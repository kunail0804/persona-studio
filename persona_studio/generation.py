"""Background scene-image generation: start it, watch it, cancel it, recover.

An image takes minutes on this machine, so the whole job runs off the request
path, on the shape `summarizer.py` established: the route does the reads and
the first write, returns, and a daemon thread polls comfy-gateway until the
render lands, fails, or passes its deadline.

The image is made by comfy-gateway, never by this application: it names the
template chosen in the settings (and its preset), sends the prompt as the
template's `prompt` role, and fetches the image the job returns. See
`gateway.py`.

The order of operations is issue #17's first criterion and is not negotiable:

1. **Check the setup first.** With no template chosen, the request is refused
   before anything is written. Whether the gateway accepts the template is
   only known when it answers the submission, and a refusal there deletes
   the pending row again (step 3).
2. **Persist the pending message before submitting.** The row carries
   `started_at` (the real age of the job, across reloads) and its `image_id`,
   and it is committed before the gateway ever hears about the job. A navigation
   or a reload finds it, whatever happens next.
3. **Submit, then record the gateway's job id** as the row's `gen_id`. Between the
   two sits a window where a pending row has no `gen_id`: if the process
   dies exactly there, the reader must not wait forever on an image that can
   never arrive — `recover_pending` marks those rows failed at startup.
4. **Watch on a background thread** and return immediately.

Finishing is guarded, because two jobs can reach the same row: the watcher
that brings a completion first re-reads the message inside a BEGIN IMMEDIATE
transaction and writes only if the message is **still pending** and still
carries **the same `gen_id` it was started with**. A completion arriving
after a cancellation is discarded whole — status, row and PNG file.

Cancelling asks the gateway to cancel that job id, and the gateway stops that
prompt in ComfyUI and nothing else, whether it is queued or rendering. The
local cancel is written first, so an unreachable gateway still leaves the
player with a cancelled image rather than a spinner.

That cancel cannot reach into the persist-to-submit window: there, a cancel
lands on a row whose `gen_id` does not exist yet, so the stop path has
nothing to aim at. `start` closes it: after the submission returns it
re-reads the row under the write lock, and only a still-pending row gets the
id stamped and a watcher started. A row a cancel already claimed gets its
job stopped instead — the only moment that stop was ever possible — and a
refused submission deletes nothing but its own still-pending row.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass

from . import db, gateway, gpu, settings

# A render takes minutes: a poll every couple of seconds is a human cadence,
# not a hot loop. Both are read from the module globals on every pass so a
# test can tighten them.
POLL_INTERVAL_SECONDS = 2.0
DEADLINE_SECONDS = 3600.0

CANCEL_MESSAGE = "Generation cancelled."


class ImageSetupNotReady(Exception):
    """Images cannot be made yet: the message says what to set up."""


NO_TEMPLATE_MESSAGE = (
    "No image template is chosen: choose a comfy-gateway template in the settings."
)

# Every failure the module records — watcher errors, cancellations, the two
# startup recoveries — marks its message row with the same statement.
_MARK_FAILED_SQL = "UPDATE message SET status = 'error', error = ? WHERE id = ?"


@dataclass(frozen=True)
class _Plan:
    """Everything the watcher needs, read before the thread starts.

    The route thread does the reads and returns; the thread only talks to
    the gateway and writes, so no SQLite connection is ever shared across threads
    or held across an HTTP call.
    """

    party_id: str
    message_id: int
    image_id: str
    gen_id: str
    prompt: str
    instruction: str


@dataclass(frozen=True)
class StartResult:
    """What the route needs to answer with: the message that now shows the
    pending image, and when its clock started."""

    message_id: int
    started_at: float


RENDER_IN_PROGRESS_DETAIL = (
    "An image is being rendered. The narrator waits until it is done: on this "
    "card the two do not fit in memory together."
)


def render_in_progress(con: sqlite3.Connection) -> bool:
    """Whether any image, in any party, is rendering right now.

    Global on purpose: the GPU is one card, not one per party, so a render in
    one party and a narrator call in another are the same out-of-memory as
    both in the same party. Ollama is emptied before a render starts
    (`gpu.make_room_for_render`); this is the other half — nothing loads the
    narrator back while the render runs.

    A `pending` row whose watcher died with the process is marked failed at
    startup by `recover_pending`, so a crash cannot leave this true forever.
    """
    return (
        con.execute(
            "SELECT 1 FROM message WHERE kind = 'image' AND status = 'pending' LIMIT 1"
        ).fetchone()
        is not None
    )


def start(party_id: str, prompt: str, instruction: str) -> StartResult:
    """Start one generation: persist pending, submit, watch in the background.

    Raises `LookupError` for an unknown party and `ImageSetupNotReady` when
    no template is chosen, both before a single row is written. A gateway
    that refuses or misses the submission deletes the pending row it created,
    unless a cancel landed during the round trip and already recorded its
    cancellation, which stands. The `gateway` error is raised again for the
    route to answer.
    """
    with db.connect() as con:
        party = con.execute("SELECT id FROM instance WHERE id = ?", (party_id,)).fetchone()
        if party is None:
            raise LookupError(f"Party {party_id!r} not found")
        template = settings.get_image_template(con)
        if template is None:
            raise ImageSetupNotReady(NO_TEMPLATE_MESSAGE)
        preset = settings.get_image_gateway_preset(con)
        now = time.time()
        # The image's id is drawn here because the PNG will carry it, but the
        # `image` row itself is only inserted at completion: `message.image_id`
        # references `image(id)`, so a pending message cannot point at a row
        # that does not exist yet.
        image_id = uuid.uuid4().hex
        cursor = con.execute(
            "INSERT INTO message "
            "(instance_id, role, kind, content, ts, status, started_at) "
            "VALUES (?, 'assistant', 'image', ?, ?, 'pending', ?)",
            (party_id, instruction, now, now),
        )
        message_id = cursor.lastrowid
        if message_id is None:
            raise RuntimeError("The pending INSERT succeeded but returned no rowid")

    # Every Ollama model, not only the narrator's, and waited for: see `gpu`.
    # This lengthens the persist-to-submit window, which is already handled:
    # a cancel landing in it is recorded, and the submission that returns
    # afterwards respects it instead of stamping a live job id over it.
    gpu.make_room_for_render()

    try:
        gen_id = gateway.submit(template, preset, prompt)
    except gateway.GatewayError:
        # Only a still-pending row is ours to clean up: a cancel that landed
        # during the round trip already turned it into a recorded
        # cancellation, and deleting that would erase the player's message.
        with db.connect(immediate=True) as con:
            row = con.execute("SELECT status FROM message WHERE id = ?", (message_id,)).fetchone()
            if row is not None and row["status"] == "pending":
                con.execute("DELETE FROM message WHERE id = ?", (message_id,))
        raise

    # The submission was the window a cancel could land in: its stop path saw
    # a row with no `gen_id` and could do nothing. Re-read under the write
    # lock before assuming the row is still pending — a cancel either
    # committed before this transaction begins (this read sees it) or waits
    # for this commit and then finds the `gen_id` it needs itself.
    with db.connect(immediate=True) as con:
        row = con.execute("SELECT status FROM message WHERE id = ?", (message_id,)).fetchone()
        if row is not None and row["status"] == "pending":
            con.execute("UPDATE message SET gen_id = ? WHERE id = ?", (gen_id, message_id))
            cancelled = False
        else:
            # Cancelled, or gone with its party: either way nothing is
            # watching the job now, and this is the only moment its id is
            # known — stop it through the same queue-checked path.
            cancelled = True

    if cancelled:
        _stop_job(gen_id)
        return StartResult(message_id=message_id, started_at=now)

    plan = _Plan(
        party_id=party_id,
        message_id=message_id,
        image_id=image_id,
        gen_id=gen_id,
        prompt=prompt,
        instruction=instruction,
    )
    threading.Thread(
        target=_watch, args=(plan,), daemon=True, name=f"image-gen-{message_id}"
    ).start()
    return StartResult(message_id=message_id, started_at=now)


def _watch(plan: _Plan) -> None:
    """Poll the job until it is done, fails, or passes its deadline.

    An unusable answer at any point fails the job with its real reason;
    nothing is retried and nothing is guessed.
    """
    started = time.time()
    while True:
        if time.time() > started + DEADLINE_SECONDS:
            # Abandoned here, the job must not go on rendering: the narrator
            # is unblocked as soon as the row stops being pending.
            if _finish(plan, done=False, error=_deadline_message()):
                _stop_job(plan.gen_id)
            return
        try:
            state = gateway.job(plan.gen_id)
        except gateway.GatewayError as exc:
            _finish(plan, done=False, error=str(exc))
            return
        if state.status not in gateway.FINISHED:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if state.status == "cancelled":
            # Cancelled here, the row is no longer pending and nothing is
            # written. Cancelled from the gateway's own interface, it is.
            _finish(plan, done=False, error="The job was cancelled in comfy-gateway.")
            return
        if state.status == "error":
            reason = state.error or "no reason given"
            _finish(plan, done=False, error=f"comfy-gateway failed: {reason}")
            return
        try:
            if not state.images:
                _finish(
                    plan, done=False, error="comfy-gateway reported success but returned no image."
                )
                return
            # The first image is the one a workflow saves first; a comparison
            # or a test grid saved by the same workflow comes after it.
            png = gateway.image_bytes(state.images[0])
        except gateway.GatewayError as exc:
            _finish(plan, done=False, error=str(exc))
            return
        _finish(plan, done=True, png=png)
        return


def _deadline_message() -> str:
    minutes = max(1, round(DEADLINE_SECONDS / 60))
    return (
        f"The render did not finish within {minutes} minutes and was abandoned: "
        "comfy-gateway reported neither success nor failure, and the job was "
        "cancelled there so it stops holding the graphics card."
    )


def _finish(plan: _Plan, *, done: bool, error: str | None = None, png: bytes | None = None) -> bool:
    """Resolve the message to `done` or `error`, or discard the result.

    The read-check-write runs in one BEGIN IMMEDIATE transaction, so a cancel
    that lands either commits before the read (and the completion sees it) or
    waits for this commit and lands after it — it can no longer fall into the
    gap between the read and the write. A result whose message is no longer
    pending, or whose `gen_id` moved on, is dropped without writing anything
    — a late completion must not overwrite a cancellation, and the PNG it
    brought must not be left behind on disk.

    Returns True when this call wrote the outcome.
    """
    with db.connect(immediate=True) as con:
        row = con.execute(
            "SELECT status, gen_id, started_at FROM message WHERE id = ?", (plan.message_id,)
        ).fetchone()
        if row is None:
            # The party was deleted while the job ran: nothing left to update.
            return False
        if row["status"] != "pending" or row["gen_id"] != plan.gen_id:
            return False
        if done:
            seconds = time.time() - row["started_at"] if row["started_at"] is not None else None
            con.execute(
                "INSERT INTO image (id, prompt, instruction, seconds, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (plan.image_id, plan.prompt, plan.instruction, seconds, time.time()),
            )
            con.execute(
                "UPDATE message SET status = 'done', image_id = ?, error = NULL WHERE id = ?",
                (plan.image_id, plan.message_id),
            )
            # An image landing is story activity, like a played turn.
            con.execute(
                "UPDATE instance SET updated_at = ? WHERE id = ?", (time.time(), plan.party_id)
            )
        else:
            con.execute(
                _MARK_FAILED_SQL,
                (error, plan.message_id),
            )
    if done and png is not None:
        # Written after the commit: a rolled-back transaction never leaves a
        # file whose row does not exist.
        db.image_path(plan.image_id).write_bytes(png)
    return True


def cancel(party_id: str, message_id: int) -> sqlite3.Row:
    """Cancel a pending generation and stop its job on the gateway.

    The message is marked cancelled before the gateway is touched, so a
    cancelled image stays cancelled whether the gateway answers or not.

    Raises `LookupError` for an unknown message and `ValueError` when the
    message is not a pending image generation.
    """
    with db.connect(immediate=True) as con:
        row = con.execute(
            "SELECT kind, status, gen_id FROM message WHERE id = ? AND instance_id = ?",
            (message_id, party_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"Message {message_id} not found in party {party_id!r}")
        if row["kind"] != "image" or row["status"] != "pending":
            raise ValueError("Only a pending image generation can be cancelled.")
        con.execute(
            _MARK_FAILED_SQL,
            (CANCEL_MESSAGE, message_id),
        )
        gen_id = row["gen_id"]
    _stop_job(gen_id)
    with db.connect() as con:
        return con.execute("SELECT * FROM message WHERE id = ?", (message_id,)).fetchone()


def _stop_job(gen_id: str | None) -> None:
    """Cancel the gateway job, if there is one.

    The gateway cancels that job's prompt in ComfyUI by id, so another render
    on the card is never touched. An unreachable gateway changes nothing
    here: the cancellation is already recorded, and a late completion is
    discarded against it.
    """
    if gen_id is None:
        return
    try:
        gateway.cancel(gen_id)
    except gateway.GatewayError:
        pass


def recover_pending() -> None:
    """Mark every still-pending image failed; called once at startup.

    A process death can leave a pending row in two states, and both can
    never complete once the watcher thread is gone:

    - no `gen_id`: the process died inside the persist-to-submit window, so
      the job was never queued (or its id was never recorded);
    - a `gen_id`: the job may still be queued or rendering on the gateway,
      and while the prompt and instruction the completion needs were held
      only in memory no result could be recorded honestly, but the job can
      be stopped. Before the rows are marked, each one carrying a `gen_id`
      is cancelled the way the cancel does it. Stopping happens first,
      outside the write transaction: a process dying again mid-way leaves
      the rows pending and lets the next startup try once more, and an
      unreachable gateway (whose errors the stop swallows) cannot prevent the
      recovery from completing.

    Running at startup means a reader never sees one of these claim to be
    working: the application is down while it is dead, and the first startup
    after it heals the rows.
    """
    with db.connect(immediate=True) as con:
        rows = con.execute(
            "SELECT id, gen_id FROM message WHERE kind = 'image' AND status = 'pending'"
        ).fetchall()
    if not rows:
        return
    for row in rows:
        _stop_job(row["gen_id"])
    with db.connect(immediate=True) as con:
        for row in rows:
            message = (
                "The application restarted before the generation could start."
                if row["gen_id"] is None
                else (
                    "The application restarted while the image was rendering; "
                    "the render was abandoned."
                )
            )
            con.execute(
                _MARK_FAILED_SQL,
                (message, row["id"]),
            )


def recover_missing_done_files() -> None:
    """Heal `done` image messages whose PNG file is gone; called once at startup.

    The completion commits its rows first and writes the PNG after, so a
    process death exactly between the two leaves a message claiming `done`
    with an image id that resolves to nothing. That ordering is deliberate —
    a rolled-back transaction must never leave an orphan file — so the
    inconsistency is healed here rather than reordered there. Such a message
    is not pending, so `recover_pending` passes it by and it can never be
    cancelled: this is the only reader that can notice it.
    """
    with db.connect(immediate=True) as con:
        rows = con.execute(
            "SELECT id, image_id FROM message "
            "WHERE kind = 'image' AND status = 'done' AND image_id IS NOT NULL"
        ).fetchall()
        for row in rows:
            if db.image_path(row["image_id"]).is_file():
                continue
            con.execute(
                _MARK_FAILED_SQL,
                (
                    "The image was recorded as complete, but its file is missing "
                    f"({db.image_path(row['image_id']).name}); it is shown as failed.",
                    row["id"],
                ),
            )
