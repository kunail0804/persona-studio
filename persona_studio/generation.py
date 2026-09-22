"""Background scene-image generation: start it, watch it, cancel it, recover.

An image takes minutes on this machine, so the whole job runs off the request
path, on the shape `summarizer.py` established: the route does the reads and
the first write, returns, and a daemon thread polls ComfyUI until the render
lands, fails, or passes its deadline.

The order of operations is issue #17's first criterion and is not negotiable:

1. **Prepare the graph first.** A workflow that cannot generate is refused
   before anything is written.
2. **Persist the pending message before submitting.** The row carries
   `started_at` (the real age of the job, across reloads) and its `image_id`,
   and it is committed before ComfyUI ever hears about the job. A navigation
   or a reload finds it, whatever happens next.
3. **Submit, then record the prompt id** as the row's `gen_id`. Between the
   two sits a window where a pending row has no `gen_id`: if the process
   dies exactly there, the reader must not wait forever on an image that can
   never arrive — `recover_pending` marks those rows failed at startup.
4. **Watch on a background thread** and return immediately.

Finishing is guarded, because two jobs can reach the same row: the watcher
that brings a completion first re-reads the message inside a BEGIN IMMEDIATE
transaction and writes only if the message is **still pending** and still
carries **the same `gen_id` it was started with**. A completion arriving
after a cancellation is discarded whole — status, row and PNG file.

Cancelling is guarded the same way at the ComfyUI end. `POST /interrupt`
stops whatever is executing *now*, not a job of our choosing, so the queue is
read first: our prompt id running → interrupt; waiting → remove the queue
entry, never interrupt (an unrelated render is on the GPU); in neither →
finished or never queued, ComfyUI is not touched at all. The local cancel is
written before any of that, so an unreachable ComfyUI still leaves the player
with a cancelled image rather than a spinner.

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

from . import comfyui, db, ollama, settings, workflows
from .workflows import WorkflowNotReady

# A render takes minutes: a poll every couple of seconds is a human cadence,
# not a hot loop. Both are read from the module globals on every pass so a
# test can tighten them.
POLL_INTERVAL_SECONDS = 2.0
DEADLINE_SECONDS = 3600.0

CANCEL_MESSAGE = "Generation cancelled."

# Every failure the module records — watcher errors, cancellations, the two
# startup recoveries — marks its message row with the same statement.
_MARK_FAILED_SQL = "UPDATE message SET status = 'error', error = ? WHERE id = ?"


@dataclass(frozen=True)
class _Plan:
    """Everything the watcher needs, read before the thread starts.

    The route thread does the reads and returns; the thread only talks to
    ComfyUI and writes, so no SQLite connection is ever shared across threads
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


def _evict_narrator(model: str | None) -> None:
    """Free the narration model's VRAM before a render takes the GPU.

    Narration and image generation share one card. Measured on this machine:
    a chat model still resident when a diffusion pipeline loads is an
    out-of-memory error, not a slowdown — so the narrator is evicted before
    the render starts rather than after it has already failed. The reverse
    direction needs nothing here: the workflows in use carry their own VRAM
    and RAM management nodes, so narration reloads normally once a render is
    done.

    Best effort, and deliberately so. This runs after the pending message is
    committed and before the submission, so the player has already asked for
    a render: an unreachable or unhappy Ollama must not turn that into an
    error, and it is holding no VRAM anyway. The cost of a failed eviction is
    one render that may not fit — which is the situation this call was trying
    to improve, never one it makes worse.

    Where this sits lengthens the persist-to-submit window by one HTTP call.
    That window is already handled: a cancel landing in it is recorded, and
    the submission that returns afterwards respects the cancellation instead
    of stamping a live job id over it.
    """
    if model is None:
        return
    try:
        ollama.unload(model)
    except ollama.OllamaError:
        return


def start(party_id: str, prompt: str, instruction: str) -> StartResult:
    """Start one generation: persist pending, submit, watch in the background.

    Raises `LookupError` for an unknown party and `WorkflowNotReady` (with
    the message `workflows.problem()` returns) when the active workflow
    cannot generate — both before a single row is written. A ComfyUI that
    refuses the submission deletes the pending row it created — unless a
    cancel landed during the round trip and already recorded its
    cancellation, which stands.
    """
    with db.connect() as con:
        party = con.execute("SELECT id FROM instance WHERE id = ?", (party_id,)).fetchone()
        if party is None:
            raise LookupError(f"Party {party_id!r} not found")
        row = workflows.resolve_active(con)
        if row is None:
            raise WorkflowNotReady(
                "No workflow is imported: import one in the settings and map its prompt field."
            )
        # Read here, used after this connection closes: the eviction below is
        # an HTTP call, and nothing in this project holds a SQLite connection
        # across one.
        narration_model = settings.get_llm_model(con)
        graph = workflows.prepare_graph(row, prompt)
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

    _evict_narrator(narration_model)

    try:
        gen_id = comfyui.submit(graph)
    except comfyui.ComfyUIError:
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
        _stop_on_comfy(gen_id)
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
    """Poll the job's history until it completes, fails, or passes its deadline.

    The first poll answers None — the job just queued — and every poll until
    ComfyUI has an entry. An unusable answer at any point fails the job with
    its real reason; nothing is retried and nothing is guessed.
    """
    started = time.time()
    while True:
        if time.time() > started + DEADLINE_SECONDS:
            _finish(plan, done=False, error=_deadline_message())
            return
        try:
            entry = comfyui.history(plan.gen_id)
        except comfyui.ComfyUIError as exc:
            _finish(plan, done=False, error=str(exc))
            return
        # The order is the real API's shape: a failed execution carries
        # `status.completed: false` — the error must be read before the
        # completion flag, or the watcher would keep polling a failed job.
        if entry is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if entry.error is not None:
            _finish(plan, done=False, error=f"ComfyUI failed: {entry.error}")
            return
        if not entry.completed:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        try:
            if not entry.images:
                _finish(plan, done=False, error="ComfyUI reported success but produced no image.")
                return
            png = comfyui.image_bytes(entry.images[0])
        except comfyui.ComfyUIError as exc:
            _finish(plan, done=False, error=str(exc))
            return
        _finish(plan, done=True, png=png)
        return


def _deadline_message() -> str:
    minutes = max(1, round(DEADLINE_SECONDS / 60))
    return (
        f"The render did not finish within {minutes} minutes and was abandoned. "
        "ComfyUI reported neither success nor failure."
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
    """Cancel a pending generation and stop it on ComfyUI as far as it is safe.

    The message is marked cancelled before ComfyUI is touched, so a cancelled
    image stays cancelled whatever the queue says or whether ComfyUI answers
    at all. See the module docstring for why the queue decides between
    interrupt, queue removal, and doing nothing.

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
    _stop_on_comfy(gen_id)
    with db.connect() as con:
        return con.execute("SELECT * FROM message WHERE id = ?", (message_id,)).fetchone()


def _stop_on_comfy(gen_id: str | None) -> None:
    """Act on ComfyUI only in the way the queue makes safe.

    Interrupting while someone else's job is rendering would kill it, so the
    queue is read first and the action follows what it says. An unreachable
    ComfyUI changes nothing here: the cancellation is already recorded, and a
    late completion will be discarded against it.
    """
    if gen_id is None:
        return
    try:
        # Residual the queue check cannot close: between this read and the
        # action below, the running job can finish and the next one start, so
        # an interrupt can still land on a render that is not ours. ComfyUI's
        # interrupt carries no target, so read-then-act is the best its API
        # allows; the half that matters most — never interrupting a merely
        # queued job — is airtight.
        state = comfyui.queue()
        if gen_id in state.running:
            comfyui.interrupt()
        elif gen_id in state.pending:
            comfyui.delete_queued(gen_id)
    except comfyui.ComfyUIError:
        pass


def recover_pending() -> None:
    """Mark every still-pending image failed; called once at startup.

    A process death can leave a pending row in two states, and both can
    never complete once the watcher thread is gone:

    - no `gen_id`: the process died inside the persist-to-submit window, so
      the job was never queued (or its id was never recorded);
    - a `gen_id`: the job may still be queued or rendering on ComfyUI, and
      while the prompt and instruction the completion needs were held only
      in memory no result could be recorded honestly — but the job can be
      stopped. Before the rows are marked, each one carrying a `gen_id` goes
      through the same queue-checked stop the cancel uses. Stopping happens
      first, outside the write transaction: a process dying again mid-way
      leaves the rows pending and lets the next startup try once more, and
      an unreachable ComfyUI (whose errors the stop swallows) cannot prevent
      the recovery from completing.

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
        _stop_on_comfy(row["gen_id"])
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
