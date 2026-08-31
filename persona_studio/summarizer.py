"""The rolling summary: a party's long-term memory.

The `instance` columns `summary_text`, `summary_upto` and `world_state` carry
everything (schema v1); this module owns writing them. `summary_upto` is a
**message id**, not a position: every message with `id <= summary_upto` is
covered by the summary and never re-sent verbatim. Being an id, deleting or
editing a message never shifts it.

The shape of the job, and why:

- **Off the request path.** `schedule` runs after a reply is persisted, checks
  the trigger with one cheap query, and either returns or spawns a daemon
  thread that does the model call and the write. The player's stream closes on
  the model's last token, never on the summarizer's.
- **One at a time per party.** A per-party lock, acquired without blocking: a
  second trigger while one summarization runs is skipped, not queued. The
  lock table is in-process by design — this application runs as a single
  uvicorn process, so a process-local lock is the whole mutual exclusion the
  design needs — and it is swept after each job so it cannot grow without
  bound across a long session.
- **Write only on success.** An unreachable Ollama, an error, an empty or
  unparseable answer leaves `summary_text`, `summary_upto` and `world_state`
  exactly as they were. No retry state exists and none is needed: the trigger
  condition is still true on the next turn, so the retry is automatic.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

from . import db, narrator, ollama, settings

# Continuity, not literature: the summary carries who the protagonist is,
# where they are, who is present, what is established, what is unresolved.
# The previous summary and world state ride along so each pass extends the
# memory instead of restarting it.
_SUMMARY_SYSTEM = (
    "You are the memory-keeper of an interactive story. The messages below are "
    "the oldest part of the transcript; they are being replaced by a summary "
    "to keep the narrator's context bounded.\n"
    "Answer with JSON only, exactly this shape:\n"
    '{"summary": "...", "world_state": {"location": "...", "present": [...], '
    '"established": [...], "unresolved": [...]}}\n'
    "The summary must carry forward the story memory below it: who the "
    "protagonist is, where the scene stands, who is present, what has been "
    "established, and what is unresolved. Write it in the language of the "
    "transcript. Keep established facts; drop nothing important. Output only "
    "the JSON object, nothing before or after it."
)

_SUMMARY_ASK = "Now write the JSON object summarizing the messages above."


@dataclass(frozen=True)
class _SummaryPlan:
    """Everything the background job needs, read before the thread starts.

    The route thread does the reads and returns; the thread only calls the
    model and writes, so no SQLite connection is ever shared across threads
    or held across the model call.
    """

    party_id: str
    frontier: int
    history: list[narrator.HistoryMessage]
    previous_summary: str
    world_state: dict[str, Any]
    model: str
    num_ctx: int


# The locks are process-local on purpose: this application runs as a single
# uvicorn process, so an in-process lock keyed by party id is the complete
# mutual exclusion, not an approximation of it.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(party_id: str) -> threading.Lock | None:
    """The party's lock, acquired without blocking.

    None means a summarization for this party is already running and this
    trigger is skipped, not queued: the condition that made it necessary is
    still true, so the next turn will trigger again.
    """
    with _locks_guard:
        lock = _locks.get(party_id)
        if lock is None:
            lock = threading.Lock()
            _locks[party_id] = lock
        if not lock.acquire(blocking=False):
            return None
        return lock


def _sweep_locks() -> None:
    """Drop the locks no job holds, so the map cannot grow without bound.

    Safe because map access and the acquire inside `_lock_for` both hold
    `_locks_guard`: a lock another thread is about to acquire cannot be swept,
    because that acquire waits for the guard too.
    """
    with _locks_guard:
        for party_id in [pid for pid, lock in _locks.items() if lock.acquire(blocking=False)]:
            _locks.pop(party_id).release()


def schedule(party_id: str) -> None:
    """Start a summarization if the trigger is met; return immediately.

    Called after a reply is persisted, and only then. The trigger: the number
    of uncovered text messages — those with `id > summary_upto` — exceeds the
    configured history window. Under the trigger this does nothing at all.
    The model call and the write run on a background thread; this function is
    a cheap read plus, at most, a thread spawn.
    """
    with db.connect() as con:
        plan = _plan(con, party_id)
    if plan is None:
        return
    lock = _lock_for(party_id)
    if lock is None:
        return
    threading.Thread(
        target=_run, args=(plan, lock), daemon=True, name=f"rolling-summary-{party_id[:8]}"
    ).start()


def _plan(con: sqlite3.Connection, party_id: str) -> _SummaryPlan | None:
    """Read the trigger state and, when it is met, everything the job needs.

    Returns None when the party is unknown, has no configured model, or is
    under the trigger. The frontier only ever advances over messages that
    already existed when this range was chosen, so new turns committed while
    the job runs are harmless — see `_commit`.
    """
    row = con.execute(
        "SELECT COALESCE(summary_upto, 0), summary_text, world_state FROM instance WHERE id = ?",
        (party_id,),
    ).fetchone()
    if row is None:
        return None
    frontier, previous_summary, world_state_raw = row
    window = settings.get_history_window(con)
    uncovered = con.execute(
        "SELECT COUNT(*) FROM message WHERE instance_id = ? AND kind = 'text' AND id > ?",
        (party_id, frontier),
    ).fetchone()[0]
    if uncovered <= window:
        return None
    to_compress = uncovered - window
    message_rows = con.execute(
        "SELECT id, role, content, ooc FROM message "
        "WHERE instance_id = ? AND kind = 'text' AND id > ? "
        "ORDER BY id LIMIT ?",
        (party_id, frontier, to_compress),
    ).fetchall()
    model = settings.get_llm_model(con)
    if model is None:
        return None
    return _SummaryPlan(
        party_id=party_id,
        frontier=message_rows[-1]["id"],
        history=[
            narrator.HistoryMessage(
                id=row["id"], role=row["role"], content=row["content"], ooc=bool(row["ooc"])
            )
            for row in message_rows
        ],
        previous_summary=previous_summary,
        world_state=parse_world_state(world_state_raw),
        model=model,
        num_ctx=settings.get_num_ctx(con),
    )


def parse_world_state(raw: str) -> dict[str, Any]:
    """The stored world state as an object; anything malformed is empty."""
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _summary_messages(plan: _SummaryPlan) -> list[dict[str, str]]:
    """The call's messages: the instruction with prior memory, then the transcript.

    The instruction rides twice — as the system message, and as a final user
    message after the transcript. The transcript's last message is the
    narrator's, so a model that continues conversations instead of answering
    them (measured on qwen3:4b) would write the next story beat and never
    reach the JSON; ending on a user message asks for the answer instead.
    """
    system = (
        f"{_SUMMARY_SYSTEM}\n\n"
        f"Story memory so far:\n{plan.previous_summary or '(nothing yet)'}\n\n"
        f"Current world state:\n{json.dumps(plan.world_state, ensure_ascii=False)}"
    )
    # The compressed messages themselves ride in the call: they are what is
    # being summarized. `build_history` keeps out-of-game turns as system
    # messages, exactly as the narrator's prompt does.
    return [
        {"role": "system", "content": system},
        *narrator.build_history(plan.history),
        {"role": "user", "content": _SUMMARY_ASK},
    ]


def _parse_reply(raw: str) -> tuple[str, dict[str, Any] | None] | None:
    """`(summary, world_state)` from the model's answer, or None.

    A local model will get this wrong sometimes, and that is a job failure,
    not a crash: not valid JSON, not an object, or no usable `summary` all
    return None and leave the stored state untouched. `world_state` is None
    when the model omitted it — the caller then keeps the previous one, so an
    omission cannot erase established facts. A `world_state` that is present
    but not an object is a malformed answer and fails the whole reply.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    if "world_state" not in data:
        return summary, None
    world_state = data["world_state"]
    if not isinstance(world_state, dict):
        return None
    return summary, world_state


def _run(plan: _SummaryPlan, lock: threading.Lock) -> None:
    """The background job: call the model, then commit — or change nothing."""
    try:
        _summarize(plan)
    finally:
        lock.release()
        _sweep_locks()


def _summarize(plan: _SummaryPlan) -> None:
    messages = _summary_messages(plan)
    try:
        raw = ollama.chat(plan.model, messages, num_ctx=plan.num_ctx, format="json")
    except ollama.OllamaError:
        # Nothing is written: the previous summary stands, and the next turn
        # re-triggers because the uncovered range only grew.
        return
    parsed = _parse_reply(raw)
    if parsed is None:
        return
    summary, world_state = parsed
    _commit(plan.party_id, plan.frontier, summary, world_state, plan.world_state)


def _commit(
    party_id: str,
    frontier: int,
    summary_text: str,
    world_state: dict[str, Any] | None,
    previous_world_state: dict[str, Any],
) -> bool:
    """Write the new summary, or discard it if the frontier moved backwards.

    Every caller through `schedule()` holds the per-party lock, so two
    scheduled jobs for the same party cannot race here. The IMMEDIATE
    transaction and the frontier re-read exist for the caller that is not
    behind that lock — `_commit` invoked directly, as the tests do and a
    future maintenance path (an admin "resummarize now", a backfill) might.
    Committing a frontier older than the stored one would un-cover messages
    the summary already replaced, which is what the re-read discards.

    Between the job's read and this write, new turns can also arrive — that
    is normal and harmless, because the frontier only advances over messages
    that already existed when the range was chosen.

    A missing `world_state` in the model's answer keeps the previous one.
    """
    if world_state is None:
        world_state = previous_world_state
    with db.connect(immediate=True) as con:
        row = con.execute(
            "SELECT COALESCE(summary_upto, 0) FROM instance WHERE id = ?", (party_id,)
        ).fetchone()
        if row is None:
            # The party was deleted while the job ran: nothing left to update.
            return False
        if row[0] > frontier:
            return False
        con.execute(
            "UPDATE instance SET summary_text = ?, summary_upto = ?, world_state = ? WHERE id = ?",
            (summary_text, frontier, json.dumps(world_state, ensure_ascii=False), party_id),
        )
    return True
