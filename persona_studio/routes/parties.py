"""Parties: one playthrough of a scenario, with its transcript.

Naming: the SQL table is `instance` — it carried that name before the feature
had an API. Everywhere a reader can act on it (routes, Pydantic models, front
end) it is a **party**. That mapping lives here and nowhere else; SQL says
`instance`, Python and TypeScript say `party`.

Creating a party generates the opening scene BEFORE anything is written: an
unreachable Ollama must leave no empty party behind. The route therefore holds
no open transaction across the model call — it reads what it needs, closes the
connection, calls Ollama, and only then opens a new one to insert the row and
its opening message in one transaction.

Playing a turn follows the same discipline, with one addition: the player's
text is written BEFORE generation starts — a broken stream must not lose what
the player did — and the reply, whatever arrived of it, is written in a
`finally` after the stream ends, on a fresh connection. No connection is ever
held open across the model call, which a local model can make run for minutes.

Variants: `message.content` always holds the **active** text. It is what
`narrator.load_history` reads, what the full-text index mirrors, and what
every other part of the application already relies on — none of that changes.
The `variant` table is the archive beside it, written only by editing and
regenerating. The invariant every path here preserves:

- a message with no variant rows has exactly one text: `message.content`;
- a message with variant rows has exactly one with `active = 1`, and
  `message.content` equals that row's content.

`_set_message_text` is the single writer that establishes it; no route may
touch `message.content` or the `active` flags directly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import anyio.to_thread
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from .. import db, narrator, ollama, settings, summarizer

router = APIRouter(tags=["parties"])


class MessageVariant(BaseModel):
    id: int
    content: str
    active: bool


MessageStatus = Literal["pending", "done", "error"]


class PartyMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    ts: float
    variants: list[MessageVariant] = []
    # Image messages only: the schema's own columns, surfaced so the interface
    # can show the generation's state. A text message carries the defaults.
    kind: Literal["text", "image"] = "text"
    image_id: str | None = None
    status: MessageStatus | None = None
    started_at: float | None = None
    error: str | None = None


class PartySummary(BaseModel):
    id: str
    scenario_id: str
    scenario_title: str
    label: str
    created_at: float
    updated_at: float


class ContextUsage(BaseModel):
    """How full the next turn's prompt is against the configured window.

    Ollama truncates an over-long prompt in silence, from the front, system
    prompt first — the narrator then "forgets" the scenario for no visible
    reason. This is what lets the interface say so before it happens.
    """

    estimated_tokens: int
    num_ctx: int
    near_limit: bool


class PromptBlock(BaseModel):
    """One message of the next turn's call, with what it weighs."""

    role: str
    content: str
    estimated_tokens: int


class PromptView(BaseModel):
    """Exactly what would be sent to the narrator, block by block."""

    blocks: list[PromptBlock]
    context: ContextUsage


class MemoryInput(BaseModel):
    """A correction to the party's memory. Only the fields sent are written."""

    model_config = ConfigDict(extra="forbid")

    summary_text: str | None = None
    world_state: dict[str, Any] | None = None


class Party(PartySummary):
    messages: list[PartyMessage]
    summary_text: str
    summary_upto: int | None
    world_state: dict[str, Any]
    context: ContextUsage


class PartyInput(BaseModel):
    label: str = ""


class TurnInput(BaseModel):
    content: str


class MessageEditInput(BaseModel):
    content: str


class VariantInput(BaseModel):
    variant_id: int


_SUMMARY_COLUMNS = (
    "instance.id, instance.scenario_id, instance.label, "
    "instance.created_at, instance.updated_at, scenario.title AS scenario_title, "
    "instance.summary_text, instance.summary_upto, instance.world_state"
)

# Everything a message response carries, including the image columns.
_MESSAGE_COLUMNS = "id, role, kind, content, ts, image_id, status, started_at, error"


# --- One turn at a time, per party ----------------------------------------------
#
# A turn commits the player's text, releases its connection, calls the model
# for as long as that takes, then appends the reply on a fresh connection.
# That ordering is deliberate: a SQLite connection must not be held open for
# minutes. It leaves a window, and nothing used to keep a second turn out of
# it — two turns played at once interleaved into `user, user, assistant,
# assistant`, each reply answering a prompt that held the other's unanswered
# turn. `message.id` is the order of the story, so that damage is permanent:
# there is no delete and no reorder endpoint to repair it with.
#
# The front end guards one tab (`controllerRef`, plus `disabled={streaming}`),
# and that is per-tab React state — useless against a second tab, a second
# window or a script. The guard has to be here.
#
# Process-local, keyed by party, and non-blocking, exactly like the
# summarizer's: this application runs as a single uvicorn process, so an
# in-process lock is the complete mutual exclusion rather than an
# approximation of it. A second turn is refused outright rather than queued —
# it was typed without knowledge of the first, so playing it afterwards would
# answer a story the player has not read yet.
_turn_locks: dict[str, threading.Lock] = {}
_turn_locks_guard = threading.Lock()


def _acquire_turn_lock(party_id: str) -> Callable[[], None]:
    """Take the party's turn lock, or 409. Returns the matching release.

    The release runs on the event loop thread while the acquire ran on a
    worker thread, which `threading.Lock` allows — it is not an RLock and has
    no owning thread.
    """
    with _turn_locks_guard:
        lock = _turn_locks.get(party_id)
        if lock is None:
            lock = threading.Lock()
            _turn_locks[party_id] = lock
        acquired = lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="A turn is already being played in this party. Wait for it to finish.",
        )

    def release() -> None:
        lock.release()
        _sweep_turn_locks()

    return release


def _sweep_turn_locks() -> None:
    """Drop the locks no turn holds, so the map cannot grow without bound.

    Safe for the same reason the summarizer's sweep is: every acquire happens
    under `_turn_locks_guard`, so a lock another request is about to take
    cannot be swept out from under it.
    """
    with _turn_locks_guard:
        idle = [pid for pid, lock in _turn_locks.items() if lock.acquire(blocking=False)]
        for party_id in idle:
            _turn_locks.pop(party_id).release()


def _next_turn_messages(
    con: sqlite3.Connection, row: sqlite3.Row
) -> tuple[list[dict[str, str]], int]:
    """The call the next played turn would make, and the configured window.

    One place assembles it, so the estimate the interface shows and the
    prompt the player can read are the same object the turn would send — not
    a second rendering of it that can drift.
    """
    scenario = narrator.load_scenario(con, row["scenario_id"])
    persona = narrator.load_active_persona(con)
    history = narrator.load_history(
        con,
        row["id"],
        settings.get_history_window(con),
        after_id=row["summary_upto"],
    )
    messages = narrator.build_chat_messages(scenario, persona, history, row["summary_text"])
    return messages, settings.get_num_ctx(con)


def _context_usage(messages: list[dict[str, str]], num_ctx: int) -> ContextUsage:
    usage = narrator.context_usage(messages, num_ctx)
    return ContextUsage(
        estimated_tokens=usage.estimated_tokens,
        num_ctx=usage.num_ctx,
        near_limit=usage.near_limit,
    )


def _party_from_row(row: sqlite3.Row) -> PartySummary:
    return PartySummary(
        id=row["id"],
        scenario_id=row["scenario_id"],
        scenario_title=row["scenario_title"],
        label=row["label"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _party_label(raw: str, scenario_title: str) -> str:
    """The stored label: the stripped input, or the scenario title when blank.

    One rule for create and rename alike: a blank name in a list is useless.
    """
    return raw.strip() or scenario_title


def _get_scenario_row(con: sqlite3.Connection, scenario_id: str) -> sqlite3.Row:
    row = con.execute("SELECT id, title FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id!r} not found")
    return row


def _get_party_row(con: sqlite3.Connection, party_id: str) -> sqlite3.Row:
    row = con.execute(
        f"SELECT {_SUMMARY_COLUMNS} "
        "FROM instance JOIN scenario ON scenario.id = instance.scenario_id "
        "WHERE instance.id = ?",
        (party_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Party {party_id!r} not found")
    return row


def _get_message_row(con: sqlite3.Connection, party_id: str, message_id: int) -> sqlite3.Row:
    """The party's message, or 404 — an unknown id and a message owned by
    another party are the same mistake from the caller's side: nothing to act
    on, so both are 404 rather than 403."""
    row = con.execute(
        f"SELECT {_MESSAGE_COLUMNS} FROM message WHERE id = ? AND instance_id = ?",
        (message_id, party_id),
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Message {message_id} not found in party {party_id!r}"
        )
    return row


def _archive_texts(original: str | None, current: str) -> list[str]:
    """The texts a regeneration must archive, oldest first.

    `current` is what the row holds when the archive runs; `original` is what
    it held when the regeneration was requested. They differ when an edit
    landed while the model was writing — an ordinary second tab, seconds
    apart, no race to arrange — and archiving only one of them destroys the
    other: the edit is an in-place UPDATE that keeps no copy, so the text the
    regeneration started from would exist nowhere at all. Both go in, in the
    order the player produced them.
    """
    if original is None or original == current:
        return [current]
    return [original, current]


def _set_message_text(
    con: sqlite3.Connection,
    message_id: int,
    content: str,
    *,
    archive_current: bool = False,
    from_variant_id: int | None = None,
    original: str | None = None,
) -> None:
    """Make `content` the message's one active text.

    Every path that changes which text a message shows goes through here, so
    the variant invariant (see the module docstring) holds after each of them.

    - **Edit** (default): `message.content` is corrected in place, and the
      active variant row — when there is one — is corrected with it. Without
      the second write, switching variants would resurrect the text the user
      just replaced.
    - **Regenerate** (`archive_current`): the text being replaced is archived
      as a variant first, unless the active row already holds it — on the
      first regeneration the message has no rows yet, afterwards the active
      row *is* the archive of the current text. `content` then becomes the
      new active row. `original` is the text the regeneration started from,
      and it is archived alongside the current one when an edit changed the
      row in between; see `_archive_texts`.
    - **Switch** (`from_variant_id`): the flags flip and the chosen row's
      content becomes the message text. No row is created — switching
      navigates the archive, it does not grow it.

    Editing is an UPDATE, never a delete followed by an insert: the message's
    id is the order of the story, and `instance.summary_upto` is a message id
    — the rolling summary's frontier. Recreating the row would move both.
    """
    if from_variant_id is not None:
        con.execute(
            "UPDATE variant SET active = (id = ?) WHERE message_id = ?",
            (from_variant_id, message_id),
        )
    elif archive_current:
        current = con.execute("SELECT content FROM message WHERE id = ?", (message_id,)).fetchone()[
            "content"
        ]
        already_archived = (
            con.execute(
                "SELECT 1 FROM variant WHERE message_id = ? AND active = 1", (message_id,)
            ).fetchone()
            is not None
        )
        if not already_archived:
            for text in _archive_texts(original, current):
                con.execute(
                    "INSERT INTO variant (message_id, content, created_at, active) "
                    "VALUES (?, ?, ?, 1)",
                    (message_id, text, time.time()),
                )
        con.execute("UPDATE variant SET active = 0 WHERE message_id = ?", (message_id,))
        con.execute(
            "INSERT INTO variant (message_id, content, created_at, active) VALUES (?, ?, ?, 1)",
            (message_id, content, time.time()),
        )
    else:
        con.execute(
            "UPDATE variant SET content = ? WHERE message_id = ? AND active = 1",
            (content, message_id),
        )
    con.execute("UPDATE message SET content = ? WHERE id = ?", (content, message_id))


def _message_response(con: sqlite3.Connection, row: sqlite3.Row) -> PartyMessage:
    """One message with its variant archive, in response shape."""
    variant_rows = con.execute(
        "SELECT id, content, active FROM variant WHERE message_id = ? ORDER BY id",
        (row["id"],),
    ).fetchall()
    return PartyMessage(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        ts=row["ts"],
        kind=row["kind"],
        image_id=row["image_id"],
        status=row["status"],
        started_at=row["started_at"],
        error=row["error"],
        variants=[
            MessageVariant(id=v["id"], content=v["content"], active=bool(v["active"]))
            for v in variant_rows
        ],
    )


def _require_model(con: sqlite3.Connection) -> str:
    model = settings.get_llm_model(con)
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="No narration model is configured. Choose a model in the settings page.",
        )
    return model


def _generate_opening(
    model: str,
    scenario: narrator.PromptScenario,
    persona: narrator.PromptPersona | None,
    num_ctx: int,
) -> str:
    """The opening text, or an HTTPException — never a written row."""
    messages = narrator.build_opening_messages(scenario, persona)
    try:
        opening = ollama.chat(model, messages, num_ctx=num_ctx)
    except ollama.OllamaUnreachable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ollama.OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not opening.strip():
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty reply, so no opening scene was generated.",
        )
    return opening


@router.post("/scenarios/{scenario_id}/parties", response_model=PartySummary, status_code=201)
def create_party(scenario_id: str, body: PartyInput) -> PartySummary:
    with db.connect() as con:
        scenario_row = _get_scenario_row(con, scenario_id)
        model = _require_model(con)
        scenario = narrator.load_scenario(con, scenario_id)
        persona = narrator.load_active_persona(con)
        num_ctx = settings.get_num_ctx(con)

    # Nothing above wrote a row: a failure past this point — including the
    # whole model call — leaves the database untouched.
    opening = _generate_opening(model, scenario, persona, num_ctx)

    now = time.time()
    party_id = uuid.uuid4().hex
    try:
        with db.connect() as con:
            con.execute(
                "INSERT INTO instance (id, scenario_id, label, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (party_id, scenario_id, _party_label(body.label, scenario_row["title"]), now, now),
            )
            con.execute(
                "INSERT INTO message (instance_id, role, kind, content, ts) "
                "VALUES (?, 'assistant', 'text', ?, ?)",
                (party_id, opening, now),
            )
    except sqlite3.IntegrityError as exc:
        # 404, not 409: the scenario the party would belong to is genuinely
        # gone, so the caller's next sensible move is to stop, not to retry.
        raise HTTPException(
            status_code=404,
            detail=f"Scenario {scenario_id!r} was deleted while the opening scene "
            "was being generated.",
        ) from exc
    return PartySummary(
        id=party_id,
        scenario_id=scenario_id,
        scenario_title=scenario_row["title"],
        label=_party_label(body.label, scenario_row["title"]),
        created_at=now,
        updated_at=now,
    )


@router.get("/parties", response_model=list[PartySummary])
def list_parties() -> list[PartySummary]:
    with db.connect() as con:
        rows = con.execute(
            f"SELECT {_SUMMARY_COLUMNS} "
            "FROM instance JOIN scenario ON scenario.id = instance.scenario_id "
            # Pinned parties first, then most recently active: this is exactly
            # the `instance_by_recency` index. `pinned` has no API yet.
            "ORDER BY instance.pinned DESC, instance.updated_at DESC"
        ).fetchall()
    return [_party_from_row(row) for row in rows]


@router.get("/parties/{party_id}", response_model=Party)
def get_party(party_id: str) -> Party:
    with db.connect() as con:
        row = _get_party_row(con, party_id)
        message_rows = con.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM message WHERE instance_id = ? ORDER BY id",
            (party_id,),
        ).fetchall()
        variant_rows = con.execute(
            "SELECT v.id, v.message_id, v.content, v.active FROM variant v "
            "JOIN message m ON m.id = v.message_id "
            "WHERE m.instance_id = ? ORDER BY v.id",
            (party_id,),
        ).fetchall()
        # What the next turn's prompt would weigh, assembled exactly as the
        # turn assembles it. The estimate rides with the party rather than on
        # an endpoint of its own: the page already reads this one, and a
        # warning nobody fetches is the state this feature was already in.
        messages, num_ctx = _next_turn_messages(con, row)
    usage = _context_usage(messages, num_ctx)
    # Variants ride inline rather than behind a second endpoint: the interface
    # navigates them with arrow presses, and a round trip per press would show
    # visible lag on a local app whose payload is text it is already sending.
    variants_by_message: dict[int, list[MessageVariant]] = {}
    for v in variant_rows:
        variants_by_message.setdefault(v["message_id"], []).append(
            MessageVariant(id=v["id"], content=v["content"], active=bool(v["active"]))
        )
    return Party(
        id=row["id"],
        scenario_id=row["scenario_id"],
        scenario_title=row["scenario_title"],
        label=row["label"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        # Read-only for now: making the summary editable is a separate,
        # highly rated feature (see issue #14's notes).
        summary_text=row["summary_text"],
        summary_upto=row["summary_upto"],
        world_state=summarizer.parse_world_state(row["world_state"]),
        context=usage,
        messages=[
            PartyMessage(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                ts=m["ts"],
                kind=m["kind"],
                image_id=m["image_id"],
                status=m["status"],
                started_at=m["started_at"],
                error=m["error"],
                variants=variants_by_message.get(m["id"], []),
            )
            for m in message_rows
        ],
    )


@router.get("/parties/{party_id}/prompt", response_model=PromptView)
def get_party_prompt(party_id: str) -> PromptView:
    """Exactly what the next turn would send the narrator, block by block.

    The debugging tool the application was missing: when the narration goes
    wrong, the question is almost always whether a section made it into the
    prompt at all — an empty field is omitted by design, a summary can have
    replaced the turns you remember, and Ollama truncates from the front in
    silence. Reading the thing itself answers all three.

    Assembled by `_next_turn_messages`, the same function the estimate on the
    party uses, so this cannot drift from what is really sent.
    """
    with db.connect() as con:
        row = _get_party_row(con, party_id)
        messages, num_ctx = _next_turn_messages(con, row)
    return PromptView(
        blocks=[
            PromptBlock(
                role=message["role"],
                content=message["content"],
                estimated_tokens=narrator.estimate_tokens([message]),
            )
            for message in messages
        ],
        context=_context_usage(messages, num_ctx),
    )


@router.patch("/parties/{party_id}/memory", response_model=Party)
def update_party_memory(party_id: str, body: MemoryInput) -> Party:
    """Correct the rolling summary or the world state by hand.

    The summariser is a local model compressing a transcript, and it gets
    things wrong. Until now the panel was read-only, so a bad summary stayed
    wrong for the rest of the party and every later turn read it.

    The frontier is not touched: this rewrites what the summary says, never
    how much of the story it stands for — the same rule
    `summarizer.schedule_revision` follows. A summarisation committing right
    after a manual edit can still overwrite it; that window is small, the
    edit is one field away from being redone, and closing it would mean
    holding the party's lock across a model call.
    """
    fields: dict[str, Any] = {}
    if body.summary_text is not None:
        fields["summary_text"] = body.summary_text
    if body.world_state is not None:
        fields["world_state"] = json.dumps(body.world_state, ensure_ascii=False)
    clause, values = db.assignments(("summary_text", "world_state"), fields)
    with db.connect() as con:
        _get_party_row(con, party_id)
        if clause:
            con.execute(f"UPDATE instance SET {clause} WHERE id = ?", (*values, party_id))
    return get_party(party_id)


@router.patch("/parties/{party_id}", response_model=PartySummary)
def rename_party(party_id: str, body: PartyInput) -> PartySummary:
    with db.connect() as con:
        existing = _get_party_row(con, party_id)
        label = _party_label(body.label, existing["scenario_title"])
        # A rename is metadata, not story activity: `updated_at` stays put, so
        # renaming never reorders the most-recently-active list.
        con.execute("UPDATE instance SET label = ? WHERE id = ?", (label, party_id))
        row = _get_party_row(con, party_id)
    return _party_from_row(row)


@router.delete("/parties/{party_id}", status_code=204)
def delete_party(party_id: str) -> None:
    with db.connect() as con:
        _get_party_row(con, party_id)
        # The schema's `ON DELETE CASCADE` removes the party's messages with it.
        con.execute("DELETE FROM instance WHERE id = ?", (party_id,))


# --- Editing a message ----------------------------------------------------------


@router.patch("/parties/{party_id}/messages/{message_id}", response_model=PartyMessage)
def edit_message(party_id: str, message_id: int, body: MessageEditInput) -> PartyMessage:
    """Correct a message's text in place.

    The message keeps its id — see `_set_message_text` for why a delete plus
    insert is never an option. Editing is not story activity: like a rename,
    it leaves `updated_at` alone so it never reorders the
    most-recently-active list.
    """
    content = body.content.strip()
    if not content:
        # Emptying a turn is a delete, and deleting is not this feature.
        raise HTTPException(status_code=400, detail="A message cannot be edited to empty text.")
    with db.connect() as con:
        _get_party_row(con, party_id)
        row = _get_message_row(con, party_id, message_id)
        if row["kind"] != "text":
            raise HTTPException(status_code=400, detail="Only a text message can be edited.")
        _set_message_text(con, message_id, content)
        response = _message_response(
            con,
            con.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM message WHERE id = ?", (message_id,)
            ).fetchone(),
        )
    # After the commit, so the revision job reads the corrected text. A message
    # behind the rolling summary's frontier is never re-sent verbatim and the
    # frontier does not move, so without this the correction would reach the
    # screen and stop there while the summary kept describing the old text.
    # Under the frontier this returns at once.
    summarizer.schedule_revision(party_id, message_id, content)
    return response


@router.put("/parties/{party_id}/messages/{message_id}/variant", response_model=PartyMessage)
def activate_variant(party_id: str, message_id: int, body: VariantInput) -> PartyMessage:
    """Make one archived variant the message's active text."""
    with db.connect() as con:
        _get_party_row(con, party_id)
        _get_message_row(con, party_id, message_id)
        variant = con.execute(
            "SELECT id, content FROM variant WHERE id = ? AND message_id = ?",
            (body.variant_id, message_id),
        ).fetchone()
        if variant is None:
            raise HTTPException(
                status_code=404,
                detail=f"Variant {body.variant_id} not found for message {message_id}",
            )
        _set_message_text(con, message_id, variant["content"], from_variant_id=body.variant_id)
        return _message_response(
            con,
            con.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM message WHERE id = ?", (message_id,)
            ).fetchone(),
        )


# --- Playing a turn -------------------------------------------------------------


@dataclass(frozen=True)
class _TurnContext:
    """Everything the stream needs, gathered before generation starts.

    `persist` is what becomes of the accumulated text once the stream ends —
    appending a reply for a played turn, replacing an archived one for a
    regeneration — so one async generator serves both callers without knowing
    which it is running. `summary` is the rolling summary standing in for the
    covered history, and `summarize` says whether the end of this stream may
    trigger one: a played turn appends to the story, a regeneration replaces
    an existing text and leaves the covered range exactly as it was.

    `opening` marks the one message generated under different rules: a party's
    opening scene comes from `build_opening_messages`, with the direction that
    tells the narrator to set the scene and no history at all. Regenerating it
    through the ordinary turn path sent the model a system prompt and
    literally nothing else, because the history before the first message is
    empty and the direction was never repeated.

    `release` gives the party's turn lock back. It is held from the moment the
    request is accepted until the stream has ended and its text is persisted,
    which is the whole window a second turn must not enter.
    """

    party_id: str
    model: str
    scenario: narrator.PromptScenario
    persona: narrator.PromptPersona | None
    history: list[narrator.HistoryMessage]
    summary: str
    num_ctx: int
    summarize: bool
    persist: Callable[[str], int | None]
    release: Callable[[], None]
    opening: bool = False


def _ndjson_line(payload: dict[str, object]) -> str:
    """One NDJSON record.

    NDJSON rather than Server-Sent Events: the client needs the final message
    id and the real error text, and SSE carries neither without inventing a
    second channel — here every line is a plain JSON object on one response.
    """
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _prepare_turn(party_id: str, body: TurnInput) -> _TurnContext:
    """Validate the turn, persist it, and load the prompt inputs.

    The order matters and is tested: 404 on an unknown party, then the shared
    no-model rule, then 400 on a blank turn — all before anything is written.
    The player's text is written BEFORE generation starts, so it survives
    whatever the model does next, and the connection closes before the stream
    runs: nothing holds a SQLite connection across a model call.

    The turn lock is taken first of all, before the player's text is written,
    because writing it is already half the interleaving. Anything that raises
    from here gives the lock straight back — the turn never happened, so the
    next one must not be refused.
    """
    release = _acquire_turn_lock(party_id)
    try:
        return _prepare_turn_locked(party_id, body, release)
    except BaseException:
        release()
        raise


def _prepare_turn_locked(
    party_id: str, body: TurnInput, release: Callable[[], None]
) -> _TurnContext:
    with db.connect() as con:
        party = _get_party_row(con, party_id)
        model = _require_model(con)
        content = body.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="A turn cannot be empty.")
        scenario = narrator.load_scenario(con, party["scenario_id"])
        persona = narrator.load_active_persona(con)
        num_ctx = settings.get_num_ctx(con)
        now = time.time()
        con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts) "
            "VALUES (?, 'user', 'text', ?, ?)",
            (party_id, content, now),
        )
        # A played turn is story activity: it reorders the most-recently-active
        # list, unlike a rename.
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (now, party_id))
        # Loaded after the insert, so the turn just persisted rides in the
        # history exactly as the model should see it — never hand-appended.
        # `after_id` is the summary frontier: the summarised messages are
        # covered by `summary_text` and must not be re-sent verbatim.
        history = narrator.load_history(
            con,
            party_id,
            settings.get_history_window(con),
            after_id=party["summary_upto"],
        )
    return _TurnContext(
        party_id=party_id,
        model=model,
        scenario=scenario,
        persona=persona,
        history=history,
        summary=party["summary_text"],
        num_ctx=num_ctx,
        summarize=True,
        persist=partial(_persist_reply, party_id),
        release=release,
    )


def _persist_reply(party_id: str, text: str) -> int | None:
    """Save whatever the stream accumulated, on a fresh connection.

    An empty or whitespace-only accumulation appends nothing at all — a blank
    assistant row is noise, while the player's turn above it was persisted on
    purpose. Returns the new message's id, or None when nothing was appended.
    """
    if not text.strip():
        return None
    with db.connect() as con:
        cursor = con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts) "
            "VALUES (?, 'assistant', 'text', ?, ?)",
            (party_id, text, time.time()),
        )
        message_id = cursor.lastrowid
        if message_id is None:
            raise RuntimeError("The reply INSERT succeeded but returned no rowid")
        return message_id


def _persist_regenerated(
    message_id: int, party_id: str, text: str, *, original: str | None = None
) -> int | None:
    """Replace a regenerated message's active text, on a fresh connection.

    An empty or whitespace-only accumulation replaces nothing: the message
    keeps its text, its variants, and which of them is active — the party is
    exactly as it was before the call. A non-empty partial from a broken
    stream is kept, the same rule a played turn's reply follows. Only a text
    that survived `.strip()` is written, and writing it is story activity.

    `original` is the text this regeneration started from, read when the
    request arrived rather than now. Passing it is what keeps an edit that
    landed in between from erasing it — see `_archive_texts`.
    """
    if not text.strip():
        return None
    # The archive inside `_set_message_text` is a read-check-write: the text it
    # saves is the one read from `message.content` moments earlier. SQLite
    # takes the write lock at the first INSERT, not at that read, so on a
    # deferred transaction an edit could commit in between — the pre-edit text
    # would be archived over it and the edited text would exist nowhere. The
    # write lock must therefore be held from before the read: the edit then
    # either committed before it (and is archived) or waits and lands after
    # the commit (and stays the active text); it can no longer fall into the
    # gap. The same lock also keeps two concurrent regenerations from both
    # seeing `already_archived = False` and archiving the original twice.
    with db.connect(immediate=True) as con:
        _set_message_text(con, message_id, text, archive_current=True, original=original)
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (time.time(), party_id))
    # The new text may sit behind the rolling summary's frontier, which does
    # not move: the summary would keep describing the reply just replaced, and
    # the message itself is never re-sent verbatim, so the regeneration would
    # reach the screen and nothing else. Under the frontier this returns at
    # once. See `summarizer.schedule_revision`.
    summarizer.schedule_revision(party_id, message_id, text)
    return message_id


def _prepare_regenerate(party_id: str, message_id: int) -> _TurnContext:
    """Validate a regeneration request and load the prompt inputs.

    Nothing is written here — the old reply must survive even a generation
    that never starts, so the archive happens at persist time, only once new
    text has actually arrived. The history is loaded up to but excluding the
    message being regenerated: with its old text in the prompt, the model
    would continue its own reply instead of writing an alternative.

    The message's current text is read here and carried to persist time. That
    is the text this regeneration is replacing, and reading it now rather than
    later is what stops an edit landing in between from erasing it.

    A regeneration takes the same turn lock a played turn does: it rewrites
    the transcript, and two of those at once on one party is the same defect.
    """
    release = _acquire_turn_lock(party_id)
    try:
        return _prepare_regenerate_locked(party_id, message_id, release)
    except BaseException:
        release()
        raise


def _prepare_regenerate_locked(
    party_id: str, message_id: int, release: Callable[[], None]
) -> _TurnContext:
    with db.connect() as con:
        party = _get_party_row(con, party_id)
        model = _require_model(con)
        row = _get_message_row(con, party_id, message_id)
        if row["role"] != "assistant" or row["kind"] != "text":
            raise HTTPException(
                status_code=400,
                detail="Only a narrator's text message can be regenerated.",
            )
        scenario = narrator.load_scenario(con, party["scenario_id"])
        persona = narrator.load_active_persona(con)
        num_ctx = settings.get_num_ctx(con)
        history = narrator.load_history(
            con,
            party_id,
            settings.get_history_window(con),
            before_id=message_id,
            after_id=party["summary_upto"],
        )
        # The opening scene is the party's first text message: nothing in the
        # story precedes it. It was generated with a direction the ordinary
        # turn path does not carry, and regenerating it without that direction
        # hands the model a system prompt and an empty conversation.
        opening = (
            con.execute(
                "SELECT 1 FROM message WHERE instance_id = ? AND kind = 'text' AND id < ? LIMIT 1",
                (party_id, message_id),
            ).fetchone()
            is None
        )
    return _TurnContext(
        party_id=party_id,
        model=model,
        scenario=scenario,
        persona=persona,
        history=history,
        summary=party["summary_text"],
        num_ctx=num_ctx,
        summarize=False,
        persist=partial(_persist_regenerated, message_id, party_id, original=row["content"]),
        release=release,
        opening=opening,
    )


def _next_fragment(gen: Iterator[str]) -> str | None:
    """One pull of the Ollama stream, for a worker thread.

    StopIteration must never cross a thread boundary — it corrupts the await
    chain — so end-of-stream comes back as None instead.
    """
    try:
        return next(gen)
    except StopIteration:
        return None


async def _turn_events(ctx: _TurnContext) -> AsyncIterator[str]:
    """The response body: one NDJSON line per fragment, then the outcome.

    Async, and pulling the sync Ollama stream itself, for one reason: on a
    client hang-up Starlette cancels its streaming task but never closes a
    sync body iterator, whose `finally` then never runs. Here the
    cancellation lands inside this frame — the pull is abandoned rather than
    shielded — so the `finally` below runs on every ending: normal end,
    broken stream, client hang-up. The persistence itself is shielded, so a
    cancellation mid-unwind cannot interrupt the write either.

    An error line carries the real reason, so the interface can say what
    happened instead of "something failed".
    """
    messages = (
        # Regenerating the opening: the same call that produced it in the
        # first place, direction included, rather than an empty conversation.
        narrator.build_opening_messages(ctx.scenario, ctx.persona)
        if ctx.opening
        else narrator.build_chat_messages(ctx.scenario, ctx.persona, ctx.history, ctx.summary)
    )
    fragments: list[str] = []
    message_id: int | None = None
    stream: Generator[str] | None = None
    try:
        # A real `chat_stream` call cannot raise — it returns a generator and
        # the error only surfaces on the first pull — but the call sits inside
        # the guarded block anyway, so an error at call time from any shape of
        # stream client still becomes an error line rather than a broken route.
        stream = ollama.chat_stream(ctx.model, messages, ctx.num_ctx)
        while True:
            fragment = await anyio.to_thread.run_sync(
                _next_fragment, stream, abandon_on_cancel=True
            )
            if fragment is None:
                break
            fragments.append(fragment)
            yield _ndjson_line({"delta": fragment})
    except ollama.OllamaError as exc:
        yield _ndjson_line({"error": str(exc)})
    finally:
        try:
            # Shielded: the cancellation that ended the stream is still being
            # delivered, and anyio re-raises it at the next checkpoint —
            # without the shield the persist await never starts (its first
            # checkpoint re-raises before the worker thread begins) and the
            # partial is lost. The write is one fast statement, so letting it
            # finish is safe.
            with anyio.CancelScope(shield=True):
                message_id = await anyio.to_thread.run_sync(ctx.persist, "".join(fragments))
                # Only after a persisted reply, and only for a played turn: the
                # scheduler checks the trigger and returns immediately — the model
                # call runs on its own thread, so the stream below closes on the
                # narrator's last token, never on the summariser's.
                if message_id is not None and ctx.summarize:
                    await anyio.to_thread.run_sync(summarizer.schedule, ctx.party_id)
            if stream is not None:
                try:
                    # Stop pulling Ollama when the stream is being abandoned; a
                    # pull still running in a worker thread makes this a no-op.
                    stream.close()
                except ValueError:
                    pass
                # And stop Ollama itself. The close above does nothing at all
                # when a worker thread is mid-pull — which is exactly the
                # case a player pressing stop creates — so the request stays
                # open and the model keeps generating a reply nobody will
                # read, holding the GPU. Closing the response reaches it from
                # here; on a stream that ended normally this is a no-op.
                ollama.close_stream(stream)
        finally:
            # Last of all, and unconditionally: the next turn in this party
            # must not be refused because this one failed to let go.
            ctx.release()
    yield _ndjson_line({"done": True, "message_id": message_id})


@router.post("/parties/{party_id}/messages")
def send_turn(party_id: str, body: TurnInput) -> StreamingResponse:
    """Play one turn: persist the player's text, then stream the narration."""
    return StreamingResponse(
        _turn_events(_prepare_turn(party_id, body)),
        media_type="application/x-ndjson",
    )


@router.post("/parties/{party_id}/messages/{message_id}/regenerate")
def regenerate_message(party_id: str, message_id: int) -> StreamingResponse:
    """Ask for another reply to one narrator message, streamed like a turn.

    The old reply is archived, not overwritten, and only once the new text
    has arrived — see `_prepare_regenerate` and `_persist_regenerated`.
    """
    return StreamingResponse(
        _turn_events(_prepare_regenerate(party_id, message_id)),
        media_type="application/x-ndjson",
    )
