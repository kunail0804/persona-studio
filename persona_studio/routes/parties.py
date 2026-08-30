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
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Generator, Iterator
from dataclasses import dataclass
from typing import Literal

import anyio.to_thread
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import db, narrator, ollama, settings

router = APIRouter(tags=["parties"])


class PartyMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    ts: float


class PartySummary(BaseModel):
    id: str
    scenario_id: str
    scenario_title: str
    label: str
    created_at: float
    updated_at: float


class Party(PartySummary):
    messages: list[PartyMessage]


class PartyInput(BaseModel):
    label: str = ""


class TurnInput(BaseModel):
    content: str


_SUMMARY_COLUMNS = (
    "instance.id, instance.scenario_id, instance.label, "
    "instance.created_at, instance.updated_at, scenario.title AS scenario_title"
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
            "SELECT id, role, content, ts FROM message WHERE instance_id = ? ORDER BY id",
            (party_id,),
        ).fetchall()
    return Party(
        id=row["id"],
        scenario_id=row["scenario_id"],
        scenario_title=row["scenario_title"],
        label=row["label"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        messages=[
            PartyMessage(id=m["id"], role=m["role"], content=m["content"], ts=m["ts"])
            for m in message_rows
        ],
    )


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


# --- Playing a turn -------------------------------------------------------------


@dataclass(frozen=True)
class _TurnContext:
    """Everything the turn stream needs, gathered before generation starts."""

    party_id: str
    model: str
    scenario: narrator.PromptScenario
    persona: narrator.PromptPersona | None
    history: list[narrator.HistoryMessage]
    num_ctx: int


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
    """
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
        history = narrator.load_history(con, party_id, settings.get_history_window(con))
    return _TurnContext(
        party_id=party_id,
        model=model,
        scenario=scenario,
        persona=persona,
        history=history,
        num_ctx=num_ctx,
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
    messages = narrator.build_chat_messages(ctx.scenario, ctx.persona, ctx.history)
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
        # Shielded: the cancellation that ended the stream is still being
        # delivered, and anyio re-raises it at the next checkpoint — without
        # the shield the persist await never starts (its first checkpoint
        # re-raises before the worker thread begins) and the partial is
        # lost. The write is one fast INSERT, so letting it finish is safe.
        with anyio.CancelScope(shield=True):
            message_id = await anyio.to_thread.run_sync(
                _persist_reply, ctx.party_id, "".join(fragments)
            )
        if stream is not None:
            try:
                # Stop pulling Ollama when the stream is being abandoned; a
                # pull still running in a worker thread makes this a no-op.
                stream.close()
            except ValueError:
                pass
    yield _ndjson_line({"done": True, "message_id": message_id})


@router.post("/parties/{party_id}/messages")
def send_turn(party_id: str, body: TurnInput) -> StreamingResponse:
    """Play one turn: persist the player's text, then stream the narration."""
    return StreamingResponse(
        _turn_events(_prepare_turn(party_id, body)),
        media_type="application/x-ndjson",
    )
