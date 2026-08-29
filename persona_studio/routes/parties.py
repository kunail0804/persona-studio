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
"""

from __future__ import annotations

import sqlite3
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, narrator, ollama, settings

router = APIRouter(tags=["parties"])


class PartyMessage(BaseModel):
    id: int
    role: str
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
    with db.connect() as con:
        con.execute(
            "INSERT INTO instance (id, scenario_id, label, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (party_id, scenario_id, body.label.strip() or scenario_row["title"], now, now),
        )
        con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts) "
            "VALUES (?, 'assistant', 'text', ?, ?)",
            (party_id, opening, now),
        )
    return PartySummary(
        id=party_id,
        scenario_id=scenario_id,
        scenario_title=scenario_row["title"],
        label=body.label.strip() or scenario_row["title"],
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
        _get_party_row(con, party_id)
        # A rename is metadata, not story activity: `updated_at` stays put, so
        # renaming never reorders the most-recently-active list.
        con.execute("UPDATE instance SET label = ? WHERE id = ?", (body.label, party_id))
        row = _get_party_row(con, party_id)
    return _party_from_row(row)


@router.delete("/parties/{party_id}", status_code=204)
def delete_party(party_id: str) -> None:
    with db.connect() as con:
        _get_party_row(con, party_id)
        # The schema's `ON DELETE CASCADE` removes the party's messages with it.
        con.execute("DELETE FROM instance WHERE id = ?", (party_id,))
