"""Player personas: the protagonist the player plays.

`appearance` is documented to the player as physical description only — it
feeds image prompts later, so it must never carry the persona's name. The
active persona is application-wide, stored in the `setting` table; per-party
personas are a later improvement. Deleting a persona that is active clears the
setting in the same transaction, so no dangling id is ever left behind.
"""

from __future__ import annotations

import sqlite3
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, settings

router = APIRouter(tags=["personas"])


class Persona(BaseModel):
    id: str
    name: str
    description: str
    appearance: str
    traits: str
    created_at: float
    is_active: bool


class PersonaInput(BaseModel):
    name: str = ""
    description: str = ""
    appearance: str = ""
    traits: str = ""


class ActivePersonaInput(BaseModel):
    id: str


def _persona_from_row(row: sqlite3.Row, active_id: str | None) -> Persona:
    return Persona(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        appearance=row["appearance"],
        traits=row["traits"],
        created_at=row["created_at"],
        is_active=row["id"] == active_id,
    )


def _get_persona_row(con: sqlite3.Connection, persona_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM persona WHERE id = ?", (persona_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Persona {persona_id!r} not found")
    return row


@router.get("/personas", response_model=list[Persona])
def list_personas() -> list[Persona]:
    with db.connect() as con:
        active_id = settings.get_active_persona_id(con)
        rows = con.execute("SELECT * FROM persona ORDER BY created_at").fetchall()
    return [_persona_from_row(row, active_id) for row in rows]


@router.put("/personas/active", response_model=Persona)
def set_active_persona(body: ActivePersonaInput) -> Persona:
    with db.connect() as con:
        row = _get_persona_row(con, body.id)
        settings.set_active_persona_id(con, body.id)
    return _persona_from_row(row, body.id)


@router.post("/personas", response_model=Persona, status_code=201)
def create_persona(body: PersonaInput) -> Persona:
    persona_id = uuid.uuid4().hex
    now = time.time()
    with db.connect() as con:
        con.execute(
            "INSERT INTO persona (id, name, description, appearance, traits, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (persona_id, body.name, body.description, body.appearance, body.traits, now),
        )
    return Persona(
        id=persona_id,
        name=body.name,
        description=body.description,
        appearance=body.appearance,
        traits=body.traits,
        created_at=now,
        is_active=False,
    )


@router.get("/personas/{persona_id}", response_model=Persona)
def get_persona(persona_id: str) -> Persona:
    with db.connect() as con:
        row = _get_persona_row(con, persona_id)
        active_id = settings.get_active_persona_id(con)
    return _persona_from_row(row, active_id)


@router.patch("/personas/{persona_id}", response_model=Persona)
def update_persona(persona_id: str, body: PersonaInput) -> Persona:
    with db.connect() as con:
        _get_persona_row(con, persona_id)
        con.execute(
            "UPDATE persona SET name = ?, description = ?, appearance = ?, traits = ? WHERE id = ?",
            (body.name, body.description, body.appearance, body.traits, persona_id),
        )
        row = _get_persona_row(con, persona_id)
        active_id = settings.get_active_persona_id(con)
    return _persona_from_row(row, active_id)


@router.delete("/personas/{persona_id}", status_code=204)
def delete_persona(persona_id: str) -> None:
    with db.connect() as con:
        _get_persona_row(con, persona_id)
        con.execute("DELETE FROM persona WHERE id = ?", (persona_id,))
        if settings.get_active_persona_id(con) == persona_id:
            settings.set_active_persona_id(con, None)
