"""Character sheets: six fields per character, ordered within a scenario.

`appearance` is documented to the player as physical description only — it is
the sole source used to draw the character. `secrets` is narrator-only; making
the narrator actually respect that is a later pull request's job, this one
just stores and labels the field correctly.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db

router = APIRouter(tags=["characters"])


class Character(BaseModel):
    id: int
    scenario_id: str
    position: int
    name: str
    appearance: str
    personality: str
    story: str
    relationships: str
    secrets: str


class CharacterInput(BaseModel):
    name: str = ""
    appearance: str = ""
    personality: str = ""
    story: str = ""
    relationships: str = ""
    secrets: str = ""


class CharacterOrder(BaseModel):
    order: list[int]


def _character_from_row(row: sqlite3.Row) -> Character:
    return Character(
        id=row["id"],
        scenario_id=row["scenario_id"],
        position=row["position"],
        name=row["name"],
        appearance=row["appearance"],
        personality=row["personality"],
        story=row["story"],
        relationships=row["relationships"],
        secrets=row["secrets"],
    )


def _require_scenario(con: sqlite3.Connection, scenario_id: str) -> None:
    row = con.execute("SELECT 1 FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id!r} not found")


def _get_character_row(con: sqlite3.Connection, scenario_id: str, character_id: int) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM character WHERE id = ? AND scenario_id = ?",
        (character_id, scenario_id),
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Character {character_id} not found in scenario {scenario_id!r}",
        )
    return row


def _resequence_positions(con: sqlite3.Connection, scenario_id: str) -> None:
    rows = con.execute(
        "SELECT id FROM character WHERE scenario_id = ? ORDER BY position",
        (scenario_id,),
    ).fetchall()
    for index, row in enumerate(rows):
        con.execute("UPDATE character SET position = ? WHERE id = ?", (index, row["id"]))


@router.get("/scenarios/{scenario_id}/characters", response_model=list[Character])
def list_characters(scenario_id: str) -> list[Character]:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        rows = con.execute(
            "SELECT * FROM character WHERE scenario_id = ? ORDER BY position",
            (scenario_id,),
        ).fetchall()
    return [_character_from_row(row) for row in rows]


@router.post("/scenarios/{scenario_id}/characters", response_model=Character, status_code=201)
def create_character(scenario_id: str, body: CharacterInput) -> Character:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        # The position is computed inside the INSERT itself, as a correlated
        # subquery, so the read and the write are one statement under one
        # write lock. A separate SELECT beforehand would let two concurrent
        # requests both read the same MAX(position) before either had
        # written, handing out duplicate positions.
        cursor = con.execute(
            """
            INSERT INTO character
                (scenario_id, position, name, appearance, personality,
                 story, relationships, secrets)
            VALUES (
                :scenario_id,
                (SELECT COALESCE(MAX(position) + 1, 0)
                 FROM character WHERE scenario_id = :scenario_id),
                :name, :appearance, :personality, :story, :relationships, :secrets
            )
            """,
            {
                "scenario_id": scenario_id,
                "name": body.name,
                "appearance": body.appearance,
                "personality": body.personality,
                "story": body.story,
                "relationships": body.relationships,
                "secrets": body.secrets,
            },
        )
        character_id = cursor.lastrowid
        if character_id is None:
            raise HTTPException(status_code=500, detail="Character insert did not return an id")
        row = con.execute("SELECT * FROM character WHERE id = ?", (character_id,)).fetchone()
    return _character_from_row(row)


@router.put(
    "/scenarios/{scenario_id}/characters/order",
    response_model=list[Character],
)
def reorder_characters(scenario_id: str, body: CharacterOrder) -> list[Character]:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        existing_ids = {
            row["id"]
            for row in con.execute(
                "SELECT id FROM character WHERE scenario_id = ?", (scenario_id,)
            ).fetchall()
        }
        # A set comparison alone accepts a repeated id as long as the set of
        # distinct ids still matches — the length check catches that case.
        if len(body.order) != len(existing_ids) or set(body.order) != existing_ids:
            raise HTTPException(
                status_code=400,
                detail="The order must list exactly the scenario's current characters, once each",
            )
        for index, character_id in enumerate(body.order):
            con.execute(
                "UPDATE character SET position = ? WHERE id = ? AND scenario_id = ?",
                (index, character_id, scenario_id),
            )
        rows = con.execute(
            "SELECT * FROM character WHERE scenario_id = ? ORDER BY position",
            (scenario_id,),
        ).fetchall()
    return [_character_from_row(row) for row in rows]


@router.patch("/scenarios/{scenario_id}/characters/{character_id:int}", response_model=Character)
def update_character(scenario_id: str, character_id: int, body: CharacterInput) -> Character:
    with db.connect() as con:
        _get_character_row(con, scenario_id, character_id)
        con.execute(
            """
            UPDATE character
            SET name = ?, appearance = ?, personality = ?, story = ?, relationships = ?, secrets = ?
            WHERE id = ? AND scenario_id = ?
            """,
            (
                body.name,
                body.appearance,
                body.personality,
                body.story,
                body.relationships,
                body.secrets,
                character_id,
                scenario_id,
            ),
        )
        row = _get_character_row(con, scenario_id, character_id)
    return _character_from_row(row)


@router.delete("/scenarios/{scenario_id}/characters/{character_id:int}", status_code=204)
def delete_character(scenario_id: str, character_id: int) -> None:
    with db.connect() as con:
        _get_character_row(con, scenario_id, character_id)
        con.execute(
            "DELETE FROM character WHERE id = ? AND scenario_id = ?", (character_id, scenario_id)
        )
        _resequence_positions(con, scenario_id)
