"""Places of a scenario: name, visual description, sensory atmosphere.

The two text fields have separate jobs. `description` is factual and visual:
it is what an image prompt receives in place of the place's name, the same way
a character's appearance stands in for theirs. `atmosphere` is sensory, for
the narrator. Places ride in every narrator prompt, like characters — the
narrator has to know a place exists to take the story there.

`PATCH` means partial, like every other update route here.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import db

router = APIRouter(tags=["places"])

# The columns a PATCH may write. A whitelist the module owns, never request data.
_UPDATABLE = ("name", "description", "atmosphere")


class Place(BaseModel):
    id: int
    scenario_id: str
    position: int
    name: str
    description: str
    atmosphere: str


class PlaceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    description: str = ""
    atmosphere: str = ""


class PlaceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    atmosphere: str | None = None


def _place_from_row(row: sqlite3.Row) -> Place:
    return Place(
        id=row["id"],
        scenario_id=row["scenario_id"],
        position=row["position"],
        name=row["name"],
        description=row["description"],
        atmosphere=row["atmosphere"],
    )


def _require_scenario(con: sqlite3.Connection, scenario_id: str) -> None:
    if con.execute("SELECT 1 FROM scenario WHERE id = ?", (scenario_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id!r} not found")


def _get_place_row(con: sqlite3.Connection, scenario_id: str, place_id: int) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM place WHERE id = ? AND scenario_id = ?", (place_id, scenario_id)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Place {place_id} not found in scenario {scenario_id!r}"
        )
    return row


@router.get("/scenarios/{scenario_id}/places", response_model=list[Place])
def list_places(scenario_id: str) -> list[Place]:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        rows = con.execute(
            "SELECT * FROM place WHERE scenario_id = ? ORDER BY position", (scenario_id,)
        ).fetchall()
    return [_place_from_row(row) for row in rows]


@router.post("/scenarios/{scenario_id}/places", response_model=Place, status_code=201)
def create_place(scenario_id: str, body: PlaceInput) -> Place:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        # The position is computed inside the INSERT, one statement under one
        # write lock — the rule `create_character` learned from a race.
        cursor = con.execute(
            """
            INSERT INTO place (scenario_id, position, name, description, atmosphere)
            VALUES (
                :scenario_id,
                (SELECT COALESCE(MAX(position) + 1, 0) FROM place WHERE scenario_id = :scenario_id),
                :name, :description, :atmosphere
            )
            """,
            {"scenario_id": scenario_id, **body.model_dump()},
        )
        place_id = cursor.lastrowid
        if place_id is None:
            raise HTTPException(status_code=500, detail="Place insert did not return an id")
        row = con.execute("SELECT * FROM place WHERE id = ?", (place_id,)).fetchone()
    return _place_from_row(row)


@router.patch("/scenarios/{scenario_id}/places/{place_id:int}", response_model=Place)
def update_place(scenario_id: str, place_id: int, body: PlaceUpdate) -> Place:
    """Write the fields this request carried, and only those."""
    fields: dict[str, Any] = {
        name: value
        for name, value in body.model_dump(exclude_unset=True).items()
        if value is not None
    }
    clause, values = db.assignments(_UPDATABLE, fields)
    with db.connect() as con:
        _get_place_row(con, scenario_id, place_id)
        if clause:
            con.execute(
                f"UPDATE place SET {clause} WHERE id = ? AND scenario_id = ?",
                (*values, place_id, scenario_id),
            )
        row = _get_place_row(con, scenario_id, place_id)
    return _place_from_row(row)


@router.delete("/scenarios/{scenario_id}/places/{place_id:int}", status_code=204)
def delete_place(scenario_id: str, place_id: int) -> None:
    with db.connect() as con:
        _get_place_row(con, scenario_id, place_id)
        con.execute("DELETE FROM place WHERE id = ? AND scenario_id = ?", (place_id, scenario_id))
