"""Scenario CRUD: the reusable world definition every party replays."""

from __future__ import annotations

import sqlite3
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db
from ..images import delete_orphan_images, unlink_images

router = APIRouter(tags=["scenarios"])


class ScenarioSummary(BaseModel):
    id: str
    title: str
    synopsis: str
    character_count: int


class Scenario(BaseModel):
    id: str
    title: str
    synopsis: str
    created_at: float
    updated_at: float


class ScenarioInput(BaseModel):
    title: str = "Sans titre"
    synopsis: str = ""


def _scenario_from_row(row: sqlite3.Row) -> Scenario:
    return Scenario(
        id=row["id"],
        title=row["title"],
        synopsis=row["synopsis"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _get_scenario_row(con: sqlite3.Connection, scenario_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id!r} not found")
    return row


@router.get("/scenarios", response_model=list[ScenarioSummary])
def list_scenarios() -> list[ScenarioSummary]:
    with db.connect() as con:
        rows = con.execute(
            """
            SELECT scenario.id, scenario.title, scenario.synopsis,
                   COUNT(character.id) AS character_count
            FROM scenario
            LEFT JOIN character ON character.scenario_id = scenario.id
            GROUP BY scenario.id
            ORDER BY scenario.updated_at DESC
            """
        ).fetchall()
    return [
        ScenarioSummary(
            id=row["id"],
            title=row["title"],
            synopsis=row["synopsis"],
            character_count=row["character_count"],
        )
        for row in rows
    ]


@router.post("/scenarios", response_model=Scenario, status_code=201)
def create_scenario(body: ScenarioInput) -> Scenario:
    scenario_id = uuid.uuid4().hex
    now = time.time()
    with db.connect() as con:
        con.execute(
            "INSERT INTO scenario (id, title, synopsis, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (scenario_id, body.title, body.synopsis, now, now),
        )
    return Scenario(
        id=scenario_id, title=body.title, synopsis=body.synopsis, created_at=now, updated_at=now
    )


@router.get("/scenarios/{scenario_id}", response_model=Scenario)
def get_scenario(scenario_id: str) -> Scenario:
    with db.connect() as con:
        row = _get_scenario_row(con, scenario_id)
    return _scenario_from_row(row)


@router.patch("/scenarios/{scenario_id}", response_model=Scenario)
def update_scenario(scenario_id: str, body: ScenarioInput) -> Scenario:
    now = time.time()
    with db.connect() as con:
        _get_scenario_row(con, scenario_id)
        con.execute(
            "UPDATE scenario SET title = ?, synopsis = ?, updated_at = ? WHERE id = ?",
            (body.title, body.synopsis, now, scenario_id),
        )
        row = _get_scenario_row(con, scenario_id)
    return _scenario_from_row(row)


@router.delete("/scenarios/{scenario_id}", status_code=204)
def delete_scenario(scenario_id: str) -> None:
    with db.connect() as con:
        _get_scenario_row(con, scenario_id)
        # `ON DELETE CASCADE` removes the scenario's characters, parties and
        # messages, but not the `image` rows they were the last reference to
        # — nothing in the schema points from an image back to its owner.
        con.execute("DELETE FROM scenario WHERE id = ?", (scenario_id,))
        orphan_ids = delete_orphan_images(con)
    unlink_images(orphan_ids)
