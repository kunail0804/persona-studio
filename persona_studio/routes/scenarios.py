"""Scenario CRUD: the reusable world definition every party replays.

`world_rules` and `arcs` are written here and read by `narrator.load_scenario`:
they are the two biggest levers a roleplay scenario has, and the prompt builder
omits either one when it is blank, so leaving them empty costs nothing.

`PATCH` means partial. Every update model below has optional fields and only
the ones actually sent are written — a body that omits a field leaves the
stored value alone instead of resetting it, and an unknown key is refused
rather than dropped.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import db
from ..images import delete_orphan_images, unlink_images

router = APIRouter(tags=["scenarios"])

DEFAULT_TITLE = "Sans titre"

# The columns a PATCH may write, in the order they are written. A whitelist the
# module owns, never anything that came from the request.
_UPDATABLE = ("title", "synopsis", "world_rules", "arcs")


def _title(raw: str) -> str:
    """The stored title: the stripped input, or the default when blank.

    A blank title is not harmless: `_party_label` falls back to it for a party
    with no name of its own, so an empty title produces exactly the nameless
    row in a list that the fallback exists to prevent.
    """
    return raw.strip() or DEFAULT_TITLE


class ScenarioSummary(BaseModel):
    id: str
    title: str
    synopsis: str
    character_count: int


class Scenario(BaseModel):
    id: str
    title: str
    synopsis: str
    world_rules: str
    arcs: str
    created_at: float
    updated_at: float


class ScenarioInput(BaseModel):
    """A whole scenario, for a create: every field has a default."""

    model_config = ConfigDict(extra="forbid")

    title: str = DEFAULT_TITLE
    synopsis: str = ""
    world_rules: str = ""
    arcs: str = ""


class ScenarioUpdate(BaseModel):
    """A PATCH: only the fields actually sent are written.

    `None` is the "not sent" marker rather than a value, so a field left out —
    or explicitly null — keeps whatever is stored. `extra="forbid"` makes a
    mistyped key a 422 instead of a silent no-op that answers 200.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    synopsis: str | None = None
    world_rules: str | None = None
    arcs: str | None = None


def _scenario_from_row(row: sqlite3.Row) -> Scenario:
    return Scenario(
        id=row["id"],
        title=row["title"],
        synopsis=row["synopsis"],
        world_rules=row["world_rules"],
        arcs=row["arcs"],
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
    title = _title(body.title)
    now = time.time()
    with db.connect() as con:
        con.execute(
            "INSERT INTO scenario (id, title, synopsis, world_rules, arcs, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scenario_id, title, body.synopsis, body.world_rules, body.arcs, now, now),
        )
    return Scenario(
        id=scenario_id,
        title=title,
        synopsis=body.synopsis,
        world_rules=body.world_rules,
        arcs=body.arcs,
        created_at=now,
        updated_at=now,
    )


@router.get("/scenarios/{scenario_id}", response_model=Scenario)
def get_scenario(scenario_id: str) -> Scenario:
    with db.connect() as con:
        row = _get_scenario_row(con, scenario_id)
    return _scenario_from_row(row)


@router.patch("/scenarios/{scenario_id}", response_model=Scenario)
def update_scenario(scenario_id: str, body: ScenarioUpdate) -> Scenario:
    """Write the fields this request carried, and only those."""
    fields: dict[str, Any] = {
        name: value
        for name, value in body.model_dump(exclude_unset=True).items()
        if value is not None
    }
    if "title" in fields:
        fields["title"] = _title(fields["title"])
    clause, values = db.assignments(_UPDATABLE, fields)
    now = time.time()
    with db.connect() as con:
        _get_scenario_row(con, scenario_id)
        if clause:
            con.execute(
                f"UPDATE scenario SET {clause}, updated_at = ? WHERE id = ?",
                (*values, now, scenario_id),
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
