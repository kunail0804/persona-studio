"""The lorebook: world knowledge that enters the prompt only when relevant.

An entry is a list of keywords and a text. It reaches the narrator on a turn
only if one of its keywords appears in the recent messages — see
`narrator.select_lore`. That is what lets a world hold many facts without
every prompt carrying all of them.

An object that must exist and come back is an entry keyed by its own name;
there is no separate item editor.

`PATCH` means partial, like every other update route here.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import db, narrator

router = APIRouter(tags=["lore"])


class LoreEntry(BaseModel):
    id: int
    scenario_id: str
    position: int
    keywords: list[str]
    text: str


class LoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = []
    text: str = ""


class LoreUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] | None = None
    text: str | None = None


def normalise_keywords(keywords: list[str]) -> list[str]:
    """Stripped, blanks dropped, repeats dropped case-insensitively, order kept.

    Matching is case-insensitive, so "Ordre" and "ordre" are one keyword; a
    blank one could never match anything and would only confuse the editor.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for keyword in keywords:
        cleaned = keyword.strip()
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            kept.append(cleaned)
    return kept


def _entry_from_row(row: sqlite3.Row) -> LoreEntry:
    return LoreEntry(
        id=row["id"],
        scenario_id=row["scenario_id"],
        position=row["position"],
        keywords=list(narrator.parse_keywords(row["keywords"])),
        text=row["text"],
    )


def _require_scenario(con: sqlite3.Connection, scenario_id: str) -> None:
    if con.execute("SELECT 1 FROM scenario WHERE id = ?", (scenario_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id!r} not found")


def _get_entry_row(con: sqlite3.Connection, scenario_id: str, entry_id: int) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM lore WHERE id = ? AND scenario_id = ?", (entry_id, scenario_id)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Lore entry {entry_id} not found in scenario {scenario_id!r}",
        )
    return row


@router.get("/scenarios/{scenario_id}/lore", response_model=list[LoreEntry])
def list_lore(scenario_id: str) -> list[LoreEntry]:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        rows = con.execute(
            "SELECT * FROM lore WHERE scenario_id = ? ORDER BY position", (scenario_id,)
        ).fetchall()
    return [_entry_from_row(row) for row in rows]


@router.post("/scenarios/{scenario_id}/lore", response_model=LoreEntry, status_code=201)
def create_lore(scenario_id: str, body: LoreInput) -> LoreEntry:
    with db.connect() as con:
        _require_scenario(con, scenario_id)
        cursor = con.execute(
            """
            INSERT INTO lore (scenario_id, position, keywords, text)
            VALUES (
                :scenario_id,
                (SELECT COALESCE(MAX(position) + 1, 0) FROM lore WHERE scenario_id = :scenario_id),
                :keywords, :text
            )
            """,
            {
                "scenario_id": scenario_id,
                "keywords": json.dumps(normalise_keywords(body.keywords), ensure_ascii=False),
                "text": body.text,
            },
        )
        entry_id = cursor.lastrowid
        if entry_id is None:
            raise HTTPException(status_code=500, detail="Lore insert did not return an id")
        row = con.execute("SELECT * FROM lore WHERE id = ?", (entry_id,)).fetchone()
    return _entry_from_row(row)


@router.patch("/scenarios/{scenario_id}/lore/{entry_id:int}", response_model=LoreEntry)
def update_lore(scenario_id: str, entry_id: int, body: LoreUpdate) -> LoreEntry:
    """Write the fields this request carried, and only those."""
    fields: dict[str, Any] = {}
    if body.keywords is not None:
        fields["keywords"] = json.dumps(normalise_keywords(body.keywords), ensure_ascii=False)
    if body.text is not None:
        fields["text"] = body.text
    clause, values = db.assignments(("keywords", "text"), fields)
    with db.connect() as con:
        _get_entry_row(con, scenario_id, entry_id)
        if clause:
            con.execute(
                f"UPDATE lore SET {clause} WHERE id = ? AND scenario_id = ?",
                (*values, entry_id, scenario_id),
            )
        row = _get_entry_row(con, scenario_id, entry_id)
    return _entry_from_row(row)


@router.delete("/scenarios/{scenario_id}/lore/{entry_id:int}", status_code=204)
def delete_lore(scenario_id: str, entry_id: int) -> None:
    with db.connect() as con:
        _get_entry_row(con, scenario_id, entry_id)
        con.execute("DELETE FROM lore WHERE id = ? AND scenario_id = ?", (entry_id, scenario_id))
