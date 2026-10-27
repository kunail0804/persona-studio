"""The world of one party: its own characters, places and lorebook.

A party receives a copy of its scenario's characters, places and lore when it
is created (schema v5). The player edits that copy while playing: create a
character the story needs, change a place, drop a lore entry. The scenario and
the other parties never see these edits. Title, synopsis, rules and arcs stay
in the scenario and are not editable here.

The narrator reads this copy through `narrator.load_party_scenario`, and the
image prompt through `image_prompt.load_inputs`, so an edit reaches the next
turn and the next image composition.

Validation is the scenario routes' own: the same input models, the same
keyword normalisation, the same partial PATCH.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, narrator
from .characters import CharacterInput, CharacterUpdate
from .lore import LoreInput, LoreUpdate, normalise_keywords
from .places import PlaceInput, PlaceUpdate

router = APIRouter(tags=["party world"])

# The columns a PATCH may write. Whitelists the module owns, never request data.
_CHARACTER_COLUMNS = ("name", "appearance", "personality", "story", "relationships", "secrets")
_PLACE_COLUMNS = ("name", "description", "atmosphere")
_LORE_COLUMNS = ("keywords", "text")

# What a 404 calls each table's row.
_ROW_LABELS = {"party_character": "Character", "party_place": "Place", "party_lore": "Lore entry"}


class PartyCharacter(BaseModel):
    id: int
    position: int
    name: str
    appearance: str
    personality: str
    story: str
    relationships: str
    secrets: str


class PartyPlace(BaseModel):
    id: int
    position: int
    name: str
    description: str
    atmosphere: str


class PartyLoreEntry(BaseModel):
    id: int
    position: int
    keywords: list[str]
    text: str


class PartyWorld(BaseModel):
    characters: list[PartyCharacter]
    places: list[PartyPlace]
    lore: list[PartyLoreEntry]


def _character_from_row(row: sqlite3.Row) -> PartyCharacter:
    return PartyCharacter(
        id=row["id"],
        position=row["position"],
        name=row["name"],
        appearance=row["appearance"],
        personality=row["personality"],
        story=row["story"],
        relationships=row["relationships"],
        secrets=row["secrets"],
    )


def _place_from_row(row: sqlite3.Row) -> PartyPlace:
    return PartyPlace(
        id=row["id"],
        position=row["position"],
        name=row["name"],
        description=row["description"],
        atmosphere=row["atmosphere"],
    )


def _lore_from_row(row: sqlite3.Row) -> PartyLoreEntry:
    return PartyLoreEntry(
        id=row["id"],
        position=row["position"],
        keywords=list(narrator.parse_keywords(row["keywords"])),
        text=row["text"],
    )


def _require_party(con: sqlite3.Connection, party_id: str) -> None:
    if con.execute("SELECT 1 FROM instance WHERE id = ?", (party_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"Party {party_id!r} not found")


def _get_row(con: sqlite3.Connection, table: str, party_id: str, row_id: int) -> sqlite3.Row:
    """The party's row in `table`, or 404. `table` is one of this module's constants."""
    row = con.execute(
        f"SELECT * FROM {table} WHERE id = ? AND instance_id = ?", (row_id, party_id)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"{_ROW_LABELS[table]} {row_id} not found in party {party_id!r}"
        )
    return row


def _insert(con: sqlite3.Connection, table: str, party_id: str, fields: dict[str, Any]) -> int:
    """Append a row at the end of the party's list and return its id."""
    names = list(fields)
    cursor = con.execute(
        f"INSERT INTO {table} (instance_id, position, {', '.join(names)}) "
        f"VALUES (?, (SELECT COALESCE(MAX(position) + 1, 0) FROM {table} WHERE instance_id = ?), "
        f"{', '.join('?' for _ in names)})",
        (party_id, party_id, *fields.values()),
    )
    if cursor.lastrowid is None:
        raise HTTPException(status_code=500, detail=f"{table} insert did not return an id")
    return cursor.lastrowid


def _update(
    con: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    party_id: str,
    row_id: int,
    fields: dict[str, Any],
) -> sqlite3.Row:
    """Write the fields the request carried, and only those."""
    _get_row(con, table, party_id, row_id)
    clause, values = db.assignments(columns, fields)
    if clause:
        con.execute(
            f"UPDATE {table} SET {clause} WHERE id = ? AND instance_id = ?",
            (*values, row_id, party_id),
        )
    return _get_row(con, table, party_id, row_id)


def _delete(con: sqlite3.Connection, table: str, party_id: str, row_id: int) -> None:
    _get_row(con, table, party_id, row_id)
    con.execute(f"DELETE FROM {table} WHERE id = ? AND instance_id = ?", (row_id, party_id))


# --- The whole world, in one read ----------------------------------------------


@router.get("/parties/{party_id}/world", response_model=PartyWorld)
def get_party_world(party_id: str) -> PartyWorld:
    with db.connect() as con:
        _require_party(con, party_id)
        characters = con.execute(
            "SELECT * FROM party_character WHERE instance_id = ? ORDER BY position", (party_id,)
        ).fetchall()
        places = con.execute(
            "SELECT * FROM party_place WHERE instance_id = ? ORDER BY position", (party_id,)
        ).fetchall()
        lore = con.execute(
            "SELECT * FROM party_lore WHERE instance_id = ? ORDER BY position", (party_id,)
        ).fetchall()
    return PartyWorld(
        characters=[_character_from_row(row) for row in characters],
        places=[_place_from_row(row) for row in places],
        lore=[_lore_from_row(row) for row in lore],
    )


# --- Characters --------------------------------------------------------------------


@router.post("/parties/{party_id}/characters", response_model=PartyCharacter, status_code=201)
def create_party_character(party_id: str, body: CharacterInput) -> PartyCharacter:
    with db.connect() as con:
        _require_party(con, party_id)
        row_id = _insert(con, "party_character", party_id, body.model_dump())
        row = _get_row(con, "party_character", party_id, row_id)
    return _character_from_row(row)


@router.patch("/parties/{party_id}/characters/{character_id:int}", response_model=PartyCharacter)
def update_party_character(
    party_id: str, character_id: int, body: CharacterUpdate
) -> PartyCharacter:
    fields = {name: value for name, value in body.model_dump().items() if value is not None}
    with db.connect() as con:
        row = _update(con, "party_character", _CHARACTER_COLUMNS, party_id, character_id, fields)
    return _character_from_row(row)


@router.delete("/parties/{party_id}/characters/{character_id:int}", status_code=204)
def delete_party_character(party_id: str, character_id: int) -> None:
    with db.connect() as con:
        _delete(con, "party_character", party_id, character_id)


# --- Places ----------------------------------------------------------------------


@router.post("/parties/{party_id}/places", response_model=PartyPlace, status_code=201)
def create_party_place(party_id: str, body: PlaceInput) -> PartyPlace:
    with db.connect() as con:
        _require_party(con, party_id)
        row_id = _insert(con, "party_place", party_id, body.model_dump())
        row = _get_row(con, "party_place", party_id, row_id)
    return _place_from_row(row)


@router.patch("/parties/{party_id}/places/{place_id:int}", response_model=PartyPlace)
def update_party_place(party_id: str, place_id: int, body: PlaceUpdate) -> PartyPlace:
    fields = {name: value for name, value in body.model_dump().items() if value is not None}
    with db.connect() as con:
        row = _update(con, "party_place", _PLACE_COLUMNS, party_id, place_id, fields)
    return _place_from_row(row)


@router.delete("/parties/{party_id}/places/{place_id:int}", status_code=204)
def delete_party_place(party_id: str, place_id: int) -> None:
    with db.connect() as con:
        _delete(con, "party_place", party_id, place_id)


# --- Lore ------------------------------------------------------------------------


@router.post("/parties/{party_id}/lore", response_model=PartyLoreEntry, status_code=201)
def create_party_lore(party_id: str, body: LoreInput) -> PartyLoreEntry:
    fields = {
        "keywords": json.dumps(normalise_keywords(body.keywords), ensure_ascii=False),
        "text": body.text,
    }
    with db.connect() as con:
        _require_party(con, party_id)
        row_id = _insert(con, "party_lore", party_id, fields)
        row = _get_row(con, "party_lore", party_id, row_id)
    return _lore_from_row(row)


@router.patch("/parties/{party_id}/lore/{entry_id:int}", response_model=PartyLoreEntry)
def update_party_lore(party_id: str, entry_id: int, body: LoreUpdate) -> PartyLoreEntry:
    fields: dict[str, Any] = {}
    if body.keywords is not None:
        fields["keywords"] = json.dumps(normalise_keywords(body.keywords), ensure_ascii=False)
    if body.text is not None:
        fields["text"] = body.text
    with db.connect() as con:
        row = _update(con, "party_lore", _LORE_COLUMNS, party_id, entry_id, fields)
    return _lore_from_row(row)


@router.delete("/parties/{party_id}/lore/{entry_id:int}", status_code=204)
def delete_party_lore(party_id: str, entry_id: int) -> None:
    with db.connect() as con:
        _delete(con, "party_lore", party_id, entry_id)
