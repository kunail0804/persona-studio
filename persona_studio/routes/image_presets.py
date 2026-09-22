"""Image-prompt presets: the master prompt the composer is given, as data.

The instruction that turns a scene into an image prompt used to be a constant
in `image_prompt.py`, and what it encodes is a grammar: "a short
comma-separated list of English keywords" is how Stable Diffusion is
addressed. Another image model wants prose. So the text is a named preset, one
active at a time, chosen here.

Same shape as personas, deliberately: the active choice lives in `setting`,
deleting the active one clears the setting in the same transaction, and a
dangling id degrades to the built-in instruction rather than breaking a
composition. With no preset at all the application behaves exactly as it did
before this existed.

This is the master prompt of the **image prompt composition**, not the
narrator's system prompt. They are different texts with different jobs, and
nothing here touches narration.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import db, image_prompt, settings

router = APIRouter(tags=["image-presets"])

# The columns a PATCH may write. A whitelist the module owns, never request data.
_UPDATABLE = ("name", "instruction")


class ImagePreset(BaseModel):
    id: str
    name: str
    instruction: str
    created_at: float
    is_active: bool


class ImagePresetInput(BaseModel):
    """A whole preset, for a create."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    instruction: str = ""


class ImagePresetUpdate(BaseModel):
    """A PATCH: only the fields actually sent are written."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    instruction: str | None = None


class ActivePresetInput(BaseModel):
    # None is "use the built-in instruction", which is a real choice and not
    # the absence of one — so it is expressible rather than only reachable by
    # deleting every preset.
    id: str | None = None


class DefaultInstruction(BaseModel):
    instruction: str


def _preset_from_row(row: sqlite3.Row, active_id: str | None) -> ImagePreset:
    return ImagePreset(
        id=row["id"],
        name=row["name"],
        instruction=row["instruction"],
        created_at=row["created_at"],
        is_active=row["id"] == active_id,
    )


def _get_preset_row(con: sqlite3.Connection, preset_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM image_preset WHERE id = ?", (preset_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Image preset {preset_id!r} not found")
    return row


@router.get("/image-presets/default", response_model=DefaultInstruction)
def get_default_instruction() -> DefaultInstruction:
    """The built-in instruction, so the interface can show what it falls back
    to and offer it as a starting point to edit rather than a blank box."""
    return DefaultInstruction(instruction=image_prompt.DEFAULT_INSTRUCTION)


@router.get("/image-presets", response_model=list[ImagePreset])
def list_image_presets() -> list[ImagePreset]:
    with db.connect() as con:
        rows = con.execute("SELECT * FROM image_preset ORDER BY created_at").fetchall()
        active_id = settings.get_active_image_preset_id(con)
    return [_preset_from_row(row, active_id) for row in rows]


@router.post("/image-presets", response_model=ImagePreset, status_code=201)
def create_image_preset(body: ImagePresetInput) -> ImagePreset:
    preset_id = uuid.uuid4().hex
    now = time.time()
    with db.connect() as con:
        con.execute(
            "INSERT INTO image_preset (id, name, instruction, created_at) VALUES (?, ?, ?, ?)",
            (preset_id, body.name, body.instruction, now),
        )
        active_id = settings.get_active_image_preset_id(con)
    return ImagePreset(
        id=preset_id,
        name=body.name,
        instruction=body.instruction,
        created_at=now,
        is_active=preset_id == active_id,
    )


@router.put("/image-presets/active", response_model=list[ImagePreset])
def set_active_image_preset(body: ActivePresetInput) -> list[ImagePreset]:
    """Choose the active preset, or `null` to go back to the built-in text."""
    with db.connect() as con:
        if body.id is not None:
            _get_preset_row(con, body.id)
        settings.set_active_image_preset_id(con, body.id)
        rows = con.execute("SELECT * FROM image_preset ORDER BY created_at").fetchall()
    return [_preset_from_row(row, body.id) for row in rows]


@router.patch("/image-presets/{preset_id}", response_model=ImagePreset)
def update_image_preset(preset_id: str, body: ImagePresetUpdate) -> ImagePreset:
    """Write the fields this request carried, and only those."""
    fields: dict[str, Any] = {
        name: value
        for name, value in body.model_dump(exclude_unset=True).items()
        if value is not None
    }
    clause, values = db.assignments(_UPDATABLE, fields)
    with db.connect() as con:
        _get_preset_row(con, preset_id)
        if clause:
            con.execute(f"UPDATE image_preset SET {clause} WHERE id = ?", (*values, preset_id))
        row = _get_preset_row(con, preset_id)
        active_id = settings.get_active_image_preset_id(con)
    return _preset_from_row(row, active_id)


@router.delete("/image-presets/{preset_id}", status_code=204)
def delete_image_preset(preset_id: str) -> None:
    with db.connect() as con:
        _get_preset_row(con, preset_id)
        con.execute("DELETE FROM image_preset WHERE id = ?", (preset_id,))
        # In the same transaction: a setting pointing at a deleted preset
        # would still compose correctly, but it would lie in the interface.
        if settings.get_active_image_preset_id(con) == preset_id:
            settings.set_active_image_preset_id(con, None)
