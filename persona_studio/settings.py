"""Typed access to the `setting` table.

Every value lives in `setting.value` as JSON. These accessors are the single
read path the narrator and the settings routes share, so the rules live in one
place: a malformed or missing value falls back to the default instead of
raising — a hand-edited or half-written database must never break a request.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

# The context window sent to Ollama on every call, and the bounds the settings
# route accepts. Ollama truncates silently from the front when the prompt
# exceeds it, so the default errs generous rather than tiny.
DEFAULT_NUM_CTX = 8192
MIN_NUM_CTX = 512
MAX_NUM_CTX = 1_048_576

# How many recent messages the narrator sees. The bound is what keeps a long
# party's prompt from growing until the model truncates it.
DEFAULT_HISTORY_WINDOW = 20
MIN_HISTORY_WINDOW = 2
MAX_HISTORY_WINDOW = 200

LLM_MODEL_KEY = "llm.model"
NUM_CTX_KEY = "llm.num_ctx"
HISTORY_WINDOW_KEY = "narrator.history_window"
ACTIVE_PERSONA_KEY = "persona.active_id"
# The comfy-gateway template images are made with, by name, and the gateway
# preset applied on top of it. `workflow.active_id`, the key of the workflows
# this application used to store itself, is no longer read; its rows stay in
# the database, untouched.
IMAGE_TEMPLATE_KEY = "image.gateway_template"
IMAGE_GATEWAY_PRESET_KEY = "image.gateway_preset"
IMAGE_PRESET_ACTIVE_KEY = "image_prompt.active_preset_id"


def _read(con: sqlite3.Connection, key: str) -> Any:
    row = con.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return None


def _write(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute(
        "INSERT INTO setting (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )


def _clear(con: sqlite3.Connection, key: str) -> None:
    con.execute("DELETE FROM setting WHERE key = ?", (key,))


def _valid_bounded_int(value: Any, minimum: int, maximum: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        # `bool` is a subclass of `int`: `True` must not pass for 1.
        return False
    return minimum <= value <= maximum


def get_llm_model(con: sqlite3.Connection) -> str | None:
    """The configured model name, or None. Anything but a non-empty string is None."""
    value = _read(con, LLM_MODEL_KEY)
    return value if isinstance(value, str) and value else None


def set_llm_model(con: sqlite3.Connection, model: str | None) -> None:
    if model is None:
        _clear(con, LLM_MODEL_KEY)
    else:
        _write(con, LLM_MODEL_KEY, model)


def get_num_ctx(con: sqlite3.Connection) -> int:
    value = _read(con, NUM_CTX_KEY)
    return value if _valid_bounded_int(value, MIN_NUM_CTX, MAX_NUM_CTX) else DEFAULT_NUM_CTX


def set_num_ctx(con: sqlite3.Connection, num_ctx: int) -> None:
    _write(con, NUM_CTX_KEY, num_ctx)


def get_history_window(con: sqlite3.Connection) -> int:
    value = _read(con, HISTORY_WINDOW_KEY)
    return (
        value
        if _valid_bounded_int(value, MIN_HISTORY_WINDOW, MAX_HISTORY_WINDOW)
        else DEFAULT_HISTORY_WINDOW
    )


def set_history_window(con: sqlite3.Connection, window: int) -> None:
    _write(con, HISTORY_WINDOW_KEY, window)


def get_active_persona_id(con: sqlite3.Connection) -> str | None:
    value = _read(con, ACTIVE_PERSONA_KEY)
    return value if isinstance(value, str) and value else None


def set_active_persona_id(con: sqlite3.Connection, persona_id: str | None) -> None:
    if persona_id is None:
        _clear(con, ACTIVE_PERSONA_KEY)
    else:
        _write(con, ACTIVE_PERSONA_KEY, persona_id)


def get_image_template(con: sqlite3.Connection) -> str | None:
    """The comfy-gateway template name images are made with, or None."""
    value = _read(con, IMAGE_TEMPLATE_KEY)
    return value if isinstance(value, str) and value else None


def get_image_gateway_preset(con: sqlite3.Connection) -> str | None:
    """The comfy-gateway preset applied to the template, or None for its own values."""
    value = _read(con, IMAGE_GATEWAY_PRESET_KEY)
    return value if isinstance(value, str) and value else None


def set_image_generation(con: sqlite3.Connection, template: str | None, preset: str | None) -> None:
    """Both at once: a preset belongs to one template's name, so it never
    outlives a change of template."""
    if template is None:
        _clear(con, IMAGE_TEMPLATE_KEY)
    else:
        _write(con, IMAGE_TEMPLATE_KEY, template)
    if template is None or preset is None:
        _clear(con, IMAGE_GATEWAY_PRESET_KEY)
    else:
        _write(con, IMAGE_GATEWAY_PRESET_KEY, preset)


def get_active_image_preset_id(con: sqlite3.Connection) -> str | None:
    """The chosen image-prompt preset, or None. None means the built-in text."""
    value = _read(con, IMAGE_PRESET_ACTIVE_KEY)
    return value if isinstance(value, str) and value else None


def set_active_image_preset_id(con: sqlite3.Connection, preset_id: str | None) -> None:
    if preset_id is None:
        _clear(con, IMAGE_PRESET_ACTIVE_KEY)
    else:
        _write(con, IMAGE_PRESET_ACTIVE_KEY, preset_id)
