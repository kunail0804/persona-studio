"""Application settings the narrator reads: the Ollama model, its context size,
and the history window.

The configured model is stored even when Ollama no longer has it — the whole
point is to flag that mismatch instead of failing at the first message. The
same holds when Ollama is unreachable: the real reason is surfaced and the
stored setting is never touched.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import db, ollama, settings

router = APIRouter(tags=["settings"])


class LlmSettings(BaseModel):
    model: str | None
    num_ctx: int
    history_window: int
    # The accepted bounds travel with the response so the client never
    # hand-copies them. Defaults, not constructor arguments: the constants in
    # `settings` stay the single owner.
    min_num_ctx: int = settings.MIN_NUM_CTX
    max_num_ctx: int = settings.MAX_NUM_CTX
    min_history_window: int = settings.MIN_HISTORY_WINDOW
    max_history_window: int = settings.MAX_HISTORY_WINDOW
    # None when Ollama could not be asked: "not installed" is then unknown,
    # not false.
    installed_models: list[str] | None
    model_missing: bool | None
    ollama_error: str | None


class LlmSettingsInput(BaseModel):
    model: str | None
    num_ctx: int = Field(ge=settings.MIN_NUM_CTX, le=settings.MAX_NUM_CTX)
    history_window: int = Field(ge=settings.MIN_HISTORY_WINDOW, le=settings.MAX_HISTORY_WINDOW)


def _read_settings() -> LlmSettings:
    with db.connect() as con:
        model = settings.get_llm_model(con)
        num_ctx = settings.get_num_ctx(con)
        history_window = settings.get_history_window(con)
    try:
        installed = ollama.list_models()
    except ollama.OllamaError as exc:
        # OllamaUnreachable is a subclass: its message carries the real reason.
        return LlmSettings(
            model=model,
            num_ctx=num_ctx,
            history_window=history_window,
            installed_models=None,
            model_missing=None,
            ollama_error=str(exc),
        )
    return LlmSettings(
        model=model,
        num_ctx=num_ctx,
        history_window=history_window,
        installed_models=installed,
        model_missing=model is not None and model not in installed,
        ollama_error=None,
    )


@router.get("/settings/llm", response_model=LlmSettings)
def get_llm_settings() -> LlmSettings:
    return _read_settings()


@router.put("/settings/llm", response_model=LlmSettings)
def update_llm_settings(body: LlmSettingsInput) -> LlmSettings:
    with db.connect() as con:
        settings.set_llm_model(con, body.model)
        settings.set_num_ctx(con, body.num_ctx)
        settings.set_history_window(con, body.history_window)
    return _read_settings()
