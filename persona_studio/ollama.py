"""The Ollama client: the single boundary every Ollama call goes through.

Routes and future services call into this module, never `httpx` directly. Two
rules are enforced here rather than trusted to callers:

- every request carries an explicit timeout;
- `chat` assembles the request options itself, with `num_ctx` as a required
  parameter — the type checker makes it impossible to send a call without a
  context window, which is what lets Ollama truncate the prompt silently from
  the front, system prompt first.
"""

from __future__ import annotations

from typing import Any

import httpx

OLLAMA_BASE_URL = "http://localhost:11434"

# Connecting to a local Ollama is immediate, but a chat completion on a slow
# local model can legitimately run for minutes: the generous read timeout is
# for that, the 5 s connect timeout is so an unreachable Ollama fails fast.
TIMEOUT = httpx.Timeout(300.0, connect=5.0)


class OllamaError(RuntimeError):
    """Ollama answered with an error, or its response was not the expected shape."""


class OllamaUnreachable(OllamaError):
    """Ollama could not be reached. The message carries the underlying reason."""


def _request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=TIMEOUT) as client:
        try:
            response = client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaUnreachable(f"Ollama is unreachable at {OLLAMA_BASE_URL}: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip()[:500] or response.reason_phrase
        raise OllamaError(f"Ollama returned HTTP {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise OllamaError(f"Ollama returned a non-JSON response: {response.text[:200]}") from exc


def list_models() -> list[str]:
    """Model names actually installed in Ollama, sorted."""
    data = _request_json("GET", "/api/tags")
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise OllamaError("Unexpected /api/tags response shape")
    names = []
    for entry in data["models"]:
        # `model` is the exact string to send back (name plus tag); `name`
        # predates tags and is only a fallback for very old versions.
        name = entry.get("model") if isinstance(entry, dict) else None
        if not name:
            name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(names)


def chat(model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
    """Send a chat completion and return the assistant's message content.

    The options payload is assembled here, once: callers pass `num_ctx`, never
    hand-built options.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    data = _request_json("POST", "/api/chat", payload)
    if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
        raise OllamaError("Unexpected /api/chat response shape")
    content = data["message"].get("content")
    if not isinstance(content, str):
        raise OllamaError("Ollama's reply carries no message content")
    return content
