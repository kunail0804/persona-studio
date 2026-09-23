"""The Ollama client: the single boundary every Ollama call goes through.

Routes and future services call into this module, never `httpx` directly. Two
rules are enforced here rather than trusted to callers:

- every request carries an explicit timeout;
- `chat` and `chat_stream` assemble the request options themselves, with
  `num_ctx` as a required parameter — the type checker makes it impossible to
  send a call without a context window, which is what lets Ollama truncate the
  prompt silently from the front, system prompt first.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Generator, Iterator
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


# Unloading frees memory, it does not generate: there is no model work to wait
# for, so it gets a far shorter read timeout than a completion. The one thing
# that can still make it slow is a generation already running on the same
# model, which Ollama serialises this request behind.
UNLOAD_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


# A status pill must answer while the page renders, not after a model call's
# read timeout. The probe gets its own short budget for that reason.
PROBE_TIMEOUT = httpx.Timeout(2.0, connect=2.0)


def probe() -> str | None:
    """None when Ollama answers, else why it did not.

    Cheap on purpose — the model list is the lightest endpoint that proves
    the server is really there — and short-timed, because this runs on a
    header refresh and must never make the interface wait.
    """
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=PROBE_TIMEOUT) as client:
            response = client.get("/api/tags")
    except httpx.HTTPError as exc:
        return f"Ollama is unreachable at {OLLAMA_BASE_URL}: {exc}"
    if response.status_code >= 400:
        return f"Ollama returned HTTP {response.status_code}"
    return None


def unload(model: str) -> None:
    """Ask Ollama to drop `model` from memory now, instead of at its idle timer.

    `keep_alive: 0` with no prompt is Ollama's documented way to do this. It
    exists here for one reason: narration and image generation share a single
    GPU, and on this machine's 12 GB a resident chat model plus a diffusion
    pipeline is an out-of-memory error rather than a slowdown. The cost is
    paid by the next narration turn, which reloads the model from disk.

    Raises the same two errors every other call in this module raises. The
    image path is expected to swallow them: an Ollama that cannot be reached
    is holding no VRAM to begin with, and an unload that failed costs at worst
    one render — never a row, and never the player's message.
    """
    with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=UNLOAD_TIMEOUT) as client:
        try:
            response = client.post("/api/generate", json={"model": model, "keep_alive": 0})
        except httpx.HTTPError as exc:
            raise OllamaUnreachable(f"Ollama is unreachable at {OLLAMA_BASE_URL}: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip()[:500] or response.reason_phrase
        raise OllamaError(f"Ollama returned HTTP {response.status_code}: {detail}")


def chat(
    model: str, messages: list[dict[str, str]], num_ctx: int, *, format: str | None = None
) -> str:
    """Send a chat completion and return the assistant's message content.

    The options payload is assembled here, once: callers pass `num_ctx`, never
    hand-built options. `format` is Ollama's response-format hint — `"json"`
    asks the model for a JSON answer — and defaults to absent, so every
    existing caller keeps sending exactly what it sent before.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }
    if format is not None:
        payload["format"] = format
    data = _request_json("POST", "/api/chat", payload)
    if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
        raise OllamaError("Unexpected /api/chat response shape")
    content = data["message"].get("content")
    if not isinstance(content, str):
        raise OllamaError("Ollama's reply carries no message content")
    return content


def chat_stream(model: str, messages: list[dict[str, str]], num_ctx: int) -> Generator[str]:
    """Send a chat completion and yield content fragments as they arrive.

    The streamed twin of `chat`, with the same contract: the options payload
    is assembled here so no caller can send a call without `num_ctx`, and the
    two errors mean the same things they mean for `chat` — `OllamaUnreachable`
    for a request that never completed, `OllamaError` for a response that
    broke the protocol. Both can surface mid-stream, after fragments were
    already yielded; the caller is expected to keep whatever arrived.

    The wire format is Ollama's NDJSON: one JSON object per line, each with
    `message.content`, the last carrying `done: true`.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }
    holder: list[httpx.Response] = []
    stream = _stream_chat(payload, holder)
    _OPEN_RESPONSES[stream] = holder
    return stream


# The live response behind each stream this module handed out, so `close_stream`
# can reach it. Weak keys: a caller that drops its generator without closing it
# takes the entry with it, and nothing here keeps a stream alive.
_OPEN_RESPONSES: weakref.WeakKeyDictionary[Generator[str], list[httpx.Response]] = (
    weakref.WeakKeyDictionary()
)


def close_stream(stream: Generator[str]) -> None:
    """Stop a chat stream at the HTTP level, from any thread.

    `Generator.close()` is the ordinary way to end a stream, and it is not
    enough on the path that matters. When the player presses stop, the
    cancellation lands while a worker thread is still inside a pull; closing
    the generator then raises `ValueError: generator already executing` and
    does nothing, so the request to Ollama stays open and the model keeps
    generating a reply nobody will ever read — holding the GPU for the rest of
    its evaluation. Closing the response object works there: the in-flight
    read fails, and the generator unwinds through its own `with` blocks.

    Idempotent, and safe on a stream that already finished. A stream this
    module did not create — a test stub, say — is not in the registry and is
    left to its own `close()`.
    """
    try:
        holder = _OPEN_RESPONSES.pop(stream, None)
    except TypeError:
        # Not weak-referenceable, so it cannot be one of ours: a plain
        # iterator standing in for a stream, which owns no HTTP request.
        return
    if holder:
        holder[0].close()


def _stream_chat(payload: dict[str, Any], holder: list[httpx.Response]) -> Generator[str]:
    """The streaming call itself; `holder` receives the response once it exists.

    The response is published rather than returned because the caller holds a
    generator, and a generator has nothing to hand back before its first pull.
    `close_stream` is what reads it.
    """
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=TIMEOUT) as client:
            with client.stream("POST", "/api/chat", json=payload) as response:
                holder.append(response)
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace").strip()[:500]
                    raise OllamaError(
                        f"Ollama returned HTTP {response.status_code}: "
                        f"{body or response.reason_phrase}"
                    )
                yield from _iter_stream_content(response)
    except httpx.HTTPError as exc:
        # Catches a stream that dies mid-way too: after ten fragments or none,
        # a connection that dropped is the same cause as an unreachable
        # Ollama, and the fragments already yielded are the caller's to keep.
        raise OllamaUnreachable(f"Ollama is unreachable at {OLLAMA_BASE_URL}: {exc}") from exc


def _iter_stream_content(response: httpx.Response) -> Iterator[str]:
    for line in response.iter_lines():
        line = line.strip()
        if not line:
            continue
        try:
            data: Any = json.loads(line)
        except ValueError as exc:
            raise OllamaError(f"Ollama sent a non-JSON stream line: {line[:200]}") from exc
        if not isinstance(data, dict):
            raise OllamaError("Unexpected /api/chat stream line shape")
        error = data.get("error")
        if isinstance(error, str) and error:
            raise OllamaError(f"Ollama stream failed: {error}")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("Ollama's stream line carries no message content")
        if content:
            yield content
        if data.get("done") is True:
            return
