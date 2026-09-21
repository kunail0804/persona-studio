"""Tests for the streaming Ollama client.

The HTTP layer is real — only the transport is faked, by swapping `httpx.Client`
for one built on an `httpx.MockTransport`. That exercises the NDJSON parsing,
the status handling and the error mapping (`OllamaUnreachable` for a request
that never completed or died mid-way, `OllamaError` for a response that broke
the protocol) without ever touching a running Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import httpx
import pytest

from persona_studio import ollama
from persona_studio.ollama import OllamaError, OllamaUnreachable

MESSAGES = [{"role": "user", "content": "J'avance dans le brouillard."}]


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    real_client = httpx.Client

    def client_factory(**kwargs: object) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)


def _ndjson(*objects: str) -> httpx.Response:
    return httpx.Response(200, content=("\n".join(objects) + "\n").encode("utf-8"))


def test_stream_yields_fragments_and_stops_at_done(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payloads.append(json.loads(request.content))
        return _ndjson(
            '{"message": {"content": "Le guide "}}',
            '{"message": {"content": "sourit."}}',
            '{"message": {"content": "", "done": true}}',
        )

    _install_transport(monkeypatch, handler)

    fragments = list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))

    assert fragments == ["Le guide ", "sourit."]
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["model"] == "test-model"
    assert payload["messages"] == MESSAGES
    assert payload["stream"] is True
    # num_ctx is assembled here, in the payload, never by the caller.
    assert payload["options"] == {"num_ctx": 4096}


def test_stream_raises_ollama_error_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model 'test-model' not found")

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaError, match="HTTP 404.*not found"):
        list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))


def test_stream_raises_ollama_error_on_non_json_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson("this is not json")

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaError, match="non-JSON stream line"):
        list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))


def test_stream_raises_ollama_error_on_error_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson('{"error": "model requires more context"}')

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaError, match="model requires more context"):
        list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))


def test_stream_raises_ollama_error_on_line_without_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson('{"message": {"role": "assistant"}}')

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaError, match="no message content"):
        list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))


def test_stream_skips_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson("", '{"message": {"content": "Un mot."}}', "")

    _install_transport(monkeypatch, handler)

    assert list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096)) == ["Un mot."]


def test_stream_raises_ollama_error_on_non_object_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson("[1, 2, 3]")

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaError, match="stream line shape"):
        list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))


def test_stream_raises_ollama_unreachable_when_connection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaUnreachable, match="unreachable"):
        list(ollama.chat_stream("test-model", MESSAGES, num_ctx=4096))


def test_stream_raises_ollama_unreachable_mid_stream_after_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def body() -> Iterator[bytes]:
        yield b'{"message": {"content": "Premi"}}\n'
        raise httpx.ReadError("connection reset by peer")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    _install_transport(monkeypatch, handler)

    stream = ollama.chat_stream("test-model", MESSAGES, num_ctx=4096)
    assert next(stream) == "Premi"
    with pytest.raises(OllamaUnreachable, match="unreachable"):
        list(stream)


def test_chat_still_answers_with_one_non_streamed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`chat` keeps its non-streaming payload: the opening scene of pull
    request 4 depends on it."""
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payloads.append(json.loads(request.content))
        return _ndjson('{"message": {"content": "Une scène."}}')

    _install_transport(monkeypatch, handler)

    assert ollama.chat("test-model", MESSAGES, num_ctx=4096) == "Une scène."
    assert payloads[0]["stream"] is False


# --- Unloading, and closing a stream from another thread -------------------------


def test_unload_asks_ollama_to_drop_the_model_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """`keep_alive: 0` with no prompt is Ollama's documented way to free a
    loaded model immediately. It exists because narration and image
    generation share one GPU and do not fit on it together."""
    seen: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"model": "test-model", "done": True})

    _install_transport(monkeypatch, handler)

    ollama.unload("test-model")

    assert seen == [("/api/generate", {"model": "test-model", "keep_alive": 0})]


def test_unload_maps_a_dead_connection_to_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaUnreachable):
        ollama.unload("test-model")


def test_unload_maps_an_http_error_to_ollama_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model 'test-model' not found")

    _install_transport(monkeypatch, handler)

    with pytest.raises(OllamaError) as excinfo:
        ollama.unload("test-model")
    assert "not found" in str(excinfo.value)


def test_close_stream_closes_the_open_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason this exists: `Generator.close()` raises
    `ValueError: generator already executing` when a worker thread is mid-pull,
    which is exactly what pressing stop creates — the request to Ollama then
    stays open and the model keeps generating for a reply nobody will read.
    Closing the response reaches it from any thread."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _ndjson(
            json.dumps({"message": {"content": "Le guide "}, "done": False}),
            json.dumps({"message": {"content": "sourit."}, "done": False}),
        )

    _install_transport(monkeypatch, handler)

    stream = ollama.chat_stream("test-model", MESSAGES, 4096)
    assert next(stream) == "Le guide "
    response = ollama._OPEN_RESPONSES[stream][0]
    # Spying on the response rather than reading `is_closed`: MockTransport
    # hands back a fully buffered body, so the real transport's open-socket
    # state cannot be staged here. What must be asserted is that
    # `close_stream` reaches the response at all — that is the whole
    # difference from `Generator.close()`.
    closed: list[str] = []
    monkeypatch.setattr(response, "close", lambda: closed.append("closed"))

    ollama.close_stream(stream)

    assert closed == ["closed"], "the HTTP response was never closed"
    # The registry lets the stream go with it.
    assert stream not in ollama._OPEN_RESPONSES


def test_close_stream_is_a_no_op_on_a_stream_it_did_not_create() -> None:
    """A test stub standing in for a real stream owns no HTTP request, so
    there is nothing to close and nothing to complain about."""

    def stub() -> Iterator[str]:
        yield "fragment"

    ollama.close_stream(stub())


def test_close_stream_ignores_an_iterator_that_cannot_be_weakly_referenced() -> None:
    """A plain `list_iterator` cannot be a weak-dictionary key at all. It is
    not one of ours either, so it is left alone rather than raising."""
    ollama.close_stream(iter(["fragment"]))  # type: ignore[arg-type]
