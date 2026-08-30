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
