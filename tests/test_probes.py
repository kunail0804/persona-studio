"""The two service probes behind the header pills.

The route that crosses them is tested with the probes stubbed out, which
proves the route and nothing about the probes. These drive the real
functions through a mocked transport: the HTTP layer is genuine, only the
network is faked, and no test here can reach a running Ollama or comfy-gateway.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from persona_studio import gateway, ollama


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[str]:
    """Route every `httpx.Client` through `handler`; return the paths it saw."""
    seen: list[str] = []
    real_client = httpx.Client

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return handler(request)

    def client_factory(**kwargs: object) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(recording), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    return seen


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 111] Connection refused")


def test_ollama_probe_answers_none_when_ollama_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_transport(monkeypatch, lambda r: httpx.Response(200, json={"models": []}))

    assert ollama.probe() is None
    # The lightest endpoint that proves the server is really there.
    assert seen == ["GET /api/tags"]


def test_ollama_probe_names_a_refused_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, _refused)

    reason = ollama.probe()

    assert reason is not None
    assert "unreachable" in reason
    assert "Connection refused" in reason, "the pill's tooltip must carry the real cause"


def test_ollama_probe_names_an_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    assert ollama.probe() == "Ollama returned HTTP 500"


def test_gateway_probe_answers_none_when_gateway_and_comfyui_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_transport(
        monkeypatch,
        lambda r: httpx.Response(200, json={"ok": True, "comfy_reachable": True}),
    )

    assert gateway.probe() is None
    # Read-only on purpose: this runs on a timer and must never disturb a
    # render. A probe that POSTed would be one bug away from cancelling one.
    assert seen == ["GET /api/health"]


def test_gateway_probe_says_when_comfyui_behind_it_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway answering is not enough to make an image: the pill must
    say that ComfyUI, behind it, is the one missing."""
    _install_transport(
        monkeypatch,
        lambda r: httpx.Response(
            200, json={"ok": True, "comfy_url": "http://127.0.0.1:8188", "comfy_reachable": False}
        ),
    )

    reason = gateway.probe()

    assert reason is not None
    assert "ComfyUI does not answer" in reason and "8188" in reason


def test_gateway_probe_names_a_refused_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, _refused)

    reason = gateway.probe()

    assert reason is not None
    assert "unreachable" in reason
    assert "Connection refused" in reason


def test_gateway_probe_names_an_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, lambda r: httpx.Response(503, text="busy"))
    assert gateway.probe() == "comfy-gateway returned HTTP 503"


@pytest.mark.parametrize("module", [ollama, gateway])
def test_a_probe_gets_a_short_budget_not_a_render_timeout(module: object) -> None:
    """A header refresh must not wait out a 60 or 300 second read timeout.
    Pinned on the value because the failure mode is a slow page, which no
    functional test would ever notice."""
    timeout = module.PROBE_TIMEOUT  # type: ignore[attr-defined]
    assert timeout.read is not None and timeout.read <= 2.0
    assert timeout.connect is not None and timeout.connect <= 2.0
