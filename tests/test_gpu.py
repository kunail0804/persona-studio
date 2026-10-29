"""Who gets the graphics card: the narrator or a render, and the buttons that
free it.

The HTTP clients are driven through a mocked transport, so the request paths
and bodies are genuine; the routes are driven with the service calls stubbed
where they are looked up. No test here reaches the Ollama or the comfy-gateway
running on this machine (`conftest._no_real_services`), and no test sends a
signal to a real process: `os.kill` and the process lookups are replaced.
"""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from persona_studio import db, gateway, gateway_process, gpu, ollama, settings, summarizer
from persona_studio.routes import parties as parties_routes

OPENING = "The harbour gate looms ahead. What do you do?"


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[str]:
    """Route every `httpx.Client` through `handler`; return what it saw."""
    seen: list[str] = []
    real_client = httpx.Client

    def recording(request: httpx.Request) -> httpx.Response:
        body = request.content.decode() if request.content else ""
        seen.append(f"{request.method} {request.url.path} {body}".strip())
        return handler(request)

    def client_factory(**kwargs: object) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(recording), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    return seen


# --- Ollama: every model, and waited for ---------------------------------------


def test_unload_all_unloads_every_model_and_waits_until_ollama_lets_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not only the narrator's model: the previous one after a change in the
    settings, or another application's, holds the same VRAM. And Ollama
    answers the unload before the model is gone, so `/api/ps` is polled."""
    monkeypatch.setattr(ollama, "UNLOAD_POLL_SECONDS", 0.01)
    ps_answers = [
        [{"model": "gemma:12b", "size_vram": 9}, {"model": "qwen3:4b", "size_vram": 3}],
        [{"model": "qwen3:4b", "size_vram": 3}],  # still letting go
        [],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": ps_answers.pop(0)})
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    seen = _install_transport(monkeypatch, handler)

    assert ollama.unload_all() == []
    unloads = [json.loads(s.split(" ", 2)[2]) for s in seen if s.startswith("POST /api/generate")]
    assert unloads == [
        {"model": "gemma:12b", "keep_alive": 0},
        {"model": "qwen3:4b", "keep_alive": 0},
    ]
    assert ps_answers == []


def test_unload_all_reports_what_ollama_still_holds_after_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ollama, "UNLOAD_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(ollama, "UNLOAD_POLL_SECONDS", 0.01)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"model": "stuck:1b", "size_vram": 1}]})
        return httpx.Response(200, json={"done": True})

    _install_transport(monkeypatch, handler)

    assert ollama.unload_all() == ["stuck:1b"]


# --- comfy-gateway: queue and free ----------------------------------------------


def test_an_idle_gateway_is_not_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(
        monkeypatch, lambda r: httpx.Response(200, json={"running": None, "queued": []})
    )
    assert gateway.busy() is None


def test_a_render_for_another_application_makes_the_gateway_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CharCreator's render is invisible to this application's own rows: only
    the gateway's queue says the card is taken."""
    _install_transport(
        monkeypatch,
        lambda r: httpx.Response(
            200, json={"running": {"id": 3, "client": "charcreator"}, "queued": [{"id": 4}]}
        ),
    )
    reason = gateway.busy()
    assert reason is not None
    assert "rendering 2 image(s)" in reason and "charcreator" in reason


def test_freeing_comfy_asks_for_vram_and_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_transport(monkeypatch, lambda r: httpx.Response(200, json={"mode": "all"}))
    gateway.free_comfy()
    assert seen == ['POST /api/comfy/free {"mode":"all"}']


def test_a_comfy_that_does_not_answer_has_nothing_to_free(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(
        monkeypatch, lambda r: httpx.Response(503, json={"detail": "ComfyUI does not answer"})
    )
    gateway.free_comfy()  # no error


# --- The arbitration ------------------------------------------------------------


def test_the_narrator_is_refused_while_the_gateway_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[bool] = []
    monkeypatch.setattr(gpu.gateway, "busy", lambda: "rendering for charcreator")
    monkeypatch.setattr(gpu.gateway, "free_comfy", lambda: freed.append(True))

    with pytest.raises(gpu.GpuBusy, match="charcreator"):
        gpu.make_room_for_llm()
    assert freed == []


def test_comfy_is_emptied_before_the_narrator_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[bool] = []
    monkeypatch.setattr(gpu.gateway, "busy", lambda: None)
    monkeypatch.setattr(gpu.gateway, "free_comfy", lambda: freed.append(True))

    gpu.make_room_for_llm()
    assert freed == [True]


def test_an_unreachable_gateway_never_stops_the_story() -> None:
    """`conftest` points the gateway at a closed port: nothing to free there."""
    gpu.make_room_for_llm()


# --- The routes that call the narrator -------------------------------------------


@pytest.fixture
def party_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    with db.connect() as con:
        settings.set_llm_model(con, "test-model")
    monkeypatch.setattr(parties_routes.ollama, "chat", lambda *a, **k: OPENING)
    scenario = client.post("/api/scenarios", json={"title": "GPU", "synopsis": ""}).json()["id"]
    party = client.post(f"/api/scenarios/{scenario}/parties", json={"label": ""}).json()["id"]
    yield party
    with db.connect() as con:
        con.execute("DELETE FROM instance WHERE id = ?", (party,))
        con.execute("DELETE FROM scenario WHERE id = ?", (scenario,))


def _message_count(party: str) -> int:
    with db.connect() as con:
        return con.execute(
            "SELECT COUNT(*) FROM message WHERE instance_id = ?", (party,)
        ).fetchone()[0]


def test_a_turn_during_a_gateway_render_is_refused_and_writes_nothing(
    client: TestClient, party_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu.gateway, "busy", lambda: "comfy-gateway is rendering for charcreator")
    before = _message_count(party_id)

    response = client.post(f"/api/parties/{party_id}/messages", json={"content": "I knock."})

    assert response.status_code == 409
    assert "charcreator" in response.json()["detail"]
    assert _message_count(party_id) == before
    # The turn lock was given back: the next turn is not refused for it.
    monkeypatch.setattr(gpu.gateway, "busy", lambda: None)
    monkeypatch.setattr(gpu.gateway, "free_comfy", lambda: None)

    def one_reply(*args: Any, **kwargs: Any) -> Iterator[str]:
        yield "Ok."

    monkeypatch.setattr(parties_routes.ollama, "chat_stream", one_reply)
    assert (
        client.post(f"/api/parties/{party_id}/messages", json={"content": "Again."}).status_code
        == 200
    )


def test_a_turn_empties_comfy_before_the_model_is_called(
    client: TestClient, party_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    monkeypatch.setattr(gpu.gateway, "busy", lambda: None)
    monkeypatch.setattr(gpu.gateway, "free_comfy", lambda: order.append("free"))

    def fake_stream(*args: Any, **kwargs: Any) -> Iterator[str]:
        order.append("model")
        yield "The gate opens."

    monkeypatch.setattr(parties_routes.ollama, "chat_stream", fake_stream)

    response = client.post(f"/api/parties/{party_id}/messages", json={"content": "I knock."})

    assert response.status_code == 200
    assert order == ["free", "model"]


# --- Background summaries wait for the card ----------------------------------------


def test_a_background_summary_waits_for_a_render_to_end(
    client: TestClient, party_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary loading the narrator under a render is the out-of-memory the
    routes refuse. A background job cannot answer 409, so it waits."""
    monkeypatch.setattr(summarizer, "GPU_WAIT_POLL_SECONDS", 0.01)
    with db.connect() as con:
        con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts, status, started_at) "
            "VALUES (?, 'assistant', 'image', 'x', 0, 'pending', 0)",
            (party_id,),
        )
    checks: list[float] = []

    def end_render_after_three_checks() -> None:
        checks.append(time.monotonic())
        if len(checks) == 3:
            with db.connect() as con:
                con.execute(
                    "UPDATE message SET status = 'error' WHERE instance_id = ? AND kind = 'image'",
                    (party_id,),
                )

    real_connect = db.connect

    def counting_connect(*args: Any, **kwargs: Any) -> Any:
        end_render_after_three_checks()
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(summarizer.db, "connect", counting_connect)
    freed: list[bool] = []
    monkeypatch.setattr(gpu.gateway, "busy", lambda: None)
    monkeypatch.setattr(gpu.gateway, "free_comfy", lambda: freed.append(True))

    assert summarizer._wait_for_the_card() is True
    assert len(checks) >= 3
    assert freed == [True]


def test_a_background_summary_gives_up_when_the_wait_runs_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summarizer, "GPU_WAIT_POLL_SECONDS", 0.01)
    monkeypatch.setattr(summarizer, "GPU_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(gpu.gateway, "busy", lambda: "rendering forever")

    assert summarizer._wait_for_the_card() is False


# --- The gateway process ------------------------------------------------------------

_TCP = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when   uid  timeout inode
   0: 0100007F:1FFE 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 111 1 0
   1: 0100007F:1FFE 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000        0 222 1 0
   2: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 333 1 0
"""


def test_only_the_listening_socket_on_the_gateway_port_is_found() -> None:
    """8190 is 0x1FFE. An established connection on the same port is a
    client, not the server."""
    assert gateway_process.listening_inodes(_TCP, 8190) == {"111"}


class _Processes:
    """Fake processes: which pids hold the port, their command lines, which
    are alive. `os.kill` is recorded instead of sent."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, cmdlines: dict[int, list[str]]) -> None:
        self.cmdlines = cmdlines
        self.alive = set(cmdlines)
        self.signals: list[tuple[int, int]] = []
        self.listening: list[int] = []
        self.parents: dict[int, int] = {}
        self.exit_on_term = True
        monkeypatch.setattr(gateway_process, "_listening_pids", lambda port: list(self.listening))
        monkeypatch.setattr(gateway_process, "_cmdline", lambda pid: self.cmdlines.get(pid, []))
        monkeypatch.setattr(gateway_process, "_parent", lambda pid: self.parents.get(pid))
        monkeypatch.setattr(gateway_process, "_alive", lambda pid: pid in self.alive)
        monkeypatch.setattr(gateway_process.os, "kill", self.kill)
        monkeypatch.setattr(gateway_process, "POLL_SECONDS", 0.01)

    def kill(self, pid: int, sig: int) -> None:
        self.signals.append((pid, sig))
        if sig == signal.SIGKILL or (sig == signal.SIGTERM and self.exit_on_term):
            self.alive.discard(pid)
            if pid in self.listening:
                self.listening.remove(pid)


_GATEWAY_CMD = ["/x/.venv/bin/python", "/x/.venv/bin/uvicorn", "comfy_gateway.main:app"]


def test_stop_terminates_the_gateway_then_closes_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = [
        "/usr/bin/alacritty",
        "--class",
        "comfy-gateway",
        "--hold",
        "-e",
        str(gateway_process.LAUNCH_SCRIPT),
    ]
    procs = _Processes(monkeypatch, {30: _GATEWAY_CMD, 20: ["uv", "run"], 10: window})
    procs.listening = [30]
    procs.parents = {30: 20, 20: 10, 10: 1}

    outcome = gateway_process.stop()

    assert procs.signals == [(30, signal.SIGTERM), (10, signal.SIGTERM)]
    assert outcome.running is False


def test_a_stuck_gateway_is_killed_after_its_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _Processes(monkeypatch, {30: _GATEWAY_CMD})
    procs.listening = [30]
    procs.exit_on_term = False
    monkeypatch.setattr(gateway_process, "STOP_WAIT_SECONDS", 0.05)

    outcome = gateway_process.stop()

    assert procs.signals == [(30, signal.SIGTERM), (30, signal.SIGKILL)]
    assert "killed" in outcome.message


def test_a_process_that_is_not_the_gateway_is_never_signalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    procs = _Processes(monkeypatch, {44: ["/usr/bin/python", "-m", "http.server", "8190"]})
    procs.listening = [44]

    with pytest.raises(gateway_process.GatewayProcessError, match="not comfy-gateway"):
        gateway_process.stop()
    assert procs.signals == []


def test_start_does_not_open_a_second_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _Processes(monkeypatch, {30: _GATEWAY_CMD})
    procs.listening = [30]
    spawned: list[Any] = []
    monkeypatch.setattr(gateway_process.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    outcome = gateway_process.start()

    assert outcome.running is True
    assert spawned == []


def test_start_runs_the_shortcut_command_in_its_own_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    procs = _Processes(monkeypatch, {})
    launch = tmp_path / "scripts" / "launch.sh"
    launch.parent.mkdir()
    launch.write_text("#!/bin/bash\n")
    monkeypatch.setattr(gateway_process, "GATEWAY_DIR", tmp_path)
    monkeypatch.setattr(gateway_process, "LAUNCH_SCRIPT", launch)
    monkeypatch.setattr(gateway_process.shutil, "which", lambda name: "/usr/bin/alacritty")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    calls: list[tuple[Any, Any]] = []

    def fake_popen(args: list[str], **kwargs: Any) -> None:
        calls.append((args, kwargs))
        procs.cmdlines[30] = _GATEWAY_CMD
        procs.alive.add(30)
        procs.listening.append(30)

    monkeypatch.setattr(gateway_process.subprocess, "Popen", fake_popen)

    outcome = gateway_process.start()

    assert outcome.running is True
    args, kwargs = calls[0]
    assert args == [
        "/usr/bin/alacritty",
        "--class",
        "comfy-gateway,comfy-gateway",
        "--hold",
        "-e",
        str(launch),
    ]
    assert kwargs["start_new_session"] is True


# --- The service routes ----------------------------------------------------------


def test_the_unload_button_says_what_it_unloaded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ollama, "loaded_models", lambda: [ollama.LoadedModel(name="gemma:12b", size_vram=9)]
    )
    monkeypatch.setattr(ollama, "unload_all", lambda: [])

    response = client.post("/api/services/ollama/unload")

    assert response.status_code == 200
    assert "gemma:12b" in response.json()["message"]


def test_the_unload_button_says_when_ollama_did_not_let_go(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ollama, "loaded_models", lambda: [])
    monkeypatch.setattr(ollama, "unload_all", lambda: ["gemma:12b"])

    response = client.post("/api/services/ollama/unload")

    assert response.status_code == 504
    assert "gemma:12b" in response.json()["detail"]


def test_a_refused_stop_is_a_409_with_the_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise gateway_process.GatewayProcessError(
            "Port 8190 is held by a process that is not comfy-gateway"
        )

    monkeypatch.setattr(gateway_process, "stop", refuse)

    response = client.post("/api/services/gateway/stop")

    assert response.status_code == 409
    assert "not comfy-gateway" in response.json()["detail"]


def test_memory_reports_each_service_on_its_own(client: TestClient) -> None:
    """Both unreachable (conftest): each side says why, neither fails the page."""
    response = client.get("/api/services/memory")

    assert response.status_code == 200
    body = response.json()
    assert body["ollama"]["error"] and body["gateway"]["error"]
    assert body["gateway"]["memory"] is None
