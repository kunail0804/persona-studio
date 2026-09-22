from __future__ import annotations

import json
import threading
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from persona_studio import db, ollama, settings
from persona_studio.routes import settings as settings_routes


@pytest.fixture(autouse=True)
def _clean_settings(client: TestClient) -> None:
    """The setting table is global state in the shared test database.

    Depends on `client` so the app's startup migration has run first — the
    tables must exist even when this file runs alone.
    """
    with db.connect() as con:
        con.execute("DELETE FROM setting")


@pytest.fixture
def installed_models(monkeypatch: pytest.MonkeyPatch):
    """Stub the Ollama client; no test ever talks to a real Ollama."""

    def stub(models: list[str]) -> None:
        monkeypatch.setattr(ollama, "list_models", lambda: models)

    return stub


def test_defaults_when_nothing_is_stored(client: TestClient, installed_models) -> None:
    installed_models(["qwen38-27b:Q6_K_XL"])

    body = client.get("/api/settings/llm").json()
    assert body["model"] is None
    assert body["num_ctx"] == settings.DEFAULT_NUM_CTX
    assert body["min_num_ctx"] == settings.MIN_NUM_CTX
    assert body["max_num_ctx"] == settings.MAX_NUM_CTX
    assert body["history_window"] == settings.DEFAULT_HISTORY_WINDOW
    assert body["min_history_window"] == settings.MIN_HISTORY_WINDOW
    assert body["max_history_window"] == settings.MAX_HISTORY_WINDOW
    assert body["model_missing"] is False
    assert body["installed_models"] == ["qwen38-27b:Q6_K_XL"]
    assert body["ollama_error"] is None


def test_malformed_stored_settings_fall_back_to_defaults(
    client: TestClient, installed_models
) -> None:
    installed_models(["a:latest"])
    with db.connect() as con:
        con.execute(
            "INSERT INTO setting (key, value) VALUES (?, ?)", (settings.NUM_CTX_KEY, "not-json{")
        )
        con.execute(
            "INSERT INTO setting (key, value) VALUES (?, ?)", (settings.LLM_MODEL_KEY, "[1, 2]")
        )
        con.execute(
            "INSERT INTO setting (key, value) VALUES (?, ?)",
            (settings.HISTORY_WINDOW_KEY, "not-json{"),
        )

    body = client.get("/api/settings/llm").json()
    assert body["num_ctx"] == settings.DEFAULT_NUM_CTX
    assert body["history_window"] == settings.DEFAULT_HISTORY_WINDOW
    assert body["model"] is None


def test_bounded_int_validator_rejects_bool_even_inside_the_range() -> None:
    """`bool` is a subclass of `int`: `True` must not pass for 1.

    Pinned on the shared validator with bounds where 1 sits inside the range.
    The settings-level minima (512, 2) reject `True == 1` on range alone, so
    only this test fails when the bool branch is deleted.
    """
    assert settings._valid_bounded_int(True, 1, 5) is False
    assert settings._valid_bounded_int(1, 1, 5) is True


def test_stored_json_true_falls_back_to_the_default(client: TestClient, installed_models) -> None:
    """A stored `true` falls back to the default, end to end.

    Wiring, not the guard itself: this proves the value read from the table
    goes through the validator. `True == 1` is also below every minimum here,
    so this test holds on range rejection alone — the bool branch is pinned
    by `test_bounded_int_validator_rejects_bool_even_inside_the_range`.
    """
    installed_models(["a:latest"])
    with db.connect() as con:
        con.execute(
            "INSERT INTO setting (key, value) VALUES (?, ?)", (settings.NUM_CTX_KEY, "true")
        )

    body = client.get("/api/settings/llm").json()
    assert body["num_ctx"] == settings.DEFAULT_NUM_CTX

    with db.connect() as con:
        con.execute("DELETE FROM setting WHERE key = ?", (settings.NUM_CTX_KEY,))
        con.execute(
            "INSERT INTO setting (key, value) VALUES (?, ?)",
            (settings.HISTORY_WINDOW_KEY, "true"),
        )

    body = client.get("/api/settings/llm").json()
    assert body["history_window"] == settings.DEFAULT_HISTORY_WINDOW


def test_storing_and_reading_settings(client: TestClient, installed_models) -> None:
    installed_models(["qwen38-27b:Q6_K_XL"])

    response = client.put(
        "/api/settings/llm",
        json={"model": "qwen38-27b:Q6_K_XL", "num_ctx": 16384, "history_window": 50},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "qwen38-27b:Q6_K_XL"
    assert body["num_ctx"] == 16384
    assert body["history_window"] == 50
    assert body["model_missing"] is False

    read_back = client.get("/api/settings/llm").json()
    assert read_back["num_ctx"] == 16384
    assert read_back["history_window"] == 50


def test_clearing_the_model_removes_the_stored_value(client: TestClient, installed_models) -> None:
    installed_models(["a:latest"])
    with db.connect() as con:
        settings.set_llm_model(con, "a:latest")

    response = client.put(
        "/api/settings/llm", json={"model": None, "num_ctx": 8192, "history_window": 20}
    )
    assert response.status_code == 200
    assert response.json()["model"] is None
    with db.connect() as con:
        row = con.execute("SELECT value FROM setting WHERE key = 'llm.model'").fetchone()
    assert row is None


def test_rejects_a_num_ctx_outside_the_accepted_bounds(
    client: TestClient, installed_models
) -> None:
    installed_models(["a:latest"])
    for num_ctx in [0, 100, settings.MAX_NUM_CTX + 1]:
        response = client.put(
            "/api/settings/llm", json={"model": None, "num_ctx": num_ctx, "history_window": 20}
        )
        assert response.status_code == 422


def test_rejects_a_history_window_outside_the_accepted_bounds(
    client: TestClient, installed_models
) -> None:
    installed_models(["a:latest"])
    for window in [0, 1, settings.MAX_HISTORY_WINDOW + 1]:
        response = client.put(
            "/api/settings/llm", json={"model": None, "num_ctx": 8192, "history_window": window}
        )
        assert response.status_code == 422


def test_configured_model_no_longer_installed_is_still_listed_and_flagged(
    client: TestClient, installed_models
) -> None:
    installed_models(["still-here:latest"])
    with db.connect() as con:
        settings.set_llm_model(con, "removed-elsewhere:latest")

    body = client.get("/api/settings/llm").json()
    assert body["model"] == "removed-elsewhere:latest"
    assert body["model_missing"] is True
    assert body["installed_models"] == ["still-here:latest"]


def test_unreachable_ollama_reports_the_real_error_and_keeps_the_setting(
    client: TestClient, installed_models, monkeypatch: pytest.MonkeyPatch
) -> None:
    reason = "[Errno 111] Connect call failed ('127.0.0.1', 11434)"

    def raise_unreachable() -> list[str]:
        raise ollama.OllamaUnreachable(
            f"Ollama is unreachable at {ollama.OLLAMA_BASE_URL}: {reason}"
        )

    monkeypatch.setattr(ollama, "list_models", raise_unreachable)
    with db.connect() as con:
        settings.set_llm_model(con, "kept:latest")
        settings.set_num_ctx(con, 4096)

    body = client.get("/api/settings/llm").json()
    assert reason in body["ollama_error"]
    assert body["installed_models"] is None
    assert body["model_missing"] is None
    assert body["model"] == "kept:latest"

    # A save must not fail, and must not empty the stored model, when Ollama
    # is down.
    response = client.put(
        "/api/settings/llm", json={"model": "kept:latest", "num_ctx": 8192, "history_window": 20}
    )
    assert response.status_code == 200
    assert response.json()["model"] == "kept:latest"
    with db.connect() as con:
        assert settings.get_llm_model(con) == "kept:latest"


# --- The Ollama client itself -------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("No JSON")
        return self._json_data


class FakeClient:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def request(self, method: str, path: str, json: Any = None) -> FakeResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_chat_sends_num_ctx_in_the_request_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        captured.update({"method": method, "path": path, "payload": payload})
        return {"message": {"role": "assistant", "content": "Bonjour."}}

    monkeypatch.setattr(ollama, "_request_json", fake_request)

    messages = [{"role": "system", "content": "Tu narres."}, {"role": "user", "content": "Hi"}]
    reply = ollama.chat("qwen38-27b:Q6_K_XL", messages, 4096)

    assert reply == "Bonjour."
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["model"] == "qwen38-27b:Q6_K_XL"
    assert captured["payload"]["messages"] == messages
    assert captured["payload"]["options"]["num_ctx"] == 4096


def test_chat_rejects_a_response_without_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "_request_json", lambda *args, **kwargs: {"message": {}})
    with pytest.raises(ollama.OllamaError):
        ollama.chat("m:latest", [{"role": "user", "content": "hi"}], 8192)


def test_chat_rejects_a_response_without_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "_request_json", lambda *args, **kwargs: {"done": True})
    with pytest.raises(ollama.OllamaError, match="/api/chat"):
        ollama.chat("m:latest", [{"role": "user", "content": "hi"}], 8192)


def test_list_models_returns_the_installed_names_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama,
        "_request_json",
        lambda *args, **kwargs: {
            "models": [
                {"name": "b:latest", "model": "b:latest"},
                {"name": "a:Q6", "model": "a:Q6"},
            ]
        },
    )
    assert ollama.list_models() == ["a:Q6", "b:latest"]


def test_list_models_falls_back_to_name_when_model_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama,
        "_request_json",
        lambda *args, **kwargs: {"models": [{"name": "legacy:latest"}]},
    )
    assert ollama.list_models() == ["legacy:latest"]


def test_list_models_surfaces_unexpected_response_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "_request_json", lambda *args, **kwargs: {"unexpected": True})
    with pytest.raises(ollama.OllamaError):
        ollama.list_models()


def test_http_error_from_ollama_raises_with_the_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama.httpx,
        "Client",
        lambda **kwargs: FakeClient(FakeResponse(status_code=404, text="not found")),
    )
    with pytest.raises(ollama.OllamaError, match="HTTP 404"):
        ollama.list_models()


def test_non_json_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ollama.httpx, "Client", lambda **kwargs: FakeClient(FakeResponse(text="<html>"))
    )
    with pytest.raises(ollama.OllamaError, match="non-JSON"):
        ollama.list_models()


def test_unreachable_ollama_carries_the_underlying_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_error = httpx.ConnectError("[Errno 111] Connect call failed ('127.0.0.1', 11434)")
    monkeypatch.setattr(ollama.httpx, "Client", lambda **kwargs: FakeClient(connect_error))

    with pytest.raises(ollama.OllamaUnreachable) as exc_info:
        ollama.list_models()
    assert "[Errno 111]" in str(exc_info.value)


def test_settings_round_trip_through_json_storage(client: TestClient) -> None:
    """The setting table stores JSON: reading back what was written must not depend on luck."""
    with db.connect() as con:
        settings.set_num_ctx(con, 2048)
        settings.set_active_persona_id(con, "abc123")
        settings.set_active_workflow_id(con, "wf456")
        raw = {row["key"]: row["value"] for row in con.execute("SELECT key, value FROM setting")}
    assert json.loads(raw[settings.NUM_CTX_KEY]) == 2048
    assert json.loads(raw[settings.ACTIVE_PERSONA_KEY]) == "abc123"
    assert json.loads(raw[settings.WORKFLOW_ACTIVE_KEY]) == "wf456"


# --- Service status (V1.2) -------------------------------------------------------
#
# Two pills in the header. Until they existed you learned Ollama was down when
# a turn failed, halfway through writing one.


def test_status_reports_both_services_reachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_routes.ollama, "probe", lambda: None)
    monkeypatch.setattr(settings_routes.comfyui, "probe", lambda: None)

    body = client.get("/api/status").json()

    assert body == {
        "ollama": {"reachable": True, "detail": None},
        "comfyui": {"reachable": True, "detail": None},
    }


def test_status_carries_the_real_reason_for_each_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detail is what the pill's tooltip shows: "unreachable" alone sends
    the reader to a terminal, which is the state this replaces."""
    monkeypatch.setattr(settings_routes.ollama, "probe", lambda: "connection refused")
    monkeypatch.setattr(settings_routes.comfyui, "probe", lambda: None)

    body = client.get("/api/status").json()

    assert body["ollama"] == {"reachable": False, "detail": "connection refused"}
    assert body["comfyui"]["reachable"] is True


def test_one_service_being_down_does_not_hide_the_other(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two probes run concurrently and are independent: a failing one must
    not take the other's answer with it."""
    monkeypatch.setattr(settings_routes.ollama, "probe", lambda: "ollama est parti")
    monkeypatch.setattr(settings_routes.comfyui, "probe", lambda: "comfyui aussi")

    body = client.get("/api/status").json()

    assert body["ollama"]["detail"] == "ollama est parti"
    assert body["comfyui"]["detail"] == "comfyui aussi"


def test_the_probes_run_at_the_same_time_not_one_after_the_other(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each probe has a two-second budget. Run in sequence the worst case is
    four seconds of a header refresh; the route exists to make it two.

    Measured rather than asserted about: both probes block on the same
    barrier, and neither can pass it until the other has arrived.
    """
    barrier = threading.Barrier(2, timeout=5.0)

    def blocking() -> str | None:
        barrier.wait()
        return None

    monkeypatch.setattr(settings_routes.ollama, "probe", blocking)
    monkeypatch.setattr(settings_routes.comfyui, "probe", blocking)

    # Deadlocks if they are sequential: the first would wait for a second
    # arrival that only comes after it returns.
    assert client.get("/api/status").status_code == 200
