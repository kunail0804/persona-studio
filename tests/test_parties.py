"""Tests for the party routes.

Ollama is never called from a test: `ollama.chat` is stubbed where the route
looks it up. `parties.py` does `from .. import ollama` and resolves
`ollama.chat` at call time, so patching the attribute on that module takes
effect. The configured model is written straight to the `setting` table — the
settings PUT route would call the real `ollama.list_models`.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi.testclient import TestClient

from persona_studio import db, settings
from persona_studio.narrator import OPENING_INSTRUCTION
from persona_studio.ollama import OllamaError, OllamaUnreachable
from persona_studio.routes import parties as parties_routes

OPENING = "The harbour gate looms ahead, barnacled and half open. What do you do?"


def _configure_model(model: str | None, num_ctx: int | None = None) -> None:
    with db.connect() as con:
        settings.set_llm_model(con, model)
        if num_ctx is not None:
            settings.set_num_ctx(con, num_ctx)


def _stub_chat(
    monkeypatch: Any, reply: str | None = None, error: Exception | None = None
) -> list[dict[str, Any]]:
    """Replace `ollama.chat` with a recording stub; return the calls it saw."""
    calls: list[dict[str, Any]] = []

    def fake_chat(model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
        calls.append({"model": model, "messages": messages, "num_ctx": num_ctx})
        if error is not None:
            raise error
        assert reply is not None
        return reply

    monkeypatch.setattr(parties_routes.ollama, "chat", fake_chat)
    return calls


def _create_scenario(client: TestClient, title: str = "La Cité Noyée") -> str:
    response = client.post("/api/scenarios", json={"title": title, "synopsis": ""})
    assert response.status_code == 201
    return response.json()["id"]


def _create_party(client: TestClient, scenario_id: str, monkeypatch: Any) -> dict[str, Any]:
    _configure_model("test-model")
    _stub_chat(monkeypatch, reply=OPENING)
    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})
    assert response.status_code == 201
    return response.json()


def _party_counts(scenario_id: str) -> tuple[int, int]:
    """(parties, messages) for one scenario — the "nothing was written" check.

    The suite shares one database, and `tests/test_narrator.py` legitimately
    inserts `instance` rows for its loader tests, so a global COUNT(*) would
    never be zero there. Counting only the scenario under test states the same
    criterion at the scope that can actually hold it.
    """
    with db.connect() as con:
        parties = con.execute(
            "SELECT COUNT(*) FROM instance WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()[0]
        messages = con.execute(
            "SELECT COUNT(*) FROM message "
            "WHERE instance_id IN (SELECT id FROM instance WHERE scenario_id = ?)",
            (scenario_id,),
        ).fetchone()[0]
    return parties, messages


def _message_count_for_party(party_id: str) -> int:
    with db.connect() as con:
        return con.execute(
            "SELECT COUNT(*) FROM message WHERE instance_id = ?", (party_id,)
        ).fetchone()[0]


# --- The headline criterion: a failed opening leaves nothing behind ------------


def test_unreachable_ollama_leaves_nothing_behind(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")
    _stub_chat(
        monkeypatch,
        error=OllamaUnreachable(
            "Ollama is unreachable at http://localhost:11434: connection refused"
        ),
    )

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
    assert _party_counts(scenario_id) == (0, 0)


def test_empty_reply_leaves_nothing_behind(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")
    _stub_chat(monkeypatch, reply="   \n\t ")

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 502
    assert _party_counts(scenario_id) == (0, 0)


def test_ollama_error_is_502_and_leaves_nothing_behind(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")
    _stub_chat(monkeypatch, error=OllamaError("Ollama returned HTTP 404: model not found"))

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 502
    assert "model not found" in response.json()["detail"]
    assert _party_counts(scenario_id) == (0, 0)


def test_no_model_configured_is_400_and_writes_nothing(client: TestClient) -> None:
    scenario_id = _create_scenario(client)
    _configure_model(None)

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 400
    # The detail must name where the fix lives.
    assert "settings" in response.json()["detail"]
    assert _party_counts(scenario_id) == (0, 0)


# --- A successful create -------------------------------------------------------


def test_create_writes_one_party_with_the_opening_as_assistant_message(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    party = _create_party(client, scenario_id, monkeypatch)

    assert party["label"] == "La Cité Noyée"
    assert party["scenario_title"] == "La Cité Noyée"

    detail = client.get(f"/api/parties/{party['id']}")
    assert detail.status_code == 200
    messages = detail.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == OPENING
    assert _party_counts(scenario_id) == (1, 1)


def test_create_uses_the_provided_label_when_there_is_one(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    _configure_model("test-model")
    _stub_chat(monkeypatch, reply=OPENING)

    response = client.post(
        f"/api/scenarios/{scenario_id}/parties", json={"label": "Campagne d'Ash"}
    )

    assert response.status_code == 201
    assert response.json()["label"] == "Campagne d'Ash"


def test_create_sends_the_assembled_prompt_and_the_configured_num_ctx(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    _configure_model("test-model", num_ctx=2048)
    calls = _stub_chat(monkeypatch, reply=OPENING)

    client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["num_ctx"] == 2048
    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system"
    # The system prompt is the assembled scenario prompt, not a stub.
    assert "Scenario: La Cité Noyée" in messages[0]["content"]
    # The opening direction rides as a second system message.
    assert messages[1] == {"role": "system", "content": OPENING_INSTRUCTION}


# --- Listing, renaming, deleting ------------------------------------------------


def test_list_orders_parties_most_recently_active_first(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    first = _create_party(client, scenario_id, monkeypatch)["id"]
    second = _create_party(client, scenario_id, monkeypatch)["id"]
    third = _create_party(client, scenario_id, monkeypatch)["id"]
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (1000.0, first))
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (2000.0, second))
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (3000.0, third))

    listed = [party["id"] for party in client.get("/api/parties").json()]

    # Other tests may have left parties behind; only the relative order of
    # these three is under test here.
    mine = [party_id for party_id in listed if party_id in {first, second, third}]
    assert mine == [third, second, first]


def test_rename_changes_the_label_without_touching_updated_at(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    older = _create_party(client, scenario_id, monkeypatch)["id"]
    newer = _create_party(client, scenario_id, monkeypatch)["id"]
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (1000.0, older))
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (2000.0, newer))
    updated_at_before = client.get(f"/api/parties/{older}").json()["updated_at"]

    response = client.patch(f"/api/parties/{older}", json={"label": "Nouveau nom"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Nouveau nom"
    assert body["updated_at"] == updated_at_before
    # The renamed party stays in its original place: a rename is not activity.
    listed = [party["id"] for party in client.get("/api/parties").json()]
    mine = [party_id for party_id in listed if party_id in {older, newer}]
    assert mine == [newer, older]


def test_delete_removes_the_party_and_its_messages(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]

    response = client.delete(f"/api/parties/{party_id}")

    assert response.status_code == 204
    assert client.get(f"/api/parties/{party_id}").status_code == 404
    assert _message_count_for_party(party_id) == 0


def test_unknown_party_and_scenario_ids_are_404(client: TestClient) -> None:
    missing = uuid.uuid4().hex
    assert client.get(f"/api/parties/{missing}").status_code == 404
    assert client.patch(f"/api/parties/{missing}", json={"label": "x"}).status_code == 404
    assert client.delete(f"/api/parties/{missing}").status_code == 404
    assert client.post(f"/api/scenarios/{missing}/parties", json={"label": ""}).status_code == 404
