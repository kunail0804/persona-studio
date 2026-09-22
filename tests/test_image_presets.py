"""Image-prompt presets: the master prompt of the composer, as data.

What matters here is the fallback chain, because every link of it is a way the
composition can silently stop working: no preset, a preset id pointing at a
deleted row, and a preset whose text is blank all have to land on the built-in
instruction rather than on an empty system message.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, image_prompt, settings
from persona_studio.routes import images as images_routes


@pytest.fixture(autouse=True)
def _clean_presets(client: TestClient) -> None:
    """Presets and the active choice are global state in the shared database.

    Surgical rather than `DELETE FROM setting`: the configured model and the
    history window belong to other files, and wiping them here would make the
    suite depend on file order.
    """
    with db.connect() as con:
        con.execute("DELETE FROM image_preset")
        settings.set_active_image_preset_id(con, None)


def _create(client: TestClient, name: str = "Krea 2", instruction: str = "Write prose.") -> dict:
    response = client.post("/api/image-presets", json={"name": name, "instruction": instruction})
    assert response.status_code == 201
    return response.json()


def _active_instruction() -> str:
    with db.connect() as con:
        return image_prompt.load_active_instruction(con)


def test_no_preset_means_the_built_in_instruction(client: TestClient) -> None:
    with db.connect() as con:
        settings.set_active_image_preset_id(con, None)
    assert _active_instruction() == image_prompt.DEFAULT_INSTRUCTION


def test_the_active_preset_replaces_the_built_in_instruction(client: TestClient) -> None:
    preset = _create(client, instruction="Answer with one English sentence of prose.")
    response = client.put("/api/image-presets/active", json={"id": preset["id"]})
    assert response.status_code == 200
    assert [p["is_active"] for p in response.json() if p["id"] == preset["id"]] == [True]

    assert _active_instruction() == "Answer with one English sentence of prose."


def test_choosing_no_preset_goes_back_to_the_built_in_text(client: TestClient) -> None:
    """`null` is a real choice, not the absence of one: it must be reachable
    without deleting every preset."""
    preset = _create(client)
    client.put("/api/image-presets/active", json={"id": preset["id"]})

    assert client.put("/api/image-presets/active", json={"id": None}).status_code == 200
    assert _active_instruction() == image_prompt.DEFAULT_INSTRUCTION


def test_a_dangling_preset_id_degrades_to_the_built_in_text(client: TestClient) -> None:
    """The setting is written by hand here: deleting through the API clears it,
    so this is the state a hand-edited database or a future bug would leave."""
    with db.connect() as con:
        settings.set_active_image_preset_id(con, "does-not-exist")
    assert _active_instruction() == image_prompt.DEFAULT_INSTRUCTION


def test_a_blank_preset_degrades_to_the_built_in_text(client: TestClient) -> None:
    """An empty system message is worse than no preset: the model would be
    given the scene with no rules at all."""
    preset = _create(client, instruction="   \n  ")
    client.put("/api/image-presets/active", json={"id": preset["id"]})
    assert _active_instruction() == image_prompt.DEFAULT_INSTRUCTION


def test_deleting_the_active_preset_clears_the_setting(client: TestClient) -> None:
    preset = _create(client)
    client.put("/api/image-presets/active", json={"id": preset["id"]})

    assert client.delete(f"/api/image-presets/{preset['id']}").status_code == 204

    with db.connect() as con:
        assert settings.get_active_image_preset_id(con) is None
    assert _active_instruction() == image_prompt.DEFAULT_INSTRUCTION


def test_patch_writes_only_what_was_sent(client: TestClient) -> None:
    preset = _create(client, name="Krea 2", instruction="Write prose.")

    response = client.patch(f"/api/image-presets/{preset['id']}", json={"name": "Krea 2 — large"})

    assert response.status_code == 200
    assert response.json()["name"] == "Krea 2 — large"
    assert response.json()["instruction"] == "Write prose.", "an omitted field was wiped"


def test_an_unknown_key_is_refused_rather_than_dropped(client: TestClient) -> None:
    preset = _create(client)
    assert client.patch(f"/api/image-presets/{preset['id']}", json={"nmae": "x"}).status_code == 422


def test_activating_and_deleting_an_unknown_preset_are_404(client: TestClient) -> None:
    assert client.put("/api/image-presets/active", json={"id": "nope"}).status_code == 404
    assert client.delete("/api/image-presets/nope").status_code == 404
    assert client.patch("/api/image-presets/nope", json={"name": "x"}).status_code == 404


def test_the_default_endpoint_serves_the_built_in_text(client: TestClient) -> None:
    """The editor starts from it rather than from a blank box."""
    response = client.get("/api/image-presets/default")
    assert response.status_code == 200
    assert response.json()["instruction"] == image_prompt.DEFAULT_INSTRUCTION


def test_the_composer_sends_the_active_preset_as_its_system_message(
    client: TestClient, monkeypatch: Any
) -> None:
    """The point of the whole feature, observed where it lands: the system
    message of the composition call is the preset's text."""
    scenario = client.post("/api/scenarios", json={"title": "La Cité Noyée", "synopsis": ""})
    scenario_id = scenario.json()["id"]
    with db.connect() as con:
        settings.set_llm_model(con, "test-model")
    monkeypatch.setattr(images_routes.ollama, "chat", lambda *a, **k: "a drowned city, dusk")
    party = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})
    party_id = party.json()["id"]

    preset = _create(client, instruction="Answer with one English sentence of prose.")
    client.put("/api/image-presets/active", json={"id": preset["id"]})

    calls: list[list[dict[str, str]]] = []

    def fake_chat(model: str, messages: list[dict[str, str]], num_ctx: int, **kwargs: Any) -> str:
        calls.append(messages)
        return "a drowned city, dusk"

    monkeypatch.setattr(images_routes.ollama, "chat", fake_chat)
    response = client.post(
        f"/api/parties/{party_id}/image-prompt", json={"instruction": "la cité vue d'en haut"}
    )

    assert response.status_code == 200
    assert calls[0][0] == {
        "role": "system",
        "content": "Answer with one English sentence of prose.",
    }
    # And the scene still rides in the second message, unchanged by the preset.
    assert "la cité vue d'en haut" in calls[0][1]["content"]


def test_presets_are_listed_oldest_first_with_their_active_flag(client: TestClient) -> None:
    _create(client, name="SD")
    second = _create(client, name="Krea")
    client.put("/api/image-presets/active", json={"id": second["id"]})

    listed = client.get("/api/image-presets").json()

    assert [p["name"] for p in listed] == ["SD", "Krea"]
    assert [p["is_active"] for p in listed] == [False, True]
    # The stored value is JSON, like every other setting.
    with db.connect() as con:
        raw = con.execute(
            "SELECT value FROM setting WHERE key = ?", (settings.IMAGE_PRESET_ACTIVE_KEY,)
        ).fetchone()
    assert json.loads(raw["value"]) == second["id"]
