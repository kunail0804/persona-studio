"""The world of one party: a copy of the scenario's characters, places and lore
that the player edits while playing, without touching the scenario or any
other party."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, image_prompt, settings
from persona_studio.routes import parties as parties_routes


@pytest.fixture
def scenario_id(client: TestClient) -> str:
    response = client.post("/api/scenarios", json={"title": "La Cité Noyée", "synopsis": ""})
    assert response.status_code == 201
    scenario = response.json()["id"]
    client.post(f"/api/scenarios/{scenario}/characters", json={"name": "Elena"})
    client.post(f"/api/scenarios/{scenario}/places", json={"name": "Le Port"})
    client.post(
        f"/api/scenarios/{scenario}/lore", json={"keywords": ["Ordre"], "text": "Un ordre ancien."}
    )
    return scenario


def _party(client: TestClient, scenario_id: str, monkeypatch: Any) -> str:
    with db.connect() as con:
        settings.set_llm_model(con, "test-model")
    monkeypatch.setattr(parties_routes.ollama, "chat", lambda *a, **k: "Le port s'ouvre.")
    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})
    assert response.status_code == 201
    return response.json()["id"]


def _world(client: TestClient, party_id: str) -> dict[str, Any]:
    response = client.get(f"/api/parties/{party_id}/world")
    assert response.status_code == 200
    return response.json()


def _system_prompt(client: TestClient, party_id: str) -> str:
    return client.get(f"/api/parties/{party_id}/prompt").json()["blocks"][0]["content"]


def test_a_new_party_starts_with_a_copy_of_the_scenario_world(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    world = _world(client, _party(client, scenario_id, monkeypatch))

    assert [c["name"] for c in world["characters"]] == ["Elena"]
    assert [p["name"] for p in world["places"]] == ["Le Port"]
    assert [entry["keywords"] for entry in world["lore"]] == [["Ordre"]]


def test_a_character_created_in_a_party_reaches_its_prompt_and_nowhere_else(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    first = _party(client, scenario_id, monkeypatch)
    second = _party(client, scenario_id, monkeypatch)

    created = client.post(
        f"/api/parties/{first}/characters", json={"name": "Ivo", "personality": "Rusé."}
    )

    assert created.status_code == 201
    assert created.json()["position"] == 1
    assert "Name: Ivo" in _system_prompt(client, first)
    assert "Name: Ivo" not in _system_prompt(client, second)
    scenario_names = [
        c["name"] for c in client.get(f"/api/scenarios/{scenario_id}/characters").json()
    ]
    assert scenario_names == ["Elena"]


def test_editing_the_scenario_after_the_party_started_does_not_reach_it(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)

    client.post(f"/api/scenarios/{scenario_id}/places", json={"name": "La Cathédrale"})

    assert [p["name"] for p in _world(client, party)["places"]] == ["Le Port"]


def test_a_patch_writes_only_the_fields_it_carries(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)
    elena = _world(client, party)["characters"][0]["id"]
    client.patch(f"/api/parties/{party}/characters/{elena}", json={"secrets": "L'héritière."})

    renamed = client.patch(f"/api/parties/{party}/characters/{elena}", json={"name": "Elena II"})

    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Elena II"
    assert renamed.json()["secrets"] == "L'héritière."


def test_a_deleted_place_leaves_the_prompt(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)
    port = _world(client, party)["places"][0]["id"]

    assert client.delete(f"/api/parties/{party}/places/{port}").status_code == 204

    assert _world(client, party)["places"] == []
    assert "Le Port" not in _system_prompt(client, party)


def test_lore_keywords_are_normalised_like_the_scenario_lorebook(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)

    created = client.post(
        f"/api/parties/{party}/lore", json={"keywords": [" Clé ", "clé", ""], "text": "Rouillée."}
    )
    entry = created.json()
    updated = client.patch(f"/api/parties/{party}/lore/{entry['id']}", json={"text": "Neuve."})

    assert entry["keywords"] == ["Clé"]
    assert updated.json() == {**entry, "text": "Neuve."}


def test_a_row_of_another_party_is_404(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    first = _party(client, scenario_id, monkeypatch)
    second = _party(client, scenario_id, monkeypatch)
    elena_of_first = _world(client, first)["characters"][0]["id"]

    assert (
        client.patch(f"/api/parties/{second}/characters/{elena_of_first}", json={"name": "X"})
    ).status_code == 404
    assert client.delete(f"/api/parties/{second}/characters/{elena_of_first}").status_code == 404
    assert client.get("/api/parties/nope/world").status_code == 404
    assert client.post("/api/parties/nope/places", json={"name": "X"}).status_code == 404


def test_the_image_prompt_reads_the_party_characters_and_places(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)
    client.post(f"/api/parties/{party}/characters", json={"name": "Ivo", "appearance": "tall"})
    client.post(f"/api/parties/{party}/places", json={"name": "La Forge", "description": "a forge"})

    with db.connect() as con:
        inputs = image_prompt.load_inputs(con, party)

    assert [c.name for c in inputs.characters] == ["Elena", "Ivo"]
    assert [p.name for p in inputs.places] == ["Le Port", "La Forge"]


def test_deleting_a_party_deletes_its_world(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)

    assert client.delete(f"/api/parties/{party}").status_code == 204

    with db.connect() as con:
        for table in ("party_character", "party_place", "party_lore"):
            count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE instance_id = ?", (party,)
            ).fetchone()[0]
            assert count == 0


# --- The v5 migration, on a database that predates it ---------------------------


@pytest.fixture
def v4_database(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[sqlite3.Connection]:
    """A database migrated to v4, with one party of a scenario that has a world."""
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "studio.db")
    monkeypatch.setattr(db, "IMAGES_DIR", tmp_path / "images")
    full = list(db.MIGRATIONS)
    monkeypatch.setattr(db, "MIGRATIONS", full[:4])
    assert db.migrate() == 4
    monkeypatch.setattr(db, "MIGRATIONS", full)
    con = sqlite3.connect(tmp_path / "studio.db")
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("INSERT INTO scenario (id, title, created_at, updated_at) VALUES ('s', 't', 0, 0)")
    con.execute(
        "INSERT INTO character (scenario_id, position, name, secrets) "
        "VALUES ('s', 0, 'Elena', 'heir')"
    )
    con.execute("INSERT INTO place (scenario_id, position, name) VALUES ('s', 0, 'Le Port')")
    con.execute(
        "INSERT INTO lore (scenario_id, position, keywords, text) "
        "VALUES ('s', 0, '[\"Ordre\"]', 'Un ordre.')"
    )
    con.execute(
        "INSERT INTO instance (id, scenario_id, created_at, updated_at) VALUES ('p', 's', 0, 0)"
    )
    con.commit()
    try:
        yield con
    finally:
        con.close()


def test_v5_gives_existing_parties_the_world_they_were_already_reading(
    v4_database: sqlite3.Connection,
) -> None:
    assert db.migrate() == 5

    character = v4_database.execute(
        "SELECT name, secrets FROM party_character WHERE instance_id = 'p'"
    ).fetchone()
    place = v4_database.execute("SELECT name FROM party_place WHERE instance_id = 'p'").fetchone()
    lore = v4_database.execute(
        "SELECT keywords, text FROM party_lore WHERE instance_id = 'p'"
    ).fetchone()
    assert (character["name"], character["secrets"]) == ("Elena", "heir")
    assert place["name"] == "Le Port"
    assert (lore["keywords"], lore["text"]) == ('["Ordre"]', "Un ordre.")
