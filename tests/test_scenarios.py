from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi.testclient import TestClient

from persona_studio import db


def _create_scenario(client: TestClient, title: str = "Test", synopsis: str = "") -> dict[str, Any]:
    response = client.post("/api/scenarios", json={"title": title, "synopsis": synopsis})
    assert response.status_code == 201
    return response.json()


def test_create_and_edit_scenario(client: TestClient) -> None:
    created = _create_scenario(client, "Le Vaisseau", "Une histoire de vaisseau perdu.")
    assert created["title"] == "Le Vaisseau"
    assert created["synopsis"] == "Une histoire de vaisseau perdu."
    assert created["created_at"] == created["updated_at"]

    response = client.patch(
        f"/api/scenarios/{created['id']}",
        json={"title": "Le Grand Vaisseau", "synopsis": "Révisé."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["title"] == "Le Grand Vaisseau"
    assert body["synopsis"] == "Révisé."


def test_list_shows_title_synopsis_and_character_count(client: TestClient) -> None:
    scenario = _create_scenario(client, "Avec personnages", "Un synopsis.")
    client.post(f"/api/scenarios/{scenario['id']}/characters", json={"name": "Alice"})
    client.post(f"/api/scenarios/{scenario['id']}/characters", json={"name": "Bob"})

    response = client.get("/api/scenarios")
    assert response.status_code == 200
    listed = next(row for row in response.json() if row["id"] == scenario["id"])
    assert listed["title"] == "Avec personnages"
    assert listed["synopsis"] == "Un synopsis."
    assert listed["character_count"] == 2


def test_list_counts_zero_characters_without_a_party_row(client: TestClient) -> None:
    scenario = _create_scenario(client, "Sans personnages")

    response = client.get("/api/scenarios")
    listed = next(row for row in response.json() if row["id"] == scenario["id"])
    assert listed["character_count"] == 0


def test_get_missing_scenario_is_404(client: TestClient) -> None:
    response = client.get("/api/scenarios/does-not-exist")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_patch_missing_scenario_is_404(client: TestClient) -> None:
    response = client.patch("/api/scenarios/does-not-exist", json={"title": "x", "synopsis": ""})
    assert response.status_code == 404


def test_delete_missing_scenario_is_404(client: TestClient) -> None:
    response = client.delete("/api/scenarios/does-not-exist")
    assert response.status_code == 404


def test_delete_scenario_removes_its_characters(client: TestClient) -> None:
    scenario = _create_scenario(client)
    client.post(f"/api/scenarios/{scenario['id']}/characters", json={"name": "Alice"})

    response = client.delete(f"/api/scenarios/{scenario['id']}")
    assert response.status_code == 204
    assert client.get(f"/api/scenarios/{scenario['id']}").status_code == 404


def test_delete_scenario_frees_orphan_images_and_their_files(client: TestClient) -> None:
    scenario = _create_scenario(client, "À supprimer")
    other_scenario = _create_scenario(client, "À garder")

    orphan_image_id = uuid.uuid4().hex
    shared_image_id = uuid.uuid4().hex
    now = time.time()

    with db.connect() as con:
        con.execute(
            "INSERT INTO image (id, prompt, instruction, created_at) VALUES (?, '', '', ?)",
            (orphan_image_id, now),
        )
        con.execute(
            "INSERT INTO image (id, prompt, instruction, created_at) VALUES (?, '', '', ?)",
            (shared_image_id, now),
        )
        # `scenario` is the only reference to the first image; the second
        # belongs to `other_scenario`, which survives the deletion below.
        con.execute(
            "UPDATE scenario SET portrait_id = ? WHERE id = ?",
            (orphan_image_id, scenario["id"]),
        )
        con.execute(
            "UPDATE scenario SET portrait_id = ? WHERE id = ?",
            (shared_image_id, other_scenario["id"]),
        )

    db.image_path(orphan_image_id).write_bytes(b"fake-png")
    db.image_path(shared_image_id).write_bytes(b"fake-png")

    response = client.delete(f"/api/scenarios/{scenario['id']}")
    assert response.status_code == 204

    with db.connect() as con:
        remaining = {row["id"] for row in con.execute("SELECT id FROM image").fetchall()}
    assert orphan_image_id not in remaining
    assert shared_image_id in remaining
    assert not db.image_path(orphan_image_id).exists()
    assert db.image_path(shared_image_id).exists()
