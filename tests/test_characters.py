from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

CHARACTER_FIELDS = {
    "name": "Alice",
    "appearance": "Cheveux roux, veste de cuir usée.",
    "personality": "Curieuse et prudente.",
    "story": "Ancienne pilote reconvertie en contrebandière.",
    "relationships": "Méfiante envers Bob.",
    "secrets": "Travaille en secret pour la faction rivale.",
}


def _create_scenario(client: TestClient, title: str = "Scénario") -> dict[str, Any]:
    response = client.post("/api/scenarios", json={"title": title, "synopsis": ""})
    assert response.status_code == 201
    return response.json()


def _create_character(client: TestClient, scenario_id: str, name: str) -> dict[str, Any]:
    response = client.post(
        f"/api/scenarios/{scenario_id}/characters",
        json={**CHARACTER_FIELDS, "name": name},
    )
    assert response.status_code == 201
    return response.json()


def test_add_and_edit_character(client: TestClient) -> None:
    scenario = _create_scenario(client)

    character = _create_character(client, scenario["id"], "Alice")
    assert character["name"] == "Alice"
    assert character["appearance"] == CHARACTER_FIELDS["appearance"]
    assert character["secrets"] == CHARACTER_FIELDS["secrets"]
    assert character["position"] == 0

    response = client.patch(
        f"/api/scenarios/{scenario['id']}/characters/{character['id']}",
        json={**CHARACTER_FIELDS, "name": "Alicia"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Alicia"


def test_list_characters_is_ordered_by_position(client: TestClient) -> None:
    scenario = _create_scenario(client)
    _create_character(client, scenario["id"], "Alice")
    _create_character(client, scenario["id"], "Bob")
    _create_character(client, scenario["id"], "Charlie")

    response = client.get(f"/api/scenarios/{scenario['id']}/characters")
    assert response.status_code == 200
    names = [row["name"] for row in response.json()]
    assert names == ["Alice", "Bob", "Charlie"]


def test_characters_require_an_existing_scenario(client: TestClient) -> None:
    assert client.get("/api/scenarios/does-not-exist/characters").status_code == 404
    assert (
        client.post("/api/scenarios/does-not-exist/characters", json=CHARACTER_FIELDS).status_code
        == 404
    )


def test_missing_character_is_404(client: TestClient) -> None:
    scenario = _create_scenario(client)

    response = client.patch(
        f"/api/scenarios/{scenario['id']}/characters/999999", json=CHARACTER_FIELDS
    )
    assert response.status_code == 404

    response = client.delete(f"/api/scenarios/{scenario['id']}/characters/999999")
    assert response.status_code == 404


def test_reorder_characters(client: TestClient) -> None:
    scenario = _create_scenario(client)
    ids = [
        _create_character(client, scenario["id"], name)["id"]
        for name in ["Alice", "Bob", "Charlie"]
    ]

    reversed_order = list(reversed(ids))
    response = client.put(
        f"/api/scenarios/{scenario['id']}/characters/order",
        json={"order": reversed_order},
    )
    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body] == reversed_order
    assert [row["position"] for row in body] == [0, 1, 2]


def test_reorder_rejects_an_order_that_is_not_a_permutation(client: TestClient) -> None:
    scenario = _create_scenario(client)
    _create_character(client, scenario["id"], "Alice")

    response = client.put(
        f"/api/scenarios/{scenario['id']}/characters/order", json={"order": [999999]}
    )
    assert response.status_code == 400


def test_removing_a_character_keeps_positions_contiguous(client: TestClient) -> None:
    scenario = _create_scenario(client)
    ids = [
        _create_character(client, scenario["id"], name)["id"]
        for name in ["Alice", "Bob", "Charlie"]
    ]

    response = client.delete(f"/api/scenarios/{scenario['id']}/characters/{ids[0]}")
    assert response.status_code == 204

    response = client.get(f"/api/scenarios/{scenario['id']}/characters")
    remaining = response.json()
    assert [row["name"] for row in remaining] == ["Bob", "Charlie"]
    assert [row["position"] for row in remaining] == [0, 1]
