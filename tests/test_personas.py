from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from persona_studio import db


@pytest.fixture(autouse=True)
def _clean_personas(client: TestClient) -> None:
    """The active persona id is global state in the shared test database.

    Depends on `client` so the app's startup migration has run first — the
    tables must exist even when this file runs alone.
    """
    with db.connect() as con:
        con.execute("DELETE FROM persona")
        con.execute("DELETE FROM setting")


PERSONA_FIELDS = {
    "name": "Elara",
    "description": "Une archéologue qui ne croit qu'à ce qu'elle touche.",
    "appearance": "Cheveux noirs courts, veste kaki, cicatrice à la main gauche.",
    "traits": "Obstinée, curieuse, méfiante.",
}


def _create_persona(client: TestClient, name: str = "Elara") -> dict:
    response = client.post("/api/personas", json={**PERSONA_FIELDS, "name": name})
    assert response.status_code == 201
    return response.json()


def test_create_edit_and_get_persona(client: TestClient) -> None:
    persona = _create_persona(client)
    assert persona["name"] == "Elara"
    assert persona["appearance"] == PERSONA_FIELDS["appearance"]
    assert persona["is_active"] is False

    response = client.patch(
        f"/api/personas/{persona['id']}", json={**PERSONA_FIELDS, "name": "Elara Vey"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Elara Vey"
    assert response.json()["description"] == PERSONA_FIELDS["description"]

    response = client.get(f"/api/personas/{persona['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == persona["id"]


def test_selecting_a_persona_marks_it_active_alone(client: TestClient) -> None:
    first = _create_persona(client, "Première")
    second = _create_persona(client, "Deuxième")

    response = client.put("/api/personas/active", json={"id": first["id"]})
    assert response.status_code == 200
    assert response.json()["is_active"] is True

    listed = client.get("/api/personas").json()
    active = [row["is_active"] for row in listed if row["id"] == first["id"]]
    others = [row["is_active"] for row in listed if row["id"] != first["id"]]
    assert active == [True]
    assert all(is_active is False for is_active in others)

    response = client.put("/api/personas/active", json={"id": second["id"]})
    assert response.status_code == 200
    listed = client.get("/api/personas").json()
    assert next(row for row in listed if row["id"] == second["id"])["is_active"] is True
    assert next(row for row in listed if row["id"] == first["id"])["is_active"] is False


def test_deleting_the_active_persona_leaves_no_dangling_id(client: TestClient) -> None:
    persona = _create_persona(client)

    assert client.put("/api/personas/active", json={"id": persona["id"]}).status_code == 200
    assert client.delete(f"/api/personas/{persona['id']}").status_code == 204

    # No persona is active any more, and the setting row itself is gone —
    # the stored value must not survive as a dangling id.
    assert all(row["is_active"] is False for row in client.get("/api/personas").json())
    with db.connect() as con:
        row = con.execute("SELECT value FROM setting WHERE key = 'persona.active_id'").fetchone()
    assert row is None


def test_deleting_an_inactive_persona_keeps_the_active_one(client: TestClient) -> None:
    kept = _create_persona(client, "Gardée")
    removed = _create_persona(client, "Supprimée")
    assert client.put("/api/personas/active", json={"id": kept["id"]}).status_code == 200

    assert client.delete(f"/api/personas/{removed['id']}").status_code == 204

    assert client.get(f"/api/personas/{kept['id']}").json()["is_active"] is True


def test_active_persona_must_exist(client: TestClient) -> None:
    response = client.put("/api/personas/active", json={"id": "does-not-exist"})
    assert response.status_code == 404


def test_missing_persona_is_404(client: TestClient) -> None:
    assert client.get("/api/personas/does-not-exist").status_code == 404
    assert client.patch("/api/personas/does-not-exist", json=PERSONA_FIELDS).status_code == 404
    assert client.delete("/api/personas/does-not-exist").status_code == 404
