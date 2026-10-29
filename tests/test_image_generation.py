"""The settings route that chooses the comfy-gateway template and preset.

It runs against the same fake gateway as the generation tests: the lists it
offers and the problems it reports come from what the gateway answers.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, gateway, generation, settings
from tests.test_scene_images import TEMPLATE, FakeGateway


@pytest.fixture
def gw(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGateway]:
    fake = FakeGateway()
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", fake.url)
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def _clean_settings(client: TestClient) -> Iterator[None]:
    with db.connect() as con:
        con.execute("DELETE FROM setting")
    yield
    with db.connect() as con:
        con.execute("DELETE FROM setting")


def test_nothing_chosen_offers_the_gateway_templates(client: TestClient, gw: FakeGateway) -> None:
    gw.templates["Other"] = ["prompt"]

    body = client.get("/api/image-generation").json()

    assert body["template"] is None and body["preset"] is None
    assert body["templates"] == [TEMPLATE, "Other"]
    assert body["presets"] == []
    assert body["problem"] == generation.NO_TEMPLATE_MESSAGE
    assert body["gateway_url"] == gw.url


def test_choosing_a_template_and_preset_stores_both(client: TestClient, gw: FakeGateway) -> None:
    response = client.put("/api/image-generation", json={"template": TEMPLATE, "preset": "fast"})

    assert response.status_code == 200
    body = response.json()
    assert (body["template"], body["preset"], body["problem"]) == (TEMPLATE, "fast", None)
    assert body["presets"] == ["fast"]
    with db.connect() as con:
        assert settings.get_image_template(con) == TEMPLATE
        assert settings.get_image_gateway_preset(con) == "fast"


def test_names_the_gateway_does_not_have_are_refused(client: TestClient, gw: FakeGateway) -> None:
    unknown = client.put("/api/image-generation", json={"template": "Nope"})
    assert unknown.status_code == 422 and "Nope" in unknown.json()["detail"]
    preset = client.put("/api/image-generation", json={"template": TEMPLATE, "preset": "slow"})
    assert preset.status_code == 422 and "slow" in preset.json()["detail"]
    with db.connect() as con:
        assert settings.get_image_template(con) is None


def test_no_template_clears_the_preset_too(client: TestClient, gw: FakeGateway) -> None:
    client.put("/api/image-generation", json={"template": TEMPLATE, "preset": "fast"})

    body = client.put("/api/image-generation", json={"template": None, "preset": "fast"}).json()

    assert body["template"] is None and body["preset"] is None
    with db.connect() as con:
        assert settings.get_image_gateway_preset(con) is None


def test_a_setup_that_no_longer_holds_says_why(client: TestClient, gw: FakeGateway) -> None:
    with db.connect() as con:
        settings.set_image_generation(con, "Gone", None)
    assert "no longer in comfy-gateway" in client.get("/api/image-generation").json()["problem"]

    with db.connect() as con:
        settings.set_image_generation(con, TEMPLATE, "deleted-preset")
    assert "deleted-preset" in client.get("/api/image-generation").json()["problem"]

    gw.templates["Bare"] = ["seed"]
    with db.connect() as con:
        settings.set_image_generation(con, "Bare", None)
    assert "no prompt role" in client.get("/api/image-generation").json()["problem"]


def test_an_unreachable_gateway_keeps_the_choice_and_says_so(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", "http://127.0.0.1:1")
    with db.connect() as con:
        settings.set_image_generation(con, TEMPLATE, None)

    body = client.get("/api/image-generation").json()

    assert body["template"] == TEMPLATE
    assert body["templates"] == []
    assert "unreachable" in body["problem"]
    put = client.put("/api/image-generation", json={"template": TEMPLATE})
    assert put.status_code == 503
