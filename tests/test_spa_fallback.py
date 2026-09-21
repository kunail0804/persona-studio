"""The SPA catch-all: serves index.html for client-side routes, 404s for /api and /assets.

`persona_studio.main` only registers this route when `web/dist` exists, and
the `python` CI job never runs `pnpm build`, so these lines are otherwise
uncovered. `register_spa_fallback` takes the built directory as a parameter
specifically so a test can point it at a throwaway one instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from persona_studio.main import is_reserved_path, register_spa_fallback


@pytest.mark.parametrize(
    ("full_path", "expected"),
    [
        ("api", True),
        ("api/health", True),
        ("api/scenarios/1", True),
        ("assets", True),
        ("assets/app.js", True),
        ("", False),
        ("scenarios/abc", False),
        ("apix", False),
        ("assetsy", False),
    ],
)
def test_is_reserved_path(full_path: str, expected: bool) -> None:
    assert is_reserved_path(full_path) is expected


@pytest.fixture
def spa_client(tmp_path: Path) -> TestClient:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("// fake bundle")
    (tmp_path / "index.html").write_text("<!doctype html><title>Persona Studio</title>")

    app = FastAPI()
    register_spa_fallback(app, tmp_path)
    return TestClient(app)


def test_deep_link_returns_index_html(spa_client: TestClient) -> None:
    response = spa_client.get("/scenarios/some-id")
    assert response.status_code == 200
    assert "Persona Studio" in response.text


def test_unmatched_api_path_is_a_404(spa_client: TestClient) -> None:
    response = spa_client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"


def test_unmatched_asset_path_is_a_404(spa_client: TestClient) -> None:
    response = spa_client.get("/assets/nope.js")
    assert response.status_code == 404


def test_bare_api_is_a_404(spa_client: TestClient) -> None:
    response = spa_client.get("/api")
    assert response.status_code == 404


def test_bare_assets_is_a_404(spa_client: TestClient) -> None:
    response = spa_client.get("/assets")
    assert response.status_code == 404


def test_unmatched_api_path_is_a_404_for_every_method(spa_client: TestClient) -> None:
    """The catch-all above is GET-only, so every other method used to be
    rejected at Starlette's method check: a 405 with no body, which reads as
    "wrong verb on a real endpoint" when no such endpoint exists at all."""
    for method in ("post", "put", "patch", "delete"):
        response = getattr(spa_client, method)("/api/nope")
        assert response.status_code == 404, f"{method.upper()} answered {response.status_code}"
        assert response.json()["detail"] == "Not found"


def test_a_non_get_on_a_client_side_route_is_also_a_404(spa_client: TestClient) -> None:
    """Serving index.html here would be worse than the 405 was: a POST to a
    React Router path is not a page request."""
    assert spa_client.post("/scenarios/abc").status_code == 404
