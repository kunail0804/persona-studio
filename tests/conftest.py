"""Shared fixtures.

`persona_studio.db` reads `PS_DATA` once, at its own import time, to compute
`DATA_DIR`. A fixture is too late to redirect it: pytest imports every test
module during collection, before any fixture runs, and those modules import
`persona_studio.db` at their own top level. So the redirect happens here
instead, as a side effect of importing this file — conftest.py is loaded
before collection ever reaches a test module, which is the one place early
enough to guarantee no test touches the real `data/`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ["PS_DATA"] = tempfile.mkdtemp(prefix="persona-studio-tests-")

from persona_studio import gateway, ollama  # noqa: E402 -- must follow the PS_DATA redirect above
from persona_studio.main import app  # noqa: E402 -- must follow the PS_DATA redirect above

# A port nothing listens on: a refused connection, at once.
_NOWHERE = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def _no_real_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the Ollama or the comfy-gateway running on this machine.

    Every model call now empties ComfyUI through the gateway first, and every
    render unloads Ollama: a test that forgot to stub one of them would free
    the developer's real memory, or stop a real render. Pointed at a closed
    port, those calls fail fast and are treated as "nothing to free". A test
    that needs a service points the URL at its own fake, after this runs.
    """
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", _NOWHERE)
    monkeypatch.setattr(ollama, "OLLAMA_BASE_URL", _NOWHERE)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
