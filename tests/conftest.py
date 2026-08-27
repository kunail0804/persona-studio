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

from persona_studio.main import app  # noqa: E402 -- must follow the PS_DATA redirect above


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
