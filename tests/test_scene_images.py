"""Tests for background scene-image generation (issue #17), through comfy-gateway.

The centrepiece is a **fake comfy-gateway**: a real HTTP server on an
ephemeral port, shaped like the real gateway's API, whose answers each test
controls — a test can hold a submission, keep a job "running", make one fail
with a specific error, or stage a completion at a chosen moment. Nothing here
ever touches the gateway on `127.0.0.1:8190`: the client's base URL is
monkeypatched to the fake server for the duration of every test, and the only
requests it can see are the ones this suite makes.

The `gw` fixture also tightens `generation`'s poll cadence: a render takes
minutes in production, and a two-second poll would make the tests crawl.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterator
from http.server import ThreadingHTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, gateway, generation, settings
from persona_studio.main import app
from persona_studio.routes import parties as parties_routes

WAIT = 5.0
OPENING = "Le port s'ouvre."
PROMPT = "ruined castle at dusk, blue lantern"
INSTRUCTION = "Le château en ruine au crépuscule."
TEMPLATE = "Krea"


# --- The fake comfy-gateway -------------------------------------------------------


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Answers the gateway endpoints the client uses, as the test configured them."""

    def log_message(self, *args: Any) -> None:  # noqa: N802 - stdlib signature
        pass

    def _read_body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else None

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, body: bytes, status: int = 200, content_type: str = "text/html") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        fake: FakeGateway = self.server.fake  # type: ignore[attr-defined]
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        parts = path.strip("/").split("/")
        if path == "/api/health":
            fake.log(method="GET", path="/api/health")
            self._json({"ok": True, "comfy_url": "http://comfy", "comfy_reachable": fake.comfy_up})
        elif path == "/api/templates":
            fake.log(method="GET", path="/api/templates")
            with fake.lock:
                names = list(fake.templates)
            self._json([{"id": i + 1, "name": n} for i, n in enumerate(names)])
        elif path == "/api/templates/latest":
            name = urllib.parse.parse_qs(parsed.query).get("name", [""])[0]
            fake.log(method="GET", path="/api/templates/latest", name=name)
            with fake.lock:
                names = list(fake.templates)
                roles = fake.templates.get(name)
            if roles is None:
                self._json({"detail": f"there is no template named '{name}'"}, status=404)
            else:
                description = {"roles": {r: {} for r in roles}, "role_errors": {}}
                self._json({"id": names.index(name) + 1, "name": name, "description": description})
        elif len(parts) == 4 and parts[:2] == ["api", "templates"] and parts[3] == "presets":
            fake.log(method="GET", path="/api/templates/presets", template_id=parts[2])
            names = list(fake.templates)
            name = names[int(parts[2]) - 1]
            self._json([{"name": p} for p in fake.presets.get(name, [])])
        elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            job_id = parts[2]
            fake.log(method="GET", path="/api/jobs/{id}", job_id=job_id)
            with fake.lock:
                raw, override = fake.raw_answer, fake.job_answer
                outcome = fake.outcomes.get(job_id)
            if raw is not None:
                self._raw(raw)
            elif override is not None:
                self._json(override)
            elif job_id in fake.deleted:
                self._json({"detail": f"there is no job {job_id}"}, status=404)
            elif outcome is None:
                self._json({"id": int(job_id), "status": "running", "error": None, "images": []})
            else:
                self._json({"id": int(job_id), **outcome})
        elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "images":
            fake.log(method="GET", path="/api/jobs/{id}/images", job_id=parts[2])
            with fake.lock:
                status, payload = fake.view_status, fake.view_bytes
            self._raw(payload, status=status, content_type="image/png")
        else:
            self._json({"detail": f"unknown path {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        fake: FakeGateway = self.server.fake  # type: ignore[attr-defined]
        body = self._read_body()
        parts = self.path.strip("/").split("/")
        if self.path == "/api/jobs":
            fake.log(method="POST", path="/api/jobs", body=body)
            # A test that armed `hold_submit` parks the request thread here,
            # so it can read the database while the submission is in flight.
            if fake.hold_submit.is_set():
                fake.release_submit.wait(WAIT)
            with fake.lock:
                status, payload = fake.submit_status, fake.submit_body
            if status >= 400:
                self._json({"detail": fake.submit_detail}, status=status)
            elif payload is not None:
                self._json(payload)
            else:
                # Incrementing ids under the lock: two submissions are two
                # distinct jobs, as on the real gateway.
                with fake.lock:
                    fake._job_counter += 1
                    job_id = fake._job_counter
                self._json({"id": job_id, "status": "queued"}, status=201)
        elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
            fake.log(method="POST", path="/api/jobs/{id}/cancel", job_id=parts[2])
            self._json({"cancelled": True, "job": {}})
        else:
            self._json({"detail": f"unknown path {self.path}"}, status=404)


class FakeGateway:
    """A fake comfy-gateway on an ephemeral port whose answers each test controls.

    State is plain attributes mutated between calls: `outcomes` holds what
    `GET /api/jobs/{id}` answers for a finished job (no entry = still
    running), `submit_status` / `submit_detail` / `submit_body` shape the
    `POST /api/jobs` answer, `templates` maps a template name to its roles and
    `presets` a name to its presets, and the `hold_submit` / `release_submit`
    event pair parks a submission until the test releases it. Each accepted
    submission gets the next job id, so two live jobs are distinguishable.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.hold_submit = threading.Event()
        self.release_submit = threading.Event()
        self._job_counter = 0
        self.submit_status = 201
        self.submit_detail = "refused"
        self.submit_body: dict[str, Any] | None = None
        self.outcomes: dict[str, dict[str, Any]] = {}
        # Jobs deleted from the gateway: their state answers 404.
        self.deleted: set[str] = set()
        self.view_status = 200
        self.view_bytes = b"fake-png-bytes"
        self.templates: dict[str, list[str]] = {TEMPLATE: ["prompt", "seed"]}
        self.presets: dict[str, list[str]] = {TEMPLATE: ["fast"]}
        self.comfy_up = True
        # Overrides for the protocol-break tests: `job_answer` replaces the
        # whole job JSON, `raw_answer` the whole body.
        self.job_answer: Any = None
        self.raw_answer: bytes | None = None
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
        self._httpd.fake = self  # type: ignore[attr-defined]
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def log(self, **entry: Any) -> None:
        with self.lock:
            self.requests.append(entry)

    def calls(self, method: str, path: str) -> list[dict[str, Any]]:
        with self.lock:
            return [r for r in self.requests if r["method"] == method and r["path"] == path]

    def logged(self, method: str, path: str) -> bool:
        return bool(self.calls(method, path))

    def complete(self, job_id: str) -> None:
        """Make the job succeed with one image to fetch."""
        with self.lock:
            self.outcomes[job_id] = {
                "status": "done",
                "error": None,
                "images": [f"/api/jobs/{job_id}/images/0"],
            }

    def fail(self, job_id: str, error: str) -> None:
        """Make the job fail with the gateway's own error text."""
        with self.lock:
            self.outcomes[job_id] = {"status": "error", "error": error, "images": []}


@pytest.fixture
def gw(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGateway]:
    fake = FakeGateway()
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", fake.url)
    monkeypatch.setattr(generation, "POLL_INTERVAL_SECONDS", 0.05)
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def _clean_state(client: TestClient) -> Iterator[None]:
    """Fresh settings per test; the suite shares one database.

    Depends on `client` so the startup migration has run first. On the way
    out it also asserts that no watcher thread leaked: a daemon thread that
    outlives the `gw` fixture would reach the real gateway's address once the
    fake's URL is undone.
    """
    with db.connect() as con:
        con.execute("DELETE FROM setting")
    yield
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("image-gen-")]
    assert not leaked, f"a generation watcher never finished: {leaked}"
    with db.connect() as con:
        con.execute("DELETE FROM image")
        con.execute("DELETE FROM instance")  # cascades messages and variants
        con.execute("DELETE FROM setting")
    for path in db.IMAGES_DIR.glob("*.png"):
        path.unlink()


# --- Shared helpers --------------------------------------------------------------


def _wait_until(predicate: Callable[[], bool], timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _wait_watcher_done(message_id: int, timeout: float = WAIT) -> bool:
    """Wait until the watcher thread for this message is gone.

    Every test that starts a generation settles its job this way, before the
    fixture undoes the fake's URL.
    """
    return _wait_until(
        lambda: (
            not any(
                t.is_alive() and t.name == f"image-gen-{message_id}" for t in threading.enumerate()
            )
        ),
        timeout,
    )


def _stub_opening(monkeypatch: Any, reply: str) -> None:
    """Stub `ollama.chat` for the party's opening scene, as test_parties does.

    `parties.py` resolves the function at call time from its own import of
    the `ollama` module, so patching the attribute there takes effect.
    """
    monkeypatch.setattr(parties_routes.ollama, "chat", lambda *a, **k: reply)


def _stub_stream(monkeypatch: Any, replies: list[str]) -> None:
    """Stub `ollama.chat_stream` with a real generator: the route closes the
    stream in its finally, which a bare list iterator does not survive."""

    def fake_stream(*a: Any, **k: Any) -> Iterator[str]:
        yield from replies

    monkeypatch.setattr(parties_routes.ollama, "chat_stream", fake_stream)


def _setup_ready_workflow() -> None:
    """Choose the fake gateway's template, straight in the database."""
    with db.connect() as con:
        settings.set_image_generation(con, TEMPLATE, None)


def _create_party(client: TestClient, monkeypatch: Any) -> str:
    with db.connect() as con:
        settings.set_llm_model(con, "test-model")
    _stub_opening(monkeypatch, "Le port s'ouvre.")
    scenario = client.post("/api/scenarios", json={"title": "La Cité Noyée", "synopsis": ""})
    assert scenario.status_code == 201
    party = client.post(f"/api/scenarios/{scenario.json()['id']}/parties", json={"label": ""})
    assert party.status_code == 201
    return party.json()["id"]


def _start(client: TestClient, party_id: str, prompt: str = PROMPT) -> Any:
    return client.post(
        f"/api/parties/{party_id}/images",
        json={"prompt": prompt, "instruction": INSTRUCTION},
    )


def _cancel(client: TestClient, party_id: str, message_id: int) -> Any:
    return client.post(f"/api/parties/{party_id}/images/{message_id}/cancel")


def _message_rows(party_id: str) -> list[dict[str, Any]]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT id, kind, content, status, gen_id, image_id, started_at, error "
            "FROM message WHERE instance_id = ? ORDER BY id",
            (party_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _image_messages(party_id: str) -> list[dict[str, Any]]:
    return [row for row in _message_rows(party_id) if row["kind"] == "image"]


def _insert_pending(party_id: str, gen_id: str) -> int:
    """A pending message as `start` would have left it, with its gen_id.

    No `image_id`: the column references `image(id)`, and the image row only
    exists once the render completes — exactly what the real start does.
    """
    with db.connect() as con:
        cursor = con.execute(
            "INSERT INTO message "
            "(instance_id, role, kind, content, ts, status, gen_id, started_at) "
            "VALUES (?, 'assistant', 'image', 'la scène', ?, 'pending', ?, ?)",
            (party_id, time.time(), gen_id or None, time.time()),
        )
        assert cursor.lastrowid is not None
    return cursor.lastrowid


def _start_unfinished(client: TestClient, party_id: str) -> tuple[int, str]:
    """Start a generation the fake never completes.

    Returns (message_id, prompt_id); the caller must settle the job —
    complete, fail, or cancel it — before returning, so the watcher thread
    ends inside the test and never outlives the fake's URL.
    """
    response = _start(client, party_id)
    assert response.status_code == 201
    message_id = response.json()["message_id"]
    row = _image_messages(party_id)[0]
    assert row["status"] == "pending"
    return message_id, row["gen_id"]


def _settle(gw: FakeGateway, message_id: int, prompt_id: str) -> None:
    """Finish the job successfully and wait for its watcher to be gone."""
    gw.complete(prompt_id)
    assert _wait_watcher_done(message_id)


# --- 1. The pending message exists before the gateway is ever called --------------


def test_pending_message_is_persisted_before_the_submission(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Issue #17's first criterion, observed while the fake server is still
    holding the submission: the row exists, pending, stamped, and without a
    gen_id — and only the release lets the request finish."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    # Arming the hold parks the submission; releasing it lets the route answer.
    gw.hold_submit.set()

    holder: dict[str, Any] = {}

    def submit_request() -> None:
        # A fresh client without lifespan: the long request blocks in the
        # route while the test reads the database through the main client.
        with TestClient(app) as blocking_client:
            holder["response"] = _start(blocking_client, party_id)

    thread = threading.Thread(target=submit_request)
    thread.start()

    assert _wait_until(lambda: gw.logged("POST", "/api/jobs")), "the submission never arrived"
    rows = _image_messages(party_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["started_at"] is not None
    assert rows[0]["image_id"] is None  # the image row only exists once done
    assert rows[0]["gen_id"] is None

    gw.release_submit.set()
    thread.join(WAIT)
    assert not thread.is_alive()

    response = holder["response"]
    assert response.status_code == 201
    body = response.json()
    assert body["message_id"] == rows[0]["id"]
    assert body["started_at"] == rows[0]["started_at"]
    with db.connect() as con:
        stored = con.execute(
            "SELECT gen_id, status FROM message WHERE id = ?", (body["message_id"],)
        ).fetchone()
    assert stored["gen_id"] == "1"
    assert stored["status"] == "pending"

    _settle(gw, body["message_id"], "1")


# --- 2. The turn is never blocked -------------------------------------------------


def test_the_start_returns_immediately_and_the_narrator_waits_for_the_render(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The start still returns at once and the party still answers. What
    changed in V1.3 is the turn: issue #17 asked to keep chatting while an
    image renders, and on this machine's single 12 GB card the narrator
    loading back in mid-render is an out-of-memory error. The turn is refused
    with the reason, and nothing is written."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()

    message_id, prompt_id = _start_unfinished(client, party_id)

    # The start returned while the job is still unfinished: the fake holds no
    # history entry for it at all.
    assert gw.outcomes == {}

    # The party endpoint answers, with the pending message in place.
    detail = client.get(f"/api/parties/{party_id}")
    assert detail.status_code == 200
    assert any(m["kind"] == "image" and m["status"] == "pending" for m in detail.json()["messages"])

    # A turn is refused while the render runs, before the player's text is
    # written — a refused turn must leave nothing behind.
    _stub_stream(monkeypatch, ["Réponse."])
    before = len(_message_rows(party_id))
    turn = client.post(f"/api/parties/{party_id}/messages", json={"content": "J'avance."})
    assert turn.status_code == 409
    assert turn.json()["detail"] == generation.RENDER_IN_PROGRESS_DETAIL
    assert len(_message_rows(party_id)) == before

    _settle(gw, message_id, "1")


# --- 3. A completion arriving after a cancellation changes nothing ----------------


def test_a_completion_after_a_cancellation_is_discarded(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A completion that lands while the message is already cancelled must
    change nothing — status, image row, and disk included. The synchronisation
    point is the image fetch: the watcher provably saw the completion before
    the assertions below judge what it did with it."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, prompt_id = _start_unfinished(client, party_id)

    assert _cancel(client, party_id, message_id).status_code == 200
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert row["error"] == generation.CANCEL_MESSAGE

    gw.complete(prompt_id)

    # The watcher fetched the produced image, then attempted its write
    # against the cancelled message — and left the message exactly as it was.
    assert _wait_until(lambda: gw.logged("GET", "/api/jobs/{id}/images"))
    assert _wait_watcher_done(message_id)

    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert row["error"] == generation.CANCEL_MESSAGE
    assert row["image_id"] is None
    # Neither the image row nor the PNG it would have carried exists.
    with db.connect() as con:
        count = con.execute(
            "SELECT COUNT(*) FROM image WHERE id IN (SELECT image_id FROM message WHERE id = ?)",
            (message_id,),
        ).fetchone()[0]
    assert count == 0
    assert list(db.IMAGES_DIR.glob("*.png")) == []


def test_a_result_whose_gen_id_moved_on_is_discarded(client: TestClient, monkeypatch: Any) -> None:
    """The second half of the criterion, unit-level: a result carrying a
    `gen_id` other than the one the message was started with writes nothing,
    and a result for a message that no longer exists is a quiet no."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "started-with-this")

    stale = generation._Plan(
        party_id=party_id,
        message_id=message_id,
        image_id=uuid.uuid4().hex,
        gen_id="a-different-job",
        prompt=PROMPT,
        instruction=INSTRUCTION,
    )
    assert generation._finish(stale, done=True, png=b"late bytes") is False
    assert _image_messages(party_id)[0]["status"] == "pending"

    orphan = generation._Plan(
        party_id=party_id,
        message_id=999999,
        image_id=uuid.uuid4().hex,
        gen_id="whatever",
        prompt=PROMPT,
        instruction=INSTRUCTION,
    )
    assert generation._finish(orphan, done=True, png=b"x") is False


def test_a_cancel_during_the_submit_window_stops_the_job_and_stays_cancelled(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A cancel landing while the submission is in flight cannot stop the job:
    the row carries no `gen_id` yet, so the stop path has nothing to aim at.
    When the submission then returns, it must not stamp the id onto the
    cancelled row, must not start a watcher, and must stop the job itself —
    now that the id is finally known — the way the cancel does it. Also names
    the stop path's no-id-yet line: the cancel above runs it with `gen_id`
    still None, and the gateway is asked nothing."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    gw.hold_submit.set()

    holder: dict[str, Any] = {}

    def submit_request() -> None:
        with TestClient(app) as blocking_client:
            holder["response"] = _start(blocking_client, party_id)

    thread = threading.Thread(target=submit_request)
    thread.start()
    assert _wait_until(lambda: gw.logged("POST", "/api/jobs")), "the submission never arrived"

    # The row is pending and id-less while the submission is held.
    rows = _image_messages(party_id)
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["gen_id"] is None
    message_id = rows[0]["id"]

    # The cancel sees no gen_id: it records the cancellation and can ask the
    # gateway for nothing yet.
    assert _cancel(client, party_id, message_id).status_code == 200
    assert gw.calls("POST", "/api/jobs/{id}/cancel") == []

    gw.release_submit.set()
    thread.join(WAIT)
    assert not thread.is_alive()

    # Whatever the implementation under test started, end it while the fake
    # is still the addressed server — a red run must never leave a watcher
    # polling the real address.
    history_calls = gw.calls("GET", "/api/jobs/{id}")
    gw.complete("1")
    assert _wait_until(
        lambda: (
            not any(t.is_alive() and t.name.startswith("image-gen-") for t in threading.enumerate())
        )
    )

    assert holder["response"].status_code == 201
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert row["error"] == generation.CANCEL_MESSAGE
    assert row["gen_id"] is None, "the job id must not be stamped onto a cancelled row"
    assert history_calls == [], "no watcher may poll a job the cancel already recorded"
    cancels = gw.calls("POST", "/api/jobs/{id}/cancel")
    assert [c["job_id"] for c in cancels] == ["1"], "the job must be cancelled once its id is known"


def test_a_refused_submission_after_a_cancel_keeps_the_recorded_cancellation(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The other face of the same window: the submission is refused after a
    cancel already turned the row into a recorded cancellation. The refused
    path deletes the pending row it created — but a row a cancel has claimed
    is no longer pending, and deleting it would erase the cancellation from
    the story entirely."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    gw.submit_status = 500
    gw.hold_submit.set()

    holder: dict[str, Any] = {}

    def submit_request() -> None:
        with TestClient(app) as blocking_client:
            holder["response"] = _start(blocking_client, party_id)

    thread = threading.Thread(target=submit_request)
    thread.start()
    assert _wait_until(lambda: gw.logged("POST", "/api/jobs")), "the submission never arrived"

    rows = _image_messages(party_id)
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["gen_id"] is None
    message_id = rows[0]["id"]
    assert _cancel(client, party_id, message_id).status_code == 200

    gw.release_submit.set()
    thread.join(WAIT)
    assert not thread.is_alive()

    assert holder["response"].status_code == 502
    rows = _image_messages(party_id)
    assert len(rows) == 1, "the recorded cancellation must not be deleted"
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == generation.CANCEL_MESSAGE
    # The job id was never known: the stop path ran with None and the gateway
    # was asked for nothing at all.
    assert gw.calls("POST", "/api/jobs/{id}/cancel") == []


# --- 4. Cancelling stops the job on the gateway, and only that job ----------------


def test_cancelling_a_generation_cancels_its_gateway_job(
    gw: FakeGateway, client: TestClient, monkeypatch: Any
) -> None:
    """The gateway cancels a job by id, queued or running, and touches no
    other render: the cancel asks it for that job and nothing else."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "41")

    response = _cancel(client, party_id, message_id)

    assert response.status_code == 200
    assert [c["job_id"] for c in gw.calls("POST", "/api/jobs/{id}/cancel")] == ["41"]
    assert _image_messages(party_id)[0]["status"] == "error"


def test_an_unreachable_gateway_still_leaves_a_cancelled_message(
    client: TestClient, monkeypatch: Any
) -> None:
    """Whatever the gateway says — including silence — the message itself is
    cancelled locally, so the player never sees a spinner forever."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "some-job")
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", "http://127.0.0.1:1")

    response = _cancel(client, party_id, message_id)

    assert response.status_code == 200
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert row["error"] == generation.CANCEL_MESSAGE


def test_the_cancel_is_recorded_locally_before_the_gateway_is_touched(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Pins the order the module docstring states: the row is marked
    cancelled before the gateway is asked, so a process death between the
    two leaves a pending row the next startup heals — never a rendering job a
    cancellation never mentioned."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "7")

    observed: list[str] = []
    real_cancel = gateway.cancel

    def spy_cancel(job_id: str) -> None:
        with db.connect() as con:
            row = con.execute("SELECT status FROM message WHERE id = ?", (message_id,)).fetchone()
        observed.append(row["status"])
        real_cancel(job_id)

    monkeypatch.setattr(gateway, "cancel", spy_cancel)

    assert _cancel(client, party_id, message_id).status_code == 200
    assert observed == ["error"], "the gateway was asked before the cancellation was recorded"
    assert _image_messages(party_id)[0]["status"] == "error"


# --- 5. A failed job surfaces the gateway's real error ---------------------------


def test_a_failed_job_surfaces_the_gateway_real_error(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Issue #17's fifth criterion: the message carries the reason the job
    failed with — not a timeout, not a generic failure."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, job_id = _start_unfinished(client, party_id)
    gw.fail(job_id, "KSampler failed: CUDA out of memory on device 0")

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    error = _image_messages(party_id)[0]["error"]
    assert "KSampler" in error
    assert "CUDA out of memory on device 0" in error
    assert "did not finish within" not in error


def test_a_job_cancelled_in_the_gateway_fails_the_message(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A job stopped from the gateway's own interface ends the watcher with
    a message that says where it was stopped."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, job_id = _start_unfinished(client, party_id)
    gw.outcomes[job_id] = {"status": "cancelled", "error": None, "images": []}

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    assert "cancelled in comfy-gateway" in _image_messages(party_id)[0]["error"]


def test_the_gateway_becoming_unreachable_mid_render_fails_the_job_with_its_reason(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Criterion 5's promise applied to the failure mode nothing tested: the
    watcher is polling and the gateway goes away. The first unusable answer fails
    the job with the real reason — not a deadline, not a cancellation."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, _ = _start_unfinished(client, party_id)

    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", "http://127.0.0.1:1")

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    error = _image_messages(party_id)[0]["error"]
    assert "unreachable" in error
    assert "did not finish within" not in error


def test_a_success_without_any_image_is_a_failure(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A job done without any image must not hang the watcher: it is a
    failure with its own message."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, job_id = _start_unfinished(client, party_id)
    gw.outcomes[job_id] = {"status": "done", "error": None, "images": []}

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    assert "returned no image" in _image_messages(party_id)[0]["error"]
    assert list(db.IMAGES_DIR.glob("*.png")) == []


def test_the_deadline_expires_with_its_own_message(
    client: TestClient, gw: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline that genuinely expires is its own message: not the
    gateway's error, not a cancellation."""
    monkeypatch.setattr(generation, "DEADLINE_SECONDS", 0.2)
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, _ = _start_unfinished(client, party_id)

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    error = _image_messages(party_id)[0]["error"]
    assert "did not finish within" in error
    assert list(db.IMAGES_DIR.glob("*.png")) == []
    # Abandoned, the job must stop rendering: the narrator is unblocked now.
    assert gw.logged("POST", "/api/jobs/{id}/cancel")


# --- 6. A reload sees the pending image with its original started_at -------------


def test_a_reload_sees_the_pending_image_with_its_original_started_at(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The elapsed time comes from the stored `started_at`, so a reload shows
    the true age of the job rather than restarting from zero."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    response = _start(client, party_id)
    assert response.status_code == 201
    started_at = response.json()["started_at"]
    message_id = response.json()["message_id"]

    # A fresh GET — what a page reload does — carries the same row, with the
    # stamp the job started with.
    detail = client.get(f"/api/parties/{party_id}").json()
    image_messages = [m for m in detail["messages"] if m["kind"] == "image"]
    assert len(image_messages) == 1
    assert image_messages[0]["id"] == message_id
    assert image_messages[0]["status"] == "pending"
    assert image_messages[0]["started_at"] == started_at
    assert image_messages[0]["image_id"] is None  # the image row only exists once done

    _settle(gw, message_id, "1")


# --- 7. An unreachable gateway fails the start cleanly ---------------------------


def test_an_unreachable_gateway_leaves_no_pending_message(
    client: TestClient, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    before = len(_message_rows(party_id))
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", "http://127.0.0.1:1")

    response = _start(client, party_id)

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
    assert len(_message_rows(party_id)) == before
    assert not any(m["status"] == "pending" for m in _message_rows(party_id))


def test_a_refused_submission_leaves_no_pending_message(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A gateway that accepts the connection but answers a server error is
    the same clean failure: 502, nothing left behind."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    before = len(_message_rows(party_id))
    gw.submit_status = 500

    response = _start(client, party_id)

    assert response.status_code == 502
    assert len(_message_rows(party_id)) == before


def test_a_garbage_submission_answer_is_502(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    gw.submit_body = {"unexpected": "shape"}

    response = _start(client, party_id)

    assert response.status_code == 502
    assert "shape" in response.json()["detail"]


# --- 8. The image route refuses a path that escapes the images directory ---------


def test_the_image_route_refuses_anything_that_is_not_a_plain_id(
    client: TestClient,
) -> None:
    # Every path keeps the route's own shape (`.../<image_id>/file`), so the
    # id guard itself is what refuses, not the router.
    for image_id in (
        "..%2F..%2Fstudio.db",
        "..%2fsecret.png",
        "..",
        "%2e%2e%2e",
        uuid.uuid4().hex.upper(),
        "0" * 33,
        f"{uuid.uuid4().hex}/../studio.db",
    ):
        response = client.get(f"/api/images/{image_id}/file")
        assert response.status_code == 404, image_id
        assert "studio" not in response.text

    # A value that passes the id pattern but resolves outside the images
    # directory — here a symlink planted among the PNGs — is refused the
    # same way: the served file must sit inside the directory, not merely
    # carry a well-formed name.
    image_id = uuid.uuid4().hex
    outside = db.IMAGES_DIR.parent / "outside-the-images-dir.png"
    outside.write_bytes(b"not-an-image")
    link = db.IMAGES_DIR / f"{image_id}.png"
    link.symlink_to(outside)
    try:
        response = client.get(f"/api/images/{image_id}/file")
        assert response.status_code == 404, image_id
        assert "studio" not in response.text
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_the_image_route_serves_a_real_image(client: TestClient) -> None:
    image_id = uuid.uuid4().hex
    db.image_path(image_id).write_bytes(b"fake-png-bytes")
    try:
        response = client.get(f"/api/images/{image_id}/file")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")
        assert response.content == b"fake-png-bytes"
    finally:
        db.image_path(image_id).unlink(missing_ok=True)


def test_the_image_route_404s_an_unknown_id(client: TestClient) -> None:
    assert client.get(f"/api/images/{uuid.uuid4().hex}/file").status_code == 404


# --- 9. A setup that cannot generate refuses, and leaves nothing ------------------


def test_a_template_the_gateway_refuses_leaves_no_pending_message(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The gateway refuses the job and says why — here, a template without a
    prompt role. The player gets that reason as a setup problem (400), and
    the pending row the start wrote is gone again."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    gw.submit_status = 400
    gw.submit_detail = "this template has no 'prompt' role group"
    before = len(_message_rows(party_id))

    response = _start(client, party_id)

    assert response.status_code == 400
    assert "has no 'prompt' role group" in response.json()["detail"]
    assert len(_message_rows(party_id)) == before


def test_the_job_names_the_template_its_preset_and_the_prompt(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """What reaches the gateway: the chosen template and preset, the prompt
    as the `prompt` role exactly as the player wrote it, this application's
    client name, and the request to empty ComfyUI once the job is over, so
    the narrator finds the card free. Nothing else is sent."""
    party_id = _create_party(client, monkeypatch)
    with db.connect() as con:
        settings.set_image_generation(con, TEMPLATE, "fast")
    message_id, job_id = _start_unfinished(client, party_id)

    assert gw.calls("POST", "/api/jobs")[0]["body"] == {
        "template": TEMPLATE,
        "preset": "fast",
        "roles": {"prompt": PROMPT},
        "client": "persona-studio",
        "free_after": "all",
    }
    _settle(gw, message_id, job_id)


def test_no_template_chosen_refuses_before_anything_is_written(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    before = len(_message_rows(party_id))

    response = _start(client, party_id)

    assert response.status_code == 400
    assert "No image template" in response.json()["detail"]
    assert len(_message_rows(party_id)) == before
    assert not gw.logged("POST", "/api/jobs")


# --- The happy path ---------------------------------------------------------------


def test_a_completed_render_writes_everything_and_bumps_updated_at(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    updated_at_before = client.get(f"/api/parties/{party_id}").json()["updated_at"]
    message_id, prompt_id = _start_unfinished(client, party_id)
    gw.complete(prompt_id)

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "done")
    assert _wait_watcher_done(message_id)

    row = _image_messages(party_id)[0]
    image_id = row["image_id"]
    assert image_id is not None
    assert row["error"] is None
    assert db.image_path(image_id).read_bytes() == gw.view_bytes
    with db.connect() as con:
        image = con.execute("SELECT * FROM image WHERE id = ?", (image_id,)).fetchone()
        updated_at = con.execute(
            "SELECT updated_at FROM instance WHERE id = ?", (party_id,)
        ).fetchone()[0]
    assert image is not None
    assert image["prompt"] == PROMPT
    assert image["instruction"] == INSTRUCTION
    assert image["seconds"] is not None and image["seconds"] > 0
    assert updated_at > updated_at_before


def test_a_cancelled_message_then_reloaded_stays_cancelled(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """After a cancel, a reload shows the message — not a spinner: the party
    endpoint reports `error` status with the cancellation as its message."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, _ = _start_unfinished(client, party_id)
    assert _cancel(client, party_id, message_id).status_code == 200

    message = next(
        m for m in client.get(f"/api/parties/{party_id}").json()["messages"] if m["kind"] == "image"
    )
    assert message["status"] == "error"
    assert message["error"] == generation.CANCEL_MESSAGE

    # The watcher keeps polling until the gateway reports anything; letting the
    # (already discarded) completion through is what ends it.
    gw.complete("1")
    assert _wait_watcher_done(message_id)


# --- The client against the fake: the shapes the watcher relies on -----------------


def test_the_client_reads_a_job_tolerantly(gw: FakeGateway) -> None:
    """Missing or malformed fields read as absent rather than breaking."""
    gw.job_answer = {"status": "done", "error": 3, "images": ["/a", 4, "/b"]}
    state = gateway.job("1")
    assert state == gateway.JobState(status="done", error=None, images=["/a", "/b"])


def test_a_job_answer_without_a_status_is_a_protocol_break(gw: FakeGateway) -> None:
    gw.job_answer = {"images": []}
    with pytest.raises(gateway.GatewayError, match="job response shape"):
        gateway.job("1")


def test_a_non_json_answer_is_a_protocol_break(gw: FakeGateway) -> None:
    gw.raw_answer = b"<html>oops</html>"
    with pytest.raises(gateway.GatewayError, match="non-JSON"):
        gateway.job("1")


def test_a_job_deleted_from_the_gateway_fails_the_message_with_its_reason(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, job_id = _start_unfinished(client, party_id)
    gw.deleted.add(job_id)

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    assert f"there is no job {job_id}" in _image_messages(party_id)[0]["error"]


def test_an_image_error_is_surfaced_with_its_status(gw: FakeGateway) -> None:
    gw.view_status = 503
    with pytest.raises(gateway.GatewayError, match="503"):
        gateway.image_bytes("/api/jobs/1/images/0")


def test_an_unreachable_gateway_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(gateway.GatewayUnreachable, match="unreachable"):
        gateway.image_bytes("/api/jobs/1/images/0")


def test_a_template_without_a_prompt_role_says_so(gw: FakeGateway) -> None:
    gw.templates["Bare"] = ["seed"]
    assert gateway.template(TEMPLATE).problem is None
    problem = gateway.template("Bare").problem
    assert problem is not None and "no prompt role" in problem
    with pytest.raises(gateway.GatewayRefused, match="no template named"):
        gateway.template("Unknown")


# --- The startup recovery ----------------------------------------------------------


def test_recovery_marks_every_pending_image_failed(client: TestClient, monkeypatch: Any) -> None:
    """A process death leaves a pending row either without a `gen_id` (died
    inside the persist-to-submit window) or with one whose watcher is gone;
    neither can ever complete. Startup marks both, with a message that says
    which happened."""
    party_id = _create_party(client, monkeypatch)
    _insert_pending(party_id, "had-a-gen-id")
    _insert_pending(party_id, "")  # empty means NULL on the row

    generation.recover_pending()

    rows = _image_messages(party_id)
    assert {row["status"] for row in rows} == {"error"}
    errors = [row["error"] for row in rows]
    assert any("before the generation could start" in error for error in errors)
    assert any("while the image was rendering" in error for error in errors)


def test_startup_cancels_a_job_that_was_still_live(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A restart abandons a live render's record — the prompt lived only in
    the dead thread's memory — but the job itself can still be stopped.
    Before a pending row is marked error, its job is cancelled on the
    gateway, the way the cancel does it."""
    party_id = _create_party(client, monkeypatch)
    _insert_pending(party_id, "12")

    generation.recover_pending()

    assert [c["job_id"] for c in gw.calls("POST", "/api/jobs/{id}/cancel")] == ["12"]
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert "while the image was rendering" in row["error"]


def test_startup_recovery_completes_even_when_the_gateway_is_unreachable(
    client: TestClient, monkeypatch: Any
) -> None:
    """The stop attempt must not prevent the recovery: an unreachable gateway
    still leaves every pending row marked failed."""
    party_id = _create_party(client, monkeypatch)
    _insert_pending(party_id, "unreachable-job")
    monkeypatch.setattr(gateway, "GATEWAY_BASE_URL", "http://127.0.0.1:1")

    generation.recover_pending()

    rows = _image_messages(party_id)
    assert {row["status"] for row in rows} == {"error"}
    assert "while the image was rendering" in rows[0]["error"]


def test_a_done_message_whose_file_is_missing_becomes_an_error_at_startup(
    client: TestClient, monkeypatch: Any
) -> None:
    """The completion commits its rows and writes the PNG after, so a death
    between the two leaves a `done` message whose image id resolves to
    nothing — not pending, so `recover_pending` passes it by and it can never
    be cancelled. A fresh startup (a new lifespan, as a restart does) heals
    it into an error that says what happened; a `done` message whose file is
    present is left alone."""
    party_id = _create_party(client, monkeypatch)
    lost_id, intact_id = uuid.uuid4().hex, uuid.uuid4().hex
    with db.connect() as con:
        con.execute(
            "INSERT INTO image (id, created_at) VALUES (?, ?), (?, ?)",
            (lost_id, time.time(), intact_id, time.time()),
        )
        con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts, status, image_id) "
            "VALUES (?, 'assistant', 'image', 'perdue', ?, 'done', ?), "
            "(?, 'assistant', 'image', 'intacte', ?, 'done', ?)",
            (party_id, time.time(), lost_id, party_id, time.time(), intact_id),
        )
    db.image_path(intact_id).write_bytes(b"still-there")

    with TestClient(app) as restarted:
        assert restarted.get("/api/health").status_code == 200

    rows = {row["content"]: row for row in _image_messages(party_id)}
    assert rows["perdue"]["status"] == "error"
    assert "missing" in rows["perdue"]["error"]
    assert rows["intacte"]["status"] == "done"
    assert rows["intacte"]["error"] is None


def test_two_live_generations_keep_their_own_rows_files_and_outcomes(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Two concurrent generations are independent end to end: their own rows,
    their own `gen_id`s (the fake assigns incrementing ids), their own files.
    Cancelling one leaves the other rendering, and each completion lands on
    its own message."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()

    first = _start(client, party_id)
    second = _start(client, party_id)
    assert first.status_code == 201 and second.status_code == 201
    first_id, second_id = first.json()["message_id"], second.json()["message_id"]

    rows = _image_messages(party_id)
    assert [row["id"] for row in rows] == [first_id, second_id]
    first_prompt = rows[0]["gen_id"]
    second_prompt = rows[1]["gen_id"]
    assert first_prompt and second_prompt and first_prompt != second_prompt

    # Cancelling one leaves the other rendering.
    assert _cancel(client, party_id, first_id).status_code == 200
    rows = _image_messages(party_id)
    assert rows[0]["status"] == "error"
    assert rows[1]["status"] == "pending"

    gw.complete(first_prompt)
    gw.complete(second_prompt)

    assert _wait_watcher_done(first_id)
    assert _wait_watcher_done(second_id)
    rows = _image_messages(party_id)
    assert rows[0]["status"] == "error"
    assert rows[0]["image_id"] is None
    assert rows[1]["status"] == "done"
    assert rows[1]["gen_id"] == second_prompt
    assert db.image_path(rows[1]["image_id"]).read_bytes() == gw.view_bytes
    assert [p.name for p in db.IMAGES_DIR.glob("*.png")] == [f"{rows[1]['image_id']}.png"]


# --- Small extras -------------------------------------------------------------------


def test_a_blank_prompt_is_400(client: TestClient, monkeypatch: Any) -> None:
    party_id = _create_party(client, monkeypatch)
    response = client.post(
        f"/api/parties/{party_id}/images", json={"prompt": "   ", "instruction": "x"}
    )
    assert response.status_code == 400


def test_an_unknown_party_is_404(client: TestClient) -> None:
    missing = uuid.uuid4().hex
    assert _start(client, missing).status_code == 404
    assert _cancel(client, missing, 1).status_code == 404


def test_cancelling_a_done_or_text_message_is_400(client: TestClient, monkeypatch: Any) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id = _insert_pending(party_id, "to-finish")
    with db.connect() as con:
        con.execute("UPDATE message SET status = 'done' WHERE id = ?", (message_id,))
    assert _cancel(client, party_id, message_id).status_code == 400

    opening_id = client.get(f"/api/parties/{party_id}").json()["messages"][0]["id"]
    assert _cancel(client, party_id, opening_id).status_code == 400


# --- Freeing Ollama's VRAM before a render ---------------------------------------
#
# Measured on this machine: a chat model still resident when a diffusion
# pipeline loads is an out-of-memory error, or layers pushed onto the CPU.


def test_starting_a_generation_empties_ollama_before_submitting(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The wiring, through the real `start`: every Ollama model is unloaded,
    and before the gateway is asked to render anything."""
    order: list[str] = []
    monkeypatch.setattr(
        generation.gpu.ollama, "unload_all", lambda: order.append("unload_all") or []
    )
    real_submit = gateway.submit

    def spy_submit(*args: Any, **kwargs: Any) -> str:
        order.append("submit")
        return real_submit(*args, **kwargs)

    monkeypatch.setattr(gateway, "submit", spy_submit)
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)

    message_id, prompt_id = _start_unfinished(client, party_id)
    try:
        assert order == ["unload_all", "submit"]
    finally:
        _settle(gw, message_id, prompt_id)


def test_an_eviction_that_fails_does_not_fail_the_generation(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Best effort on purpose. At that point the pending message is committed
    and the player has asked for a render: an unreachable Ollama must not turn
    that into an error, and it is holding no VRAM to begin with."""

    def boom() -> list[str]:
        raise generation.gpu.ollama.OllamaUnreachable("Ollama est injoignable")

    monkeypatch.setattr(generation.gpu.ollama, "unload_all", boom)
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)

    message_id, prompt_id = _start_unfinished(client, party_id)
    _settle(gw, message_id, prompt_id)

    assert _image_messages(party_id)[0]["status"] == "done"


# --- Deleting an image (V1.2) ----------------------------------------------------
#
# The first delete path of the project: no message could be removed before
# this. The PNG goes with the row, and only after the transaction commits.


def _completed_image(client: TestClient, gw: FakeGateway, monkeypatch: Any) -> tuple[str, int, str]:
    """A party with one finished image. Returns (party_id, message_id, image_id)."""
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    message_id, prompt_id = _start_unfinished(client, party_id)
    _settle(gw, message_id, prompt_id)
    row = _image_messages(party_id)[0]
    assert row["status"] == "done"
    return party_id, message_id, row["image_id"]


def test_deleting_an_image_removes_its_message_row_and_file(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    party_id, message_id, image_id = _completed_image(client, gw, monkeypatch)
    assert db.image_path(image_id).is_file()

    response = client.delete(f"/api/parties/{party_id}/images/{message_id}")

    assert response.status_code == 204
    assert _image_messages(party_id) == []
    with db.connect() as con:
        assert con.execute("SELECT 1 FROM image WHERE id = ?", (image_id,)).fetchone() is None
    assert not db.image_path(image_id).exists(), "the PNG outlived its row"


def test_deleting_an_image_leaves_the_story_alone(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Images are messages, so a delete that reached too far would take the
    narration with it."""
    party_id, message_id, _ = _completed_image(client, gw, monkeypatch)
    before = [row["id"] for row in _message_rows(party_id) if row["kind"] == "text"]

    client.delete(f"/api/parties/{party_id}/images/{message_id}")

    after = [row["id"] for row in _message_rows(party_id) if row["kind"] == "text"]
    assert after == before


def test_a_generation_still_running_is_refused_rather_than_torn_out(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """Its watcher is about to write to that row. A completion landing on a
    deleted message is a state nothing here handles, so the delete refuses and
    names the way out."""
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    message_id, prompt_id = _start_unfinished(client, party_id)
    try:
        response = client.delete(f"/api/parties/{party_id}/images/{message_id}")

        assert response.status_code == 409
        assert "Cancel it" in response.json()["detail"]
        assert _image_messages(party_id)[0]["status"] == "pending"
    finally:
        _settle(gw, message_id, prompt_id)


def test_a_cancelled_image_can_be_deleted(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The way out the refusal names has to actually work.

    The job is settled on the fake's side after the cancel so the watcher
    ends inside the test: cancelling marks the row locally and does not, on
    its own, stop the thread from polling.
    """
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    message_id, prompt_id = _start_unfinished(client, party_id)
    assert _cancel(client, party_id, message_id).status_code == 200
    gw.complete(prompt_id)
    assert _wait_watcher_done(message_id)
    assert _image_messages(party_id)[0]["status"] == "error"

    assert client.delete(f"/api/parties/{party_id}/images/{message_id}").status_code == 204
    assert _image_messages(party_id) == []


def test_deleting_a_text_message_through_the_image_route_is_refused(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    opening_id = _message_rows(party_id)[0]["id"]

    response = client.delete(f"/api/parties/{party_id}/images/{opening_id}")

    assert response.status_code == 400
    assert len(_message_rows(party_id)) == 1, "the narration was deleted"


def test_deleting_an_unknown_image_is_404(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    assert client.delete(f"/api/parties/{party_id}/images/999999").status_code == 404


# --- The narrator waits for the render (V1.3) -------------------------------------
#
# The other half of the GPU guard. The narrator is evicted before a render
# starts; nothing may load it back while the render runs. Every route that
# calls the narrator is refused with the same reason, and once the render is
# done they all work again.


def test_every_narrator_call_is_refused_during_a_render(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    opening_id = _message_rows(party_id)[0]["id"]
    scenario = client.post("/api/scenarios", json={"title": "Autre monde", "synopsis": ""})
    message_id, prompt_id = _start_unfinished(client, party_id)
    try:
        regenerate = client.post(f"/api/parties/{party_id}/messages/{opening_id}/regenerate")
        new_party = client.post(
            f"/api/scenarios/{scenario.json()['id']}/parties", json={"label": ""}
        )
        compose = client.post(
            f"/api/parties/{party_id}/image-prompt", json={"instruction": "le port"}
        )

        for name, response in (
            ("regenerate", regenerate),
            ("new party", new_party),
            ("compose", compose),
        ):
            assert response.status_code == 409, f"{name} answered {response.status_code}"
            assert response.json()["detail"] == generation.RENDER_IN_PROGRESS_DETAIL
    finally:
        _settle(gw, message_id, prompt_id)


def test_the_narrator_answers_again_once_the_render_is_done(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """A guard that never lets go would be worse than the out-of-memory."""
    _setup_ready_workflow()
    party_id = _create_party(client, monkeypatch)
    message_id, prompt_id = _start_unfinished(client, party_id)
    _settle(gw, message_id, prompt_id)

    _stub_stream(monkeypatch, ["Le quai s'éveille."])
    turn = client.post(f"/api/parties/{party_id}/messages", json={"content": "J'avance."})

    assert turn.status_code == 200
    assert '"delta"' in turn.text


def test_a_render_in_another_party_blocks_this_one_too(
    client: TestClient, gw: FakeGateway, monkeypatch: Any
) -> None:
    """The GPU is one card, not one per party."""
    _setup_ready_workflow()
    rendering = _create_party(client, monkeypatch)
    other = _create_party(client, monkeypatch)
    message_id, prompt_id = _start_unfinished(client, rendering)
    try:
        _stub_stream(monkeypatch, ["Réponse."])
        turn = client.post(f"/api/parties/{other}/messages", json={"content": "J'avance."})
        assert turn.status_code == 409
    finally:
        _settle(gw, message_id, prompt_id)
