"""Tests for background scene-image generation (issue #17).

The centrepiece is a **fake ComfyUI**: a real HTTP server on an ephemeral
port, shaped like the real API, whose responses each test controls — a test
can hold a submission, keep a job "running", make one fail with a specific
error, or stage a completion at a chosen moment. Nothing here ever touches
the ComfyUI on `127.0.0.1:8188`: the client's base URL is monkeypatched to
the fake server for the duration of every test, and the only requests it can
see are the ones this suite makes.

The `comfy` fixture also tightens `generation`'s poll cadence: a render takes
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

from persona_studio import comfyui, db, generation, settings
from persona_studio.main import app
from persona_studio.routes import parties as parties_routes

WAIT = 5.0
OPENING = "Le port s'ouvre."
PROMPT = "ruined castle at dusk, blue lantern"
INSTRUCTION = "Le château en ruine au crépuscule."


# --- The fake ComfyUI -----------------------------------------------------------


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Answers the five endpoints the client uses, as the test configured them."""

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

    def _empty_ok(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _raw(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        fake: FakeComfyUI = self.server.fake  # type: ignore[attr-defined]
        path = self.path.split("?")[0]
        if path.startswith("/history/"):
            prompt_id = path.removeprefix("/history/")
            fake.log(method="GET", path="/history", prompt_id=prompt_id)
            with fake.lock:
                raw, override = fake.raw_answer, fake.history_answer
            if raw is not None:
                self._raw(raw)
            elif override is not None:
                self._json(override)
            else:
                with fake.lock:
                    entry = fake.history_entries.get(prompt_id)
                self._json({prompt_id: entry} if entry is not None else {})
        elif path == "/queue":
            with fake.lock:
                running, pending = fake.queue_running, fake.queue_pending
            fake.log(method="GET", path="/queue")
            self._json({"queue_running": running, "queue_pending": pending})
        elif path == "/view":
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            with fake.lock:
                status, payload = fake.view_status, fake.view_bytes
            fake.log(method="GET", path="/view", params=params)
            self.send_response(status)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self._json({"error": f"unknown path {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        fake: FakeComfyUI = self.server.fake  # type: ignore[attr-defined]
        body = self._read_body()
        if self.path == "/prompt":
            fake.log(method="POST", path="/prompt", body=body)
            # A test that armed `hold_submit` parks the request thread here,
            # so it can read the database while the submission is in flight.
            if fake.hold_submit.is_set():
                fake.release_submit.wait(WAIT)
            with fake.lock:
                status, payload = fake.submit_status, fake.submit_body
            if status >= 400:
                self._json({"error": "refused"}, status=status)
            elif payload is not None:
                self._json(payload)
            else:
                # Incrementing ids under the lock: two submissions are two
                # distinct jobs, which is what a real ComfyUI does and what
                # the concurrency test below needs to observe.
                with fake.lock:
                    fake._prompt_counter += 1
                    prompt_id = f"fake-prompt-{fake._prompt_counter}"
                self._json({"prompt_id": prompt_id, "number": 1, "node_errors": {}})
        elif self.path == "/interrupt":
            fake.log(method="POST", path="/interrupt")
            self._empty_ok()
        elif self.path == "/queue":
            fake.log(method="POST", path="/queue", body=body)
            self._empty_ok()
        else:
            self._json({"error": f"unknown path {self.path}"}, status=404)

    def _empty_ok(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


class FakeComfyUI:
    """A fake ComfyUI on an ephemeral port whose answers each test controls.

    State is plain attributes mutated between calls: `history_entries` holds
    what `GET /history/{id}` answers (no entry = the job is still queued or
    running), `queue_running` / `queue_pending` hold queue entries as lists
    whose second element is the prompt id, `submit_status` / `submit_body`
    shape the `POST /prompt` answer, and the `hold_submit` / `release_submit`
    event pair parks a submission until the test releases it. Each accepted
    submission is assigned the next incrementing prompt id, so two live jobs
    are distinguishable the way they are on the real server.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.hold_submit = threading.Event()
        self.release_submit = threading.Event()
        self._prompt_counter = 0
        self.submit_status = 200
        self.submit_body: dict[str, Any] | None = None
        self.history_entries: dict[str, dict[str, Any]] = {}
        self.view_status = 200
        self.view_bytes = b"fake-png-bytes"
        # Overrides for the protocol-break tests: `history_answer` replaces
        # the whole /history JSON, `raw_answer` the whole body.
        self.history_answer: Any = None
        self.raw_answer: bytes | None = None
        self.queue_running: list[list[Any]] = []
        self.queue_pending: list[list[Any]] = []
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

    def complete(self, prompt_id: str, filename: str = "scene.png") -> None:
        """Make the job succeed with one produced image to fetch from /view."""
        self.history_entries[prompt_id] = {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {
                "9": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}
            },
        }

    def fail(self, prompt_id: str, reason: dict[str, Any]) -> None:
        """Make the job fail with a real `execution_error` status message."""
        self.history_entries[prompt_id] = {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [["execution_error", reason]],
            },
            "outputs": {},
        }


@pytest.fixture
def comfy(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeComfyUI]:
    fake = FakeComfyUI()
    monkeypatch.setattr(comfyui, "COMFYUI_BASE_URL", fake.url)
    monkeypatch.setattr(generation, "POLL_INTERVAL_SECONDS", 0.05)
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def _clean_state(client: TestClient) -> Iterator[None]:
    """Fresh workflows and settings per test; the suite shares one database.

    Depends on `client` so the startup migration has run first. On the way
    out it also asserts that no watcher thread leaked: a daemon thread that
    outlives the `comfy` fixture would reach the real ComfyUI's address once
    the fake's URL is undone.
    """
    with db.connect() as con:
        con.execute("DELETE FROM workflow")
        con.execute("DELETE FROM setting")
    yield
    leaked = [t.name for t in threading.enumerate() if t.name.startswith("image-gen-")]
    assert not leaked, f"a generation watcher never finished: {leaked}"
    with db.connect() as con:
        con.execute("DELETE FROM image")
        con.execute("DELETE FROM instance")  # cascades messages and variants
        con.execute("DELETE FROM workflow")
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


def _setup_ready_workflow() -> str:
    """Insert a mapped workflow and make it active, straight in the database."""
    workflow_id = uuid.uuid4().hex
    graph = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a castle", "clip": ["4", 1]}},
        "7": {"class_type": "KSampler", "inputs": {"seed": 42, "model": ["4", 0]}},
    }
    with db.connect() as con:
        con.execute(
            "INSERT INTO workflow (id, name, graph, prompt_node, prompt_field, created_at) "
            "VALUES (?, 'Krea', ?, '6', 'text', ?)",
            (workflow_id, json.dumps(graph), time.time()),
        )
        settings.set_active_workflow_id(con, workflow_id)
    return workflow_id


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


def _settle(comfy: FakeComfyUI, message_id: int, prompt_id: str) -> None:
    """Finish the job successfully and wait for its watcher to be gone."""
    comfy.complete(prompt_id)
    assert _wait_watcher_done(message_id)


# --- 1. The pending message exists before ComfyUI is ever called -----------------


def test_pending_message_is_persisted_before_the_submission(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Issue #17's first criterion, observed while the fake server is still
    holding the submission: the row exists, pending, stamped, and without a
    gen_id — and only the release lets the request finish."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    # Arming the hold parks the submission; releasing it lets the route answer.
    comfy.hold_submit.set()

    holder: dict[str, Any] = {}

    def submit_request() -> None:
        # A fresh client without lifespan: the long request blocks in the
        # route while the test reads the database through the main client.
        with TestClient(app) as blocking_client:
            holder["response"] = _start(blocking_client, party_id)

    thread = threading.Thread(target=submit_request)
    thread.start()

    assert _wait_until(lambda: comfy.logged("POST", "/prompt")), "the submission never arrived"
    rows = _image_messages(party_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["started_at"] is not None
    assert rows[0]["image_id"] is None  # the image row only exists once done
    assert rows[0]["gen_id"] is None

    comfy.release_submit.set()
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
    assert stored["gen_id"] == "fake-prompt-1"
    assert stored["status"] == "pending"

    _settle(comfy, body["message_id"], "fake-prompt-1")


# --- 2. The turn is never blocked -------------------------------------------------


def test_the_start_returns_immediately_and_the_party_keeps_answering(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()

    message_id, prompt_id = _start_unfinished(client, party_id)

    # The start returned while the job is still unfinished: the fake holds no
    # history entry for it at all.
    assert comfy.history_entries == {}

    # The party endpoint answers, with the pending message in place.
    detail = client.get(f"/api/parties/{party_id}")
    assert detail.status_code == 200
    assert any(m["kind"] == "image" and m["status"] == "pending" for m in detail.json()["messages"])

    # And a turn can still be played — stubbed, but through the whole
    # streaming path, which a blocked start would have starved.
    _stub_stream(monkeypatch, ["Réponse."])
    turn = client.post(f"/api/parties/{party_id}/messages", json={"content": "J'avance."})
    assert turn.status_code == 200
    assert '"delta"' in turn.text

    _settle(comfy, message_id, "fake-prompt-1")


# --- 3. A completion arriving after a cancellation changes nothing ----------------


def test_a_completion_after_a_cancellation_is_discarded(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """A completion that lands while the message is already cancelled must
    change nothing — status, image row, and disk included. The synchronisation
    point is the /view fetch: the watcher provably saw the completion before
    the assertions below judge what it did with it."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, prompt_id = _start_unfinished(client, party_id)

    assert _cancel(client, party_id, message_id).status_code == 200
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert row["error"] == generation.CANCEL_MESSAGE

    comfy.complete(prompt_id)

    # The watcher fetched the produced image, then attempted its write
    # against the cancelled message — and left the message exactly as it was.
    assert _wait_until(lambda: comfy.logged("GET", "/view"))
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
        seed=1,
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
        seed=1,
    )
    assert generation._finish(orphan, done=True, png=b"x") is False


def test_a_cancel_during_the_submit_window_stops_the_job_and_stays_cancelled(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """A cancel landing while the submission is in flight cannot stop the job:
    the row carries no `gen_id` yet, so the stop path has nothing to aim at.
    When the submission then returns, it must not stamp the id onto the
    cancelled row, must not start a watcher, and must stop the job itself —
    now that the id is finally known — through the same queue-checked path
    the cancel uses. Also names the stop path's no-id-yet line: the cancel
    above runs it with `gen_id` still None, and ComfyUI is asked nothing."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    comfy.hold_submit.set()

    holder: dict[str, Any] = {}

    def submit_request() -> None:
        with TestClient(app) as blocking_client:
            holder["response"] = _start(blocking_client, party_id)

    thread = threading.Thread(target=submit_request)
    thread.start()
    assert _wait_until(lambda: comfy.logged("POST", "/prompt")), "the submission never arrived"

    # The row is pending and id-less while the submission is held.
    rows = _image_messages(party_id)
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["gen_id"] is None
    message_id = rows[0]["id"]

    # The cancel sees no gen_id: it records the cancellation and can do
    # nothing on ComfyUI. The queue entry staged below is answered only once
    # the id becomes known — whoever reads it then must act on it.
    comfy.queue_running = [["entry", "fake-prompt-1"]]
    assert _cancel(client, party_id, message_id).status_code == 200
    assert comfy.calls("GET", "/queue") == []

    comfy.release_submit.set()
    thread.join(WAIT)
    assert not thread.is_alive()

    # Whatever the implementation under test started, end it while the fake
    # is still the addressed server — a red run must never leave a watcher
    # polling the real address.
    history_calls = comfy.calls("GET", "/history")
    comfy.complete("fake-prompt-1")
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
    assert comfy.logged("GET", "/queue"), "the stopped job must be found through the queue"
    assert comfy.logged("POST", "/interrupt"), "the job must be interrupted once its id is known"


def test_a_refused_submission_after_a_cancel_keeps_the_recorded_cancellation(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """The other face of the same window: the submission is refused after a
    cancel already turned the row into a recorded cancellation. The refused
    path deletes the pending row it created — but a row a cancel has claimed
    is no longer pending, and deleting it would erase the cancellation from
    the story entirely."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    comfy.submit_status = 500
    comfy.hold_submit.set()

    holder: dict[str, Any] = {}

    def submit_request() -> None:
        with TestClient(app) as blocking_client:
            holder["response"] = _start(blocking_client, party_id)

    thread = threading.Thread(target=submit_request)
    thread.start()
    assert _wait_until(lambda: comfy.logged("POST", "/prompt")), "the submission never arrived"

    rows = _image_messages(party_id)
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["gen_id"] is None
    message_id = rows[0]["id"]
    assert _cancel(client, party_id, message_id).status_code == 200

    comfy.release_submit.set()
    thread.join(WAIT)
    assert not thread.is_alive()

    assert holder["response"].status_code == 502
    rows = _image_messages(party_id)
    assert len(rows) == 1, "the recorded cancellation must not be deleted"
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == generation.CANCEL_MESSAGE
    # The job id was never known: the stop path ran with None and ComfyUI was
    # asked for nothing at all.
    assert comfy.calls("GET", "/queue") == []
    assert comfy.calls("POST", "/interrupt") == []


# --- 4. Cancelling acts on the queue only as far as it is safe --------------------


def test_cancelling_a_running_generation_interrupts(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Our prompt id is the one running: interrupt, and nothing else."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "running-job")
    comfy.queue_running = [["entry", "running-job"]]

    response = _cancel(client, party_id, message_id)

    assert response.status_code == 200
    assert comfy.logged("POST", "/interrupt")
    assert not comfy.logged("POST", "/queue")
    assert _image_messages(party_id)[0]["status"] == "error"


def test_cancelling_a_queued_generation_removes_it_without_interrupting(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Our prompt id is waiting while someone else's renders: remove the
    queue entry and do not interrupt — the interrupt would kill the
    unrelated job. Asserted on what the fake server was asked, which is the
    criterion itself."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "queued-job")
    comfy.queue_pending = [["entry", "queued-job"]]

    response = _cancel(client, party_id, message_id)

    assert response.status_code == 200
    deletes = comfy.calls("POST", "/queue")
    assert len(deletes) == 1
    assert deletes[0]["body"] == {"delete": ["queued-job"]}
    assert not comfy.logged("POST", "/interrupt")
    assert _image_messages(party_id)[0]["status"] == "error"


def test_cancelling_a_generation_not_on_comfyui_asks_it_for_nothing(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Our prompt id is in neither list: it finished or never queued —
    ComfyUI is never mutated. The queue read is the look that makes the
    decision safe; nothing more is asked of it."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "already-gone")

    response = _cancel(client, party_id, message_id)

    assert response.status_code == 200
    assert comfy.calls("POST", "/interrupt") == []
    assert comfy.calls("POST", "/queue") == []
    assert _image_messages(party_id)[0]["status"] == "error"


def test_an_unreachable_comfyui_still_leaves_a_cancelled_message(
    client: TestClient, monkeypatch: Any
) -> None:
    """Whatever the queue says — including silence — the message itself is
    cancelled locally, so the player never sees a spinner forever."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "some-job")
    monkeypatch.setattr(comfyui, "COMFYUI_BASE_URL", "http://127.0.0.1:1")

    response = _cancel(client, party_id, message_id)

    assert response.status_code == 200
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert row["error"] == generation.CANCEL_MESSAGE


def test_the_cancel_is_recorded_locally_before_comfyui_is_touched(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Pins the order the module docstring states: the row is marked
    cancelled before the queue is read, so a process death between the two
    leaves a pending row the next startup heals — never a rendering job a
    cancellation never mentioned."""
    party_id = _create_party(client, monkeypatch)
    message_id = _insert_pending(party_id, "order-job")
    comfy.queue_running = [["entry", "order-job"]]

    observed: list[str] = []
    real_queue = comfyui.queue

    def spy_queue() -> comfyui.QueueState:
        with db.connect() as con:
            row = con.execute("SELECT status FROM message WHERE id = ?", (message_id,)).fetchone()
        observed.append(row["status"])
        return real_queue()

    monkeypatch.setattr(comfyui, "queue", spy_queue)

    assert _cancel(client, party_id, message_id).status_code == 200
    assert observed == ["error"], "the queue was read before the cancellation was recorded"
    assert _image_messages(party_id)[0]["status"] == "error"


# --- 5. A failed workflow surfaces ComfyUI's real error --------------------------


def test_a_failed_execution_surfaces_comfyui_real_error(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Issue #17's fifth criterion: the message carries the reason from the
    history entry — not a timeout, not a generic failure."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, prompt_id = _start_unfinished(client, party_id)
    comfy.fail(
        prompt_id,
        {
            "node_type": "KSampler",
            "exception_type": "OutOfMemoryError",
            "exception_message": "CUDA out of memory on device 0",
        },
    )

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    error = _image_messages(party_id)[0]["error"]
    assert "KSampler" in error
    assert "CUDA out of memory on device 0" in error
    assert "did not finish within" not in error


def test_comfyui_becoming_unreachable_mid_render_fails_the_job_with_its_reason(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """Criterion 5's promise applied to the failure mode nothing tested: the
    watcher is polling and ComfyUI goes away. The first unusable answer fails
    the job with the real reason — not a deadline, not a cancellation."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, _ = _start_unfinished(client, party_id)

    monkeypatch.setattr(comfyui, "COMFYUI_BASE_URL", "http://127.0.0.1:1")

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    error = _image_messages(party_id)[0]["error"]
    assert "unreachable" in error
    assert "did not finish within" not in error


def test_a_success_without_any_image_is_a_failure(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """An entry that completed without any output image must not hang the
    watcher: it is a failure with its own message."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, prompt_id = _start_unfinished(client, party_id)
    comfy.history_entries[prompt_id] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {},
    }

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    assert "produced no image" in _image_messages(party_id)[0]["error"]
    assert list(db.IMAGES_DIR.glob("*.png")) == []


def test_the_deadline_expires_with_its_own_message(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline that genuinely expires is its own message: not ComfyUI's
    error, not a cancellation."""
    monkeypatch.setattr(generation, "DEADLINE_SECONDS", 0.2)
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    message_id, _ = _start_unfinished(client, party_id)

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "error")
    assert _wait_watcher_done(message_id)
    error = _image_messages(party_id)[0]["error"]
    assert "did not finish within" in error
    assert list(db.IMAGES_DIR.glob("*.png")) == []


# --- 6. A reload sees the pending image with its original started_at -------------


def test_a_reload_sees_the_pending_image_with_its_original_started_at(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
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

    _settle(comfy, message_id, "fake-prompt-1")


# --- 7. An unreachable ComfyUI fails the start cleanly ---------------------------


def test_an_unreachable_comfyui_leaves_no_pending_message(
    client: TestClient, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    before = len(_message_rows(party_id))
    monkeypatch.setattr(comfyui, "COMFYUI_BASE_URL", "http://127.0.0.1:1")

    response = _start(client, party_id)

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
    assert len(_message_rows(party_id)) == before
    assert not any(m["status"] == "pending" for m in _message_rows(party_id))


def test_a_refused_submission_leaves_no_pending_message(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """A ComfyUI that accepts the connection but answers an HTTP error is the
    same clean failure: 502, nothing left behind."""
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    before = len(_message_rows(party_id))
    comfy.submit_status = 500

    response = _start(client, party_id)

    assert response.status_code == 502
    assert len(_message_rows(party_id)) == before


def test_a_garbage_submission_answer_is_502(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    comfy.submit_body = {"unexpected": "shape"}

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


# --- 9. A workflow that cannot generate refuses before anything is written -------


def test_an_unmapped_workflow_refuses_before_anything_is_written(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """An imported but unmapped workflow — the refusal carries the exact
    message the settings page shows — and nothing reaches ComfyUI or the
    database."""
    party_id = _create_party(client, monkeypatch)
    with db.connect() as con:
        con.execute(
            "INSERT INTO workflow (id, name, graph, created_at) VALUES (?, 'Krea', ?, ?)",
            (
                uuid.uuid4().hex,
                json.dumps({"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}),
                time.time(),
            ),
        )
        # The first insert became active on its own; it is unmapped.
    before = len(_message_rows(party_id))

    response = _start(client, party_id)

    assert response.status_code == 400
    assert "No prompt field" in response.json()["detail"]
    assert len(_message_rows(party_id)) == before
    assert not comfy.logged("POST", "/prompt")


def test_no_workflow_at_all_refuses_before_anything_is_written(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    before = len(_message_rows(party_id))

    response = _start(client, party_id)

    assert response.status_code == 400
    assert "No workflow" in response.json()["detail"]
    assert len(_message_rows(party_id)) == before


# --- The happy path ---------------------------------------------------------------


def test_a_completed_render_writes_everything_and_bumps_updated_at(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    party_id = _create_party(client, monkeypatch)
    _setup_ready_workflow()
    updated_at_before = client.get(f"/api/parties/{party_id}").json()["updated_at"]
    message_id, prompt_id = _start_unfinished(client, party_id)
    comfy.complete(prompt_id, filename="the-scene.png")

    assert _wait_until(lambda: _image_messages(party_id)[0]["status"] == "done")
    assert _wait_watcher_done(message_id)

    row = _image_messages(party_id)[0]
    image_id = row["image_id"]
    assert image_id is not None
    assert row["error"] is None
    assert db.image_path(image_id).read_bytes() == comfy.view_bytes
    with db.connect() as con:
        image = con.execute("SELECT * FROM image WHERE id = ?", (image_id,)).fetchone()
        updated_at = con.execute(
            "SELECT updated_at FROM instance WHERE id = ?", (party_id,)
        ).fetchone()[0]
    assert image is not None
    assert image["prompt"] == PROMPT
    assert image["instruction"] == INSTRUCTION
    assert isinstance(image["seed"], int)
    assert image["seconds"] is not None and image["seconds"] > 0
    assert updated_at > updated_at_before


def test_a_cancelled_message_then_reloaded_stays_cancelled(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
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

    # The watcher keeps polling until ComfyUI reports anything; letting the
    # (already discarded) completion through is what ends it.
    comfy.complete("fake-prompt-1")
    assert _wait_watcher_done(message_id)


# --- The client against the fake: the shapes the watcher relies on -----------------


def test_the_client_parses_a_complete_history_entry(comfy: FakeComfyUI) -> None:
    """A real entry's `outputs` map is read tolerantly: malformed nodes and
    items are skipped, and missing subfolder/type fall back to what /view
    needs to be asked with."""
    comfy.history_entries["p1"] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            "9": {
                "images": [
                    {"filename": "a.png", "subfolder": "out", "type": "output"},
                    {"filename": "b.png"},
                    "junk",
                ]
            },
            "junk-node": "not an object",
            "10": {"images": "not a list"},
        },
    }
    entry = comfyui.history("p1")
    assert entry is not None
    assert entry.completed is True
    assert entry.error is None
    assert entry.images == [
        comfyui.ImageRef("a.png", "out", "output"),
        comfyui.ImageRef("b.png", "", "output"),
    ]


def test_a_history_entry_without_a_status_object_is_a_protocol_break(comfy: FakeComfyUI) -> None:
    comfy.history_entries["p1"] = {"outputs": {}}
    with pytest.raises(comfyui.ComfyUIError, match="status object"):
        comfyui.history("p1")


def test_a_history_entry_that_is_not_an_object_is_a_protocol_break(comfy: FakeComfyUI) -> None:
    comfy.history_entries["p1"] = "junk"
    with pytest.raises(comfyui.ComfyUIError, match="entry shape"):
        comfyui.history("p1")


def test_a_history_answer_that_is_not_an_object_is_a_protocol_break(comfy: FakeComfyUI) -> None:
    comfy.history_answer = ["a", "list"]
    with pytest.raises(comfyui.ComfyUIError, match="history response"):
        comfyui.history("p1")


def test_a_non_json_answer_is_a_protocol_break(comfy: FakeComfyUI) -> None:
    comfy.raw_answer = b"<html>oops</html>"
    with pytest.raises(comfyui.ComfyUIError, match="non-JSON"):
        comfyui.history("p1")


def test_a_view_error_is_surfaced_with_its_status(comfy: FakeComfyUI) -> None:
    comfy.view_status = 503
    with pytest.raises(comfyui.ComfyUIError, match="/view"):
        comfyui.image_bytes(comfyui.ImageRef("a.png", "", "output"))


def test_an_unreachable_view_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfyui, "COMFYUI_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(comfyui.ComfyUIUnreachable):
        comfyui.image_bytes(comfyui.ImageRef("a.png", "", "output"))


def test_queue_entries_that_carry_no_prompt_id_are_skipped(comfy: FakeComfyUI) -> None:
    comfy.queue_running = [["x"], ["x", 3], "junk", ["a", "b", "c"]]
    comfy.queue_pending = [["ok", "waiting"]]
    state = comfyui.queue()
    assert state.running == ["b"]
    assert state.pending == ["waiting"]


def test_the_error_reason_falls_back_when_the_fields_are_missing() -> None:
    assert comfyui._error_from_messages("junk") is None
    assert comfyui._error_from_messages([["other", {}]]) is None
    assert comfyui._error_from_messages([["execution_error", {"node_id": "9"}]]) == (
        "9: unknown error"
    )
    assert comfyui._error_from_messages([["execution_error", {"exception_type": "OOM"}]]) == (
        "?: OOM"
    )


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


def test_startup_stops_a_render_that_was_still_live(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """A restart abandons a live render's record — the prompt lived only in
    the dead thread's memory — but the render itself can still be stopped.
    Before a pending row is marked error, its job goes through the same
    queue-checked stop the cancel uses: running means interrupt."""
    party_id = _create_party(client, monkeypatch)
    _insert_pending(party_id, "still-live-job")
    comfy.queue_running = [["entry", "still-live-job"]]

    generation.recover_pending()

    assert comfy.logged("GET", "/queue")
    assert comfy.logged("POST", "/interrupt")
    assert not comfy.logged("POST", "/queue")
    row = _image_messages(party_id)[0]
    assert row["status"] == "error"
    assert "while the image was rendering" in row["error"]


def test_startup_stops_a_merely_queued_job_by_removing_it(
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
) -> None:
    """The same recovery, for a job that never started rendering: removed
    from the queue, and never interrupted."""
    party_id = _create_party(client, monkeypatch)
    _insert_pending(party_id, "still-queued-job")
    comfy.queue_pending = [["entry", "still-queued-job"]]

    generation.recover_pending()

    deletes = comfy.calls("POST", "/queue")
    assert len(deletes) == 1
    assert deletes[0]["body"] == {"delete": ["still-queued-job"]}
    assert not comfy.logged("POST", "/interrupt")
    assert _image_messages(party_id)[0]["status"] == "error"


def test_startup_recovery_completes_even_when_comfyui_is_unreachable(
    client: TestClient, monkeypatch: Any
) -> None:
    """The stop attempt must not prevent the recovery: an unreachable ComfyUI
    still leaves every pending row marked failed."""
    party_id = _create_party(client, monkeypatch)
    _insert_pending(party_id, "unreachable-job")
    monkeypatch.setattr(comfyui, "COMFYUI_BASE_URL", "http://127.0.0.1:1")

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
    client: TestClient, comfy: FakeComfyUI, monkeypatch: Any
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

    comfy.complete(first_prompt)
    comfy.complete(second_prompt, filename="second.png")

    assert _wait_watcher_done(first_id)
    assert _wait_watcher_done(second_id)
    rows = _image_messages(party_id)
    assert rows[0]["status"] == "error"
    assert rows[0]["image_id"] is None
    assert rows[1]["status"] == "done"
    assert rows[1]["gen_id"] == second_prompt
    assert db.image_path(rows[1]["image_id"]).read_bytes() == comfy.view_bytes
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
