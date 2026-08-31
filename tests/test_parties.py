"""Tests for the party routes.

Ollama is never called from a test: `ollama.chat` and `ollama.chat_stream` are
stubbed where the route looks them up. `parties.py` does `from .. import
ollama` and resolves the functions at call time, so patching the attribute on
that module takes effect. The configured model is written straight to the
`setting` table — the settings PUT route would call the real
`ollama.list_models`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient

from persona_studio import db, settings, summarizer
from persona_studio.narrator import OPENING_INSTRUCTION
from persona_studio.ollama import OllamaError, OllamaUnreachable
from persona_studio.routes import parties as parties_routes

OPENING = "The harbour gate looms ahead, barnacled and half open. What do you do?"


@pytest.fixture(autouse=True)
def _default_history_window(client: TestClient) -> Iterator[None]:
    """The suite shares one database, and a window left low by a summariser
    test would make unrelated turns trigger background jobs mid-test. Depends
    on `client` because only its startup runs the migration."""
    _set_window(settings.DEFAULT_HISTORY_WINDOW)
    yield
    _set_window(settings.DEFAULT_HISTORY_WINDOW)


def _configure_model(model: str | None, num_ctx: int | None = None) -> None:
    with db.connect() as con:
        settings.set_llm_model(con, model)
        if num_ctx is not None:
            settings.set_num_ctx(con, num_ctx)


def _stub_chat(
    monkeypatch: Any, reply: str | None = None, error: Exception | None = None
) -> list[dict[str, Any]]:
    """Replace `ollama.chat` with a recording stub; return the calls it saw."""
    calls: list[dict[str, Any]] = []

    def fake_chat(model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
        calls.append({"model": model, "messages": messages, "num_ctx": num_ctx})
        if error is not None:
            raise error
        assert reply is not None
        return reply

    monkeypatch.setattr(parties_routes.ollama, "chat", fake_chat)
    return calls


def _stub_chat_stream(
    monkeypatch: Any, replies: list[str] | None = None, error: Exception | None = None
) -> list[dict[str, Any]]:
    """Replace `ollama.chat_stream` with a recording stub; return the calls.

    The stub yields the fragments in order, then raises `error` if one was
    given — the shape a stream dying mid-turn has from the route's side.
    """
    calls: list[dict[str, Any]] = []

    def fake_chat_stream(model: str, messages: list[dict[str, str]], num_ctx: int) -> Iterator[str]:
        calls.append({"model": model, "messages": messages, "num_ctx": num_ctx})
        if replies is not None:
            yield from replies
        if error is not None:
            raise error

    monkeypatch.setattr(parties_routes.ollama, "chat_stream", fake_chat_stream)
    return calls


def _create_scenario(client: TestClient, title: str = "La Cité Noyée") -> str:
    response = client.post("/api/scenarios", json={"title": title, "synopsis": ""})
    assert response.status_code == 201
    return response.json()["id"]


def _create_party(client: TestClient, scenario_id: str, monkeypatch: Any) -> dict[str, Any]:
    _configure_model("test-model")
    _stub_chat(monkeypatch, reply=OPENING)
    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})
    assert response.status_code == 201
    return response.json()


def _party_counts(scenario_id: str) -> tuple[int, int]:
    """(parties, messages) for one scenario — the "nothing was written" check.

    The suite shares one database, and `tests/test_narrator.py` legitimately
    inserts `instance` rows for its loader tests, so a global COUNT(*) would
    never be zero there. Counting only the scenario under test states the same
    criterion at the scope that can actually hold it.
    """
    with db.connect() as con:
        parties = con.execute(
            "SELECT COUNT(*) FROM instance WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()[0]
        messages = con.execute(
            "SELECT COUNT(*) FROM message "
            "WHERE instance_id IN (SELECT id FROM instance WHERE scenario_id = ?)",
            (scenario_id,),
        ).fetchone()[0]
    return parties, messages


def _message_count_for_party(party_id: str) -> int:
    with db.connect() as con:
        return con.execute(
            "SELECT COUNT(*) FROM message WHERE instance_id = ?", (party_id,)
        ).fetchone()[0]


# --- The headline criterion: a failed opening leaves nothing behind ------------


def test_unreachable_ollama_leaves_nothing_behind(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")
    _stub_chat(
        monkeypatch,
        error=OllamaUnreachable(
            "Ollama is unreachable at http://localhost:11434: connection refused"
        ),
    )

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
    assert _party_counts(scenario_id) == (0, 0)


def test_empty_reply_leaves_nothing_behind(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")
    _stub_chat(monkeypatch, reply="   \n\t ")

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 502
    assert _party_counts(scenario_id) == (0, 0)


def test_ollama_error_is_502_and_leaves_nothing_behind(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")
    _stub_chat(monkeypatch, error=OllamaError("Ollama returned HTTP 404: model not found"))

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 502
    assert "model not found" in response.json()["detail"]
    assert _party_counts(scenario_id) == (0, 0)


def test_no_model_configured_is_400_and_writes_nothing(client: TestClient) -> None:
    scenario_id = _create_scenario(client)
    _configure_model(None)

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 400
    # The detail must name where the fix lives.
    assert "settings" in response.json()["detail"]
    assert _party_counts(scenario_id) == (0, 0)


def test_scenario_deleted_during_generation_is_404_and_writes_nothing(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    _configure_model("test-model")

    def delete_scenario_then_reply(model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
        with db.connect() as con:
            con.execute("DELETE FROM scenario WHERE id = ?", (scenario_id,))
        return OPENING

    monkeypatch.setattr(parties_routes.ollama, "chat", delete_scenario_then_reply)

    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert response.status_code == 404
    assert "deleted while the opening scene was being generated" in response.json()["detail"]
    # The rollback already held; the point here is that the surfaced error is a
    # 404 and the no-write guarantee survived it.
    assert _party_counts(scenario_id) == (0, 0)


# --- A successful create -------------------------------------------------------


def test_create_writes_one_party_with_the_opening_as_assistant_message(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    party = _create_party(client, scenario_id, monkeypatch)

    assert party["label"] == "La Cité Noyée"
    assert party["scenario_title"] == "La Cité Noyée"

    detail = client.get(f"/api/parties/{party['id']}")
    assert detail.status_code == 200
    messages = detail.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == OPENING
    assert _party_counts(scenario_id) == (1, 1)


def test_create_uses_the_provided_label_when_there_is_one(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    _configure_model("test-model")
    _stub_chat(monkeypatch, reply=OPENING)

    response = client.post(
        f"/api/scenarios/{scenario_id}/parties", json={"label": "Campagne d'Ash"}
    )

    assert response.status_code == 201
    assert response.json()["label"] == "Campagne d'Ash"


def test_create_sends_the_assembled_prompt_and_the_configured_num_ctx(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    _configure_model("test-model", num_ctx=2048)
    calls = _stub_chat(monkeypatch, reply=OPENING)

    client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})

    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["num_ctx"] == 2048
    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system"
    # The system prompt is the assembled scenario prompt, not a stub.
    assert "Scenario: La Cité Noyée" in messages[0]["content"]
    # The opening direction rides as a second system message.
    assert messages[1] == {"role": "system", "content": OPENING_INSTRUCTION}


# --- Listing, renaming, deleting ------------------------------------------------


def test_list_orders_parties_most_recently_active_first(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    first = _create_party(client, scenario_id, monkeypatch)["id"]
    second = _create_party(client, scenario_id, monkeypatch)["id"]
    third = _create_party(client, scenario_id, monkeypatch)["id"]
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (1000.0, first))
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (2000.0, second))
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (3000.0, third))

    listed = [party["id"] for party in client.get("/api/parties").json()]

    # Other tests may have left parties behind; only the relative order of
    # these three is under test here.
    mine = [party_id for party_id in listed if party_id in {first, second, third}]
    assert mine == [third, second, first]


def test_rename_changes_the_label_without_touching_updated_at(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    older = _create_party(client, scenario_id, monkeypatch)["id"]
    newer = _create_party(client, scenario_id, monkeypatch)["id"]
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (1000.0, older))
        con.execute("UPDATE instance SET updated_at = ? WHERE id = ?", (2000.0, newer))
    updated_at_before = client.get(f"/api/parties/{older}").json()["updated_at"]

    response = client.patch(f"/api/parties/{older}", json={"label": "Nouveau nom"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Nouveau nom"
    assert body["updated_at"] == updated_at_before
    # The renamed party stays in its original place: a rename is not activity.
    listed = [party["id"] for party in client.get("/api/parties").json()]
    mine = [party_id for party_id in listed if party_id in {older, newer}]
    assert mine == [newer, older]


def test_rename_to_blank_falls_back_to_the_scenario_title(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]

    for blank in ("", "   "):
        response = client.patch(f"/api/parties/{party_id}", json={"label": blank})

        assert response.status_code == 200
        assert response.json()["label"] == "La Cité Noyée"


def test_rename_stores_the_sent_label_stripped(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]

    response = client.patch(f"/api/parties/{party_id}", json={"label": "  Campagne d'Ash  "})

    assert response.status_code == 200
    assert response.json()["label"] == "Campagne d'Ash"


def test_delete_removes_the_party_and_its_messages(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]

    response = client.delete(f"/api/parties/{party_id}")

    assert response.status_code == 204
    assert client.get(f"/api/parties/{party_id}").status_code == 404
    assert _message_count_for_party(party_id) == 0


def test_unknown_party_and_scenario_ids_are_404(client: TestClient) -> None:
    missing = uuid.uuid4().hex
    assert client.get(f"/api/parties/{missing}").status_code == 404
    assert client.patch(f"/api/parties/{missing}", json={"label": "x"}).status_code == 404
    assert client.delete(f"/api/parties/{missing}").status_code == 404
    assert client.post(f"/api/scenarios/{missing}/parties", json={"label": ""}).status_code == 404


# --- Playing a turn -------------------------------------------------------------


def _send_turn(client: TestClient, party_id: str, content: str) -> Any:
    return client.post(f"/api/parties/{party_id}/messages", json={"content": content})


def _ndjson_events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _party_message_rows(party_id: str) -> list[dict[str, Any]]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT id, role, content FROM message WHERE instance_id = ? ORDER BY id",
            (party_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def test_turn_is_persisted_before_generation(client: TestClient, monkeypatch: Any) -> None:
    """Issue #9's first criterion, observed from inside the generation itself:
    a stream stub that reads the database when it starts sees the player's
    turn already there and no reply yet. An insert moved after the generation
    call would show only the opening here."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    observed: list[list[str]] = []

    def spy_stream(model: str, messages: list[dict[str, str]], num_ctx: int) -> Iterator[str]:
        # The real `chat_stream` is a generator factory: the call itself cannot
        # fail, the error surfaces on the first pull. The stub models that —
        # the database is read from the first pull, not from the call.
        observed.append([m["role"] for m in _party_message_rows(party_id)])
        raise OllamaError("Ollama died on the first token")
        yield ""  # never reached; its presence is what makes this a generator

    monkeypatch.setattr(parties_routes.ollama, "chat_stream", spy_stream)

    response = _send_turn(client, party_id, "J'entre dans le port.")

    assert response.status_code == 200
    assert observed == [["assistant", "user"]]
    assert [m["role"] for m in _party_message_rows(party_id)] == ["assistant", "user"]
    assert _party_message_rows(party_id)[-1]["content"] == "J'entre dans le port."


def test_partial_reply_survives_a_broken_stream(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    reason = (
        "Ollama is unreachable at http://localhost:11434: "
        "peer closed connection without sending complete message body"
    )
    _stub_chat_stream(monkeypatch, replies=["Le guide ", "sourit."], error=OllamaError(reason))

    response = _send_turn(client, party_id, "Je m'approche.")

    assert response.status_code == 200
    events = _ndjson_events(response.text)
    assert events[0] == {"delta": "Le guide "}
    assert events[1] == {"delta": "sourit."}
    # The error line carries the real reason, not "something failed".
    assert events[2] == {"error": reason}
    # The partial was persisted, and the final line says which message it is.
    assert events[3]["done"] is True
    messages = _party_message_rows(party_id)
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert messages[-1]["content"] == "Le guide sourit."
    assert events[3]["message_id"] == messages[-1]["id"]


def test_partial_reply_survives_a_client_hangup(client: TestClient, monkeypatch: Any) -> None:
    """TestClient cannot stage a real mid-stream disconnect, so the generator
    is disposed the way Starlette's finalizer disposes it: consumed up to the
    first fragment, then aclose()d — a GeneratorExit at the yield, which skips
    everything but the `finally` that persists the partial."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _stub_chat_stream(monkeypatch, replies=["Le guide sourit et ", "tend la main."])

    ctx = parties_routes._prepare_turn(party_id, parties_routes.TurnInput(content="Je le suis."))

    async def hang_up_after_first_fragment() -> None:
        stream = parties_routes._turn_events(ctx)
        first = await stream.__anext__()
        assert json.loads(first) == {"delta": "Le guide sourit et "}
        await stream.aclose()

    asyncio.run(hang_up_after_first_fragment())

    messages = _party_message_rows(party_id)
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert messages[-1]["content"] == "Le guide sourit et "


def test_cancellation_during_an_outstanding_pull_persists_the_partial(
    client: TestClient, monkeypatch: Any
) -> None:
    """The real hang-up Starlette delivers is a task cancellation raised while
    the frame awaits inside `anyio.to_thread.run_sync` — not the GeneratorExit
    at a suspended yield that `test_partial_reply_survives_a_client_hangup`
    stages. Here a real anyio scope is cancelled while a fragment pull is
    outstanding — the worker thread blocked mid-pull on an event — and two
    guards the route relies on are pinned:

    - the partial is persisted inside the shielded scope (shield=True): with
      the shield gone, the persist await re-raises the still-delivered
      cancellation before the worker thread even starts, and no row appears;
    - the unwind does not wait for the pull (abandon_on_cancel=True): the
      task group exits while the pull is still in flight. With it False,
      anyio defers the cancellation until the pull returns — so the pull must
      finish before the group can exit, and `pull_finished` is already set.

    Because the finally's `stream.close()` then runs while the worker thread
    is inside `next()`, it raises `ValueError: generator already executing` —
    the race the `except ValueError` guard exists to swallow.

    Every wait is bounded, so a failing run ends instead of deadlocking on a
    blocked worker thread.
    """
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    first_fragment = "Le guide sourit et "

    # Signal plumbing between the test and the stub's worker thread:
    # `pull_started` says a pull is outstanding, `release_pull` unblocks it,
    # `pull_finished` says the pull is over.
    pull_started = threading.Event()
    pull_finished = threading.Event()
    release_pull = threading.Event()
    unblock_timeout = 5.0

    def fake_chat_stream(model: str, messages: list[dict[str, str]], num_ctx: int) -> Iterator[str]:
        yield first_fragment
        # The second pull announces itself, then blocks mid-pull: the
        # cancellation must land here, not between fragments.
        pull_started.set()
        release_pull.wait(unblock_timeout)
        pull_finished.set()
        yield "tend la main."

    monkeypatch.setattr(parties_routes.ollama, "chat_stream", fake_chat_stream)

    ctx = parties_routes._prepare_turn(party_id, parties_routes.TurnInput(content="Je le suis."))

    async def run_turn() -> None:
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                stream = parties_routes._turn_events(ctx)
                first = await stream.__anext__()
                assert json.loads(first) == {"delta": first_fragment}
                # This await is cancelled while its worker thread sits blocked
                # mid-pull; the unwinding runs the shielded persist.
                await stream.__anext__()

            async def cancel_when_pull_is_outstanding() -> None:
                # Wait on a worker thread — the event loop must stay free —
                # for the signal that a pull is blocked mid-stream, then
                # cancel the response task exactly while it is outstanding.
                await anyio.to_thread.run_sync(pull_started.wait, unblock_timeout)
                tg.cancel_scope.cancel()

            tg.start_soon(consume)
            tg.start_soon(cancel_when_pull_is_outstanding)

        # The task group has exited while the pull is still blocked: the
        # unwind abandoned it rather than waiting for it.
        assert not pull_finished.is_set()

        release_pull.set()
        assert pull_finished.wait(unblock_timeout)

    anyio.run(run_turn)

    # The partial — the first fragment only — was persisted as the reply.
    messages = _party_message_rows(party_id)
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert messages[-1]["content"] == first_fragment


def test_empty_reply_appends_nothing(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _stub_chat_stream(monkeypatch, replies=[])

    response = _send_turn(client, party_id, "Tour sans réponse.")

    assert response.status_code == 200
    assert _ndjson_events(response.text) == [{"done": True, "message_id": None}]
    # The player's turn stays — it was persisted on purpose. No assistant row.
    assert [m["role"] for m in _party_message_rows(party_id)] == ["assistant", "user"]


def test_whitespace_reply_appends_nothing(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _stub_chat_stream(monkeypatch, replies=["  ", "\n\t"])

    response = _send_turn(client, party_id, "Tour sans réponse.")

    assert response.status_code == 200
    # Whitespace fragments ride the stream; only the database row is skipped.
    assert _ndjson_events(response.text)[-1] == {"done": True, "message_id": None}
    assert [m["role"] for m in _party_message_rows(party_id)] == ["assistant", "user"]


def test_turn_sends_the_system_prompt_the_history_and_the_turn(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client, "La Cité Noyée")
    _configure_model("test-model", num_ctx=2048)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    calls = _stub_chat_stream(monkeypatch, replies=["Bien."])

    _send_turn(client, party_id, "J'avance dans le brouillard.")

    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["num_ctx"] == 2048
    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "Scenario: La Cité Noyée" in messages[0]["content"]
    # The history: the opening, then the turn just persisted — loaded after
    # the insert, not hand-appended.
    assert messages[1] == {"role": "assistant", "content": OPENING}
    assert messages[-1] == {"role": "user", "content": "J'avance dans le brouillard."}


def test_history_window_limits_the_messages_sent(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _stub_chat_stream(monkeypatch, replies=["Réponse 1."])
    _send_turn(client, party_id, "Tour 1.")
    _stub_chat_stream(monkeypatch, replies=["Réponse 2."])
    _send_turn(client, party_id, "Tour 2.")
    with db.connect() as con:
        settings.set_history_window(con, 2)
    calls = _stub_chat_stream(monkeypatch, replies=["Réponse 3."])
    # Six text messages are now uncovered — past the window — so this turn
    # also schedules a background summarisation. Stub its model call and wait
    # for the job to finish, so no daemon thread ever reaches the real
    # Ollama after the stubs are undone.
    summary_calls = _stub_summary_chat(
        monkeypatch, reply=json.dumps({"summary": "Résumé.", "world_state": {}})
    )

    _send_turn(client, party_id, "Tour 3.")

    messages = calls[0]["messages"]
    assert len(messages) == 3  # the system prompt, plus the last two only
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "Réponse 2."}
    assert messages[2] == {"role": "user", "content": "Tour 3."}
    assert _wait_until(lambda: len(summary_calls) == 1)
    assert _wait_unlocked(party_id)


def test_turn_bumps_updated_at_but_rename_does_not(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = 1000.0 WHERE id = ?", (party_id,))

    _stub_chat_stream(monkeypatch, replies=["Bien."])
    _send_turn(client, party_id, "Un vrai tour.")

    updated_after_turn = client.get(f"/api/parties/{party_id}").json()["updated_at"]
    assert updated_after_turn > 1000.0

    client.patch(f"/api/parties/{party_id}", json={"label": "Nouveau nom"})

    assert client.get(f"/api/parties/{party_id}").json()["updated_at"] == updated_after_turn


def test_blank_turn_is_400_and_writes_nothing(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    before = _message_count_for_party(party_id)

    for blank in ("", "   \n\t "):
        response = _send_turn(client, party_id, blank)
        assert response.status_code == 400
        assert response.json()["detail"] == "A turn cannot be empty."

    assert _message_count_for_party(party_id) == before


def test_turn_without_model_is_400_and_writes_nothing(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _configure_model(None)
    before = _message_count_for_party(party_id)

    response = _send_turn(client, party_id, "Un tour.")

    assert response.status_code == 400
    assert "settings" in response.json()["detail"]
    assert _message_count_for_party(party_id) == before


def test_turn_on_unknown_party_is_404(client: TestClient) -> None:
    missing = uuid.uuid4().hex

    response = _send_turn(client, missing, "Un tour.")

    assert response.status_code == 404
    assert client.get(f"/api/parties/{missing}").status_code == 404


# --- Editing, regenerating, variants --------------------------------------------
#
# The variant invariant, asserted directly against the database after every
# path in this section: a message with no variant rows has exactly one text
# (`message.content`); a message with variant rows has exactly one with
# `active = 1` and `message.content` equals it.


def _regenerate(client: TestClient, party_id: str, message_id: int) -> Any:
    return client.post(f"/api/parties/{party_id}/messages/{message_id}/regenerate")


def _opening_message_id(client: TestClient, party_id: str) -> int:
    detail = client.get(f"/api/parties/{party_id}")
    assert detail.status_code == 200
    return detail.json()["messages"][0]["id"]


def _edit(client: TestClient, party_id: str, message_id: int, content: str) -> Any:
    return client.patch(f"/api/parties/{party_id}/messages/{message_id}", json={"content": content})


def _switch_variant(client: TestClient, party_id: str, message_id: int, variant_id: int) -> Any:
    return client.put(
        f"/api/parties/{party_id}/messages/{message_id}/variant", json={"variant_id": variant_id}
    )


def _variant_rows(message_id: int) -> list[dict[str, Any]]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT id, content, active FROM variant WHERE message_id = ? ORDER BY id",
            (message_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _message_content(message_id: int) -> str:
    with db.connect() as con:
        return con.execute("SELECT content FROM message WHERE id = ?", (message_id,)).fetchone()[
            "content"
        ]


def _assert_invariant(message_id: int) -> None:
    rows = _variant_rows(message_id)
    content = _message_content(message_id)
    actives = [row for row in rows if row["active"]]
    if not rows:
        return
    assert len(actives) == 1, f"expected exactly one active variant, got {actives}"
    assert actives[0]["content"] == content


def _insert_image_message(party_id: str) -> int:
    with db.connect() as con:
        cursor = con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts) "
            "VALUES (?, 'assistant', 'image', 'la scène', ?)",
            (party_id, time.time()),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def test_get_party_returns_variants_inline(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])

    detail = client.get(f"/api/parties/{party_id}")
    assert detail.json()["messages"][0]["variants"] == []

    _stub_chat_stream(monkeypatch, replies=["Autre version."])
    assert _regenerate(client, party_id, message_id).status_code == 200

    messages = client.get(f"/api/parties/{party_id}").json()["messages"]
    variants = messages[0]["variants"]
    assert [(v["content"], v["active"]) for v in variants] == [
        (OPENING, False),
        ("Autre version.", True),
    ]
    assert messages[0]["content"] == "Autre version."
    # Later messages keep their empty archive.
    assert len(messages) == 1


def test_edit_changes_the_text_and_keeps_the_id(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])

    response = _edit(client, party_id, message_id, "  La porte est condamnée.  ")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == message_id
    assert body["content"] == "La porte est condamnée."
    assert _message_content(message_id) == "La porte est condamnée."
    _assert_invariant(message_id)


def test_edit_of_a_message_with_variants_writes_the_active_row(
    client: TestClient, monkeypatch: Any
) -> None:
    """Issue #11's second criterion: the edit must reach the active variant
    row too, or switching away and back resurrects the replaced text."""
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    _stub_chat_stream(monkeypatch, replies=["Première variante."])
    assert _regenerate(client, party_id, message_id).status_code == 200
    first_variant_id = _variant_rows(message_id)[0]["id"]
    assert _switch_variant(client, party_id, message_id, first_variant_id).status_code == 200

    assert _edit(client, party_id, message_id, "Texte corrigé.").status_code == 200
    _assert_invariant(message_id)

    # Away, then back: the corrected text is still there.
    assert _switch_variant(client, party_id, message_id, _variant_rows(message_id)[1]["id"])
    assert _switch_variant(client, party_id, message_id, first_variant_id).status_code == 200
    assert _message_content(message_id) == "Texte corrigé."
    assert _variant_rows(message_id)[0]["content"] == "Texte corrigé."


def test_edit_keeps_the_message_id_so_the_summary_frontier_is_untouched(
    client: TestClient, monkeypatch: Any
) -> None:
    """`instance.summary_upto` is a message id — the rolling summary's
    frontier — and `message.id` is the order of the story. The edit is an
    UPDATE: recreating the row would move both."""
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    with db.connect() as con:
        con.execute("UPDATE instance SET summary_upto = ? WHERE id = ?", (message_id, party_id))

    assert _edit(client, party_id, message_id, "Texte corrigé.").status_code == 200

    with db.connect() as con:
        row = con.execute("SELECT summary_upto FROM instance WHERE id = ?", (party_id,)).fetchone()
        messages = con.execute(
            "SELECT id, content FROM message WHERE instance_id = ?", (party_id,)
        ).fetchall()
    assert [m["id"] for m in messages] == [message_id]
    assert messages[0]["content"] == "Texte corrigé."
    assert row["summary_upto"] == message_id


def test_edit_does_not_bump_updated_at(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = 1000.0 WHERE id = ?", (party_id,))

    opening_id = _opening_message_id(client, party_id)
    assert _edit(client, party_id, opening_id, "Corrigé.").status_code == 200

    assert client.get(f"/api/parties/{party_id}").json()["updated_at"] == 1000.0


def test_blank_edit_is_400(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    message_id = _opening_message_id(client, party["id"])

    for blank in ("", "   \n\t "):
        response = _edit(client, party["id"], message_id, blank)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    assert _message_content(message_id) == OPENING


def test_edit_of_an_image_message_is_400(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    image_id = _insert_image_message(party["id"])

    response = _edit(client, party["id"], image_id, "Une légende.")

    assert response.status_code == 400
    assert "text" in response.json()["detail"]


def test_regenerate_replaces_the_text_and_archives_the_old_reply(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    _stub_chat_stream(monkeypatch, replies=["Le ", "quai ", "s'éveille."])

    response = _regenerate(client, party_id, message_id)

    assert response.status_code == 200
    events = _ndjson_events(response.text)
    assert [e["delta"] for e in events[:-1]] == ["Le ", "quai ", "s'éveille."]
    assert events[-1] == {"done": True, "message_id": message_id}
    assert _message_content(message_id) == "Le quai s'éveille."
    variants = _variant_rows(message_id)
    assert [(v["content"], v["active"]) for v in variants] == [
        (OPENING, False),
        ("Le quai s'éveille.", True),
    ]
    _assert_invariant(message_id)


def test_regenerated_message_is_absent_from_the_prompt(
    client: TestClient, monkeypatch: Any
) -> None:
    """The reply being regenerated must not ride in the history: with it, the
    model reads its own text and continues instead of writing an
    alternative."""
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    opening_id = _opening_message_id(client, party["id"])
    _stub_chat_stream(monkeypatch, replies=["Réponse 1."])
    assert _send_turn(client, party_id, "Tour 1.").status_code == 200
    reply = next(m for m in _party_message_rows(party_id) if m["content"] == "Réponse 1.")
    calls = _stub_chat_stream(monkeypatch, replies=["Réponse 2."])

    assert _regenerate(client, party_id, reply["id"]).status_code == 200

    messages = calls[0]["messages"]
    assert {"role": "assistant", "content": "Réponse 1."} not in messages
    # Everything before it is still there: the opening, then the player's turn.
    assert {"role": "assistant", "content": OPENING} in messages
    assert {"role": "user", "content": "Tour 1."} in messages
    assert opening_id < reply["id"]


def test_empty_stream_leaves_everything_untouched(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = 1000.0 WHERE id = ?", (party_id,))

    for replies in ([], ["  ", "\n\t"]):
        _stub_chat_stream(monkeypatch, replies=replies)
        response = _regenerate(client, party_id, message_id)
        assert response.status_code == 200
        # Whitespace fragments ride the stream; only the database write is skipped.
        events = _ndjson_events(response.text)
        assert [e["delta"] for e in events[:-1]] == replies
        assert events[-1] == {"done": True, "message_id": None}
        assert _variant_rows(message_id) == []
        assert _message_content(message_id) == OPENING
        _assert_invariant(message_id)

    # The party looks exactly as it did — `updated_at` included.
    assert client.get(f"/api/parties/{party_id}").json()["updated_at"] == 1000.0


def test_non_empty_partial_from_a_broken_stream_becomes_the_active_variant(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    _stub_chat_stream(monkeypatch, replies=["Le navire "], error=OllamaError("stream died mid-way"))

    response = _regenerate(client, party_id, message_id)

    assert response.status_code == 200
    events = _ndjson_events(response.text)
    assert events[0] == {"delta": "Le navire "}
    assert events[1] == {"error": "stream died mid-way"}
    assert events[2]["done"] is True
    assert _message_content(message_id) == "Le navire "
    variants = _variant_rows(message_id)
    assert [(v["content"], v["active"]) for v in variants] == [
        (OPENING, False),
        ("Le navire ", True),
    ]
    _assert_invariant(message_id)


def test_three_regenerations_archive_every_previous_reply(
    client: TestClient, monkeypatch: Any
) -> None:
    """The opening plus each of the three alternatives: four rows, of which
    three are the archive. The first reply is still reachable and unchanged."""
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    for reply in ("Variante 1.", "Variante 2.", "Variante 3."):
        _stub_chat_stream(monkeypatch, replies=[reply])
        assert _regenerate(client, party_id, message_id).status_code == 200

    variants = _variant_rows(message_id)
    assert [v["content"] for v in variants] == [
        OPENING,
        "Variante 1.",
        "Variante 2.",
        "Variante 3.",
    ]
    assert [v["active"] for v in variants] == [False, False, False, True]
    assert _message_content(message_id) == "Variante 3."
    _assert_invariant(message_id)

    # The first reply is still reachable.
    assert _switch_variant(client, party_id, message_id, variants[0]["id"]).status_code == 200
    assert _message_content(message_id) == OPENING
    _assert_invariant(message_id)


def test_regenerate_bumps_updated_at(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    with db.connect() as con:
        con.execute("UPDATE instance SET updated_at = 1000.0 WHERE id = ?", (party_id,))
    _stub_chat_stream(monkeypatch, replies=["Nouvelle version."])

    assert _regenerate(client, party_id, message_id).status_code == 200

    assert client.get(f"/api/parties/{party_id}").json()["updated_at"] > 1000.0


def test_regenerate_of_a_player_turn_is_400(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    _stub_chat_stream(monkeypatch, replies=["Réponse."])
    assert _send_turn(client, party_id, "Tour du joueur.").status_code == 200
    user_id = next(m["id"] for m in _party_message_rows(party_id) if m["role"] == "user")

    response = _regenerate(client, party_id, user_id)

    assert response.status_code == 400
    assert "narrator" in response.json()["detail"]


def test_regenerate_of_an_image_message_is_400(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    image_id = _insert_image_message(party["id"])

    response = _regenerate(client, party["id"], image_id)

    assert response.status_code == 400
    assert "narrator" in response.json()["detail"]


def test_switch_variant_activates_that_variant(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    _stub_chat_stream(monkeypatch, replies=["Variante."])
    assert _regenerate(client, party_id, message_id).status_code == 200
    first_variant_id = _variant_rows(message_id)[0]["id"]

    response = _switch_variant(client, party_id, message_id, first_variant_id)

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == OPENING
    assert [(v["content"], v["active"]) for v in body["variants"]] == [
        (OPENING, True),
        ("Variante.", False),
    ]
    assert _message_content(message_id) == OPENING
    _assert_invariant(message_id)
    # Switching navigates the archive: it grows it by nothing.
    assert len(_variant_rows(message_id)) == 2


def test_switch_to_an_unknown_variant_is_404(client: TestClient, monkeypatch: Any) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])

    response = _switch_variant(client, party_id, message_id, 999999)

    assert response.status_code == 404
    assert _message_content(message_id) == OPENING


def test_edit_regenerate_and_switch_on_unknown_targets_are_404(
    client: TestClient, monkeypatch: Any
) -> None:
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    other_party_id = party["id"]
    other_message_id = _opening_message_id(client, party["id"])
    other_variant = _variant_rows(other_message_id)
    assert other_variant == []
    stranger = _create_party(client, scenario_id, monkeypatch)
    stranger_message_id = _opening_message_id(client, stranger["id"])
    _stub_chat_stream(monkeypatch, replies=["Variante."])
    assert _regenerate(client, stranger["id"], stranger_message_id).status_code == 200
    stranger_variant_id = _variant_rows(stranger_message_id)[0]["id"]
    missing = uuid.uuid4().hex
    missing_message = 999999

    # Unknown party.
    assert _edit(client, missing, missing_message, "x").status_code == 404
    assert _regenerate(client, missing, missing_message).status_code == 404
    assert _switch_variant(client, missing, missing_message, 1).status_code == 404
    # Unknown message in a real party.
    assert _edit(client, other_party_id, missing_message, "x").status_code == 404
    assert _regenerate(client, other_party_id, missing_message).status_code == 404
    assert _switch_variant(client, other_party_id, missing_message, 1).status_code == 404
    # A message, and a variant, belonging to another party.
    assert _edit(client, other_party_id, stranger_message_id, "x").status_code == 404
    assert _regenerate(client, other_party_id, stranger_message_id).status_code == 404
    assert (
        _switch_variant(client, other_party_id, other_message_id, stranger_variant_id).status_code
        == 404
    )


def test_editing_and_regenerating_run_the_shared_stream_contract(
    client: TestClient, monkeypatch: Any
) -> None:
    """A regeneration sends the assembled system prompt and the configured
    num_ctx, exactly as a played turn does — one machinery, two callers."""
    scenario_id = _create_scenario(client, "La Cité Noyée")
    _configure_model("test-model", num_ctx=4096)
    party = _create_party(client, scenario_id, monkeypatch)
    message_id = _opening_message_id(client, party["id"])
    calls = _stub_chat_stream(monkeypatch, replies=["Nouvelle ouverture."])

    assert _regenerate(client, party["id"], message_id).status_code == 200

    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["num_ctx"] == 4096
    assert "Scenario: La Cité Noyée" in calls[0]["messages"][0]["content"]


# --- The archive race ------------------------------------------------------------
#
# The archive branch of `_set_message_text` is a read-check-write: it saves the
# text it read from `message.content` moments earlier. These tests open that
# window on purpose and drive a writer through it, so the interleaving is
# deterministic instead of left to the scheduler — a threaded test that loses
# the edit 7 times out of 30 proves nothing. The window sits immediately after
# the `SELECT content FROM message` read, which is exactly where another
# writer's commit used to be archived over.

_WAIT_TIMEOUT = 5.0
# Long enough for a local edit transaction to commit many times over, short
# enough that the fix's blocked writer never approaches its 5 s busy_timeout.
_EDIT_WINDOW = 0.5
# When the write lock is held, the second regeneration cannot reach the window
# at all; the gate gives up waiting for it after this long.
_GATE_TIMEOUT = 2.0

_RACE_SQL = "SELECT content FROM message"

_WINDOW_GATE: _ArchiveGate | None = None


class _ArchiveGate:
    """A synchronisation point parked inside the archive window.

    Threads arriving through `_WindowConnection` park here until the gate lets
    them: with `expected=1` the test closes the window itself through
    `window_close`, with `expected=2` the two racing writers are released
    together. Every wait is bounded, so a broken run ends instead of hanging
    the suite.
    """

    def __init__(self, expected: int) -> None:
        self._expected = expected
        self._arrivals = 0
        self._lock = threading.Lock()
        self._all_arrived = threading.Event()
        self.window_open = threading.Event()
        self.window_close = threading.Event()

    def arrive(self) -> None:
        global _WINDOW_GATE
        with self._lock:
            self._arrivals += 1
            arrived_last = self._arrivals >= self._expected
        if self._expected == 1:
            self.window_open.set()
            self.window_close.wait(_WAIT_TIMEOUT)
        else:
            if arrived_last:
                self._all_arrived.set()
            self._all_arrived.wait(_GATE_TIMEOUT)
        # One-shot: once released, later statements and later connections must
        # run freely, or the test's own verification reads would park too.
        with self._lock:
            _WINDOW_GATE = None


class _WindowConnection(sqlite3.Connection):
    """Connection that parks in the archive window when a gate is armed.

    The park lands after `SELECT content FROM message` has returned — the
    moment the archive branch holds a text another writer may already have
    replaced.
    """

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        cursor = super().execute(sql, parameters)
        gate = _WINDOW_GATE
        if gate is not None and sql.lstrip().startswith(_RACE_SQL):
            gate.arrive()
        return cursor


@contextmanager
def _traced_connect(**kwargs: Any) -> Iterator[sqlite3.Connection]:
    """`db.connect` with `_WindowConnection` as the connection factory.

    Mirrors `db.connect`'s body — including the `immediate` transaction mode —
    because the real signature carries no factory parameter to inject.
    """
    con = sqlite3.connect(db.DB_PATH, factory=_WindowConnection)
    try:
        db._configure(con)
        if kwargs.get("immediate"):
            con.isolation_level = None
            con.execute("BEGIN IMMEDIATE")
        with con:
            yield con
    finally:
        con.close()


def _run_in_thread(fn: Callable[..., Any], *args: Any) -> tuple[threading.Thread, list[Exception]]:
    """Run `fn` on a thread, collecting anything it raises for the test to assert."""
    errors: list[Exception] = []

    def target() -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - the test asserts on what happened
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    return thread, errors


def test_edit_committing_in_the_archive_window_is_not_lost(
    client: TestClient, monkeypatch: Any
) -> None:
    """An edit committing between the archive read and the archive INSERT used
    to be lost: the regeneration archived the pre-edit text it had read and
    its final `UPDATE message` then overwrote the edit, leaving the edited
    text nowhere. The window below opens right after that read and the edit
    commits inside it — deterministically.

    With the write lock held from before the read, the edit can no longer land
    inside the window: it waits for the regeneration to commit and applies
    after it, so the edited text survives as the message's active text.
    """
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])
    edited = "La porte est condamnée."
    regenerated = "Le quai s'éveille, différent."

    gate = _ArchiveGate(expected=1)
    monkeypatch.setattr(sys.modules[__name__], "_WINDOW_GATE", gate)
    monkeypatch.setattr(db, "connect", _traced_connect)

    persist_thread, persist_errors = _run_in_thread(
        parties_routes._persist_regenerated, message_id, party_id, regenerated
    )
    assert gate.window_open.wait(_WAIT_TIMEOUT), "the archive read never reached its window"

    edit_thread, edit_errors = _run_in_thread(
        parties_routes.edit_message,
        party_id,
        message_id,
        parties_routes.MessageEditInput(content=edited),
    )
    # Under the bug the edit commits inside the window and this join returns
    # at once; under the fix the edit is blocked by the write lock the
    # regeneration already holds, so the join times out. Either way the window
    # then closes and the outcome is what the assertions below judge.
    edit_thread.join(_EDIT_WINDOW)
    gate.window_close.set()

    persist_thread.join(_WAIT_TIMEOUT)
    assert not persist_thread.is_alive(), "the regeneration never completed"
    assert persist_errors == []

    edit_thread.join(_WAIT_TIMEOUT)
    assert not edit_thread.is_alive(), "the edit never completed"
    assert edit_errors == []

    variants = _variant_rows(message_id)
    survived = _message_content(message_id) == edited or any(
        v["content"] == edited for v in variants
    )
    assert survived, (
        "the edit was lost: "
        f"message.content={_message_content(message_id)!r}, "
        f"variants={[v['content'] for v in variants]!r}"
    )
    _assert_invariant(message_id)


def test_two_regenerations_racing_the_window_archive_the_original_once(
    client: TestClient, monkeypatch: Any
) -> None:
    """Two regenerations that both read `already_archived = False` used to
    archive the same text twice — a duplicate entry in the navigator. Both
    threads here read inside the same window, deterministically.

    With the write lock held from before the read, the second regeneration
    only runs once the first has committed and finds the archive already
    there: the original is archived exactly once.
    """
    scenario_id = _create_scenario(client)
    party = _create_party(client, scenario_id, monkeypatch)
    party_id = party["id"]
    message_id = _opening_message_id(client, party["id"])

    gate = _ArchiveGate(expected=2)
    monkeypatch.setattr(sys.modules[__name__], "_WINDOW_GATE", gate)
    monkeypatch.setattr(db, "connect", _traced_connect)

    first, first_errors = _run_in_thread(
        parties_routes._persist_regenerated, message_id, party_id, "Nouvelle un."
    )
    second, second_errors = _run_in_thread(
        parties_routes._persist_regenerated, message_id, party_id, "Nouvelle deux."
    )
    first.join(_WAIT_TIMEOUT + _GATE_TIMEOUT)
    second.join(_WAIT_TIMEOUT + _GATE_TIMEOUT)
    assert not first.is_alive() and not second.is_alive(), "a regeneration never completed"
    assert first_errors == [] and second_errors == []

    originals = [v for v in _variant_rows(message_id) if v["content"] == OPENING]
    assert len(originals) == 1, f"the original reply was archived {len(originals)} times"
    _assert_invariant(message_id)


# --- The rolling summary (issue #14) ---------------------------------------------
#
# The summariser runs on a background thread. Every wait below is bounded, so
# a failing run ends instead of hanging the suite, and every test waits for
# the job it started to leave `summarizer._locks` before returning — a daemon
# thread that outlives its stubs would reach the real Ollama.


def _set_window(window: int) -> None:
    with db.connect() as con:
        settings.set_history_window(con, window)


def _fill_party(party_id: str, count: int, marker: str) -> list[int]:
    """Insert `count` text messages past the opening; return all message ids."""
    with db.connect() as con:
        existing = con.execute(
            "SELECT id FROM message WHERE instance_id = ? ORDER BY id", (party_id,)
        ).fetchall()
        ids = [row["id"] for row in existing]
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            cursor = con.execute(
                "INSERT INTO message (instance_id, role, kind, content, ts) "
                "VALUES (?, ?, 'text', ?, ?)",
                (party_id, role, f"{marker} {i}.", 0.0),
            )
            assert cursor.lastrowid is not None
            ids.append(cursor.lastrowid)
    return ids


def _set_summary(party_id: str, text: str, upto: int | None, world_state: str) -> None:
    with db.connect() as con:
        con.execute(
            "UPDATE instance SET summary_text = ?, summary_upto = ?, world_state = ? WHERE id = ?",
            (text, upto, world_state, party_id),
        )


def _summary_state(party_id: str) -> tuple[str, int | None, str]:
    """(summary_text, summary_upto, world_state) as stored, raw."""
    with db.connect() as con:
        row = con.execute(
            "SELECT summary_text, summary_upto, world_state FROM instance WHERE id = ?",
            (party_id,),
        ).fetchone()
    assert row is not None
    return row["summary_text"], row["summary_upto"], row["world_state"]


def _summary_reply(summary: str, world_state: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"summary": summary}
    if world_state is not None:
        payload["world_state"] = world_state
    return json.dumps(payload, ensure_ascii=False)


def _stub_summary_chat(
    monkeypatch: Any,
    reply: str | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Replace `ollama.chat` with a stub that recognises the summariser's call.

    The summariser is the only caller that passes `format="json"`; the stub
    records every call so a test can count them.
    """
    calls: list[dict[str, Any]] = []

    def fake_chat(
        model: str, messages: list[dict[str, str]], num_ctx: int, *, format: str | None = None
    ) -> str:
        calls.append({"model": model, "messages": messages, "num_ctx": num_ctx, "format": format})
        if error is not None:
            raise error
        assert reply is not None
        return reply

    monkeypatch.setattr(parties_routes.ollama, "chat", fake_chat)
    return calls


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _wait_unlocked(party_id: str, timeout: float = 5.0) -> bool:
    """Wait until no summarisation for this party holds its lock any more.

    Callers must first establish that a job started — a stub call recorded,
    the party seen in `_locks`, or its summary already committed. The lock is
    taken in `schedule` and given back, swept out of the map, when the job
    ends; a job fast enough to finish between two polls is never seen holding
    it, so absence is the only reliable end signal here.
    """
    return _wait_until(lambda: party_id not in summarizer._locks, timeout)


def test_summarisation_never_blocks_the_turn(client: TestClient, monkeypatch: Any) -> None:
    """Issue #14's first criterion, observed from both sides: the summariser
    blocks on an event inside its model call, and the turn's response still
    completes fully — every NDJSON line included — before the event is ever
    set. A scheduler that ran the model call inline would deadlock here."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _fill_party(party_id, 3, "Ancien")
    _set_window(2)
    entered = threading.Event()
    release = threading.Event()

    def blocking_chat(
        model: str, messages: list[dict[str, str]], num_ctx: int, *, format: str | None = None
    ) -> str:
        entered.set()
        release.wait(5.0)
        return _summary_reply("Résumé bloquant.", {"lieu": "port"})

    monkeypatch.setattr(parties_routes.ollama, "chat", blocking_chat)
    _stub_chat_stream(monkeypatch, replies=["Réponse."])

    response = _send_turn(client, party_id, "Tour 1.")

    assert response.status_code == 200
    events = _ndjson_events(response.text)
    assert events[0] == {"delta": "Réponse."}
    assert events[-1]["done"] is True
    # The summariser is now parked inside its model call — the whole response
    # arrived while it was still in there, and the test has not released it.
    # Nothing was committed either: a scheduler that ran the model call on the
    # request path would have come back from it (event timeout) before the
    # response could finish.
    assert entered.wait(5.0)
    assert not release.is_set()
    assert _summary_state(party_id)[0] == ""

    release.set()
    assert _wait_unlocked(party_id)
    text, upto, _ = _summary_state(party_id)
    assert text == "Résumé bloquant."
    assert upto is not None


def test_second_trigger_while_one_runs_is_skipped(client: TestClient, monkeypatch: Any) -> None:
    """Issue #14's second criterion, one party at a time: a trigger that
    arrives while a summarisation for the same party is running is skipped,
    not queued — the condition stays true, so the next turn re-triggers."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _fill_party(party_id, 3, "Ancien")
    _set_window(2)
    release = threading.Event()
    calls: list[dict[str, Any]] = []

    def blocking_chat(
        model: str, messages: list[dict[str, str]], num_ctx: int, *, format: str | None = None
    ) -> str:
        calls.append({"model": model, "messages": messages, "num_ctx": num_ctx, "format": format})
        release.wait(5.0)
        return _summary_reply("Résumé unique.")

    monkeypatch.setattr(parties_routes.ollama, "chat", blocking_chat)

    summarizer.schedule(party_id)
    assert _wait_until(lambda: len(calls) == 1)
    # The first job is still inside its model call: the second trigger must
    # not start a second one.
    summarizer.schedule(party_id)
    assert len(calls) == 1

    release.set()
    assert _wait_unlocked(party_id)


def test_two_parties_do_not_block_each_other(client: TestClient, monkeypatch: Any) -> None:
    """The lock is per party: a blocked summarisation for one party leaves
    another party's summarisation free to run and commit."""
    scenario_id = _create_scenario(client)
    party_a = _create_party(client, scenario_id, monkeypatch)["id"]
    party_b = _create_party(client, scenario_id, monkeypatch)["id"]
    _fill_party(party_a, 3, "PA")
    _fill_party(party_b, 3, "PB")
    _set_window(2)
    release_a = threading.Event()

    def chat_by_party(
        model: str, messages: list[dict[str, str]], num_ctx: int, *, format: str | None = None
    ) -> str:
        if "PA" in json.dumps(messages):
            release_a.wait(5.0)
            return _summary_reply("Résumé A.")
        return _summary_reply("Résumé B.")

    monkeypatch.setattr(parties_routes.ollama, "chat", chat_by_party)

    summarizer.schedule(party_a)
    assert _wait_until(lambda: party_a in summarizer._locks)
    summarizer.schedule(party_b)

    # B committed while A was still blocked; A commits once released.
    assert _wait_until(lambda: _summary_state(party_b)[0] == "Résumé B.")
    assert _wait_unlocked(party_b)
    assert _summary_state(party_a)[0] == ""
    release_a.set()
    assert _wait_unlocked(party_a)
    assert _summary_state(party_a)[0] == "Résumé A."


def test_failed_summarisation_keeps_the_previous_summary_then_retries(
    client: TestClient, monkeypatch: Any
) -> None:
    """Issue #14's third criterion, both halves: an unreachable Ollama leaves
    `summary_text`, `summary_upto` and `world_state` exactly as they were,
    and the next turn — no retry state, the trigger is still true — updates
    them."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _fill_party(party_id, 3, "Ancien")
    _set_window(2)
    _set_summary(party_id, "Ancien résumé.", None, '{"lieu": "port"}')

    failed = threading.Event()

    def failing_chat(
        model: str, messages: list[dict[str, str]], num_ctx: int, *, format: str | None = None
    ) -> str:
        failed.set()
        raise OllamaUnreachable(
            "Ollama is unreachable at http://localhost:11434: connection refused"
        )

    monkeypatch.setattr(parties_routes.ollama, "chat", failing_chat)
    summarizer.schedule(party_id)
    assert failed.wait(5.0)
    assert _wait_until(lambda: party_id not in summarizer._locks)
    assert _summary_state(party_id) == ("Ancien résumé.", None, '{"lieu": "port"}')

    # The automatic retry: the next turn's reply re-triggers the job, which
    # now answers — without a `world_state`, so the previous one is kept.
    _stub_chat_stream(monkeypatch, replies=["Réponse."])
    _stub_summary_chat(monkeypatch, reply=_summary_reply("Résumé après échec."))
    _send_turn(client, party_id, "Tour suivant.")

    assert _wait_until(lambda: _summary_state(party_id)[0] == "Résumé après échec.")
    text, upto, world_raw = _summary_state(party_id)
    assert text == "Résumé après échec."
    assert upto is not None
    assert json.loads(world_raw) == {"lieu": "port"}


def test_unparseable_summaries_change_nothing(client: TestClient, monkeypatch: Any) -> None:
    """A local model will answer wrong sometimes, and each wrong shape is a
    job failure, not a crash: not JSON, not an object, or no usable summary —
    all leave the stored state exactly as it was."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _fill_party(party_id, 3, "Ancien")
    _set_window(2)
    _set_summary(party_id, "Ancien résumé.", None, '{"lieu": "port"}')

    unparseable = [
        "pas du tout du JSON",
        '["un", "tableau"]',
        json.dumps({"world_state": {"lieu": "port"}}),
        json.dumps({"summary": "   "}),
        json.dumps({"summary": "Résumé.", "world_state": "pas un objet"}),
    ]
    for expected_reply in unparseable:
        calls = _stub_summary_chat(monkeypatch, reply=expected_reply)
        summarizer.schedule(party_id)
        assert _wait_until(lambda calls=calls: len(calls) == 1)
        assert _wait_until(lambda: party_id not in summarizer._locks)
        assert _summary_state(party_id) == ("Ancien résumé.", None, '{"lieu": "port"}')
        calls.clear()


def test_frontier_advances_and_the_prompt_carries_the_summary(
    client: TestClient, monkeypatch: Any
) -> None:
    """The frontier lands on the last compressed message's id, and the next
    turn's prompt carries the summary while the covered messages are gone —
    asserted on the messages handed to Ollama, not only on the database."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    ids = _fill_party(party_id, 3, "Ancien")
    opening = _message_content(ids[0])
    _set_window(2)
    calls = _stub_summary_chat(
        monkeypatch, reply=_summary_reply("Résumé de la digue.", {"lieu": "quai"})
    )

    summarizer.schedule(party_id)

    assert _wait_until(lambda: len(calls) == 1)
    assert _wait_unlocked(party_id)
    assert calls[0]["format"] == "json"
    # Four uncovered messages, window 2: the two oldest are compressed, and
    # the frontier is the second message's id.
    text, upto, world_raw = _summary_state(party_id)
    assert text == "Résumé de la digue."
    assert upto == ids[1]
    assert json.loads(world_raw) == {"lieu": "quai"}
    # The call itself carried the compressed messages and the prior memory,
    # ending on the user instruction that asks for the answer.
    sent = calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Résumé de la digue." not in sent[0]["content"]  # the old summary was empty
    assert sent[-1] == {
        "role": "user",
        "content": "Now write the JSON object summarizing the messages above.",
    }
    compressed = [m["content"] for m in sent[1:-1]]
    assert opening in compressed
    assert "Ancien 0." in compressed
    assert not any("Ancien 1." in c or "Ancien 2." in c for c in compressed)

    # The next turn: the summary rides in the system prompt, the covered
    # messages no longer ride in the history. The turn crosses the window
    # again, so stub the summariser with a failure and wait for that job too.
    second_calls = _stub_summary_chat(monkeypatch, error=OllamaError("not under test"))
    stream_calls = _stub_chat_stream(monkeypatch, replies=["Réponse."])
    _send_turn(client, party_id, "Tour après résumé.")
    messages = stream_calls[0]["messages"]
    assert "Résumé de la digue." in messages[0]["content"]
    history_contents = [m["content"] for m in messages[1:]]
    assert history_contents == ["Ancien 2.", "Tour après résumé."]
    assert opening not in history_contents
    assert "Ancien 1." not in history_contents
    assert _wait_until(lambda: len(second_calls) == 1)
    assert _wait_unlocked(party_id)


def test_party_under_the_trigger_never_calls_the_summariser(
    client: TestClient, monkeypatch: Any
) -> None:
    """Under the trigger, nothing runs at all: a short party's turns never
    reach the summariser's model call. The schedule runs before the response's
    final line, so a zero here is not a race."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _set_window(settings.DEFAULT_HISTORY_WINDOW)
    calls = _stub_summary_chat(monkeypatch, reply=_summary_reply("Résumé."))
    _stub_chat_stream(monkeypatch, replies=["Réponse."])

    response = _send_turn(client, party_id, "Premier tour.")

    assert response.status_code == 200
    assert _ndjson_events(response.text)[-1]["done"] is True
    assert calls == []
    assert party_id not in summarizer._locks


def test_get_party_serves_the_summary_and_the_world_state(
    client: TestClient, monkeypatch: Any
) -> None:
    """Issue #14's fourth criterion, read-only: the summary, its frontier and
    the world state are visible on the party the page loads."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    ids = _fill_party(party_id, 1, "Ancien")
    _set_summary(party_id, "Le héros a trouvé la clé.", ids[0], '{"lieu": "tour"}')

    body = client.get(f"/api/parties/{party_id}").json()
    assert body["summary_text"] == "Le héros a trouvé la clé."
    assert body["summary_upto"] == ids[0]
    assert body["world_state"] == {"lieu": "tour"}

    # A fresh party starts empty, in response shape and not raw JSON.
    fresh = _create_party(client, scenario_id, monkeypatch)["id"]
    fresh_body = client.get(f"/api/parties/{fresh}").json()
    assert fresh_body["summary_text"] == ""
    assert fresh_body["summary_upto"] is None
    assert fresh_body["world_state"] == {}


def test_a_result_behind_the_stored_frontier_is_discarded(
    client: TestClient, monkeypatch: Any
) -> None:
    """A job that read before another one committed must not move the frontier
    backwards: its result is discarded inside the IMMEDIATE transaction that
    re-reads `summary_upto`."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    ids = _fill_party(party_id, 2, "Ancien")
    _set_summary(party_id, "Résumé en place.", ids[1], '{"lieu": "port"}')

    # A stale job computed a frontier below the stored one: discarded.
    applied = summarizer._commit(
        party_id,
        frontier=ids[0],
        summary_text="Résumé périmé.",
        world_state=None,
        previous_world_state={"lieu": "port"},
    )
    assert applied is False
    assert _summary_state(party_id) == ("Résumé en place.", ids[1], '{"lieu": "port"}')

    # A frontier at or past the stored one commits.
    applied = summarizer._commit(
        party_id,
        frontier=ids[1],
        summary_text="Résumé à jour.",
        world_state={"lieu": "quai"},
        previous_world_state={"lieu": "port"},
    )
    assert applied is True
    text, upto, world_raw = _summary_state(party_id)
    assert text == "Résumé à jour."
    assert upto == ids[1]
    assert json.loads(world_raw) == {"lieu": "quai"}


def test_the_summary_parser_tolerates_no_shape_but_its_own() -> None:
    assert summarizer._parse_reply("pas du tout du JSON") is None
    assert summarizer._parse_reply('["un", "tableau"]') is None
    assert summarizer._parse_reply('{"world_state": {}}') is None
    assert summarizer._parse_reply('{"summary": "   "}') is None
    assert summarizer._parse_reply('{"summary": "s", "world_state": "pas un objet"}') is None
    # A missing world_state is not a failure: the previous one is kept.
    assert summarizer._parse_reply('{"summary": "s"}') == ("s", None)
    assert summarizer._parse_reply('{"summary": "s", "world_state": {}}') == ("s", {})


def test_schedule_without_a_party_a_model_or_a_readable_world_state_does_nothing(
    client: TestClient, monkeypatch: Any
) -> None:
    """An unknown party, a missing model and a hand-corrupted `world_state`
    are all quiet no-ops, never errors."""
    calls = _stub_summary_chat(monkeypatch, reply=_summary_reply("Résumé."))
    summarizer.schedule("no-such-party")
    assert calls == []

    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    _fill_party(party_id, 3, "Ancien")
    _set_window(2)
    _configure_model(None)
    summarizer.schedule(party_id)
    assert calls == []
    assert party_id not in summarizer._locks
    assert _summary_state(party_id) == ("", None, "{}")

    _configure_model("test-model")
    assert summarizer.parse_world_state("pas du JSON") == {}
    assert summarizer.parse_world_state('["un tableau"]') == {}


def test_a_commit_for_a_deleted_party_is_a_quiet_no(client: TestClient, monkeypatch: Any) -> None:
    """A party deleted while its summary job ran leaves nothing to update."""
    scenario_id = _create_scenario(client)
    party_id = _create_party(client, scenario_id, monkeypatch)["id"]
    ids = _fill_party(party_id, 1, "Ancien")
    client.delete(f"/api/parties/{party_id}")

    applied = summarizer._commit(
        party_id,
        frontier=ids[0],
        summary_text="Résumé orphelin.",
        world_state={"lieu": "port"},
        previous_world_state={},
    )

    assert applied is False
