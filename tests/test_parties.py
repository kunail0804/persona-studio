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
import threading
import uuid
from collections.abc import Iterator
from typing import Any

import anyio
from fastapi.testclient import TestClient

from persona_studio import db, settings
from persona_studio.narrator import OPENING_INSTRUCTION
from persona_studio.ollama import OllamaError, OllamaUnreachable
from persona_studio.routes import parties as parties_routes

OPENING = "The harbour gate looms ahead, barnacled and half open. What do you do?"


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

    _send_turn(client, party_id, "Tour 3.")

    messages = calls[0]["messages"]
    assert len(messages) == 3  # the system prompt, plus the last two only
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "Réponse 2."}
    assert messages[2] == {"role": "user", "content": "Tour 3."}


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
