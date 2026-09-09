"""Tests for image prompt composition.

Two layers, like the feature:

- the pure layer (`image_prompt.build_messages`, `scrub_names`) is tested
  with inputs built in memory — no database, no client, no HTTP;
- the route is tested through `TestClient` with `ollama.chat` stubbed where
  the route looks it up (the same technique as `test_parties.py`), so no
  test ever reaches a real Ollama.

The name pass is issue #16's first criterion, so its traps each get a test:
word boundaries (`Ana` must not touch `banana`), case-insensitivity, bare
names removed with their keyword, and the persona and scenario title being
scrubbed too.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, settings
from persona_studio.image_prompt import Appearance, ImagePromptInputs, build_messages, scrub_names
from persona_studio.ollama import OllamaError, OllamaUnreachable
from persona_studio.routes import images as images_routes


@pytest.fixture
def connection(client: TestClient) -> Iterator[sqlite3.Connection]:
    """A connection to the migrated test database; `client` ran the migration."""
    with db.connect() as con:
        yield con


# --- Shared inputs -------------------------------------------------------------


ELENA = Appearance(name="Elena", appearance="Red hair, green eyes, a scar across one brow.")
MARCO = Appearance(name="Marco", appearance="Tall, greying beard, navy coat.")
ANA = Appearance(name="Ana", appearance="Small, dark-haired, quick-eyed.")
ASH = Appearance(name="Ash", appearance="Grey coat, always damp.")
# A character the scenario never gave an appearance to: a name with nothing
# to become, the case the bare-name removal exists for.
BARE_MARCO = Appearance(name="Marco", appearance="")

INPUTS = ImagePromptInputs(
    instruction="Elena sur le quai au crépuscule, une lanterne bleue",
    narration="Elena attend près de la porte du port, la nuit tombée.",
    world_state={
        "location": "le port",
        "present": ["Elena"],
        "established": ["la porte est ouverte"],
    },
    persona=ASH,
    characters=(ELENA, MARCO),
    scenario_title="La Cité Noyée",
)


# --- The deterministic name pass (issue #16, criterion 1) -----------------------


def test_a_named_character_becomes_their_appearance_description() -> None:
    scrubbed = scrub_names("portrait of Elena at dusk, smiling", INPUTS)
    assert "Elena" not in scrubbed
    assert "Red hair, green eyes, a scar across one brow." in scrubbed
    assert "at dusk" in scrubbed
    assert "smiling" in scrubbed


def test_word_boundaries_ana_never_touches_banana() -> None:
    inputs = replace_instruction(INPUTS, characters=(ELENA, MARCO, ANA))
    assert "banana" in scrub_names("banana, Ana dancing", inputs)
    assert "Small, dark-haired, quick-eyed." in scrub_names("banana, Ana dancing", inputs)


def test_case_insensitive_matching_catches_any_casing() -> None:
    for casing in ("ELENA", "elena", "eLeNa"):
        scrubbed = scrub_names(f"{casing} on the quay", INPUTS)
        assert casing not in scrubbed
        assert "Red hair, green eyes" in scrubbed


def test_a_name_without_appearance_is_removed_with_its_keyword() -> None:
    inputs = replace_instruction(INPUTS, characters=(BARE_MARCO,))
    scrubbed = scrub_names("castle at dusk, portrait of Marco, blue lantern", inputs)
    assert "Marco" not in scrubbed
    assert "portrait" not in scrubbed
    assert "castle at dusk" in scrubbed
    assert "blue lantern" in scrubbed


def test_the_persona_name_and_the_scenario_title_are_scrubbed_too() -> None:
    scrubbed = scrub_names("Ash by the gate, night, walls of La Cité Noyée", INPUTS)
    assert "Ash" not in scrubbed
    assert "Grey coat, always damp." in scrubbed
    assert "night" in scrubbed
    assert "Noyée" not in scrubbed
    # The title has no appearance: it takes its whole keyword with it.
    assert "walls" not in scrubbed


def test_the_scenario_title_has_no_appearance_so_it_takes_its_keyword() -> None:
    scrubbed = scrub_names("La Cité Noyée, castle at dusk", INPUTS)
    assert "Noyée" not in scrubbed
    assert "castle at dusk" in scrubbed


def test_scrubbing_leaves_a_reply_without_any_name_untouched() -> None:
    reply = "harbour gate, blue lantern, dusk, wet stones"
    assert scrub_names(reply, INPUTS) == reply


def replace_instruction(inputs: ImagePromptInputs, **changes: Any) -> ImagePromptInputs:
    """A copy of the inputs with some fields replaced (test convenience)."""
    return replace(inputs, **changes)


# --- What the model is actually asked (issue #16, criterion 2) ------------------


def test_the_request_rides_verbatim_and_last() -> None:
    messages = build_messages(INPUTS)
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert INPUTS.instruction in messages[-1]["content"]
    # No section may follow the request: it is the instruction, not scenery.
    assert messages[-1]["content"].rstrip().endswith(INPUTS.instruction)


def test_the_scene_rides_whole() -> None:
    scene = build_messages(INPUTS)[-1]["content"]
    assert INPUTS.narration in scene
    for value in INPUTS.world_state.values():
        assert json.dumps(value, ensure_ascii=False) in scene


def test_only_appearance_fields_are_sent() -> None:
    scene = build_messages(INPUTS)[-1]["content"]
    assert f"Name: {ELENA.name}" in scene
    assert ELENA.appearance in scene
    assert MARCO.appearance in scene
    assert ASH.appearance in scene
    for forbidden in ("Personality", "Story", "Relationships", "Secrets"):
        assert forbidden not in scene


def test_the_system_message_asks_for_english_keywords_that_keep_the_details() -> None:
    rules = build_messages(INPUTS)[0]["content"]
    assert "English keywords" in rules
    assert "Never drop a detail" in rules
    assert "never by name" in rules


def test_no_active_persona_still_composes() -> None:
    """Issue #4's last criterion lands here too: an absent persona leaves the
    composition working, with the protagonist section simply absent."""
    inputs = replace_instruction(INPUTS, persona=None)
    messages = build_messages(inputs)
    assert "Protagonist" not in messages[-1]["content"]
    assert INPUTS.narration in messages[-1]["content"]
    # The scrub still runs over the characters: the persona is simply no
    # longer a name it knows.
    assert "Elena" not in scrub_names("Elena by the gate", inputs)


# --- The endpoint ---------------------------------------------------------------


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

    monkeypatch.setattr(images_routes.ollama, "chat", fake_chat)
    return calls


def _seed_party(
    *,
    with_persona: bool = True,
    with_world_state: bool = True,
    model: str | None = "test-model",
) -> str:
    """A scenario, two characters, a persona and a played opening, inserted
    directly: the party under test does not need a narrated history, only a
    scene to compose from. Returns the party id."""
    scenario_id, persona_id, party_id = (uuid.uuid4().hex for _ in range(3))
    with db.connect() as con:
        con.execute(
            "INSERT INTO scenario (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (scenario_id, "La Cité Noyée", 0.0, 0.0),
        )
        con.execute(
            "INSERT INTO instance (id, scenario_id, world_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                party_id,
                scenario_id,
                json.dumps({"location": "le port"}, ensure_ascii=False)
                if with_world_state
                else "{}",
                0.0,
                0.0,
            ),
        )
        con.execute(
            "INSERT INTO character (scenario_id, position, name, appearance, personality, secrets) "
            "VALUES (?, 0, ?, ?, ?, ?)",
            (scenario_id, "Elena", ELENA.appearance, "Brusque, loyal.", "She is the heir."),
        )
        con.execute(
            "INSERT INTO character (scenario_id, position, name, appearance) VALUES (?, 1, ?, ?)",
            (scenario_id, "Marco", MARCO.appearance),
        )
        con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ts) "
            "VALUES (?, 'assistant', 'text', ?, ?)",
            (party_id, "Elena attend près de la porte du port, la nuit tombée.", 0.0),
        )
        if with_persona:
            con.execute(
                "INSERT INTO persona (id, name, appearance, created_at) VALUES (?, ?, ?, ?)",
                (persona_id, ASH.name, ASH.appearance, 0.0),
            )
            settings.set_active_persona_id(con, persona_id)
        settings.set_llm_model(con, model)
    return party_id


def _compose(client: TestClient, party_id: str, instruction: str = "Elena sur le quai") -> Any:
    return client.post(f"/api/parties/{party_id}/image-prompt", json={"instruction": instruction})


def test_compose_returns_the_prompt_with_names_scrubbed(
    client: TestClient, monkeypatch: Any
) -> None:
    party_id = _seed_party()
    calls = _stub_chat(monkeypatch, reply="portrait of Elena, blue lantern, dusk")

    response = _compose(client, party_id, "Elena sur le quai au crépuscule")

    assert response.status_code == 200
    prompt = response.json()["prompt"]
    assert "Elena" not in prompt
    assert ELENA.appearance in prompt
    assert "blue lantern" in prompt
    # What the model was asked: the request verbatim, the scene, the persona,
    # the appearances — and none of the narrator-only fields.
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    scene = calls[0]["messages"][-1]["content"]
    assert "Elena sur le quai au crépuscule" in scene
    assert "Elena attend près de la porte du port" in scene
    assert "le port" in scene
    assert ELENA.appearance in scene
    assert ASH.appearance in scene
    assert "Brusque, loyal." not in scene
    assert "She is the heir." not in scene


def test_compose_sends_the_configured_num_ctx(client: TestClient, monkeypatch: Any) -> None:
    party_id = _seed_party()
    with db.connect() as con:
        settings.set_num_ctx(con, 2048)
    calls = _stub_chat(monkeypatch, reply="harbour at dusk")

    _compose(client, party_id)

    assert calls[0]["num_ctx"] == 2048


def test_unknown_party_is_404(client: TestClient, monkeypatch: Any) -> None:
    calls = _stub_chat(monkeypatch, reply="harbour")

    response = _compose(client, uuid.uuid4().hex)

    assert response.status_code == 404
    assert len(calls) == 0


def test_blank_instruction_is_400_before_any_model_call(
    client: TestClient, monkeypatch: Any
) -> None:
    party_id = _seed_party()
    calls = _stub_chat(monkeypatch, reply="harbour")

    for blank in ("", "   \n\t "):
        response = _compose(client, party_id, blank)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    assert len(calls) == 0


def test_unreachable_ollama_fails_the_composition_cleanly(
    client: TestClient, monkeypatch: Any
) -> None:
    party_id = _seed_party()
    reason = "Ollama is unreachable at http://localhost:11434: connection refused"
    _stub_chat(monkeypatch, error=OllamaUnreachable(reason))

    response = _compose(client, party_id)

    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]
    assert _nothing_written(party_id)


def test_ollama_error_is_502_and_writes_nothing(client: TestClient, monkeypatch: Any) -> None:
    party_id = _seed_party()
    _stub_chat(monkeypatch, error=OllamaError("Ollama returned HTTP 404: model not found"))

    response = _compose(client, party_id)

    assert response.status_code == 502
    assert "model not found" in response.json()["detail"]
    assert _nothing_written(party_id)


def test_empty_reply_is_502(client: TestClient, monkeypatch: Any) -> None:
    party_id = _seed_party()
    _stub_chat(monkeypatch, reply="   \n\t ")

    response = _compose(client, party_id)

    assert response.status_code == 502
    assert _nothing_written(party_id)


def test_a_reply_that_is_only_a_bare_name_scrubs_to_nothing(
    client: TestClient, monkeypatch: Any
) -> None:
    """A name with no description takes its keyword with it; when that is all
    the model returned, the composition fails rather than return an empty
    prompt for the player to wonder about. The scenario title is the seeded
    party's one name without an appearance."""
    party_id = _seed_party()
    _stub_chat(monkeypatch, reply="La Cité Noyée")

    response = _compose(client, party_id)

    assert response.status_code == 502
    assert "proper names" in response.json()["detail"]


def test_no_model_configured_is_400(client: TestClient, monkeypatch: Any) -> None:
    party_id = _seed_party(model=None)
    _stub_chat(monkeypatch, reply="harbour")

    response = _compose(client, party_id)

    assert response.status_code == 400
    assert "settings" in response.json()["detail"]


def test_composition_without_an_active_persona_still_works(
    client: TestClient, monkeypatch: Any
) -> None:
    party_id = _seed_party(with_persona=False)
    _stub_chat(monkeypatch, reply="portrait of Elena, dusk")

    response = _compose(client, party_id)

    assert response.status_code == 200
    assert "Elena" not in response.json()["prompt"]
    assert ELENA.appearance in response.json()["prompt"]


def test_composition_writes_nothing(client: TestClient, monkeypatch: Any) -> None:
    """Issue #16's third criterion, on the storage side: composition and
    generation are two steps, and the first one persists nothing at all."""
    party_id = _seed_party()
    _stub_chat(monkeypatch, reply="portrait of Elena, dusk")

    response = _compose(client, party_id)

    assert response.status_code == 200
    assert _nothing_written(party_id)


def _nothing_written(party_id: str) -> bool:
    """No `image` row exists, and the party's transcript is untouched."""
    with db.connect() as con:
        images = con.execute("SELECT COUNT(*) FROM image").fetchone()[0]
        messages = con.execute(
            "SELECT COUNT(*) FROM message WHERE instance_id = ?", (party_id,)
        ).fetchone()[0]
    return images == 0 and messages == 1
