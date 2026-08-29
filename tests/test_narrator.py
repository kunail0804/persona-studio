"""Tests for the narrator's prompt assembly, windowing, and loaders.

The pure tests build their inputs in memory: no database, no TestClient, no
HTTP. The loader tests use the shared test database through the `client`
fixture, which has run the app's startup migration.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, settings
from persona_studio.narrator import (
    OPENING_INSTRUCTION,
    HistoryMessage,
    PromptCharacter,
    PromptPersona,
    PromptScenario,
    build_chat_messages,
    build_history,
    build_opening_messages,
    build_system_prompt,
    context_usage,
    estimate_tokens,
    load_active_persona,
    load_history,
    load_scenario,
)


@pytest.fixture
def connection(client: TestClient) -> Iterator[sqlite3.Connection]:
    """A connection to the migrated test database; `client` ran the migration."""
    with db.connect() as con:
        yield con


# --- Shared inputs -------------------------------------------------------------


TITLE_ONLY = PromptScenario(
    title="The Drowned City", synopsis="", world_rules="", arcs="", characters=()
)

FULL_SCENARIO = PromptScenario(
    title="The Drowned City",
    synopsis="A city sank beneath the sea; its streets are still walkable.",
    world_rules="Magic costs a memory per casting.",
    arcs="Act one: arrival. Act two: the sunken cathedral.",
    characters=(
        PromptCharacter(
            name="Elsa",
            appearance="Red hair, green eyes, a scar across one brow.",
            personality="Brusque, loyal to a fault.",
            story="Grew up salvaging wrecks; never knew her parents.",
            relationships="Distrusts Marco, owes the harbourmaster a favour.",
            secrets="She is the heir the bounty letters describe.",
        ),
        PromptCharacter(
            name="Marco",
            appearance="Tall, greying beard, navy coat.",
            personality="Charming, evasive.",
            story="",
            relationships="",
            secrets="",
        ),
    ),
)

PERSONA = PromptPersona(
    name="Ash",
    description="A cartographer chasing the city's last map.",
    appearance="Grey coat, always damp.",
    traits="Curious, stubborn.",
)

SUMMARY = "Ash arrived at the harbour and met Elsa at the gate."


def _assert_no_empty_section(prompt: str) -> None:
    """No heading sits above nothing, and no line carries trailing whitespace."""
    lines = prompt.split("\n")
    assert not prompt.startswith("\n")
    assert all(line == line.rstrip() for line in lines)
    for index, line in enumerate(lines):
        if not line.endswith(":"):
            continue
        rest = [following for following in lines[index + 1 :] if following.strip()]
        assert rest and not rest[0].endswith(":"), f"empty section under {line!r}"


# --- The system prompt ---------------------------------------------------------


def test_title_only_scenario_omits_every_empty_section() -> None:
    prompt = build_system_prompt(TITLE_ONLY, None)
    assert "Scenario: The Drowned City" in prompt
    for heading in [
        "What happened earlier:",
        "Synopsis:",
        "World rules:",
        "Story arcs:",
        "Protagonist",
        "Characters:",
    ]:
        assert heading not in prompt
    _assert_no_empty_section(prompt)


def test_populated_scenario_produces_the_exact_prompt() -> None:
    expected = (
        "You are the narrator of an interactive story. You narrate the world and "
        "voice every character except the protagonist, whom the player alone plays.\n"
        "\n"
        "Narration rules:\n"
        "- Never speak, act, think or decide for the protagonist.\n"
        "- Never write the protagonist's dialogue.\n"
        "- End every reply on something the player can answer: a question, a "
        "decision, or a situation that awaits their character.\n"
        "\n"
        "Scenario: The Drowned City\n"
        "\n"
        "Synopsis:\n"
        "A city sank beneath the sea; its streets are still walkable.\n"
        "\n"
        "World rules:\n"
        "Magic costs a memory per casting.\n"
        "\n"
        "Story arcs:\n"
        "Act one: arrival. Act two: the sunken cathedral.\n"
        "\n"
        "Protagonist (the player's character):\n"
        "Name: Ash\n"
        "Description: A cartographer chasing the city's last map.\n"
        "Appearance: Grey coat, always damp.\n"
        "Traits: Curious, stubborn.\n"
        "\n"
        "Characters:\n"
        "\n"
        "Name: Elsa\n"
        "Appearance: Red hair, green eyes, a scar across one brow.\n"
        "Personality: Brusque, loyal to a fault.\n"
        "Story: Grew up salvaging wrecks; never knew her parents.\n"
        "Relationships: Distrusts Marco, owes the harbourmaster a favour.\n"
        "Secrets (narrator only — never reveal these to the player, "
        "only play them out): She is the heir the bounty letters describe.\n"
        "\n"
        "Name: Marco\n"
        "Appearance: Tall, greying beard, navy coat.\n"
        "Personality: Charming, evasive.\n"
        "\n"
        "What happened earlier:\n"
        f"{SUMMARY}"
    )
    assert build_system_prompt(FULL_SCENARIO, PERSONA, SUMMARY) == expected


def test_secrets_appear_under_narrator_only_wording() -> None:
    prompt = build_system_prompt(FULL_SCENARIO, None)
    assert "She is the heir the bounty letters describe." in prompt
    assert "narrator only" in prompt
    assert "never reveal these to the player" in prompt


def test_protagonist_rule_and_persona_fields_are_injected() -> None:
    prompt = build_system_prompt(FULL_SCENARIO, PERSONA)
    assert "Never speak, act, think or decide for the protagonist." in prompt
    assert "Never write the protagonist's dialogue." in prompt
    assert "Protagonist (the player's character):" in prompt
    assert "Name: Ash" in prompt
    assert PERSONA.description in prompt
    assert PERSONA.appearance in prompt
    assert PERSONA.traits in prompt


def test_no_persona_yields_the_prompt_minus_the_protagonist_section() -> None:
    with_persona = build_system_prompt(FULL_SCENARIO, PERSONA, SUMMARY)
    without_persona = build_system_prompt(FULL_SCENARIO, None, SUMMARY)
    start = with_persona.index("Protagonist (the player's character):")
    end = with_persona.index("Characters:")
    assert without_persona == with_persona[:start] + with_persona[end:]


def test_an_all_empty_persona_yields_no_protagonist_section() -> None:
    empty = PromptPersona(name="", description="", appearance="", traits="")
    assert build_system_prompt(FULL_SCENARIO, empty) == build_system_prompt(FULL_SCENARIO, None)


def test_assembly_is_pure() -> None:
    assert build_system_prompt(FULL_SCENARIO, PERSONA, SUMMARY) == build_system_prompt(
        FULL_SCENARIO, PERSONA, SUMMARY
    )


def test_a_multi_line_field_stays_inside_its_own_block() -> None:
    scenario = PromptScenario(
        title="",
        synopsis="",
        world_rules="",
        arcs="",
        characters=(
            PromptCharacter(
                name="Elsa",
                appearance="Red hair.\nGreen eyes.",
                personality="",
                story="",
                relationships="",
                secrets="",
            ),
        ),
    )
    prompt = build_system_prompt(scenario, None)
    assert "Appearance:\n  Red hair.\n  Green eyes." in prompt
    _assert_no_empty_section(prompt)


# --- History and chat messages -------------------------------------------------


def test_out_of_game_turn_becomes_a_system_message_at_its_position() -> None:
    history = [
        HistoryMessage(id=1, role="user", content="I draw my sword.", ooc=False),
        HistoryMessage(id=2, role="user", content="Describe the room more slowly.", ooc=True),
        HistoryMessage(id=3, role="assistant", content="The room is vast.", ooc=False),
    ]
    messages = build_history(history)
    assert messages[0] == {"role": "user", "content": "I draw my sword."}
    assert messages[2] == {"role": "assistant", "content": "The room is vast."}
    ooc = messages[1]
    assert ooc["role"] == "system"
    assert ooc["content"].endswith("Describe the room more slowly.")
    # Worded as an instruction from the player, never as a played action.
    assert "instruction from the player" in ooc["content"]


def test_build_chat_messages_puts_the_system_prompt_first() -> None:
    history = [HistoryMessage(id=1, role="user", content="Hello.", ooc=False)]
    messages = build_chat_messages(TITLE_ONLY, None, history)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == build_system_prompt(TITLE_ONLY, None)
    assert messages[1] == {"role": "user", "content": "Hello."}


def test_build_opening_messages_is_system_prompt_then_system_direction() -> None:
    """The opening direction is a system message, not a played turn: nothing has
    happened in the story yet, so there is no turn to play."""
    messages = build_opening_messages(FULL_SCENARIO, PERSONA)
    assert messages == [
        {"role": "system", "content": build_system_prompt(FULL_SCENARIO, PERSONA)},
        {"role": "system", "content": OPENING_INSTRUCTION},
    ]


def test_opening_instruction_directs_without_playing_the_protagonist() -> None:
    assert "opening scene" in OPENING_INSTRUCTION
    assert "Do not act" in OPENING_INSTRUCTION


def test_estimate_tokens_grows_with_content() -> None:
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "a" * 100_000}]
    assert estimate_tokens([]) == 0
    assert estimate_tokens(small) > 0
    assert estimate_tokens(large) > estimate_tokens(small)


def test_context_usage_flags_only_prompts_past_the_warning_ratio() -> None:
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "a" * 100_000}]
    quiet = context_usage(small, 8192)
    loud = context_usage(large, 8192)
    assert quiet.near_limit is False
    assert quiet.num_ctx == 8192
    assert loud.near_limit is True
    assert loud.estimated_tokens == estimate_tokens(large)


# --- Loaders -------------------------------------------------------------------


def _new_instance(con: sqlite3.Connection) -> str:
    scenario_id, instance_id = uuid.uuid4().hex, uuid.uuid4().hex
    con.execute(
        "INSERT INTO scenario (id, created_at, updated_at) VALUES (?, ?, ?)",
        (scenario_id, 0.0, 0.0),
    )
    con.execute(
        "INSERT INTO instance (id, scenario_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (instance_id, scenario_id, 0.0, 0.0),
    )
    return instance_id


def _insert_messages(
    con: sqlite3.Connection,
    instance_id: str,
    specs: list[tuple[str, str, str, int]],
) -> list[int]:
    """Insert (role, kind, content, ooc) rows; return their ids, oldest first."""
    ids = []
    for role, kind, content, ooc in specs:
        cursor = con.execute(
            "INSERT INTO message (instance_id, role, kind, content, ooc, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (instance_id, role, kind, content, ooc, 0.0),
        )
        ids.append(cursor.lastrowid)
    return ids


def test_window_returns_the_last_messages_oldest_first(connection: sqlite3.Connection) -> None:
    instance_id = _new_instance(connection)
    specs = [("user" if i % 2 == 0 else "assistant", "text", f"m{i}", 0) for i in range(50)]
    ids = _insert_messages(connection, instance_id, specs)
    messages = load_history(connection, instance_id, 10)
    assert [message.id for message in messages] == ids[-10:]
    assert [message.content for message in messages] == [f"m{i}" for i in range(40, 50)]
    assert messages[0].role == "user"
    assert messages[0].ooc is False


@pytest.mark.parametrize("window", [0, -1])
def test_load_history_rejects_a_window_below_one(
    connection: sqlite3.Connection, window: int
) -> None:
    """A non-positive window is a caller bug: raise naming the value, never clamp.

    SQLite reads a negative LIMIT as no limit at all, so an unguarded window
    would silently return the entire history.
    """
    instance_id = _new_instance(connection)
    specs = [("user" if i % 2 == 0 else "assistant", "text", f"m{i}", 0) for i in range(30)]
    _insert_messages(connection, instance_id, specs)
    with pytest.raises(ValueError, match=str(window)):
        load_history(connection, instance_id, window)


def test_image_messages_never_reach_the_narrator(connection: sqlite3.Connection) -> None:
    instance_id = _new_instance(connection)
    specs = [
        ("user", "text", "I look around.", 0),
        ("user", "image", "the generated scene", 0),
        ("assistant", "text", "You see a harbour.", 0),
    ]
    _insert_messages(connection, instance_id, specs)
    messages = load_history(connection, instance_id, 10)
    assert [message.content for message in messages] == ["I look around.", "You see a harbour."]
    received = build_history(messages)
    assert all("the generated scene" not in message["content"] for message in received)


def test_ooc_flag_survives_the_loader(connection: sqlite3.Connection) -> None:
    instance_id = _new_instance(connection)
    _insert_messages(connection, instance_id, [("user", "text", "Pause a moment.", 1)])
    messages = load_history(connection, instance_id, 10)
    assert messages[0].ooc is True
    assert build_history(messages)[0]["role"] == "system"


def test_load_scenario_reads_fields_and_ordered_characters(
    connection: sqlite3.Connection,
) -> None:
    scenario_id = uuid.uuid4().hex
    connection.execute(
        "INSERT INTO scenario (id, title, synopsis, world_rules, arcs, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (scenario_id, "T", "S", "W", "A", 0.0, 0.0),
    )
    connection.execute(
        "INSERT INTO character (scenario_id, position, name, secrets) VALUES (?, 0, ?, ?)",
        (scenario_id, "Elsa", "her secret"),
    )
    connection.execute(
        "INSERT INTO character (scenario_id, position, name) VALUES (?, 1, ?)",
        (scenario_id, "Marco"),
    )
    scenario = load_scenario(connection, scenario_id)
    assert scenario.title == "T"
    assert scenario.synopsis == "S"
    assert scenario.world_rules == "W"
    assert scenario.arcs == "A"
    assert [character.name for character in scenario.characters] == ["Elsa", "Marco"]
    assert scenario.characters[0].secrets == "her secret"
    assert scenario.characters[1].secrets == ""


def test_load_scenario_raises_when_the_scenario_is_missing(
    connection: sqlite3.Connection,
) -> None:
    with pytest.raises(LookupError):
        load_scenario(connection, uuid.uuid4().hex)


def test_load_active_persona_reads_the_active_persona(connection: sqlite3.Connection) -> None:
    persona_id = uuid.uuid4().hex
    connection.execute(
        "INSERT INTO persona (id, name, description, appearance, traits, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (persona_id, "Ash", "A cartographer.", "Grey coat.", "Stubborn.", 0.0),
    )
    settings.set_active_persona_id(connection, persona_id)
    assert load_active_persona(connection) == PromptPersona(
        name="Ash", description="A cartographer.", appearance="Grey coat.", traits="Stubborn."
    )


def test_load_active_persona_is_none_when_none_is_set(connection: sqlite3.Connection) -> None:
    settings.set_active_persona_id(connection, None)
    assert load_active_persona(connection) is None


def test_load_active_persona_is_none_when_the_id_dangles(
    connection: sqlite3.Connection,
) -> None:
    settings.set_active_persona_id(connection, "no-such-persona")
    assert load_active_persona(connection) is None
