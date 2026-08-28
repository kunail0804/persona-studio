"""The narrator's prompt: input types, pure assembly, and SQLite loaders.

Everything here is deterministic string work. The assembly functions are pure —
they consume objects built in memory, so tests need no database and no HTTP —
and the loaders at the bottom only translate rows into those objects. Nothing
in this module calls Ollama; `build_chat_messages` is the single call site the
chat routes will use.

Two rules live in exactly one place each. `load_history`'s WHERE clause is the
only filter that keeps image messages from the narrator, and `_section` is the
only mechanism that drops an empty section from the prompt.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from . import settings

# --- Input types ---------------------------------------------------------------


@dataclass(frozen=True)
class PromptCharacter:
    """One character sheet as the prompt builder sees it: plain fields, no ids."""

    name: str
    appearance: str
    personality: str
    story: str
    relationships: str
    secrets: str


@dataclass(frozen=True)
class PromptPersona:
    """The protagonist the player plays; absent when no persona is active."""

    name: str
    description: str
    appearance: str
    traits: str


@dataclass(frozen=True)
class PromptScenario:
    """The scenario's prompt-facing fields; `place`, `item` and `lore` stay out."""

    title: str
    synopsis: str
    world_rules: str
    arcs: str
    characters: tuple[PromptCharacter, ...]


@dataclass(frozen=True)
class HistoryMessage:
    """One past turn, already filtered to text messages by the loader."""

    id: int
    role: Literal["user", "assistant"]
    content: str
    ooc: bool


# --- The prompt format ---------------------------------------------------------
#
# Sections are joined by exactly one blank line. Every section builder returns
# an empty string for an empty input, and `build_system_prompt` joins only the
# non-empty ones — so a heading with nothing under it cannot exist, whatever
# the inputs look like.

_INTRO = (
    "You are the narrator of an interactive story. You narrate the world and "
    "voice every character except the protagonist, whom the player alone plays."
)

_NARRATION_RULES = (
    "Narration rules:\n"
    "- Never speak, act, think or decide for the protagonist.\n"
    "- Never write the protagonist's dialogue.\n"
    "- End every reply on something the player can answer: a question, a "
    "decision, or a situation that awaits their character."
)

_SECRETS_LABEL = "Secrets (narrator only — never reveal these to the player, only play them out)"

_OOC_PREFIX = "Out-of-game instruction from the player, not part of the story: "


def _section(heading: str, body: str) -> str:
    """A heading and its body, or nothing at all when the body is blank.

    Omission lives here, once: a blank body yields no section, so no caller can
    produce a heading with nothing under it.
    """
    text = body.strip()
    if not text:
        return ""
    return f"{heading}\n{text}"


def _field(label: str, value: str) -> str:
    """`label: value`, or nothing when the value is blank.

    A multi-line value keeps its newlines, indented under the label so the
    block still reads as one field.
    """
    text = value.strip()
    if not text:
        return ""
    if "\n" in text:
        return f"{label}:\n  " + text.replace("\n", "\n  ")
    return f"{label}: {text}"


def _intro() -> str:
    return f"{_INTRO}\n\n{_NARRATION_RULES}"


def _protagonist_section(persona: PromptPersona) -> str:
    fields = "\n".join(
        part
        for part in (
            _field("Name", persona.name),
            _field("Description", persona.description),
            _field("Appearance", persona.appearance),
            _field("Traits", persona.traits),
        )
        if part
    )
    return _section("Protagonist (the player's character):", fields)


def _character_sheet(character: PromptCharacter) -> str:
    return "\n".join(
        part
        for part in (
            _field("Name", character.name),
            _field("Appearance", character.appearance),
            _field("Personality", character.personality),
            _field("Story", character.story),
            _field("Relationships", character.relationships),
            _field(_SECRETS_LABEL, character.secrets),
        )
        if part
    )


def _characters_section(characters: tuple[PromptCharacter, ...]) -> str:
    sheets = [sheet for sheet in (_character_sheet(c) for c in characters) if sheet]
    if not sheets:
        return ""
    return "Characters:\n\n" + "\n\n".join(sheets)


def build_system_prompt(
    scenario: PromptScenario,
    persona: PromptPersona | None,
    summary: str = "",
) -> str:
    """Assemble the narrator's system prompt.

    Pure: the same inputs always yield the same string. Every empty section is
    omitted rather than sent blank — the omission is `_section`'s job, not a
    chain of ifs here. With no active persona the protagonist section is absent
    and the rest of the prompt is byte-for-byte what it would have been with
    one.
    """
    sections = [
        _intro(),
        _section("What happened earlier:", summary),
        _field("Scenario", scenario.title),
        _section("Synopsis:", scenario.synopsis),
        _section("World rules:", scenario.world_rules),
        _section("Story arcs:", scenario.arcs),
        _protagonist_section(persona) if persona is not None else "",
        _characters_section(scenario.characters),
    ]
    return "\n\n".join(section for section in sections if section)


def build_history(messages: Sequence[HistoryMessage]) -> list[dict[str, str]]:
    """Map history to the Ollama chat shape, in the order given.

    An out-of-game turn becomes a `system` message at its own position, worded
    as an instruction from the player, so the model reads it as an instruction
    and never as a played action.
    """
    return [
        {"role": "system", "content": _OOC_PREFIX + message.content}
        if message.ooc
        else {"role": message.role, "content": message.content}
        for message in messages
    ]


def build_chat_messages(
    scenario: PromptScenario,
    persona: PromptPersona | None,
    history: Sequence[HistoryMessage],
    summary: str = "",
) -> list[dict[str, str]]:
    """The full message list for a chat call: the system prompt, then history."""
    return [
        {"role": "system", "content": build_system_prompt(scenario, persona, summary)},
        *build_history(history),
    ]


# --- Context size --------------------------------------------------------------
#
# The installed Ollama exposes no tokenizer endpoint (POST /api/tokenize
# answers 404), so token counts here are estimates: characters divided by 3.5,
# plus a fixed per-message overhead for the chat framing. The divisor errs
# high on purpose — a context warning should fire early, not late. Real
# replies carry `prompt_eval_count`, which a later pull request should prefer.

CHARS_PER_TOKEN = 3.5
TOKENS_PER_MESSAGE_OVERHEAD = 4
CONTEXT_WARNING_RATIO = 0.75


@dataclass(frozen=True)
class ContextUsage:
    """Estimated tokens against the configured window, and whether to warn."""

    estimated_tokens: int
    num_ctx: int
    near_limit: bool


def estimate_tokens(messages: Sequence[dict[str, str]]) -> int:
    """Estimate a message list's token count; see the context-size note above.

    The estimate deliberately over-counts so a context warning fires early
    rather than late. A real count arrives with every reply —
    `prompt_eval_count` — and a later pull request should prefer it.
    """
    return sum(
        TOKENS_PER_MESSAGE_OVERHEAD + math.ceil(len(message.get("content", "")) / CHARS_PER_TOKEN)
        for message in messages
    )


def context_usage(messages: Sequence[dict[str, str]], num_ctx: int) -> ContextUsage:
    """Estimated tokens against `num_ctx`, flagged once past the warning ratio."""
    estimated = estimate_tokens(messages)
    return ContextUsage(
        estimated_tokens=estimated,
        num_ctx=num_ctx,
        near_limit=estimated >= num_ctx * CONTEXT_WARNING_RATIO,
    )


# --- Loaders -------------------------------------------------------------------


def load_scenario(con: sqlite3.Connection, scenario_id: str) -> PromptScenario:
    """Read a scenario and its characters, in order, as prompt inputs.

    Raises LookupError when the scenario does not exist; the route that calls
    this is the one that turns that into a 404.
    """
    row = con.execute(
        "SELECT title, synopsis, world_rules, arcs FROM scenario WHERE id = ?",
        (scenario_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"Scenario {scenario_id!r} not found")
    characters = tuple(
        PromptCharacter(
            name=character_row["name"],
            appearance=character_row["appearance"],
            personality=character_row["personality"],
            story=character_row["story"],
            relationships=character_row["relationships"],
            secrets=character_row["secrets"],
        )
        for character_row in con.execute(
            "SELECT name, appearance, personality, story, relationships, secrets "
            "FROM character WHERE scenario_id = ? ORDER BY position",
            (scenario_id,),
        ).fetchall()
    )
    return PromptScenario(
        title=row["title"],
        synopsis=row["synopsis"],
        world_rules=row["world_rules"],
        arcs=row["arcs"],
        characters=characters,
    )


def load_active_persona(con: sqlite3.Connection) -> PromptPersona | None:
    """The active persona as prompt input, or None when unset or dangling."""
    persona_id = settings.get_active_persona_id(con)
    if persona_id is None:
        return None
    row = con.execute(
        "SELECT name, description, appearance, traits FROM persona WHERE id = ?",
        (persona_id,),
    ).fetchone()
    if row is None:
        return None
    return PromptPersona(
        name=row["name"],
        description=row["description"],
        appearance=row["appearance"],
        traits=row["traits"],
    )


def load_history(con: sqlite3.Connection, instance_id: str, window: int) -> list[HistoryMessage]:
    """The last `window` text messages of a party, oldest first.

    `kind = 'text'` in the WHERE clause is the single place that keeps image
    messages away from the narrator; the pure functions must not repeat the
    guard.
    """
    rows = con.execute(
        "SELECT id, role, content, ooc FROM message "
        "WHERE instance_id = ? AND kind = 'text' "
        "ORDER BY id DESC LIMIT ?",
        (instance_id, window),
    ).fetchall()
    return [
        HistoryMessage(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            ooc=bool(row["ooc"]),
        )
        for row in reversed(rows)
    ]
