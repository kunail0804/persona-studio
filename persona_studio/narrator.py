"""The narrator's prompt: input types, pure assembly, and SQLite loaders.

Everything here is deterministic string work. The assembly functions are pure —
they consume objects built in memory, so tests need no database and no HTTP —
and the loaders at the bottom only translate rows into those objects. Nothing
in this module calls Ollama; `build_chat_messages` and `build_opening_messages`
are the two call sites the routes use.

Two rules live in exactly one place each. `load_history`'s WHERE clause is the
only filter that keeps image messages from the narrator, and `_section` is the
only mechanism that drops an empty section from the prompt.
"""

from __future__ import annotations

import json
import math
import re
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
class PromptPlace:
    """A place of the scenario: what it is and looks like, and how it feels.

    `description` is factual and visual — it is also what an image prompt
    receives in place of the name. `atmosphere` is sensory, for narration.
    """

    name: str
    description: str
    atmosphere: str


@dataclass(frozen=True)
class PromptLore:
    """One lorebook entry: text that enters the prompt only when relevant."""

    keywords: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class PromptScenario:
    """The scenario's prompt-facing fields. `item` stays out: an object that
    must exist and come back is a lore entry keyed by its own name.

    Places ride in every prompt, like characters: the narrator has to know a
    place exists to take the story there. Lore does not — `lore` holds every
    entry, and `select_lore` decides which of them a given turn receives.
    """

    title: str
    synopsis: str
    world_rules: str
    arcs: str
    characters: tuple[PromptCharacter, ...]
    places: tuple[PromptPlace, ...] = ()
    lore: tuple[PromptLore, ...] = ()


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

OPENING_INSTRUCTION = (
    "Write the opening scene of this story, in one reply. Set the scene: where "
    "the protagonist is, what is happening around them, who else is present. "
    "End on something the player can answer. Do not act, speak or decide for "
    "the protagonist — they have not acted yet."
)


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


def _place_sheet(place: PromptPlace) -> str:
    return "\n".join(
        part
        for part in (
            _field("Name", place.name),
            _field("Description", place.description),
            _field("Atmosphere", place.atmosphere),
        )
        if part
    )


def _places_section(places: tuple[PromptPlace, ...]) -> str:
    sheets = [sheet for sheet in (_place_sheet(p) for p in places) if sheet]
    if not sheets:
        return ""
    return "Places:\n\n" + "\n\n".join(sheets)


def _lore_section(entries: Sequence[PromptLore]) -> str:
    texts = [entry.text.strip() for entry in entries if entry.text.strip()]
    if not texts:
        return ""
    return "What the narrator knows that bears on this scene:\n\n" + "\n\n".join(texts)


# --- The lorebook ----------------------------------------------------------------
#
# An entry enters the prompt only when one of its keywords appears in the most
# recent messages. That is the whole point of it: everything else a scenario
# holds is sent on every turn, so a world that grows makes every prompt grow
# with it. The lorebook lets a world hold a hundred facts and send the three
# that the scene is actually touching.
#
# Both bounds exist for `num_ctx`. Scanning only the last few messages keeps
# an entry from staying in the prompt long after the story moved on; the cap
# keeps a scene that mentions everything at once from flooding the window.
# The values are the previous version's, which were used in play.

LORE_SCAN_MESSAGES = 10
LORE_MAX_ENTRIES = 8


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    # Lookarounds rather than `\b`: `\b` needs a word character at the edge,
    # so a keyword ending in punctuation ("C++", "M.") would never match.
    # Whole words either way — "Ana" must not trigger on "banana".
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)


def select_lore(
    entries: Sequence[PromptLore],
    history: Sequence[HistoryMessage],
    *,
    scan: int = LORE_SCAN_MESSAGES,
    limit: int = LORE_MAX_ENTRIES,
) -> list[PromptLore]:
    """The entries a turn receives: those whose keywords the recent story uses.

    Pure. Case-insensitive, whole-word matching against the last `scan`
    messages of `history`, out-of-game turns included — a player asking the
    narrator about the Order is talking about the Order. Entries keep the
    scenario's order, each enters at most once, and at most `limit` enter.
    An entry with no usable keyword or no text never enters.
    """
    recent = "\n".join(message.content for message in history[-scan:]) if scan > 0 else ""
    if not recent:
        return []
    selected: list[PromptLore] = []
    for entry in entries:
        if len(selected) >= limit:
            break
        if not entry.text.strip():
            continue
        keywords = [k.strip() for k in entry.keywords if k.strip()]
        if any(_keyword_pattern(keyword).search(recent) for keyword in keywords):
            selected.append(entry)
    return selected


def build_system_prompt(
    scenario: PromptScenario,
    persona: PromptPersona | None,
    summary: str = "",
    lore: Sequence[PromptLore] = (),
) -> str:
    """Assemble the narrator's system prompt.

    Pure: the same inputs always yield the same string. Every empty section is
    omitted rather than sent blank — the omission is `_section`'s job, not a
    chain of ifs here. With no active persona the protagonist section is absent
    and the rest of the prompt is byte-for-byte what it would have been with
    one. The summary comes last: it is recent state that continues into the
    history messages which follow the system prompt, not scenario definition.

    `lore` is the entries already selected for this turn — see `select_lore`
    — and sits just before the summary: it is about the current scene, not
    the world's permanent definition.
    """
    sections = [
        _intro(),
        _field("Scenario", scenario.title),
        _section("Synopsis:", scenario.synopsis),
        _section("World rules:", scenario.world_rules),
        _section("Story arcs:", scenario.arcs),
        _protagonist_section(persona) if persona is not None else "",
        _characters_section(scenario.characters),
        _places_section(scenario.places),
        _lore_section(lore),
        _section("What happened earlier:", summary),
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
    """The full message list for a chat call: the system prompt, then history.

    The lorebook is selected here, from the same history the call sends, so
    every caller — a played turn, a regeneration, the prompt view — gets the
    same entries for the same story.
    """
    lore = select_lore(scenario.lore, history)
    return [
        {"role": "system", "content": build_system_prompt(scenario, persona, summary, lore)},
        *build_history(history),
    ]


def build_opening_messages(
    scenario: PromptScenario,
    persona: PromptPersona | None,
) -> list[dict[str, str]]:
    """The message list for a party's opening call: system prompt, then direction.

    The instruction rides as a `system` message, not a played turn: nothing has
    happened in the story yet, so there is no turn to play — the same reasoning
    that makes out-of-game turns system messages in `build_history`. It is a
    direction to the narrator and is never persisted.
    """
    return [
        {"role": "system", "content": build_system_prompt(scenario, persona)},
        {"role": "system", "content": OPENING_INSTRUCTION},
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
    places = tuple(
        PromptPlace(
            name=place_row["name"],
            description=place_row["description"],
            atmosphere=place_row["atmosphere"],
        )
        for place_row in con.execute(
            "SELECT name, description, atmosphere FROM place "
            "WHERE scenario_id = ? ORDER BY position",
            (scenario_id,),
        ).fetchall()
    )
    lore = tuple(
        PromptLore(keywords=parse_keywords(lore_row["keywords"]), text=lore_row["text"])
        for lore_row in con.execute(
            "SELECT keywords, text FROM lore WHERE scenario_id = ? ORDER BY position",
            (scenario_id,),
        ).fetchall()
    )
    return PromptScenario(
        title=row["title"],
        synopsis=row["synopsis"],
        world_rules=row["world_rules"],
        arcs=row["arcs"],
        characters=characters,
        places=places,
        lore=lore,
    )


def parse_keywords(raw: str) -> tuple[str, ...]:
    """A lore entry's stored keywords; anything malformed is none at all.

    The column holds a JSON list. A hand-edited or half-written value must
    never break a turn, so a non-list, or a non-string inside one, drops out
    rather than raising — an entry with no keyword simply never triggers.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        return ()
    if not isinstance(data, list):
        return ()
    return tuple(item for item in data if isinstance(item, str))


def load_persona(con: sqlite3.Connection, persona_id: str | None) -> PromptPersona | None:
    """A persona as prompt input, or None when there is none or it is gone."""
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


def load_history(
    con: sqlite3.Connection,
    instance_id: str,
    window: int,
    before_id: int | None = None,
    after_id: int | None = None,
) -> list[HistoryMessage]:
    """The last `window` text messages of a party, oldest first.

    With `before_id`, only messages strictly before that id are read. The
    regenerate path uses it to ask for the history as it stood before the
    reply being replaced: with the old reply still in the prompt, the model
    reads its own text as history and writes a continuation instead of an
    alternative.

    With `after_id`, only messages strictly after that id are read. The turn
    passes the rolling summary's frontier (`instance.summary_upto`), so the
    messages the summary already stands in for are never re-sent. The two
    bounds compose: the regenerate path can ask for the window as it stood
    before a reply *and* after the summary frontier.

    Raises ValueError when the window is below 1: SQLite reads a non-positive
    LIMIT as no limit at all, so clamping would silently return the entire
    history. A bad window is a caller bug to surface, not input to forgive.

    `kind = 'text'` in the WHERE clause is the single place that keeps image
    messages away from the narrator; the pure functions must not repeat the
    guard.
    """
    if window < 1:
        raise ValueError(f"History window must be at least 1, got {window}")
    where = "instance_id = ? AND kind = 'text'"
    parameters: list[str | int] = [instance_id]
    if before_id is not None:
        where += " AND id < ?"
        parameters.append(before_id)
    if after_id is not None:
        where += " AND id > ?"
        parameters.append(after_id)
    rows = con.execute(
        f"SELECT id, role, content, ooc FROM message WHERE {where} ORDER BY id DESC LIMIT ?",
        (*parameters, window),
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


def load_active_persona(con: sqlite3.Connection) -> PromptPersona | None:
    """The default persona — the one a new party starts with — as prompt input."""
    return load_persona(con, settings.get_active_persona_id(con))


def load_party_persona(con: sqlite3.Connection, party_id: str) -> PromptPersona | None:
    """The persona this party is played as, or None when it has none.

    Each party carries its own since schema v4: two parties in progress can
    have two different protagonists, which a single global setting made
    impossible.
    """
    row = con.execute("SELECT persona_id FROM instance WHERE id = ?", (party_id,)).fetchone()
    if row is None:
        return None
    return load_persona(con, row["persona_id"])
