"""The image prompt: what the model is asked, and what comes back cleaned.

The same split as `narrator.py`: pure functions over in-memory inputs, plus
loaders that translate rows into those inputs. Nothing here calls Ollama;
`routes/images.py` is the call site.

The input types carry **only** what an image may show. A character's
personality, story, relationships and secrets never reach this module — not
filtered out at assembly time, simply absent from the type — so no code path
here can leak them into a prompt.

Names are the first-class problem. The instruction to the model is the first
line of defence against proper names reaching an image generator; `scrub_names`
is the deterministic second line. It works on the model's **answer**, not on
the prompt: the model may see names (it must, to tell which appearance belongs
to whom), the returned keywords may not.

Place names are excluded, not scrubbed. There is no bounded list of known
places to match against the way there is for characters — the `place` table
carries no CRUD and nothing writes to it, so the only place-shaped text is
`world_state["location"]`, free text the summariser writes. That single key
is dropped before the call rather than sent and then cleaned, because there
is nothing to clean it *against*. The rest of `world_state`, and the
narration itself, can still name a place in passing prose; no deterministic
pass can catch that without a list to check names against, the same limit
`scrub_names` already lives with for an uncommon character name.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from . import settings, summarizer
from .narrator import _field, _section

# --- Input types ---------------------------------------------------------------


@dataclass(frozen=True)
class Appearance:
    """One being as the image prompt sees it: a name and a physical look."""

    name: str
    appearance: str


@dataclass(frozen=True)
class ImagePromptInputs:
    """Everything one composition is allowed to know.

    `narration` is the party's most recent narration and `world_state` the
    stored one — together they are the current scene. The scenario title rides
    along for one reason only: it is a proper name, and `scrub_names` must
    remove it from the answer.
    """

    instruction: str
    narration: str
    world_state: dict[str, object]
    persona: Appearance | None
    characters: tuple[Appearance, ...]
    scenario_title: str


# --- The prompt format ---------------------------------------------------------


_SYSTEM_INSTRUCTION = (
    "You compose the prompt of an image generator from a scene of an "
    "interactive story.\n"
    "Rules:\n"
    "- Answer with a short comma-separated list of English keywords only: no "
    "sentences, no explanations, no quotes around the answer.\n"
    "- Keep every explicit detail the request asks for: objects, colours, "
    "counts, poses, actions. Never drop a detail and never summarise the "
    "request away.\n"
    "- Describe people by their physical appearance, using the appearance "
    "notes provided, and never by name.\n"
    "- Do not invent people, objects or events that the scene and the request "
    "do not mention."
)


# The one key excluded before the call: see the module docstring for why
# exclusion, not a scrub, is the only honest way to keep a place name out.
_EXCLUDED_WORLD_STATE_KEYS = frozenset({"location"})


def _world_state_lines(world_state: dict[str, object]) -> str:
    """The stored world state as `key: value` lines, `location` excluded.

    The shape is the summariser's (`location`, `present`, `established`, ...),
    and every key but `location` is rendered generically: a new key the
    summariser starts producing rides along without this module learning
    about it.
    """
    lines = (
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in world_state.items()
        if key not in _EXCLUDED_WORLD_STATE_KEYS
    )
    return "\n".join(lines)


def _appearance_sheet(being: Appearance) -> str:
    return "\n".join(
        part
        for part in (_field("Name", being.name), _field("Appearance", being.appearance))
        if part
    )


def _beings_section(heading: str, beings: tuple[Appearance, ...]) -> str:
    sheets = "\n\n".join(sheet for sheet in (_appearance_sheet(being) for being in beings) if sheet)
    return _section(heading, sheets)


def build_messages(inputs: ImagePromptInputs) -> list[dict[str, str]]:
    """The two-message call: the rules, then the scene and the request.

    Pure: the same inputs always yield the same strings. The player's request
    is included verbatim and last, where an instruction has the most weight;
    every empty section is omitted, exactly as the narrator's prompt does. With
    no active persona the protagonist section is absent and the rest of the
    scene is unchanged — the same rule the narrator's prompt follows.
    """
    scene_parts = [
        _section("Current scene:", inputs.narration),
        _section("World state:", _world_state_lines(inputs.world_state)),
        _beings_section(
            "Protagonist (the player's character):", (inputs.persona,) if inputs.persona else ()
        ),
        _beings_section("Characters:", inputs.characters),
        _section("The request from the player:", inputs.instruction),
    ]
    scene = "\n\n".join(part for part in scene_parts if part)
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": scene},
    ]


# --- The deterministic name pass -----------------------------------------------
#
# The criterion it implements: proper names never appear in the returned
# prompt. A model asked politely complies most of the time, which is not the
# same thing, so every known name is removed from the answer regardless of
# what the model was told.
#
# The pass is safe by construction, not by repetition: every appearance string
# is cleaned of every known name — its own included — once, before it is ever
# used as a replacement. No replacement value therefore contains a name, and a
# single substitution pass cannot introduce one. Re-running the scrub until a
# pass changes nothing would not even terminate: an appearance containing its
# own character's name would be found and substituted back forever.


_PUNCTUATION = ",.;:!?"


@dataclass(frozen=True)
class _NamePass:
    """The replacement map and the bare names, precompiled for one scrub."""

    replacements: tuple[tuple[re.Pattern[str], str], ...]
    bare: tuple[re.Pattern[str], ...]


def _word_pattern(name: str) -> re.Pattern[str]:
    # Whole words only: `\b` is what keeps a character named Ana from turning
    # "banana" into a description. Case-insensitive because a model answers in
    # any casing it likes. The optional possessive keeps "Zoe's" from leaving
    # a "'s" behind.
    return re.compile(rf"\b{re.escape(name)}\b(?:'s)?", re.IGNORECASE)


def _tidy(text: str) -> str:
    """The leftovers of a removal, pulled tight: doubled spaces collapsed,
    a space before punctuation closed up, punctuation that only dangled off
    the removed name dropped. Newlines survive — an appearance is a textarea
    and may legitimately be multi-line."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(rf"[ \t]+([{_PUNCTUATION}])", r"\1", text)
    text = re.sub(rf"([{_PUNCTUATION}])(?:[ \t]*[{_PUNCTUATION}])+", r"\1", text)
    text = re.sub(rf"^[ \t]*[{_PUNCTUATION}][ \t]*", "", text)
    return text.strip()


def _clean_description(appearance: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    """The appearance with every known name removed from it, its own included.

    Removal, not substitution: "Same sharp jaw as his sister Elena" keeps the
    jaw and the sister and loses the name. Tidying runs only when something
    was actually removed, so an appearance that never carried a name rides
    into the prompt byte for byte.
    """
    cleaned = appearance
    for pattern in patterns:
        cleaned = pattern.sub(" ", cleaned)
    if cleaned != appearance:
        cleaned = _tidy(cleaned)
    return cleaned


def _name_pass(inputs: ImagePromptInputs) -> _NamePass:
    """Names with an appearance become that appearance; the rest disappear.

    Every appearance is cleaned of every known name before it becomes a
    replacement value, so one substitution pass cannot introduce a name. A
    name whose appearance is blank — or becomes blank once its own name is
    out of it — has nothing to become: it is removed where it stands, name-
    locally, together with the punctuation left dangling around it. The
    scenario title has no appearance at all and is bare the same way. A blank
    name is skipped: the empty pattern would match between every character.
    """
    beings = [being for being in (*inputs.characters, inputs.persona) if being is not None]
    named = [(being.name.strip(), being.appearance.strip()) for being in beings]
    title = inputs.scenario_title.strip()
    if title:
        named.append((title, ""))
    named = [(name, appearance) for name, appearance in named if name]
    patterns = tuple(_word_pattern(name) for name, _ in named)
    replacements: list[tuple[re.Pattern[str], str]] = []
    bare: list[re.Pattern[str]] = []
    for (_name, appearance), pattern in zip(named, patterns, strict=True):
        cleaned = _clean_description(appearance, patterns) if appearance else ""
        if cleaned:
            replacements.append((pattern, cleaned))
        else:
            bare.append(pattern)
    return _NamePass(replacements=tuple(replacements), bare=tuple(bare))


def _has_words(text: str) -> bool:
    """Whether anything beyond punctuation and whitespace is left."""
    return re.search(r"\w", text) is not None


def _scrub_keyword(keyword: str, name_pass: _NamePass) -> str:
    """One comma-separated keyword, cleaned — or empty when nothing but
    punctuation is left of it. The removal of a bare name takes the name and
    the spacing or punctuation dangling around it, never the neighbouring
    details that happen to share the comma-segment."""
    cleaned = keyword
    for pattern, appearance in name_pass.replacements:
        cleaned = pattern.sub(appearance, cleaned)
    for pattern in name_pass.bare:
        cleaned = pattern.sub(" ", cleaned)
    if cleaned != keyword:
        cleaned = _tidy(cleaned)
    if not _has_words(cleaned):
        return ""
    return cleaned.strip()


def scrub_names(text: str, inputs: ImagePromptInputs) -> str:
    """The model's answer with every known proper name driven out.

    The answer is treated as the comma-separated keyword list the instruction
    asked for. A name that cannot become a description is removed on its own;
    a comma-segment is dropped only when nothing but punctuation or whitespace
    is left of it. A description substituted in may itself contain commas —
    that is fine for an image prompt, which is a bag of tags, not prose.
    """
    name_pass = _name_pass(inputs)
    keywords = [_scrub_keyword(keyword, name_pass) for keyword in text.split(",")]
    return ", ".join(keyword for keyword in keywords if keyword)


# --- Loaders -------------------------------------------------------------------


def load_inputs(con: sqlite3.Connection, party_id: str) -> ImagePromptInputs:
    """Read everything one composition needs, as its input type.

    Raises LookupError when the party does not exist; the route that calls
    this is the one that turns that into a 404. The player's request is not
    read here — it arrives in the request body, not the database.
    """
    party = con.execute(
        "SELECT scenario_id, world_state FROM instance WHERE id = ?", (party_id,)
    ).fetchone()
    if party is None:
        raise LookupError(f"Party {party_id!r} not found")
    scenario_row = con.execute(
        "SELECT title FROM scenario WHERE id = ?", (party["scenario_id"],)
    ).fetchone()
    narration_row = con.execute(
        "SELECT content FROM message "
        "WHERE instance_id = ? AND role = 'assistant' AND kind = 'text' "
        "ORDER BY id DESC LIMIT 1",
        (party_id,),
    ).fetchone()
    persona: Appearance | None = None
    persona_id = settings.get_active_persona_id(con)
    if persona_id is not None:
        persona_row = con.execute(
            "SELECT name, appearance FROM persona WHERE id = ?", (persona_id,)
        ).fetchone()
        if persona_row is not None:
            persona = Appearance(name=persona_row["name"], appearance=persona_row["appearance"])
    characters = tuple(
        Appearance(name=row["name"], appearance=row["appearance"])
        for row in con.execute(
            "SELECT name, appearance FROM character WHERE scenario_id = ? ORDER BY position",
            (party["scenario_id"],),
        ).fetchall()
    )
    return ImagePromptInputs(
        instruction="",
        narration=narration_row["content"] if narration_row is not None else "",
        world_state=summarizer.parse_world_state(party["world_state"]),
        persona=persona,
        characters=characters,
        scenario_title=scenario_row["title"] if scenario_row is not None else "",
    )
