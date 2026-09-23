"""The world a scenario describes: its places, its lorebook, and who plays it.

V1.3. The lorebook is the part worth the most tests, because its whole value
is a selection rule: an entry reaches the narrator only when the story is
touching it, and every way that rule can be wrong is a way the narrator either
forgets the world or drowns in it.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from persona_studio import db, settings
from persona_studio.narrator import (
    HistoryMessage,
    PromptLore,
    PromptPlace,
    PromptScenario,
    build_chat_messages,
    build_system_prompt,
    load_party_persona,
    load_scenario,
    parse_keywords,
    select_lore,
)


@pytest.fixture(autouse=True)
def _no_default_persona_left_behind(client: TestClient) -> Iterator[None]:
    """The default persona is a setting in the database every test file
    shares. These tests move it around; none may leave it pointing somewhere
    for a later file to inherit a protagonist it never asked for."""
    yield
    with db.connect() as con:
        settings.set_active_persona_id(con, None)


SCENARIO = PromptScenario(
    title="La Cité Noyée", synopsis="", world_rules="", arcs="", characters=()
)


def _said(*texts: str) -> list[HistoryMessage]:
    """A history whose messages say these things, oldest first."""
    return [
        HistoryMessage(id=i, role="user" if i % 2 else "assistant", content=text, ooc=False)
        for i, text in enumerate(texts, start=1)
    ]


# --- The lorebook's selection rule (pure) ----------------------------------------


def test_an_entry_enters_when_the_story_uses_its_keyword() -> None:
    order = PromptLore(keywords=("Ordre",), text="L'Ordre tient le port depuis cent ans.")
    selected = select_lore([order], _said("Un membre de l'ordre approche."))
    assert selected == [order], "matching must ignore case"


def test_an_entry_stays_out_when_the_story_does_not_touch_it() -> None:
    """The reason the lorebook exists: a world can hold facts the current
    scene does not need, and not pay for them in every prompt."""
    order = PromptLore(keywords=("Ordre",), text="L'Ordre tient le port.")
    assert select_lore([order], _said("Le quai est désert.")) == []


def test_a_keyword_matches_whole_words_only() -> None:
    """A character called Ana must not trigger on "banana"."""
    ana = PromptLore(keywords=("Ana",), text="Ana est la fille du passeur.")
    assert select_lore([ana], _said("Il épluche une banana.")) == []
    assert select_lore([ana], _said("Ana attend sur le ponton.")) == [ana]


def test_a_keyword_ending_in_punctuation_still_matches() -> None:
    """`\\b` needs a word character at the edge; the lookarounds do not, so a
    keyword like "C++" or "M." is not silently unmatchable."""
    entry = PromptLore(keywords=("M.",), text="M. est l'informateur.")
    assert select_lore([entry], _said("M. a laissé un message.")) == [entry]


def test_only_the_most_recent_messages_are_scanned() -> None:
    """An entry mentioned long ago must leave the prompt once the story has
    moved on, or the lorebook fills the window with yesterday's scene."""
    order = PromptLore(keywords=("Ordre",), text="L'Ordre tient le port.")
    old_mention = _said("L'Ordre surveille.", *(f"tour {i}" for i in range(10)))
    assert select_lore([order], old_mention, scan=10) == []
    assert select_lore([order], old_mention, scan=11) == [order]


def test_the_number_of_entries_is_capped_in_scenario_order() -> None:
    entries = [PromptLore(keywords=(f"mot{i}",), text=f"Fait {i}.") for i in range(12)]
    story = _said(" ".join(f"mot{i}" for i in range(12)))

    selected = select_lore(entries, story, limit=8)

    assert selected == entries[:8], "the cap keeps the first entries of the scenario, in order"


def test_an_entry_without_text_or_keywords_never_enters() -> None:
    blank_text = PromptLore(keywords=("port",), text="   ")
    no_keyword = PromptLore(keywords=("", "  "), text="Jamais injecté.")
    assert select_lore([blank_text, no_keyword], _said("Le port dort.")) == []


def test_an_out_of_game_turn_can_trigger_an_entry() -> None:
    """A player asking the narrator about the Order is talking about it."""
    order = PromptLore(keywords=("Ordre",), text="L'Ordre tient le port.")
    history = [
        HistoryMessage(id=1, role="user", content="Rappelle-moi qui est l'Ordre ?", ooc=True)
    ]
    assert select_lore([order], history) == [order]


def test_no_history_selects_nothing() -> None:
    order = PromptLore(keywords=("Ordre",), text="L'Ordre tient le port.")
    assert select_lore([order], []) == []


def test_malformed_stored_keywords_are_no_keywords_at_all() -> None:
    """A hand-edited or half-written value must never break a turn."""
    assert parse_keywords("not json") == ()
    assert parse_keywords('{"a": 1}') == ()
    assert parse_keywords('["port", 3, null, "quai"]') == ("port", "quai")


# --- What the narrator receives -------------------------------------------------


def test_selected_lore_reaches_the_system_prompt_and_the_rest_does_not() -> None:
    scenario = PromptScenario(
        title="La Cité Noyée",
        synopsis="",
        world_rules="",
        arcs="",
        characters=(),
        lore=(
            PromptLore(keywords=("Ordre",), text="L'Ordre tient le port."),
            PromptLore(keywords=("cathédrale",), text="La cathédrale est engloutie."),
        ),
    )

    system = build_chat_messages(scenario, None, _said("Un homme de l'Ordre."))[0]["content"]

    assert "L'Ordre tient le port." in system
    assert "La cathédrale est engloutie." not in system


def test_places_ride_in_every_prompt_with_their_atmosphere() -> None:
    """Unlike lore, places are always sent: the narrator must know a place
    exists to take the story there."""
    scenario = PromptScenario(
        title="La Cité Noyée",
        synopsis="",
        world_rules="",
        arcs="",
        characters=(),
        places=(
            PromptPlace(
                name="La Cathédrale",
                description="Une nef à demi immergée.",
                atmosphere="Odeur de sel, écho de gouttes.",
            ),
        ),
    )

    system = build_system_prompt(scenario, None)

    assert "Places:" in system
    assert "Name: La Cathédrale" in system
    assert "Atmosphere: Odeur de sel, écho de gouttes." in system


def test_a_scenario_without_places_or_lore_prompts_exactly_as_before() -> None:
    """V1.3 must not change a single byte for a scenario that uses none of it."""
    assert "Places:" not in build_system_prompt(SCENARIO, None)
    assert build_chat_messages(SCENARIO, None, _said("Bonjour."))[0]["content"] == (
        build_system_prompt(SCENARIO, None)
    )


# --- The HTTP surface ------------------------------------------------------------


@pytest.fixture
def scenario_id(client: TestClient) -> str:
    response = client.post("/api/scenarios", json={"title": "La Cité Noyée", "synopsis": ""})
    assert response.status_code == 201
    return response.json()["id"]


def test_places_round_trip_in_order(client: TestClient, scenario_id: str) -> None:
    for name in ("Le Port", "La Cathédrale"):
        assert (
            client.post(f"/api/scenarios/{scenario_id}/places", json={"name": name}).status_code
            == 201
        )
    listed = client.get(f"/api/scenarios/{scenario_id}/places").json()
    assert [p["name"] for p in listed] == ["Le Port", "La Cathédrale"]
    assert [p["position"] for p in listed] == [0, 1]


def test_a_place_patch_writes_only_what_it_carried(client: TestClient, scenario_id: str) -> None:
    place = client.post(
        f"/api/scenarios/{scenario_id}/places",
        json={"name": "Le Port", "description": "Des quais de pierre.", "atmosphere": "Le sel."},
    ).json()

    updated = client.patch(
        f"/api/scenarios/{scenario_id}/places/{place['id']}", json={"atmosphere": "La brume."}
    ).json()

    assert updated["atmosphere"] == "La brume."
    assert updated["description"] == "Des quais de pierre.", "an omitted field was wiped"


def test_lore_keywords_are_normalised(client: TestClient, scenario_id: str) -> None:
    """Matching ignores case, so "Ordre" and "ordre" are one keyword, and a
    blank one could never match anything."""
    entry = client.post(
        f"/api/scenarios/{scenario_id}/lore",
        json={"keywords": [" Ordre ", "ordre", "", "Chevaliers"], "text": "L'Ordre tient le port."},
    ).json()
    assert entry["keywords"] == ["Ordre", "Chevaliers"]


def test_lore_entries_reach_the_narrator_through_the_database(
    client: TestClient, scenario_id: str
) -> None:
    """End to end on the loader: what the API stored is what the turn reads."""
    client.post(
        f"/api/scenarios/{scenario_id}/lore",
        json={"keywords": ["Ordre"], "text": "L'Ordre tient le port."},
    )
    client.post(f"/api/scenarios/{scenario_id}/places", json={"name": "Le Port"})
    with db.connect() as con:
        scenario = load_scenario(con, scenario_id)
    assert [entry.text for entry in scenario.lore] == ["L'Ordre tient le port."]
    assert scenario.lore[0].keywords == ("Ordre",)
    assert [place.name for place in scenario.places] == ["Le Port"]


def test_deleting_a_scenario_takes_its_places_and_lore(
    client: TestClient, scenario_id: str
) -> None:
    client.post(f"/api/scenarios/{scenario_id}/places", json={"name": "Le Port"})
    client.post(f"/api/scenarios/{scenario_id}/lore", json={"keywords": ["x"], "text": "y"})

    assert client.delete(f"/api/scenarios/{scenario_id}").status_code == 204

    with db.connect() as con:
        for table in ("place", "lore"):
            left = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE scenario_id = ?", (scenario_id,)
            ).fetchone()[0]
            assert left == 0, f"{table} rows outlived their scenario"


@pytest.mark.parametrize("kind", ["places", "lore"])
def test_unknown_scenario_and_entry_are_404(
    client: TestClient, scenario_id: str, kind: str
) -> None:
    assert client.get(f"/api/scenarios/nope/{kind}").status_code == 404
    assert client.patch(f"/api/scenarios/{scenario_id}/{kind}/999999", json={}).status_code == 404
    assert client.delete(f"/api/scenarios/{scenario_id}/{kind}/999999").status_code == 404


@pytest.mark.parametrize("kind", ["places", "lore"])
def test_an_unknown_key_is_refused(client: TestClient, scenario_id: str, kind: str) -> None:
    assert (
        client.post(f"/api/scenarios/{scenario_id}/{kind}", json={"nmae": "x"}).status_code == 422
    )


# --- One persona per party -------------------------------------------------------


def _persona(client: TestClient, name: str) -> str:
    response = client.post("/api/personas", json={"name": name, "appearance": f"{name}, grand."})
    assert response.status_code == 201
    return response.json()["id"]


def _party(client: TestClient, scenario_id: str, monkeypatch: Any) -> dict[str, Any]:
    from persona_studio.routes import parties as parties_routes

    with db.connect() as con:
        settings.set_llm_model(con, "test-model")
    monkeypatch.setattr(parties_routes.ollama, "chat", lambda *a, **k: "Le port s'ouvre.")
    response = client.post(f"/api/scenarios/{scenario_id}/parties", json={"label": ""})
    assert response.status_code == 201
    return response.json()


def test_a_new_party_starts_as_the_default_persona(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    ash = _persona(client, "Ash")
    client.put("/api/personas/active", json={"id": ash})

    party = _party(client, scenario_id, monkeypatch)

    assert party["persona_id"] == ash


def test_changing_the_default_does_not_touch_a_party_in_progress(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    """The whole point: two parties in progress, two protagonists."""
    ash, bea = _persona(client, "Ash"), _persona(client, "Bea")
    client.put("/api/personas/active", json={"id": ash})
    first = _party(client, scenario_id, monkeypatch)

    client.put("/api/personas/active", json={"id": bea})
    second = _party(client, scenario_id, monkeypatch)

    with db.connect() as con:
        assert load_party_persona(con, first["id"]).name == "Ash"  # type: ignore[union-attr]
        assert load_party_persona(con, second["id"]).name == "Bea"  # type: ignore[union-attr]


def test_a_party_persona_can_be_changed_and_cleared(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)
    bea = _persona(client, "Bea")

    changed = client.put(f"/api/parties/{party['id']}/persona", json={"persona_id": bea})
    assert changed.status_code == 200
    assert changed.json()["persona_id"] == bea

    cleared = client.put(f"/api/parties/{party['id']}/persona", json={"persona_id": None})
    assert cleared.json()["persona_id"] is None


def test_an_unknown_persona_is_refused(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    party = _party(client, scenario_id, monkeypatch)
    response = client.put(f"/api/parties/{party['id']}/persona", json={"persona_id": "nope"})
    assert response.status_code == 404


def test_deleting_a_persona_leaves_its_parties_playable_without_one(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    """`ON DELETE SET NULL`: the party loses its protagonist, not itself."""
    ash = _persona(client, "Ash")
    client.put("/api/personas/active", json={"id": ash})
    party = _party(client, scenario_id, monkeypatch)

    assert client.delete(f"/api/personas/{ash}").status_code == 204

    detail = client.get(f"/api/parties/{party['id']}")
    assert detail.status_code == 200
    assert detail.json()["persona_id"] is None


def test_the_turn_prompt_carries_the_party_persona_not_the_default(
    client: TestClient, scenario_id: str, monkeypatch: Any
) -> None:
    ash, bea = _persona(client, "Ash"), _persona(client, "Bea")
    client.put("/api/personas/active", json={"id": ash})
    party = _party(client, scenario_id, monkeypatch)
    client.put("/api/personas/active", json={"id": bea})

    blocks = client.get(f"/api/parties/{party['id']}/prompt").json()["blocks"]

    assert "Name: Ash" in blocks[0]["content"]
    assert "Name: Bea" not in blocks[0]["content"]


# --- The v4 migration, on a database that predates it ---------------------------


@pytest.fixture
def v3_database(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[sqlite3.Connection]:
    """A fresh database migrated to v3 only, ready to be moved to v4."""
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "studio.db")
    monkeypatch.setattr(db, "IMAGES_DIR", tmp_path / "images")
    full = list(db.MIGRATIONS)
    monkeypatch.setattr(db, "MIGRATIONS", full[:3])
    assert db.migrate() == 3
    monkeypatch.setattr(db, "MIGRATIONS", full)
    con = sqlite3.connect(tmp_path / "studio.db")
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("INSERT INTO scenario (id, title, created_at, updated_at) VALUES ('s', 't', 0, 0)")
    con.execute("INSERT INTO persona (id, name, created_at) VALUES ('ash', 'Ash', 0)")
    con.execute(
        "INSERT INTO instance (id, scenario_id, created_at, updated_at) VALUES ('p', 's', 0, 0)"
    )
    con.commit()
    try:
        yield con
    finally:
        con.close()


def test_v4_gives_existing_parties_the_persona_they_were_already_played_as(
    v3_database: sqlite3.Connection,
) -> None:
    v3_database.execute(
        "INSERT INTO setting (key, value) VALUES ('persona.active_id', ?)", (json.dumps("ash"),)
    )
    v3_database.commit()

    assert db.migrate() == 4

    row = v3_database.execute("SELECT persona_id FROM instance WHERE id = 'p'").fetchone()
    assert row["persona_id"] == "ash"


def test_v4_leaves_a_party_without_persona_when_the_setting_dangles(
    v3_database: sqlite3.Connection,
) -> None:
    """A setting pointing at a deleted persona must not fail the foreign key
    and leave the application unable to start."""
    v3_database.execute(
        "INSERT INTO setting (key, value) VALUES ('persona.active_id', ?)",
        (json.dumps(uuid.uuid4().hex),),
    )
    v3_database.commit()

    assert db.migrate() == 4

    row = v3_database.execute("SELECT persona_id FROM instance WHERE id = 'p'").fetchone()
    assert row["persona_id"] is None
