"""Accès SQLite : connexion, réglages du moteur, et migrations de schéma.

Une connexion est ouverte par opération plutôt que partagée : sqlite3 interdit
de réutiliser une connexion entre threads, et FastAPI exécute les routes
synchrones dans un pool. L'ouverture coûte quelques microsecondes, le mode WAL
laisse les lecteurs travailler pendant qu'un écrivain écrit, donc rien ne
justifie de gérer un pool à la main.

Les images ne sont PAS dans la base : un PNG pèse plusieurs mégaoctets, et un
blob de cette taille coûte plus qu'il ne rapporte (sauvegarde tout-ou-rien,
espace non rendu sans VACUUM, lecture en mémoire au lieu d'un envoi direct).
La table `image` ne porte que les métadonnées ; le fichier vit dans data/images/.
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.environ.get("PS_DATA", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = DATA_DIR / "studio.db"
IMAGES_DIR = DATA_DIR / "images"

SCHEMA_VERSION = 1


def _configure(con: sqlite3.Connection) -> None:
    con.row_factory = sqlite3.Row
    # WAL : un écrivain n'empêche plus les lecteurs. C'est ce qui remplace les
    # verrous par partie de la version précédente.
    con.execute("PRAGMA journal_mode = WAL")
    # Sans ça SQLite ignore silencieusement les ON DELETE CASCADE.
    con.execute("PRAGMA foreign_keys = ON")
    # NORMAL suffit en WAL : on ne perd rien sauf coupure de courant brutale,
    # et on évite un fsync par transaction pendant le flux de chat.
    con.execute("PRAGMA synchronous = NORMAL")
    # Attendre plutôt qu'échouer si une écriture est déjà en cours.
    con.execute("PRAGMA busy_timeout = 5000")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Connexion transactionnelle : commit en sortie normale, rollback sur erreur."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        _configure(con)
        with con:
            yield con
    finally:
        con.close()


# --------------------------------------------------------------------------
# Migrations. Chaque entrée est le SQL qui fait passer de la version n-1 à n ;
# `PRAGMA user_version` retient où on en est. On n'édite jamais une migration
# déjà appliquée, on en ajoute une nouvelle.
# --------------------------------------------------------------------------

MIGRATIONS: list[str] = [
    # --- v1 : schéma initial ------------------------------------------------
    """
    CREATE TABLE setting (
        key    TEXT PRIMARY KEY,
        value  TEXT NOT NULL           -- JSON
    );

    CREATE TABLE image (
        id          TEXT PRIMARY KEY,  -- porte le nom du PNG : data/images/<id>.png
        prompt      TEXT NOT NULL DEFAULT '',   -- ce qui est parti à ComfyUI
        instruction TEXT NOT NULL DEFAULT '',   -- ce que l'utilisateur avait demandé
        seed        INTEGER,
        seconds     REAL,
        created_at  REAL NOT NULL
    );

    CREATE TABLE workflow (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        graph        TEXT NOT NULL,     -- le JSON du workflow, format API
        prompt_node  TEXT NOT NULL DEFAULT '',
        prompt_field TEXT NOT NULL DEFAULT '',
        seed_node    TEXT NOT NULL DEFAULT '',
        seed_field   TEXT NOT NULL DEFAULT '',
        created_at   REAL NOT NULL
    );

    CREATE TABLE persona (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        appearance  TEXT NOT NULL DEFAULT '',  -- seule source pour l'image
        traits      TEXT NOT NULL DEFAULT '',
        created_at  REAL NOT NULL
    );

    CREATE TABLE scenario (
        id           TEXT PRIMARY KEY,
        title        TEXT NOT NULL DEFAULT 'Sans titre',
        synopsis     TEXT NOT NULL DEFAULT '',
        world_rules  TEXT NOT NULL DEFAULT '',
        arcs         TEXT NOT NULL DEFAULT '',
        portrait_id  TEXT REFERENCES image(id) ON DELETE SET NULL,
        -- fond de discussion : un id d'image, ou la chaîne 'portrait' pour
        -- réutiliser la couverture, ou NULL pour aucun fond
        background   TEXT,
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL
    );

    CREATE TABLE character (
        id            INTEGER PRIMARY KEY,
        scenario_id   TEXT NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
        position      INTEGER NOT NULL,
        name          TEXT NOT NULL DEFAULT '',
        appearance    TEXT NOT NULL DEFAULT '',   -- physique seul : source des images
        personality   TEXT NOT NULL DEFAULT '',
        story         TEXT NOT NULL DEFAULT '',
        relationships TEXT NOT NULL DEFAULT '',
        secrets       TEXT NOT NULL DEFAULT '',   -- connus du narrateur seul
        portrait_id   TEXT REFERENCES image(id) ON DELETE SET NULL
    );
    CREATE INDEX character_by_scenario ON character(scenario_id, position);

    CREATE TABLE place (
        id          INTEGER PRIMARY KEY,
        scenario_id TEXT NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
        position    INTEGER NOT NULL,
        name        TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',   -- factuel et visuel
        atmosphere  TEXT NOT NULL DEFAULT ''    -- sensoriel, pour la narration
    );
    CREATE INDEX place_by_scenario ON place(scenario_id, position);

    CREATE TABLE item (
        id          INTEGER PRIMARY KEY,
        scenario_id TEXT NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
        position    INTEGER NOT NULL,
        name        TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX item_by_scenario ON item(scenario_id, position);

    CREATE TABLE lore (
        id          INTEGER PRIMARY KEY,
        scenario_id TEXT NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
        position    INTEGER NOT NULL,
        keywords    TEXT NOT NULL DEFAULT '[]',  -- JSON : liste de mots-clés
        text        TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX lore_by_scenario ON lore(scenario_id, position);

    CREATE TABLE instance (
        id           TEXT PRIMARY KEY,
        scenario_id  TEXT NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
        persona_id   TEXT REFERENCES persona(id) ON DELETE SET NULL,
        label        TEXT NOT NULL DEFAULT '',
        pinned       INTEGER NOT NULL DEFAULT 0,
        author_note  TEXT NOT NULL DEFAULT '',
        summary_text TEXT NOT NULL DEFAULT '',
        -- Frontière du résumé : l'identifiant du dernier message résumé, pas sa
        -- position. Supprimer un message ne décale donc plus rien.
        summary_upto INTEGER REFERENCES message(id) ON DELETE SET NULL,
        world_state  TEXT NOT NULL DEFAULT '{}',  -- JSON : lieu, présents, faits
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL
    );
    CREATE INDEX instance_by_scenario ON instance(scenario_id);
    CREATE INDEX instance_by_recency  ON instance(pinned DESC, updated_at DESC);

    CREATE TABLE message (
        -- rowid croissant : c'est lui qui donne l'ordre du récit. Aucune
        -- position à recalculer quand un message est supprimé.
        id          INTEGER PRIMARY KEY,
        instance_id TEXT NOT NULL REFERENCES instance(id) ON DELETE CASCADE,
        role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        kind        TEXT NOT NULL DEFAULT 'text' CHECK (kind IN ('text', 'image')),
        content     TEXT NOT NULL DEFAULT '',
        ooc         INTEGER NOT NULL DEFAULT 0,  -- tour hors-jeu, relayé en système
        ts          REAL NOT NULL,
        -- message d'image : la ligne image correspondante et l'état de sa génération
        image_id    TEXT REFERENCES image(id) ON DELETE SET NULL,
        status      TEXT CHECK (status IN ('pending', 'done', 'error')),
        gen_id      TEXT,
        started_at  REAL,
        error       TEXT
    );
    CREATE INDEX message_by_instance ON message(instance_id, id);
    CREATE INDEX message_pending     ON message(gen_id) WHERE gen_id IS NOT NULL;

    CREATE TABLE variant (
        id         INTEGER PRIMARY KEY,
        message_id INTEGER NOT NULL REFERENCES message(id) ON DELETE CASCADE,
        content    TEXT NOT NULL,
        created_at REAL NOT NULL,
        active     INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX variant_by_message ON variant(message_id, id);

    -- Recherche plein texte. `content=` fait de la table une vue indexée sur
    -- message : le texte n'est pas stocké deux fois, les triggers ci-dessous
    -- gardent l'index en phase.
    CREATE VIRTUAL TABLE message_fts USING fts5(
        content, content='message', content_rowid='id', tokenize='unicode61'
    );
    CREATE TRIGGER message_fts_insert AFTER INSERT ON message BEGIN
        INSERT INTO message_fts(rowid, content) VALUES (new.id, new.content);
    END;
    CREATE TRIGGER message_fts_delete AFTER DELETE ON message BEGIN
        INSERT INTO message_fts(message_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    END;
    CREATE TRIGGER message_fts_update AFTER UPDATE OF content ON message BEGIN
        INSERT INTO message_fts(message_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
        INSERT INTO message_fts(rowid, content) VALUES (new.id, new.content);
    END;
    """,
]


def migrate() -> int:
    """Applique les migrations manquantes. Retourne la version atteinte."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        _configure(con)
        current = con.execute("PRAGMA user_version").fetchone()[0]
        for version in range(current, len(MIGRATIONS)):
            with con:
                con.executescript(MIGRATIONS[version])
                # user_version n'accepte pas de paramètre lié.
                con.execute(f"PRAGMA user_version = {version + 1}")
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def image_path(image_id: str):
    return IMAGES_DIR / f"{image_id}.png"
