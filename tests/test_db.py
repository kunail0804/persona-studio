"""Migrations: applying one must be all-or-nothing.

`db.migrate()` is the only thing that runs at startup before anything else, so a
migration that fails half-way is the one failure the application cannot recover
from on its own.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from persona_studio import db


def test_a_failing_migration_leaves_the_database_untouched(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A migration that fails part-way must roll back the statements that passed.

    Without this, `user_version` stays behind while some of the migration's
    tables already exist, and every later startup dies on "table already
    exists" — an application that can never start again.
    """
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "studio.db")
    monkeypatch.setattr(db, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(
        db,
        "MIGRATIONS",
        ["CREATE TABLE first (x INTEGER); CREATE TABLE second (this is not valid sql);"],
    )

    with pytest.raises(sqlite3.OperationalError):
        db.migrate()

    con = sqlite3.connect(tmp_path / "studio.db")
    try:
        tables = [
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        assert tables == [], f"a failed migration left tables behind: {tables}"
        assert con.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        con.close()


def test_a_failed_migration_can_be_retried_after_the_fault_is_fixed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The retry is the point of rolling back: a corrected migration must apply."""
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "studio.db")
    monkeypatch.setattr(db, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(
        db, "MIGRATIONS", ["CREATE TABLE first (x INTEGER); CREATE TABLE second (broken;"]
    )
    with pytest.raises(sqlite3.OperationalError):
        db.migrate()

    monkeypatch.setattr(
        db, "MIGRATIONS", ["CREATE TABLE first (x INTEGER); CREATE TABLE second (y INTEGER);"]
    )
    assert db.migrate() == 1

    con = sqlite3.connect(tmp_path / "studio.db")
    try:
        tables = sorted(
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        )
        assert tables == ["first", "second"]
    finally:
        con.close()


def test_the_real_migrations_still_apply_and_carry_the_trigger_bodies(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The v1 migration holds FTS triggers whose bodies contain semicolons.

    Naive statement splitting would cut those trigger bodies in half, so this
    pins that the real schema still arrives whole.
    """
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "studio.db")
    monkeypatch.setattr(db, "IMAGES_DIR", tmp_path / "images")

    assert db.migrate() == db.SCHEMA_VERSION

    con = sqlite3.connect(tmp_path / "studio.db")
    try:
        triggers = sorted(
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        )
        assert triggers == ["message_fts_delete", "message_fts_insert", "message_fts_update"]
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            "INSERT INTO scenario (id, title, created_at, updated_at) VALUES ('s', 't', 0, 0)"
        )
        con.execute(
            "INSERT INTO instance (id, scenario_id, created_at, updated_at) VALUES ('i', 's', 0, 0)"
        )
        con.execute(
            "INSERT INTO message (instance_id, role, content, ts) "
            "VALUES ('i', 'user', 'la cathedrale engloutie', 0)"
        )
        con.commit()
        hit = con.execute(
            "SELECT rowid FROM message_fts WHERE message_fts MATCH 'cathedrale'"
        ).fetchone()
        assert hit is not None, "the FTS trigger did not fire, so its body was truncated"
    finally:
        con.close()
