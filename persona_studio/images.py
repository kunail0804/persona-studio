"""Shared cleanup for image rows and files that nothing references any more.

An `image` row can be pointed to from three places: `scenario.portrait_id`,
`character.portrait_id` and `message.image_id`. Deleting a scenario cascades
away the rows that reference it (`character`, `instance`, `message`, ...), but
SQLite's `ON DELETE CASCADE` only removes rows — it never touches the `image`
row itself or the PNG file on disk, since nothing in the schema points the
other way. This module is the part that does.
"""

from __future__ import annotations

import sqlite3

from . import db


def delete_orphan_images(con: sqlite3.Connection) -> list[str]:
    """Delete every `image` row no longer referenced anywhere.

    Must run inside the same transaction as the change that may have orphaned
    them, so the query sees the post-cascade state. Returns the deleted ids;
    the caller unlinks their files once the transaction has committed, so a
    rolled-back transaction never loses a file whose row survived.
    """
    orphan_ids = [
        row["id"]
        for row in con.execute(
            """
            SELECT id FROM image
            WHERE id NOT IN (SELECT portrait_id FROM scenario WHERE portrait_id IS NOT NULL)
              AND id NOT IN (SELECT portrait_id FROM character WHERE portrait_id IS NOT NULL)
              AND id NOT IN (SELECT image_id FROM message WHERE image_id IS NOT NULL)
            """
        ).fetchall()
    ]
    if orphan_ids:
        placeholders = ", ".join("?" for _ in orphan_ids)
        con.execute(f"DELETE FROM image WHERE id IN ({placeholders})", orphan_ids)
    return orphan_ids


def unlink_images(image_ids: list[str]) -> None:
    """Remove the PNG file for each id. A file already missing is not an error."""
    for image_id in image_ids:
        db.image_path(image_id).unlink(missing_ok=True)
