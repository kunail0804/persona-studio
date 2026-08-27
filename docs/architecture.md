# Architecture

What Persona Studio is, and the decisions it rests on. Behaviour changes; the
reasoning below is what stays true, so this document records **why**, not what.

## What it is

A local, single-user application for illustrated roleplay. A local LLM narrates
and voices every character while the player takes the protagonist; a local
image pipeline draws the scenes on demand. Nothing leaves the machine.

## Shape

```
browser  ──HTTP──▶  FastAPI  ──▶  SQLite  (data/studio.db)
                       │     ──▶  PNG files (data/images/)
                       ├──HTTP──▶ Ollama    :11434   narration, summaries, prompts
                       └──HTTP──▶ ComfyUI   :8188    image generation
```

In development Vite serves the front on `:5173` and proxies `/api`. In use,
FastAPI serves `web/dist` itself, so there is one process to start.

## Decisions

### The backend is Python

Not a preference. [ComfyUI-Scene-Compiler](https://github.com/kunail0804/ComfyUI-Scene-Compiler)
is a Python package — 1 486 `.py` files, zero in any other language — and the
plan is to import it directly to compile image prompts into tags whose existence
is verified. Measured outside ComfyUI, it resolves a scene in 3.5 ms, fast
enough to call inline in a request.

Any other backend language forces the alternative: wiring the analyzer into the
ComfyUI graph. That costs a second LLM call per image and moves the final prompt
out of the application, where it can no longer be shown or edited.

### The frontend is TypeScript, React, Vite, Tailwind

The previous version was 1 773 lines of untyped vanilla JavaScript, untestable
and — after a sentinel character was written raw into the file — unreadable by
`git` itself. Typing the client is the correction.

React over the alternatives because it is already in use in `ck3-mod-manager`
and `CKModManager`. A better framework that has to be learned first is worse
than a good one already known.

### The store is SQLite, and images are not in it

Storage was files-on-disk before. Measured on this machine at 60 parties ×
2 000 messages (~235 MB of JSON, a 285 MB database):

| Operation | JSON files | SQLite |
|---|---|---|
| List the home page | 274 ms | 4.7 ms |
| Search a word that is absent | 386 ms | 0.0 ms |
| Append a message | 70 ms (btrfs, fsync) | 15.6 ms |

Restructuring the files — a small header per party, messages in JSONL — beats
SQLite on listing and writing. It does not solve search: scanning every byte
costs 254 ms per keystroke and grows linearly forever. FTS5 answers instantly,
and that is the deciding difference.

Two further gains that do not show in a table. WAL mode replaces the per-party
`threading.Lock` machinery the previous version needed for concurrent writers.
And ordering the story by message rowid removes a bug class: deleting a message
used to shift every later index, so the rolling-summary frontier had to be
decremented by hand.

**Images stay on disk.** A generated PNG averaged 6.5 MB in the previous
version. As blobs they would make every backup all-or-nothing, take back their
space only under `VACUUM`, and be served through memory instead of straight from
the filesystem. The `image` table carries the metadata; the file is
`data/images/<id>.png`.

### Migrations are explicit

`PRAGMA user_version` records the schema version; `db.MIGRATIONS` is the list of
scripts to reach each one, applied at startup. A migration already applied is
never edited — a new one is added.

## Not decided yet

- Whether the LLM client stays Ollama-only or grows an OpenAI-compatible path.
- How character appearance stays consistent between two generated images. This
  is the largest known visual gap, and its cost is model work, not code.
- Whether the rolling summary and world state become editable. In the previous
  version they were read-only, so a bad summary poisoned every later turn.
