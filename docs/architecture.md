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
                       └──HTTP──▶ comfy-gateway :8190 ──▶ ComfyUI :8188
                                                     image generation
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

### Images go through comfy-gateway

This application used to keep its own copies of ComfyUI workflows: an
"Export (API)" file, a dropdown to pick the node and field the prompt went
into, and a client that submitted the graph, polled `/history` and cancelled
through ComfyUI's queue. CharCreator needed the same thing, and each app
grew its own version of it.

comfy-gateway is the one place that does it now. It takes a workflow straight
from the ComfyUI editor, exposes its settings and presets, finds the prompt
field by role, queues one job at a time, and cancels a job by id without
touching another render. This application names a template and a preset and
sends the prompt as the `prompt` role; it never builds or reads a graph.

What it costs: one more service to run for images. The chat does not need it,
the header pill says when it is down or when ComfyUI behind it is, and the
settings page says why a template cannot be used.

The `workflow` table of the old system is still in the schema and is no longer
read. It was left in place rather than dropped, so nothing the user imported
is deleted by an upgrade.

### The narrator and a render take turns on the graphics card

On 12 GB of VRAM the narration model and an image pipeline do not fit
together. Ollama's logs showed what happens when nobody lets go: the
narrator loaded with 0 of 49 layers on the GPU, RAM full, swap in use, until
a process was killed by hand. `gpu.py` decides who gets the card:

- before a render, every model Ollama holds is unloaded, and the code waits
  until `/api/ps` is empty;
- every job is sent with `free_after: "all"`, so the gateway empties
  ComfyUI (VRAM and RAM) once it is over;
- before every model call (turn, regeneration, opening, image prompt,
  summary), ComfyUI is emptied through the gateway, and the call is refused
  with a 409 while the gateway renders for any application. Background
  summaries wait instead of refusing.

The header pills open a menu with each service's memory and the buttons that
replace the commands: unload Ollama's models, empty ComfyUI, stop or restart
the gateway (in the alacritty window its shortcut opens). Ollama runs as a
system service this application has no right to stop, so it is unloaded,
not killed.

What it costs: the models reload after each switch, about 20 s for the
narrator on this machine, and about one second more per turn for the
gateway call.

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
