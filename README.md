# Persona Studio

Local application for illustrated roleplay. A local LLM narrates and voices every
character while the player takes the protagonist; a local image pipeline draws
the scenes on demand. Nothing leaves the machine.

## Status

Playable. [V1](../../releases/tag/v1.0.0) closed the sixteen issues of the
[V1 — Playable core](../../milestone/1) milestone: scenarios and character
sheets, a player persona, a narrator whose prompt and history window are
explicit, streamed narration you can edit and regenerate without losing what
was there, a rolling summary that gives a long party its memory, and scene
images composed and generated through your own ComfyUI workflow.

What it is not: a server. One machine, one player, no accounts, nothing
leaving the host.

## Requirements

| | |
|---|---|
| [uv](https://docs.astral.sh/uv/) | installs and pins Python 3.14 |
| [pnpm](https://pnpm.io/) | Node 24 |
| [Ollama](https://ollama.com/) | narration — needs at least one chat model pulled |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | images — optional, chat works without it |

## Getting started

```bash
uv sync --extra dev
pnpm --dir web install
git config core.hooksPath .githooks
```

Then, for everyday use:

```bash
pnpm --dir web build
uv run uvicorn persona_studio.main:app --port 8000
```

and open <http://localhost:8000>. The database is created and migrated on
startup; there is nothing to run by hand.

For development, run the backend with `--reload` and `pnpm --dir web dev`
alongside it, then open the Vite port instead — it proxies `/api` through.
[`CONTRIBUTING.md`](CONTRIBUTING.md) has both commands in full.

## Layout

| Path | What it holds |
|---|---|
| `persona_studio/` | FastAPI application, database access, migrations |
| `web/` | React + TypeScript front end, built by Vite |
| `data/` | SQLite database and generated PNG files — never committed |
| `docs/` | Design documents |

## Documentation

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — setup, both run modes, the checks CI
  runs, and the branch workflow.
- [`docs/architecture.md`](docs/architecture.md) — the decisions the project
  rests on and the measurements behind them.

The split is deliberate. Commands change; the reasoning behind a decision does
not. The previous version kept both in one document that described behaviour,
and it was fifty days out of date by the time anyone read it.
