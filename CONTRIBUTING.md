# Contributing

Private, single-developer project. These notes exist so a version of me six
months from now can pick the work back up without rereading the source.

## Requirements

- Python 3.14, installed and pinned by [uv](https://docs.astral.sh/uv/)
- Node 24 with [pnpm](https://pnpm.io/)
- [Ollama](https://ollama.com/) for chat, [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for images

## Setup

```bash
uv sync --extra dev
pnpm --dir web install
```

## Running

Two modes. Development keeps the front on Vite so edits appear instantly:

```bash
uv run uvicorn persona_studio.main:app --reload --port 8000   # terminal 1
pnpm --dir web dev                                            # terminal 2 → :5173
```

Vite proxies `/api` to port 8000, so open **http://localhost:5173**.

For everyday play, build the front once and let FastAPI serve it:

```bash
pnpm --dir web build
uv run uvicorn persona_studio.main:app --port 8000            # → http://localhost:8000
```

The database is created and migrated on startup. Nothing to run by hand.

## Checks

Run what CI runs, before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q          # exit code 5 is expected while there are no tests

pnpm --dir web lint
pnpm --dir web exec tsc -b --noEmit
pnpm --dir web build
```

## Workflow

1. Branch off `main`. Prefix by intent: `feat/`, `fix/`, `chore/`, `docs/`.
2. Commit in English. The message states **the problem being solved**, not the diff.
3. Open a pull request. GitHub cannot protect `main` here — branch protection
   and rulesets both require Pro on a private repository — so the guard is a
   local hook that refuses a direct push. Enable it once per clone:

   ```bash
   git config core.hooksPath .githooks
   ```

   It is client-side and `--no-verify` bypasses it. It stops the accident, not
   the decision.
4. Review the branch before merging. Reviewing your own code works better when
   you read the diff cold, from the pull request, not from the editor you wrote it in.
5. Squash or merge, then delete the branch.

## Pull requests

The body has two headings and nothing else:

```markdown
## Summary

- short bullet points, each a problem solved rather than a file touched

## Testing

- concrete checks with their outcome, or "Not run" and why
```

`.github/pull_request_template.md` already carries that shape, so a pull request
opened from the GitHub UI or by an agent starts from it. Its HTML comments are
guidance and are meant to be deleted, not filled in.

Keep the Conventional Commits prefix in the title — this repository uses them,
so `chore:`, `feat:` and `fix:` belong there. A title should be specific: the
problem, not the area it lives in.

This convention is borrowed from T3 Code, which reads a repository's template
when one exists — first match wins among `.github/pull_request_template.md`,
`.github/PULL_REQUEST_TEMPLATE.md`, then the same two names at the root and
under `docs/`, up to 8 000 bytes — and otherwise falls back to exactly these two
sections. Ours sits at the first path it looks for, so the tool and the template
agree instead of pulling in different directions.

## Ground rules

- **Code and identifiers in English.** Comments and documentation in English too.
- **Comments explain why, never what.** If a comment restates the line above it,
  delete the comment.
- **Images are files, never database rows.** See `docs/architecture.md`.
- **Every schema change is a new migration.** Never edit one already applied.
- **The narrator never receives image messages.** Image generation is out of band.
