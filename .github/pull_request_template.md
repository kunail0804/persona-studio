## What this changes

<!-- One or two sentences. State the problem being solved, not just the diff. -->

## Why

<!-- The reasoning that is not obvious from the code. Link the issue: Closes #N -->

## Checks

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy`
- [ ] `pnpm --dir web lint && pnpm --dir web exec tsc -b --noEmit && pnpm --dir web build`
- [ ] Ran the app and used the change, not just the checks

## Anything left open

<!-- Known gaps, deliberate shortcuts, follow-up issues. Write "nothing" if there are none. -->
