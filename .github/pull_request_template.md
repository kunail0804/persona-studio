## Summary

<!--
Short bullet points. Each states a problem solved, not a file touched.
Reasoning that is not obvious from the diff belongs here — that is the part
worth reading in six months.
Put known gaps, deliberate shortcuts and follow-ups in the last bullets.
Link the issue: Closes #N
-->

-

## Testing

<!--
Concrete checks with their outcome. Write "Not run" and why, rather than
leaving a check implied.
-->

- `uv run ruff check . && uv run ruff format --check . && uv run mypy` —
- `uv run pytest -q` —
- `pnpm --dir web lint && pnpm --dir web exec tsc -b --noEmit && pnpm --dir web build` —
- Ran the app and used the change —
