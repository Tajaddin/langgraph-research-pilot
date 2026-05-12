# Improvement plan

## What changes

- Rewrite README prose in spartan voice. Strip em dashes, semicolons, and banned filler words. Keep every section (hero metrics, accuracy table, architecture, install, usage, demo, configuration, tests, reproduce, deploy, caveats, layout). Keep file-tree comments. The reproduction story is the value, do not delete it.
- Pin every dependency to an exact version in `pyproject.toml` and `requirements.txt`. The two files must match.
- Add `.github/workflows/ci.yml` running pytest on Python 3.10 / 3.11 / 3.12 plus a ruff lint pass.
- Rename "Limitations" -> "Caveats" so every polished repo uses the same heading.
- Strip em dashes from `app.py`, `RESULTS.md`, and source docstrings. Leave the em dashes inside the LLM prompt strings in `nodes.py` alone, since the committed bench numbers depend on the exact prompt text.
- Add a one-line note in the bench section that the agent-vs-baseline numbers come from `bench/results_haiku_path2.json` so a reader knows which file backs each row.

## Why

- README has em dashes and banned filler words. Style rules forbid them.
- Deps are unpinned (`>=` ranges). CI reproducibility breaks on minor releases of langgraph and langchain-core, which release frequently.
- No CI workflow. The README claims 35 passing tests with no badge proving it.
- Heading consistency lets a reader scan multiple repos without re-orienting.

## Out of scope

- Adding new nodes, retrievers, or models. The agent shape is the value.
- Re-running the bench. Reuse the committed `results_haiku_path2.json`. The polish does not invalidate the numbers.
- Constrained decoding for Llama. Already a caveat. Adding it would be scope expansion.
