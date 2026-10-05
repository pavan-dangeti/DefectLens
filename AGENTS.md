# DefectLens — working notes for agents

## Commands

| Task | Command |
|---|---|
| Install deps + lockfile | `uv sync` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Typecheck | `uv run mypy src/defectlens` |
| Tests + coverage gate (85%) | `uv run pytest --cov --cov-report=term-missing` |
| Build image | `docker build -t defectlens .` |
| Compose up | `docker compose up` |

## Conventions

- Single source of dependency truth is `pyproject.toml` / `uv.lock` — do not add `pip`
  installs to the Dockerfile beyond `uv sync --frozen`.
- Data and artefacts are never committed. CI fails if `git ls-files` lists them.
- The dataset licence is recorded in each `configs/datasets/*.yaml`, and ADR-0001 controls
  what ships. Do not commit anything derived from MVTec AD.
- Tests must not download datasets or reach the network; use `httpx.MockTransport` and
  `tmp_path`. Property tests use hypothesis.
- Every number that lands in the README must come from a committed script under `results/`
  (Phase 3 onward); Phase 0–1 READMEs carry no performance figures.
- Commit style: conventional commits. Solo contributor only; no bot authors.
