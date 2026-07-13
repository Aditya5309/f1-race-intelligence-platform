# Contributing

Thanks for your interest in this project. It's primarily a portfolio/
demonstration project, but genuine bug fixes, documentation improvements,
and well-scoped feature proposals are welcome.

## Before you start

For anything beyond a small fix, please open an issue first describing
what you want to change and why — this avoids wasted effort on a change
that doesn't fit the project's design (see "Design principles" below).

## Local setup

```bash
git clone https://github.com/Aditya5309/f1-race-intelligence-platform.git
cd f1-race-intelligence-platform
pip install -r requirements.txt   # editable install + dev tools (pytest, ruff, pip-audit)

pytest tests/               # should pass — 540 tests, none need data/
python -m ruff check .      # should be clean
python scripts/smoke.py     # end-to-end synthetic check, no data/ needed
```

`data/` (raw Ergast-format CSVs) is only needed if you're changing the
data pipeline or retraining a model — everything else (the API, the
dashboard, the test suite) works from the committed `artifacts/` tree
alone. See [docs/commands.md](docs/commands.md) for the full command
reference, including how to rebuild the dataset and retrain.

Docker is also supported if you'd rather not set up a local Python
environment at all — see [README.md's Docker section](README.md#9-docker).

## Repository Structure

```text
.devcontainer/       Codespaces configuration for the Streamlit Cloud deployment
.github/workflows/   ci.yml (lint, security scans, tests, coverage gate, smoke),
                     retrain.yml (scheduled ingestion + retraining), seed-data-cache.yml
.github/dependabot.yml   Weekly dependency-update pull requests (pip, GitHub Actions, Docker)
app/                 FastAPI service, settings, eight-page Streamlit dashboard
                     (views + shared components/charts/metadata modules)
artifacts/           COMMITTED runtime tree — frozen model bundle, features snapshot,
                     display-metadata CSVs, and the live-season tracking log; this is the
                     only thing a deployed instance needs beyond source code
config/              Shared hyperparameter configuration used by both manual and
                     automated training runs
data/                Raw / interim / processed training datasets (gitignored)
docker/              Dockerfile.api, Dockerfile.dashboard, and their per-image
                     dependency lists
docs/                User guide, API reference, command reference, ingestion/retraining
                     runbook, and screenshots
notebooks/           Exploratory analysis only — no business logic
scripts/             Data ingestion, model promotion, artifact export, the local
                     development launcher, and the end-to-end smoke test
src/data/            Loading, cleaning, validation, interim parquet builders
src/integration/     Join-only master dataset builder
src/pipelines/       Dataset build orchestration
src/features/        Modular feature groups, the feature pipeline, and feature metadata
src/models/          Splits, regulation eras, registry, training, evaluation, analysis,
                     calibration, prediction, live-season monitoring, and
                     frozen runtime-artifact export/load
tests/               540 pytest tests mirroring every implemented layer
docker-compose.yml           Production-shape service definitions
docker-compose.override.yml  Local development overrides (auto-merged)
Makefile             make lint / test / coverage / quality / smoke / audit / secrets / all
pyproject.toml       PEP 621 packaging — version, dependencies, Ruff and coverage config
requirements.txt     Installer shim (`-e .[dev]`); pins live in pyproject.toml
SECURITY.md          Vulnerability reporting policy
CONTRIBUTING.md       How to propose changes
.env.example         Template for every environment variable this platform reads
```

## Design principles

Please respect these when proposing a change — they're load-bearing, not
stylistic preferences:

- **Temporal discipline is non-negotiable.** Every feature must be
  computable using only information available before a race starts.
  Rolling windows shift before they roll; standings are lagged to the
  previous round; circuit history uses only prior visits. A pull request
  that introduces even a subtle leak of race-outcome information into a
  pre-race feature will be rejected regardless of how much it improves a
  metric — a large jump in accuracy from a new feature is a leakage red
  flag, not a result to celebrate.
- **The 2024 test set and the 2025–2026 forward holdout are spent
  resources.** Do not add code paths that evaluate against them outside
  the existing, deliberately guarded one-time test run.
- **The dashboard never imports model code.** `app/views/` talks to the
  API over HTTP only; if a view needs a new capability, it needs a new (or
  extended) API route, not a direct import of `src/`.
- **The deployed API and dashboard read only from the committed
  `artifacts/` tree** — a frozen model bundle and a features snapshot —
  never from a live MLflow server or the gitignored `data/` tree at
  request time. Don't reintroduce that coupling.
- **Registering a model in MLflow does not promote it.** Only
  `scripts/promote_model.py`'s gate is allowed to change
  `artifacts/serving/`, and only after its checks pass.
- **No authentication on the demonstration API is intentional**, not a gap
  to silently "fix" — see [SECURITY.md](SECURITY.md).

## Code style

- **Ruff** is the only enforced linter/formatter gate (`python -m ruff
  check .`, zero findings expected). The Ruff *formatter* is intentionally
  not adopted — don't reformat files wholesale in an unrelated change.
- Comments should explain **why**, not restate what the code already says.
  Prefer a clear variable/function name over a comment describing what a
  line does.
- One module per concern in `src/features/` (`qualifying.py`,
  `driver_form.py`, ...) rather than a monolith — follow the existing
  pattern for new feature groups.
- Business logic lives in `src/`; `notebooks/` and `app/views/` are
  read-only consumers of it, never the other way around.

## Tests

- Every new feature or bug fix needs a test. `tests/` mirrors `src/`
  module-for-module (`tests/test_<module>.py`).
- If you add a new feature-engineering transform, add a leakage test for
  it in `tests/test_features.py` — one test per identified leakage risk is
  the existing convention, not optional coverage.
- `src/` coverage is enforced in CI at a floor of 80% (currently ~95%
  measured) — a change that drops it below the floor will fail CI. Run
  `make coverage-gate` locally before opening a pull request.
- Tests must be hermetic: no writes into the real MLflow store or the
  checkout itself. Use `tests/conftest.py`'s helpers and an explicit
  `bundle_root`/tmp path for anything that touches MLflow or a serving
  bundle.

## Submitting a change

1. Fork the repository and create a branch from `main`.
2. Make your change, following the design principles and code style above.
3. Run the full local quality gate before opening a pull request:
   ```bash
   make quality   # lint + tests
   make smoke
   ```
4. Open a pull request describing *why* the change is needed, not just
   what it does. Every pull request runs the full CI pipeline (lint,
   secret scan, dependency vulnerability scan, tests with a coverage gate,
   smoke test) automatically — it must pass before merge.
5. Nothing in this repository auto-merges. Expect a human review, even for
   automated dependency-update pull requests.

## Reporting bugs

Open a GitHub issue with: what you expected, what actually happened, and
the smallest reproduction you can manage (a specific `race_id`/driver, a
specific command, or a failing test). For anything that looks like a
security issue rather than a functional bug, see
[SECURITY.md](SECURITY.md) instead of a public issue.
