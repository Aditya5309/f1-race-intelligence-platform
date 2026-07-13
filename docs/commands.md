# Command Reference

A complete, copy-pasteable reference for working with this repository —
setup, testing, linting, security scanning, Docker, CI, data operations,
and release commands in one place.

## Setup

```bash
pip install -r requirements.txt   # shim for `-e .[dev]`: runtime + pytest/ruff/notebook/pip-audit
pip install -e .                  # runtime-only alternative
```

Requires `data/` (Ergast-format CSVs, gitignored) only if you intend to
rebuild the datasets or retrain a model — obtained manually, there is no
bootstrap/download script. Serving from the committed `artifacts/` tree
needs nothing from `data/` at all.

## Testing

```bash
pytest tests/
pytest tests/ --cov=src --cov=app --cov-report=term-missing --cov-report=html
make test           # or: make coverage
```

540 tests. Four skip cleanly when `data/` is absent (real-data checks in
`test_features.py`/`test_splits.py`) — expected on a fresh clone or CI,
not a failure.

`src/` coverage is 95%, enforced in CI at a floor of 80%:

```bash
python -m coverage report --include="src/*" --fail-under=80
make coverage-gate   # = make coverage, then the check above
```

The combined `src/`+`app/` coverage figure (reported in the commands
above, ~58%) is intentionally *not* gated — every Streamlit view module
measures 0% by design (they're exercised by the smoke test's headless
dashboard run and by manual QA, never by `pytest` itself), so gating the
combined number would dilute every time a new dashboard page is added.

## Linting

```bash
python -m ruff check .
make lint
```

Rule set: `E4 E7 E9 F W I B UP` (`[tool.ruff]` in `pyproject.toml`). Zero
findings expected; a small number of exemptions are reason-commented
directly in `pyproject.toml`. The Ruff formatter is intentionally not
adopted.

## Security scanning

```bash
python -m pip_audit          # dependency vulnerability scan (report-only in CI)
make audit

gitleaks detect --source . -v   # secret scan — requires the gitleaks binary
                                 # on PATH (not a pip package):
                                 # https://github.com/gitleaks/gitleaks/releases
make secrets
```

CI runs the equivalent of both on every push and pull request — secret
scanning is a hard gate, dependency scanning is report-only (see
[SECURITY.md](../SECURITY.md) for the current policy and known-issue
tracking).

## Smoke test

```bash
python scripts/smoke.py
make smoke
```

A synthetic, service-free end-to-end check: configuration load → MLflow
train/register + bundle export → frozen-bundle load → prediction contract
→ in-process FastAPI health/prediction (both the versioned and legacy
paths) → headless dashboard run. No `data/`, no open ports, no external
services.

## Full quality gate

```bash
make quality   # lint + tests
make all        # quality + smoke
```

## Building the data pipeline (optional — only for retraining)

Idempotent; run in order after any source CSV changes:

```bash
python -m src.data.build_interim --target all
python -m src.pipelines.build_dataset
python -m src.features.pipeline
```

One-time enrichment backfills (only needed once per machine/checkout, not
part of the regular rebuild cycle above):

```bash
python scripts/backfill_weather.py               # historical race-weekend weather
python scripts/backfill_weather.py --dry-run      # fetch + report, no write

python scripts/backfill_circuit_layouts.py        # circuit track-outline geometry
python scripts/backfill_circuit_layouts.py --dry-run
```

## Training and registration

```bash
python -m src.models.train                             # full model zoo comparison
python -m src.models.train --model logreg --tune        # randomized hyperparameter search
python -m src.models.analysis --model logreg --timing   # SHAP / permutation importance / timing
mlflow ui                                                # browse experiments

# Register a candidate — this ALSO immediately overwrites the live serving
# bundle, unchecked. Fine for dev iteration; not the sanctioned promotion path.
python -m src.models.train --model logreg --register Staging --calibrate \
    --params-file config/registered_model_params.json

# Score one historical race directly from the committed runtime artifacts
python -m src.models.predict --race-id 1120
```

## Promotion — the only sanctioned path to changing what's served

```bash
python scripts/promote_model.py --alias Staging
python scripts/promote_model.py --alias Staging --version 3
python scripts/promote_model.py --alias Staging --force-baseline   # bootstrap only
```

Loads an already-registered candidate, checks its model class, smoke-tests
it against real races, and refuses to promote if accuracy regresses
against the currently-served bundle's own recorded metrics. Only on
success does it touch `artifacts/serving/`.

## Data ingestion and the automated retrain cycle

```bash
# Ingest newly-completed race weekends from the upstream data source
python scripts/ingest_jolpica.py                       # all missing completed races
python scripts/ingest_jolpica.py --dry-run             # fetch + report, no write
python scripts/ingest_jolpica.py --year 2026 --round 7 # one race only

# The atomic ingest -> rebuild -> track -> refresh -> register orchestrator
python scripts/refresh_and_freeze.py                    # manual mode: exports immediately
python scripts/refresh_and_freeze.py --automated         # registers only; promote_model.py
                                                          # is the only thing that then
                                                          # changes what's served
python scripts/refresh_and_freeze.py --skip-ingest       # rebuild/freeze only, data/ as-is

# Continuous scoring of the served model against the live season
python -m src.models.season_tracking
python -m src.models.season_tracking --alias Production

# Refresh the runtime features snapshot independently of promotion
python scripts/refresh_features_snapshot.py

# Freeze data/'s display CSVs into the committed artifacts/display/ tree
python scripts/export_display_data.py
```

A scheduled GitHub Actions workflow (`.github/workflows/retrain.yml`) runs
this cycle weekly and on manual dispatch, opening pull requests rather
than merging automatically. See
[docs/retrain_workflow_setup.md](retrain_workflow_setup.md) for the
one-time setup this workflow needs before its first run.

## Local development

```bash
python scripts/dev.py     # starts the API if needed, waits for /health,
                           # then runs the dashboard in the foreground
make dev

# ...or the two processes separately
uvicorn app.api:app                        # API → http://localhost:8000
streamlit run app/dashboard.py             # UI  → http://localhost:8501
# module forms, if the console scripts aren't on PATH:
python -m uvicorn app.api:app
python -m streamlit run app/dashboard.py
```

## Docker

```bash
docker compose up --build                          # development shape:
                                                     # source bind-mounted, live-reload
docker compose -f docker-compose.yml up --build -d  # production shape: baked image only
docker compose down
```

Every `F1_*` environment variable can be set via a `.env` file at the
project root (copy [`.env.example`](../.env.example)); Compose picks it up
automatically.

## Git and release workflow

```bash
git status
git log --oneline -10
git diff

# Version lives in pyproject.toml only; tag matches it on a release boundary
git tag v1.4.0
git push origin v1.4.0
```

Commit style: a short, imperative summary line. Pull requests (including
automated ones from the scheduled retrain workflow and Dependabot) are
always human-reviewed before merge — nothing in this repository's CI/CD
auto-merges.

## GitHub Actions

```bash
gh run list --workflow=ci.yml
gh run watch                              # follow the most recent run
gh run view <run-id> --log                # full log for a specific run
gh workflow run retrain.yml               # manually trigger the scheduled retrain
```

## Other useful commands

```bash
python -m build          # build an sdist/wheel (not part of any current workflow)
```
