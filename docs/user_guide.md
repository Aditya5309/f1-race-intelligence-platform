# F1 Race Winner Prediction — User Guide

## What this system does

Predicts the winner of a Formula 1 Grand Prix **before the race starts**,
using only pre-race information: qualifying results, starting grid, rolling
driver/constructor form, circuit history, and championship standings from the
previous round. It serves one calibrated win probability per driver per race,
normalized within each race so the field's probabilities sum to 100%.

The serving model is an isotonic-calibrated logistic regression, selected
against a five-model comparison and registered in MLflow as
`f1-winner@Staging`. Evaluation on data it never trained on:

| Split | Top-1 accuracy | Top-3 recall | Pole-sitter baseline (top-1) |
|---|---|---|---|
| Validation 2022–2023 | 68.2% | 86.4% | 54.5% |
| Test 2024 | 45.8% | 75.0% | 45.8% |

**Read the numbers with the era caveat in mind:** the model's advantage over
"just pick the pole sitter" is concentrated in dominance seasons (2023). In
competitive seasons expect top-1 parity with the pole pick — but strong
top-3 ranking and well-calibrated probabilities throughout.

## Data & ML pipeline

| Stage | Module | Output artifact |
|---|---|---|
| Load · clean · repair · validate | `src/data` | Interim parquet (gitignored) |
| Master join (integration only) | `src/integration` + `src/pipelines` | Master dataset (gitignored) |
| Temporal feature engineering | `src/features` | Feature store (gitignored) |
| Train · tune · calibrate · register | `src/models/train.py` + `calibration.py` | MLflow registry entry |
| Freeze runtime artifacts | `src/models/serving_bundle.py` | Committed model bundle + features snapshot — no live MLflow or raw data needed to serve |
| Promote (gated) | `scripts/promote_model.py` | The only step that changes what is actually served — refuses any candidate whose accuracy regresses |
| Score | `src/models/predict.py` | Per-race normalized win probabilities |

Every stage enforces temporal discipline (rolling windows shift before they
roll, standings are lagged to the previous round, circuit history uses prior
visits only) — see the [README's Architecture section](../README.md#4-architecture)
for the full correctness-constraint explanation. The 2024 test season was
scored exactly once behind a guarded flag, and 2025–2026 data is excluded
from training/tuning and from serving by default.

The served model was trained on 31 pre-race features. The feature pipeline
has since grown to include additional experimental groups (teammate-relative
deltas, wet-weather signals); training defaults to a slightly narrower,
curated subset of those columns after an ablation study found one
experimental group (wet-weather deltas) didn't generalize from the
historical training window — that group is excluded from training by
default, while remaining fully computed and available for future evaluation
once more data accumulates.

Commands to rebuild each stage: [docs/commands.md](commands.md#building-the-data-pipeline-optional--only-for-retraining).

## Quick start

```bash
# One-time setup
pip install -r requirements.txt
pip install -e .

# 1. Start the API (terminal 1)
uvicorn app.api:app                      # http://localhost:8000, docs at /docs

# 2. Start the dashboard (terminal 2)
streamlit run app/dashboard.py           # http://localhost:8501
```

Or, for local development, start both with one command:

```bash
python scripts/dev.py                    # equivalent: make dev
```

`scripts/dev.py` starts the API only if nothing is already answering at its
`/health` endpoint, waits for it to come up, then runs the dashboard in the
foreground; it cleans up the API on exit only if it started it. This is a
local development convenience — a real deployment runs the two services
independently (see the Docker section below, or the two-terminal form above).

Prerequisites: a committed runtime features snapshot (`artifacts/
features.parquet`) and a frozen serving bundle at `artifacts/serving/
staging/` (defaults; see `F1_FEATURES_PATH`/`F1_SERVING_BUNDLE_PATH` below).
Both ship in the repository — a fresh clone already has everything the API
needs. If missing (or you want to refresh them from a new training run),
rebuild:

```bash
python -m src.data.build_interim --target all
python -m src.pipelines.build_dataset
python -m src.features.pipeline
python -m src.models.train --model logreg --register Staging --calibrate \
    --params-file config/registered_model_params.json
```

Registering a model freezes both runtime artifacts automatically: the frozen
model bundle at `artifacts/serving/staging/` and a snapshot of
`data/processed/features.parquet` copied to `artifacts/features.parquet` —
the file the deployed API actually reads. The gitignored `data/` tree
(raw CSVs, interim/processed parquet, MLflow's own store) is a training-time
concern only; the deployed API never touches it. This registration step
alone does not gate for quality — see [README.md's "Promotion &
rollback"](../README.md#11-model-performance) for the sanctioned path to
actually changing what is served.

### Or, run it in Docker

No local Python install needed at all — just Docker:

```bash
docker compose up --build
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
```

This builds two images (`docker/Dockerfile.api`, `docker/Dockerfile.dashboard`),
each a slim `python:3.11-slim` build carrying only what that service
actually imports at runtime (the API image has no streamlit/xgboost/
lightgbm/mlflow; the dashboard image has no scikit-learn/scipy/mlflow), and
bakes in the same committed `artifacts/` tree described above — you get
real predictions from the real served model with no extra setup.

`docker compose up` also picks up `docker-compose.override.yml`
automatically, which bind-mounts `src/`/`app/`/`artifacts/` read-only and
enables live-reload — edit code on your host, see it reflected in the
running container without rebuilding. For a production-shape run with
neither of those (baked image only, no mounts, no reload):

```bash
docker compose -f docker-compose.yml up --build -d
```

Every `F1_*` setting below can be set via a `.env` file at the project root
(copy `.env.example` to start) — Compose reads it automatically for the
`${VAR}` substitutions in `docker-compose.yml`.

## The dashboard

Eight pages (left navigation). Most are fan-first; all ML/technical detail is
confined to the last, clearly-labeled "Advanced" page. Every prediction
number comes from the FastAPI service over HTTP — the dashboard never
imports model code. Grand Prix names, grids, standings, and career stats are
read-only display metadata that degrades gracefully when unavailable.

- **🏠 Dashboard** — system status at a glance: model stage (Staging/
  Production), API health, latest model version, supported season range,
  and headline validation/test metrics.
- **🏎 Race Center** — pick a season and race (2010 through the currently-verified
  season, 2024 as of this writing), or the single
  upcoming race with no result yet (added as an extra picker entry); see
  the model's favorite as a hero card with a confidence level, the top-5
  contenders as cards (grid/qualifying position, rank trend vs. the
  previous round), a plain-language "why did the model choose this
  driver?" breakdown, a hit/miss badge against the actual winner (once one
  exists), race facts, and the full field as a constructor-colored bar
  chart plus table. Also hosts two interactive views: a **grid-position
  simulator** ("what if this driver started P1 instead?") that freezes
  every other feature at its real value, and a **qualifying-impact
  comparison** placing the full model's picks next to a simple pole-only
  baseline for the same race. Tied probabilities between midfield drivers
  are normal (the calibrator maps similar strength to the same probability
  step). Selecting the upcoming race shows a provenance panel (which model,
  which data snapshot, when it was generated) and any caveats about the
  prediction's completeness — see
  [docs/pre_race_materialization.md](pre_race_materialization.md) for how
  that prediction is built.
- **👤 Driver Explorer** — one driver's races, scoped to a season or their
  whole served career: profile card with current championship position,
  wins/podiums/poles/points/average-qualifying/average-finish tiles,
  qualifying/finishing/championship-points/win-share trend charts, a skill
  radar chart, and a full race log.
- **⚖️ Compare Drivers** — two drivers, one season, side by side — the same
  tiles and an overlaid radar chart.
- **🏭 Team** — one constructor's season or career view, same shape as
  Driver Explorer but at the team level.
- **🏟 Circuit Explorer** — one circuit's all-time stats, with a rendered
  track-layout map for circuits where one is available.
- **📊 Season Analytics** — how a season is unfolding: model hit rate round
  by round (plus cumulative), championship standings, most-predicted
  winners, most surprising races (biggest upsets), win-share distribution,
  and which drivers/constructors are rising or fading.
- **🤖 Model Insights** *(advanced)* — model card (algorithm, calibration
  method, registry alias, run id), validation/test results, the full
  model-comparison table, the feature classification (stable /
  era-sensitive / experimental), and the feature-importance/SHAP/
  calibration/diagnostic figures.

## Command-line prediction

```bash
python -m src.models.predict --race-id 1120     # score one race directly
```

## Configuration

All settings are environment variables prefixed `F1_` (or a local `.env`
file). The defaults work out of the box. Common overrides:

| Variable | Default | Meaning |
|---|---|---|
| `F1_SERVING_BUNDLE_PATH` | `artifacts/serving/staging` | which frozen serving bundle the API loads — no live MLflow registry needed, and no dependency on the gitignored `data/` tree |
| `F1_FEATURES_PATH` | `artifacts/features.parquet` | the committed runtime feature snapshot the API scores races from (NOT the training-side `data/processed/features.parquet`) |
| `F1_API_URL` | `http://localhost:8000` | where the dashboard finds the API |
| `F1_VERIFIED_SEASONS_THROUGH` | `2024` | newest season the API will serve (was `F1_SERVE_MAX_YEAR` — renamed 2026-07-24, Decision 055). As of Decision 056 (2026-07-24) the provenance concern this setting was protecting against is resolved; a proposal to remove it entirely is approved in direction but not yet implemented — see `docs/serving_policy.md` |
| `F1_DEBUG_ENDPOINTS` | `false` | enables `/api/v1/debug/*` inspection routes (development only) |
| `F1_LOG_LEVEL` | `INFO` | API log verbosity |
| `F1_CORS_ALLOW_ORIGINS` | `` (empty) | comma-separated allowed CORS origins, or `*` for any — empty means no cross-origin browser access |

A full annotated template is in [`.env.example`](../.env.example).

## Why some races return errors

- **404** — the raceId isn't in the built feature matrix.
- **409** — the race's season hasn't been marked verified for historical
  serving yet (`F1_VERIFIED_SEASONS_THROUGH`, currently `2024`). This is
  distinct from the model's own evaluation holdout (a separate,
  permanently-fixed mechanism) — see `docs/serving_policy.md` for the full
  distinction, and note this gate is under active revision (Decision 056).
- **500** — an unexpected server-side error. The response body is always a
  generic `{"detail": "Internal server error."}` — no internal detail is
  ever exposed; the real cause is logged server-side only.
- **`POST /predict` returns 503** if the training-side data it needs isn't
  available on this deployment (it needs more than the committed
  `artifacts/` tree every other route reads from — see
  [docs/pre_race_materialization.md](pre_race_materialization.md)), and
  **409** if the race it's asked to score already has a real result.

## Security notes

- No authentication is implemented. This is a public demonstration of a
  read-only historical-data API: every route is `GET` except `POST
  /predict`, which accepts only an identity payload (year/round, optional
  entry-list override) — never feature values — and there's no
  user-account concept.
- CORS is deny-by-default; configure `F1_CORS_ALLOW_ORIGINS` if you're
  calling this API directly from browser JavaScript on another origin.
- Report a suspected vulnerability per [SECURITY.md](../SECURITY.md).

## Limitations to keep in mind

1. Race outcomes are irreducibly noisy (safety cars, first-lap incidents) —
   even a perfect pre-race model cannot approach 100% top-1 accuracy.
2. The model's edge is era-dependent (see the caveat above).
3. Probabilities describe the *pre-race* picture; nothing in-race updates them.
4. Rookies and newly rebranded teams have little/no history — the model
   correctly treats them as long shots.
5. Already-run races through the currently-verified season (2024 as of this
   writing — see `docs/serving_policy.md`) are always servable. The single upcoming
   race with no result yet can also be scored (`POST /predict` /
   `GET /races/upcoming`), but only on a deployment with the training-side
   data that route needs — see
   [docs/pre_race_materialization.md](pre_race_materialization.md) for the
   gap and its 503 degraded-mode behavior. Its qualifying-position-as-grid
   proxy (documented in the same file) is a real, disclosed limitation, not
   a bug.

## More documentation

- [docs/api_reference.md](api_reference.md) — full REST API reference.
- [docs/pre_race_materialization.md](pre_race_materialization.md) — how
  upcoming-race predictions are built.
- [docs/commands.md](commands.md) — the complete command reference.
- [docs/retrain_workflow_setup.md](retrain_workflow_setup.md) — scheduled
  ingestion/retraining workflow setup and troubleshooting.
- Interactive API docs (Swagger UI) at `http://localhost:8000/docs`.
