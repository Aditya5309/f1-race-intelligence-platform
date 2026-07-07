# F1 Race Winner Prediction — User Guide

_User-facing documentation. (Design documents and analysis reports live in
`reports/`.)_

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
| Validation 2022–2023 | 68.2% | 88.6% | 54.5% |
| Test 2024 | 45.8% | 75.0% | 45.8% |

**Read the numbers with the era caveat in mind:** the model's advantage over
"just pick the pole sitter" is concentrated in dominance seasons (2023). In
competitive seasons expect top-1 parity with the pole pick — but strong
top-3 ranking and well-calibrated probabilities throughout.

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

Or, for local development, start both with one command (dev tooling only,
Decision 025 — production still runs the two services independently):

```bash
python scripts/dev.py                    # equivalent: make dev
```

`scripts/dev.py` starts the API only if nothing is already answering at its
`/health` endpoint, waits for it to come up, then runs the dashboard in the
foreground; it cleans up the API on exit only if it started it.

Prerequisites: the built feature matrix (`data/processed/features.parquet`)
and a frozen serving bundle at `models/serving/staging/` (default; see
`F1_SERVING_BUNDLE_PATH` below). If missing, rebuild:

```bash
python -m src.data.build_interim --target all
python -m src.pipelines.build_dataset
python -m src.features.pipeline
python -m src.models.train --model logreg --register Staging --calibrate \
    --params '{"model__C": 0.01653693718282442}'
```

## The dashboard

Five pages (left navigation). Four are fan-first; all ML/technical detail is
confined to the fifth, clearly-labeled "Advanced" page. Every prediction
number comes from the FastAPI service over HTTP — the dashboard never
imports model code. Grand Prix names, grids, standings, and career stats are
read-only display metadata that degrades gracefully when `data/` is absent.

- **🏠 Dashboard** — system status at a glance: model stage (Staging/
  Production), API health, latest model version, supported season range,
  and headline validation/test metrics. One-line links out to the other four
  pages.
- **🏎 Race Center** — pick a season and race (2010–2024); see the model's
  favorite as a hero card with a confidence level, the top-5 contenders as
  cards (grid/qualifying position, rank trend vs. the previous round), a
  plain-language "why did the model choose this driver?" breakdown, a
  hit/miss badge against the actual winner, race facts, and the full field
  as a constructor-colored bar chart plus table. Tied probabilities between
  midfield drivers are normal (the calibrator maps similar strength to the
  same probability step).
- **👤 Driver Explorer** — one driver's races, scoped to a season or their
  whole served career: profile card with current championship position,
  wins/podiums/poles/points/average-qualifying/average-finish tiles,
  qualifying/finishing/championship-points/win-share trend charts, and a
  full race log.
- **📊 Season Analytics** — how a season is unfolding: model hit rate round
  by round (plus cumulative), championship standings, most-predicted
  winners, most surprising races (biggest upsets), win-share distribution,
  and which drivers/constructors are rising or fading.
- **🤖 Model Insights** *(advanced)* — model card (algorithm, calibration
  method, registry alias, run id), validation/test results, the full
  five-model comparison table, the Decision-013 feature classification
  (stable / era-sensitive / experimental), and the feature-importance/SHAP/
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
| `F1_SERVING_BUNDLE_PATH` | `models/serving/staging` | which frozen serving bundle the API loads (Decision 026/027 — no live MLflow registry needed) |
| `F1_API_URL` | `http://localhost:8000` | where the dashboard finds the API |
| `F1_SERVE_MAX_YEAR` | `2024` | newest season the API will serve (2025+ is a reserved evaluation holdout) |
| `F1_DEBUG_ENDPOINTS` | `false` | enables `/debug/*` inspection routes (development only) |
| `F1_LOG_LEVEL` | `INFO` | API log verbosity |

## Why some races return errors

- **404** — the raceId isn't in the built feature matrix.
- **409** — the race is in the 2025–2026 *forward holdout*: data deliberately
  reserved to evaluate the system on genuinely unseen seasons later. Serving
  it casually would spoil that experiment.

## Limitations to keep in mind

1. Race outcomes are irreducibly noisy (safety cars, first-lap incidents) —
   even a perfect pre-race model cannot approach 100% top-1 accuracy.
2. The model's edge is era-dependent (see the caveat above).
3. Probabilities describe the *pre-race* picture; nothing in-race updates them.
4. Rookies and newly rebranded teams have little/no history — the model
   correctly treats them as long shots.

## More documentation

- `docs/api_reference.md` — REST API reference.
- `reports/model_selection_report.md` — full model-selection evidence.
- `reports/application_design.md` — serving architecture design.
- Interactive API docs (Swagger UI) at `http://localhost:8000/docs`.
