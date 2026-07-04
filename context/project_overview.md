# F1 Race Winner Prediction — Project Overview

_Core ML Platform baseline: 2026-07-04._

## Purpose

Predict the winner of a Formula 1 race by scoring every entered driver with
information available before the race. The system is a portfolio-quality local ML
platform, not merely a notebook: raw-data processing, temporal feature engineering,
experiment tracking, registered inference, API serving, and a dashboard are separate
layers with tested contracts.

## Implemented scope

| Phase | Status | Deliverable |
|---|---|---|
| 0 Setup | Complete | package layout, dependencies, internal context |
| 1 Data | Complete | cleaning, repair, validation, interim parquet |
| 2 EDA | Complete | domain/quality analysis and temporal split |
| 3 Features | Complete | master dataset and 31 pre-race features |
| 4 Models | Complete | zoo, CV, tuning, evaluation, analysis, calibration, registry, inference |
| 5 Application | Complete | FastAPI, Streamlit, configuration, user documentation |
| 6 Quality baseline | Current | dedicated loader tests and ≥80% measured coverage remain |
| 7 Delivery | Future | Git/CI, containers, deployment/security |
| 8 Data operations | Future | maintained ingestion, incremental refresh, scheduling, monitoring |

## Problem framing and metrics

- One binary row per driver per race: `winner = 1` for exactly one driver.
- Scores are ranked within a race; user-facing probabilities are normalized to sum
  to one per race while raw calibrated outputs remain available.
- Mandatory baseline: pick the pole sitter.
- Primary metrics: per-race top-1 accuracy, top-3 winner recall, winner MRR.
- Probability metrics: log loss, Brier score, calibration table/ECE.
- Temporal windows: train 2010–2021, validation 2022–2023, final test 2024;
  2025 onward is excluded from current split and serving policy.

Current result: validation top-1 68.2% versus pole 54.5%; final-test top-1
45.8%, equal to pole, with 75.0% top-3 recall. The edge is dominance-season
concentrated and must not be presented as uniform across seasons.

## Implemented architecture

```text
Ergast-format CSV files
  -> src/data: load, clean, repair, validate
  -> data/interim/*.parquet
  -> src/integration + src/pipelines: join-only master dataset
  -> data/processed/master_dataset.parquet
  -> src/features: prior-race feature transforms and leakage validation
  -> data/processed/features.parquet
  -> src/models: temporal CV, model zoo, evaluation, analysis, calibration
  -> MLflow SQLite tracking + model registry
  -> f1-winner v2 @ Staging (CalibratedModel)
  -> src/models/predict.py (artifact-schema validation and race normalization)
  -> app/api.py (historical prediction API)
  -> app/dashboard.py + app/views (HTTP client and presentation)
```

Layer boundaries:

- `src/data/` owns raw loading, cleaning, validation, and interim output.
- `src/integration/` joins cleaned sources; it does not engineer model features.
- `src/features/` owns temporal feature logic; transforms exclude current-race outcomes.
- `src/models/` owns fitted preprocessing, training, tracking, calibration, analysis,
  registry loading, and model-agnostic inference.
- `app/api.py` owns HTTP/startup/cache/display enrichment and delegates scoring.
- `app/views/` owns presentation and consumes only the HTTP API.
- `docs/` and `README.md` are user-facing; `context/` is internal agent memory.

## Technology baseline

| Concern | Technology |
|---|---|
| Data/features | pandas, pyarrow/parquet |
| Models | scikit-learn, XGBoost, LightGBM |
| Tracking/registry | MLflow with project-root `sqlite:///.../mlflow.db` |
| Explainability | SHAP and per-race permutation importance |
| API | FastAPI, uvicorn, pydantic |
| Dashboard | Streamlit, Plotly, httpx |
| Configuration | pydantic-settings, `F1_` environment prefix |
| Tests | pytest, 314 tests (285 at the Phase-5 milestone baseline) |

## Repository structure

```text
app/                    FastAPI, settings, Streamlit entry and views
context/                internal agent operating memory and decisions
data/                   gitignored raw/interim/processed datasets
docs/                   user guide and API reference
models/                 gitignored model artifacts when applicable
notebooks/              EDA consumers; no reusable business logic
reports/                design, selection, EDA, and analysis artifacts
src/data/               loading, cleaning, validation, interim builder
src/integration/        join-only master dataset builder
src/pipelines/          integration orchestration
src/features/           modular feature transforms and pipeline
src/models/             splits, registry, evaluation, training, analysis,
                        calibration, prediction
tests/                  unit/integration tests mirroring implemented layers
mlflow.db               local MLflow tracking/registry database
pyproject.toml          PEP 621 packaging: version, Python floor, dependencies
requirements.txt        installer shim (`-e .[dev]`) — pins live in pyproject.toml
```

## Operational baseline

```bash
pip install -r requirements.txt
pip install -e .
python -m src.data.build_interim --target all
python -m src.pipelines.build_dataset
python -m src.features.pipeline
python -m pytest tests/
uvicorn app.api:app
streamlit run app/dashboard.py
```

The application is local/trusted-use software. It has no authentication, deployment
topology, automated ingestion, scheduler, or monitoring. Those are future milestones.

## Roadmap

1. Close the quality baseline: loader tests, measured ≥80% `src/` coverage, Git.
2. Add CI after the repository/version-control policy is established.
3. Design deployment/security, then containerize and deploy if required.
4. Resolve 2025–2026 provenance and select an upstream source before ETL work.
5. Design idempotent ingestion, upcoming-race feature materialization, refresh,
   retraining/promotion, and observability as one controlled data lifecycle.

Longer-term candidates include era-aware features/training, constructor lineage,
sprint handling, weather/telemetry, live prediction, and strategy modeling. None is
part of the current baseline.

## Non-negotiable design principles

- Time and race boundaries take precedence over convenience.
- Business logic belongs in `src/`, never in notebooks or dashboard pages.
- Model artifacts and fitted preprocessing go through MLflow.
- Processed data and artifacts must be reproducible from source data and code.
- New data/features require leakage review against `domain_knowledge.md` §7 and the
  v1 exclusion registry in §11.
- Future roadmap descriptions are not authorization to implement them.
