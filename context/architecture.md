# Architecture

_As-built Core ML Platform baseline, audited 2026-07-04._

## System context

```text
                           local/trusted environment

Browser -> Streamlit dashboard -> HTTP/JSON -> FastAPI
                                             |  startup: load model + feature parquet
                                             |  request: select race rows
                                             v
                                      src.models.predict
                                        |           |
                         artifact schema|           |predict_proba
                                        v           v
                              MLflow registry   CalibratedModel
                              f1-winner@Staging  (LogReg + isotonic)

CSV -> data/clean/validate -> master join -> temporal features -> parquet
                                                         |
                                                         +-> training/evaluation/MLflow
                                                         +-> API historical feature store
```

## Dependency direction

```text
app/views -> HTTP API
app/api -> app/config + src/models/predict + src/models/registry.training_schema
src/models -> src/features/metadata + processed feature data
src/features -> master dataset + standings source tables
src/pipelines -> src/integration
src/integration -> cleaned/interim data
src/data -> raw CSV data
```

The dashboard does not import `src`. The API is a thin serving adapter; model fitting
and feature construction do not occur in `app/`. `src/models/predict.py` is the single
model-agnostic scoring contract shared by CLI and API.

## Data flow

1. `src.data.loader` reads Ergast-format CSV files and maps `\N` to null.
2. `src.data.cleaner` enforces result/qualifying types and derives result status.
3. `src.data.build_interim` owns dataset-specific repairs, validation, and parquet
   publication. Current results output has 27,279 rows; qualifying has 11,102.
4. `src.integration.build_master_dataset` left-joins results to race, circuit,
   driver, constructor, and qualifying dimensions. Join fan-out and key integrity
   are checked. The result-grain output has 27,279 rows and 43 columns.
5. `src.features.pipeline` composes qualifying, driver-form, constructor-form,
   circuit-history, and lagged-standings transforms. It writes 27,279 rows with
   31 model features plus identifiers and target.

Standings are deliberately not joined into the master table: converting them to
round N-1 information is a temporal feature operation. Post-race columns remain in
the master history table but are prohibited from the feature matrix.

## Temporal and feature contract

- One record per `(raceId, driverId)`.
- `FEATURE_COLUMNS` in `src/features/pipeline.py` is the repository feature contract.
- `POST_RACE_OUTCOME_COLUMNS` is checked for intersection at import and validation.
- Driver and constructor rolling values shift before rolling.
- Constructor calculations aggregate to race grain before rolling, preventing
  same-race teammate leakage.
- Circuit features use prior visits only.
- Standings use the previous calendar race; season-opening rows use prior-season final.
- Train/validation/test windows are explicit; rows after 2024 enter none of them.

## Model architecture and MLflow

`src/models/registry.py` defines five candidates: pole baseline, logistic regression,
random forest, XGBoost, and LightGBM. Every fitted pipeline starts with `ColumnGuard`,
which records names/order/dtypes and validates again at prediction time.

`src/models/train.py` owns season-fold CV, tuning, validation/final-test controls,
artifact/metric logging, and registry registration. MLflow uses the project-root
SQLite URI by default, experiment `f1-winner-prediction`, registered model
`f1-winner`, and aliases rather than stage APIs.

The selected tuned logistic regression is wrapped by `CalibratedModel`:

- isotonic calibration is fit on training-season OOF probabilities;
- the final base pipeline is fit independently on the chosen fit frame;
- the wrapper delegates `named_steps` so schema introspection still reaches
  `ColumnGuard`;
- v2 at `Staging` is calibrated; v1 remains registry history without the alias.

`Production` is unset. Promotion/refit is a policy decision, not a missing runtime
dependency.

## Prediction flow

```text
load_model(alias)
  -> resolve models:/f1-winner@alias
  -> load sklearn artifact and registry metadata
predict_race(model, rows)
  -> read stored artifact schema
  -> validate raceId, unique driver/race, required numeric features
  -> call predict_proba
  -> normalize raw probabilities within each race
  -> deterministic descending ranks + carried identifiers
```

Raw output is calibrated binary probability. `win_probability` is a monotonic,
within-race share for presentation; it is not a separately calibrated multinomial
probability.

## API architecture

`app.api.create_app(settings)` supports dependency injection in tests. Its lifespan:

1. resolves the configured MLflow alias;
2. loads `features.parquet` once;
3. optionally loads driver/constructor display-name maps;
4. enters degraded mode on model/feature failure rather than crash-looping.

Implemented routes:

| Route | Contract |
|---|---|
| `GET /health` | 200 with `ok` or `degraded` state and detail |
| `GET /model` | loaded registry/artifact metadata; 503 if unavailable |
| `GET /races?year=` | scoreable races through `F1_SERVE_MAX_YEAR` |
| `GET /predictions/{race_id}` | historical full-field prediction |
| `GET /debug/features/{race_id}` | artifact-ordered feature rows when enabled |
| `POST /predict` | reserved 501 stub for a future upcoming-race contract |

Prediction cache entries are FIFO bounded and keyed by `(model_version, race_id)`.
Cached responses intentionally preserve the original response body/prediction ID;
each HTTP request emits its own request log ID. The API currently relies on FastAPI's
default exception behavior; it does not implement the generic custom 500 handler
described in the original design.

## Dashboard architecture

- `app/dashboard.py`: `st.navigation` entry point.
- `app/views/common.py`: HTTP client, cache, API errors, model sidebar.
- `app/views/overview.py`: scope, metrics, and interpretation.
- `app/views/predictions.py`: race selectors, chart, winner comparison, field table.
- `app/views/insights.py`: static analysis artifacts; no model computation.

The dashboard uses `F1_API_URL`, caches GET responses for 300 seconds, and keeps the
era-performance caveat visible. It requires both API and dashboard processes locally.

## Configuration

All app settings use pydantic-settings with the `F1_` prefix:

| Variable | Default | Meaning |
|---|---|---|
| `F1_TRACKING_URI` | empty -> model-layer project SQLite default | MLflow backend |
| `F1_MODEL_ALIAS` | `Staging` | registry alias to load |
| `F1_FEATURES_PATH` | `data/processed/features.parquet` | historical feature store |
| `F1_DATA_DIR` | `data/` | display-name CSV directory |
| `F1_SERVE_MAX_YEAR` | `2024` | forward-holdout serving guard |
| `F1_DEBUG_ENDPOINTS` | `false` | debug feature route gate |
| `F1_PREDICTION_CACHE_SIZE` | `512` | FIFO prediction entries |
| `F1_LOG_LEVEL` | `INFO` | API log level |
| `F1_API_URL` | `http://localhost:8000` | dashboard API base URL |

## Implemented versus future boundary

Implemented: local batch rebuilding, local MLflow registry, historical inference,
local API/dashboard, and stdout request logging.

Not implemented: maintained upstream ingestion, upcoming-race row materialization,
working POST prediction, reload, auth, CI, containers, deployment, scheduling,
monitoring, atomic data/model refresh, remote registry, or multi-instance cache.

These require explicit decisions. Do not infer them from long-term diagrams.
