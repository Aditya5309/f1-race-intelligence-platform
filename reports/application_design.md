# Phase 5 — Application Layer Design (FastAPI + Streamlit)

_Status: APPROVED AND IMPLEMENTED 2026-07-04 (Decision 016); audited against
`app/` and `tests/test_api.py`. Historical design rationale is retained._
_Author: AI agent, Phase 5 planning session, 2026-07-04._
_Depends on: Decisions 003, 004, 008, 012, 014, 015; `src/models/predict.py`
(the serving contract); MLflow registry `f1-winner` v2 @Staging
(CalibratedModel); `data/processed/features.parquet` (27,279 × 38)._
_Approval points are collected in §16._

---

## 1. Overall Application Architecture

Two thin processes over the existing, fully tested inference stack. **No
business logic lives in `app/`** — the application layer translates HTTP/UI
concerns to exactly two calls: `predict.load_model()` (startup) and
`predict.predict_race()` (per request). Everything below the dashed line
already exists and does not change.

```
┌───────────────┐        HTTP (JSON)        ┌──────────────────────────┐
│  Streamlit    │ ────────────────────────▶ │  FastAPI (app/api.py)    │
│  dashboard    │ ◀──────────────────────── │  uvicorn, port 8000      │
│  (app/        │   PredictionResponse      │                          │
│  dashboard.py)│                           │  app.state.model         │
└───────────────┘                           │  app.state.model_info    │
      browser ▲                             │  app.state.features (df) │
              │ renders                     └───────────┬──────────────┘
              │                                         │ in-process calls
 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─
                                                        ▼
                              src/models/predict.py   load_model(alias)
                                                      predict_race(model, df)
                                                        │
                              MLflow registry (sqlite:///mlflow.db)
                              f1-winner@Staging = CalibratedModel (Dec. 015)
                                                        │
                              data/processed/features.parquet  (feature source)
```

**Key architectural choice — the dashboard consumes the API, not `src/`
directly.** Rationale: (a) it matches the target-architecture SERVING box
(project_overview.md) where the API is the single inference entry point;
(b) it forces the API contract to be complete (the dashboard is its first
real client); (c) it lets Phase 7 deploy the two processes independently.
Cost: local dev runs two processes — accepted; a `make`-style run note in
AI_AGENT.md §4 covers it. (§16.1 — needs approval.)

**Second key choice — server-side feature lookup by `raceId`.** The original
API sketch (architecture.md "API Design") had clients POST driver feature
dicts. That contract is wrong for this system: features are *derived
artifacts* of the leakage-audited pipeline (rolling windows, lagged
standings) — a client cannot be trusted, or expected, to compute them. The
API therefore accepts a race identifier and reads the feature rows from
`features.parquet` itself. Client-supplied feature payloads are deferred to
Phase 8 (upcoming-race scoring, where an ETL step materializes the feature
rows first — §12). This supersedes the architecture.md sketch. (§16.2 —
needs approval.)

## 2. FastAPI Structure (`app/api.py`)

Single module, mirroring the project's thin-orchestration style:

| Concern | Implementation |
|---|---|
| App factory | `create_app(settings) -> FastAPI` — testable with injected settings; module-level `app = create_app(Settings())` for uvicorn |
| Startup | FastAPI lifespan handler: load settings → `load_model(alias)` → read `features.parquet` (once) → stash on `app.state` |
| Routes | `GET /health`, `GET /model`, `GET /races`, `GET /predictions/{race_id}` (§5) |
| Schemas | pydantic models in the same module (few enough); split into `app/schemas.py` only if Phase 5 grows |
| Errors | FastAPI defaults plus explicit startup/route `HTTPException` behavior (§7) |

Run: `uvicorn app.api:app --reload` (unchanged from AI_AGENT.md).

## 3. Streamlit Structure (multi-page — amendment)

`app/dashboard.py` is the entry point (`st.navigation`); one module per page
under `app/views/`:

| Page | Module | Content |
|---|---|---|
| **Overview** | `app/views/overview.py` | what the system is, model card (from `/model`), headline Phase-4 metrics, the era-caveat note, how to read the numbers |
| **Predictions** | `app/views/predictions.py` | season → race selectors (`/races`), probability bar chart + field table + hit/miss badge (`/predictions/{race_id}`) |
| **Model Insights** | `app/views/insights.py` | Decision-013 feature-class summary, Phase-4 analysis figures from `reports/phase4_analysis/` (importance, SHAP, calibration), pointer to the selection report |

Shared plumbing in `app/views/common.py`: `api_get(path)` (`httpx` on
`F1_API_URL`, wrapped in `st.cache_data(ttl=300)`), error banner helper.
No `src/` imports — all project logic goes through the API; the insights
page reads static figure files only.

## 4. Request/Response Flow

```
User picks race 1120
  → dashboard GET {API}/predictions/1120
    → api: validate race exists in app.state.features
    → api: predict_race(app.state.model, feature rows for 1120)   # in-process
    → api: attach ModelInfo + actual outcome (winner known for past races)
  ← 200 PredictionResponse (JSON)
  → dashboard renders probability bars + hit/miss badge
```

One prediction of a 20-driver field costs ~44 µs of model time (measured,
Phase 4 timing analysis); end-to-end latency is dominated by HTTP overhead.
No async model work is needed — routes are plain `def` (FastAPI runs them in
a threadpool; the model and DataFrame are read-only after startup, so this
is thread-safe).

## 5. API Endpoints

| Method/Path | Purpose | Success | Errors |
|---|---|---|---|
| `GET /health` | liveness + which model is serving | 200 `HealthResponse` (`ok` or `degraded`) | — |
| `GET /model` | full `ModelInfo` for the loaded artifact | 200 `ModelInfoResponse` | 503 |
| `GET /races?year=2023` | races available for scoring (drives the dashboard selector); `year` optional filter | 200 `RaceListResponse` | 422 bad year |
| `GET /predictions/{race_id}` | score one race's full field | 200 `PredictionResponse` | 404 unknown race, 409 forward-holdout guard (§5.1), 503 |
| `POST /predict` | **RESERVED (amendment)** — Phase 8 upcoming-race entry point; the route exists and returns `501 Not Implemented` with a pointer to this section, so Phase 8 lands without an API redesign | 501 always (v1) | — |
| `GET /debug/features/{race_id}` | **(amendment)** development-only: the exact feature vector rows fed to the model for a historical race (ids + the artifact-schema columns, NaNs as nulls) | 200 `FeatureDebugResponse` | 404 route disabled unless `F1_DEBUG_ENDPOINTS=true` (default false — off in production), 404 unknown race, 409 holdout |

Explicitly **not** in v1: a working `POST /predict` body (reserved stub
only — accepts `{race_id}` for a future race OR explicit feature rows,
validated against the artifact schema, per §12); `POST /model/reload` (§8);
auth endpoints (§13).

### 5.1 Forward-holdout guard

`features.parquet` contains 2025–2026 rows reserved as the untouched forward
holdout (Decision 012 §13.1). Casually serving model-vs-actual comparisons
for those races through a UI is informal peeking that erodes the holdout's
value for Phase 8. Default policy: `/races` lists and `/predictions`
serves **years ≤ 2024 only**; requests beyond return `409 {"detail":
"forward-holdout race — reserved for Phase 8 evaluation"}`. Overridable by
config (`SERVE_MAX_YEAR`) when Phase 8 legitimately needs it. (§16.3 —
needs approval.)

## 6. Request/Response Schemas (pydantic)

```
HealthResponse    { status: "ok", model: ModelInfoSchema }
ModelInfoSchema   { name, version, alias, run_id, trained_at,
                    calibration, model_class }          # == ModelInfo.to_dict()
RaceSummary       { race_id, year, round, n_drivers }
RaceListResponse  { races: [RaceSummary] }
DriverPrediction  { driver_id, driver_name: str | null,
                    constructor_id, constructor_name: str | null,
                    predicted_rank,
                    win_probability,          # per-race-normalized share (user-facing)
                    win_probability_raw }     # calibrated model output
PredictionResponse{ prediction_id,                         # uuid4 (amendment; matches log line)
                    race_id, year, round, generated_at,    # ISO-8601 UTC
                    model: ModelInfoSchema,
                    predictions: [DriverPrediction],       # sorted by rank
                    actual_winner_driver_id: int | null,   # null if unknown
                    model_top1_hit: bool | null }
FeatureDebugResponse { race_id, model: ModelInfoSchema,    # (amendment, dev-only)
                    feature_names: [str],                  # the artifact's schema order
                    rows: [{driver_id, features: {name: float|null}}] }
ErrorResponse     { detail: str }                          # FastAPI convention
```

Notes: `win_probability` is the design-§6 user-facing number;
`win_probability_raw` is exposed for transparency/debugging. Within-race
ties in probabilities are normal (isotonic plateaus, Decision 015) —
consumers order by `predicted_rank`, which is deterministically tie-broken.
Driver/constructor *names* require joining `drivers.csv`/`constructors.csv`;
the API includes a small startup-time id→name lookup built from those CSVs
(display names are a serving concern; the feature pipeline stays id-only).

## 7. Error Handling

| Failure | Behavior |
|---|---|
| Registry/model unavailable at startup | app starts DEGRADED: `/health` → HTTP 200 with `status="degraded"`; model-dependent routes → 503. |
| Unknown `race_id` | 404 with the requested id echoed |
| Forward-holdout race | 409 (§5.1) |
| `predict_race` raises ValueError (schema/grouping) | 500 + logged at ERROR — with server-side lookup this indicates a data/artifact mismatch, i.e. a bug, not a client error |
| Unhandled exception | FastAPI default 500 behavior; the designed custom generic handler is not implemented |
| Dashboard: API unreachable / non-200 | `st.error` banner with retry hint; page stays usable |

## 8. Model Loading Strategy

- **Load once at startup** via lifespan: `load_model(alias=settings.model_alias)`
  (default `Staging`). The model (~KB-scale LogReg + isotonic) and
  features.parquet (723 KB) live in memory for the process lifetime.
- The served `ModelInfo` (version, calibration, trained_at) is attached to
  every `PredictionResponse` — consumers can always tell which artifact
  produced a number (mirrors the data-fingerprint discipline of Phase 4).
- **Picking up a new registry version requires a process restart** in v1 —
  deliberate: restart-to-deploy is unambiguous and idempotent. A
  `POST /model/reload` admin endpoint is designed but deferred to Phase 8b
  (scheduled retraining is the first thing that needs it, and it should
  arrive together with auth, §13).

## 9. Caching Strategy

| Layer | What | How | Invalidation |
|---|---|---|---|
| API | features.parquet | read once at startup into `app.state.features` | restart (v1); Phase 8 reload hook |
| API | per-race predictions | dict keyed **`(model_version, race_id)`** (amendment — model version is an explicit key component so a reload/new artifact can never serve stale entries); deterministic model ⇒ entries never stale within a version | version-keyed; bounded (~512 races, FIFO eviction) |
| Dashboard | API GETs | `st.cache_data(ttl=300)` on the fetch helper | TTL; manual "refresh" button clears |
| Dashboard | static assets (figures) | `st.cache_resource` if Phase-4 plots are embedded | n/a |

Nothing here is load-bearing for correctness — predictions are deterministic
(tested) — caching only trims repeated work.

## 10. Logging

- Std-lib `logging`, logger names `f1.api` / `f1.dashboard`; uvicorn keeps
  its access log. Level from config (`F1_LOG_LEVEL`, default INFO).
- **(amendment) Structured prediction-request fields**, emitted as
  `key=value` pairs on one INFO line (JSON formatter is a config switch
  later): `prediction_id` (uuid4 per request, also returned in the
  response for cross-referencing), `race_id`, `model_version`,
  `model_alias`, `n_drivers`, `cache_hit`, `latency_ms`, `status_code`.
- ERROR with traceback for §7's 500-class failures.
- Format: plain text v1; a JSON formatter is a config switch away when
  Phase 8c monitoring wants machine-readable logs. No log files managed by
  the app (stdout only — container/scheduler friendly, 12-factor).

## 11. Configuration Management

`app/config.py` — a single pydantic-settings `Settings` class, env-prefixed
`F1_` and `.env`-file capable (add `pydantic-settings` to requirements;
`.env` gitignored):

| Setting | Default | Used by |
|---|---|---|
| `F1_TRACKING_URI` | `sqlite:///<project>/mlflow.db` (predict.py's default) | api |
| `F1_MODEL_ALIAS` | `Staging` | api |
| `F1_FEATURES_PATH` | `data/processed/features.parquet` | api |
| `F1_SERVE_MAX_YEAR` | `2024` (§5.1) | api |
| `F1_DEBUG_ENDPOINTS` | `false` (amendment — enables /debug/*; keep false in production) | api |
| `F1_DATA_DIR` | `data` (drivers.csv / constructors.csv name lookup; names degrade to null if absent) | api |
| `F1_LOG_LEVEL` | `INFO` | both |
| `F1_API_URL` | `http://localhost:8000` | dashboard |

No hardcoded paths in `app/` (guiding principle, project_overview.md);
everything routes through `Settings` so the hundredth incremental-sync run
and the first Docker run use the same code.

## 12. Future ETL Integration (Phase 8 — design hooks only, do not build)

- **New data arrives** → ingestion re-runs build_interim → build_dataset →
  features pipeline → new `features.parquet` → **API restart or reload
  endpoint** picks it up. The feature-file read is already isolated in one
  startup function so the reload hook is a one-function change.
- **Upcoming-race scoring** (predict a race that hasn't run): requires the
  feature pipeline to emit rows for a race with no results yet (qualifying
  known, outcome columns null). That is a *pipeline* extension, not an API
  one; when it lands, the API gains `POST /predict` accepting either a
  future `race_id` or explicit feature rows validated against the
  artifact's schema. The §6 schemas were shaped so this is additive.
- **Scheduled retraining** → new registry version → reload endpoint (§8) +
  `model_version` cache keying (§9) already absorb it.

## 13. Future Authentication (not in v1)

Local, single-user tool today — v1 ships **no auth**, bound to localhost by
default. Designed path when exposure happens (Phase 7/8): static API key in
an `X-API-Key` header enforced by FastAPI dependency (`Security(api_key_header)`),
key from `F1_API_KEY` env (never in git); dashboard passes it from its own
config. OAuth/JWT is out of scope for a portfolio system; recorded here so
nobody bolts on ad-hoc auth later.

## 14. Dashboard Page Layout

Three pages (amendment, §3). **Overview** is text + model card + headline
metrics; **Model Insights** is figure panels from `reports/phase4_analysis/`
with short captions. The **Predictions** page keeps the original three-zone
layout:

```
┌────────────┬──────────────────────────────────────────────────┐
│ SIDEBAR    │  H1: F1 Race Winner Prediction                   │
│ Season  ▾  │  Race header: "2023 R22 — Abu Dhabi" + date      │
│ Race    ▾  │                                                  │
│ ────────── │  [Prediction bar chart]                          │
│ Model info │   horizontal bars, win_probability desc,         │
│  name/ver  │   top 10 + "rest of field" expander;             │
│  alias     │   actual winner's bar highlighted; hit/miss      │
│  calibr.   │   badge ("model picked the winner" / "winner     │
│  trained   │   was model's #k pick")                          │
│  era note  │                                                  │
│ (§14 copy) │  [Field table] rank | driver | constructor |     │
│            │   win share % | raw prob                         │
│            │  ▸ expander: "About this model" — Decision-013   │
│            │    class summary + link to selection report      │
└────────────┴──────────────────────────────────────────────────┘
```

- Chart: Plotly horizontal bar, percentage
  labels, winner bar in a distinct color.
- **Era-caveat copy is mandatory** (session-handoff reminder): a fixed
  sidebar note — "The model's top-1 edge over 'pick the pole sitter' is
  concentrated in dominance seasons; in competitive seasons expect parity
  on top-1 but strong top-3 ranking." Set expectations in the UI itself.
- Race selector shows ≤ 2024 races only (mirrors §5.1).

## 15. User Interaction Flow, Testing, Deployment

**Flow:** open dashboard → (auto) health check renders model panel → pick
season → pick race → predictions render with actual-outcome badge → optional
expander for model background. Errors surface as banners, never blank pages.

**Testing (`tests/test_api.py`):** FastAPI `TestClient` against
`create_app()` with a **tmp registry + tiny synthetic features frame**
(reuse the `tests/test_predict.py` fixture pattern — register a calibrated
model into tmp sqlite, point `Settings` at it). Cases: health OK + model
metadata; health 503 when registry missing; races list + year filter;
predictions happy path (sums to 1, sorted, winner id attached);
404 unknown race; 409 forward-holdout; response schema round-trips through
pydantic. The dashboard is not unit-tested in v1 (thin rendering over the
tested API); a smoke launch is part of the verification checklist.

**Deployment considerations:**
- v1: two local processes — `uvicorn app.api:app` + `streamlit run
  app/dashboard.py`. Windows-friendly, no orchestration.
- Phase 7: two containers (api, dashboard) + shared volume for `mlflow.db`,
  `mlruns/`, `data/` — compose file. Note the sqlite single-writer
  constraint: the API only *reads* the registry, so co-existing with a
  training job is safe; two writers are not (document, don't solve).
- The API is stateless beyond its startup caches — horizontal scaling and a
  future remote MLflow server require zero code change (config only).

## 16. Approval Record

**APPROVED 2026-07-04 (Decision 016 → Accepted) with six amendments, all
folded into the sections above:**

Original six points (all approved): (1) dashboard consumes the API over
HTTP; (2) server-side feature lookup by raceId, superseding the old
POST-payload sketch; (3) forward-holdout serving guard (409, config-
overridable); (4) new dependencies fastapi/uvicorn/streamlit/plotly/httpx/
pydantic-settings, `~=`-pinned; (5) `app/config.py`; (6) no auth in v1.

User amendments: (a) `POST /predict` route RESERVED now (501 stub) as the
Phase 8 upcoming-race entry point — §5; (b) development-only
`GET /debug/features/{race_id}` gated by `F1_DEBUG_ENDPOINTS` — §5;
(c) prediction cache key is explicitly `(model_version, race_id)` — §9;
(d) structured prediction-log fields incl. `prediction_id` — §10;
(e) three-page dashboard (Overview / Predictions / Model Insights) — §3/§14;
(f) **`docs/` directory for user-facing documentation** (user guide + API
reference) — `context/` remains internal AI project memory only.

Implementation order: `app/config.py` + `app/api.py` + `tests/test_api.py`
→ `app/dashboard.py` (+views) + smoke verification → `docs/` + docs sync.
