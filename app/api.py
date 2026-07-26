"""
app/api.py

FastAPI serving layer.

    uvicorn app.api:app --reload

The app is deliberately logic-free: HTTP concerns in, calls to the tested
prediction layer out — `predict.load_model()` once at startup,
`predict.predict_race()` per request. For every historical route, feature
rows are looked up SERVER-SIDE from the frozen runtime features snapshot
(settings.features_path, default artifacts/features.parquet) by raceId —
clients never send feature payloads, since features are derived artifacts
of the leakage-audited pipeline, not something a caller should be able to
inject. `POST /predict` is the one exception to "clients never
send anything feature-shaped": callers may optionally send an entry-list
override (driver/constructor pairs, not feature values) — everything
feature-shaped is still built server-side, by `materialize_features()`,
never accepted from the request body.

`predict.load_model()` reads a frozen serving bundle (settings.
serving_bundle_path, default artifacts/serving/staging) — no live MLflow
tracking server, SQLite registry, or mlruns/ directory is required at
runtime. Both runtime artifacts live under the committed artifacts/ tree,
so the deployed API needs nothing from the gitignored data/ training tree
to serve predictions. This module has no concept of experiments or
registry aliases; that machinery lives entirely on the training side
(src/models/train.py).

Endpoints, all under /api/v1:
    GET  /api/v1/health                              liveness + serving model metadata
    GET  /api/v1/model                               full ModelInfo
    GET  /api/v1/races?year=                         races available to score — every
                                              completed race present in the frozen
                                              features snapshot (see docs/serving_policy.md)
    GET  /api/v1/races/upcoming                      identity only (year/round/
                                              name/circuit/date) of the single next race
                                              with no result yet — NOT a prediction, no
                                              materialization_status/caveats/provenance;
                                              those come from POST /predict.
    GET  /api/v1/standings/{year}                    championship standings (drivers +
                                              constructors) for one season — display
                                              metadata only, read from settings.data_dir
                                              (the committed artifacts/display/ snapshot),
                                              independent of model/features readiness; the
                                              Streamlit-side equivalent is
                                              app/views/metadata.py's
                                              season_driver_standings/
                                              season_constructor_standings. Degrades to
                                              empty drivers/constructors lists rather than
                                              404/503 when data is absent for that year.
    GET  /api/v1/model-comparison                    model-selection comparison table
                                              (artifacts/analysis/model_comparison.csv) —
                                              recorded at analysis time, independent of
                                              model/features readiness.
    GET  /api/v1/feature-importance                  logreg feature importance +
                                              importance-by-class (artifacts/analysis/
                                              feature_importance_logreg.csv).
    GET  /api/v1/shap-summary                        mean |SHAP| ranking (artifacts/
                                              analysis/shap_summary_logreg.csv). All three
                                              of the above degrade to an empty response
                                              (never 404/503) when their CSV is absent —
                                              see scripts/export_analysis_data.py
                                              (Decision 063) for why only these three
                                              reports/phase4_analysis/ artifacts have a
                                              route: the rest (SHAP beeswarm/waterfall,
                                              calibration, confusion matrix, ROC,
                                              precision-recall) have no underlying tabular
                                              data anywhere in this repo's own pipeline.
    GET  /api/v1/drivers/{id}/stats?year=             historical outcome stats (races/wins/
                                              podiums/poles/points/avg_quali/avg_finish);
                                              year omitted = career.
    GET  /api/v1/drivers/{id}/race-results?year=      per-race grid/quali/finish rows.
    GET  /api/v1/drivers/{id}/standings-progression?year=  round-by-round championship
                                              points/position (year REQUIRED).
    GET  /api/v1/drivers/{id}/radar?year=             Qualifying pace/Race pace/Consistency,
                                              0-100 within-season (year REQUIRED); scores is
                                              null (not an error) when data is insufficient.
    GET  /api/v1/constructors                        every constructor in the catalog.
    GET  /api/v1/constructors/{id}/stats?year=        mirrors /drivers/{id}/stats.
    GET  /api/v1/constructors/{id}/race-results?year=  mirrors /drivers/{id}/race-results.
    GET  /api/v1/constructors/{id}/drivers?year=      roster for one season (year REQUIRED).
                                              All 8 entity-stats routes above (Decision 064,
                                              Frontend Overhaul Phase 5, app/entity_stats.py)
                                              read settings.data_dir directly, independent of
                                              model/features readiness, and degrade to
                                              zero/null/empty (never 404/503) when an entity
                                              has no data in the requested scope.
    GET  /api/v1/circuits                            every circuit in the catalog.
    GET  /api/v1/circuits/{id}/layout                track-outline GeoJSON (coordinates/
                                              attribution both null when this circuit has no
                                              OpenStreetMap backfill).
    GET  /api/v1/circuits/{id}/stats                 all-time stats: races held, years active,
                                              DNF rate, avg grid->finish change, fastest
                                              recorded lap, win-count leaders. All 3 circuit
                                              routes (Decision 065, Frontend Overhaul Phase 6,
                                              app/circuit_stats.py) are independent of model/
                                              features readiness and degrade to zero/null/empty
                                              (never 404) when a circuit has no data.
    GET  /api/v1/predictions/{race_id}               per-race-normalized field predictions
    GET  /api/v1/predictions/{race_id}/simulate/{driver_id}
                                              Prediction Simulator: re-score
                                              one driver with an overridden
                                              grid/qualifying position, everything
                                              else held at real values — see
                                              ADJUSTABLE_GRID_FEATURES below for why
                                              only 3 of the 10 qualifying-group
                                              features actually move.
    GET  /api/v1/predictions/{race_id}/vs-baseline   Qualifying Impact: the
                                              calibrated model's predictions next
                                              to MODEL_ZOO["pole_baseline"]'s (grid-
                                              only heuristic) for the same race.
    GET  /api/v1/debug/features/{race_id}  dev-only (F1_DEBUG_ENDPOINTS=true) —
                                    the exact feature vectors fed to the model
    POST /api/v1/predict            Upcoming-race prediction: materializes the
                                    single next race with no result yet
                                    (this pipeline only ever resolves the
                                    earliest unplayed race — a materialization
                                    horizon of one race, not a forward
                                    schedule) and scores it with the same
                                    served bundle. See the
                                    "Upcoming-race prediction" section below.

Upcoming-race prediction (`POST /predict`): unlike every route
above, this one needs training-side data (`settings.master_dataset_path`,
`raw_data_dir`, etc.) that has no committed `artifacts/`-tree equivalent
today — see `app/config.py`'s own comment and
`docs/pre_race_materialization.md` for why. That data is LAZILY loaded on
the first `POST /predict` request rather than at startup, since no other
route reads it — paying its full read/memory cost unconditionally at
every process start would be wasted work on a process that never receives
this request — and cached thereafter by
`app.upcoming_prediction_service.ensure_materialization_data()`. This
route itself is a genuinely thin transport layer: it parses the request,
delegates ALL orchestration (calendar/entry-list resolution,
materialization, scoring, the pre-race cache) to
`app.upcoming_prediction_service.resolve_upcoming_prediction()`, maps that
module's plain-Python exceptions to HTTP status codes, and assembles the
response. That service module reuses `src.features.upcoming.next_race()`/
`resolve_entry_list()`, `src.models.materialize.materialize_features()`,
and `src.models.predict.predict_race()` unmodified — this module imports
none of them directly anymore.

API versioning: every route above is defined ONCE on a plain `APIRouter`
(no route logic duplicated) and mounted TWICE — once under
`app.config.API_V1_PREFIX` ("/api/v1", the canonical, documented location,
listed in /docs) and once with no prefix at all (`include_in_schema=False`
— every pre-versioning URL, e.g. plain `/health`, keeps working
identically, for anything still hardcoded to it, but doesn't clutter the
public OpenAPI schema with duplicate entries). Both mounts call the exact
same handler closures and share the exact same `app.state` — there is no
behavioral difference between hitting a route at its versioned or legacy
path, only the URL differs. `docker/Dockerfile.api`'s HEALTHCHECK and this
project's own dashboard both keep working through this — the Dockerfile's
healthcheck deliberately still targets the unversioned `/health` (simpler,
and health checks are conventionally unversioned anyway), while
`app/views/common.py` calls the versioned paths (a first-party client
should use the current version, not lean on the back-compat shim).

Degraded-start policy: if the model or features cannot be loaded, the app
still starts and reports the failure via 503s — starting-and-reporting
beats crash-looping under a scheduler. The pole-baseline model (used only
by /predictions/{race_id}/vs-baseline) degrades independently of the main
model — a failure there doesn't take down the rest of the API.

API hardening: CORSMiddleware with configurable origins
(settings.cors_allow_origins, default empty — deny all cross-origin
browser access) and three exception handlers that add structured logging
to every error path while preserving the exact existing response for
anything that was already a deliberate HTTPException/RequestValidationError;
a bare, unhandled exception (a bug) is the only thing whose response
actually changes — from a generic plaintext 500 to a generic JSON one,
`{"detail": "Internal server error."}`, with nothing internal (message,
type, traceback) ever reaching the client. Full detail always goes to the
server log via `logger.exception(...)`.

**Authentication is deliberately NOT implemented.** This is a public
demonstration deployment of a read-only, historical-data prediction
service — not a multi-user production system: every historically-served
route is GET; `POST /predict` is the one exception, but it
doesn't write or persist anything server-side — like every GET route, it
computes and returns a prediction, just via a POST because it accepts a
request body. There is no concept of a user account, no genuine write/
mutation operation exists anywhere in this API, and nothing served is
private (F1 results are public record). Adding auth here would protect
nothing real while adding a credential-management surface this project has
no actual use for. Revisit only if this API ever gains a genuine
per-user/write capability — `POST /predict`, now implemented, does not
count as one.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import circuit_stats, entity_stats
from app.config import API_V1_PREFIX, Settings
from app.upcoming_prediction_service import (
    RaceAlreadyHasResult,
    resolve_upcoming_prediction,
    resolve_upcoming_race,
)
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.features.qualifying import QUALIFYING_FEATURES
from src.features.upcoming import EntryListEntry
from src.models.predict import load_model, predict_race
from src.models.registry import MODEL_ZOO, get_model, training_schema

logger = logging.getLogger("f1.api")

# Prediction Simulator — of the 10 QUALIFYING_FEATURES, only
# these 3 are literally overridden by a grid-position "what if". The other 7
# (qualifying_position, q1/q2/q3_sec, reached_q2/q3, qualifying_gap_to_pole_pct)
# stay FROZEN at the driver's real values for the race, rather than being
# fabricated to match the new grid slot. This is a deliberate "freeze, don't
# interpolate" choice: grid position legitimately diverges from qualifying
# pace in real F1 (grid penalties, pit-lane starts, grid reshuffles) — the
# training data already contains this pattern, so the model is calibrated for
# it. Interpolating a fake q3_sec/gap-to-pole for the new slot would fabricate
# a lap the driver never set, which this project avoids everywhere else (see
# the no-weather-data / no-fake-timeline decisions).
ADJUSTABLE_GRID_FEATURES: tuple[str, ...] = (
    "grid_adjusted", "grid_position_norm", "pit_lane_start",
)

# pyproject.toml is the single source of truth for the version; the
# fallback covers running from a checkout that was never `pip install -e .`
# -ed (unsupported, but should not crash /health).
try:
    API_VERSION = _package_version("f1-race-winner-prediction")
except PackageNotFoundError:                                    # pragma: no cover
    API_VERSION = "0.0.0+uninstalled"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ModelInfoSchema(BaseModel):
    name: str
    version: str
    alias: str
    run_id: str
    trained_at: str
    calibration: str
    model_class: str


class HealthResponse(BaseModel):
    status: str
    api_version: str
    model: ModelInfoSchema | None = None
    detail: str | None = None


class RaceSummary(BaseModel):
    race_id: int
    year: int
    round: int
    n_drivers: int


class RaceListResponse(BaseModel):
    races: list[RaceSummary]


class UpcomingRaceSchema(BaseModel):
    """Identity only (GET /races/upcoming) — no prediction, no
    materialization_status/caveats/provenance. Those come from POST
    /predict once the caller actually asks for a scored prediction."""
    race_id: int
    year: int
    round: int
    name: str
    circuit_id: int
    date: str


class DriverStandingSchema(BaseModel):
    position: int
    driver_id: int
    driver_name: str | None = None
    points: float
    wins: int


class ConstructorStandingSchema(BaseModel):
    position: int
    constructor_id: int
    constructor_name: str | None = None
    points: float
    wins: int


class StandingsResponse(BaseModel):
    """Final standings as of the season's last race WITH recorded
    standings — for an in-progress season this is the latest completed
    round, not necessarily the full scheduled calendar (same convention as
    app/views/metadata.py::_final_round_standings)."""
    year: int
    drivers: list[DriverStandingSchema]
    constructors: list[ConstructorStandingSchema]


class ModelComparisonResponse(BaseModel):
    """Passthrough of artifacts/analysis/model_comparison.csv — a generic
    tabular shape (not one field per column) since the source CSV's own
    columns vary by candidate (e.g. hyperparameters is only populated for
    tuned candidates) and are recorded at analysis time, not defined by
    this API. Mirrors app/views/insights.py's own
    st.dataframe(pd.read_csv(...)) — "render whatever's in the CSV,"
    verbatim."""
    columns: list[str]
    rows: list[dict[str, float | str | bool | None]]


class FeatureImportanceRow(BaseModel):
    feature: str
    importance: float
    feature_class: str


class ImportanceByClassRow(BaseModel):
    feature_class: str
    total_importance: float


class FeatureImportanceResponse(BaseModel):
    """features: full list, sorted descending by importance (the frontend
    slices to top-20, matching the Streamlit page's own chart). by_class:
    importance summed per feature_class — the same source data as
    features, reproducing app/views/insights.py's separate
    "importance_by_class" figure without a second CSV."""
    features: list[FeatureImportanceRow]
    by_class: list[ImportanceByClassRow]


class ShapSummaryRow(BaseModel):
    feature: str
    mean_abs_shap: float
    feature_class: str


class ShapSummaryResponse(BaseModel):
    """Mean |SHAP| ranking (shap_bar_logreg.png's underlying data) — the
    ONLY SHAP figure with tabular data available; the beeswarm/waterfall
    plots need a per-instance SHAP value matrix that isn't exported (see
    scripts/export_analysis_data.py's docstring, Decision 063)."""
    features: list[ShapSummaryRow]


# --- Entity-stats route family (Frontend Overhaul Phase 5, Decision 064) ---
# Driver Explorer / Compare Drivers / Team all block on this same missing
# capability — designed once here rather than per-page. Every route below
# is independent of _require_ready() (same reasoning as /standings/{year}
# and the analysis routes: this is display metadata from settings.data_dir,
# not a function of the served model) and degrades gracefully (zeros/nulls/
# empty lists, never 404/503) when the underlying entity has no data in the
# requested scope — see app/entity_stats.py for the full rationale.

class DriverStatsResponse(BaseModel):
    driver_id: int
    year: int | None
    races: int = 0
    wins: int = 0
    podiums: int = 0
    poles: int | None = None
    points: float = 0.0
    avg_quali: float | None = None
    avg_finish: float | None = None
    avg_finish_classified: float | None = None
    consistency_std: float | None = None


class RaceResultRow(BaseModel):
    race_id: int
    year: int
    round: int
    quali_position: float | None = None
    finish: float | None = None


class DriverRaceResultsResponse(BaseModel):
    driver_id: int
    year: int | None
    results: list[RaceResultRow]


class StandingsProgressionRow(BaseModel):
    round: int
    points: float
    position: int
    wins: int


class DriverStandingsProgressionResponse(BaseModel):
    driver_id: int
    year: int
    progression: list[StandingsProgressionRow]


class RadarScoresResponse(BaseModel):
    """scores is None when this driver doesn't have enough races this
    season for all three underlying stats, or the field itself is too
    small to normalize against — not an error, a real, expected outcome."""
    driver_id: int
    year: int
    scores: dict[str, float] | None = None


class ConstructorCatalogEntry(BaseModel):
    constructor_id: int
    name: str
    nationality: str | None = None


class ConstructorCatalogResponse(BaseModel):
    constructors: list[ConstructorCatalogEntry]


class ConstructorStatsResponse(BaseModel):
    constructor_id: int
    year: int | None
    races: int = 0
    wins: int = 0
    podiums: int = 0
    points: float = 0.0
    avg_quali: float | None = None


class ConstructorRaceResultsResponse(BaseModel):
    constructor_id: int
    year: int | None
    results: list[RaceResultRow]


class ConstructorDriversResponse(BaseModel):
    constructor_id: int
    year: int
    drivers: list[str]


# --- Circuit-stats routes (Frontend Overhaul Phase 6, Decision 065) --------
# Circuit Explorer's data need doesn't overlap the driver/constructor entity-
# stats family above, but reuses the same conventions: independent of
# _require_ready(), degrades gracefully (empty/null, never 404/503) when a
# circuit has no data — see app/circuit_stats.py for the full rationale.

class CircuitCatalogEntry(BaseModel):
    circuit_id: int
    name: str
    country: str | None = None
    location: str | None = None
    races_held: int = 0


class CircuitCatalogResponse(BaseModel):
    circuits: list[CircuitCatalogEntry]


class CircuitLayoutResponse(BaseModel):
    """coordinates/attribution are both null when this circuit has no
    OpenStreetMap track-outline backfill (most circuits) — not every
    circuit has one, see app/circuit_stats.py::circuit_layout."""
    circuit_id: int
    coordinates: list[list[float]] | None = None
    attribution: str | None = None


class FastestLapInfo(BaseModel):
    time: str
    year: int
    driver: str


class CircuitWinnerRow(BaseModel):
    year: int
    driver: str
    constructor: str | None = None


class MostWinsRow(BaseModel):
    driver: str
    wins: int


class CircuitStatsResponse(BaseModel):
    """Real, computed proxies — not invented "difficulty" scores. Every
    field needs the same caveats app/views/circuit_explorer.py's own UI
    states alongside it (car-performance evolution, strategy/reliability
    confounds); this API doesn't attempt to encode those caveats itself,
    the same way /model-comparison doesn't editorialize its own numbers."""
    circuit_id: int
    races_held: int = 0
    first_year: int | None = None
    last_year: int | None = None
    dnf_rate: float | None = None
    avg_position_change: float | None = None
    fastest_lap: FastestLapInfo | None = None
    most_wins: list[MostWinsRow] = []
    winners_by_year: list[CircuitWinnerRow] = []


class DriverPrediction(BaseModel):
    driver_id: int
    driver_name: str | None = None
    constructor_id: int | None = None
    constructor_name: str | None = None
    predicted_rank: int
    win_probability: float          # per-race-normalized share (user-facing)
    win_probability_raw: float      # calibrated model output


class PredictionResponse(BaseModel):
    prediction_id: str
    race_id: int
    year: int
    round: int
    generated_at: str
    model: ModelInfoSchema
    predictions: list[DriverPrediction]
    actual_winner_driver_id: int | None = None
    model_top1_hit: bool | None = None


class FeatureDebugRow(BaseModel):
    driver_id: int
    features: dict[str, float | None]


class FeatureDebugResponse(BaseModel):
    race_id: int
    model: ModelInfoSchema
    feature_names: list[str]
    rows: list[FeatureDebugRow]


class BaselineComparisonResponse(BaseModel):
    """The calibrated model vs. the grid-only heuristic
    baseline (MODEL_ZOO["pole_baseline"]) for the same race and driver set."""
    race_id: int
    year: int
    round: int
    model: ModelInfoSchema
    baseline_name: str
    baseline_description: str
    model_predictions: list[DriverPrediction]
    baseline_predictions: list[DriverPrediction]
    actual_winner_driver_id: int | None = None
    model_top1_hit: bool | None = None
    baseline_top1_hit: bool | None = None


class SimulateGridResponse(BaseModel):
    """One driver's win share re-scored under an overridden
    grid/qualifying position, everything else held at real values."""
    race_id: int
    driver_id: int
    driver_name: str | None = None
    field_size: int
    real_grid_position: float | None    # driver's actual grid_adjusted, for the slider default
    simulated_grid_position: float      # field_size + 1 when pit_lane_start=True
    pit_lane_start: bool
    real_win_probability: float         # driver's actual per-race-normalized share
    simulated_win_probability: float    # share under the override
    field: list[DriverPrediction]       # full field re-normalized under the override
    locked_qualifying_features: list[str]   # frozen (not fabricated) qualifying-group fields
    locked_features: list[str]              # historical/aggregate features — never adjustable
    model: ModelInfoSchema


class EntryListEntrySchema(BaseModel):
    """One (driverId, constructorId) pairing — the request-body shape for
    an explicit entry-list override. Mirrors
    `src.features.upcoming.EntryListEntry` field-for-field; kept as a
    separate Pydantic model rather than reusing that dataclass directly so
    request-body validation stays entirely a FastAPI/pydantic concern."""
    driver_id: int
    constructor_id: int


class PredictRequest(BaseModel):
    year: int
    round: int
    #: Omit to resolve the current-season roster (see
    #: `resolve_entry_list()`'s backward-looking inference in
    #: src/features/upcoming.py).
    entry_list: list[EntryListEntrySchema] | None = None
    #: ISO-8601, MUST be timezone-aware (e.g. "2026-07-23T10:00:00Z" or
    #: "...+00:00") — a naive timestamp's offset is genuinely unknown to
    #: this API and is rejected with 422 rather than silently assumed to
    #: be UTC. Optional — reserved for a future historical-
    #: cutoff override; validated (never later than the latest local ETL
    #: snapshot) but does not yet change which data is used (see the
    #: route's own docstring for why — no per-row ETL-ingestion timestamp
    #: exists to filter by, a known, disclosed limitation).
    as_of: str | None = None


class ProvenanceSchema(BaseModel):
    """Every prediction must be reconstructable later from its own recorded
    metadata alone: which model, which feature schema, which ETL snapshot,
    and when each step ran."""
    model_version: str
    model_alias: str
    feature_schema_version: str
    etl_snapshot_version: str
    data_as_of: str
    materialized_at: str
    predicted_at: str
    qualifying_status: str          # "not_started" | "in_progress" | "complete"
    completeness_status: str        # mirrors materialization_status


class UpcomingPredictionResponse(BaseModel):
    prediction_id: str
    year: int
    round: int
    materialization_status: str     # "post_qualifying" | "pre_qualifying"
    missing_inputs: list[str]
    generated_at: str
    model: ModelInfoSchema
    predictions: list[DriverPrediction]
    caveats: list[str]
    provenance: ProvenanceSchema


# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------

def _load_name_lookups(settings: Settings) -> tuple[dict, dict]:
    """id -> display-name maps from drivers.csv / constructors.csv.

    Display names are a serving concern only — absence of the CSVs degrades
    names to null, never fails the app."""
    drivers: dict[int, str] = {}
    constructors: dict[int, str] = {}
    try:
        d = pd.read_csv(settings.data_dir / "drivers.csv", na_values=["\\N"])
        drivers = {
            int(r.driverId): f"{r.forename} {r.surname}"
            for r in d.itertuples()
        }
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("driver name lookup unavailable: %s", exc)
    try:
        c = pd.read_csv(settings.data_dir / "constructors.csv", na_values=["\\N"])
        constructors = {int(r.constructorId): str(r.name) for r in c.itertuples()}
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("constructor name lookup unavailable: %s", exc)
    return drivers, constructors


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    logging.basicConfig(level=settings.log_level.upper())

    app.state.model = None
    app.state.model_info = None
    app.state.features = None
    app.state.load_error = None
    app.state.prediction_cache = OrderedDict()   # (model_version, race_id) -> resp
    # Loaded unconditionally, outside the model try/except below: display
    # names (and the /standings route) are a data_dir-only concern,
    # structurally independent of whether the model/features loaded — a
    # degraded model must not also blank out names/standings that are
    # otherwise perfectly available.
    app.state.driver_names, app.state.constructor_names = _load_name_lookups(settings)
    app.state.baseline_model = None
    app.state.baseline_load_error = None
    # POST /predict's own training-side data is intentionally NOT loaded
    # here — it's lazy (see app.upcoming_prediction_service.
    # ensure_materialization_data(), called from the route on first
    # request). Only cheap, no-I/O state is set up at startup.
    app.state.materialization_data = None
    app.state.materialization_load_attempted = False
    app.state.materialization_load_error = None
    app.state.pre_race_cache = OrderedDict()     # cache key -> (result, cached_at)

    try:
        model, info = load_model(settings.serving_bundle_path)
        features = pd.read_parquet(settings.features_path)
        app.state.model, app.state.model_info = model, info
        app.state.features = features
        logger.info(
            "serving model=%s v%s alias=%s calibration=%s | %d feature rows",
            info.name, info.version, info.alias, info.calibration, len(features),
        )

        # vs-baseline route: MODEL_ZOO["pole_baseline"] fit once at
        # startup against the full frozen feature snapshot. "Fit" is a no-op
        # here — PoleSitterBaseline.fit() only records classes_=[0,1] and
        # never looks at X or y (the rule is the fixed "grid_adjusted==1"
        # heuristic) — so this isn't training on serving data, just wiring
        # the ColumnGuard so predict_race() can validate/score it exactly
        # like the real model. Wrapped in its own try/except so a failure
        # here degrades only /vs-baseline, not the whole API.
        try:
            # get_model()'s feature_columns defaults to a curated subset of
            # FEATURE_COLUMNS (an experimental group is excluded from
            # training by default), not the raw full set — but this route scores
            # against the FULL frozen features.parquet snapshot below
            # (features.loc[:, list(FEATURE_COLUMNS)]), so the guard must be
            # explicitly told to expect the FULL set too, or the .fit() call
            # immediately after would raise a column mismatch. Explicit
            # override, not the new default, is deliberate here.
            baseline = get_model(
                "pole_baseline", features[TARGET_COLUMN], feature_columns=FEATURE_COLUMNS,
            )
            baseline.fit(features.loc[:, list(FEATURE_COLUMNS)], features[TARGET_COLUMN])
            app.state.baseline_model = baseline
        except Exception as exc:                                # noqa: BLE001
            app.state.baseline_load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("pole-baseline model unavailable — /vs-baseline "
                           "will 503: %s", app.state.baseline_load_error)
    except Exception as exc:                                    # noqa: BLE001
        # Degraded start: report via /health + 503s.
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        logger.error("startup failed — serving degraded: %s", app.state.load_error)
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="F1 Race Winner Prediction API",
        version=API_VERSION,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # --- CORS --------------------------------------------------------------
    # Always registered; settings.cors_allow_origins (default "") controls
    # the actual allow-list. Deliberately fixed (not configurable) beyond
    # origins: GET/POST cover every route this API actually exposes, this
    # API never uses cookies/auth headers so allow_credentials=False is
    # correct regardless of origin config, and allow_headers=["*"] is safe
    # for a public read-only API with no auth scheme to leak.
    cors_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # --- global exception handling ------------------------------------------
    # All three handlers log first, then either delegate to FastAPI's own
    # default handler (StarletteHTTPException, RequestValidationError —
    # response is BYTE-IDENTICAL to before these handlers existed) or, for
    # a bare unhandled Exception only, return a new generic-but-safe JSON
    # 500 instead of Starlette's default plaintext one. No response body
    # ever includes exception internals; full detail goes to the server
    # log only (logger.exception below captures the traceback).

    @app.exception_handler(StarletteHTTPException)
    async def _log_http_exception(request: Request, exc: StarletteHTTPException):
        logger.log(
            logging.WARNING if exc.status_code < 500 else logging.ERROR,
            "http_exception method=%s path=%s status_code=%d detail=%s",
            request.method, request.url.path, exc.status_code, exc.detail,
        )
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _log_validation_exception(request: Request, exc: RequestValidationError):
        logger.warning(
            "validation_error method=%s path=%s errors=%s",
            request.method, request.url.path, exc.errors(),
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def _handle_unexpected_exception(request: Request, exc: Exception):
        logger.exception(
            "unhandled_exception method=%s path=%s exception_type=%s",
            request.method, request.url.path, type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    # --- versioned router ----------------------------------------------------
    # Every route below is defined once on this router and mounted twice —
    # see the module docstring's "API versioning" paragraph for the full
    # rationale. `router` carries no prefix of its own; both prefixes are
    # applied at include_router() time, at the bottom of this function.
    router = APIRouter()

    # --- helpers -----------------------------------------------------------

    def _require_ready() -> None:
        if app.state.model is None or app.state.features is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {app.state.load_error or 'not loaded'}",
            )

    def _model_schema() -> ModelInfoSchema:
        return ModelInfoSchema(**app.state.model_info.to_dict())

    def _race_rows(race_id: int) -> pd.DataFrame:
        features: pd.DataFrame = app.state.features
        rows = features[features["raceId"] == race_id]
        if rows.empty:
            raise HTTPException(404, detail=f"raceId {race_id} not found.")
        return rows

    def _driver_predictions(scored: pd.DataFrame) -> list[DriverPrediction]:
        """predict_race() output -> DriverPrediction list, with display-name
        enrichment. Shared by /predictions, /simulate, and /vs-baseline so
        the three routes can never drift in how a scored frame is rendered."""
        return [
            DriverPrediction(
                driver_id=int(r.driverId),
                driver_name=app.state.driver_names.get(int(r.driverId)),
                constructor_id=(int(r.constructorId)
                                if hasattr(r, "constructorId") else None),
                constructor_name=app.state.constructor_names.get(
                    int(r.constructorId)) if hasattr(r, "constructorId") else None,
                predicted_rank=int(r.predicted_rank),
                win_probability=float(r.win_probability),
                win_probability_raw=float(r.win_probability_raw),
            )
            for r in scored.itertuples()
        ]

    def _actual_winner(rows: pd.DataFrame) -> int | None:
        if "winner" in rows.columns and rows["winner"].notna().all():
            winners = rows.loc[rows["winner"] == 1, "driverId"]
            return int(winners.iloc[0]) if len(winners) == 1 else None
        return None

    def _final_round_standings(standings_csv: str, year: int) -> pd.DataFrame:
        """Standings rows keyed to the season's LAST race that has
        standings recorded — mirrors
        app/views/metadata.py::_final_round_standings exactly (same Ergast
        convention: for an in-progress season this is the latest completed
        round, not the full scheduled calendar). Reads directly from
        settings.data_dir; degrades to an empty frame, never raises, when
        the CSVs are absent or the year has no matching rows."""
        try:
            standings = pd.read_csv(settings.data_dir / standings_csv, na_values=["\\N"])
            races = pd.read_csv(settings.data_dir / "races.csv", na_values=["\\N"])
        except Exception as exc:                                # noqa: BLE001
            logger.warning("standings lookup unavailable (%s): %s", standings_csv, exc)
            return pd.DataFrame()
        season = races.loc[races["year"] == year, ["raceId", "round"]]
        merged = standings.merge(season, on="raceId", how="inner")
        if merged.empty:
            return pd.DataFrame()
        return merged[merged["round"] == merged["round"].max()].copy()

    # --- routes ------------------------------------------------------------

    @router.get("/health", response_model=HealthResponse)
    def health():
        if app.state.model is None:
            return HealthResponse(
                status="degraded", api_version=API_VERSION,
                detail=app.state.load_error or "model not loaded",
            )
        return HealthResponse(
            status="ok", api_version=API_VERSION, model=_model_schema(),
        )

    @router.get("/model", response_model=ModelInfoSchema)
    def model_info():
        _require_ready()
        return _model_schema()

    @router.get("/races", response_model=RaceListResponse)
    def races(year: int | None = None):
        _require_ready()
        features: pd.DataFrame = app.state.features
        served = features
        if year is not None:
            served = served[served["year"] == year]
        summary = (
            served.groupby(["raceId", "year", "round"], as_index=False)
            .size()
            .sort_values(["year", "round"])
        )
        return RaceListResponse(races=[
            RaceSummary(race_id=int(r.raceId), year=int(r.year),
                        round=int(r.round), n_drivers=int(r.size))
            for r in summary.itertuples()
        ])

    @router.get("/races/upcoming", response_model=UpcomingRaceSchema)
    def upcoming_race():
        """Identity of the single next race with no result yet (this
        pipeline only ever resolves the earliest unplayed race), for the
        dashboard's picker — NOT a prediction. Thin transport only, same as
        POST /predict: delegates
        to app.upcoming_prediction_service.resolve_upcoming_race(), maps
        RuntimeError -> 503 (materialization data unavailable, mirrors
        POST /predict's own degraded-state handling) and "no upcoming
        race" -> 404 (nothing found, not a bad request)."""
        try:
            race = resolve_upcoming_race(app.state, settings)
        except RuntimeError as exc:
            raise HTTPException(
                503, detail=f"Upcoming race lookup not available: {exc}") from exc
        if race is None:
            raise HTTPException(
                404, detail="No upcoming race is scheduled — "
                            "every calendar race already has a result.")
        return UpcomingRaceSchema(
            race_id=race.race_id, year=race.year, round=race.round,
            name=race.name, circuit_id=race.circuit_id, date=race.date,
        )

    @router.get("/standings/{year}", response_model=StandingsResponse)
    def standings(year: int):
        """Championship standings (drivers + constructors) for one season —
        display metadata only, independent of _require_ready(): a degraded
        model must not also take down this route, since it reads solely
        from settings.data_dir, never the model or features snapshot.
        Degrades to empty drivers/constructors lists (not 404/503) when the
        CSVs are absent or the year has no recorded standings yet — the
        same graceful-absence behavior every other data_dir-sourced lookup
        in this API already follows (contrast with /predictions/{race_id},
        which 404s on an unknown raceId because there the caller supplied a
        bad reference, not because data happens to be missing)."""
        driver_rows = _final_round_standings("driver_standings.csv", year)
        drivers = [
            DriverStandingSchema(
                position=int(r.position),
                driver_id=int(r.driverId),
                driver_name=app.state.driver_names.get(int(r.driverId)),
                points=float(r.points),
                wins=int(r.wins),
            )
            for r in driver_rows.sort_values("position").itertuples()
        ] if not driver_rows.empty else []

        constructor_rows = _final_round_standings("constructor_standings.csv", year)
        constructors = [
            ConstructorStandingSchema(
                position=int(r.position),
                constructor_id=int(r.constructorId),
                constructor_name=app.state.constructor_names.get(int(r.constructorId)),
                points=float(r.points),
                wins=int(r.wins),
            )
            for r in constructor_rows.sort_values("position").itertuples()
        ] if not constructor_rows.empty else []

        return StandingsResponse(year=year, drivers=drivers, constructors=constructors)

    def _read_analysis_csv(name: str) -> pd.DataFrame:
        """settings.analysis_data_dir CSV -> DataFrame; EMPTY frame (never
        raises) when absent — the same graceful-absence convention as
        _final_round_standings above, for the same reason: Model Insights
        data is a recorded-at-promotion-time snapshot, independent of
        whether the live model/features loaded."""
        try:
            return pd.read_csv(settings.analysis_data_dir / name)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("analysis data unavailable (%s): %s", name, exc)
            return pd.DataFrame()

    @router.get("/model-comparison", response_model=ModelComparisonResponse)
    def model_comparison():
        """Model-selection comparison table (artifacts/analysis/
        model_comparison.csv) — independent of _require_ready(), same
        reasoning as /standings: this is recorded analysis data, not a
        function of the currently-loaded model. Degrades to an empty table
        (200, never 404/503) when the CSV is absent."""
        frame = _read_analysis_csv("model_comparison.csv")
        if frame.empty:
            return ModelComparisonResponse(columns=[], rows=[])
        return ModelComparisonResponse(
            columns=list(frame.columns),
            rows=frame.where(frame.notna(), None).to_dict(orient="records"),
        )

    @router.get("/feature-importance", response_model=FeatureImportanceResponse)
    def feature_importance():
        """Native (|coefficient|) feature importance for the logreg model
        family (artifacts/analysis/feature_importance_logreg.csv), plus the
        same data summed by feature_class — reproduces both
        feature_importance_logreg.png and importance_by_class_logreg.png
        from one source file, no second CSV needed. Independent of
        _require_ready(), same reasoning as /standings and
        /model-comparison."""
        frame = _read_analysis_csv("feature_importance_logreg.csv")
        if frame.empty:
            return FeatureImportanceResponse(features=[], by_class=[])
        sorted_frame = frame.sort_values("importance", ascending=False)
        by_class = (
            frame.groupby("feature_class")["importance"].sum()
            .sort_values(ascending=False)
        )
        return FeatureImportanceResponse(
            features=[
                FeatureImportanceRow(
                    feature=str(r.feature), importance=float(r.importance),
                    feature_class=str(r.feature_class),
                )
                for r in sorted_frame.itertuples()
            ],
            by_class=[
                ImportanceByClassRow(feature_class=str(cls), total_importance=float(total))
                for cls, total in by_class.items()
            ],
        )

    @router.get("/shap-summary", response_model=ShapSummaryResponse)
    def shap_summary():
        """Mean |SHAP| ranking for the logreg model family (artifacts/
        analysis/shap_summary_logreg.csv) — reproduces shap_bar_logreg.png.
        The SHAP beeswarm/waterfall figures are NOT reproducible from this
        CSV (they need a per-instance SHAP value matrix, not exported —
        see scripts/export_analysis_data.py's docstring, Decision 063) and
        have no route here. Independent of _require_ready(), same
        reasoning as the routes above."""
        frame = _read_analysis_csv("shap_summary_logreg.csv")
        if frame.empty:
            return ShapSummaryResponse(features=[])
        sorted_frame = frame.sort_values("mean_abs_shap", ascending=False)
        return ShapSummaryResponse(features=[
            ShapSummaryRow(
                feature=str(r.feature), mean_abs_shap=float(r.mean_abs_shap),
                feature_class=str(r.feature_class),
            )
            for r in sorted_frame.itertuples()
        ])

    def _race_result_rows(frame: pd.DataFrame) -> list[RaceResultRow]:
        """driver_race_results()/constructor_race_results() -> RaceResultRow
        list — shared by both entity grains, same column shape."""
        if frame.empty:
            return []
        return [
            RaceResultRow(
                race_id=int(r.raceId), year=int(r.year), round=int(r.round),
                quali_position=(float(r.quali_position)
                                if hasattr(r, "quali_position") and pd.notna(r.quali_position)
                                else None),
                finish=(float(r.finish) if pd.notna(r.finish) else None),
            )
            for r in frame.itertuples()
        ]

    @router.get("/drivers/{driver_id}/stats", response_model=DriverStatsResponse)
    def driver_stats(driver_id: int, year: int | None = None):
        """Historical outcome stats (races/wins/podiums/poles/points/
        avg_quali/avg_finish) for one driver — one season, or career when
        year is omitted. Degrades to all-zero/null fields (200, never
        404/503) when this driver has no data in the requested scope."""
        stats = entity_stats.driver_season_stats(settings.data_dir, driver_id, year)
        return DriverStatsResponse(driver_id=driver_id, year=year, **stats)

    @router.get("/drivers/{driver_id}/race-results", response_model=DriverRaceResultsResponse)
    def driver_race_results(driver_id: int, year: int | None = None):
        """(year, round)-sorted per-race grid/quali/finish rows for one
        driver — Driver Explorer's trend-chart source."""
        frame = entity_stats.driver_race_results(settings.data_dir, driver_id, year)
        return DriverRaceResultsResponse(
            driver_id=driver_id, year=year, results=_race_result_rows(frame))

    @router.get("/drivers/{driver_id}/standings-progression",
             response_model=DriverStandingsProgressionResponse)
    def driver_standings_progression(driver_id: int, year: int):
        """Round-by-round championship points/position for one
        driver-season — season-scoped only (a career-spanning points
        progression doesn't mean anything, year is required)."""
        frame = entity_stats.driver_standings_progression(settings.data_dir, driver_id, year)
        progression = [
            StandingsProgressionRow(
                round=int(r.round), points=float(r.points),
                position=int(r.position), wins=int(r.wins),
            )
            for r in frame.itertuples()
        ] if not frame.empty else []
        return DriverStandingsProgressionResponse(
            driver_id=driver_id, year=year, progression=progression)

    @router.get("/drivers/{driver_id}/radar", response_model=RadarScoresResponse)
    def driver_radar(driver_id: int, year: int):
        """Qualifying pace / Race pace / Consistency, 0-100, normalized
        within this season's field only (not comparable across seasons —
        see app/entity_stats.py::radar_scores). Season-scoped only, year
        is required. scores is null (not an error) when this driver lacks
        enough races this season, or the field is too small to normalize."""
        scores = entity_stats.radar_scores(settings.data_dir, driver_id, year)
        return RadarScoresResponse(driver_id=driver_id, year=year, scores=scores)

    @router.get("/constructors", response_model=ConstructorCatalogResponse)
    def constructors():
        """Every constructor in the display-metadata catalog, name-sorted —
        the Team page's picker source (independent of any season/scope)."""
        frame = entity_stats.constructor_catalog(settings.data_dir)
        return ConstructorCatalogResponse(constructors=[
            ConstructorCatalogEntry(
                constructor_id=int(r.constructorId), name=str(r.name),
                nationality=(str(r.nationality) if pd.notna(r.nationality) else None),
            )
            for r in frame.itertuples()
        ] if not frame.empty else [])

    @router.get("/constructors/{constructor_id}/stats", response_model=ConstructorStatsResponse)
    def constructor_stats(constructor_id: int, year: int | None = None):
        """Historical outcome stats for one constructor — mirrors
        /drivers/{id}/stats at constructor grain."""
        stats = entity_stats.constructor_season_stats(settings.data_dir, constructor_id, year)
        return ConstructorStatsResponse(constructor_id=constructor_id, year=year, **stats)

    @router.get("/constructors/{constructor_id}/race-results",
             response_model=ConstructorRaceResultsResponse)
    def constructor_race_results(constructor_id: int, year: int | None = None):
        """(year, round)-sorted per-race rows for one constructor: mean
        qualifying position across both cars, best finish that round —
        the Team page's trend-chart source."""
        frame = entity_stats.constructor_race_results(settings.data_dir, constructor_id, year)
        return ConstructorRaceResultsResponse(
            constructor_id=constructor_id, year=year, results=_race_result_rows(frame))

    @router.get("/constructors/{constructor_id}/drivers", response_model=ConstructorDriversResponse)
    def constructor_drivers(constructor_id: int, year: int):
        """Driver display names who raced for this constructor in this
        season — season-scoped only, year is required (a roster is a
        per-season concept)."""
        names = entity_stats.constructor_current_drivers(settings.data_dir, constructor_id, year)
        return ConstructorDriversResponse(constructor_id=constructor_id, year=year, drivers=names)

    @router.get("/circuits", response_model=CircuitCatalogResponse)
    def circuits():
        """Every circuit in the display-metadata catalog, name-sorted —
        the Circuit Explorer picker source."""
        frame = circuit_stats.circuit_catalog(settings.data_dir)
        return CircuitCatalogResponse(circuits=[
            CircuitCatalogEntry(
                circuit_id=int(r.circuitId), name=str(r.name),
                country=(str(r.country) if pd.notna(r.country) else None),
                location=(str(r.location) if pd.notna(r.location) else None),
                races_held=int(r.races_held),
            )
            for r in frame.itertuples()
        ] if not frame.empty else [])

    @router.get("/circuits/{circuit_id}/layout", response_model=CircuitLayoutResponse)
    def circuit_layout(circuit_id: int):
        """Track-outline GeoJSON for one circuit — coordinates/attribution
        are both null (200, never 404) when this circuit has no
        OpenStreetMap backfill, matching every other data_dir-sourced
        lookup's graceful-absence behavior."""
        feature = circuit_stats.circuit_layout(settings.data_dir, circuit_id)
        if feature is None:
            return CircuitLayoutResponse(circuit_id=circuit_id)
        return CircuitLayoutResponse(
            circuit_id=circuit_id,
            coordinates=feature["geometry"]["coordinates"],
            attribution=feature["properties"]["attribution"],
        )

    @router.get("/circuits/{circuit_id}/stats", response_model=CircuitStatsResponse)
    def circuit_stats_route(circuit_id: int):
        """All-time stats for one circuit — races held, years active, DNF
        rate, average grid->finish position change, fastest recorded race
        lap, and win-count leaders. Degrades to zero/null/empty (200,
        never 404) when this circuit has no race history in the served
        data."""
        stats = circuit_stats.circuit_stats(settings.data_dir, circuit_id)
        most_wins = stats.get("most_wins")
        winners_by_year = stats.get("winners_by_year")
        fastest_lap = stats.get("fastest_lap")
        return CircuitStatsResponse(
            circuit_id=circuit_id,
            races_held=stats.get("races_held", 0),
            first_year=stats.get("first_year"),
            last_year=stats.get("last_year"),
            dnf_rate=stats.get("dnf_rate"),
            avg_position_change=stats.get("avg_position_change"),
            fastest_lap=(FastestLapInfo(**fastest_lap) if fastest_lap else None),
            most_wins=(
                [MostWinsRow(driver=str(r.driver), wins=int(r.wins))
                 for r in most_wins.itertuples()]
                if most_wins is not None else []
            ),
            winners_by_year=(
                [CircuitWinnerRow(
                    year=int(r.year), driver=str(r.driver),
                    constructor=(str(r.constructor)
                                 if hasattr(r, "constructor") and pd.notna(r.constructor)
                                 else None),
                )
                 for r in winners_by_year.itertuples()]
                if winners_by_year is not None else []
            ),
        )

    @router.get("/predictions/{race_id}", response_model=PredictionResponse)
    def predictions(race_id: int, request: Request):
        _require_ready()
        started = time.perf_counter()
        prediction_id = str(uuid.uuid4())
        info = app.state.model_info

        cache: OrderedDict = app.state.prediction_cache
        cache_key = (info.version, race_id)
        cache_hit = cache_key in cache

        if cache_hit:
            response = cache[cache_key]
        else:
            rows = _race_rows(race_id)
            scored = predict_race(app.state.model, rows)

            winner_id = _actual_winner(rows)
            top_pick = int(scored.iloc[0]["driverId"])

            response = PredictionResponse(
                prediction_id=prediction_id,
                race_id=race_id,
                year=int(rows["year"].iloc[0]),
                round=int(rows["round"].iloc[0]),
                generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                model=_model_schema(),
                predictions=_driver_predictions(scored),
                actual_winner_driver_id=winner_id,
                model_top1_hit=(winner_id == top_pick) if winner_id is not None else None,
            )
            cache[cache_key] = response
            while len(cache) > settings.prediction_cache_size:
                cache.popitem(last=False)                       # FIFO eviction

        # Structured prediction log. A cache hit
        # reuses the cached body but gets its own prediction_id in the log.
        latency_ms = (time.perf_counter() - started) * 1e3
        logger.info(
            "prediction prediction_id=%s race_id=%d model_version=%s "
            "model_alias=%s n_drivers=%d cache_hit=%s latency_ms=%.2f status_code=200",
            prediction_id, race_id, info.version, info.alias,
            len(response.predictions), cache_hit, latency_ms,
        )
        return response

    @router.get("/predictions/{race_id}/simulate/{driver_id}",
             response_model=SimulateGridResponse)
    def simulate_grid(race_id: int, driver_id: int,
                      grid_position: int | None = None, pit_lane: bool = False):
        """Prediction Simulator (grid/qualifying group only).

        Re-scores ONE driver's row with `grid_adjusted`/`grid_position_norm`/
        `pit_lane_start` overridden (see ADJUSTABLE_GRID_FEATURES); every
        other feature — including the rest of the qualifying group and all
        21 historical/standings aggregates — is held at the driver's real
        value for this race. Rescoring the whole field (not just the one
        row) means the returned `field` reflects the real, sum-normalization
        redistribution: the target driver's `win_probability_raw` is the
        only one that changes, but every driver's normalized `win_probability`
        shifts a little to keep the race summing to 1 — that redistribution
        IS the "other drivers' context" the simulator provides, at the cost
        of one extra predict_proba call already this cheap for a single race.
        """
        _require_ready()
        rows = _race_rows(race_id)
        if driver_id not in set(rows["driverId"]):
            raise HTTPException(
                404, detail=f"driverId {driver_id} not in race {race_id}.")
        if not pit_lane and grid_position is None:
            raise HTTPException(
                422, detail="Provide grid_position (1..field size) or set pit_lane=true.")

        field_size = len(rows)
        if not pit_lane and not (1 <= grid_position <= field_size):
            raise HTTPException(
                422,
                detail=f"grid_position must be between 1 and {field_size} for "
                       f"this race ({field_size} entries) — or set pit_lane=true.",
            )

        real_row = rows.loc[rows["driverId"] == driver_id].iloc[0]
        mask = rows["driverId"] == driver_id
        modified = rows.copy()
        # Normalize to float64 up front — ColumnGuard casts the whole design
        # matrix to float64 at score time anyway (registry.py), so this loses
        # nothing, and it avoids a pandas dtype-mismatch warning when writing
        # a Python bool into what may be a bool- or float-dtyped column
        # depending on the source frame.
        for col in ADJUSTABLE_GRID_FEATURES:
            modified[col] = modified[col].astype(float)
        if pit_lane:
            modified.loc[mask, "pit_lane_start"] = 1.0
            modified.loc[mask, "grid_adjusted"] = float(field_size + 1)
            modified.loc[mask, "grid_position_norm"] = 1.0
            sim_grid = float(field_size + 1)
        else:
            modified.loc[mask, "pit_lane_start"] = 0.0
            modified.loc[mask, "grid_adjusted"] = float(grid_position)
            modified.loc[mask, "grid_position_norm"] = float(grid_position) / field_size
            sim_grid = float(grid_position)

        real_scored = predict_race(app.state.model, rows)
        sim_scored = predict_race(app.state.model, modified)
        real_prob = float(
            real_scored.loc[real_scored["driverId"] == driver_id, "win_probability"].iloc[0])
        sim_prob = float(
            sim_scored.loc[sim_scored["driverId"] == driver_id, "win_probability"].iloc[0])

        schema = training_schema(app.state.model)["feature_names"]
        qualifying_in_schema = [f for f in QUALIFYING_FEATURES if f in schema]
        locked_qualifying = [f for f in qualifying_in_schema
                             if f not in ADJUSTABLE_GRID_FEATURES]
        locked_other = [f for f in schema if f not in qualifying_in_schema]

        real_grid = real_row.get("grid_adjusted")
        return SimulateGridResponse(
            race_id=race_id,
            driver_id=driver_id,
            driver_name=app.state.driver_names.get(driver_id),
            field_size=field_size,
            real_grid_position=(float(real_grid) if pd.notna(real_grid) else None),
            simulated_grid_position=sim_grid,
            pit_lane_start=bool(pit_lane),
            real_win_probability=real_prob,
            simulated_win_probability=sim_prob,
            field=_driver_predictions(sim_scored),
            locked_qualifying_features=locked_qualifying,
            locked_features=locked_other,
            model=_model_schema(),
        )

    @router.get("/predictions/{race_id}/vs-baseline",
             response_model=BaselineComparisonResponse)
    def predictions_vs_baseline(race_id: int):
        """Qualifying Impact: the calibrated model next to
        MODEL_ZOO["pole_baseline"] (P(win)=1 iff grid_adjusted==1) for the
        same race and driver set — "here's what qualifying position alone
        predicts vs. what the full model predicts," grounded entirely in
        artifacts that already exist (no fabricated FP1-FP3 narrative)."""
        _require_ready()
        if app.state.baseline_model is None:
            raise HTTPException(
                503,
                detail="Baseline model not available: "
                       f"{app.state.baseline_load_error or 'not loaded'}",
            )
        rows = _race_rows(race_id)
        model_scored = predict_race(app.state.model, rows)
        baseline_scored = predict_race(app.state.baseline_model, rows)

        winner_id = _actual_winner(rows)
        model_top1 = int(model_scored.iloc[0]["driverId"])
        baseline_top1 = int(baseline_scored.iloc[0]["driverId"])
        spec = MODEL_ZOO["pole_baseline"]

        return BaselineComparisonResponse(
            race_id=race_id,
            year=int(rows["year"].iloc[0]),
            round=int(rows["round"].iloc[0]),
            model=_model_schema(),
            baseline_name=spec.name,
            baseline_description=spec.description,
            model_predictions=_driver_predictions(model_scored),
            baseline_predictions=_driver_predictions(baseline_scored),
            actual_winner_driver_id=winner_id,
            model_top1_hit=(winner_id == model_top1) if winner_id is not None else None,
            baseline_top1_hit=(winner_id == baseline_top1) if winner_id is not None else None,
        )

    @router.get("/debug/features/{race_id}", response_model=FeatureDebugResponse)
    def debug_features(race_id: int):
        if not settings.debug_endpoints:
            # Indistinguishable from an unknown route in production.
            raise HTTPException(404, detail="Not Found")
        _require_ready()
        rows = _race_rows(race_id)
        schema = training_schema(app.state.model)["feature_names"]
        X = rows.loc[:, schema].astype(float)
        return FeatureDebugResponse(
            race_id=race_id,
            model=_model_schema(),
            feature_names=schema,
            rows=[
                FeatureDebugRow(
                    driver_id=int(driver_id),
                    features={
                        name: (None if pd.isna(value) else float(value))
                        for name, value in zip(schema, values)
                    },
                )
                for driver_id, values in zip(rows["driverId"], X.to_numpy())
            ],
        )

    @router.post("/predict", response_model=UpcomingPredictionResponse)
    def predict_upcoming(body: PredictRequest):
        """Score the single next race with no result yet (this pipeline
        only ever resolves the earliest unplayed race) with the served
        bundle.

        Thin transport only: parses the request, delegates every bit of
        orchestration (calendar/entry-list resolution, materialization,
        scoring, the pre-race cache) to
        `app.upcoming_prediction_service.resolve_upcoming_prediction()`,
        maps its plain-Python exceptions to HTTP status codes, and
        assembles the response. See that module for why the composition
        lives there instead of `src.models.predict_upcoming.
        predict_upcoming_race()`.
        """
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {app.state.load_error or 'not loaded'}",
            )

        override = (
            [EntryListEntry(driver_id=e.driver_id, constructor_id=e.constructor_id)
             for e in body.entry_list]
            if body.entry_list is not None else None
        )

        try:
            result, cache_hit = resolve_upcoming_prediction(
                app.state, settings, app.state.model, app.state.model_info,
                year=body.year, round_=body.round,
                entry_list_override=override, as_of=body.as_of,
            )
        except RaceAlreadyHasResult as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                503, detail=f"Upcoming-race prediction not available: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

        info = app.state.model_info
        prediction_id = str(uuid.uuid4())
        caveats = [
            "grid_adjusted/grid_position_norm are sourced from qualifying_position — a "
            "post-qualifying grid penalty or pit-lane start is not yet reflected "
            "(a documented, interim proxy — see docs/pre_race_materialization.md).",
        ]
        if result.materialization_status == "pre_qualifying":
            caveats.append(
                "Qualifying has not been fully recorded for this race yet; this prediction "
                "relies on informative-missingness handling that has not been separately "
                "validated for a pre-qualifying regime (deliberately deferred — scoring "
                "before qualifying results exist at all is a future extension, not yet "
                "implemented)."
            )

        response = UpcomingPredictionResponse(
            prediction_id=prediction_id,
            year=body.year,
            round=body.round,
            materialization_status=result.materialization_status,
            missing_inputs=result.missing_inputs,
            generated_at=result.predicted_at.isoformat(timespec="seconds"),
            model=_model_schema(),
            predictions=_driver_predictions(result.predictions),
            caveats=caveats,
            provenance=ProvenanceSchema(
                model_version=info.version,
                model_alias=info.alias,
                feature_schema_version=result.feature_schema_version,
                etl_snapshot_version=result.etl_snapshot_version.isoformat(),
                data_as_of=(body.as_of or result.etl_snapshot_version.isoformat()),
                materialized_at=result.materialized_at.isoformat(timespec="seconds"),
                predicted_at=result.predicted_at.isoformat(timespec="seconds"),
                qualifying_status=result.qualifying_status,
                completeness_status=result.materialization_status,
            ),
        )

        logger.info(
            "upcoming_prediction prediction_id=%s year=%d round=%d model_version=%s "
            "cache_hit=%s status_code=200",
            prediction_id, body.year, body.round, info.version, cache_hit,
        )
        return response

    # --- mount the router twice ----------------------------------------------
    # Canonical, documented: /api/v1/... . Legacy back-compat: every
    # pre-versioning path (e.g. bare /health) keeps working identically —
    # hidden from /docs/OpenAPI so the schema only ever shows one contract
    # per route, not two. Both mounts share the same handler closures and
    # app.state; there is no behavioral difference, only the URL.
    app.include_router(router, prefix=API_V1_PREFIX)
    app.include_router(router, include_in_schema=False)

    return app


app = create_app()
