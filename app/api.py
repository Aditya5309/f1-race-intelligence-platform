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
inject. `POST /predict` (Phase 7) is the one exception to "clients never
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
    GET  /api/v1/races?year=                         races available to score (<= serve_max_year)
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
    POST /api/v1/predict            Upcoming-race prediction (Phase 7,
                                    Decisions 049/050): materializes the
                                    single next race with no result yet
                                    (Decision 050's horizon=1) and scores it
                                    with the same served bundle. See the
                                    "Upcoming-race prediction" section below.

Upcoming-race prediction (`POST /predict`, Phase 7): unlike every route
above, this one needs training-side data (`settings.master_dataset_path`,
`raw_data_dir`, etc.) that has no committed `artifacts/`-tree equivalent
today — see `app/config.py`'s own comment and `context/decisions.md`
Decision 052 for why. That data is LAZILY loaded on the first `POST
/predict` request (not at startup — a stability-review finding: no other
route reads it, so paying its full read/memory cost unconditionally at
every process start would be wasted work on a process that never receives
this request) and cached thereafter by
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
route is GET; `POST /predict` (Phase 7) is the one exception, but it
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

from app.config import API_V1_PREFIX, Settings
from app.upcoming_prediction_service import (
    RaceAlreadyHasResult,
    resolve_upcoming_prediction,
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
    #: `resolve_entry_list()`'s backward-looking inference — Phase 1).
    entry_list: list[EntryListEntrySchema] | None = None
    #: ISO-8601, MUST be timezone-aware (e.g. "2026-07-23T10:00:00Z" or
    #: "...+00:00") — a naive timestamp's offset is genuinely unknown to
    #: this API and is rejected with 422 rather than silently assumed to
    #: be UTC (Decision 052). Optional — reserved for a future historical-
    #: cutoff override; validated (never later than the latest local ETL
    #: snapshot) but does not yet change which data is used (see the
    #: route's own docstring for why — no per-row ETL-ingestion timestamp
    #: exists to filter by, a known, disclosed limitation).
    as_of: str | None = None


class ProvenanceSchema(BaseModel):
    """Decision 049 Refinement 6: every prediction must be reconstructable
    later from its own recorded metadata alone."""
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
    app.state.driver_names, app.state.constructor_names = {}, {}
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
        app.state.driver_names, app.state.constructor_names = (
            _load_name_lookups(settings)
        )
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
        year = int(rows["year"].iloc[0])
        if year > settings.serve_max_year:
            raise HTTPException(
                409,
                detail=f"raceId {race_id} ({year}) is in the forward holdout "
                       f"(> {settings.serve_max_year}) — reserved as a "
                       "genuinely unseen evaluation window.",
            )
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
        served = features[features["year"] <= settings.serve_max_year]
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
        """Score the single next race with no result yet (Decision 050's
        horizon=1) with the served bundle.

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
            "(Decision 049/050's documented interim proxy).",
        ]
        if result.materialization_status == "pre_qualifying":
            caveats.append(
                "Qualifying has not been fully recorded for this race yet; this prediction "
                "relies on informative-missingness handling that has not been separately "
                "validated for a pre-qualifying regime (design doc Phase 9, deferred)."
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
