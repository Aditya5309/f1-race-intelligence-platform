"""
app/api.py

FastAPI serving layer (Decision 016; reports/application_design.md;
Decision 026/027/029).

    uvicorn app.api:app --reload

The app is deliberately logic-free: HTTP concerns in, exactly two calls to
the tested prediction layer out — `predict.load_model()` once at startup,
`predict.predict_race()` per request. Feature rows are looked up SERVER-SIDE
from the frozen runtime features snapshot (settings.features_path, default
artifacts/features.parquet) by raceId (clients never send feature payloads —
features are derived artifacts of the leakage-audited pipeline; design §1).

`predict.load_model()` reads a frozen serving bundle (settings.
serving_bundle_path, default artifacts/serving/staging) — no live MLflow
tracking server, SQLite registry, or mlruns/ directory is required at
runtime (Decision 026/027). Both runtime artifacts live under the committed
artifacts/ tree (Decision 029) — the deployed API needs nothing from the
gitignored data/ training tree to serve predictions. This module has no
concept of experiments or registry aliases; that machinery lives entirely
on the training side (src/models/train.py).

Endpoints (design §5):
    GET  /health                              liveness + serving model metadata
    GET  /model                               full ModelInfo
    GET  /races?year=                         races available to score (<= serve_max_year)
    GET  /predictions/{race_id}               per-race-normalized field predictions
    GET  /predictions/{race_id}/simulate/{driver_id}
                                              Phase 3 Item 1 (Prediction Simulator):
                                              re-score one driver with an overridden
                                              grid/qualifying position, everything
                                              else held at real values — see
                                              ADJUSTABLE_GRID_FEATURES below for why
                                              only 3 of the 10 qualifying-group
                                              features actually move.
    GET  /predictions/{race_id}/vs-baseline   Phase 3 Item 2 (Qualifying Impact):
                                              the calibrated model's predictions next
                                              to MODEL_ZOO["pole_baseline"]'s (grid-
                                              only heuristic) for the same race.
    GET  /debug/features/{race_id}  dev-only (F1_DEBUG_ENDPOINTS=true) —
                                    the exact feature vectors fed to the model
    POST /predict                   RESERVED for Phase 8 upcoming-race scoring;
                                    returns 501 (design §5 amendment)

Degraded-start policy (§7): if the model or features cannot be loaded, the
app still starts and reports the failure via 503s — starting-and-reporting
beats crash-looping under a scheduler. The pole-baseline model (used only by
/predictions/{race_id}/vs-baseline) degrades independently of the main
model — a failure there doesn't take down the rest of the API.
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
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.config import Settings
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.features.qualifying import QUALIFYING_FEATURES
from src.models.predict import load_model, predict_race
from src.models.registry import MODEL_ZOO, get_model, training_schema

logger = logging.getLogger("f1.api")

# Phase 3 Item 1 (Prediction Simulator) — of the 10 QUALIFYING_FEATURES, only
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

# Single-source version (Decision 020): pyproject.toml is the only place the
# version is defined; the fallback covers running from a checkout that was
# never `pip install -e .`-ed (unsupported, but should not crash /health).
try:
    API_VERSION = _package_version("f1-race-winner-prediction")
except PackageNotFoundError:                                    # pragma: no cover
    API_VERSION = "0.0.0+uninstalled"


# ---------------------------------------------------------------------------
# Response schemas (design §6)
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
    """Phase 3 Item 2 — the calibrated model vs. the grid-only heuristic
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
    """Phase 3 Item 1 — one driver's win share re-scored under an overridden
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


# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------

def _load_name_lookups(settings: Settings) -> tuple[dict, dict]:
    """id -> display-name maps from drivers.csv / constructors.csv.

    Display names are a serving concern only — absence of the CSVs degrades
    names to null, never fails the app (design §6)."""
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

        # Phase 3 Item 2 (vs-baseline): MODEL_ZOO["pole_baseline"] fit once at
        # startup against the full frozen feature snapshot. "Fit" is a no-op
        # here — PoleSitterBaseline.fit() only records classes_=[0,1] and
        # never looks at X or y (the rule is the fixed "grid_adjusted==1"
        # heuristic) — so this isn't training on serving data, just wiring
        # the ColumnGuard so predict_race() can validate/score it exactly
        # like the real model. Wrapped in its own try/except so a failure
        # here degrades only /vs-baseline, not the whole API.
        try:
            baseline = get_model("pole_baseline", features[TARGET_COLUMN])
            baseline.fit(features.loc[:, list(FEATURE_COLUMNS)], features[TARGET_COLUMN])
            app.state.baseline_model = baseline
        except Exception as exc:                                # noqa: BLE001
            app.state.baseline_load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("pole-baseline model unavailable — /vs-baseline "
                           "will 503: %s", app.state.baseline_load_error)
    except Exception as exc:                                    # noqa: BLE001
        # Degraded start (design §7): report via /health + 503s.
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
                       f"(> {settings.serve_max_year}) — reserved for Phase 8 "
                       "evaluation.",
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

    @app.get("/health", response_model=HealthResponse)
    def health():
        if app.state.model is None:
            return HealthResponse(
                status="degraded", api_version=API_VERSION,
                detail=app.state.load_error or "model not loaded",
            )
        return HealthResponse(
            status="ok", api_version=API_VERSION, model=_model_schema(),
        )

    @app.get("/model", response_model=ModelInfoSchema)
    def model_info():
        _require_ready()
        return _model_schema()

    @app.get("/races", response_model=RaceListResponse)
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

    @app.get("/predictions/{race_id}", response_model=PredictionResponse)
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

        # Structured prediction log (design §10 amendment). A cache hit
        # reuses the cached body but gets its own prediction_id in the log.
        latency_ms = (time.perf_counter() - started) * 1e3
        logger.info(
            "prediction prediction_id=%s race_id=%d model_version=%s "
            "model_alias=%s n_drivers=%d cache_hit=%s latency_ms=%.2f status_code=200",
            prediction_id, race_id, info.version, info.alias,
            len(response.predictions), cache_hit, latency_ms,
        )
        return response

    @app.get("/predictions/{race_id}/simulate/{driver_id}",
             response_model=SimulateGridResponse)
    def simulate_grid(race_id: int, driver_id: int,
                      grid_position: int | None = None, pit_lane: bool = False):
        """Phase 3 Item 1 — Prediction Simulator (grid/qualifying group only).

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

    @app.get("/predictions/{race_id}/vs-baseline",
             response_model=BaselineComparisonResponse)
    def predictions_vs_baseline(race_id: int):
        """Phase 3 Item 2 — Qualifying Impact: the calibrated model next to
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

    @app.get("/debug/features/{race_id}", response_model=FeatureDebugResponse)
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

    @app.post("/predict", status_code=501)
    def predict_reserved():
        """RESERVED (design §5 amendment): Phase 8 upcoming-race predictions
        will accept a future race_id or explicit feature rows here. Routing
        it now means Phase 8 lands without an API redesign."""
        raise HTTPException(
            501,
            detail="Reserved for Phase 8 upcoming-race predictions — see "
                   "reports/application_design.md §5/§12. Use "
                   "GET /predictions/{race_id} for historical races.",
        )

    return app


app = create_app()
