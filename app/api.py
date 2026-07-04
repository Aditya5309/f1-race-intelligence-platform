"""
app/api.py

FastAPI serving layer (Decision 016; reports/application_design.md).

    uvicorn app.api:app --reload

The app is deliberately logic-free: HTTP concerns in, exactly two calls to
the tested prediction layer out — `predict.load_model()` once at startup,
`predict.predict_race()` per request. Feature rows are looked up SERVER-SIDE
from features.parquet by raceId (clients never send feature payloads —
features are derived artifacts of the leakage-audited pipeline; design §1).

Endpoints (design §5):
    GET  /health                    liveness + serving model metadata
    GET  /model                     full ModelInfo
    GET  /races?year=               races available to score (<= serve_max_year)
    GET  /predictions/{race_id}     per-race-normalized field predictions
    GET  /debug/features/{race_id}  dev-only (F1_DEBUG_ENDPOINTS=true) —
                                    the exact feature vectors fed to the model
    POST /predict                   RESERVED for Phase 8 upcoming-race scoring;
                                    returns 501 (design §5 amendment)

Degraded-start policy (§7): if the model or features cannot be loaded, the
app still starts and reports the failure via 503s — starting-and-reporting
beats crash-looping under a scheduler.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _package_version

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.config import Settings
from src.models.predict import load_model, predict_race
from src.models.registry import training_schema

logger = logging.getLogger("f1.api")

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

    try:
        kwargs = {"alias": settings.model_alias}
        if settings.tracking_uri:
            kwargs["tracking_uri"] = settings.tracking_uri
        model, info = load_model(**kwargs)
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

            winner_id = None
            if "winner" in rows.columns and rows["winner"].notna().all():
                winners = rows.loc[rows["winner"] == 1, "driverId"]
                winner_id = int(winners.iloc[0]) if len(winners) == 1 else None
            top_pick = int(scored.iloc[0]["driverId"])

            response = PredictionResponse(
                prediction_id=prediction_id,
                race_id=race_id,
                year=int(rows["year"].iloc[0]),
                round=int(rows["round"].iloc[0]),
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model=_model_schema(),
                predictions=[
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
                ],
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
