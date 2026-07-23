"""
app/upcoming_prediction_service.py

Orchestration layer for `POST /predict` (Phase 7, Decisions 049/050/052).

`app/api.py` is a thin HTTP transport layer (request parsing, status-code
mapping, response assembly) and `src.models.predict_upcoming.
predict_upcoming_race()` is a narrow, HTTP-agnostic composition
(materialize_features() -> predict_race(), nothing else) reused elsewhere
too — widening its return contract just to carry HTTP-response metadata
(materialization_status/missing_inputs/qualifying_status) would leak an
API concern into the prediction pipeline. This module is the dedicated
place that gap is closed instead: it composes materialize_features() and
predict_race() a SECOND time (the same two-line sequence
predict_upcoming_race() already does — neither function's own logic is
duplicated, only that two-call sequence is) specifically because it also
needs the INTERMEDIATE materialized frame to derive those fields by
inspecting already-computed null-ness.

Also owns `POST /predict`'s training-side data loading (settings.
master_dataset_path, raw_data_dir, etc. — see app/config.py's own comment
+ Decision 052) — LAZILY, on first request rather than at app startup.
Stability-review finding: this data (a full master_dataset.parquet + 6 raw
CSVs + a weather CSV) is read by no other route; loading it unconditionally
at startup would pay that full read/memory cost even for a process that
never receives a single POST /predict request. `ensure_materialization_data()`
loads and caches it exactly once per process — including caching a FAILURE,
since a missing/misconfigured data/ tree does not self-heal between
requests, and retrying would just repeat the same expensive disk I/O on
every subsequent call.

The pre-race prediction cache (keyed on model version, race identity, the
served model's own feature-schema hash, the local ETL snapshot's mtime,
and the resolved entry list) also lives here — it exists to avoid
re-running materialize_and_score() (and therefore re-reading nothing new,
but re-doing real feature-engineering work) for a request that already
answered the exact same question under the exact same data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from app.config import Settings
from src.data.loader import load_csv
from src.features.upcoming import (
    EntryListEntry,
    UpcomingRace,
    next_race,
    resolve_entry_list,
)
from src.features.weather import load_race_weather
from src.models.materialize import materialize_features
from src.models.predict import predict_race
from src.models.registry import training_schema

logger = logging.getLogger("f1.api")

# Feature columns that read null when qualifying hasn't happened (or hasn't
# landed locally) yet for a materialized row — checked for nullness only,
# never recomputed, to report materialization_status/missing_inputs.
# Excludes reached_q2/reached_q3/pit_lane_start/grid_penalty_applied: those
# are booleans that correctly compute to False on missing data
# (src/features/qualifying.py), never null themselves, so they're not
# "missing inputs" in this sense.
QUALIFYING_DEPENDENT_COLUMNS: tuple[str, ...] = (
    "qualifying_position", "q1_sec", "q2_sec", "q3_sec",
    "qualifying_gap_to_pole_pct", "grid_adjusted", "grid_position_norm",
)


class RaceAlreadyHasResult(ValueError):
    """The requested (year, round) already has a real result on file.

    A ValueError subclass (not a bare RuntimeError/new base) so a caller
    that only catches ValueError still nets it; app/api.py catches this
    specifically first to map it to 409 instead of the generic 422 every
    other rejection here maps to."""


@dataclass
class UpcomingPredictionResult:
    """Everything app/api.py needs to build UpcomingPredictionResponse,
    computed here so the route itself stays pure HTTP transport."""

    predictions: pd.DataFrame
    materialization_status: str      # "post_qualifying" | "pre_qualifying"
    missing_inputs: list[str] = field(default_factory=list)
    qualifying_status: str = "not_started"   # "not_started"|"in_progress"|"complete"
    materialized_at: datetime | None = None
    predicted_at: datetime | None = None
    feature_schema_version: str = ""
    etl_snapshot_version: datetime | None = None


# ---------------------------------------------------------------------------
# Lazy, cached training-side data load
# ---------------------------------------------------------------------------

def ensure_materialization_data(state, settings: Settings) -> dict:
    """Lazily load POST /predict's training-side data exactly once per
    process, caching either the loaded tables or the failure on `state`
    (FastAPI's `app.state`).

    Raises RuntimeError (with the original cause chained) on failure — the
    caller (app/api.py) is expected to turn that into a 503. A failure is
    cached, not retried on subsequent calls (see module docstring).
    """
    if state.materialization_data is not None:
        return state.materialization_data
    if state.materialization_load_attempted:
        raise RuntimeError(state.materialization_load_error)

    state.materialization_load_attempted = True
    try:
        master = pd.read_parquet(settings.master_dataset_path)
        races_raw = load_csv("races.csv", settings.raw_data_dir)
        drivers_raw = load_csv("drivers.csv", settings.raw_data_dir)
        constructors_raw = load_csv("constructors.csv", settings.raw_data_dir)
        circuits_raw = load_csv("circuits.csv", settings.raw_data_dir)
        qualifying_raw = pd.read_parquet(settings.qualifying_interim_path)
        # load_csv, not load_standings() — the latter hardcodes its own
        # default data_dir and doesn't accept an override, so it wouldn't
        # respect settings.raw_data_dir if ever configured away from the
        # default. Same underlying primitive either way, just called
        # directly here for correct path configurability.
        driver_standings = load_csv("driver_standings.csv", settings.raw_data_dir)
        constructor_standings = load_csv("constructor_standings.csv", settings.raw_data_dir)
        weather = load_race_weather(settings.weather_csv_path)
    except Exception as exc:                                    # noqa: BLE001
        state.materialization_load_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "materialization data unavailable — POST /predict will 503: %s",
            state.materialization_load_error,
        )
        raise RuntimeError(state.materialization_load_error) from exc

    data = {
        "master": master, "races": races_raw, "drivers": drivers_raw,
        "constructors": constructors_raw, "circuits": circuits_raw,
        "qualifying": qualifying_raw, "driver_standings": driver_standings,
        "constructor_standings": constructor_standings, "weather": weather,
    }
    state.materialization_data = data
    logger.info(
        "materialization data loaded (lazy, first request): %d historical rows, "
        "%d races on calendar", len(master), len(races_raw),
    )
    return data


# ---------------------------------------------------------------------------
# Cache-key components
# ---------------------------------------------------------------------------

def _feature_schema_version(model) -> str:
    """Short, stable hash of the served model's OWN recorded training
    schema (`training_schema()`, reused unmodified) — changes iff the
    model's feature contract does."""
    payload = json.dumps(training_schema(model), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _etl_snapshot_version(settings: Settings) -> datetime:
    """Proxy for "when did the ETL last land data this route reads." No
    per-row or per-run ingestion timestamp is persisted anywhere in this
    project today — the latest mtime among the files actually read is the
    most honest available signal, not a fabricated precision."""
    paths = (
        settings.master_dataset_path,
        settings.qualifying_interim_path,
        settings.weather_csv_path,
    )
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    if not mtimes:
        return datetime.now(UTC)
    return datetime.fromtimestamp(max(mtimes), tz=UTC)


def _entry_list_hash(entry_list: list[EntryListEntry]) -> str:
    payload = json.dumps(
        sorted((e.driver_id, e.constructor_id) for e in entry_list)
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _pre_race_cache_get(
    state, settings: Settings, key: tuple
) -> UpcomingPredictionResult | None:
    cache: OrderedDict = state.pre_race_cache
    entry = cache.get(key)
    if entry is None:
        return None
    result, cached_at = entry
    age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
    if age_seconds > settings.pre_race_cache_ttl_seconds:
        del cache[key]      # expired -- treat exactly like a miss
        return None
    return result


def _pre_race_cache_set(
    state, settings: Settings, key: tuple, result: UpcomingPredictionResult
) -> None:
    cache: OrderedDict = state.pre_race_cache
    cache[key] = (result, datetime.now(UTC))
    while len(cache) > settings.prediction_cache_size:
        cache.popitem(last=False)                              # FIFO eviction


# ---------------------------------------------------------------------------
# Materialize + score (the one deliberate, disclosed duplication — see
# module docstring)
# ---------------------------------------------------------------------------

def materialize_and_score(
    model,
    race: UpcomingRace,
    entry_list: list[EntryListEntry],
    dimension_inputs: dict[str, pd.DataFrame],
    historical_master: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    feature_schema_version: str,
    etl_snapshot_version: datetime,
) -> UpcomingPredictionResult:
    """materialize_features() -> predict_race(), reused verbatim — the
    same two-call sequence `predict_upcoming_race()` already composes,
    duplicated here (not imported from there) only because this caller
    also needs the intermediate materialized frame. Neither
    materialize_features() nor predict_race() gains a line for this;
    predict_upcoming_race() isn't touched or widened."""
    materialized_at = datetime.now(UTC)
    materialized = materialize_features(
        race, entry_list, dimension_inputs, historical_master,
        driver_standings, constructor_standings, weather,
    )
    scored = predict_race(model, materialized)
    predicted_at = datetime.now(UTC)

    missing_inputs = [
        col for col in QUALIFYING_DEPENDENT_COLUMNS
        if col in materialized.columns and materialized[col].isna().any()
    ]
    materialization_status = "pre_qualifying" if missing_inputs else "post_qualifying"
    has_any_quali = materialized["qualifying_position"].notna().any()
    has_all_quali = materialized["qualifying_position"].notna().all()
    qualifying_status = (
        "complete" if has_all_quali else "in_progress" if has_any_quali else "not_started"
    )

    return UpcomingPredictionResult(
        predictions=scored,
        materialization_status=materialization_status,
        missing_inputs=missing_inputs,
        qualifying_status=qualifying_status,
        materialized_at=materialized_at,
        predicted_at=predicted_at,
        feature_schema_version=feature_schema_version,
        etl_snapshot_version=etl_snapshot_version,
    )


# ---------------------------------------------------------------------------
# Upcoming-race IDENTITY lookup (Phase 8 — GET /races/upcoming)
# ---------------------------------------------------------------------------

def resolve_upcoming_race(state, settings: Settings) -> UpcomingRace | None:
    """Identity of the single next race with no result yet (Decision 050
    horizon=1), or None if every calendar race already has a result.

    Deliberately identity-only — no materialization_status/caveats/
    provenance here, and no call to materialize_and_score(). Those stay
    owned exclusively by POST /predict (the same "single source of truth,
    no duplicated status logic" principle Decision 052 established for
    materialize_and_score() vs. predict_upcoming_race()). This function
    exists so a client (the dashboard) can cheaply discover WHICH race to
    ask POST /predict about, without that being a full, cache-populating
    materialize+score call just to populate a picker.

    Reuses ensure_materialization_data() and next_race() verbatim — no new
    data-loading or calendar-resolution logic. Raises RuntimeError if
    materialization data isn't available (app/api.py maps this to 503).
    """
    data = ensure_materialization_data(state, settings)
    races_df: pd.DataFrame = data["races"]
    master: pd.DataFrame = data["master"]
    results_for_horizon = master[["raceId"]].drop_duplicates()
    return next_race(races_df, results_for_horizon)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_upcoming_prediction(
    state,
    settings: Settings,
    model,
    model_info,
    *,
    year: int,
    round_: int,
    entry_list_override: list[EntryListEntry] | None,
    as_of: str | None,
) -> tuple[UpcomingPredictionResult, bool]:
    """Full Phase 7 orchestration for one POST /predict request: resolve
    the target race, validate it against Decision 050's horizon=1 policy,
    resolve its entry list, and materialize+score it (cached).

    Returns (result, cache_hit). Raises RuntimeError if materialization
    data isn't available (app/api.py maps this to 503),
    RaceAlreadyHasResult if the race already ran (-> 409), or ValueError
    for every other rejection (-> 422) — this function owns no HTTP
    concept itself, only these plain-Python exceptions.
    """
    data = ensure_materialization_data(state, settings)
    races_df: pd.DataFrame = data["races"]
    master: pd.DataFrame = data["master"]

    target = races_df.loc[(races_df["year"] == year) & (races_df["round"] == round_)]
    if target.empty:
        raise ValueError(f"{year} round {round_} is not on the races calendar.")
    target_row = target.iloc[0]
    target_race_id = int(target_row["raceId"])

    if target_race_id in set(master["raceId"].unique()):
        raise RaceAlreadyHasResult(
            f"{year} round {round_} already has a result — "
            "use GET /predictions/{race_id} for historical races."
        )

    results_for_horizon = master[["raceId"]].drop_duplicates()
    resolved_next = next_race(races_df, results_for_horizon)
    if resolved_next is None:
        raise ValueError(
            "No upcoming race is scheduled — every calendar race already has a result.")
    if target_race_id != resolved_next.race_id:
        raise ValueError(
            f"{year} round {round_} is beyond the current materialization "
            f"horizon (={settings.materialization_horizon}) — only "
            f"{resolved_next.year} round {resolved_next.round} can be materialized "
            "right now (Decision 050)."
        )

    etl_snapshot = _etl_snapshot_version(settings)
    if as_of is not None:
        try:
            as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"as_of '{as_of}' is not a valid ISO-8601 timestamp.") from exc
        # Contract (Decision 052): as_of must be timezone-aware. A naive
        # timestamp's UTC offset is genuinely unknown to this API — this
        # field exists to pin a precise provenance cutoff, so silently
        # assuming UTC would be a guess dressed up as a fact. Comparing a
        # naive value against etl_snapshot (always UTC-aware) would also
        # raise an unhandled TypeError if not rejected explicitly here.
        if as_of_dt.tzinfo is None:
            raise ValueError(
                f"as_of '{as_of}' has no UTC offset — pass a timezone-aware "
                "ISO-8601 timestamp (e.g. '2026-07-23T10:00:00Z' or "
                "'2026-07-23T10:00:00+00:00')."
            )
        if as_of_dt > etl_snapshot:
            raise ValueError(
                f"as_of ({as_of}) is later than the latest local ETL snapshot "
                f"({etl_snapshot.isoformat()})."
            )

    entry_list = resolve_entry_list(target_race_id, races_df, master, override=entry_list_override)

    cache_key = (
        model_info.version, year, round_,
        _feature_schema_version(model), etl_snapshot.isoformat(),
        _entry_list_hash(entry_list),
    )
    cached = _pre_race_cache_get(state, settings, cache_key)
    if cached is not None:
        return cached, True

    race = UpcomingRace(
        race_id=target_race_id, year=year, round=round_,
        circuit_id=int(target_row["circuitId"]),
        name=str(target_row["name"]), date=str(target_row["date"]),
    )
    dimension_inputs = {
        "races": races_df, "circuits": data["circuits"], "drivers": data["drivers"],
        "constructors": data["constructors"], "qualifying": data["qualifying"],
    }
    result = materialize_and_score(
        model, race, entry_list, dimension_inputs, master,
        data["driver_standings"], data["constructor_standings"], data["weather"],
        feature_schema_version=cache_key[3], etl_snapshot_version=etl_snapshot,
    )
    _pre_race_cache_set(state, settings, cache_key, result)
    return result, False
