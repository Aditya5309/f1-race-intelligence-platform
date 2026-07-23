"""
Tests for app/upcoming_prediction_service.py — the orchestration layer
behind POST /predict (see docs/pre_race_materialization.md).

Coverage: lazy initialization of POST /predict's training-side data (loads
on first call, not before), successful cache reuse (a second call reuses
the already-loaded tables without touching disk again), a cached
initialization FAILURE (a second call after a failed first one raises the
same cached error without retrying disk I/O), the resolve_upcoming_
prediction() pre-race cache (a second identical request is a hit and never
re-invokes materialize_and_score()), one test per named cache-invalidation
trigger (see "Cache invalidation" in docs/pre_race_materialization.md), and the plain-Python
exception types this module raises for app/api.py to map to HTTP status
codes (RaceAlreadyHasResult, ValueError for a bad/out-of-horizon/naive-
as_of request, RuntimeError for unavailable materialization data) — the
actual HTTP status codes those map to are covered end-to-end in
tests/test_api.py, which also has the provenance round-trip test.

Isolated, synthetic tmp-path data throughout — never the real project's
data/ tree (test isolation; also proves this module works from any
correctly-shaped Settings, not just this repo's own checkout).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app.upcoming_prediction_service as ups
from app.config import Settings
from app.upcoming_prediction_service import (
    RaceAlreadyHasResult,
    UpcomingPredictionResult,
    ensure_materialization_data,
    resolve_upcoming_prediction,
    resolve_upcoming_race,
)
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.features.upcoming import EntryListEntry
from src.models.registry import get_model
from src.models.splits import to_xy


def _new_state():
    return SimpleNamespace(
        materialization_data=None,
        materialization_load_attempted=False,
        materialization_load_error=None,
        pre_race_cache=OrderedDict(),
    )


def _races_frame():
    return pd.DataFrame([
        {"raceId": 1, "year": 2026, "round": 1, "circuitId": 1, "name": "R1", "date": "2026-01-01"},
        {"raceId": 2, "year": 2026, "round": 2, "circuitId": 1, "name": "R2", "date": "2026-01-08"},
        {"raceId": 3, "year": 2026, "round": 3, "circuitId": 1, "name": "R3", "date": "2026-01-15"},
    ])


def _master_frame(race_ids):
    return pd.DataFrame({"raceId": list(race_ids)})


def _write_raw_files(tmp_path, races_df, master_df) -> Settings:
    """Minimal, valid files for every path ensure_materialization_data()
    reads. Only races/master content matters to any given test; the rest
    are trivial stubs just so the load itself succeeds."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    races_df.to_csv(raw_dir / "races.csv", index=False)
    pd.DataFrame([{"driverId": 1, "driverRef": "d1", "code": "D1", "forename": "F",
                   "surname": "L", "dob": "1990-01-01", "nationality": "N"}]).to_csv(
        raw_dir / "drivers.csv", index=False)
    pd.DataFrame([{"constructorId": 1, "constructorRef": "c1", "name": "C1",
                   "nationality": "N"}]).to_csv(raw_dir / "constructors.csv", index=False)
    pd.DataFrame([{"circuitId": 1, "circuitRef": "circ1", "name": "Circuit",
                   "location": "L", "country": "C", "lat": 0.0, "lng": 0.0, "alt": 0}]).to_csv(
        raw_dir / "circuits.csv", index=False)
    pd.DataFrame([{"raceId": 1, "driverId": 1, "constructorId": 1, "points": 0,
                   "position": 1, "positionText": "1", "wins": 0}]).to_csv(
        raw_dir / "driver_standings.csv", index=False)
    pd.DataFrame([{"raceId": 1, "constructorId": 1, "points": 0, "position": 1,
                   "positionText": "1", "wins": 0}]).to_csv(
        raw_dir / "constructor_standings.csv", index=False)
    pd.DataFrame(columns=["raceId", "race_precip_mm", "race_temp_c",
                          "quali_precip_mm", "conditions_changed"]).to_csv(
        raw_dir / "race_weather.csv", index=False)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    master_path = processed_dir / "master_dataset.parquet"
    master_df.to_parquet(master_path)

    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    qualifying_path = interim_dir / "qualifying.parquet"
    pd.DataFrame(columns=["raceId", "driverId", "constructorId", "number",
                          "position", "q1", "q2", "q3"]).to_parquet(qualifying_path)

    return Settings(
        raw_data_dir=raw_dir, master_dataset_path=master_path,
        qualifying_interim_path=qualifying_path,
        weather_csv_path=raw_dir / "race_weather.csv",
    )


def _synthetic_training_frame():
    rng = np.random.default_rng(0)
    rows = []
    for race_id in range(1, 11):
        grid = rng.permutation(3) + 1
        for driver in range(3):
            row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            row.update({
                "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                "circuitId": 1, "year": 2020 + race_id, "round": 1,
                TARGET_COLUMN: int(grid[driver] == 1),
            })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def fitted_model():
    """A real, minimal fitted zoo pipeline. training_schema() (used by the
    service's cache-key computation) requires a real 'guard' step — a
    stand-in object won't do. Fit on the default (curated) feature set."""
    frame = _synthetic_training_frame()
    X, y, _ = to_xy(frame)
    model = get_model("logreg", y)
    model.fit(X, y)
    return model


@pytest.fixture()
def fitted_model_full_schema():
    """Same synthetic data, but fit on the RAW full FEATURE_COLUMNS set
    (an explicit override, not the curated default) — a genuinely
    different `training_schema()` from `fitted_model`, for the
    feature-schema-change cache-invalidation trigger test."""
    frame = _synthetic_training_frame()
    X, y, _ = to_xy(frame, feature_columns=FEATURE_COLUMNS)
    model = get_model("logreg", y, feature_columns=FEATURE_COLUMNS)
    model.fit(X, y)
    return model


def _counting_materialize_and_score(call_count: dict):
    """A materialize_and_score() stand-in that just counts invocations and
    echoes the cache-key components it was given back into the result --
    real cache-key/hit-miss logic in resolve_upcoming_prediction() is what
    these tests exercise, not real feature engineering."""
    def _fake(*_args, **kwargs):
        call_count["n"] += 1
        return UpcomingPredictionResult(
            predictions=pd.DataFrame({"driverId": [1]}),
            materialization_status="post_qualifying",
            feature_schema_version=kwargs["feature_schema_version"],
            etl_snapshot_version=kwargs["etl_snapshot_version"],
        )
    return _fake


def _bump_mtime(path) -> None:
    """Force a file's mtime strictly forward, regardless of filesystem
    timestamp resolution (some report only whole seconds)."""
    future = time.time() + 10
    os.utime(path, (future, future))


# ---------------------------------------------------------------------------
# ensure_materialization_data — lazy load, cache reuse, cached failure
# ---------------------------------------------------------------------------

def test_not_attempted_before_first_call(tmp_path):
    _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    assert state.materialization_data is None
    assert state.materialization_load_attempted is False


def test_loads_on_first_call(tmp_path):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()

    data = ensure_materialization_data(state, settings)

    assert state.materialization_load_attempted is True
    assert state.materialization_data is data
    assert list(data["races"]["raceId"]) == [1, 2, 3]
    assert list(data["master"]["raceId"]) == [1]


def test_second_call_reuses_cache_without_reloading(tmp_path, monkeypatch):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    first = ensure_materialization_data(state, settings)

    def _boom(*_a, **_k):
        raise AssertionError("read_parquet should not be called again on a cache hit")
    monkeypatch.setattr(ups.pd, "read_parquet", _boom)

    second = ensure_materialization_data(state, settings)
    assert second is first


def test_failure_is_cached_and_not_retried(tmp_path, monkeypatch):
    missing_dir = tmp_path / "no-such-dir"
    settings = Settings(
        raw_data_dir=missing_dir,
        master_dataset_path=missing_dir / "master_dataset.parquet",
        qualifying_interim_path=missing_dir / "qualifying.parquet",
        weather_csv_path=missing_dir / "race_weather.csv",
    )
    state = _new_state()

    with pytest.raises(RuntimeError):
        ensure_materialization_data(state, settings)
    assert state.materialization_load_attempted is True
    first_error = state.materialization_load_error
    assert first_error

    def _boom(*_a, **_k):
        raise AssertionError("read_parquet should not be retried after a cached failure")
    monkeypatch.setattr(ups.pd, "read_parquet", _boom)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_materialization_data(state, settings)
    assert str(exc_info.value) == first_error


# ---------------------------------------------------------------------------
# resolve_upcoming_race — identity-only lookup (GET /races/upcoming)
# ---------------------------------------------------------------------------

def test_resolve_upcoming_race_returns_identity(tmp_path):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()

    race = resolve_upcoming_race(state, settings)

    assert race is not None
    assert race.race_id == 2
    assert race.year == 2026
    assert race.round == 2


def test_resolve_upcoming_race_none_when_every_race_has_a_result(tmp_path):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1, 2, 3]))
    state = _new_state()

    assert resolve_upcoming_race(state, settings) is None


def test_resolve_upcoming_race_raises_runtime_error_when_data_unavailable(tmp_path):
    missing_dir = tmp_path / "no-such-dir"
    settings = Settings(
        raw_data_dir=missing_dir,
        master_dataset_path=missing_dir / "master_dataset.parquet",
        qualifying_interim_path=missing_dir / "qualifying.parquet",
        weather_csv_path=missing_dir / "race_weather.csv",
    )
    state = _new_state()

    with pytest.raises(RuntimeError):
        resolve_upcoming_race(state, settings)


def test_resolve_upcoming_race_reuses_already_loaded_data(tmp_path, monkeypatch):
    """Confirms resolve_upcoming_race() shares ensure_materialization_data()'s
    cache with resolve_upcoming_prediction() -- calling this identity lookup
    first must not force a second, independent load."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()

    resolve_upcoming_race(state, settings)

    def _boom(*_a, **_k):
        raise AssertionError("read_parquet should not be called again on a cache hit")
    monkeypatch.setattr(ups.pd, "read_parquet", _boom)

    race = resolve_upcoming_race(state, settings)
    assert race is not None


# ---------------------------------------------------------------------------
# resolve_upcoming_prediction — plain-Python exception types
# (RaceAlreadyHasResult / ValueError; app/api.py maps these to HTTP status
# codes, verified end-to-end in tests/test_api.py)
# ---------------------------------------------------------------------------

def test_race_not_on_calendar_raises_value_error(tmp_path):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    with pytest.raises(ValueError, match="not on the races calendar"):
        resolve_upcoming_prediction(
            state, settings, model=object(), model_info=SimpleNamespace(version="1"),
            year=2099, round_=1, entry_list_override=None, as_of=None,
        )


def test_race_already_has_result_raises_race_already_has_result(tmp_path):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    with pytest.raises(RaceAlreadyHasResult, match="already has a result"):
        resolve_upcoming_prediction(
            state, settings, model=object(), model_info=SimpleNamespace(version="1"),
            year=2026, round_=1, entry_list_override=None, as_of=None,
        )


def test_beyond_horizon_raises_value_error(tmp_path):
    """3 races on the calendar; only round 2 has no result. Requesting
    round 3 (further out, also result-less) must still be rejected --
    the materialization horizon is 1 (only the single next race), not
    "any not-yet-run race"."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    with pytest.raises(ValueError, match="materialization horizon"):
        resolve_upcoming_prediction(
            state, settings, model=object(), model_info=SimpleNamespace(version="1"),
            year=2026, round_=3, entry_list_override=None, as_of=None,
        )


def test_naive_as_of_raises_value_error(tmp_path):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    with pytest.raises(ValueError, match="no UTC offset"):
        resolve_upcoming_prediction(
            state, settings, model=object(), model_info=SimpleNamespace(version="1"),
            year=2026, round_=2,
            entry_list_override=[EntryListEntry(driver_id=1, constructor_id=1)],
            as_of="2026-01-01T00:00:00",
        )


def test_timezone_aware_as_of_is_accepted(tmp_path, monkeypatch, fitted_model):
    """A valid, timezone-aware, past as_of must NOT be rejected -- reaches
    the (mocked) materialization step successfully."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    fake_result = UpcomingPredictionResult(
        predictions=pd.DataFrame({"driverId": [1]}),
        materialization_status="post_qualifying",
    )
    monkeypatch.setattr(ups, "materialize_and_score", lambda *a, **k: fake_result)

    result, cache_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, SimpleNamespace(version="v1"),
        year=2026, round_=2,
        entry_list_override=[EntryListEntry(driver_id=1, constructor_id=1)],
        as_of="2020-01-01T00:00:00+00:00",
    )
    assert result is fake_result
    assert cache_hit is False


def test_materialization_data_unavailable_raises_runtime_error(tmp_path):
    missing_dir = tmp_path / "no-such-dir"
    settings = Settings(
        raw_data_dir=missing_dir,
        master_dataset_path=missing_dir / "master_dataset.parquet",
        qualifying_interim_path=missing_dir / "qualifying.parquet",
        weather_csv_path=missing_dir / "race_weather.csv",
    )
    state = _new_state()
    with pytest.raises(RuntimeError):
        resolve_upcoming_prediction(
            state, settings, model=object(), model_info=SimpleNamespace(version="1"),
            year=2026, round_=2, entry_list_override=None, as_of=None,
        )


# ---------------------------------------------------------------------------
# resolve_upcoming_prediction — the pre-race cache itself
# ---------------------------------------------------------------------------

def test_identical_request_is_a_cache_hit_and_skips_materialization(tmp_path, monkeypatch, fitted_model):
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    model_info = SimpleNamespace(version="v1")
    entry_list_override = [EntryListEntry(driver_id=1, constructor_id=1)]

    call_count = {"n": 0}

    def _fake_materialize_and_score(*_args, **kwargs):
        call_count["n"] += 1
        return UpcomingPredictionResult(
            predictions=pd.DataFrame({"driverId": [1]}),
            materialization_status="post_qualifying",
            feature_schema_version=kwargs["feature_schema_version"],
            etl_snapshot_version=kwargs["etl_snapshot_version"],
        )

    monkeypatch.setattr(ups, "materialize_and_score", _fake_materialize_and_score)

    first_result, first_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert first_hit is False
    assert call_count["n"] == 1

    second_result, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert second_hit is True
    assert call_count["n"] == 1                 # not called again
    assert second_result is first_result


# ---------------------------------------------------------------------------
# Cache-invalidation triggers (see "Cache invalidation" in
# docs/pre_race_materialization.md) — one test per named row. "Qualifying
# completed", "grid penalties
# adjudicated", and "ETL refresh (any file)" all share the SAME mechanism
# (etl_snapshot_version = the max mtime across three watched files) — each
# gets its own test against a DIFFERENT one of those files, covering the
# "any of these" generality honestly rather than testing one file thrice.
# ---------------------------------------------------------------------------

def test_trigger_qualifying_completed_busts_cache(tmp_path, monkeypatch, fitted_model):
    """'Qualifying completed' lands as a qualifying.parquet write."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    model_info = SimpleNamespace(version="v1")
    entry_list_override = [EntryListEntry(driver_id=1, constructor_id=1)]
    call_count = {"n": 0}
    monkeypatch.setattr(ups, "materialize_and_score", _counting_materialize_and_score(call_count))

    resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert call_count["n"] == 1

    _bump_mtime(settings.qualifying_interim_path)

    _, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert second_hit is False
    assert call_count["n"] == 2


def test_trigger_grid_penalties_adjudicated_busts_cache(tmp_path, monkeypatch, fitted_model):
    """'Grid penalties adjudicated' lands via the same ETL scope, as a
    master_dataset.parquet write -- a different watched file from the
    qualifying trigger above."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    model_info = SimpleNamespace(version="v1")
    entry_list_override = [EntryListEntry(driver_id=1, constructor_id=1)]
    call_count = {"n": 0}
    monkeypatch.setattr(ups, "materialize_and_score", _counting_materialize_and_score(call_count))

    resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert call_count["n"] == 1

    _bump_mtime(settings.master_dataset_path)

    _, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert second_hit is False
    assert call_count["n"] == 2


def test_trigger_etl_refresh_any_file_busts_cache(tmp_path, monkeypatch, fitted_model):
    """'ETL refresh (any run touching this race's inputs)' -- covered here
    via the third watched file (weather), completing the "any of the
    three" generality this cache-invalidation contract claims."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    model_info = SimpleNamespace(version="v1")
    entry_list_override = [EntryListEntry(driver_id=1, constructor_id=1)]
    call_count = {"n": 0}
    monkeypatch.setattr(ups, "materialize_and_score", _counting_materialize_and_score(call_count))

    resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert call_count["n"] == 1

    _bump_mtime(settings.weather_csv_path)

    _, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert second_hit is False
    assert call_count["n"] == 2


def test_trigger_updated_entry_list_busts_cache(tmp_path, monkeypatch, fitted_model):
    """'Updated entry list' -- an explicit override that differs from the
    previous request's resolved entry list."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    model_info = SimpleNamespace(version="v1")
    call_count = {"n": 0}
    monkeypatch.setattr(ups, "materialize_and_score", _counting_materialize_and_score(call_count))

    resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2,
        entry_list_override=[EntryListEntry(driver_id=1, constructor_id=1)],
        as_of=None,
    )
    assert call_count["n"] == 1

    _, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2,
        entry_list_override=[EntryListEntry(driver_id=2, constructor_id=1)],
        as_of=None,
    )
    assert second_hit is False
    assert call_count["n"] == 2


def test_trigger_model_promotion_busts_cache(tmp_path, monkeypatch, fitted_model):
    """'Model promotion' -- already covered by model_version, the existing
    mechanism, unchanged in the new pre-race cache namespace."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    entry_list_override = [EntryListEntry(driver_id=1, constructor_id=1)]
    call_count = {"n": 0}
    monkeypatch.setattr(ups, "materialize_and_score", _counting_materialize_and_score(call_count))

    resolve_upcoming_prediction(
        state, settings, fitted_model, SimpleNamespace(version="v1"),
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert call_count["n"] == 1

    _, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model, SimpleNamespace(version="v2"),
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert second_hit is False
    assert call_count["n"] == 2


def test_trigger_feature_schema_change_busts_cache(
    tmp_path, monkeypatch, fitted_model, fitted_model_full_schema,
):
    """'Feature schema change' -- guards against a Materializer built
    against an old FEATURE_COLUMNS shape silently reusing a cached row
    under a new one. Same model_version deliberately, to isolate this
    trigger from the model-promotion one above."""
    settings = _write_raw_files(tmp_path, _races_frame(), _master_frame([1]))
    state = _new_state()
    model_info = SimpleNamespace(version="v1")
    entry_list_override = [EntryListEntry(driver_id=1, constructor_id=1)]
    call_count = {"n": 0}
    monkeypatch.setattr(ups, "materialize_and_score", _counting_materialize_and_score(call_count))

    resolve_upcoming_prediction(
        state, settings, fitted_model, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert call_count["n"] == 1

    _, second_hit = resolve_upcoming_prediction(
        state, settings, fitted_model_full_schema, model_info,
        year=2026, round_=2, entry_list_override=entry_list_override, as_of=None,
    )
    assert second_hit is False
    assert call_count["n"] == 2
