"""
Tests for src/models/predict_upcoming.py, the composition step of the
pre-race materialization pipeline (see docs/pre_race_materialization.md).

Coverage:
  - pure composition: predict_upcoming_race() produces byte-identical
    output to calling materialize_features() then predict_race()
    manually — proving the glue adds no logic of its own
  - schema-conformance against the REAL committed served bundle
    (artifacts/serving/staging) — the actual production model, not a
    synthetic stand-in, scoring a materialized (not batch-built) row
  - NaN-handling: a driver with no qualifying data yet still produces a
    valid, finite win_probability, via the served model's OWN already-
    fitted imputer -- no new imputation logic anywhere in this path
  - backward compatibility: predict_race() itself is untouched, still
    importable and callable exactly as before

Requires real local data + the real served bundle for the schema-
conformance/NaN-handling tests; skipped (not failed) if either is
missing, matching this project's existing convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_csv
from src.features.pipeline import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    MASTER_DATASET_PATH,
    TARGET_COLUMN,
)
from src.features.standings import load_standings
from src.features.upcoming import EntryListEntry, UpcomingRace
from src.features.weather import WEATHER_CSV_PATH, load_race_weather
from src.models.materialize import materialize_features
from src.models.predict import load_model, predict_race
from src.models.predict_upcoming import predict_upcoming_race
from src.models.registry import get_model
from src.models.serving_bundle import bundle_dir_for_alias

# ---------------------------------------------------------------------------
# Pure composition — no real model or real data needed, just a fitted one
# ---------------------------------------------------------------------------

def _synthetic_training_races(n_races=20, n_drivers=4, seed=0) -> pd.DataFrame:
    """Pole-always-wins synthetic data, same pattern as
    test_season_tracking.py's fitted_bundle fixture -- enough for a LogReg
    to fit cleanly. Unrelated to the small materialize scenario below;
    predict_race() only needs the model's schema columns present, not a
    realistic distributional match."""
    rng = np.random.default_rng(seed)
    rows = []
    for race_id in range(1, n_races + 1):
        grid = rng.permutation(n_drivers) + 1
        for driver in range(n_drivers):
            row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            row["grid_adjusted"] = float(grid[driver])
            row["grid_position_norm"] = float(grid[driver]) / n_drivers
            row.update({
                "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                "circuitId": 1, "year": 2010 + race_id, "round": 1,
                TARGET_COLUMN: int(grid[driver] == 1),
            })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_model():
    from src.models.splits import to_xy
    frame = _synthetic_training_races()
    X, y, _ = to_xy(frame)
    model = get_model("logreg", y)
    model.fit(X, y)
    return model


def _historical_row(**overrides) -> dict:
    base = {
        "raceId": 1, "driverId": 1, "constructorId": 10, "circuitId": 100,
        "year": 2026, "round": 1,
        "race_name": "Race", "race_date": "2026-01-01",
        "circuit_ref": "circA", "circuit_name": "Circuit A", "circuit_location": "Loc",
        "circuit_country": "Country", "circuit_lat": 1.0, "circuit_lng": 1.0, "circuit_alt": 1,
        "driver_ref": "driver1", "driver_code": "D1", "driver_forename": "First",
        "driver_surname": "Last", "driver_dob": "1990-01-01", "driver_nationality": "Nat",
        "constructor_ref": "team10", "constructor_name": "Team 10", "constructor_nationality": "Nat",
        "grid": 1, "qualifying_position": 1, "q1": "1:20.000", "q2": "1:19.000", "q3": "1:18.000",
        "position": 1, "positionText": "1", "positionOrder": 1, "points": 25.0, "laps": 50,
        "milliseconds": 1000000, "rank": 1, "fastestLap": 30, "fastestLapTime": "1:20.000",
        "fastestLapSpeed": 200.0, "statusId": 1, "result_status": "Finished", "finished": True,
        "winner": 1,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def small_upcoming_scenario():
    historical_master = pd.DataFrame([
        _historical_row(raceId=1, driverId=1, constructorId=10, driver_ref="driver1",
                        constructor_ref="team10", positionOrder=1, winner=1),
        _historical_row(raceId=1, driverId=2, constructorId=20, driver_ref="driver2",
                        constructor_ref="team20", positionOrder=2, winner=0, grid=2,
                        qualifying_position=2),
    ])
    race = UpcomingRace(race_id=2, year=2026, round=2, circuit_id=100, name="Race 2", date="2026-01-08")
    entry_list = [EntryListEntry(driver_id=1, constructor_id=10), EntryListEntry(driver_id=2, constructor_id=20)]
    dimension_inputs = {
        "races": pd.DataFrame([{"raceId": 2, "year": 2026, "round": 2, "circuitId": 100,
                                 "name": "Race 2", "date": "2026-01-08"}]),
        "circuits": pd.DataFrame([{"circuitId": 100, "circuitRef": "circA", "name": "Circuit A",
                                    "location": "Loc", "country": "Country", "lat": 1.0, "lng": 1.0, "alt": 1}]),
        "drivers": pd.DataFrame([
            {"driverId": 1, "driverRef": "driver1", "code": "D1", "forename": "F", "surname": "L",
             "dob": "1990-01-01", "nationality": "N"},
            {"driverId": 2, "driverRef": "driver2", "code": "D2", "forename": "F", "surname": "L",
             "dob": "1990-01-01", "nationality": "N"},
        ]),
        "constructors": pd.DataFrame([
            {"constructorId": 10, "constructorRef": "team10", "name": "Team 10", "nationality": "N"},
            {"constructorId": 20, "constructorRef": "team20", "name": "Team 20", "nationality": "N"},
        ]),
        "qualifying": pd.DataFrame([
            {"raceId": 2, "driverId": 1, "constructorId": 10, "number": 1, "position": 1,
             "q1": "1:19.5", "q2": "1:18.5", "q3": "1:17.5"},
            {"raceId": 2, "driverId": 2, "constructorId": 20, "number": 2, "position": 2,
             "q1": "1:19.6", "q2": "1:18.6", "q3": "1:17.6"},
        ]),
    }
    driver_standings = pd.DataFrame([
        {"raceId": 1, "driverId": 1, "points": 25, "position": 1, "positionText": "1", "wins": 1},
        {"raceId": 1, "driverId": 2, "points": 18, "position": 2, "positionText": "2", "wins": 0},
    ])
    constructor_standings = pd.DataFrame([
        {"raceId": 1, "constructorId": 10, "points": 25, "position": 1, "positionText": "1", "wins": 1},
        {"raceId": 1, "constructorId": 20, "points": 18, "position": 2, "positionText": "2", "wins": 0},
    ])
    weather = pd.DataFrame(columns=["raceId", "race_precip_mm", "race_temp_c",
                                     "quali_precip_mm", "conditions_changed"])
    return {
        "race": race, "entry_list": entry_list, "dimension_inputs": dimension_inputs,
        "historical_master": historical_master, "driver_standings": driver_standings,
        "constructor_standings": constructor_standings, "weather": weather,
    }


def test_predict_upcoming_race_matches_manual_composition(fitted_model, small_upcoming_scenario):
    """The glue must add nothing: calling predict_upcoming_race() gives
    byte-identical output to calling materialize_features() then
    predict_race() manually, in two separate calls."""
    s = small_upcoming_scenario
    materialized = materialize_features(
        s["race"], s["entry_list"], s["dimension_inputs"], s["historical_master"],
        s["driver_standings"], s["constructor_standings"], s["weather"],
    )
    expected = predict_race(fitted_model, materialized)

    actual = predict_upcoming_race(
        fitted_model, s["race"], s["entry_list"], s["dimension_inputs"], s["historical_master"],
        s["driver_standings"], s["constructor_standings"], s["weather"],
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_predict_upcoming_race_returns_predict_race_contract(fitted_model, small_upcoming_scenario):
    s = small_upcoming_scenario
    result = predict_upcoming_race(
        fitted_model, s["race"], s["entry_list"], s["dimension_inputs"], s["historical_master"],
        s["driver_standings"], s["constructor_standings"], s["weather"],
    )
    assert list(result.columns) == [
        "raceId", "driverId", "constructorId", "year", "round",
        "win_probability_raw", "win_probability", "predicted_rank",
    ]
    assert len(result) == 2
    assert abs(result["win_probability"].sum() - 1.0) < 1e-9
    assert set(result["predicted_rank"]) == {1, 2}


def test_predict_race_itself_is_unmodified_and_still_importable():
    """Backward-compatibility sanity check: the historical prediction
    path's own entry point is untouched by this phase."""
    import inspect

    from src.models.predict import predict_race as pr
    assert "race_df" in inspect.signature(pr).parameters


# ---------------------------------------------------------------------------
# Schema-conformance + NaN-handling against the REAL committed served bundle
# ---------------------------------------------------------------------------

_STAGING_BUNDLE_DIR = bundle_dir_for_alias("Staging")


@pytest.fixture(scope="module")
def real_data():
    return {
        "master": pd.read_parquet(MASTER_DATASET_PATH),
        "races": load_csv("races.csv"),
        "drivers": load_csv("drivers.csv"),
        "constructors": load_csv("constructors.csv"),
        "circuits": load_csv("circuits.csv"),
        "qualifying": pd.read_parquet(MASTER_DATASET_PATH.parent.parent / "interim" / "qualifying.parquet"),
        "driver_standings": load_standings()[0],
        "constructor_standings": load_standings()[1],
        "weather": load_race_weather(),
    }


def _materialize_real_race(race_id: int, real_data: dict, entry_list=None) -> tuple:
    race_row = real_data["races"].loc[real_data["races"]["raceId"] == race_id].iloc[0]
    year, rnd = int(race_row["year"]), int(race_row["round"])
    historical_master = real_data["master"][
        (real_data["master"]["year"] < year)
        | ((real_data["master"]["year"] == year) & (real_data["master"]["round"] < rnd))
    ].copy()
    real_rows = real_data["master"].loc[real_data["master"]["raceId"] == race_id]
    if entry_list is None:
        entry_list = [
            EntryListEntry(driver_id=int(r.driverId), constructor_id=int(r.constructorId))
            for r in real_rows.itertuples()
        ]
    race = UpcomingRace(
        race_id=race_id, year=year, round=rnd, circuit_id=int(race_row["circuitId"]),
        name=str(race_row["name"]), date=str(race_row["date"]),
    )
    dimension_inputs = {
        "races": real_data["races"], "circuits": real_data["circuits"],
        "drivers": real_data["drivers"], "constructors": real_data["constructors"],
        "qualifying": real_data["qualifying"],
    }
    return race, entry_list, dimension_inputs, historical_master


@pytest.mark.skipif(not MASTER_DATASET_PATH.exists(), reason="master_dataset.parquet not built")
@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="features.parquet not built")
@pytest.mark.skipif(not WEATHER_CSV_PATH.exists(), reason="race_weather.csv not built")
@pytest.mark.skipif(not _STAGING_BUNDLE_DIR.exists(), reason="artifacts/serving/staging not present")
def test_schema_conformance_against_real_served_bundle(real_data):
    """Materialize a real 2023 race and score it with the ACTUAL committed
    production bundle -- not a synthetic stand-in. Proves the served
    model's own ColumnGuard accepts a materialized row exactly as it
    accepts a batch-built features.parquet row."""
    model, _info = load_model(_STAGING_BUNDLE_DIR)
    race_id = int(
        real_data["races"].loc[
            (real_data["races"].year == 2023) & (real_data["races"]["round"] == 5), "raceId"
        ].iloc[0]
    )
    race, entry_list, dimension_inputs, historical_master = _materialize_real_race(race_id, real_data)

    result = predict_upcoming_race(
        model, race, entry_list, dimension_inputs, historical_master,
        real_data["driver_standings"], real_data["constructor_standings"], real_data["weather"],
    )

    assert len(result) == len(entry_list)
    assert not result["win_probability"].isna().any()
    assert abs(result["win_probability"].sum() - 1.0) < 1e-6
    assert (result["win_probability"] >= 0).all()


@pytest.mark.skipif(not MASTER_DATASET_PATH.exists(), reason="master_dataset.parquet not built")
@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="features.parquet not built")
@pytest.mark.skipif(not WEATHER_CSV_PATH.exists(), reason="race_weather.csv not built")
@pytest.mark.skipif(not _STAGING_BUNDLE_DIR.exists(), reason="artifacts/serving/staging not present")
def test_nan_handling_partial_qualifying_produces_valid_prediction(real_data):
    """A driver with no qualifying row yet (a genuinely partial, pre-
    qualifying materialization) still produces a valid, finite
    win_probability -- via the served model's OWN already-fitted
    SimpleImputer(add_indicator=True), not any new imputation logic. If
    the model's imputer couldn't handle this, predict_race() itself would
    raise ValueError('Model produced NaN probabilities...')."""
    model, _info = load_model(_STAGING_BUNDLE_DIR)
    race_id = int(
        real_data["races"].loc[
            (real_data["races"].year == 2023) & (real_data["races"]["round"] == 5), "raceId"
        ].iloc[0]
    )
    real_rows = real_data["master"].loc[real_data["master"]["raceId"] == race_id]
    entry_list = [
        EntryListEntry(driver_id=int(r.driverId), constructor_id=int(r.constructorId))
        for r in real_rows.itertuples()
    ]
    race, _, dimension_inputs, historical_master = _materialize_real_race(race_id, real_data, entry_list)

    # Strip the first driver's qualifying row -- as if their session
    # hadn't posted yet when this race was materialized.
    missing_driver_id = entry_list[0].driver_id
    partial_qualifying = dimension_inputs["qualifying"][
        ~((dimension_inputs["qualifying"]["raceId"] == race_id)
          & (dimension_inputs["qualifying"]["driverId"] == missing_driver_id))
    ]
    dimension_inputs = {**dimension_inputs, "qualifying": partial_qualifying}

    result = predict_upcoming_race(
        model, race, entry_list, dimension_inputs, historical_master,
        real_data["driver_standings"], real_data["constructor_standings"], real_data["weather"],
    )

    assert not result["win_probability"].isna().any()
    assert np.isfinite(result["win_probability"]).all()
