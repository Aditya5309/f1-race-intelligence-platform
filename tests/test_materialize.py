"""
Tests for src/models/materialize.py (the Materializer — Phase 3 of the
pre-race materialization plan, Decisions 049/050).

Coverage, per the design doc's own input tiers (§3):
  - structural (identity/historical aggregates): a rookie with zero prior
    results gets null driver-form features, never a fabricated one; an
    ambiguous/duplicate dimension key raises via the REUSED
    `build_master_dataset._join_and_check`, not a new check
  - session-dependent (qualifying/grid): a driver with no qualifying row
    yet gets null qualifying/grid-derived features, never fabricated
  - entry-list: an explicit entry list drives which rows get built, not
    inference (Phase 1's concern, exercised here only via its output shape)
Plus: output contract (schema, no target column, sorted, one row per
entrant), the grid-proxy invariant (grid == qualifying_position, so
pit_lane_start/grid_penalty_applied always read False), reuse of
`validate_features()`/`validate_output()` (a genuinely bad frame raises
through the REUSED validator, never a new check), the qualifying join's
scoping fix (an unrelated duplicate elsewhere in qualifying history must
NOT block materializing the target race — a `/review` finding, resolved),
and the ValueError paths (empty entry list, duplicate raceId in
historical_master, an unresolved driver reference, a duplicate driverId in
the entry list, a validate_features() failure).

No network calls, no file I/O anywhere in this file — every input is an
in-memory synthetic DataFrame, by construction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.upcoming import EntryListEntry, UpcomingRace
from src.models.materialize import MATERIALIZED_COLUMNS, materialize_features

# ---------------------------------------------------------------------------
# Fixture builders — minimal synthetic tables matching the real Ergast
# schema subset the Materializer + build_master_dataset actually read.
# ---------------------------------------------------------------------------

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


def _races_dim_row(**overrides) -> dict:
    base = {"raceId": 3, "year": 2026, "round": 3, "circuitId": 100,
            "name": "Race 3", "date": "2026-01-15"}
    base.update(overrides)
    return base


def _circuits_dim_row(**overrides) -> dict:
    base = {"circuitId": 100, "circuitRef": "circA", "name": "Circuit A",
            "location": "Loc", "country": "Country", "lat": 1.0, "lng": 1.0, "alt": 1}
    base.update(overrides)
    return base


def _drivers_dim_row(**overrides) -> dict:
    base = {"driverId": 1, "driverRef": "driver1", "code": "D1", "forename": "First",
            "surname": "Last", "dob": "1990-01-01", "nationality": "Nat"}
    base.update(overrides)
    return base


def _constructors_dim_row(**overrides) -> dict:
    base = {"constructorId": 10, "constructorRef": "team10", "name": "Team 10",
            "nationality": "Nat"}
    base.update(overrides)
    return base


def _empty_weather() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["raceId", "race_precip_mm", "race_temp_c", "quali_precip_mm", "conditions_changed"]
    )


@pytest.fixture()
def two_driver_scenario():
    """Two completed historical races (raceId 1, 2), drivers 1 & 2, driver 1
    wins both. Upcoming race 3: both entered; only driver 1's qualifying has
    landed (Phase 2 already ran for driver 1, not driver 2 — the
    session-dependent-tier test case)."""
    historical_master = pd.DataFrame([
        _historical_row(raceId=1, driverId=1, constructorId=10, driver_ref="driver1",
                        constructor_ref="team10", positionOrder=1, winner=1),
        _historical_row(raceId=1, driverId=2, constructorId=20, driver_ref="driver2",
                        constructor_ref="team20", positionOrder=2, winner=0, grid=2,
                        qualifying_position=2),
        _historical_row(raceId=2, driverId=1, constructorId=10, driver_ref="driver1",
                        constructor_ref="team10", positionOrder=1, winner=1, round=2),
        _historical_row(raceId=2, driverId=2, constructorId=20, driver_ref="driver2",
                        constructor_ref="team20", positionOrder=2, winner=0, grid=2,
                        qualifying_position=2, round=2),
    ])

    race = UpcomingRace(race_id=3, year=2026, round=3, circuit_id=100, name="Race 3", date="2026-01-15")
    entry_list = [EntryListEntry(driver_id=1, constructor_id=10), EntryListEntry(driver_id=2, constructor_id=20)]

    dimension_inputs = {
        "races": pd.DataFrame([_races_dim_row()]),
        "circuits": pd.DataFrame([_circuits_dim_row()]),
        "drivers": pd.DataFrame([_drivers_dim_row(driverId=1, driverRef="driver1"),
                                 _drivers_dim_row(driverId=2, driverRef="driver2")]),
        "constructors": pd.DataFrame([_constructors_dim_row(constructorId=10, constructorRef="team10"),
                                      _constructors_dim_row(constructorId=20, constructorRef="team20")]),
        # driver 2 has no row here yet -- qualifying not ingested for them.
        "qualifying": pd.DataFrame([
            {"raceId": 3, "driverId": 1, "constructorId": 10, "number": 1, "position": 1,
             "q1": "1:19.500", "q2": "1:18.500", "q3": "1:17.500"},
        ]),
    }

    driver_standings = pd.DataFrame([
        {"raceId": 1, "driverId": 1, "points": 25, "position": 1, "positionText": "1", "wins": 1},
        {"raceId": 1, "driverId": 2, "points": 18, "position": 2, "positionText": "2", "wins": 0},
        {"raceId": 2, "driverId": 1, "points": 50, "position": 1, "positionText": "1", "wins": 2},
        {"raceId": 2, "driverId": 2, "points": 36, "position": 2, "positionText": "2", "wins": 0},
    ])
    constructor_standings = pd.DataFrame([
        {"raceId": 1, "constructorId": 10, "points": 25, "position": 1, "positionText": "1", "wins": 1},
        {"raceId": 1, "constructorId": 20, "points": 18, "position": 2, "positionText": "2", "wins": 0},
        {"raceId": 2, "constructorId": 10, "points": 50, "position": 1, "positionText": "1", "wins": 2},
        {"raceId": 2, "constructorId": 20, "points": 36, "position": 2, "positionText": "2", "wins": 0},
    ])

    return {
        "race": race, "entry_list": entry_list, "dimension_inputs": dimension_inputs,
        "historical_master": historical_master, "driver_standings": driver_standings,
        "constructor_standings": constructor_standings, "weather": _empty_weather(),
    }


def _materialize(scenario, **overrides):
    kwargs = {**scenario, **overrides}
    return materialize_features(
        kwargs["race"], kwargs["entry_list"], kwargs["dimension_inputs"],
        kwargs["historical_master"], kwargs["driver_standings"],
        kwargs["constructor_standings"], kwargs["weather"],
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_schema_matches_materialized_columns_exactly(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    assert list(result.columns) == list(MATERIALIZED_COLUMNS)


def test_output_never_contains_a_target_column(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    assert "winner" not in result.columns


def test_one_row_per_entry_list_entry_sorted_by_driver(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    assert len(result) == 2
    assert list(result["driverId"]) == [1, 2]
    assert (result["raceId"] == 3).all()


# ---------------------------------------------------------------------------
# Structural tier: historical aggregates reflect PRIOR races only
# ---------------------------------------------------------------------------

def test_prior_wins_and_experience_reflect_historical_races_only(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    driver1 = result.loc[result.driverId == 1].iloc[0]
    assert driver1["driver_experience_races"] == 2
    assert driver1["driver_wins_last_3"] == 2.0
    assert driver1["driver_circuit_wins"] == 2.0  # both prior races at this circuit


def test_rookie_with_no_history_gets_null_form_features():
    """Entry-list tier: a genuinely new driver (never raced) gets NaN
    driver-form features, not zero -- 'no information' and '0 wins' are
    different signals (context/domain_knowledge.md)."""
    historical_master = pd.DataFrame([_historical_row(raceId=1, driverId=1, constructorId=10)])
    race = UpcomingRace(race_id=2, year=2026, round=2, circuit_id=100, name="Race 2", date="2026-01-08")
    entry_list = [EntryListEntry(driver_id=99, constructor_id=10)]  # rookie, never raced

    dimension_inputs = {
        "races": pd.DataFrame([_races_dim_row(raceId=2, round=2)]),
        "circuits": pd.DataFrame([_circuits_dim_row()]),
        "drivers": pd.DataFrame([_drivers_dim_row(driverId=99, driverRef="rookie")]),
        "constructors": pd.DataFrame([_constructors_dim_row()]),
        "qualifying": pd.DataFrame([
            {"raceId": 2, "driverId": 99, "constructorId": 10, "number": 1, "position": 1,
             "q1": "1:19.000", "q2": "1:18.000", "q3": "1:17.000"},
        ]),
    }
    driver_standings = pd.DataFrame(columns=["raceId", "driverId", "points", "position", "positionText", "wins"])
    constructor_standings = pd.DataFrame(
        columns=["raceId", "constructorId", "points", "position", "positionText", "wins"]
    )

    result = materialize_features(
        race, entry_list, dimension_inputs, historical_master,
        driver_standings, constructor_standings, _empty_weather(),
    )
    assert pd.isna(result.iloc[0]["driver_wins_last_3"])
    assert result.iloc[0]["driver_experience_races"] == 0


# ---------------------------------------------------------------------------
# Session-dependent tier: qualifying/grid absence is surfaced, not imputed
# ---------------------------------------------------------------------------

def test_driver_with_no_qualifying_yet_gets_null_not_fabricated(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    driver2 = result.loc[result.driverId == 2].iloc[0]
    assert pd.isna(driver2["qualifying_position"])
    assert pd.isna(driver2["grid_adjusted"])


def test_driver_with_qualifying_gets_real_values(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    driver1 = result.loc[result.driverId == 1].iloc[0]
    assert driver1["qualifying_position"] == 1.0


def test_qualifying_join_ignores_unrelated_duplicate_elsewhere_in_history(two_driver_scenario):
    """/review finding (Important, resolved): validate="one_to_one" checks
    the right side's key uniqueness GLOBALLY, not just the matched subset.
    A duplicate (raceId, driverId) pair for a completely unrelated,
    already-completed race must NOT block materializing the target race —
    the qualifying table is pre-filtered to `race.race_id` before the
    (unchanged) `_join_and_check` call specifically to eliminate this."""
    unrelated_duplicate = pd.DataFrame([
        {"raceId": 1, "driverId": 5, "constructorId": 10, "number": 5, "position": 9,
         "q1": "9:99.999", "q2": "9:99.999", "q3": "9:99.999"},
        {"raceId": 1, "driverId": 5, "constructorId": 10, "number": 5, "position": 9,
         "q1": "9:99.999", "q2": "9:99.999", "q3": "9:99.999"},
    ])
    qualifying_with_unrelated_dupe = pd.concat([
        two_driver_scenario["dimension_inputs"]["qualifying"], unrelated_duplicate,
    ], ignore_index=True)
    dimension_inputs = {
        **two_driver_scenario["dimension_inputs"], "qualifying": qualifying_with_unrelated_dupe,
    }

    result = _materialize(two_driver_scenario, dimension_inputs=dimension_inputs)

    assert len(result) == 2
    assert result.loc[result.driverId == 1, "qualifying_position"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Grid-penalty-proxy invariant (design doc §1/§3, still unresolved)
# ---------------------------------------------------------------------------

def test_grid_equals_qualifying_position_proxy_so_penalty_never_detected(two_driver_scenario):
    result = _materialize(two_driver_scenario)
    driver1 = result.loc[result.driverId == 1].iloc[0]
    assert driver1["grid_adjusted"] == driver1["qualifying_position"]
    assert bool(driver1["pit_lane_start"]) is False
    assert bool(driver1["grid_penalty_applied"]) is False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

def test_empty_entry_list_raises(two_driver_scenario):
    with pytest.raises(ValueError, match="entry_list is empty"):
        _materialize(two_driver_scenario, entry_list=[])


def test_duplicate_race_id_in_historical_master_raises(two_driver_scenario):
    bad_master = pd.concat([
        two_driver_scenario["historical_master"],
        pd.DataFrame([_historical_row(raceId=3, driverId=1, constructorId=10)]),
    ], ignore_index=True)

    with pytest.raises(ValueError, match="already contains raceId 3"):
        _materialize(two_driver_scenario, historical_master=bad_master)


def test_duplicate_dimension_key_propagates_join_and_check_error(two_driver_scenario):
    """Reuse verification: a duplicated circuitId in the circuits dimension
    table raises via build_master_dataset's OWN reused merge/`_join_and_check`
    path, not a new check invented in materialize.py. `validate="many_to_one"`
    (also reused, unchanged) catches this even before `_join_and_check`'s own
    row-count comparison would — `pandas.errors.MergeError` is itself a
    `ValueError` subclass."""
    bad_circuits = pd.concat([
        two_driver_scenario["dimension_inputs"]["circuits"],
        two_driver_scenario["dimension_inputs"]["circuits"],
    ], ignore_index=True)
    bad_dimension_inputs = {**two_driver_scenario["dimension_inputs"], "circuits": bad_circuits}

    with pytest.raises(ValueError, match="not a many-to-one merge"):
        _materialize(two_driver_scenario, dimension_inputs=bad_dimension_inputs)


def test_unresolved_driver_reference_raises(two_driver_scenario):
    """/review finding (Important, resolved): an entry_list driverId absent
    from the drivers dimension table must hard-fail (design doc §3's
    "structural, always-available identity" rule), not silently carry a
    null driver_ref through to the feature pipeline. Caught by reusing
    build_master_dataset.validate_output()'s own referential-integrity
    check, not a new one."""
    entry_list = [EntryListEntry(driver_id=999, constructor_id=10)]  # not in "drivers" below
    dimension_inputs = {
        **two_driver_scenario["dimension_inputs"],
        "qualifying": pd.DataFrame([
            {"raceId": 3, "driverId": 999, "constructorId": 10, "number": 1, "position": 1,
             "q1": "1:19", "q2": "1:18", "q3": "1:17"},
        ]),
    }

    with pytest.raises(ValueError, match="no matching row in drivers.csv"):
        _materialize(two_driver_scenario, entry_list=entry_list, dimension_inputs=dimension_inputs)


def test_duplicate_driver_id_in_entry_list_raises(two_driver_scenario):
    """A malformed entry_list (the same driver listed twice) must hard-fail
    rather than silently produce a duplicate (raceId, driverId) pair. In
    practice this is caught even earlier than validate_output(): the
    qualifying join's reused `_join_and_check(..., validate="one_to_one")`
    requires the LEFT side's keys unique too, so a duplicate entry_list
    driverId raises there first (`pandas.errors.MergeError`, itself a
    `ValueError` subclass) — still zero new validation logic invented."""
    entry_list = [
        EntryListEntry(driver_id=1, constructor_id=10),
        EntryListEntry(driver_id=1, constructor_id=10),
    ]

    with pytest.raises(ValueError, match="not unique in left dataset"):
        _materialize(two_driver_scenario, entry_list=entry_list)


def test_validate_features_failure_raises(two_driver_scenario):
    """Reuse verification: a historical_master that already violates
    validate_features()'s own duplicate-pair check must fail loudly through
    THAT reused validator, not silently produce a materialized row."""
    bad_master = pd.concat([
        two_driver_scenario["historical_master"],
        pd.DataFrame([_historical_row(raceId=1, driverId=1, constructorId=10)]),  # duplicate (1,1)
    ], ignore_index=True)

    with pytest.raises(ValueError, match="failed validate_features"):
        _materialize(two_driver_scenario, historical_master=bad_master)
