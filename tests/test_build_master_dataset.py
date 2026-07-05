"""
Tests for src/integration/build_master_dataset.py and src/pipelines/build_dataset.py

Covers:
  - Correct joins across all 6 source tables at the (raceId, driverId) grain
  - Row count is preserved through every join step (no fan-out)
  - Left-join semantics: a driver missing from qualifying still appears, with
    null qualifying columns, rather than being dropped
  - Target (`winner`) derivation from positionOrder
  - Column renaming avoids cross-table collisions (name, nationality, position)
  - validate_inputs(): duplicate primary keys, duplicate (raceId, driverId)
    pairs, null join keys
  - validate_output(): schema mismatch, duplicate pairs, null identifiers,
    referential integrity (orphan foreign keys), winner-count warning
  - Fan-out protection: a duplicated dimension key raises rather than silently
    multiplying rows
  - End-to-end smoke test against the real project data via the pipeline entry
    point (src/pipelines/build_dataset.py)
"""

import pandas as pd
import pytest

from src.integration.build_master_dataset import (
    MASTER_DATASET_COLUMNS,
    POST_RACE_OUTCOME_COLUMNS,
    build_master_dataset,
    validate_inputs,
    validate_output,
)

# ---------------------------------------------------------------------------
# Fixture builders — minimal synthetic tables matching the real Ergast schema
# subset this module actually reads.
# ---------------------------------------------------------------------------

def _results_row(**overrides) -> dict:
    base = {
        "raceId": 1, "driverId": 1, "constructorId": 1,
        "grid": 1, "position": 1, "positionText": "1", "positionOrder": 1,
        "points": 25.0, "laps": 58, "milliseconds": 5000000, "rank": 1,
        "fastestLap": 40, "fastestLapTime": "1:20.000", "fastestLapSpeed": 220.0,
        "statusId": 1, "result_status": "Finished", "finished": True,
    }
    base.update(overrides)
    return base


def _qualifying_row(**overrides) -> dict:
    base = {
        "raceId": 1, "driverId": 1, "constructorId": 1,
        "position": 1, "q1": "1:25.000", "q2": "1:24.000", "q3": "1:23.000",
    }
    base.update(overrides)
    return base


def _races_row(**overrides) -> dict:
    base = {"raceId": 1, "year": 2020, "round": 1, "circuitId": 1,
            "name": "Test Grand Prix", "date": "2020-03-01"}
    base.update(overrides)
    return base


def _circuits_row(**overrides) -> dict:
    base = {"circuitId": 1, "circuitRef": "test_circuit", "name": "Test Circuit",
            "location": "Testville", "country": "Testland",
            "lat": 1.0, "lng": 2.0, "alt": 3}
    base.update(overrides)
    return base


def _drivers_row(**overrides) -> dict:
    base = {"driverId": 1, "driverRef": "driver_one", "code": "DR1",
            "forename": "Driver", "surname": "One", "dob": "1990-01-01",
            "nationality": "Testland"}
    base.update(overrides)
    return base


def _constructors_row(**overrides) -> dict:
    base = {"constructorId": 1, "constructorRef": "team_one", "name": "Team One",
            "nationality": "Testland"}
    base.update(overrides)
    return base


def _basic_inputs() -> dict[str, pd.DataFrame]:
    """
    Two races, two drivers each, one constructor per driver. Driver 2 has no
    qualifying row in race 2 (tests left-join semantics). Race 1: driver 1
    wins. Race 2: driver 3 wins, driver 4 does not qualify-match.
    """
    results = pd.DataFrame([
        _results_row(raceId=1, driverId=1, constructorId=1, positionOrder=1, position=1),
        _results_row(raceId=1, driverId=2, constructorId=2, positionOrder=2, position=2,
                     result_status="Finished", finished=True),
        _results_row(raceId=2, driverId=3, constructorId=1, positionOrder=1, position=1),
        _results_row(raceId=2, driverId=4, constructorId=2, positionOrder=2, position=None,
                     positionText="R", result_status="Retired", finished=False),
    ])
    qualifying = pd.DataFrame([
        _qualifying_row(raceId=1, driverId=1, constructorId=1, position=2),
        _qualifying_row(raceId=1, driverId=2, constructorId=2, position=1),
        _qualifying_row(raceId=2, driverId=3, constructorId=1, position=1),
        # driverId=4 deliberately has no qualifying row in race 2 (DNQ scenario)
    ])
    races = pd.DataFrame([
        _races_row(raceId=1, year=2020, round=1, circuitId=1),
        _races_row(raceId=2, year=2020, round=2, circuitId=2, name="Second Grand Prix"),
    ])
    circuits = pd.DataFrame([
        _circuits_row(circuitId=1),
        _circuits_row(circuitId=2, circuitRef="second_circuit", name="Second Circuit"),
    ])
    drivers = pd.DataFrame([
        _drivers_row(driverId=1),
        _drivers_row(driverId=2, driverRef="driver_two", code="DR2", surname="Two"),
        _drivers_row(driverId=3, driverRef="driver_three", code="DR3", surname="Three"),
        _drivers_row(driverId=4, driverRef="driver_four", code="DR4", surname="Four"),
    ])
    constructors = pd.DataFrame([
        _constructors_row(constructorId=1),
        _constructors_row(constructorId=2, constructorRef="team_two", name="Team Two"),
    ])
    return {
        "results": results, "qualifying": qualifying, "races": races,
        "circuits": circuits, "drivers": drivers, "constructors": constructors,
    }


# ---------------------------------------------------------------------------
# validate_inputs()
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_inputs_pass(self):
        result = validate_inputs(_basic_inputs())
        assert result.passed
        assert result.errors == []

    def test_detects_duplicate_primary_key_in_dimension_table(self):
        inputs = _basic_inputs()
        inputs["drivers"] = pd.concat(
            [inputs["drivers"], pd.DataFrame([_drivers_row(driverId=1)])],
            ignore_index=True,
        )
        result = validate_inputs(inputs)
        assert not result.passed
        assert any("drivers" in e and "duplicate" in e for e in result.errors)

    def test_detects_duplicate_race_driver_pair_in_results(self):
        inputs = _basic_inputs()
        inputs["results"] = pd.concat(
            [inputs["results"], pd.DataFrame([_results_row(raceId=1, driverId=1)])],
            ignore_index=True,
        )
        result = validate_inputs(inputs)
        assert not result.passed
        assert any("results" in e for e in result.errors)

    def test_detects_duplicate_race_driver_pair_in_qualifying(self):
        inputs = _basic_inputs()
        inputs["qualifying"] = pd.concat(
            [inputs["qualifying"], pd.DataFrame([_qualifying_row(raceId=1, driverId=1)])],
            ignore_index=True,
        )
        result = validate_inputs(inputs)
        assert not result.passed
        assert any("qualifying" in e for e in result.errors)

    def test_detects_null_join_key(self):
        inputs = _basic_inputs()
        inputs["results"].loc[0, "constructorId"] = None
        result = validate_inputs(inputs)
        assert not result.passed
        assert any("constructorId" in e for e in result.errors)


# ---------------------------------------------------------------------------
# build_master_dataset()
# ---------------------------------------------------------------------------

class TestBuildMasterDataset:
    def test_row_count_matches_results(self):
        inputs = _basic_inputs()
        master = build_master_dataset(inputs)
        assert len(master) == len(inputs["results"])

    def test_schema_matches_expected_columns(self):
        master = build_master_dataset(_basic_inputs())
        assert list(master.columns) == list(MASTER_DATASET_COLUMNS)

    def test_left_join_preserves_driver_with_no_qualifying_row(self):
        master = build_master_dataset(_basic_inputs())
        row = master[(master.raceId == 2) & (master.driverId == 4)].iloc[0]
        assert pd.isna(row["qualifying_position"])
        assert pd.isna(row["q1"])
        # but dimension data (not qualifying-dependent) still resolved
        assert row["driver_ref"] == "driver_four"

    def test_winner_derived_from_position_order(self):
        master = build_master_dataset(_basic_inputs())
        winners = master[master["winner"] == 1][["raceId", "driverId"]]
        assert set(winners.itertuples(index=False, name=None)) == {(1, 1), (2, 3)}

    def test_exactly_one_winner_per_race_in_clean_fixture(self):
        master = build_master_dataset(_basic_inputs())
        counts = master.groupby("raceId")["winner"].sum()
        assert (counts == 1).all()

    def test_no_column_name_collisions_after_join(self):
        """race_name vs circuit_name, driver_nationality vs constructor_nationality."""
        master = build_master_dataset(_basic_inputs())
        row = master.iloc[0]
        assert row["race_name"] != row["circuit_name"]
        assert "race_name" in master.columns and "circuit_name" in master.columns
        assert "driver_nationality" in master.columns
        assert "constructor_nationality" in master.columns

    def test_post_race_outcome_columns_present_for_history_use(self):
        master = build_master_dataset(_basic_inputs())
        assert POST_RACE_OUTCOME_COLUMNS <= set(master.columns)

    def test_fan_out_from_duplicate_dimension_key_raises(self):
        """A duplicate circuitId in circuits.csv would otherwise silently
        double every row that joins to it -- must raise, not corrupt the grain."""
        inputs = _basic_inputs()
        inputs["circuits"] = pd.concat(
            [inputs["circuits"], pd.DataFrame([_circuits_row(circuitId=1, circuitRef="dupe")])],
            ignore_index=True,
        )
        with pytest.raises(Exception):
            build_master_dataset(inputs)


# ---------------------------------------------------------------------------
# validate_output()
# ---------------------------------------------------------------------------

class TestValidateOutput:
    def test_valid_output_passes(self):
        master = build_master_dataset(_basic_inputs())
        result = validate_output(master, expected_row_count=len(master))
        assert result.passed

    def test_detects_row_count_mismatch(self):
        master = build_master_dataset(_basic_inputs())
        result = validate_output(master, expected_row_count=len(master) + 1)
        assert not result.passed
        assert any("Row count mismatch" in e for e in result.errors)

    def test_detects_missing_column(self):
        master = build_master_dataset(_basic_inputs()).drop(columns=["winner"])
        result = validate_output(master, expected_row_count=len(master))
        assert not result.passed
        assert any("missing expected column" in e for e in result.errors)

    def test_detects_extra_column(self):
        master = build_master_dataset(_basic_inputs())
        master["unexpected_column"] = 1
        result = validate_output(master, expected_row_count=len(master))
        assert not result.passed
        assert any("extra column" in e for e in result.errors)

    def test_detects_duplicate_race_driver_pair(self):
        master = build_master_dataset(_basic_inputs())
        dup = pd.concat([master, master.iloc[[0]]], ignore_index=True)
        result = validate_output(dup, expected_row_count=len(dup))
        assert not result.passed
        assert any("duplicate" in e for e in result.errors)

    def test_detects_null_identifier(self):
        master = build_master_dataset(_basic_inputs())
        master.loc[0, "year"] = None
        result = validate_output(master, expected_row_count=len(master))
        assert not result.passed
        assert any("year" in e for e in result.errors)

    def test_detects_orphan_foreign_key(self):
        """Simulates a driverId with no resolved dimension row (referential
        integrity violation) even though the row count didn't change."""
        master = build_master_dataset(_basic_inputs())
        master.loc[0, "driver_ref"] = None
        result = validate_output(master, expected_row_count=len(master))
        assert not result.passed
        assert any("driverId" in e for e in result.errors)

    def test_warns_on_missing_winner(self):
        inputs = _basic_inputs()
        # Make race 1 have zero winners by flipping the winning row's positionOrder
        inputs["results"].loc[
            (inputs["results"].raceId == 1) & (inputs["results"].driverId == 1),
            "positionOrder",
        ] = 5
        master = build_master_dataset(inputs)
        result = validate_output(master, expected_row_count=len(master))
        assert result.passed  # warning only, not blocking
        assert any("winner" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# End-to-end smoke test against real project data
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_dry_run_against_real_interim_data(self):
        """
        Full pipeline against the actual data/interim/*.parquet + data/*.csv
        files in this project. Skips if the interim datasets haven't been
        built yet (fresh checkout before running build_interim.py).
        """
        from pathlib import Path

        from src.pipelines.build_dataset import build_dataset

        project_root = Path(__file__).resolve().parents[1]
        interim_dir = project_root / "data" / "interim"
        if not (interim_dir / "results.parquet").exists() or \
           not (interim_dir / "qualifying.parquet").exists():
            pytest.skip("Interim datasets not built — run src/data/build_interim.py first.")

        master = build_dataset(dry_run=True)
        assert list(master.columns) == list(MASTER_DATASET_COLUMNS)
        assert len(master) > 0
        assert not master.duplicated(subset=["raceId", "driverId"]).any()
