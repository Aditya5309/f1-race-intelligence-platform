"""
Tests for scripts/ingest_jolpica.py (Phase 4 Tranche D, Part 2).

No live network calls — fixtures below mirror the REAL jolpica-f1 JSON
shapes verified directly against api.jolpi.ca while building this script
(2026 rounds 7-9: 22 results/qualifying rows, "R" positionText for
non-classified drivers, FastestLap/Time blocks present for finishers).

scripts/ is not a package (mirrors every other scripts/*.py — none have
tests either except promote_model.py, which set this importlib pattern),
so the module is loaded directly via importlib.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_jolpica", _PROJECT_ROOT / "scripts" / "ingest_jolpica.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest_jolpica = _load_module()


# ---------------------------------------------------------------------------
# normalize_finish — the Decision 035 positionText/status handling
# ---------------------------------------------------------------------------

def test_normalize_finish_classified_row():
    position, position_text, status_id = ingest_jolpica.normalize_finish("6")
    assert position == "6"
    assert position_text == "6"
    assert status_id == ingest_jolpica.GENERIC_FINISHED_STATUS_ID


def test_normalize_finish_non_classified_row():
    """jolpica already uses Ergast's own "R" code for every non-classified
    row (verified live) — this must NOT be reinterpreted as historical
    Ergast "N" (Did Not Start); it's taken at face value and bucketed into
    the existing generic "Retired" statusId (31), never invented."""
    position, position_text, status_id = ingest_jolpica.normalize_finish("R")
    assert position == ingest_jolpica.NA_TOKEN
    assert position_text == "R"
    assert status_id == ingest_jolpica.GENERIC_RETIRED_STATUS_ID


def test_normalize_finish_downstream_classification_is_correct(tmp_path):
    """End-to-end proof the normalization produces correct output through
    THIS project's own unmodified classification logic — not just that
    normalize_finish() looks right in isolation."""
    from src.data.cleaner import clean_results

    rows = []
    for i, pos_text in enumerate(["1", "2", "R", "R"], start=1):
        position, position_text, status_id = ingest_jolpica.normalize_finish(pos_text)
        rows.append({
            "resultId": i, "raceId": 9001, "driverId": i, "constructorId": 1,
            "number": i, "grid": i, "position": position, "positionText": position_text,
            "positionOrder": i, "points": "0", "laps": "50", "time": ingest_jolpica.NA_TOKEN,
            "milliseconds": ingest_jolpica.NA_TOKEN, "fastestLap": ingest_jolpica.NA_TOKEN,
            "rank": ingest_jolpica.NA_TOKEN, "fastestLapTime": ingest_jolpica.NA_TOKEN,
            "fastestLapSpeed": ingest_jolpica.NA_TOKEN, "statusId": status_id,
        })
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path, na_values=[ingest_jolpica.NA_TOKEN])

    cleaned = clean_results(frame)
    assert list(cleaned["result_status"]) == ["Finished", "Finished", "Retired", "Retired"]
    assert list(cleaned["finished"]) == [True, True, False, False]


# ---------------------------------------------------------------------------
# IdReconciler — *Ref-based lookup, minting new IDs off the max sequence
# ---------------------------------------------------------------------------

@pytest.fixture()
def drivers_df():
    return pd.DataFrame({
        "driverId": [1, 2, 3],
        "driverRef": ["hamilton", "verstappen", "russell"],
        "forename": ["Lewis", "Max", "George"],
        "surname": ["Hamilton", "Verstappen", "Russell"],
    })


def test_id_reconciler_resolves_existing_ref(drivers_df):
    reconciler = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    assert reconciler.resolve("hamilton", {"forename": "Lewis"}) == 1
    assert reconciler.new_rows == []


def test_id_reconciler_mints_new_id_for_new_ref(drivers_df):
    reconciler = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    new_id = reconciler.resolve("piastri", {"forename": "Oscar", "surname": "Piastri"})
    assert new_id == 4   # max(1,2,3) + 1
    assert reconciler.new_rows == [
        {"driverId": 4, "driverRef": "piastri", "forename": "Oscar", "surname": "Piastri"}
    ]


def test_id_reconciler_mints_sequential_ids_for_multiple_new_refs(drivers_df):
    reconciler = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    first = reconciler.resolve("piastri", {})
    second = reconciler.resolve("antonelli", {})
    assert (first, second) == (4, 5)
    assert len(reconciler.new_rows) == 2


def test_id_reconciler_same_new_ref_resolves_to_same_id(drivers_df):
    """A driver appearing in both results AND qualifying for the same race
    must resolve to the same newly-minted ID both times, not two different
    ones — resolve() is called once per row in each payload."""
    reconciler = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    first = reconciler.resolve("piastri", {"forename": "Oscar"})
    second = reconciler.resolve("piastri", {"forename": "Oscar"})
    assert first == second == 4
    assert len(reconciler.new_rows) == 1


# ---------------------------------------------------------------------------
# missing_completed_races
# ---------------------------------------------------------------------------

def test_missing_completed_races_finds_races_with_no_results():
    races = pd.DataFrame({
        "raceId": [1, 2, 3], "year": [2026, 2026, 2026], "round": [7, 8, 9],
        "name": ["A", "B", "C"],
    })
    results = pd.DataFrame({"raceId": [1, 1]})   # only race 1 has results
    missing = ingest_jolpica.missing_completed_races(races, results)
    assert sorted(missing["raceId"]) == [2, 3]


def test_missing_completed_races_empty_when_all_have_results():
    races = pd.DataFrame({"raceId": [1, 2], "year": [2026, 2026], "round": [7, 8]})
    results = pd.DataFrame({"raceId": [1, 1, 2]})
    assert ingest_jolpica.missing_completed_races(races, results).empty


# ---------------------------------------------------------------------------
# build_results_rows — real jolpica JSON shape (verified live)
# ---------------------------------------------------------------------------

_RAW_RESULT_FINISHER = {
    "number": "44", "position": "1", "positionText": "1", "points": "25",
    "Driver": {"driverId": "hamilton", "givenName": "Lewis", "familyName": "Hamilton",
               "code": "HAM", "permanentNumber": "44", "dateOfBirth": "1985-01-07",
               "nationality": "British", "url": "http://x/Lewis_Hamilton"},
    "Constructor": {"constructorId": "ferrari", "name": "Ferrari",
                    "nationality": "Italian", "url": "http://x/Ferrari"},
    "grid": "2", "laps": "66", "status": "Finished",
    "Time": {"millis": "5548105", "time": "1:32:28.105"},
    "FastestLap": {"rank": "1", "lap": "44", "Time": {"time": "1:20.122"}},
}

_RAW_RESULT_RETIRED = {
    "number": "14", "position": "18", "positionText": "R", "points": "0",
    "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso",
               "code": "ALO", "permanentNumber": "14", "dateOfBirth": "1981-07-29",
               "nationality": "Spanish", "url": "http://x/Fernando_Alonso"},
    "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin",
                    "nationality": "British", "url": "http://x/Aston_Martin"},
    "grid": "10", "laps": "40", "status": "Retired",
}


def test_build_results_rows_finisher(drivers_df):
    constructors_df = pd.DataFrame({"constructorId": [1], "constructorRef": ["mercedes"]})
    drivers = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    constructors = ingest_jolpica.IdReconciler(constructors_df, "constructorId", "constructorRef")

    rows = ingest_jolpica.build_results_rows(
        [_RAW_RESULT_FINISHER], race_id=9001, next_result_id=500,
        drivers=drivers, constructors=constructors,
    )
    row = rows[0]
    assert row["resultId"] == 500
    assert row["raceId"] == 9001
    assert row["driverId"] == 1          # existing "hamilton" ref -> driverId 1
    assert row["constructorId"] == 2     # new "ferrari" ref -> minted id (max 1 + 1)
    assert row["position"] == "1"
    assert row["positionOrder"] == 1
    assert row["statusId"] == ingest_jolpica.GENERIC_FINISHED_STATUS_ID
    assert row["milliseconds"] == "5548105"
    assert row["fastestLapTime"] == "1:20.122"
    assert row["fastestLapSpeed"] == ingest_jolpica.NA_TOKEN  # jolpica never provides this


def test_build_results_rows_retired_gets_generic_status(drivers_df):
    constructors_df = pd.DataFrame({"constructorId": [1], "constructorRef": ["mercedes"]})
    drivers = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    constructors = ingest_jolpica.IdReconciler(constructors_df, "constructorId", "constructorRef")

    rows = ingest_jolpica.build_results_rows(
        [_RAW_RESULT_RETIRED], race_id=9001, next_result_id=500,
        drivers=drivers, constructors=constructors,
    )
    row = rows[0]
    assert row["position"] == ingest_jolpica.NA_TOKEN
    assert row["positionText"] == "R"
    assert row["statusId"] == ingest_jolpica.GENERIC_RETIRED_STATUS_ID
    assert drivers.new_rows[0]["driverRef"] == "alonso"


def test_build_results_rows_positionorder_matches_array_order(drivers_df):
    """jolpica's Results array is already finish-order sorted (verified
    live) — positionOrder is derived from array position, not re-sorted."""
    constructors_df = pd.DataFrame({"constructorId": [1], "constructorRef": ["mercedes"]})
    drivers = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    constructors = ingest_jolpica.IdReconciler(constructors_df, "constructorId", "constructorRef")

    rows = ingest_jolpica.build_results_rows(
        [_RAW_RESULT_FINISHER, _RAW_RESULT_RETIRED], race_id=9001, next_result_id=500,
        drivers=drivers, constructors=constructors,
    )
    assert [r["positionOrder"] for r in rows] == [1, 2]
    assert [r["resultId"] for r in rows] == [500, 501]
