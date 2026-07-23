"""
Tests for scripts/ingest_jolpica.py.

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
import json
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
# normalize_finish — positionText/status handling
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
# resolve_upcoming_qualifying_target — Phase 2 (Decisions 049/050)
# ---------------------------------------------------------------------------

def _races_calendar() -> pd.DataFrame:
    return pd.DataFrame({
        "raceId": [1, 2, 3], "year": [2026, 2026, 2026], "round": [7, 8, 9],
        "circuitId": [10, 11, 12], "name": ["A", "B", "C"],
        "date": ["2026-01-01", "2026-01-08", "2026-01-15"],
    })


def test_resolve_upcoming_qualifying_target_finds_next_race_with_no_qualifying():
    races = _races_calendar()
    results = pd.DataFrame({"raceId": [1]})       # race 1 completed
    qualifying = pd.DataFrame({"raceId": []})      # nothing ingested yet

    target = ingest_jolpica.resolve_upcoming_qualifying_target(races, results, qualifying)

    assert target.race_id == 2
    assert target.year == 2026
    assert target.round == 8


def test_resolve_upcoming_qualifying_target_none_when_qualifying_already_on_file():
    """Idempotent: a prior run already landed this race's qualifying."""
    races = _races_calendar()
    results = pd.DataFrame({"raceId": [1]})
    qualifying = pd.DataFrame({"raceId": [2]})     # race 2's qualifying already ingested

    assert ingest_jolpica.resolve_upcoming_qualifying_target(races, results, qualifying) is None


def test_resolve_upcoming_qualifying_target_none_when_ingested_this_run():
    """The 'upcoming' race turned out to have just finished and was already
    fully ingested (results + qualifying) earlier in the SAME run — must not
    be double-fetched here."""
    races = _races_calendar()
    results = pd.DataFrame({"raceId": [1]})
    qualifying = pd.DataFrame({"raceId": []})

    target = ingest_jolpica.resolve_upcoming_qualifying_target(
        races, results, qualifying, already_ingested_race_ids={2},
    )

    assert target is None


def test_resolve_upcoming_qualifying_target_none_when_no_upcoming_race():
    races = _races_calendar()
    results = pd.DataFrame({"raceId": [1, 2, 3]})  # every race already completed
    qualifying = pd.DataFrame({"raceId": []})

    assert ingest_jolpica.resolve_upcoming_qualifying_target(races, results, qualifying) is None


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


# ---------------------------------------------------------------------------
# build_qualifying_rows — real jolpica JSON shape, including partial-session
# states (a driver eliminated in Q1 has no Q2/Q3; eliminated in Q2 has no
# Q3) — the normal, expected shape of EVERY qualifying session (15 of 20
# drivers never see Q3), not a special "session in progress" case. First
# coverage for this function (previously exercised only implicitly, never
# asserted on directly).
# ---------------------------------------------------------------------------

_RAW_QUALIFYING_ALL_SESSIONS = {
    "number": "1", "position": "1",
    "Driver": {"driverId": "max_verstappen", "givenName": "Max", "familyName": "Verstappen",
               "code": "VER", "permanentNumber": "1", "dateOfBirth": "1997-09-30",
               "nationality": "Dutch", "url": "http://x/Max_Verstappen"},
    "Constructor": {"constructorId": "red_bull", "name": "Red Bull",
                    "nationality": "Austrian", "url": "http://x/Red_Bull"},
    "Q1": "1:29.000", "Q2": "1:28.500", "Q3": "1:28.000",
}

_RAW_QUALIFYING_ELIMINATED_Q2 = {
    "number": "14", "position": "12",
    "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso",
               "code": "ALO", "permanentNumber": "14", "dateOfBirth": "1981-07-29",
               "nationality": "Spanish", "url": "http://x/Fernando_Alonso"},
    "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin",
                    "nationality": "British", "url": "http://x/Aston_Martin"},
    "Q1": "1:29.800", "Q2": "1:29.400",
    # no Q3 key — eliminated in Q2, exactly like the real jolpica payload.
}

_RAW_QUALIFYING_ELIMINATED_Q1 = {
    "number": "27", "position": "18",
    "Driver": {"driverId": "hulkenberg", "givenName": "Nico", "familyName": "Hulkenberg",
               "code": "HUL", "permanentNumber": "27", "dateOfBirth": "1987-08-19",
               "nationality": "German", "url": "http://x/Nico_Hulkenberg"},
    "Constructor": {"constructorId": "haas", "name": "Haas F1 Team",
                    "nationality": "American", "url": "http://x/Haas"},
    "Q1": "1:30.500",
    # no Q2/Q3 keys — eliminated in Q1.
}


def test_build_qualifying_rows_all_sessions_present(drivers_df):
    constructors_df = pd.DataFrame({"constructorId": [1], "constructorRef": ["mercedes"]})
    drivers = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    constructors = ingest_jolpica.IdReconciler(constructors_df, "constructorId", "constructorRef")

    rows = ingest_jolpica.build_qualifying_rows(
        [_RAW_QUALIFYING_ALL_SESSIONS], race_id=9001, next_qualify_id=700,
        drivers=drivers, constructors=constructors,
    )
    row = rows[0]
    assert row["qualifyId"] == 700
    assert row["raceId"] == 9001
    assert row["position"] == "1"
    assert row["q1"] == "1:29.000"
    assert row["q2"] == "1:28.500"
    assert row["q3"] == "1:28.000"


def test_build_qualifying_rows_partial_session_states_get_na_token(drivers_df):
    """The named Phase 2 test scenario: Q1 done for everyone, Q2/Q3 only for
    those who advanced. Missing sessions become NA_TOKEN, never fabricated
    or dropped — same discipline as every other informative-missingness
    column in this project (domain_knowledge.md §8)."""
    constructors_df = pd.DataFrame({"constructorId": [1], "constructorRef": ["mercedes"]})
    drivers = ingest_jolpica.IdReconciler(drivers_df, "driverId", "driverRef")
    constructors = ingest_jolpica.IdReconciler(constructors_df, "constructorId", "constructorRef")

    rows = ingest_jolpica.build_qualifying_rows(
        [_RAW_QUALIFYING_ALL_SESSIONS, _RAW_QUALIFYING_ELIMINATED_Q2, _RAW_QUALIFYING_ELIMINATED_Q1],
        race_id=9001, next_qualify_id=700, drivers=drivers, constructors=constructors,
    )

    eliminated_q2_row = rows[1]
    assert eliminated_q2_row["q1"] == "1:29.800"
    assert eliminated_q2_row["q2"] == "1:29.400"
    assert eliminated_q2_row["q3"] == ingest_jolpica.NA_TOKEN

    eliminated_q1_row = rows[2]
    assert eliminated_q1_row["q1"] == "1:30.500"
    assert eliminated_q1_row["q2"] == ingest_jolpica.NA_TOKEN
    assert eliminated_q1_row["q3"] == ingest_jolpica.NA_TOKEN
    assert drivers.new_rows[-1]["driverRef"] == "hulkenberg"


# ---------------------------------------------------------------------------
# write_ingest_report — Part 2 weekly-verification artifact
# ---------------------------------------------------------------------------

def test_write_ingest_report_writes_summary_and_new_row_csvs(tmp_path):
    report_dir = tmp_path / "ingest_report"
    ingested = [{
        "year": 2026, "round": 10, "name": "Test GP", "raceId": 9001,
        "n_results": 2, "n_qualifying": 2, "n_driver_standings": 2,
        "n_constructor_standings": 2,
    }]
    skipped = [{"year": 2026, "round": 11, "name": "Future GP"}]
    new_results_rows = [{"resultId": 1, "raceId": 9001}, {"resultId": 2, "raceId": 9001}]
    new_drivers = [{"driverId": 4, "driverRef": "piastri"}]

    returned = ingest_jolpica.write_ingest_report(
        report_dir, dry_run=False, ingested=ingested, skipped=skipped,
        new_drivers=new_drivers, new_constructors=[],
        new_results_rows=new_results_rows, new_qualifying_rows=[],
        new_driver_standings_rows=[], new_constructor_standings_rows=[],
    )

    assert returned == report_dir
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["dry_run"] is False
    assert summary["ingested_races"] == ingested
    assert summary["skipped_races"] == skipped
    assert summary["upcoming_qualifying"] is None  # not passed -> default
    assert summary["totals"] == {
        "races_ingested": 1, "races_skipped": 1, "results_rows": 2,
        "qualifying_rows": 0, "driver_standings_rows": 0,
        "constructor_standings_rows": 0, "new_drivers": 1, "new_constructors": 0,
    }
    assert "generated_at" in summary

    assert (report_dir / "new_results.csv").exists()
    assert len(pd.read_csv(report_dir / "new_results.csv")) == 2
    assert (report_dir / "new_drivers.csv").exists()
    # nothing new on these endpoints this run — no empty file clutter
    assert not (report_dir / "new_qualifying.csv").exists()
    assert not (report_dir / "new_constructors.csv").exists()


def test_write_ingest_report_includes_upcoming_qualifying_when_given(tmp_path):
    """Phase 2 (Decisions 049/050): the upcoming-race qualifying summary
    rides along in the same report, additive to the existing fields."""
    report_dir = tmp_path / "ingest_report"
    upcoming = {
        "year": 2026, "round": 11, "name": "Upcoming GP", "raceId": 9002,
        "n_qualifying_rows": 20,
    }

    ingest_jolpica.write_ingest_report(
        report_dir, dry_run=False, ingested=[], skipped=[],
        new_drivers=[], new_constructors=[], new_results_rows=[],
        new_qualifying_rows=[], new_driver_standings_rows=[],
        new_constructor_standings_rows=[], upcoming_qualifying=upcoming,
    )

    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["upcoming_qualifying"] == upcoming


def test_write_ingest_report_nothing_new_still_writes_summary(tmp_path):
    report_dir = tmp_path / "ingest_report"
    ingest_jolpica.write_ingest_report(
        report_dir, dry_run=False, ingested=[], skipped=[],
        new_drivers=[], new_constructors=[], new_results_rows=[],
        new_qualifying_rows=[], new_driver_standings_rows=[],
        new_constructor_standings_rows=[],
    )
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["totals"]["races_ingested"] == 0
    assert list(report_dir.iterdir()) == [report_dir / "summary.json"]


def test_write_ingest_report_creates_report_dir(tmp_path):
    report_dir = tmp_path / "nested" / "ingest_report"
    assert not report_dir.exists()
    ingest_jolpica.write_ingest_report(
        report_dir, dry_run=True, ingested=[], skipped=[],
        new_drivers=[], new_constructors=[], new_results_rows=[],
        new_qualifying_rows=[], new_driver_standings_rows=[],
        new_constructor_standings_rows=[],
    )
    assert report_dir.exists()
