"""
Tests for src/data/build_interim.py (Decision 007 repair logic + pipeline)

Covers:
  - _repair_duplicates: Finished preferred over non-Finished, ties broken by
    highest resultId, multiple groups, no-op on clean data, idempotency,
    input not mutated
  - _repair_positions: numeric positionText fills a null position on finished
    rows, non-finished rows untouched, idempotency, non-numeric positionText
    left null
  - build_interim end-to-end: both repairs applied, validation passes,
    parquet written / dry-run skips the write, validation failure raises
  - build_qualifying_interim end-to-end: clean + validate + write / dry-run
"""

import pandas as pd
import pytest

from src.data.build_interim import (
    _repair_duplicates,
    _repair_positions,
    build_interim,
    build_qualifying_interim,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleaned_row(**overrides) -> dict:
    """Minimal row matching clean_results() output; override as needed."""
    base = {
        "resultId":      1,
        "raceId":        100,
        "driverId":      1,
        "constructorId": 1,
        "grid":          1,
        "position":      1,
        "positionText":  "1",
        "points":        25.0,
        "statusId":      1,
        "result_status": "Finished",
        "finished":      True,
    }
    base.update(overrides)
    return base


def _cleaned_df(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df["position"] = df["position"].astype("Int64")
    return df


def _raw_results_df() -> pd.DataFrame:
    """
    Raw results.csv-shaped frame exercising both Decision-007 repairs:
      - driver 2 has a duplicate (raceId, driverId) pair: one Finished row
        (resultId=2) and one Retired row with a HIGHER resultId (5) — the
        Finished row must win despite the lower resultId.
      - driver 3 finished but has a null position with numeric positionText.
    """
    rows = [
        dict(resultId=1, raceId=1, driverId=1, constructorId=1, grid=1,
             position=1, positionText="1", points=25.0, laps=57, statusId=1),
        dict(resultId=2, raceId=1, driverId=2, constructorId=1, grid=2,
             position=2, positionText="2", points=18.0, laps=57, statusId=1),
        dict(resultId=5, raceId=1, driverId=2, constructorId=1, grid=2,
             position=None, positionText="R", points=0.0, laps=10, statusId=5),
        dict(resultId=3, raceId=1, driverId=3, constructorId=2, grid=3,
             position=None, positionText="3", points=15.0, laps=57, statusId=1),
        dict(resultId=4, raceId=1, driverId=4, constructorId=2, grid=4,
             position=None, positionText="R", points=0.0, laps=30, statusId=4),
    ]
    return pd.DataFrame(rows)


def _raw_qualifying_df() -> pd.DataFrame:
    rows = [
        dict(qualifyId=1, raceId=1, driverId=1, constructorId=1, number=44,
             position=1, q1="1:20.1", q2="1:19.9", q3="1:19.5"),
        dict(qualifyId=2, raceId=1, driverId=2, constructorId=1, number=63,
             position=11, q1="1:21.0", q2="1:20.8", q3=None),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _repair_duplicates — Decision 007 arbitration order
# ---------------------------------------------------------------------------

class TestRepairDuplicates:
    def test_prefers_finished_over_non_finished(self):
        """Finished wins even when the non-Finished row has a higher resultId."""
        df = _cleaned_df(
            _cleaned_row(resultId=10, raceId=1, driverId=1,
                         result_status="Finished", finished=True),
            _cleaned_row(resultId=99, raceId=1, driverId=1, position=None,
                         positionText="R", result_status="Retired", finished=False),
        )
        deduped, n_dropped = _repair_duplicates(df)
        assert n_dropped == 1
        assert deduped["resultId"].tolist() == [10]
        assert deduped.iloc[0]["result_status"] == "Finished"

    def test_finished_tie_broken_by_highest_result_id(self):
        """Two Finished rows: the most recently added (highest resultId) wins."""
        df = _cleaned_df(
            _cleaned_row(resultId=10, raceId=1, driverId=1),
            _cleaned_row(resultId=20, raceId=1, driverId=1),
        )
        deduped, n_dropped = _repair_duplicates(df)
        assert n_dropped == 1
        assert deduped["resultId"].tolist() == [20]

    def test_non_finished_tie_broken_by_highest_result_id(self):
        df = _cleaned_df(
            _cleaned_row(resultId=10, raceId=1, driverId=1, position=None,
                         positionText="R", result_status="Retired", finished=False),
            _cleaned_row(resultId=20, raceId=1, driverId=1, position=None,
                         positionText="R", result_status="Retired", finished=False),
        )
        deduped, n_dropped = _repair_duplicates(df)
        assert n_dropped == 1
        assert deduped["resultId"].tolist() == [20]

    def test_no_duplicates_is_a_no_op(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, raceId=1, driverId=1),
            _cleaned_row(resultId=2, raceId=1, driverId=2),
            _cleaned_row(resultId=3, raceId=2, driverId=1),
        )
        deduped, n_dropped = _repair_duplicates(df)
        assert n_dropped == 0
        assert sorted(deduped["resultId"].tolist()) == [1, 2, 3]

    def test_same_driver_different_races_is_not_a_duplicate(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, raceId=1, driverId=1),
            _cleaned_row(resultId=2, raceId=2, driverId=1),
        )
        _, n_dropped = _repair_duplicates(df)
        assert n_dropped == 0

    def test_multiple_groups_resolved_independently(self):
        df = _cleaned_df(
            # group A: Finished vs Retired
            _cleaned_row(resultId=1, raceId=1, driverId=1),
            _cleaned_row(resultId=2, raceId=1, driverId=1, position=None,
                         positionText="R", result_status="Retired", finished=False),
            # group B: Finished tie
            _cleaned_row(resultId=3, raceId=1, driverId=2),
            _cleaned_row(resultId=4, raceId=1, driverId=2),
            # untouched singleton
            _cleaned_row(resultId=5, raceId=1, driverId=3),
        )
        deduped, n_dropped = _repair_duplicates(df)
        assert n_dropped == 2
        assert sorted(deduped["resultId"].tolist()) == [1, 4, 5]

    def test_three_way_group_keeps_exactly_one(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, raceId=1, driverId=1, position=None,
                         positionText="R", result_status="Retired", finished=False),
            _cleaned_row(resultId=2, raceId=1, driverId=1),
            _cleaned_row(resultId=3, raceId=1, driverId=1, position=None,
                         positionText="W", result_status="Withdrawn", finished=False),
        )
        deduped, n_dropped = _repair_duplicates(df)
        assert n_dropped == 2
        assert deduped["resultId"].tolist() == [2]

    def test_idempotent(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, raceId=1, driverId=1),
            _cleaned_row(resultId=2, raceId=1, driverId=1),
        )
        once, _ = _repair_duplicates(df)
        twice, n_second = _repair_duplicates(once)
        assert n_second == 0
        pd.testing.assert_frame_equal(once, twice)

    def test_helper_column_not_leaked(self):
        df = _cleaned_df(_cleaned_row())
        deduped, _ = _repair_duplicates(df)
        assert "_is_finished" not in deduped.columns

    def test_input_not_mutated(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, raceId=1, driverId=1),
            _cleaned_row(resultId=2, raceId=1, driverId=1),
        )
        original = df.copy()
        _repair_duplicates(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# _repair_positions — Decision 007 positionText fill
# ---------------------------------------------------------------------------

class TestRepairPositions:
    def test_fills_null_position_from_numeric_position_text(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=None, positionText="7"),
        )
        repaired, n_fixed = _repair_positions(df)
        assert n_fixed == 1
        assert repaired.loc[0, "position"] == 7

    def test_untouched_rows_keep_their_position(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=1, positionText="1"),
            _cleaned_row(resultId=2, driverId=2, position=None, positionText="4"),
        )
        repaired, n_fixed = _repair_positions(df)
        assert n_fixed == 1
        assert repaired.loc[0, "position"] == 1
        assert repaired.loc[1, "position"] == 4

    def test_non_finished_null_position_is_not_filled(self):
        """A retired driver's null position is correct — never fabricate one."""
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=None, positionText="R",
                         result_status="Retired", finished=False),
        )
        repaired, n_fixed = _repair_positions(df)
        assert n_fixed == 0
        assert pd.isna(repaired.loc[0, "position"])

    def test_non_numeric_position_text_stays_null(self):
        """Extraction only accepts pure digit strings."""
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=None, positionText="3rd"),
        )
        repaired, _ = _repair_positions(df)
        assert pd.isna(repaired.loc[0, "position"])

    def test_no_repairs_needed_is_a_no_op(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=1),
            _cleaned_row(resultId=2, driverId=2, position=2, positionText="2"),
        )
        repaired, n_fixed = _repair_positions(df)
        assert n_fixed == 0
        pd.testing.assert_frame_equal(repaired, df)

    def test_idempotent(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=None, positionText="5"),
        )
        once, n_first = _repair_positions(df)
        twice, n_second = _repair_positions(once)
        assert n_first == 1
        assert n_second == 0
        pd.testing.assert_frame_equal(once, twice)

    def test_input_not_mutated(self):
        df = _cleaned_df(
            _cleaned_row(resultId=1, position=None, positionText="5"),
        )
        original = df.copy()
        _repair_positions(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# build_interim — end-to-end pipeline (load monkeypatched)
# ---------------------------------------------------------------------------

class TestBuildInterim:
    @pytest.fixture
    def patched_load(self, monkeypatch):
        monkeypatch.setattr(
            "src.data.build_interim.load_csv",
            lambda name: _raw_results_df(),
        )

    def test_applies_both_repairs_and_validates(self, patched_load, tmp_path):
        out = tmp_path / "results.parquet"
        df = build_interim(output_path=out)
        # duplicate pair for driver 2 resolved to the Finished row
        driver2 = df[df["driverId"] == 2]
        assert len(driver2) == 1
        assert driver2.iloc[0]["resultId"] == 2
        assert driver2.iloc[0]["result_status"] == "Finished"
        # driver 3's null position filled from positionText
        driver3 = df[df["driverId"] == 3]
        assert driver3.iloc[0]["position"] == 3
        # one row per (raceId, driverId)
        assert not df.duplicated(subset=["raceId", "driverId"]).any()
        assert len(df) == 4

    def test_writes_parquet(self, patched_load, tmp_path):
        out = tmp_path / "results.parquet"
        returned = build_interim(output_path=out)
        assert out.exists()
        written = pd.read_parquet(out)
        assert len(written) == len(returned)

    def test_dry_run_skips_write(self, patched_load, tmp_path):
        out = tmp_path / "results.parquet"
        build_interim(dry_run=True, output_path=out)
        assert not out.exists()

    def test_validation_failure_raises(self, monkeypatch, tmp_path):
        bad = _raw_results_df()
        bad["points"] = -1.0  # violates the points >= 0 constraint
        monkeypatch.setattr(
            "src.data.build_interim.load_csv", lambda name: bad
        )
        with pytest.raises(ValueError):
            build_interim(output_path=tmp_path / "results.parquet")


# ---------------------------------------------------------------------------
# build_qualifying_interim — end-to-end pipeline (load monkeypatched)
# ---------------------------------------------------------------------------

class TestBuildQualifyingInterim:
    @pytest.fixture
    def patched_load(self, monkeypatch):
        monkeypatch.setattr(
            "src.data.build_interim.load_csv",
            lambda name: _raw_qualifying_df(),
        )

    def test_cleans_and_writes_parquet(self, patched_load, tmp_path):
        out = tmp_path / "qualifying.parquet"
        df = build_qualifying_interim(output_path=out)
        assert out.exists()
        assert str(df["raceId"].dtype) == "Int64"
        # q times stay raw strings / NaN
        assert df.loc[0, "q3"] == "1:19.5"
        assert pd.isna(df.loc[1, "q3"])

    def test_dry_run_skips_write(self, patched_load, tmp_path):
        out = tmp_path / "qualifying.parquet"
        build_qualifying_interim(dry_run=True, output_path=out)
        assert not out.exists()

    def test_duplicate_qualifying_rows_fail_validation(self, monkeypatch, tmp_path):
        """Qualifying has no repair step — duplicates must raise, not be fixed."""
        dupes = pd.concat(
            [_raw_qualifying_df(), _raw_qualifying_df()], ignore_index=True
        )
        monkeypatch.setattr(
            "src.data.build_interim.load_csv", lambda name: dupes
        )
        with pytest.raises(ValueError):
            build_qualifying_interim(output_path=tmp_path / "qualifying.parquet")
