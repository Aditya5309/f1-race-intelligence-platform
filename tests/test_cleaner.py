"""
Tests for src/data/cleaner.py

Covers:
  - Normal race finishers (various grid positions)
  - Lapped classified finishers (+N Laps)
  - Retired drivers (mechanical failure, accident)
  - Disqualified drivers (DSQ, Excluded)
  - Did Not Start (DNS, Withdrawn, 107% rule, DNQ)
  - Null position is preserved for non-finishers
  - Dtype enforcement (Int64 nullable, float64)
  - Validation errors for missing/null required columns
  - statusId fallback when positionText is absent
"""

import pandas as pd
import pytest

from src.data.cleaner import clean_qualifying, clean_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**overrides) -> dict:
    """Return a minimal valid result row; override any field as needed."""
    base = {
        "resultId":      1,
        "raceId":        18,
        "driverId":      1,
        "constructorId": 1,
        "number":        22,
        "grid":          1,
        "position":      1.0,
        "positionText":  "1",
        "positionOrder": 1,
        "points":        25.0,
        "laps":          57,
        "time":          "1:34:50.616",
        "milliseconds":  5690616.0,
        "fastestLap":    39.0,
        "rank":          2.0,
        "fastestLapTime": "1:27.452",
        "fastestLapSpeed": 218.3,
        "statusId":      1,
    }
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    """Build a DataFrame from one or more row dicts."""
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Finished drivers
# ---------------------------------------------------------------------------

class TestFinishedDrivers:
    def test_winner_is_finished(self):
        df = clean_results(_df(_make_row(positionText="1", position=1.0, statusId=1)))
        assert df.loc[0, "finished"] == True
        assert df.loc[0, "result_status"] == "Finished"

    def test_midfield_finisher(self):
        df = clean_results(_df(_make_row(positionText="10", position=10.0, statusId=11)))
        assert df.loc[0, "finished"] == True
        assert df.loc[0, "result_status"] == "Finished"

    def test_last_classified_finisher(self):
        df = clean_results(_df(_make_row(positionText="15", position=15.0, statusId=12)))
        assert df.loc[0, "finished"] == True
        assert df.loc[0, "result_status"] == "Finished"

    def test_position_is_non_null_for_finisher(self):
        df = clean_results(_df(_make_row(positionText="3", position=3.0, statusId=1)))
        assert pd.notna(df.loc[0, "position"])

    def test_finished_via_statusid_fallback(self):
        """statusId=1 should resolve to Finished when positionText is absent."""
        row = _make_row(statusId=1)
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "finished"] == True
        assert df.loc[0, "result_status"] == "Finished"


# ---------------------------------------------------------------------------
# Retired drivers
# ---------------------------------------------------------------------------

class TestRetiredDrivers:
    def test_retirement_via_position_text(self):
        df = clean_results(_df(_make_row(
            positionText="R", position=None, statusId=5  # Engine failure
        )))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Retired"

    def test_retirement_position_is_null(self):
        df = clean_results(_df(_make_row(positionText="R", position=None, statusId=5)))
        assert pd.isna(df.loc[0, "position"])

    def test_retirement_via_statusid_fallback(self):
        """Mechanical statusId with no positionText → Retired."""
        row = _make_row(position=None, statusId=5)  # Engine
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Retired"

    def test_accident_retirement(self):
        df = clean_results(_df(_make_row(positionText="R", position=None, statusId=3)))
        assert df.loc[0, "result_status"] == "Retired"

    def test_collision_retirement(self):
        df = clean_results(_df(_make_row(positionText="R", position=None, statusId=4)))
        assert df.loc[0, "result_status"] == "Retired"


# ---------------------------------------------------------------------------
# Disqualified drivers
# ---------------------------------------------------------------------------

class TestDisqualifiedDrivers:
    def test_dsq_via_position_text_D(self):
        df = clean_results(_df(_make_row(positionText="D", position=None, statusId=2)))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Disqualified"

    def test_dsq_via_position_text_E(self):
        """'E' (Excluded) should map to Disqualified."""
        df = clean_results(_df(_make_row(positionText="E", position=None, statusId=96)))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Disqualified"

    def test_dsq_via_statusid_underweight(self):
        row = _make_row(position=None, statusId=92)  # Underweight
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "result_status"] == "Disqualified"

    def test_dsq_via_statusid_excluded(self):
        row = _make_row(position=None, statusId=96)  # Excluded
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "result_status"] == "Disqualified"


# ---------------------------------------------------------------------------
# Did Not Start / Withdrawn
# ---------------------------------------------------------------------------

class TestDidNotStart:
    def test_dns_via_position_text_N(self):
        df = clean_results(_df(_make_row(positionText="N", position=None, statusId=81)))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Did Not Start"

    def test_failed_107_rule_via_position_text_F(self):
        df = clean_results(_df(_make_row(positionText="F", position=None, statusId=77)))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Did Not Start"

    def test_did_not_qualify_via_statusid(self):
        row = _make_row(position=None, statusId=81)
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "result_status"] == "Did Not Start"

    def test_did_not_prequalify_via_statusid(self):
        row = _make_row(position=None, statusId=97)
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "result_status"] == "Did Not Start"

    def test_withdrawn_via_position_text_W(self):
        df = clean_results(_df(_make_row(positionText="W", position=None, statusId=54)))
        assert df.loc[0, "finished"] == False
        assert df.loc[0, "result_status"] == "Withdrawn"

    def test_withdrew_via_statusid(self):
        row = _make_row(position=None, statusId=54)
        row.pop("positionText")
        df = clean_results(_df(row))
        assert df.loc[0, "result_status"] == "Did Not Start"


# ---------------------------------------------------------------------------
# Dtype enforcement
# ---------------------------------------------------------------------------

class TestDtypes:
    def setup_method(self):
        self.df = clean_results(_df(_make_row()))

    def test_raceId_is_Int64(self):
        assert self.df["raceId"].dtype == pd.Int64Dtype()

    def test_driverId_is_Int64(self):
        assert self.df["driverId"].dtype == pd.Int64Dtype()

    def test_constructorId_is_Int64(self):
        assert self.df["constructorId"].dtype == pd.Int64Dtype()

    def test_position_is_nullable_Int64(self):
        assert self.df["position"].dtype == pd.Int64Dtype()

    def test_points_is_float64(self):
        assert self.df["points"].dtype == "float64"

    def test_milliseconds_is_nullable_Int64(self):
        assert self.df["milliseconds"].dtype == pd.Int64Dtype()

    def test_rank_is_nullable_Int64(self):
        assert self.df["rank"].dtype == pd.Int64Dtype()

    def test_position_null_preserved_for_non_finisher(self):
        df = clean_results(_df(_make_row(positionText="R", position=None, statusId=5)))
        assert pd.isna(df.loc[0, "position"])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_raises_on_null_raceId(self):
        row = _make_row()
        row["raceId"] = None
        with pytest.raises(ValueError, match="raceId"):
            clean_results(_df(row))

    def test_raises_on_null_driverId(self):
        row = _make_row()
        row["driverId"] = None
        with pytest.raises(ValueError, match="driverId"):
            clean_results(_df(row))

    def test_raises_on_null_constructorId(self):
        row = _make_row()
        row["constructorId"] = None
        with pytest.raises(ValueError, match="constructorId"):
            clean_results(_df(row))

    def test_raises_on_missing_raceId_column(self):
        df = _df(_make_row())
        df = df.drop(columns=["raceId"])
        with pytest.raises(ValueError, match="raceId"):
            clean_results(df)

    def test_raises_on_missing_driverId_column(self):
        df = _df(_make_row())
        df = df.drop(columns=["driverId"])
        with pytest.raises(ValueError, match="driverId"):
            clean_results(df)


# ---------------------------------------------------------------------------
# clean_qualifying()
# ---------------------------------------------------------------------------

def _make_qualifying_row(**overrides) -> dict:
    """Return a minimal valid qualifying row; override any field as needed."""
    base = {
        "qualifyId":     1,
        "raceId":        18,
        "driverId":      1,
        "constructorId": 1,
        "number":        22,
        "position":      1,
        "q1":            "1:26.572",
        "q2":            "1:25.187",
        "q3":            "1:26.714",
    }
    base.update(overrides)
    return base


class TestCleanQualifying:
    def test_casts_id_columns_to_nullable_int(self):
        df = clean_qualifying(_df(_make_qualifying_row()))
        assert str(df["raceId"].dtype) == "Int64"
        assert str(df["driverId"].dtype) == "Int64"
        assert str(df["constructorId"].dtype) == "Int64"
        assert str(df["position"].dtype) == "Int64"

    def test_q1_q2_q3_left_as_raw_strings(self):
        df = clean_qualifying(_df(_make_qualifying_row()))
        assert df.loc[0, "q1"] == "1:26.572"
        assert df.loc[0, "q2"] == "1:25.187"
        assert df.loc[0, "q3"] == "1:26.714"

    def test_null_q3_preserved_not_imputed(self):
        """A driver eliminated in Q2 has no Q3 time -- this is informative, not missing."""
        df = clean_qualifying(_df(_make_qualifying_row(q3=None)))
        assert pd.isna(df.loc[0, "q3"])

    def test_raises_on_null_raceId(self):
        row = _make_qualifying_row()
        row["raceId"] = None
        with pytest.raises(ValueError, match="raceId"):
            clean_qualifying(_df(row))

    def test_raises_on_null_driverId(self):
        row = _make_qualifying_row()
        row["driverId"] = None
        with pytest.raises(ValueError, match="driverId"):
            clean_qualifying(_df(row))

    def test_raises_on_null_constructorId(self):
        row = _make_qualifying_row()
        row["constructorId"] = None
        with pytest.raises(ValueError, match="constructorId"):
            clean_qualifying(_df(row))

    def test_raises_on_missing_raceId_column(self):
        df = _df(_make_qualifying_row()).drop(columns=["raceId"])
        with pytest.raises(ValueError, match="raceId"):
            clean_qualifying(df)

    def test_original_columns_preserved(self):
        df = clean_qualifying(_df(_make_qualifying_row()))
        assert "qualifyId" in df.columns
        assert "number" in df.columns

    def test_does_not_mutate_input(self):
        raw = _df(_make_qualifying_row())
        raw_copy = raw.copy()
        clean_qualifying(raw)
        pd.testing.assert_frame_equal(raw, raw_copy)

    def test_raises_when_some_rows_have_null_raceId(self):
        rows = _df(_make_row(raceId=18), _make_row(raceId=None, driverId=2))
        with pytest.raises(ValueError, match="raceId"):
            clean_results(rows)


# ---------------------------------------------------------------------------
# Original columns preserved
# ---------------------------------------------------------------------------

class TestOriginalColumnsPreserved:
    def test_positionText_unchanged(self):
        df = clean_results(_df(_make_row(positionText="R", position=None, statusId=5)))
        assert df.loc[0, "positionText"] == "R"

    def test_statusId_unchanged(self):
        df = clean_results(_df(_make_row(statusId=5)))
        assert df.loc[0, "statusId"] == 5

    def test_new_columns_appended(self):
        df = clean_results(_df(_make_row()))
        assert "result_status" in df.columns
        assert "finished" in df.columns

    def test_column_count_increases_by_two(self):
        raw = _df(_make_row())
        cleaned = clean_results(raw)
        assert len(cleaned.columns) == len(raw.columns) + 2


# ---------------------------------------------------------------------------
# Multi-row / mixed scenario
# ---------------------------------------------------------------------------

class TestMixedRace:
    def test_mixed_race_outcome_counts(self):
        rows = _df(
            _make_row(driverId=1, positionText="1",  position=1.0,  statusId=1),
            _make_row(driverId=2, positionText="2",  position=2.0,  statusId=1),
            _make_row(driverId=3, positionText="R",  position=None, statusId=5),
            _make_row(driverId=4, positionText="D",  position=None, statusId=2),
            _make_row(driverId=5, positionText="W",  position=None, statusId=54),
        )
        df = clean_results(rows)
        assert df["finished"].sum() == 2
        assert (df["result_status"] == "Finished").sum() == 2
        assert (df["result_status"] == "Retired").sum() == 1
        assert (df["result_status"] == "Disqualified").sum() == 1
        assert (df["result_status"] == "Withdrawn").sum() == 1

    def test_all_non_finishers_have_null_position(self):
        rows = _df(
            _make_row(driverId=1, positionText="R", position=None, statusId=5),
            _make_row(driverId=2, positionText="D", position=None, statusId=2),
            _make_row(driverId=3, positionText="N", position=None, statusId=81),
        )
        df = clean_results(rows)
        assert df["position"].isna().all()
        assert not df["finished"].any()
