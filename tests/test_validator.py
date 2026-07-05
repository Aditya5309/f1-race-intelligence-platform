"""
Tests for src/data/validator.py

Covers:
  - Valid cleaned data passes all checks
  - Each individual error check fires correctly
  - Warnings fire without blocking the pipeline
  - raise_on_error=False returns result instead of raising
  - ValidationResult.passed and summary() reflect state correctly
  - Column presence guard: checks skip gracefully when columns are absent
"""

import pandas as pd
import pytest

from src.data.validator import (
    ALLOWED_RESULT_STATUSES,
    ValidationResult,
    validate_qualifying,
    validate_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**overrides) -> dict:
    """Return a minimal *cleaned* result row (post clean_results output)."""
    base = {
        "resultId":      1,
        "raceId":        18,
        "driverId":      1,
        "constructorId": 1,
        "grid":          1,
        "position":      pd.NA,
        "positionText":  "1",
        "points":        25.0,
        "laps":          57,
        "statusId":      1,
        "result_status": "Finished",
        "finished":      True,
    }
    # A finished driver always has a position
    if base["result_status"] == "Finished":
        base["position"] = 1
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _valid_df(n: int = 3) -> pd.DataFrame:
    """Build a small multi-row DataFrame where every row is valid."""
    rows = [
        _make_row(driverId=i, position=i, result_status="Finished", finished=True)
        for i in range(1, n + 1)
    ]
    return _df(*rows)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestValidData:
    def test_clean_data_passes(self):
        result = validate_results(_valid_df())
        assert result.passed is True
        assert result.errors == []

    def test_returns_validation_result(self):
        result = validate_results(_valid_df(), raise_on_error=False)
        assert isinstance(result, ValidationResult)

    def test_row_count_is_set(self):
        df = _valid_df(5)
        result = validate_results(df)
        assert result.row_count == 5

    def test_mixed_statuses_pass(self):
        rows = _df(
            _make_row(driverId=1, position=1,    result_status="Finished",     finished=True),
            _make_row(driverId=2, position=pd.NA, result_status="Retired",      finished=False),
            _make_row(driverId=3, position=pd.NA, result_status="Disqualified", finished=False),
            _make_row(driverId=4, position=pd.NA, result_status="Did Not Start",finished=False),
            _make_row(driverId=5, position=pd.NA, result_status="Withdrawn",    finished=False),
        )
        result = validate_results(rows)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_missing_result_status_errors(self):
        df = _valid_df().drop(columns=["result_status"])
        result = validate_results(df, raise_on_error=False)
        assert not result.passed
        assert any("result_status" in e for e in result.errors)

    def test_missing_finished_errors(self):
        df = _valid_df().drop(columns=["finished"])
        result = validate_results(df, raise_on_error=False)
        assert not result.passed

    def test_missing_raceId_errors(self):
        df = _valid_df().drop(columns=["raceId"])
        result = validate_results(df, raise_on_error=False)
        assert not result.passed
        assert any("raceId" in e for e in result.errors)

    def test_missing_column_skips_downstream_checks(self):
        """When columns are missing, downstream checks should not crash."""
        df = _valid_df().drop(columns=["result_status", "finished"])
        result = validate_results(df, raise_on_error=False)
        assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Non-null ID checks
# ---------------------------------------------------------------------------

class TestNonNullIds:
    @pytest.mark.parametrize("col", ["raceId", "driverId", "constructorId"])
    def test_null_id_raises(self, col):
        row = _make_row()
        row[col] = None
        with pytest.raises(ValueError, match=col):
            validate_results(_df(row))

    @pytest.mark.parametrize("col", ["raceId", "driverId", "constructorId"])
    def test_null_id_captured_without_raise(self, col):
        row = _make_row()
        row[col] = None
        result = validate_results(_df(row), raise_on_error=False)
        assert not result.passed
        assert any(col in e for e in result.errors)

    def test_partial_null_raceId_fails(self):
        rows = _df(
            _make_row(driverId=1),
            _make_row(driverId=2, raceId=None),
        )
        result = validate_results(rows, raise_on_error=False)
        assert not result.passed


# ---------------------------------------------------------------------------
# Duplicate (raceId, driverId) check
# ---------------------------------------------------------------------------

class TestDuplicateEntries:
    def test_duplicate_pair_raises(self):
        rows = _df(_make_row(), _make_row())  # same raceId=18, driverId=1
        with pytest.raises(ValueError, match="duplicate"):
            validate_results(rows)

    def test_duplicate_pair_captured(self):
        rows = _df(_make_row(), _make_row())
        result = validate_results(rows, raise_on_error=False)
        assert not result.passed
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_different_drivers_same_race_ok(self):
        rows = _df(_make_row(driverId=1, position=1), _make_row(driverId=2, position=2))
        result = validate_results(rows)
        assert result.passed

    def test_same_driver_different_races_ok(self):
        rows = _df(_make_row(raceId=1), _make_row(raceId=2))
        result = validate_results(rows)
        assert result.passed


# ---------------------------------------------------------------------------
# Non-negative points check
# ---------------------------------------------------------------------------

class TestNonNegativePoints:
    def test_negative_points_raises(self):
        row = _make_row(points=-1.0)
        with pytest.raises(ValueError, match="points"):
            validate_results(_df(row))

    def test_negative_points_captured(self):
        row = _make_row(points=-5.0)
        result = validate_results(_df(row), raise_on_error=False)
        assert not result.passed
        assert any("points" in e for e in result.errors)

    def test_zero_points_ok(self):
        row = _make_row(points=0.0)
        result = validate_results(_df(row))
        assert result.passed

    def test_max_points_ok(self):
        row = _make_row(points=26.0)  # 25 + fastest lap bonus
        result = validate_results(_df(row))
        assert result.passed


# ---------------------------------------------------------------------------
# result_status allowed values
# ---------------------------------------------------------------------------

class TestAllowedResultStatuses:
    def test_unknown_status_raises(self):
        row = _make_row(result_status="Teleported", finished=False, position=pd.NA)
        with pytest.raises(ValueError, match="result_status"):
            validate_results(_df(row))

    def test_unknown_status_captured(self):
        row = _make_row(result_status="Teleported", finished=False, position=pd.NA)
        result = validate_results(_df(row), raise_on_error=False)
        assert not result.passed
        assert any("result_status" in e for e in result.errors)

    @pytest.mark.parametrize("status", sorted(ALLOWED_RESULT_STATUSES))
    def test_each_allowed_status_passes(self, status):
        finished = status == "Finished"
        position = 1 if finished else pd.NA
        row = _make_row(result_status=status, finished=finished, position=position)
        result = validate_results(_df(row))
        assert result.passed


# ---------------------------------------------------------------------------
# finished / position consistency
# ---------------------------------------------------------------------------

class TestFinishedPositionConsistency:
    def test_finished_true_null_position_raises(self):
        row = _make_row(result_status="Finished", finished=True, position=pd.NA)
        with pytest.raises(ValueError, match="finished=True"):
            validate_results(_df(row))

    def test_finished_true_null_position_captured(self):
        row = _make_row(result_status="Finished", finished=True, position=pd.NA)
        result = validate_results(_df(row), raise_on_error=False)
        assert not result.passed
        assert any("finished=True" in e for e in result.errors)

    def test_finished_false_non_null_position_is_warning(self):
        """Post-race DSQ: driver has a provisional position but finished=False."""
        row = _make_row(result_status="Disqualified", finished=False, position=3)
        result = validate_results(_df(row))
        assert result.passed  # warnings do not fail
        assert any("finished=False" in w for w in result.warnings)

    def test_finished_false_null_position_no_warning(self):
        row = _make_row(result_status="Retired", finished=False, position=pd.NA)
        result = validate_results(_df(row))
        assert result.passed
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Position range warnings
# ---------------------------------------------------------------------------

class TestPositionRange:
    def test_position_zero_triggers_warning(self):
        row = _make_row(result_status="Finished", finished=True, position=0)
        result = validate_results(_df(row))
        assert result.passed  # warning only
        assert any("position" in w for w in result.warnings)

    def test_position_35_triggers_warning(self):
        row = _make_row(result_status="Finished", finished=True, position=35)
        result = validate_results(_df(row))
        assert result.passed
        assert result.warnings != []

    def test_position_26_no_warning(self):
        row = _make_row(result_status="Finished", finished=True, position=26)
        result = validate_results(_df(row))
        assert result.passed
        assert result.warnings == []


# ---------------------------------------------------------------------------
# raise_on_error=False behaviour
# ---------------------------------------------------------------------------

class TestRaiseOnErrorFalse:
    def test_does_not_raise_on_duplicate(self):
        rows = _df(_make_row(), _make_row())
        result = validate_results(rows, raise_on_error=False)
        assert isinstance(result, ValidationResult)
        assert not result.passed

    def test_does_not_raise_on_null_id(self):
        row = _make_row(raceId=None)
        result = validate_results(_df(row), raise_on_error=False)
        assert not result.passed

    def test_all_errors_collected_before_return(self):
        """Multiple errors should all appear in result.errors, not just the first."""
        row = _make_row(raceId=None, points=-1.0)
        result = validate_results(_df(row), raise_on_error=False)
        assert len(result.errors) >= 2


# ---------------------------------------------------------------------------
# ValidationResult.summary()
# ---------------------------------------------------------------------------

class TestValidationResultSummary:
    def test_summary_contains_passed(self):
        result = validate_results(_valid_df())
        assert "PASSED" in result.summary()

    def test_summary_contains_failed_on_error(self):
        row = _make_row(raceId=None)
        result = validate_results(_df(row), raise_on_error=False)
        assert "FAILED" in result.summary()

    def test_str_equals_summary(self):
        result = validate_results(_valid_df())
        assert str(result) == result.summary()

    def test_summary_lists_error_text(self):
        rows = _df(_make_row(), _make_row())  # duplicate
        result = validate_results(rows, raise_on_error=False)
        assert "ERROR" in result.summary()

    def test_summary_lists_warning_text(self):
        row = _make_row(result_status="Disqualified", finished=False, position=3)
        result = validate_results(_df(row))
        assert "WARN" in result.summary()


# ---------------------------------------------------------------------------
# validate_qualifying()
# ---------------------------------------------------------------------------

def _make_qualifying_row(**overrides) -> dict:
    """Return a minimal *cleaned* qualifying row (post clean_qualifying output)."""
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


def _valid_qualifying_df(n: int = 3) -> pd.DataFrame:
    rows = [
        _make_qualifying_row(driverId=i, position=i)
        for i in range(1, n + 1)
    ]
    return _df(*rows)


class TestValidateQualifying:
    def test_valid_data_passes(self):
        result = validate_qualifying(_valid_qualifying_df())
        assert result.passed
        assert result.errors == []

    def test_raises_on_null_raceId(self):
        row = _make_qualifying_row(raceId=None)
        with pytest.raises(ValueError, match="raceId"):
            validate_qualifying(_df(row))

    def test_raises_on_null_driverId(self):
        row = _make_qualifying_row(driverId=None)
        with pytest.raises(ValueError, match="driverId"):
            validate_qualifying(_df(row))

    def test_raises_on_duplicate_race_driver_pair(self):
        rows = _df(_make_qualifying_row(), _make_qualifying_row())
        with pytest.raises(ValueError, match="duplicate"):
            validate_qualifying(rows)

    def test_raises_on_missing_required_column(self):
        df = _valid_qualifying_df().drop(columns=["position"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_qualifying(df)

    def test_warns_on_position_out_of_range(self):
        row = _make_qualifying_row(position=99)
        result = validate_qualifying(_df(row))
        assert result.passed
        assert any("position" in w for w in result.warnings)

    def test_raise_on_error_false_returns_result(self):
        row = _make_qualifying_row(raceId=None)
        result = validate_qualifying(_df(row), raise_on_error=False)
        assert not result.passed
        assert len(result.errors) >= 1
