"""
src/data/validator.py

Post-cleaning validation for the Ergast race results DataFrame.
Run this on the OUTPUT of clean_results(), not on raw loaded data.

Entry point
-----------
    validate_results(df, raise_on_error=True) -> ValidationResult

Design notes
------------
Two severity levels:

  Error   — a constraint that, if violated, makes the data unsafe to use
            downstream (e.g. duplicate rows would double-count a race entry
            in rolling feature calculations, corrupting every model trained on it).
            All errors are collected before raising so the caller sees the full
            picture in one exception, not one failure at a time.

  Warning — an anomaly worth knowing about but not pipeline-blocking
            (e.g. a non-finisher who retained a provisional position after DSQ).
            Warnings are returned in ValidationResult and printed; they never
            cause an exception.

The function always returns a ValidationResult. When raise_on_error=True (default)
and errors exist, it raises ValueError AFTER returning would have been possible —
callers that need the result regardless should pass raise_on_error=False and
inspect result.passed themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_RESULT_STATUSES: frozenset[str] = frozenset({
    "Finished",
    "Retired",
    "Disqualified",
    "Did Not Start",
    "Withdrawn",
    "Other",
})

# Columns that must be present for the validator to run its full suite.
# Subset of the cleaner output schema.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "raceId",
    "driverId",
    "constructorId",
    "points",
    "position",
    "result_status",
    "finished",
)

# Reasonable upper bound on grid size across all F1 eras.
_MAX_GRID_POSITION: int = 34

# Columns that must be present for validate_qualifying() to run its full suite.
# Subset of the cleaner output schema for clean_qualifying().
_QUALIFYING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "raceId",
    "driverId",
    "constructorId",
    "position",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Holds the outcome of a validate_results() call.

    Attributes
    ----------
    passed : bool
        True only when there are zero errors (warnings are allowed).
    errors : list[str]
        Critical constraint violations. Any error means the DataFrame is
        unsafe to use downstream.
    warnings : list[str]
        Non-blocking anomalies. Logged for awareness; do not fail pipelines.
    row_count : int
        Number of rows in the validated DataFrame.
    """

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'} "
            f"({self.row_count:,} rows)",
        ]
        if self.errors:
            lines.append(f"  {len(self.errors)} error(s):")
            for e in self.errors:
                lines.append(f"    [ERROR] {e}")
        if self.warnings:
            lines.append(f"  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"    [WARN]  {w}")
        if self.passed and not self.warnings:
            lines.append("  All checks passed.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_required_columns(
    df: pd.DataFrame,
    errors: list[str],
) -> bool:
    """Return True if all required columns are present; append errors otherwise."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(
            f"Missing required columns: {missing}. "
            "Run clean_results() before validate_results()."
        )
        return False
    return True


def _check_non_null_ids(df: pd.DataFrame, errors: list[str]) -> None:
    for col in ("raceId", "driverId", "constructorId"):
        n = int(df[col].isnull().sum())
        if n:
            errors.append(
                f"'{col}' has {n:,} null value(s). "
                "Every result row must reference a valid race, driver, and constructor."
            )


def _check_no_duplicate_entries(df: pd.DataFrame, errors: list[str]) -> None:
    """Each driver must appear at most once per race."""
    dupes = df.duplicated(subset=["raceId", "driverId"], keep=False)
    n = int(dupes.sum())
    if n:
        sample = (
            df.loc[dupes, ["raceId", "driverId"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        errors.append(
            f"{n:,} rows form duplicate (raceId, driverId) pairs "
            f"(sample: {sample}). "
            "Duplicates would corrupt rolling feature calculations."
        )


def _check_non_negative_points(df: pd.DataFrame, errors: list[str]) -> None:
    if "points" not in df.columns:
        return
    neg = int((df["points"] < 0).sum())
    if neg:
        errors.append(
            f"'points' contains {neg:,} negative value(s). "
            "Points are always >= 0 in F1."
        )


def _check_allowed_result_statuses(df: pd.DataFrame, errors: list[str]) -> None:
    if "result_status" not in df.columns:
        return
    unknown = set(df["result_status"].dropna().unique()) - ALLOWED_RESULT_STATUSES
    if unknown:
        errors.append(
            f"'result_status' contains unexpected value(s): {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_RESULT_STATUSES)}."
        )


def _check_finished_position_consistency(
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    Enforce the contract defined in cleaner.py:
      finished=True  => position must be non-null  (error if violated)
      finished=False => position is usually null   (warning if non-null — valid for post-race DSQ)
    """
    if "finished" not in df.columns or "position" not in df.columns:
        return

    finished_no_pos = df["finished"] & df["position"].isna()
    n_err = int(finished_no_pos.sum())
    if n_err:
        errors.append(
            f"{n_err:,} row(s) have finished=True but position is null. "
            "A classified finisher must always have a finishing position."
        )

    not_finished_with_pos = ~df["finished"] & df["position"].notna()
    n_warn = int(not_finished_with_pos.sum())
    if n_warn:
        warnings.append(
            f"{n_warn:,} row(s) have finished=False but position is non-null. "
            "This can occur for drivers disqualified after crossing the finish line "
            "(provisional position retained). Verify these rows are intentional."
        )


def _check_position_range(
    df: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Warn if any classified finishing position is outside the plausible grid range."""
    if "position" not in df.columns:
        return
    pos = df["position"].dropna()
    out_of_range = pos[(pos < 1) | (pos > _MAX_GRID_POSITION)]
    if not out_of_range.empty:
        warnings.append(
            f"{len(out_of_range):,} finished row(s) have position outside "
            f"[1, {_MAX_GRID_POSITION}]: {sorted(out_of_range.unique().tolist())}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_results(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> ValidationResult:
    """
    Validate a cleaned Ergast race results DataFrame.

    Expects the output of ``clean_results()``. Checks are run in full even
    when early checks fail, so the caller receives a complete error list.

    Checks performed
    ----------------
    Errors (pipeline-blocking):
      1. All required columns present
      2. raceId, driverId, constructorId are non-null
      3. No duplicate (raceId, driverId) pairs
      4. points >= 0
      5. result_status only contains allowed values
      6. finished=True implies position is non-null

    Warnings (non-blocking):
      7. finished=False rows with non-null position (valid for post-race DSQ)
      8. position outside plausible range [1, 34]

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ``clean_results()``.
    raise_on_error : bool, default True
        If True, raises ValueError listing all errors when any are found.
        If False, returns the ValidationResult for the caller to inspect.

    Returns
    -------
    ValidationResult
        Always returned. Check ``.passed`` and ``.errors`` programmatically
        when raise_on_error=False.

    Raises
    ------
    ValueError
        When raise_on_error=True and one or more errors are found.
        The message lists every error collected.
    """
    errors: list[str] = []
    warnings: list[str] = []

    columns_ok = _check_required_columns(df, errors)

    if columns_ok:
        _check_non_null_ids(df, errors)
        _check_no_duplicate_entries(df, errors)
        _check_non_negative_points(df, errors)
        _check_allowed_result_statuses(df, errors)
        _check_finished_position_consistency(df, errors, warnings)
        _check_position_range(df, warnings)

    result = ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
    )

    if raise_on_error and not result.passed:
        error_block = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"validate_results() found {len(errors)} error(s):\n{error_block}"
        )

    return result


# ---------------------------------------------------------------------------
# Qualifying validation
# ---------------------------------------------------------------------------

def _check_qualifying_required_columns(df: pd.DataFrame, errors: list[str]) -> bool:
    missing = [c for c in _QUALIFYING_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(
            f"Missing required columns: {missing}. "
            "Run clean_qualifying() before validate_qualifying()."
        )
        return False
    return True


def _check_qualifying_no_duplicate_entries(df: pd.DataFrame, errors: list[str]) -> None:
    """Each driver must appear at most once per race's qualifying session."""
    dupes = df.duplicated(subset=["raceId", "driverId"], keep=False)
    n = int(dupes.sum())
    if n:
        sample = (
            df.loc[dupes, ["raceId", "driverId"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        errors.append(
            f"{n:,} rows form duplicate (raceId, driverId) pairs "
            f"(sample: {sample}). "
            "Duplicates would corrupt the join onto results.parquet."
        )


def validate_qualifying(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> ValidationResult:
    """
    Validate a cleaned Ergast qualifying DataFrame.

    Expects the output of ``clean_qualifying()``.

    Checks performed
    ----------------
    Errors (pipeline-blocking):
      1. All required columns present
      2. raceId, driverId, constructorId are non-null
      3. No duplicate (raceId, driverId) pairs

    Warnings (non-blocking):
      4. position outside plausible range [1, 34]

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ``clean_qualifying()``.
    raise_on_error : bool, default True
        If True, raises ValueError listing all errors when any are found.

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    columns_ok = _check_qualifying_required_columns(df, errors)

    if columns_ok:
        for col in ("raceId", "driverId", "constructorId"):
            n = int(df[col].isnull().sum())
            if n:
                errors.append(
                    f"'{col}' has {n:,} null value(s). "
                    "Every qualifying row must reference a valid race, driver, and constructor."
                )
        _check_qualifying_no_duplicate_entries(df, errors)
        _check_position_range(df, warnings)

    result = ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
    )

    if raise_on_error and not result.passed:
        error_block = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"validate_qualifying() found {len(errors)} error(s):\n{error_block}"
        )

    return result
