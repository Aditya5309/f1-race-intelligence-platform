"""
src/data/cleaner.py

Responsible for DataFrame-level transformations on raw Ergast race results.
Does NOT load files, call APIs, discover paths, or perform feature engineering.

Entry point
-----------
    clean_results(df) -> pd.DataFrame

Design notes
------------
Non-finishers (DNF, DNS, DSQ, Withdrawn) are intentionally retained:
  - They appear in qualifying data and carry feature-relevant information
    (constructor, grid position, driver experience) needed by downstream
    feature engineering.
  - Dropping them would silently bias the training set toward finishers only,
    distorting any model that learns from grid-position distributions.

`position` remains nullable (Int64):
  - A null position is semantically meaningful: the driver did not receive a
    classified finishing position, which is distinct from 0 or any sentinel.
  - Use `finished` (bool) or `result_status` (str) for outcome filtering.
  - Do not test `position.isna()` for control flow; use `finished` instead.

`finished` vs `position`:
  - `finished=True`  => `position` is always non-null.
  - `finished=False` => `position` is usually null, but a driver disqualified
    *after* taking the flag may retain a provisional numeric position.
    Use `finished` for the intent; use `position` for the numerical rank.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Single-character positionText codes used by Ergast for non-finishers.
_POSITION_TEXT_MAP: dict[str, str] = {
    "R": "Retired",
    "D": "Disqualified",
    "E": "Disqualified",   # Excluded post-race (same outcome as DSQ)
    "N": "Did Not Start",
    "F": "Did Not Start",  # Failed 107% qualifying rule
    "W": "Withdrawn",
}

# statusId sets used as fallback when positionText is null.
# Source: status.csv from the Ergast dataset.
_DISQUALIFIED_STATUS_IDS: frozenset[int] = frozenset({
    2,   # Disqualified
    92,  # Underweight
    96,  # Excluded
})

_DID_NOT_START_STATUS_IDS: frozenset[int] = frozenset({
    54,  # Withdrew
    77,  # 107% Rule
    81,  # Did not qualify
    97,  # Did not prequalify
})

# statusId=1 is "Finished"; lapped cars (+N Laps, statusId 11-20+) are
# already caught by the numeric positionText check, so this set is small.
_FINISHED_STATUS_IDS: frozenset[int] = frozenset({1})

# Columns that must never be null — each result row must reference a
# valid race, driver, and constructor or the join graph breaks downstream.
_REQUIRED_COLUMNS: tuple[str, ...] = ("raceId", "driverId", "constructorId")

# Target dtypes: Int64 (capital I) is pandas nullable integer.
_DTYPE_MAP: dict[str, str] = {
    "raceId":        "Int64",
    "driverId":      "Int64",
    "constructorId": "Int64",
    "grid":          "Int64",
    "laps":          "Int64",
    "position":      "Int64",   # nullable — non-finishers have no position
    "points":        "float64",
    "rank":          "Int64",   # nullable — only set for classified finishers
    "milliseconds":  "Int64",   # nullable — only set for classified finishers
}

# Same required-key contract as results.csv: every qualifying row must
# reference a valid race, driver, and constructor.
_QUALIFYING_REQUIRED_COLUMNS: tuple[str, ...] = ("raceId", "driverId", "constructorId")

# qualifying.csv dtypes. q1/q2/q3 are intentionally left as raw "M:SS.sss"
# strings (or NaN) — parsing them into seconds is a feature-engineering
# transform (reports/master_dataset_design.md Section 5.3), not a cleaning
# concern, so it does not happen here.
_QUALIFYING_DTYPE_MAP: dict[str, str] = {
    "raceId":        "Int64",
    "driverId":      "Int64",
    "constructorId": "Int64",
    "number":        "Int64",
    "position":      "Int64",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, required_columns: tuple[str, ...] = _REQUIRED_COLUMNS,
              source_hint: str = "results.csv") -> None:
    """Raise ValueError if required columns are missing or contain nulls."""
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing from the DataFrame. "
                f"Pass a DataFrame loaded from {source_hint}."
            )
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            raise ValueError(
                f"Required column '{col}' contains {null_count} null value(s). "
                "Every row must reference a valid race, driver, and constructor."
            )


def _cast_dtypes(df: pd.DataFrame, dtype_map: dict[str, str] = _DTYPE_MAP) -> pd.DataFrame:
    """Return a copy of df with columns cast to canonical dtypes."""
    df = df.copy()
    for col, dtype in dtype_map.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot cast column '{col}' to {dtype}: {exc}"
            ) from exc
    return df


def _derive_result_status(df: pd.DataFrame) -> pd.Series:
    """
    Return a string Series with one status category per row.

    Resolution priority
    -------------------
    1. positionText is a digit string              → "Finished"
    2. positionText is a known non-finish code     → mapped category
    3. positionText is null, statusId is present   → statusId lookup
    4. Everything else                             → "Other"
    """
    status = pd.Series("Other", index=df.index, dtype="string")

    has_pos_text = "positionText" in df.columns
    has_status_id = "statusId" in df.columns

    if has_pos_text:
        pos = df["positionText"].astype("string")

        # Classified finishers: positionText is a pure integer string
        is_numeric = pos.str.match(r"^\d+$", na=False)
        status[is_numeric] = "Finished"

        # Non-finish single-character codes
        for code, label in _POSITION_TEXT_MAP.items():
            status[pos == code] = label

    if has_status_id:
        sid = df["statusId"].astype("Int64")
        unresolved = status == "Other"

        status[unresolved & sid.isin(_FINISHED_STATUS_IDS)]      = "Finished"
        status[unresolved & sid.isin(_DISQUALIFIED_STATUS_IDS)]  = "Disqualified"
        status[unresolved & sid.isin(_DID_NOT_START_STATUS_IDS)] = "Did Not Start"

        # Any remaining row with a known statusId is a mechanical or
        # accident retirement — the catch-all for ~130 failure modes.
        all_special = (
            _FINISHED_STATUS_IDS
            | _DISQUALIFIED_STATUS_IDS
            | _DID_NOT_START_STATUS_IDS
        )
        status[unresolved & sid.notna() & ~sid.isin(all_special)] = "Retired"

    return status


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw Ergast race results DataFrame.

    Steps
    -----
    1. Validate that raceId, driverId, constructorId are present and non-null.
    2. Cast columns to canonical dtypes (Int64 nullable integers, float64 points).
    3. Add ``result_status`` (string): one of
       "Finished" | "Retired" | "Disqualified" | "Did Not Start" | "Withdrawn" | "Other"
    4. Add ``finished`` (bool): True when the driver holds a classified position.

    Original columns are preserved unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by ``loader.load_csv("results.csv")``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with all original columns plus ``result_status``
        and ``finished`` appended.

    Raises
    ------
    ValueError
        If a required column is missing or contains null values.
    """
    _validate(df)
    df = _cast_dtypes(df)
    df["result_status"] = _derive_result_status(df)
    df["finished"] = df["result_status"] == "Finished"
    return df


def clean_qualifying(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw Ergast qualifying DataFrame.

    Steps
    -----
    1. Validate that raceId, driverId, constructorId are present and non-null.
    2. Cast columns to canonical dtypes (Int64 nullable integers).

    ``q1``, ``q2``, ``q3`` are intentionally left as raw "M:SS.sss" strings
    (or NaN for a session a driver didn't reach). Parsing them into seconds
    and deriving a gap-to-pole percentage are feature-engineering transforms
    (reports/master_dataset_design.md Section 5.3) — out of scope here.

    A null ``q3`` is not missing data: only the top 10 qualifiers reach Q3.
    Downstream consumers must treat it as informative, not impute it.

    Original columns are preserved unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by ``loader.load_csv("qualifying.csv")``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with canonical dtypes applied.

    Raises
    ------
    ValueError
        If a required column is missing or contains null values.
    """
    _validate(df, required_columns=_QUALIFYING_REQUIRED_COLUMNS, source_hint="qualifying.csv")
    df = _cast_dtypes(df, dtype_map=_QUALIFYING_DTYPE_MAP)
    return df
