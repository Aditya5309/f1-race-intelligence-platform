"""
src/integration/build_master_dataset.py

Reusable integration logic for joining cleaned interim/raw F1 datasets into a
single master modeling dataset: one row per (raceId, driverId), covering the
full available race history (no year filtering — the Decision-008 train/val/
test split is applied later, at training time, not here).

Scope (Decision 009 / reports/master_dataset_design.md)
--------------------------------------------------------
This module performs INTEGRATION ONLY:
  - straight joins on stable keys (raceId, driverId, constructorId, circuitId)
  - column renaming to avoid collisions across source tables
  - target derivation (winner)
  - schema/integrity validation

It explicitly does NOT do feature engineering:
  - no rolling driver/constructor form
  - no circuit-history aggregates
  - no championship-standings lag (driver_standings.csv / constructor_standings.csv
    are deliberately excluded — lagging to round N-1 is a temporal transform, not
    a plain join, and belongs in the future feature-engineering phase)
  - no parsing of qualifying q1/q2/q3 time strings into seconds

Post-race outcome columns (position, points, laps, statusId, etc.) ARE included
in the output — they're needed to derive the `winner` target and will be needed
by future feature engineering to compute prior-race rolling history. They are
NOT pre-race-safe and must never be selected as a model feature for a row's own
race. See POST_RACE_OUTCOME_COLUMNS below and
reports/master_dataset_design.md Section 6.1.

Entry points
------------
    load_inputs() -> dict[str, pd.DataFrame]
    validate_inputs(inputs) -> ValidationResult
    build_master_dataset(inputs) -> pd.DataFrame
    validate_output(df, expected_row_count) -> ValidationResult
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_csv
from src.data.validator import ValidationResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INTERIM_DIR = _PROJECT_ROOT / "data" / "interim"

# ---------------------------------------------------------------------------
# Column renames — applied per source table to avoid collisions. Several
# Ergast tables share raw column names ("name", "nationality", "position",
# "url") that would silently clobber each other in a naive merge.
# ---------------------------------------------------------------------------

_RACES_COLUMNS: tuple[str, ...] = ("raceId", "year", "round", "circuitId", "name", "date")
_RACES_RENAME: dict[str, str] = {"name": "race_name", "date": "race_date"}

_CIRCUITS_COLUMNS: tuple[str, ...] = (
    "circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt",
)
_CIRCUITS_RENAME: dict[str, str] = {
    "circuitRef": "circuit_ref",
    "name": "circuit_name",
    "location": "circuit_location",
    "country": "circuit_country",
    "lat": "circuit_lat",
    "lng": "circuit_lng",
    "alt": "circuit_alt",
}

_DRIVERS_COLUMNS: tuple[str, ...] = (
    "driverId", "driverRef", "code", "forename", "surname", "dob", "nationality",
)
_DRIVERS_RENAME: dict[str, str] = {
    "driverRef": "driver_ref",
    "code": "driver_code",
    "forename": "driver_forename",
    "surname": "driver_surname",
    "dob": "driver_dob",
    "nationality": "driver_nationality",
}

_CONSTRUCTORS_COLUMNS: tuple[str, ...] = ("constructorId", "constructorRef", "name", "nationality")
_CONSTRUCTORS_RENAME: dict[str, str] = {
    "constructorRef": "constructor_ref",
    "name": "constructor_name",
    "nationality": "constructor_nationality",
}

_QUALIFYING_COLUMNS: tuple[str, ...] = ("raceId", "driverId", "position", "q1", "q2", "q3")
_QUALIFYING_RENAME: dict[str, str] = {"position": "qualifying_position"}

# Columns that describe the OUTCOME of the race being predicted. Kept in the
# master dataset for target derivation and future rolling-history computation,
# but must never be used as a model feature for a row's own race.
POST_RACE_OUTCOME_COLUMNS: frozenset[str] = frozenset({
    "position", "positionText", "positionOrder", "points", "laps",
    "milliseconds", "rank", "fastestLap", "fastestLapTime", "fastestLapSpeed",
    "statusId", "result_status", "finished",
})

# Full expected schema of the master dataset, in output column order.
MASTER_DATASET_COLUMNS: tuple[str, ...] = (
    # Identifiers
    "raceId", "driverId", "constructorId", "circuitId", "year", "round",
    # Race / circuit context
    "race_name", "race_date",
    "circuit_ref", "circuit_name", "circuit_location", "circuit_country",
    "circuit_lat", "circuit_lng", "circuit_alt",
    # Driver dimension
    "driver_ref", "driver_code", "driver_forename", "driver_surname",
    "driver_dob", "driver_nationality",
    # Constructor dimension
    "constructor_ref", "constructor_name", "constructor_nationality",
    # Pre-race grid / qualifying (safe)
    "grid", "qualifying_position", "q1", "q2", "q3",
    # Post-race outcome (raw, reference only — see POST_RACE_OUTCOME_COLUMNS)
    "position", "positionText", "positionOrder", "points", "laps",
    "milliseconds", "rank", "fastestLap", "fastestLapTime", "fastestLapSpeed",
    "statusId", "result_status", "finished",
    # Target
    "winner",
)

assert POST_RACE_OUTCOME_COLUMNS <= set(MASTER_DATASET_COLUMNS)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_inputs(interim_dir: Path = _INTERIM_DIR) -> dict[str, pd.DataFrame]:
    """
    Load every table required to build the master dataset.

    `results` and `qualifying` come from data/interim/ (already cleaned by
    src/data/cleaner.py + src/data/build_interim.py). `races`, `drivers`,
    `constructors`, and `circuits` are loaded directly from data/*.csv — per
    Decision 005 / current_status.md, these dimension tables join cleanly
    as-is and have no dedicated clean_* step.

    Raises
    ------
    FileNotFoundError
        If an interim parquet is missing (run src/data/build_interim.py first).
    """
    results_path = interim_dir / "results.parquet"
    qualifying_path = interim_dir / "qualifying.parquet"

    for path in (results_path, qualifying_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required interim dataset not found: {path}. "
                "Run `python src/data/build_interim.py` first."
            )

    return {
        "results": pd.read_parquet(results_path),
        "qualifying": pd.read_parquet(qualifying_path),
        "races": load_csv("races.csv"),
        "drivers": load_csv("drivers.csv"),
        "constructors": load_csv("constructors.csv"),
        "circuits": load_csv("circuits.csv"),
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _check_primary_key_unique(
    df: pd.DataFrame, key: str, table_name: str, errors: list[str],
) -> None:
    if key not in df.columns:
        errors.append(f"'{table_name}' is missing its primary key column '{key}'.")
        return
    n_dupes = int(df[key].duplicated().sum())
    if n_dupes:
        errors.append(
            f"'{table_name}' has {n_dupes:,} duplicate value(s) in primary key "
            f"'{key}'. A dimension table must have a unique key or joins will fan out."
        )


def _check_no_duplicate_pair(
    df: pd.DataFrame, keys: list[str], table_name: str, errors: list[str],
) -> None:
    dupes = df.duplicated(subset=keys, keep=False)
    n = int(dupes.sum())
    if n:
        errors.append(
            f"'{table_name}' has {n:,} rows forming duplicate {tuple(keys)} pairs."
        )


def _check_non_null(
    df: pd.DataFrame, columns: list[str], table_name: str, errors: list[str],
) -> None:
    for col in columns:
        if col not in df.columns:
            errors.append(f"'{table_name}' is missing expected column '{col}'.")
            continue
        n = int(df[col].isnull().sum())
        if n:
            errors.append(f"'{table_name}.{col}' has {n:,} null value(s).")


def validate_inputs(inputs: dict[str, pd.DataFrame]) -> ValidationResult:
    """
    Validate every source table before joining.

    Checks
    ------
    - Dimension tables (races, drivers, constructors, circuits) have a unique
      primary key — a duplicate key would silently fan out every join.
    - results and qualifying have no duplicate (raceId, driverId) pairs.
    - Join-key columns are non-null everywhere they're required.
    """
    errors: list[str] = []
    warnings: list[str] = []

    _check_primary_key_unique(inputs["races"], "raceId", "races", errors)
    _check_primary_key_unique(inputs["drivers"], "driverId", "drivers", errors)
    _check_primary_key_unique(inputs["constructors"], "constructorId", "constructors", errors)
    _check_primary_key_unique(inputs["circuits"], "circuitId", "circuits", errors)

    _check_no_duplicate_pair(inputs["results"], ["raceId", "driverId"], "results", errors)
    _check_no_duplicate_pair(inputs["qualifying"], ["raceId", "driverId"], "qualifying", errors)

    _check_non_null(inputs["results"], ["raceId", "driverId", "constructorId"], "results", errors)
    _check_non_null(inputs["races"], ["raceId", "circuitId"], "races", errors)
    _check_non_null(inputs["qualifying"], ["raceId", "driverId"], "qualifying", errors)

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=len(inputs["results"]),
    )


# ---------------------------------------------------------------------------
# Join logic
# ---------------------------------------------------------------------------

def _join_and_check(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    validate: str,
    step_name: str,
    expected_row_count: int,
) -> pd.DataFrame:
    """Left-join right onto left, then assert the row count did not change."""
    merged = left.merge(right, on=on, how="left", validate=validate)
    if len(merged) != expected_row_count:
        raise ValueError(
            f"Join step '{step_name}' changed row count: "
            f"{expected_row_count:,} -> {len(merged):,}. "
            f"This means a duplicate key exists in the '{step_name}' table "
            "that validate_inputs() should have caught — investigate before proceeding."
        )
    return merged


def build_master_dataset(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join the interim/raw tables into the master modeling dataset.

    Grain: one row per (raceId, driverId), identical to `results`. Every join
    is a left join FROM results, keyed on stable IDs, with row-count parity
    enforced after each step (see _join_and_check) — a silent fan-out would
    corrupt the (raceId, driverId) grain that every downstream feature and the
    target label depend on.

    Does not filter by year — the full race history is retained so that a
    future feature-engineering step can compute rolling/circuit-history
    features using pre-2010 races as context, even though only 2010+ rows are
    used for training (Decision 008).
    """
    results = inputs["results"]
    base_row_count = len(results)

    df = results.copy()

    races = inputs["races"][list(_RACES_COLUMNS)].rename(columns=_RACES_RENAME)
    df = _join_and_check(df, races, on="raceId", validate="many_to_one",
                         step_name="races", expected_row_count=base_row_count)

    circuits = inputs["circuits"][list(_CIRCUITS_COLUMNS)].rename(columns=_CIRCUITS_RENAME)
    df = _join_and_check(df, circuits, on="circuitId", validate="many_to_one",
                         step_name="circuits", expected_row_count=base_row_count)

    drivers = inputs["drivers"][list(_DRIVERS_COLUMNS)].rename(columns=_DRIVERS_RENAME)
    df = _join_and_check(df, drivers, on="driverId", validate="many_to_one",
                         step_name="drivers", expected_row_count=base_row_count)

    constructors = inputs["constructors"][list(_CONSTRUCTORS_COLUMNS)].rename(columns=_CONSTRUCTORS_RENAME)
    df = _join_and_check(df, constructors, on="constructorId", validate="many_to_one",
                         step_name="constructors", expected_row_count=base_row_count)

    qualifying = inputs["qualifying"][list(_QUALIFYING_COLUMNS)].rename(columns=_QUALIFYING_RENAME)
    df = _join_and_check(df, qualifying, on=["raceId", "driverId"], validate="one_to_one",
                         step_name="qualifying", expected_row_count=base_row_count)

    # Target: winner. Uses positionOrder (Ergast's canonical finishing-order
    # column), not position/positionText, per reports/master_dataset_design.md
    # Section 5.2 — avoids the string/nullable-int ambiguity clean_results()
    # already resolved once.
    df["winner"] = (df["positionOrder"] == 1).astype(int)

    return df[list(MASTER_DATASET_COLUMNS)]


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(df: pd.DataFrame, expected_row_count: int) -> ValidationResult:
    """
    Validate the joined master dataset.

    Checks
    ------
    Errors (pipeline-blocking):
      - Schema matches MASTER_DATASET_COLUMNS exactly (no missing/extra columns)
      - Row count matches the input results row count exactly
      - No duplicate (raceId, driverId) pairs
      - Identifier columns (raceId, driverId, constructorId, circuitId, year,
        round) are non-null
      - Referential integrity: every row's dimension columns resolved (a
        left join that didn't match would leave e.g. driver_ref null)

    Warnings (non-blocking):
      - A race with a winner count != 1 (should be exactly 1 for 2010+ data;
        older/incomplete historical races may legitimately vary)
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in MASTER_DATASET_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Output is missing expected column(s): {missing_cols}")
    extra_cols = [c for c in df.columns if c not in MASTER_DATASET_COLUMNS]
    if extra_cols:
        errors.append(f"Output has unexpected extra column(s): {extra_cols}")

    if len(df) != expected_row_count:
        errors.append(
            f"Row count mismatch: expected {expected_row_count:,}, got {len(df):,}."
        )

    if {"raceId", "driverId"} <= set(df.columns):
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
                f"{n:,} rows form duplicate (raceId, driverId) pairs (sample: {sample})."
            )

    for col in ("raceId", "driverId", "constructorId", "circuitId", "year", "round"):
        if col in df.columns:
            n = int(df[col].isnull().sum())
            if n:
                errors.append(f"Identifier column '{col}' has {n:,} null value(s).")

    # Referential integrity: a left join that failed to match leaves the
    # dimension's non-key columns null for that row.
    _ref_checks = [
        ("driverId", "driver_ref", "drivers.csv"),
        ("constructorId", "constructor_ref", "constructors.csv"),
        ("circuitId", "circuit_ref", "circuits.csv"),
    ]
    for fk_col, resolved_col, source in _ref_checks:
        if {fk_col, resolved_col} <= set(df.columns):
            unresolved = df[fk_col].notna() & df[resolved_col].isna()
            n = int(unresolved.sum())
            if n:
                errors.append(
                    f"{n:,} row(s) have a '{fk_col}' with no matching row in {source} "
                    f"('{resolved_col}' is null)."
                )

    if {"raceId", "winner"} <= set(df.columns):
        winners_per_race = df.groupby("raceId")["winner"].sum()
        bad_races = winners_per_race[winners_per_race != 1]
        if not bad_races.empty:
            warnings.append(
                f"{len(bad_races):,} race(s) do not have exactly one winner "
                f"(sample raceIds: {bad_races.index[:5].tolist()})."
            )

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
    )
