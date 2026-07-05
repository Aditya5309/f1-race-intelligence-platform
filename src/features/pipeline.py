"""
src/features/pipeline.py

Feature-engineering pipeline: composes the five feature groups, in order, to
transform data/processed/master_dataset.parquet into
data/processed/features.parquet.

    python -m src.features.pipeline            # build + validate + write
    python -m src.features.pipeline --dry-run  # build + validate, no write

Design notes
------------
- Every transform here is STATELESS: rolling windows, lags, and string
  parsing computed deterministically from history. There is nothing to fit,
  so this is a functional composition + CLI (like src/pipelines/
  build_dataset.py), not a fitted sklearn Pipeline. Fitted preprocessing
  (imputation, scaling — fit on the 2010-2021 training window only, per
  Decision 008) belongs in the Phase 4 model pipeline, where the estimator
  lives. Recorded as Decision 011.
- The output covers the FULL race history, like the master dataset — rolling
  features for 2010+ rows are computed using pre-2010 races as context
  (Decision 010); the Decision-008 year split is applied at training time.
- POST_RACE_OUTCOME_COLUMNS is imported from the integration layer — the
  single source of truth for what is not pre-race-safe — and the disjointness
  of FEATURE_COLUMNS is asserted at import time: the pipeline cannot even be
  imported in a state that leaks a same-race outcome column into the feature
  set (design doc Section 6.1).
- Deferred by design, deliberately absent from FEATURE_COLUMNS:
  `is_home_circuit` (needs a hand-built nationality->country mapping, design
  doc Section 6.3) and sprint enrichment (Section 6.4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.validator import ValidationResult
from src.features.circuit_history import (
    CIRCUIT_HISTORY_FEATURES,
    add_circuit_history_features,
)
from src.features.constructor_form import (
    CONSTRUCTOR_FORM_FEATURES,
    add_constructor_form_features,
)
from src.features.driver_form import DRIVER_FORM_FEATURES, add_driver_form_features
from src.features.qualifying import QUALIFYING_FEATURES, add_qualifying_features
from src.features.standings import (
    STANDINGS_FEATURES,
    add_standings_features,
    load_standings,
)
from src.integration.build_master_dataset import POST_RACE_OUTCOME_COLUMNS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
MASTER_DATASET_PATH = _PROCESSED_DIR / "master_dataset.parquet"
FEATURES_PATH = _PROCESSED_DIR / "features.parquet"

# Kept for joins, grouping, and the train/val/test split — not model features.
ID_COLUMNS: tuple[str, ...] = (
    "raceId", "driverId", "constructorId", "circuitId", "year", "round",
)

TARGET_COLUMN: str = "winner"

# The model-facing feature set, one tuple per group, in execution order.
FEATURE_COLUMNS: tuple[str, ...] = (
    QUALIFYING_FEATURES
    + DRIVER_FORM_FEATURES
    + CONSTRUCTOR_FORM_FEATURES
    + CIRCUIT_HISTORY_FEATURES
    + STANDINGS_FEATURES
)

FEATURES_DATASET_COLUMNS: tuple[str, ...] = (
    ID_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,)
)

# Import-time leakage guard (design doc Section 6.1): no same-race outcome
# column may ever be a feature. Note this also keeps raw `grid` (pit-lane
# sentinel, Section 6.5) and raw q1/q2/q3 strings out of the model's view —
# only their engineered, safe forms are in FEATURE_COLUMNS.
_leaked = set(FEATURE_COLUMNS) & POST_RACE_OUTCOME_COLUMNS
assert not _leaked, f"Post-race outcome column(s) in FEATURE_COLUMNS: {sorted(_leaked)}"
assert len(set(FEATURES_DATASET_COLUMNS)) == len(FEATURES_DATASET_COLUMNS), \
    "Duplicate column name across feature groups"


def build_features(
    master: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the feature matrix from the master dataset.

    Applies the feature groups in order (qualifying -> driver form ->
    constructor form -> circuit history -> lagged standings), then selects
    FEATURES_DATASET_COLUMNS. Returns one row per (raceId, driverId), sorted
    chronologically; row count equals the input's.
    """
    expected_rows = len(master)

    df = add_qualifying_features(master)
    df = add_driver_form_features(df)
    df = add_constructor_form_features(df)
    df = add_circuit_history_features(df)
    df = add_standings_features(df, driver_standings, constructor_standings)

    if len(df) != expected_rows:
        raise ValueError(
            f"Feature pipeline changed row count: {expected_rows:,} -> {len(df):,}."
        )

    df = df.sort_values(
        ["year", "round", "raceId", "driverId"], kind="mergesort"
    ).reset_index(drop=True)
    return df[list(FEATURES_DATASET_COLUMNS)]


def validate_features(df: pd.DataFrame, expected_row_count: int) -> ValidationResult:
    """
    Validate the built feature matrix.

    Errors (pipeline-blocking):
      - Schema matches FEATURES_DATASET_COLUMNS exactly
      - Row count preserved from the master dataset
      - No duplicate (raceId, driverId) pairs
      - Identifier columns and the target are non-null
      - No post-race outcome column present anywhere in the output
    Warnings (non-blocking):
      - High null rates in standings/qualifying features (expected for early
        history and pre-knockout-era qualifying, but worth surfacing)
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in FEATURES_DATASET_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Output is missing expected column(s): {missing}")
    extra = [c for c in df.columns if c not in FEATURES_DATASET_COLUMNS]
    if extra:
        errors.append(f"Output has unexpected extra column(s): {extra}")

    leaked = POST_RACE_OUTCOME_COLUMNS & set(df.columns)
    if leaked:
        errors.append(f"Post-race outcome column(s) present in output: {sorted(leaked)}")

    if len(df) != expected_row_count:
        errors.append(
            f"Row count mismatch: expected {expected_row_count:,}, got {len(df):,}."
        )

    if {"raceId", "driverId"} <= set(df.columns):
        n_dupes = int(df.duplicated(subset=["raceId", "driverId"], keep=False).sum())
        if n_dupes:
            errors.append(f"{n_dupes:,} rows form duplicate (raceId, driverId) pairs.")

    for col in ID_COLUMNS + (TARGET_COLUMN,):
        if col in df.columns:
            n = int(df[col].isnull().sum())
            if n:
                errors.append(f"Column '{col}' has {n:,} null value(s).")

    for col in ("driver_standing_position_prev", "qualifying_gap_to_pole_pct"):
        if col in df.columns and len(df):
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 60:
                warnings.append(f"'{col}' is {null_pct:.1f}% null.")

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=len(df),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build data/processed/features.parquet from the master dataset."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build and validate but do not write the output parquet.",
    )
    args = parser.parse_args(argv)

    if not MASTER_DATASET_PATH.exists():
        print(
            f"ERROR: {MASTER_DATASET_PATH} not found. "
            "Run `python -m src.pipelines.build_dataset` first.",
            file=sys.stderr,
        )
        return 1

    master = pd.read_parquet(MASTER_DATASET_PATH)
    print(f"Loaded master dataset: {len(master):,} rows x {master.shape[1]} cols")

    driver_standings, constructor_standings = load_standings()
    features = build_features(master, driver_standings, constructor_standings)
    print(f"Built feature matrix: {len(features):,} rows x {features.shape[1]} cols")

    result = validate_features(features, expected_row_count=len(master))
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if not result.passed:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run — features.parquet not written.")
        return 0

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    size_kb = FEATURES_PATH.stat().st_size / 1024
    print(f"Wrote {FEATURES_PATH} ({size_kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
