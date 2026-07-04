"""
src/data/build_interim.py

Runs the full Phase 1 data pipeline for race results:

    results.csv
      → clean_results()      dtype enforcement, result_status, finished
      → _repair_duplicates() resolve known Ergast duplicate entries
      → _repair_positions()  fix position null where positionText is numeric
      → validate_results()   assert clean constraints
      → data/interim/results.parquet

Usage
-----
    python src/data/build_interim.py             # full pipeline
    python src/data/build_interim.py --dry-run   # validate only, do not write

The output parquet is the canonical input for Phase 2 (EDA) and
Phase 3 (feature engineering). Nothing downstream should read results.csv directly.

Known data quality issues repaired here
----------------------------------------
The Ergast dataset contains two categories of real-world data quality issues
that are fixed in this script rather than in the cleaner (which only transforms,
never decides between competing rows):

1. Duplicate (raceId, driverId) entries — 85 pairs, 176 rows affected.
   Likely caused by: sprint-race entries added to the main results table,
   or post-race result corrections added as new rows instead of updates.
   Resolution: for each pair, keep the row that best represents the final
   classified result — Finished rows are preferred over non-Finished; ties
   are broken by highest resultId (most recently added / corrected entry).

2. Null position where positionText is numeric — 2 rows in raceId=71.
   The raw CSV has a null `position` value but a valid numeric `positionText`.
   Resolution: fill position from positionText when finished=True and
   position is null.

See decisions.md Decision 007 for the rationale behind these choices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INTERIM_DIR = _PROJECT_ROOT / "data" / "interim"
_OUTPUT_PATH = _INTERIM_DIR / "results.parquet"
_QUALIFYING_OUTPUT_PATH = _INTERIM_DIR / "qualifying.parquet"

from src.data.cleaner import clean_qualifying, clean_results
from src.data.loader import load_csv
from src.data.validator import validate_qualifying, validate_results


# ---------------------------------------------------------------------------
# Repair helpers
# ---------------------------------------------------------------------------

def _repair_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Resolve duplicate (raceId, driverId) entries.

    Strategy
    --------
    For each duplicate group, keep exactly one row using this priority:
      1. Prefer rows where result_status == "Finished" (classified result)
      2. Among ties, keep the row with the highest resultId (latest entry)

    Returns
    -------
    (deduplicated_df, n_rows_dropped)
    """
    is_finished = (df["result_status"] == "Finished").astype(int)
    df_sorted = df.assign(_is_finished=is_finished).sort_values(
        ["raceId", "driverId", "_is_finished", "resultId"],
        ascending=[True, True, False, False],
    )
    deduped = df_sorted.drop_duplicates(subset=["raceId", "driverId"], keep="first")
    deduped = deduped.drop(columns=["_is_finished"])
    n_dropped = len(df) - len(deduped)
    return deduped, n_dropped


def _repair_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fix rows where finished=True but position is null.

    The Ergast CSV occasionally has a null `position` column despite a valid
    numeric `positionText`. Derive position from positionText in those cases.

    Returns
    -------
    (repaired_df, n_rows_fixed)
    """
    df = df.copy()
    needs_repair = df["finished"] & df["position"].isna()

    if needs_repair.any() and "positionText" in df.columns:
        extracted = (
            df.loc[needs_repair, "positionText"]
            .astype(str)
            .str.extract(r"^(\d+)$")[0]
            .astype("Int64")
        )
        df.loc[needs_repair, "position"] = extracted

    n_fixed = int(needs_repair.sum())
    return df, n_fixed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_interim(
    dry_run: bool = False,
    output_path: Path = _OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load, clean, repair, validate, and optionally save the race results dataset.

    Parameters
    ----------
    dry_run : bool
        If True, run the full pipeline but skip writing the parquet file.
    output_path : Path
        Destination parquet path. Defaults to data/interim/results.parquet.

    Returns
    -------
    pd.DataFrame
        The validated, cleaned DataFrame.

    Raises
    ------
    ValueError
        If validation fails after repairs.
    FileNotFoundError
        If results.csv is not found in data/.
    """
    print("=== Phase 1 — Build Interim Dataset ===\n")

    # Step 1: Load
    print("1/4  Loading results.csv ...")
    raw = load_csv("results.csv")
    print(f"     Loaded {len(raw):,} rows x {len(raw.columns)} columns.\n")

    # Step 2: Clean
    print("2/4  Cleaning ...")
    cleaned = clean_results(raw)
    status_counts = cleaned["result_status"].value_counts().to_dict()
    print(f"     result_status breakdown: {status_counts}")
    print(f"     Finished: {cleaned['finished'].sum():,} / {len(cleaned):,} rows\n")

    # Step 3: Repair known data quality issues
    print("3/4  Repairing known data quality issues ...")
    repaired, n_dupes_dropped = _repair_duplicates(cleaned)
    repaired, n_pos_fixed = _repair_positions(repaired)
    if n_dupes_dropped:
        print(f"     Dropped {n_dupes_dropped} duplicate (raceId, driverId) rows.")
    if n_pos_fixed:
        print(f"     Fixed {n_pos_fixed} null position(s) from positionText.")
    if not n_dupes_dropped and not n_pos_fixed:
        print("     No repairs needed.")
    print()

    # Step 4: Validate
    print("4/4  Validating ...")
    result = validate_results(repaired, raise_on_error=True)
    print(result.summary())

    # Save
    if dry_run:
        print("\nDry run — skipping write.")
    else:
        _INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        repaired.to_parquet(output_path, index=False)
        size_kb = output_path.stat().st_size / 1024
        print(f"\nSaved: {output_path}  ({size_kb:.1f} KB)")

    return repaired


def build_qualifying_interim(
    dry_run: bool = False,
    output_path: Path = _QUALIFYING_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load, clean, validate, and optionally save the qualifying dataset.

    No repair step is needed: qualifying.csv has no duplicate (raceId,
    driverId) pairs and no null keys as-is (verified against the current
    data snapshot) — unlike results.csv, which required the repairs in
    Decision 007.

    Parameters
    ----------
    dry_run : bool
        If True, run the full pipeline but skip writing the parquet file.
    output_path : Path
        Destination parquet path. Defaults to data/interim/qualifying.parquet.

    Returns
    -------
    pd.DataFrame
        The validated, cleaned DataFrame.

    Raises
    ------
    ValueError
        If validation fails.
    FileNotFoundError
        If qualifying.csv is not found in data/.
    """
    print("=== Build Interim Dataset — Qualifying ===\n")

    print("1/3  Loading qualifying.csv ...")
    raw = load_csv("qualifying.csv")
    print(f"     Loaded {len(raw):,} rows x {len(raw.columns)} columns.\n")

    print("2/3  Cleaning ...")
    cleaned = clean_qualifying(raw)
    print(f"     Dtypes cast; q1/q2/q3 left as raw time strings.\n")

    print("3/3  Validating ...")
    result = validate_qualifying(cleaned, raise_on_error=True)
    print(result.summary())

    if dry_run:
        print("\nDry run — skipping write.")
    else:
        _INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(output_path, index=False)
        size_kb = output_path.stat().st_size / 1024
        print(f"\nSaved: {output_path}  ({size_kb:.1f} KB)")

    return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build data/interim/*.parquet from raw Ergast CSVs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline(s) without writing output file(s).",
    )
    parser.add_argument(
        "--target",
        choices=["results", "qualifying", "all"],
        default="all",
        help="Which interim dataset(s) to build (default: all).",
    )
    args = parser.parse_args()

    try:
        if args.target in ("results", "all"):
            build_interim(dry_run=args.dry_run)
        if args.target in ("qualifying", "all"):
            print()
            build_qualifying_interim(dry_run=args.dry_run)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
