"""
src/pipelines/build_dataset.py

Orchestration entry point for the Master Dataset Integration layer (Decision 009).

    data/interim/results.parquet   ─┐
    data/interim/qualifying.parquet ─┤
    data/races.csv                  ─┼─► build_master_dataset() ─► data/processed/master_dataset.parquet
    data/drivers.csv                 │
    data/constructors.csv            │
    data/circuits.csv               ─┘

This script is orchestration only — it has no join/validation logic of its
own. All reusable logic lives in src/integration/build_master_dataset.py so it
can be called by future incremental-sync/ETL workflows without depending on
this CLI wrapper.

Usage
-----
    python -m src.pipelines.build_dataset             # full pipeline
    python -m src.pipelines.build_dataset --dry-run   # validate only, do not write

Scope
-----
Integration only — no feature engineering, no rolling statistics, no lagged
standings, no training data, no model training, no inference. See
reports/master_dataset_design.md and Decision 009.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.integration.build_master_dataset import (
    build_master_dataset,
    load_inputs,
    validate_inputs,
    validate_output,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_OUTPUT_PATH = _PROCESSED_DIR / "master_dataset.parquet"


def build_dataset(dry_run: bool = False, output_path: Path = _OUTPUT_PATH):
    """
    Run the full master-dataset integration pipeline.

    Steps
    -----
    1. Load cleaned interim datasets (+ raw dimension CSVs)
    2. Validate inputs (primary keys, duplicate pairs, non-null join keys)
    3. Build the master dataset (joins + target derivation)
    4. Validate output (schema, row counts, referential integrity, duplicates)
    5. Save data/processed/master_dataset.parquet

    Raises
    ------
    ValueError
        If input or output validation fails.
    FileNotFoundError
        If a required interim dataset is missing.
    """
    print("=== Master Dataset Integration Pipeline (Decision 009) ===\n")

    print("1/5  Loading interim datasets ...")
    inputs = load_inputs()
    for name, df in inputs.items():
        print(f"     {name:<15} {len(df):,} rows x {len(df.columns)} columns")
    print()

    print("2/5  Validating inputs ...")
    input_result = validate_inputs(inputs)
    print(input_result.summary())
    if not input_result.passed:
        raise ValueError(
            f"Input validation failed with {len(input_result.errors)} error(s). "
            "See summary above."
        )
    print()

    print("3/5  Building master dataset (joins) ...")
    master = build_master_dataset(inputs)
    expected_row_count = len(inputs["results"])
    print(f"     Built {len(master):,} rows x {len(master.columns)} columns "
          f"(base results row count: {expected_row_count:,})\n")

    print("4/5  Validating output ...")
    output_result = validate_output(master, expected_row_count=expected_row_count)
    print(output_result.summary())
    if not output_result.passed:
        raise ValueError(
            f"Output validation failed with {len(output_result.errors)} error(s). "
            "See summary above."
        )
    print()

    print("5/5  Saving ...")
    if dry_run:
        print("     Dry run — skipping write.")
    else:
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        master.to_parquet(output_path, index=False)
        size_kb = output_path.stat().st_size / 1024
        print(f"     Saved: {output_path}  ({size_kb:.1f} KB)")

    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build data/processed/master_dataset.parquet from interim datasets."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline without writing output file.",
    )
    args = parser.parse_args()

    try:
        build_dataset(dry_run=args.dry_run)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
