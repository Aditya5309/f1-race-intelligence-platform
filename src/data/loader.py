"""
Data loader for the Ergast-format F1 CSV dataset.

Usage:
    python src/data/loader.py                  # profile all CSVs in data/
    python src/data/loader.py results.csv      # profile one file
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Ergast MySQL dumps encode NULL as \N
_NA_VALUES = ["\\N", "\\\\N", "", "NA", "N/A"]


def load_csv(filename: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a single CSV from data_dir, treating Ergast \\N tokens as NaN."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, na_values=_NA_VALUES, keep_default_na=True)


def profile(df: pd.DataFrame, name: str) -> None:
    """Print shape, dtypes, summary statistics, and missing-value report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)

    # Shape
    rows, cols = df.shape
    print(f"\nShape: {rows:,} rows  x  {cols} columns")

    # Column names and dtypes
    print("\nColumns and dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<30} {dtype}")

    # Summary statistics (numeric columns only)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        stats = numeric.agg(["mean", "std", "min", "max"]).T
        stats.columns = ["mean", "std", "min", "max"]
        print("\nNumeric summary:")
        print(
            stats.to_string(
                float_format=lambda x: f"{x:>12.4f}"
            )
        )

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\nMissing values: none")
    else:
        print("\nMissing values:")
        pct = (missing / rows * 100).round(2)
        report = pd.DataFrame({"count": missing, "pct": pct})
        report.columns = ["count", "%"]
        print(report.to_string())


def load_all(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load every CSV in data_dir and return a {filename: DataFrame} dict."""
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {data_dir}")
        return {}
    datasets = {}
    for path in csvs:
        datasets[path.name] = load_csv(path.name, data_dir)
    return datasets


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        df = load_csv(target)
        profile(df, target)
    else:
        datasets = load_all()
        for name, df in datasets.items():
            profile(df, name)
        print(f"\nLoaded {len(datasets)} files from {DATA_DIR}")
