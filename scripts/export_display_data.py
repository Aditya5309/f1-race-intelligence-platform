"""
scripts/export_display_data.py

Freezes a small, committed subset of the Ergast display-metadata CSVs from
the gitignored training-side data/ tree into artifacts/display/ — the
directory app/config.py's Settings.data_dir now defaults to.

Why this exists: data/ is entirely gitignored (Phase 5 / Decision 016). A
fresh clone (Streamlit Cloud, Render, CI) has no data/ at all, so every
display-name lookup (app/api.py::_load_name_lookups, app/views/metadata.py)
silently degrades to null/fallback text — the "race/driver name missing in
prod" bug. Decision 029 already solved this exact shape of problem for the
model itself (frozen artifacts/features.parquet + artifacts/serving/); this
script does the same for display data.

The exported set is deliberately the FULL set of files any app/-side reader
touches (found by grepping the whole repo for data_dir/read_csv usage, not
by eyeballing known call sites — that under-count is what caused the bug):
races, circuits, drivers, constructors, results, qualifying,
driver_standings, constructor_standings, status. lap_times.csv (24MB) and
the other Ergast tables are excluded — nothing under app/ reads them.

    python scripts/export_display_data.py                # data/ -> artifacts/display/
    python scripts/export_display_data.py --source PATH --dest PATH
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _PROJECT_ROOT / "data"
DEFAULT_DEST = _PROJECT_ROOT / "artifacts" / "display"

#: Every CSV any app/-side reader touches via Settings().data_dir
#: (app/api.py::_load_name_lookups, app/views/metadata.py) — enumerated by
#: grepping the whole repo for data_dir/read_csv, not by call-site sampling.
#: status.csv has no reader yet but is required by the Historical Replay /
#: "how wrong" DNF-reason lookup added alongside this fix.
DISPLAY_FILES = (
    "races.csv",
    "circuits.csv",
    "drivers.csv",
    "constructors.csv",
    "results.csv",
    "qualifying.csv",
    "driver_standings.csv",
    "constructor_standings.csv",
    "status.csv",
)


def export_display_data(source: Path, dest: Path) -> list[str]:
    """Copy DISPLAY_FILES from source to dest. Returns the filenames copied;
    raises FileNotFoundError listing any that are missing from source."""
    missing = [f for f in DISPLAY_FILES if not (source / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing from {source}: {missing}. Run this script from a "
            "checkout with the full training-side data/ tree present."
        )
    dest.mkdir(parents=True, exist_ok=True)
    for filename in DISPLAY_FILES:
        shutil.copy2(source / filename, dest / filename)
    return list(DISPLAY_FILES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"Training-side data/ tree (default: {DEFAULT_SOURCE}).")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Committed display-data tree (default: {DEFAULT_DEST}).")
    args = parser.parse_args(argv)

    try:
        copied = export_display_data(args.source, args.dest)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total_bytes = sum((args.dest / f).stat().st_size for f in copied)
    print(f"Exported {len(copied)} files ({total_bytes / 1e6:.1f} MB) "
          f"from {args.source} to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
