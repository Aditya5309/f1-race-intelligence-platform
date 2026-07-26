"""
scripts/export_analysis_data.py

Freezes a small, committed subset of the gitignored reports/ tree (Decision
021) into artifacts/analysis/ — the directory app/config.py's
Settings.analysis_data_dir defaults to. Same shape of fix as
export_display_data.py (Decision 030): a fresh clone (Streamlit Cloud,
Render, CI) has no reports/ at all, so Model Insights' backing routes would
otherwise have nothing to serve.

Exports ONLY the files with genuine, already-tabular chart data — not the
PNG images themselves (those stay screenshots-only in reports/, never
served; the frontend redraws these three as native charts instead, reading
the same underlying numbers):
  - model_comparison.csv          (reports/)
  - feature_importance_logreg.csv (reports/phase4_analysis/)
  - shap_summary_logreg.csv       (reports/phase4_analysis/)

Deliberately NOT exported (Frontend Overhaul Phase 4, Decision 063): the
remaining reports/phase4_analysis/ artifacts — SHAP beeswarm/waterfall/
dependence plots, the calibration reliability diagram, confusion matrix,
ROC curve, precision-recall curve. src/models/analysis.py never computes
tabular data for the latter four (no calibration_curve/confusion_matrix/
roc_curve/precision_recall_curve call exists anywhere in this repo's own
pipeline) — they exist as PNGs from an out-of-repo, untracked process.
Exporting them would mean writing new analysis code, a real scope decision
left for a follow-up, not something to invent as a side effect of a
frontend-conversion phase.

    python scripts/export_analysis_data.py                # reports/ -> artifacts/analysis/
    python scripts/export_analysis_data.py --source PATH --dest PATH
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _PROJECT_ROOT / "reports"
DEFAULT_DEST = _PROJECT_ROOT / "artifacts" / "analysis"

#: (path relative to source, destination filename) — see module docstring
#: for why only these three files qualify.
ANALYSIS_FILES = (
    ("model_comparison.csv", "model_comparison.csv"),
    ("phase4_analysis/feature_importance_logreg.csv", "feature_importance_logreg.csv"),
    ("phase4_analysis/shap_summary_logreg.csv", "shap_summary_logreg.csv"),
)


def export_analysis_data(source: Path, dest: Path) -> list[str]:
    """Copy ANALYSIS_FILES from source to dest. Returns the destination
    filenames copied; raises FileNotFoundError listing any source files
    that are missing."""
    missing = [rel for rel, _ in ANALYSIS_FILES if not (source / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing from {source}: {missing}. Run `python -m src.models.analysis` "
            "first (writes reports/phase4_analysis/) from a checkout with the full "
            "training-side data/ tree present."
        )
    dest.mkdir(parents=True, exist_ok=True)
    for rel, dest_name in ANALYSIS_FILES:
        shutil.copy2(source / rel, dest / dest_name)
    return [dest_name for _, dest_name in ANALYSIS_FILES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"Local reports/ tree (default: {DEFAULT_SOURCE}).")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Committed analysis-data tree (default: {DEFAULT_DEST}).")
    args = parser.parse_args(argv)

    try:
        copied = export_analysis_data(args.source, args.dest)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total_bytes = sum((args.dest / f).stat().st_size for f in copied)
    print(f"Exported {len(copied)} files ({total_bytes / 1e3:.1f} KB) "
          f"from {args.source} to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
