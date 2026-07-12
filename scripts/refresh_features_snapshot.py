"""
scripts/refresh_features_snapshot.py

Freezes the training-side data/processed/features.parquet as the runtime
serving snapshot artifacts/features.parquet (Decision 029) — independent of
model registration or promotion.

Why this exists as its OWN script (Part 1 fix, post-Tranche-D display-data
post-mortem): this snapshot answers "which races/drivers exist" — by
src/models/serving_bundle.py's own docstring, orthogonal to which model
version serves them (it is NOT alias-scoped). Despite that, it previously
only ever refreshed as a side effect of `train.py --register`/
`promote_model.py` succeeding. In the scheduled retrain workflow that meant
a real production gap: every week promotion is (correctly) refused, this
snapshot stayed silently frozen at whatever it was after the last
successful promotion, even though a full week of newly-ingested real race
results already existed in data/processed/features.parquet. A new race
could complete and never become predictable via the deployed API, with no
indication anything was stale.

This script lets scripts/refresh_and_freeze.py refresh the snapshot
unconditionally, every run — alongside display data (Decision 030) and the
2026 tracking set (src/models/season_tracking.py) — as "current vs. stale"
ground-truth data, never a model-quality judgment, so it belongs in the
SAME always-open PR as those two, not the gated model-promotion PR.
train.py::register_model()'s and promote_model.py's own calls to
export_features_snapshot() are UNCHANGED and still also refresh it at
registration/promotion time — this is additive, not a replacement, so a
human running either of those manually still gets the behavior already
documented for them.

    python scripts/refresh_features_snapshot.py
    python scripts/refresh_features_snapshot.py --source PATH --artifacts-root PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))          # runnable without pip install

from src.features.pipeline import FEATURES_PATH  # noqa: E402
from src.models.serving_bundle import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    export_features_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=FEATURES_PATH,
                        help=f"Training-side features.parquet (default: {FEATURES_PATH}).")
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help=f"Committed runtime artifacts root (default: {DEFAULT_ARTIFACTS_ROOT}).")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: {args.source} not found — run "
              "`python -m src.features.pipeline` first.", file=sys.stderr)
        return 1

    dest = export_features_snapshot(args.source, artifacts_root=args.artifacts_root)
    print(f"Refreshed runtime features snapshot: {args.source} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
