"""
src/models/season_tracking.py

Continuous out-of-sample tracking for the current, ONGOING regulation era
(the "Phase 8 monitoring/retraining rehearsal" named in
reports/model_development_design.md S13.1/S14; era boundaries from
src/models/eras.py, Decision 019). Scores each newly completed race under
the live era with whatever model is CURRENTLY SERVED
(artifacts/serving/<alias>) and appends one row per race to a running CSV —
e.g. artifacts/tracking/2026_running_eval.csv while `future_engine` (start
year 2026) is the open era.

THIS IS NOT A TRAINING INPUT. Structural guarantee, not just this
docstring's claim: this module imports only src.models.{eras, predict,
evaluate, serving_bundle} plus src.features.pipeline's data-artifact path
constants (FEATURES_PATH, MASTER_DATASET_PATH, TARGET_COLUMN) — never
src.models.train or src.models.splits. `temporal_split` is never called,
and no design matrix is ever assembled for fitting; predict_race() builds a
read-only inference input against a model that was already fit elsewhere.
tests/test_season_tracking.py verifies this by importing the module in a
clean subprocess and asserting src.models.train / src.models.splits never
enter sys.modules.

Why score the CURRENTLY SERVED bundle rather than a freshly retrained
candidate: this answers "how is the model actually doing in the wild", a
different question from "would a new candidate be better" (that's
promote_model.py's job — untouched by this module). Hooked into
scripts/refresh_and_freeze.py as the step right after the feature rebuild
and before registration, so it always sees this week's newly completed
race(s) but scores them with the bundle that was serving BEFORE this run —
never the not-yet-promoted candidate about to be registered later in the
same run.

Why one row per race, not only a rolling aggregate: a single running number
hides which specific races drove it and can't be re-sliced later (by round,
by a mid-season rule tweak, etc.). Each row is computed by calling
evaluate.evaluate_all() on that ONE race's rows — reusing the exact metric
definitions/keys used everywhere else in this project rather than
inventing new ones. Aggregates in `summarize()` are therefore a plain mean
across each race's own evaluate_all() output — an approximation for
row-count-sensitive metrics (log_loss, brier_score, ece), clearly named
`mean_*` for that reason — not a row-level pooled recomputation.

Sample-size honesty (same discipline as the Tranche B ablation's "1/24 test
races" framing): the ongoing era starts with a handful of races.
`summarize()` always returns the race count alongside every aggregate, and
the CLI never prints a headline number without that count on the same
line, plus an explicit small-sample banner below SMALL_SAMPLE_THRESHOLD.

    python -m src.models.season_tracking                     # score + print summary
    python -m src.models.season_tracking --alias Production  # track the Production bundle instead
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import eras
from src.models.evaluate import evaluate_all
from src.models.predict import DEFAULT_ALIAS, load_model, predict_race
from src.models.serving_bundle import bundle_dir_for_alias

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_DIR = _PROJECT_ROOT / "artifacts" / "tracking"

#: One full F1 season (~22-24 races). Below this, any aggregate is a
#: directional read only — flagged loudly, never silently trusted.
SMALL_SAMPLE_THRESHOLD = 24

_METRIC_COLUMNS = (
    "top1_accuracy", "top3_recall", "winner_mrr", "avg_winner_probability",
    "median_winner_rank", "log_loss", "brier_score", "ece",
)


def _ongoing_era() -> eras.RegulationEra:
    """The one era table entry with end_year=None (import-time-guaranteed
    to be exactly the final entry — see src/models/eras.py)."""
    for era in eras.REGULATION_ERAS:
        if era.is_ongoing:
            return era
    raise RuntimeError(
        "No ongoing regulation era in eras.REGULATION_ERAS — every era "
        "table must end with one open-ended entry."
    )


def default_report_path(
    era: eras.RegulationEra | None = None, tracking_dir: Path | None = None,
) -> Path:
    """artifacts/tracking/{era.start_year}_running_eval.csv for the ongoing
    era by default — e.g. 2026_running_eval.csv today. Automatically shifts
    to the new era's start year once `future_engine` is closed and a new
    era opens in src/models/eras.py; no change needed here."""
    era = era or _ongoing_era()
    tracking_dir = tracking_dir or DEFAULT_TRACKING_DIR
    return tracking_dir / f"{era.start_year}_running_eval.csv"


#: Metadata columns that must round-trip through CSV as strings — without
#: an explicit dtype, pandas infers a purely-numeric model_version (e.g.
#: "1") as int64 on reload, which would silently disagree with the object
#: (str) dtype of a freshly-scored, not-yet-persisted frame.
_STRING_COLUMNS = (
    "model_name", "model_version", "model_alias", "model_run_id",
    "model_calibration", "scored_at",
)


def _load_report(report_path: Path) -> pd.DataFrame:
    if report_path.exists():
        return pd.read_csv(report_path, dtype={c: str for c in _STRING_COLUMNS})
    return pd.DataFrame()


def _load_position_order(master_path: Path) -> pd.Series:
    """`positionOrder` indexed by (raceId, driverId).

    Deliberately NOT imported from src.models.train.load_position_order —
    it is the same three lines, reimplemented here so this module never
    imports src.models.train (see the module docstring's structural
    guarantee).
    """
    master = pd.read_parquet(master_path, columns=["raceId", "driverId", "positionOrder"])
    return master.set_index(["raceId", "driverId"])["positionOrder"]


def score_new_races(
    *,
    alias: str = DEFAULT_ALIAS,
    bundle_root: Path | None = None,
    features_path: Path | None = None,
    master_path: Path | None = None,
    report_path: Path | None = None,
) -> pd.DataFrame:
    """
    Score every newly completed race in the ongoing era that isn't already
    in the report, append the results, and return the FULL updated report
    (not just the new rows). Idempotent: re-running with nothing new
    completed returns the existing report unchanged, and never rescoring an
    already-tracked raceId keeps a race's recorded score pinned to whichever
    bundle was actually serving when it first completed.

    `features_path`/`master_path` default to the TRAINING-side
    data/processed/ files (src.features.pipeline.FEATURES_PATH /
    MASTER_DATASET_PATH) — the freshly rebuilt files a weekly ingest run
    just produced — NOT the frozen artifacts/features.parquet serving
    snapshot, which may still be last week's and would miss this week's
    newly completed race entirely.
    """
    from src.features.pipeline import FEATURES_PATH, MASTER_DATASET_PATH, TARGET_COLUMN

    era = _ongoing_era()
    features_path = features_path or FEATURES_PATH
    master_path = master_path or MASTER_DATASET_PATH
    report_path = report_path or default_report_path(era)

    features = pd.read_parquet(features_path)
    if "year" not in features.columns:
        raise KeyError(f"{features_path} has no 'year' column.")

    existing = _load_report(report_path)
    already_tracked = set(existing["race_id"]) if "race_id" in existing.columns else set()

    mask = features["year"] >= era.start_year
    if era.end_year is not None:
        mask &= features["year"] <= era.end_year
    era_rows = features.loc[mask]

    new_rows = era_rows.loc[~era_rows["raceId"].isin(already_tracked)]
    if new_rows.empty:
        return existing

    # Defense in depth: a race with no confirmed single winner yet (result
    # not final) must never be scored as if it were. In practice the
    # feature pipeline only emits fully-raced rows, but this module must
    # not assume that silently.
    completeness = new_rows.groupby("raceId")[TARGET_COLUMN].sum()
    incomplete = completeness[completeness != 1].index
    if len(incomplete):
        print(
            f"NOTE: skipping {len(incomplete)} race(s) with no confirmed "
            f"single winner yet (raceId(s): {sorted(incomplete)}) — not "
            "complete, not scored this run."
        )
        new_rows = new_rows.loc[~new_rows["raceId"].isin(incomplete)]
    if new_rows.empty:
        return existing

    model, info = load_model(bundle_dir_for_alias(alias, bundle_root))
    predictions = predict_race(model, new_rows)
    merged = predictions.merge(
        new_rows[["raceId", "driverId", TARGET_COLUMN]], on=["raceId", "driverId"], how="left",
    )

    position_order = _load_position_order(master_path) if master_path.exists() else None
    scored_at = datetime.now(UTC).isoformat(timespec="seconds")

    new_report_rows = []
    for race_id, group in merged.groupby("raceId", sort=True):
        po = None
        if position_order is not None:
            idx = pd.MultiIndex.from_frame(group[["raceId", "driverId"]])
            po_values = position_order.reindex(idx).to_numpy(dtype=float)
            if not np.isnan(po_values).any():
                po = po_values

        metrics = evaluate_all(
            group[TARGET_COLUMN].to_numpy(),
            group["win_probability_raw"].to_numpy(),
            group["raceId"].to_numpy(),
            position_order=po,
        )
        new_report_rows.append({
            "race_id": int(race_id),
            "year": int(group["year"].iloc[0]),
            "round": int(group["round"].iloc[0]),
            "n_drivers": len(group),
            **metrics,
            "model_name": info.name,
            "model_version": info.version,
            "model_alias": info.alias,
            "model_run_id": info.run_id,
            "model_calibration": info.calibration,
            "scored_at": scored_at,
        })

    updated = pd.concat([existing, pd.DataFrame(new_report_rows)], ignore_index=True)
    updated = updated.sort_values("race_id").reset_index(drop=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(report_path, index=False)
    return updated


def summarize(
    report_path: Path | None = None, era: eras.RegulationEra | None = None,
) -> dict:
    """Race-count-qualified aggregates over the accumulated report so far.

    Always includes `n_races` and `small_sample` (n_races < 24) — never
    return a headline metric without them; the CLI relies on this to keep
    the sample size next to every printed number, not a footnote.
    """
    era = era or _ongoing_era()
    report_path = report_path or default_report_path(era)
    report = _load_report(report_path)
    if report.empty:
        return {"era": era.name, "n_races": 0, "small_sample": True}

    n = len(report)
    out: dict = {
        "era": era.name,
        "n_races": n,
        "small_sample": n < SMALL_SAMPLE_THRESHOLD,
        **{
            ("mean_" + col if col in ("log_loss", "brier_score", "ece") else col):
                float(report[col].mean())
            for col in _METRIC_COLUMNS
        },
    }
    if "spearman_corr" in report.columns and report["spearman_corr"].notna().any():
        out["spearman_corr"] = float(report["spearman_corr"].mean())
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(summary: dict, report_path: Path) -> None:
    n = summary["n_races"]
    if n == 0:
        print(f"No completed '{summary['era']}' races tracked yet at {report_path}.")
        return

    banner = " *** SMALL SAMPLE — directional only, not conclusive *** " if summary["small_sample"] else ""
    print(f"Era '{summary['era']}' tracking: n={n} race(s) scored.{banner}")
    for key, label in (
        ("top1_accuracy", "top1_accuracy"),
        ("top3_recall", "top3_recall"),
        ("winner_mrr", "winner_mrr"),
        ("avg_winner_probability", "avg_winner_probability"),
        ("mean_log_loss", "mean_log_loss"),
        ("mean_brier_score", "mean_brier_score"),
        ("spearman_corr", "spearman_corr"),
    ):
        if key in summary:
            print(f"  {label:<24} = {summary[key]:.4f}  (n={n} races)")
    print(f"Report: {report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score newly completed races in the current regulation "
                     "era with the currently-served model bundle (read-only "
                     "tracking, never a training input).")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, choices=["Staging", "Production"],
                        help="Which served bundle's alias to score with.")
    parser.add_argument("--bundle-root", type=Path, default=None,
                        help="Serving bundle root (default: artifacts/serving/).")
    parser.add_argument("--features-path", type=Path, default=None,
                        help="Features parquet to scan for new races "
                             "(default: data/processed/features.parquet — "
                             "the freshest training-side build).")
    parser.add_argument("--master-path", type=Path, default=None,
                        help="master_dataset.parquet for the Spearman join "
                             "(default: data/processed/master_dataset.parquet).")
    parser.add_argument("--tracking-dir", type=Path, default=None,
                        help="Directory for the running eval CSV "
                             "(default: artifacts/tracking/).")
    args = parser.parse_args(argv)

    era = _ongoing_era()
    report_path = default_report_path(era, tracking_dir=args.tracking_dir)

    score_new_races(
        alias=args.alias, bundle_root=args.bundle_root,
        features_path=args.features_path, master_path=args.master_path,
        report_path=report_path,
    )
    summary = summarize(report_path=report_path, era=era)
    _print_summary(summary, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
