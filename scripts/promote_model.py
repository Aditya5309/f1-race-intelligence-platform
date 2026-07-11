"""
scripts/promote_model.py

The sanctioned promotion gate (Phase 4 Tranche C) between "a candidate is
registered in MLflow" and "the candidate is what the deployed API serves".

Problem this closes: src.models.train.register_model() ALWAYS calls
serving_bundle.export_bundle(), which `shutil.rmtree`s whatever is at
artifacts/serving/<alias>/model/ and writes the new candidate over it — with
zero validation in between. Every "don't promote" decision made so far
(e.g. Tranche B, Decision 034) was pure human discipline: the operator
looked at metrics and chose not to run `--register`. Nothing in the tooling
stopped a bad promotion from overwriting a good one.

This script does NOT fit or register anything new — it operates on a model
version that ALREADY EXISTS in the MLflow registry (produced by a prior
`python -m src.models.train --register <alias> ...` call) and decides
whether that version is allowed to become the live serving bundle.

    python scripts/promote_model.py --alias Staging
    python scripts/promote_model.py --alias Staging --version 3
    python scripts/promote_model.py --alias Production --top1-tolerance 0.02

Checks, in order (all against the CANDIDATE, none of them retrain anything):

  1. Loads without error — mlflow.sklearn.load_model("models:/f1-winner/N").
  2. ColumnGuard self-consistency + non-degenerate predictions — scores a
     handful of real, in-window races (drawn from --features-source) via
     src.models.predict.predict_race(), which routes through the
     candidate's own recorded training schema. predict_race() already
     raises on NaN probabilities and guarantees the per-race sum-to-1
     normalization; this step additionally rejects a race where every
     driver gets an identical probability (a constant-output model would
     pass both of predict_race's own checks while being useless).
  3. Metric non-regression — the candidate is scored fresh on the
     Decision-008 validation split (2022-2023, ~44 races) via evaluate_all
     (the same evaluation code every other module in this project uses,
     including the Tranche A Spearman baseline) and compared against
     whatever's recorded in the CURRENTLY-SERVED bundle's manifest.json
     (populated by register_model() since Tranche C Item 1). Refuses to
     promote if top1_accuracy or spearman_corr regress by more than the
     configured tolerance.

     Default tolerances: 0.03 (top1_accuracy) / 0.015 (spearman_corr). The
     2022-2023 validation split is ~44 races (README §10), so one race
     flipping from a correct to incorrect top-1 pick moves top1_accuracy by
     ~1/44 ≈ 0.023 — the default tolerance covers a little more than one
     race of sampling noise without masking a real regression. Tranche B's
     genuine regressions (reports/phase4_tranche_b_retrain_findings.md)
     moved spearman_corr by 0.017-0.021 on the 2024 test split; 0.015 would
     have caught every one of those on this val-set analogue while
     tolerating ordinary float-level noise.

  Cheap by design: no retraining, no CV — just loading an already-fitted
  model, one predict_proba pass over the val split, and a handful of extra
  predict_race calls (seconds, not minutes).

Only on success: freezes the candidate as artifacts/serving/<alias>/
(serving_bundle.export_bundle) and refreshes the runtime features snapshot
(serving_bundle.export_features_snapshot) — the same two calls
register_model() makes internally, just gated. On any failure: exits 1,
prints the specific reason, and touches nothing under artifacts/.

Calling register_model()/export_bundle() directly still works (see their
own docstrings) but bypasses every check above — don't use them for a real
promotion; this script is the sanctioned path going forward.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))          # runnable without pip install

from src.features.pipeline import FEATURES_PATH, MASTER_DATASET_PATH  # noqa: E402
from src.models.evaluate import evaluate_all  # noqa: E402
from src.models.predict import predict_race  # noqa: E402
from src.models.serving_bundle import (  # noqa: E402
    ModelInfo,
    bundle_dir_for_alias,
    export_bundle,
    export_features_snapshot,
    load_bundle,
)
from src.models.splits import temporal_split, to_xy  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    load_position_order,
)

DEFAULT_TOP1_TOLERANCE = 0.03
DEFAULT_SPEARMAN_TOLERANCE = 0.015
#: How many of the most recent real races to smoke-test predict_race() on.
SMOKE_RACE_SAMPLE = 5


class PromotionRefused(Exception):
    """Raised with a human-readable reason; caught once in main()."""


def resolve_version(client: mlflow.MlflowClient, version: str | None) -> str:
    """--version if given, else the highest existing version number
    ("the most recent --register invocation's output")."""
    if version is not None:
        return version
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise PromotionRefused(
            f"No versions of '{REGISTERED_MODEL_NAME}' exist in the registry yet — "
            "run `python -m src.models.train --register <alias>` first."
        )
    return str(max(int(v.version) for v in versions))


def load_candidate(client: mlflow.MlflowClient, version: str):
    """Check 1: loads without error. Returns (model, ModelVersion)."""
    mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    try:
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    except Exception as exc:
        raise PromotionRefused(f"Candidate v{version} failed to load: {exc}") from exc
    return model, mv


def check_schema_and_predictions(model, features: pd.DataFrame) -> None:
    """Check 2: predict_race() on real races — schema validation, no NaN,
    sums to 1 (all enforced inside predict_race itself), and NOT uniform
    across the field (the one thing predict_race does not check itself)."""
    race_ids = features["raceId"].drop_duplicates().sort_values()
    sample = race_ids.tail(SMOKE_RACE_SAMPLE)
    if sample.empty:
        raise PromotionRefused(f"No races found in {len(features)} rows to smoke-test against.")
    for race_id in sample:
        race_df = features[features["raceId"] == race_id]
        try:
            out = predict_race(model, race_df)
        except Exception as exc:
            raise PromotionRefused(
                f"predict_race failed on real race {race_id}: {exc}"
            ) from exc
        if len(out) > 1 and out["win_probability"].nunique() <= 1:
            raise PromotionRefused(
                f"Race {race_id}: every driver got an identical win_probability "
                f"({out['win_probability'].iloc[0]:.4f}) — degenerate output."
            )


def compute_candidate_metrics(model, split, position_order) -> dict[str, float]:
    """Check 3 (part 1): fresh evaluate_all() on split.val — never split.test
    (Section 11.3). Mirrors register_model()'s own manifest-metrics logic."""
    X_val, y_val, races_val = to_xy(split.val)
    y_prob = model.predict_proba(X_val)[:, 1]
    po_val = None
    if position_order is not None:
        index = pd.MultiIndex.from_frame(split.val[["raceId", "driverId"]])
        po_val = position_order.reindex(index).to_numpy(dtype=float)
    return evaluate_all(y_val, y_prob, races_val, position_order=po_val)


def check_regression(
    candidate: dict, served: dict, top1_tolerance: float, spearman_tolerance: float,
) -> None:
    """Check 3 (part 2): refuse if top1_accuracy or spearman_corr regressed
    past tolerance vs. the currently-served bundle. Missing keys on either
    side (e.g. no position_order -> no spearman_corr) skip that comparison
    rather than erroring — absence isn't evidence of regression."""
    for key, tolerance in (("top1_accuracy", top1_tolerance), ("spearman_corr", spearman_tolerance)):
        cand_val, served_val = candidate.get(key), served.get(key)
        if cand_val is None or served_val is None:
            continue
        if cand_val < served_val - tolerance:
            raise PromotionRefused(
                f"{key} regressed: candidate={cand_val:.4f} vs. currently-served="
                f"{served_val:.4f} (tolerance {tolerance})."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate + promote an already-registered MLflow model "
                     "version to the live serving bundle (Phase 4 Tranche C).")
    parser.add_argument("--alias", default="Staging", choices=["Staging", "Production"],
                        help="Which serving bundle directory to promote into.")
    parser.add_argument("--version", default=None,
                        help=f"MLflow version of '{REGISTERED_MODEL_NAME}' to promote "
                             "(default: the most recently registered version).")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--bundle-root", type=Path, default=None,
                        help="Default: artifacts/serving/ (see serving_bundle.py).")
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help="Default: artifacts/ (see serving_bundle.py).")
    parser.add_argument("--features-source", type=Path, default=FEATURES_PATH,
                        help="Training-side features.parquet to smoke-test "
                             "against and freeze as the new runtime snapshot "
                             "(default: data/processed/features.parquet).")
    parser.add_argument("--top1-tolerance", type=float, default=DEFAULT_TOP1_TOLERANCE)
    parser.add_argument("--spearman-tolerance", type=float, default=DEFAULT_SPEARMAN_TOLERANCE)
    args = parser.parse_args(argv)

    if not args.features_source.exists():
        print(f"ERROR: {args.features_source} not found — run "
              "`python -m src.features.pipeline` or pass --features-source.",
              file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    try:
        version = resolve_version(client, args.version)
        model, mv = load_candidate(client, version)
        print(f"Candidate: {REGISTERED_MODEL_NAME} v{version} "
              f"({type(model).__name__}, run {mv.run_id})")

        features = pd.read_parquet(args.features_source)
        check_schema_and_predictions(model, features)
        print(f"[1/2] Smoke checks (load, schema, non-degenerate on "
              f"{SMOKE_RACE_SAMPLE} real races) PASS")

        split = temporal_split(features)
        position_order = load_position_order() if MASTER_DATASET_PATH.exists() else None
        candidate_metrics = compute_candidate_metrics(model, split, position_order)

        bundle_dir = bundle_dir_for_alias(args.alias, args.bundle_root)
        try:
            _, served_info = load_bundle(bundle_dir)
            served_metrics = served_info.metrics
        except FileNotFoundError:
            served_metrics = {}

        if served_metrics:
            check_regression(candidate_metrics, served_metrics,
                             args.top1_tolerance, args.spearman_tolerance)
        else:
            print(f"No existing bundle (or no recorded metrics) at {bundle_dir} — "
                  "regression check skipped.")
        print(f"[2/2] Metric non-regression PASS "
              f"(candidate top1_accuracy={candidate_metrics['top1_accuracy']:.4f}, "
              f"spearman_corr={candidate_metrics.get('spearman_corr', float('nan')):.4f})")

    except PromotionRefused as exc:
        print(f"PROMOTION REFUSED: {exc}", file=sys.stderr)
        return 1

    info = ModelInfo(
        name=REGISTERED_MODEL_NAME,
        version=str(version),
        alias=args.alias,
        run_id=mv.run_id,
        trained_at=datetime.fromtimestamp(
            mv.creation_timestamp / 1000, tz=UTC
        ).isoformat(timespec="seconds"),
        calibration=getattr(model, "calibration", "none"),
        model_class=type(model).__name__,
        metrics=candidate_metrics,
    )
    bundle_dir = export_bundle(model, info, bundle_root=args.bundle_root)
    export_features_snapshot(args.features_source, artifacts_root=args.artifacts_root)
    print(f"PROMOTED {REGISTERED_MODEL_NAME} v{version} to @{args.alias} -> {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
