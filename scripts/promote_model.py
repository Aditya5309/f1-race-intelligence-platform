"""
scripts/promote_model.py

The sanctioned promotion gate between "a candidate is
registered in MLflow" and "the candidate is what the deployed API serves".

Problem this closes: src.models.train.register_model() ALWAYS calls
serving_bundle.export_bundle(), which `shutil.rmtree`s whatever is at
artifacts/serving/<alias>/model/ and writes the new candidate over it — with
zero validation in between. Historically, every "don't promote this
candidate" call was pure human discipline: the operator
looked at metrics and chose not to run `--register`. Nothing in the tooling
stopped a bad promotion from overwriting a good one.

This script does NOT fit or register anything new — it operates on a model
version that ALREADY EXISTS in the MLflow registry (produced by a prior
`python -m src.models.train --register <alias> ...` call) and decides
whether that version is allowed to become the live serving bundle.

    python scripts/promote_model.py --alias Staging
    python scripts/promote_model.py --alias Staging --version 3
    python scripts/promote_model.py --alias Production --top1-tolerance 0.02
    python scripts/promote_model.py --alias Staging --force-baseline   # bootstrap only — see Check 5

Checks, in order (all against the CANDIDATE, none of them retrain anything):

  1. Loads without error — mlflow.sklearn.load_model("models:/f1-winner/N").
  2. Model-class check — the candidate's actual
     fitted estimator's module (e.g. 'sklearn', 'xgboost', 'lightgbm') must
     be in --allowed-model-modules (default: permissive, all three — no
     live deployment is constrained yet). Catches "an XGBoost/LightGBM
     candidate was promoted but the serving environment only has
     scikit-learn installed" HERE, as a clear pre-deploy refusal, instead
     of as an unpickle-time ImportError at the deployed Lambda's cold
     start. Project-native estimators (e.g. PoleSitterBaseline) are always
     allowed — no third-party dependency to check.
  3. Excluded-feature check — refuses a candidate whose OWN
     recorded training schema (never a repository constant) contains any
     feature belonging to a group in src.features.metadata's
     EXCLUDED_FROM_TRAINING (currently wet_form, an experimental feature
     group a per-group ablation isolated as a real accuracy regression that
     doesn't generalize past the training window). to_xy()/get_model()
     already default to the
     exclusion-applied set, so a normal training run can't trip this — it
     exists as a last line of defense against a silently-bypassed
     exclusion, the same defense-in-
     depth spirit as the model-class check above.
  4. ColumnGuard self-consistency + non-degenerate predictions — scores a
     handful of real, in-window races (drawn from --features-source) via
     src.models.predict.predict_race(), which routes through the
     candidate's own recorded training schema. predict_race() already
     raises on NaN probabilities and guarantees the per-race sum-to-1
     normalization; this step additionally rejects a race where every
     driver gets an identical probability (a constant-output model would
     pass both of predict_race's own checks while being useless).
  5. Metric non-regression — the candidate is scored fresh on the
     VALIDATION split (2022-2023, ~44 races) via evaluate_all
     (the same evaluation code every other module in this project uses)
     and compared against whatever's recorded in the CURRENTLY-SERVED
     bundle's manifest.json (populated by register_model()) — ALSO a
     val-split number, never test. Refuses to promote if
     top1_accuracy or spearman_corr regress by more than the configured
     tolerance.

     DO NOT compare either side of this check against the frequently-quoted
     "0.749 Spearman" figure found elsewhere in this project's
     documentation — that figure is explicitly, consistently the 2024 TEST
     split's score (nothing outside `train.py --final-test` may ever touch
     split.test, so this gate structurally cannot use it). The currently-
     served model's OWN val-split
     spearman_corr, measured directly, is ~0.60 — a real, large, permanent
     gap from the oft-quoted 0.749 that has nothing to do with any
     regression; it is simply a different evaluation surface. A candidate
     whose val spearman_corr looks alarming next to "0.749" may be
     completely unremarkable next to the served bundle's ACTUAL val
     spearman_corr — always compare against the manifest.json numbers this
     gate itself prints, never against a remembered test-split figure.

     No baseline to compare against is handled as TWO DIFFERENT cases, not
     one (a real bug found and fixed after the first real automated run let
     a candidate promote with zero comparison, because both cases were
     originally treated as "skip the check"):
     no bundle exists at all -> genuinely nothing to compare, skip is safe
     (first-ever promotion for an alias). A bundle EXISTS but its manifest
     has no metrics recorded (e.g. a legacy bundle predating this field)
     -> REFUSE by default — a real served model is out there, we just
     don't know its numbers, and silently letting anything through
     uncompared is exactly the failure mode this gate exists to prevent.

     --force-baseline deliberately bootstraps this second case ONLY: it
     skips the regression comparison (checks 1-4 above still run
     normally), prints a loud warning, and records
     baseline_bootstrapped=true permanently in the promoted manifest —
     never silent, never the default. It has NO effect if the served
     bundle already has real metrics; a real comparison is always enforced
     when one is actually possible. Use it exactly once per genuine gap
     (e.g. right after this project's own metrics field was added and
     nothing had been re-registered yet) — every promotion after that
     should have a real baseline again and never need the flag.

     Default tolerances: 0.03 (top1_accuracy) / 0.015 (spearman_corr),
     both measured on THIS gate's own val-split comparison (see above —
     not calibrated against the test-split 0.749 figure at all). The
     2022-2023 validation split is ~44 races, so one race
     flipping from a correct to incorrect top-1 pick moves top1_accuracy by
     ~1/44 ≈ 0.023 — the default tolerance covers a little more than one
     race of sampling noise without masking a real regression. A genuine
     regression investigated during this gate's development
     moved spearman_corr by 0.017-0.021 on the 2024 test split; 0.015 is
     carried over from that as a same-order-of-magnitude heuristic for the
     val-split comparison actually performed here — it has not been
     independently validated against real val-split regression sizes.

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

from src.features.metadata import EXCLUDED_FROM_TRAINING, FEATURE_GROUPS  # noqa: E402
from src.features.pipeline import FEATURES_PATH, MASTER_DATASET_PATH  # noqa: E402
from src.models.evaluate import evaluate_all  # noqa: E402
from src.models.predict import predict_race  # noqa: E402
from src.models.registry import training_schema  # noqa: E402
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
#: Third-party ML library modules a deployment target may support serving.
#: Permissive by default — no specific
#: deployment is constrained to a subset yet; this exists so ONE becomes
#: constrainable later (e.g. a minimal sklearn-only Lambda) without any
#: candidate silently promoting into an environment that can't unpickle it.
DEFAULT_ALLOWED_MODEL_MODULES = ("sklearn", "xgboost", "lightgbm")


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


def _underlying_estimator(model):
    """The actual fitted estimator a candidate wraps: CalibratedModel's own
    base_pipeline's 'model' step, or a bare zoo Pipeline's 'model' step."""
    pipeline = getattr(model, "base_pipeline", model)
    return pipeline.named_steps["model"]


def check_model_class(model, allowed_modules: set[str]) -> None:
    """Refuse a candidate whose fitted estimator comes from a third-party
    library outside the deployment target's declared allowed set — e.g. an
    XGBoost/LightGBM candidate promoted while the serving environment only
    has scikit-learn installed. Without this, that mismatch surfaces as an
    unpickle-time ImportError at the deployed Lambda's cold start instead of
    here, at promotion time, where it's cheap and gets a clear message.

    Project-native estimators (module prefix 'src', e.g. PoleSitterBaseline)
    are always allowed — they carry no third-party dependency beyond what
    every deployment target already needs (numpy/pandas), so there is
    nothing environment-specific to check.
    """
    estimator = _underlying_estimator(model)
    module_root = type(estimator).__module__.split(".")[0]
    if module_root == "src":
        return
    if module_root not in allowed_modules:
        raise PromotionRefused(
            f"Candidate model class {type(estimator).__name__} (module "
            f"'{module_root}') is not supported by this deployment target's "
            f"allowed model modules {sorted(allowed_modules)} — install "
            f"{module_root} there, or choose a different candidate."
        )


def check_excluded_features(
    model, excluded_groups: tuple[str, ...] = EXCLUDED_FROM_TRAINING,
) -> None:
    """Refuse a candidate whose OWN recorded training schema
    (`training_schema()` reads the fitted ColumnGuard's actual recorded
    columns, never a repository constant) contains any feature belonging
    to a CURRENTLY-excluded `FEATURE_GROUPS` group.

    This is the last line of defense: `to_xy()`/`get_model()` default to
    the exclusion-applied
    feature set, so a normal training run can't reintroduce
    an excluded group by accident — but this check catches it anyway if a
    candidate was registered some other way (e.g. an explicit
    `feature_columns=FEATURE_COLUMNS` override used carelessly), the same
    defense-in-depth spirit as `check_model_class`.
    """
    excluded_features = {f for g in excluded_groups for f in FEATURE_GROUPS[g]}
    trained_features = set(training_schema(model)["feature_names"])
    present = trained_features & excluded_features
    if present:
        raise PromotionRefused(
            f"Candidate was trained on {len(present)} feature(s) belonging to "
            f"a currently-excluded group: {sorted(present)} (excluded groups: "
            f"{sorted(excluded_groups)}, see src/features/metadata.py's "
            "EXCLUDED_FROM_TRAINING). Retrain via the default (no explicit "
            "feature_columns override) unless the exclusion itself is being "
            "deliberately, carefully changed."
        )


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
    """Check 3 (part 1): fresh evaluate_all() on split.val — never split.test.
    Mirrors register_model()'s own manifest-metrics logic."""
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
                     "version to the live serving bundle.")
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
    parser.add_argument("--allowed-model-modules",
                        default=",".join(DEFAULT_ALLOWED_MODEL_MODULES),
                        help="Comma-separated third-party ML modules this "
                             "deployment target supports (default: "
                             f"{','.join(DEFAULT_ALLOWED_MODEL_MODULES)} — "
                             "permissive, no live deployment is constrained "
                             "yet). E.g. 'sklearn' for a minimal Lambda "
                             "build with no xgboost/lightgbm installed.")
    parser.add_argument("--force-baseline", action="store_true",
                        help="Bootstrap: allow promotion when the served "
                             "bundle exists but has NO metrics recorded, "
                             "skipping ONLY the regression comparison (the "
                             "model-class and smoke checks still run "
                             "normally). Has NO effect if the served bundle "
                             "already has real metrics — regression is "
                             "still enforced in that case exactly as "
                             "without this flag. Recorded permanently in "
                             "the promoted bundle's own manifest.json "
                             "(baseline_bootstrapped: true) so this is "
                             "never silent. See PromotionRefused's message "
                             "for when this is needed.")
    args = parser.parse_args(argv)
    allowed_model_modules = {m.strip() for m in args.allowed_model_modules.split(",") if m.strip()}

    if not args.features_source.exists():
        print(f"ERROR: {args.features_source} not found — run "
              "`python -m src.features.pipeline` or pass --features-source.",
              file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    bootstrapped = False

    try:
        version = resolve_version(client, args.version)
        model, mv = load_candidate(client, version)
        print(f"Candidate: {REGISTERED_MODEL_NAME} v{version} "
              f"({type(model).__name__}, run {mv.run_id})")

        check_model_class(model, allowed_model_modules)
        print(f"[1/4] Model-class check PASS "
              f"(allowed: {sorted(allowed_model_modules)})")

        check_excluded_features(model)
        print(f"[2/4] Excluded-feature check PASS "
              f"(excluded groups: {sorted(EXCLUDED_FROM_TRAINING) or 'none'})")

        features = pd.read_parquet(args.features_source)
        check_schema_and_predictions(model, features)
        print(f"[3/4] Smoke checks (load, schema, non-degenerate on "
              f"{SMOKE_RACE_SAMPLE} real races) PASS")

        split = temporal_split(features)
        position_order = load_position_order() if MASTER_DATASET_PATH.exists() else None
        candidate_metrics = compute_candidate_metrics(model, split, position_order)

        bundle_dir = bundle_dir_for_alias(args.alias, args.bundle_root)
        bundle_exists = True
        try:
            _, served_info = load_bundle(bundle_dir)
            served_metrics = served_info.metrics
        except FileNotFoundError:
            bundle_exists = False
            served_metrics = {}

        if not bundle_exists:
            # Genuinely nothing to compare against — the only case where
            # skipping the regression check is safe. Distinct from the
            # elif below: a bundle that EXISTS but has no metrics recorded
            # is NOT the same as no baseline at all, and must not be
            # treated as one (see that branch's comment).
            print(f"No existing bundle at {bundle_dir} — first-ever "
                  "promotion for this alias, regression check skipped.")
        elif not served_metrics:
            # A real, currently-served model IS out there — we just don't
            # have its metrics on record (e.g. a legacy bundle exported
            # before this manifest field existed). Silently
            # skipping here would mean literally any candidate, however
            # bad, sails through uncompared — exactly what let a candidate
            # promote uncompared the first time this gate ran for
            # real. Refuse UNLESS the operator
            # explicitly opts into bootstrapping via --force-baseline — no
            # metrics baseline means no comparison is possible, and that
            # must be a deliberate, loud, recorded choice, never a default.
            if not args.force_baseline:
                raise PromotionRefused(
                    f"Bundle at {bundle_dir} exists but its manifest has no "
                    "recorded metrics — cannot check for regression against "
                    "an unknown baseline. Either establish a real baseline "
                    "first (re-register the currently-served model so its "
                    "metrics get recorded) or, to deliberately bootstrap "
                    "one now, re-run with --force-baseline."
                )
            bootstrapped = True
            print(f"[4/4] WARNING: --force-baseline used — bundle at "
                  f"{bundle_dir} had no recorded metrics, so the regression "
                  "check was SKIPPED, not passed. This candidate's own "
                  "metrics become the new baseline for future promotions. "
                  "Recorded as baseline_bootstrapped=true in the promoted "
                  "manifest.")
        else:
            # A real baseline exists — --force-baseline has NO effect here.
            # It means "allow bootstrapping when there's nothing to compare
            # against", not "skip the check on request"; a real comparison
            # is always enforced when one is actually possible.
            check_regression(candidate_metrics, served_metrics,
                             args.top1_tolerance, args.spearman_tolerance)
        if not bootstrapped:
            print(f"[4/4] Metric non-regression PASS "
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
        baseline_bootstrapped=bootstrapped,
    )
    bundle_dir = export_bundle(model, info, bundle_root=args.bundle_root)
    export_features_snapshot(args.features_source, artifacts_root=args.artifacts_root)
    bootstrap_note = " (baseline_bootstrapped=true — no regression check was performed)" if bootstrapped else ""
    print(f"PROMOTED {REGISTERED_MODEL_NAME} v{version} to @{args.alias} -> {bundle_dir}{bootstrap_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
