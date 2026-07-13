"""
src/models/train.py

Training orchestration.

    python -m src.models.train                              # stage 1: all candidates
    python -m src.models.train --model xgboost              # one candidate
    python -m src.models.train --model xgboost --tune       # stage 2: randomized search
    python -m src.models.train --model xgboost --final-test # ONE-TIME 2024 evaluation
    python -m src.models.train --model xgboost --register Staging|Production

Orchestration only — splitting lives in splits.py, model definitions in
registry.py, metrics in evaluate.py. This module wires them together and
talks to MLflow.

MLflow layout: experiment `f1-winner-prediction`; one
parent run per candidate training invocation with one child run per CV fold
(tune mode logs per-fold metrics on the config's parent run instead of
spawning 240 child runs). Tags carry ModelSpec.to_metadata() plus a data
fingerprint (row count + file hash of features.parquet) so every result is
attributable to a dataset version. Artifacts: the fitted pipeline
(mlflow.sklearn), the training schema recorded by ColumnGuard, per-season
evaluation, calibration table + plot, and feature importances grouped by
era-robustness class.

Tracking store: defaults to `sqlite:///mlflow.db` in the project root — a
local store, chosen over the bare `mlruns/` file store because the MLflow
Model Registry (needed by --register / predict.py) does not work on the
file store. Artifacts still land in ./mlruns.

Leakage guards implemented here:
- Test-set discipline: ONLY final_test() ever touches split.test, only
  behind the explicit --final-test flag, and it tags the run final=true.
  train_candidate()/tune_candidate() are structurally unable to see test
  rows — their signatures take train/val frames only.
- Fitted-state containment: a fresh pipeline is built and fit inside
  every fold, so imputer/scaler statistics never see the fold's future.
- Too-good-to-be-true tripwire: any per-race top-1 above
  TOP1_TRIPWIRE (0.70) emits a loud warning demanding investigation —
  pole predicts ~50% in-window; >70% smells like leakage, not brilliance.
(A shuffled-target canary — fitting on a permuted target should destroy all
signal — is a test in tests/test_train.py.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from src.features.metadata import (
    EXCLUDED_FROM_TRAINING,
    FEATURE_CLASSIFICATION,
    active_feature_columns,
)
from src.features.pipeline import FEATURES_PATH, MASTER_DATASET_PATH
from src.models.evaluate import (
    calibration_table,
    evaluate_all,
    evaluate_by_season,
    per_race_table,
)
from src.models.registry import MODEL_ZOO, get_model, training_schema
from src.models.splits import (
    DEFAULT_N_FOLDS,
    TemporalSplit,
    season_folds,
    temporal_split,
    to_xy,
)

EXPERIMENT_NAME = "f1-winner-prediction"
REGISTERED_MODEL_NAME = "f1-winner"
# Anchored to the project root so the CLI produces ONE tracking store no
# matter what directory it is invoked from (a CWD-relative sqlite path would
# silently split experiment history across stray mlflow.db files).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{(_PROJECT_ROOT / 'mlflow.db').as_posix()}"
TOP1_TRIPWIRE = 0.70           # >70% top-1 is a leakage alarm, not a win
DEFAULT_TUNE_ITER = 40
SEED = 42
#: Shared source of truth for retrain hyperparameters —
#: {"model", "calibrate", "params"}. Read by the manual `--params-file` CLI
#: flag below AND by scripts/refresh_and_freeze.py's automated retrain path,
#: so both cannot silently drift onto different configs. Update this file
#: (after actually re-running --tune) whenever a real retune picks a new
#: config — do not hand-edit the params without re-tuning.
DEFAULT_PARAMS_CONFIG_PATH = _PROJECT_ROOT / "config" / "registered_model_params.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_fingerprint(path: Path = FEATURES_PATH) -> str:
    """`<rows>rows-<sha256[:12]>` of features.parquet, for run attribution."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    n_rows = len(pd.read_parquet(path, columns=["raceId"]))
    return f"{n_rows}rows-{digest}"


def load_position_order(master_path: Path = MASTER_DATASET_PATH) -> pd.Series:
    """
    `positionOrder` indexed by (raceId, driverId), for the Spearman metric
    (evaluate.spearman_rank_correlation). This is an EVALUATION-TIME join
    from master_dataset.parquet, never a model feature — positionOrder stays
    excluded from FEATURE_COLUMNS via POST_RACE_OUTCOME_COLUMNS regardless.
    """
    master = pd.read_parquet(master_path, columns=["raceId", "driverId", "positionOrder"])
    return master.set_index(["raceId", "driverId"])["positionOrder"]


def _align_position_order(df: pd.DataFrame, position_order: pd.Series | None) -> np.ndarray | None:
    """Reindex `position_order` to df's (raceId, driverId) row order."""
    if position_order is None:
        return None
    index = pd.MultiIndex.from_frame(df[["raceId", "driverId"]])
    return position_order.reindex(index).to_numpy(dtype=float)


def load_registered_model_config(path: Path = DEFAULT_PARAMS_CONFIG_PATH) -> dict:
    """Read {"model", "calibrate", "params"} from the shared retrain-config
    file — the source both the manual `--params-file`
    CLI flag and scripts/refresh_and_freeze.py's automated retrain read, so
    a real re-tune's chosen config only needs updating in one place."""
    return json.loads(path.read_text())


def check_tripwire(metrics: dict[str, float], context: str) -> None:
    """Top-1 accuracy above 70% is a leakage alarm, not a win — see the module docstring."""
    top1 = metrics.get("top1_accuracy", 0.0)
    if top1 > TOP1_TRIPWIRE:
        warnings.warn(
            f"TRIPWIRE ({context}): per-race top-1 = {top1:.1%} exceeds "
            f"{TOP1_TRIPWIRE:.0%}. The pole baseline is ~50% in-window; this "
            "result is more likely leakage than brilliance — investigate "
            "before trusting or registering anything.",
            stacklevel=2,
        )


def _fit_and_score(
    pipeline: Pipeline, train_df: pd.DataFrame, eval_df: pd.DataFrame,
    position_order: pd.Series | None = None,
) -> tuple[Pipeline, dict[str, float], np.ndarray]:
    """Fit on train_df, evaluate on eval_df; returns (pipeline, metrics, probs).

    `position_order` (see load_position_order), if given, is aligned to
    eval_df's row order and threaded into evaluate_all for the Spearman
    metric.
    """
    X_tr, y_tr, _ = to_xy(train_df)
    X_ev, y_ev, races_ev = to_xy(eval_df)
    pipeline.fit(X_tr, y_tr)
    y_prob = pipeline.predict_proba(X_ev)[:, 1]
    po = _align_position_order(eval_df, position_order)
    return pipeline, evaluate_all(y_ev, y_prob, races_ev, position_order=po), y_prob


def run_cv(
    name: str,
    train_df: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    params: dict | None = None,
) -> tuple[list[dict], list[Pipeline]]:
    """
    Expanding-window CV for one candidate over the training split.

    A FRESH pipeline is built per fold — with class weights computed from
    that fold's own training target and any imputer/scaler fit inside the
    fold. Returns per-fold metric dicts (with 'val_year') and
    the fitted fold pipelines (kept for tests and diagnostics).
    """
    fold_metrics: list[dict] = []
    fold_pipelines: list[Pipeline] = []
    for fold in season_folds(train_df, n_folds=n_folds):
        _, y_tr, _ = to_xy(fold.train)
        pipeline = get_model(name, y_tr)
        if params:
            pipeline.set_params(**params)
        pipeline, metrics, _ = _fit_and_score(pipeline, fold.train, fold.val)
        fold_metrics.append({**metrics, "val_year": fold.val_year, "fold": fold.fold})
        fold_pipelines.append(pipeline)
    return fold_metrics, fold_pipelines


def _cv_aggregate(fold_metrics: list[dict]) -> dict[str, float]:
    """cv_{metric}_mean/std across folds (excluding bookkeeping keys)."""
    skip = {"val_year", "fold", "n_races", "n_rows"}
    out: dict[str, float] = {}
    for key in fold_metrics[0]:
        if key in skip:
            continue
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        out[f"cv_{key}_mean"] = float(values.mean())
        out[f"cv_{key}_std"] = float(values.std())
    return out


# ---------------------------------------------------------------------------
# Artifact logging
# ---------------------------------------------------------------------------

def feature_importance_frame(pipeline: Pipeline) -> pd.DataFrame | None:
    """
    Importances with era-robustness classes. Uses the preprocessing steps'
    get_feature_names_out so imputer missing-indicator columns are named and
    mapped back to their base feature's class. Returns None for estimators
    with no importance concept (the pole heuristic).
    """
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        return None

    names = list(pipeline[:-1].get_feature_names_out())
    frame = pd.DataFrame({"feature": names, "importance": values})
    base = frame["feature"].str.replace("missingindicator_", "", regex=False)
    frame["is_missing_indicator"] = frame["feature"].str.startswith("missingindicator_")
    frame["feature_class"] = base.map(FEATURE_CLASSIFICATION).fillna("derived")
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _log_evaluation_artifacts(
    pipeline: Pipeline, y_true, y_prob, race_ids, years, prefix: str,
    position_order=None,
) -> None:
    """Per-season CSV, per-race CSV, calibration table + plot, importances.

    `position_order`, if given, adds the Spearman metric/column to both CSVs
    (see evaluate.evaluate_by_season / per_race_table).
    """
    mlflow.log_text(
        evaluate_by_season(y_true, y_prob, race_ids, years, position_order=position_order).to_csv(),
        f"{prefix}/metrics_by_season.csv",
    )
    mlflow.log_text(
        per_race_table(y_true, y_prob, race_ids, position_order=position_order).to_csv(index=False),
        f"{prefix}/per_race_metrics.csv",
    )

    cal = calibration_table(y_true, y_prob)
    mlflow.log_text(cal.to_csv(index=False), f"{prefix}/calibration_table.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(cal["mean_predicted"], cal["fraction_positive"], "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of winners")
    ax.set_title("Calibration (reliability diagram)")
    ax.legend()
    mlflow.log_figure(fig, f"{prefix}/calibration_plot.png")
    plt.close(fig)

    importances = feature_importance_frame(pipeline)
    if importances is not None:
        mlflow.log_text(
            importances.to_csv(index=False), f"{prefix}/feature_importance.csv"
        )
        by_class = (
            importances.groupby("feature_class")["importance"].sum().sort_values(ascending=False)
        )
        mlflow.log_text(by_class.to_csv(), f"{prefix}/importance_by_class.csv")


def _log_common(name: str, fingerprint: str, stage: str) -> None:
    spec = MODEL_ZOO[name]
    active_cols = active_feature_columns()
    mlflow.set_tags({
        "model_family": spec.family,
        "stage": stage,
        "data_fingerprint": fingerprint,
        "feature_count": str(len(FEATURE_CLASSIFICATION)),
        "code_phase": "phase4",
        # Which FEATURE_GROUPS this run's design matrix
        # actually excluded, and how many columns that left — the
        # queryable, human-readable side of reproducibility. The
        # ground-truth side is training_schema()'s own recorded column
        # list (already logged as training_schema.json by every caller
        # of this function), which is immune to a later edit of what
        # EXCLUDED_FROM_TRAINING means.
        "excluded_feature_groups": ",".join(EXCLUDED_FROM_TRAINING) or "none",
        "active_feature_count": str(len(active_cols)),
        **{f"spec_{k}": str(v) for k, v in spec.to_metadata().items()
           if k not in ("description", "tuned_params")},
    })
    mlflow.log_param("seed", SEED)


# ---------------------------------------------------------------------------
# Stage 1 — train one candidate with zoo defaults
# ---------------------------------------------------------------------------

def train_candidate(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fingerprint: str = "unfingerprinted",
    n_folds: int = DEFAULT_N_FOLDS,
    params: dict | None = None,
    stage: str = "default",
    position_order: pd.Series | None = None,
) -> dict[str, float]:
    """
    CV on the training split -> refit on the full training split -> score
    validation. Logs one MLflow parent run with per-fold child runs. Never
    sees test data — the signature makes it impossible.
    Returns the logged metrics (cv_* aggregates + val_*).

    `position_order` (see load_position_order), if given, adds val_* Spearman
    reporting only — CV fold metrics (run_cv) deliberately do not thread it
    through, to keep this addition contained to the val/test reporting the
    metric was asked for.
    """
    with mlflow.start_run(run_name=f"{name}-{stage}"):
        _log_common(name, fingerprint, stage)
        if params:
            mlflow.log_params(params)

        fold_metrics, _ = run_cv(name, train_df, n_folds=n_folds, params=params)
        for fm in fold_metrics:
            with mlflow.start_run(run_name=f"{name}-fold{fm['fold']}", nested=True):
                mlflow.set_tags({"val_year": str(fm["val_year"]), "stage": "cv-fold"})
                mlflow.log_metrics(
                    {k: v for k, v in fm.items() if k not in ("fold", "val_year")}
                )

        cv_agg = _cv_aggregate(fold_metrics)

        _, y_tr, _ = to_xy(train_df)
        pipeline = get_model(name, y_tr)
        if params:
            pipeline.set_params(**params)
        pipeline, val_metrics, y_prob = _fit_and_score(
            pipeline, train_df, val_df, position_order=position_order
        )
        val_named = {f"val_{k}": v for k, v in val_metrics.items()}

        mlflow.log_metrics({**cv_agg, **val_named})
        mlflow.log_dict(training_schema(pipeline), "training_schema.json")
        mlflow.sklearn.log_model(pipeline, name="model")

        _, y_val, races_val = to_xy(val_df)
        _log_evaluation_artifacts(
            pipeline, y_val, y_prob, races_val, val_df["year"], prefix="val",
            position_order=_align_position_order(val_df, position_order),
        )

        check_tripwire(val_metrics, context=f"{name} validation")
        check_tripwire({"top1_accuracy": cv_agg["cv_top1_accuracy_mean"]},
                       context=f"{name} CV mean")

        return {**cv_agg, **val_named}


# ---------------------------------------------------------------------------
# Stage 2 — randomized search over a candidate's declared distributions
# ---------------------------------------------------------------------------

def tune_candidate(
    name: str,
    train_df: pd.DataFrame,
    fingerprint: str = "unfingerprinted",
    n_iter: int = DEFAULT_TUNE_ITER,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = SEED,
) -> tuple[dict, dict[str, float]]:
    """
    Randomized search: sample n_iter configs from the
    zoo's declared distributions; score each by expanding-window CV.
    Selection statistic: mean CV per-race top-1, ties broken by mean CV
    log-loss. Each config is one MLflow run tagged stage=tune (per-fold
    metrics logged flat on the run, not as child runs, to keep the run count
    sane). Validation data is deliberately NOT an argument — the val split
    arbitrates finalists via train_candidate afterwards, it is not searched
    against.

    Returns (best_params, best_cv_metrics).
    """
    spec = MODEL_ZOO[name]
    if not spec.param_distributions:
        raise ValueError(f"'{name}' declares no tunable distributions.")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}.")

    best_params: dict = {}
    best_metrics: dict[str, float] = {}
    best_key: tuple[float, float] | None = None

    sampler = ParameterSampler(spec.param_distributions, n_iter=n_iter, random_state=seed)
    for i, params in enumerate(sampler, start=1):
        with mlflow.start_run(run_name=f"{name}-tune-{i:03d}"):
            _log_common(name, fingerprint, stage="tune")
            mlflow.log_params(params)
            fold_metrics, _ = run_cv(name, train_df, n_folds=n_folds, params=params)
            cv_agg = _cv_aggregate(fold_metrics)
            for fm in fold_metrics:
                mlflow.log_metric(f"top1_fold_{fm['val_year']}", fm["top1_accuracy"])
            mlflow.log_metrics(cv_agg)

            # Maximize top-1; minimize log-loss on ties.
            key = (cv_agg["cv_top1_accuracy_mean"], -cv_agg["cv_log_loss_mean"])
            if best_key is None or key > best_key:
                best_key, best_params, best_metrics = key, params, cv_agg
                mlflow.set_tag("tune_best_so_far", "true")

    check_tripwire(
        {"top1_accuracy": best_metrics["cv_top1_accuracy_mean"]},
        context=f"{name} tuning best",
    )
    return best_params, best_metrics


# ---------------------------------------------------------------------------
# Final test (guarded) and registration
# ---------------------------------------------------------------------------

def final_test(
    name: str,
    split: TemporalSplit,
    fingerprint: str = "unfingerprinted",
    params: dict | None = None,
    position_order: pd.Series | None = None,
) -> dict[str, float]:
    """
    THE one-time 2024 evaluation. Fits the
    selected configuration on the training split (pre-refit)
    and scores the test split once. Tags the run final=true. Callable only
    from the --final-test CLI path — nothing else in this module touches
    split.test.

    `position_order` (see load_position_order), if given, adds test_*
    Spearman reporting.
    """
    with mlflow.start_run(run_name=f"{name}-FINAL-TEST"):
        _log_common(name, fingerprint, stage="final-test")
        mlflow.set_tag("final", "true")
        if params:
            mlflow.log_params(params)

        _, y_tr, _ = to_xy(split.train)
        pipeline = get_model(name, y_tr)
        if params:
            pipeline.set_params(**params)
        pipeline, test_metrics, y_prob = _fit_and_score(
            pipeline, split.train, split.test, position_order=position_order
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_dict(training_schema(pipeline), "training_schema.json")

        _, y_te, races_te = to_xy(split.test)
        _log_evaluation_artifacts(
            pipeline, y_te, y_prob, races_te, split.test["year"], prefix="test",
            position_order=_align_position_order(split.test, position_order),
        )
        check_tripwire(test_metrics, context=f"{name} FINAL TEST")
        return {f"test_{k}": v for k, v in test_metrics.items()}


def register_model(
    name: str,
    split: TemporalSplit,
    alias: str,
    fingerprint: str = "unfingerprinted",
    params: dict | None = None,
    calibrate: bool = False,
    bundle_root: Path | None = None,
    features_source: Path = FEATURES_PATH,
    artifacts_root: Path | None = None,
    position_order: pd.Series | None = None,
    export: bool = True,
) -> str:
    """
    Register a fitted pipeline as `f1-winner` and point `alias` at it.

    alias="Staging": fit on train only (pre-test finalist).
    alias="Production": fit on train+val 2010-2023 (the post-test refit —
    the registered model must not ignore the two most
    recent completed seasons it was validated on).

    calibrate=True: wrap the fitted pipeline in the OOF isotonic
    calibrator. The calibrator is ALWAYS learned from
    the training split's season folds (calibration.py enforces this), even
    when the base model is refit on train+val for Production.

    Requires a registry-capable tracking store (the default sqlite URI is).

    The manifest ALSO records honest evaluation
    metrics (evaluate_all on split.val) — what this bundle's model actually
    scored, not just identity fields. Scored on a model fit on split.train
    ONLY, regardless of alias: for alias="Staging" that's model_obj itself
    (fit_df == split.train, so scoring it on val is safe); for
    alias="Production" (fit_df == train+val) model_obj has already seen val,
    so scoring IT on val would leak — a throwaway train-only refit is used
    for the metrics instead, never touching split.test.
    `position_order` (see load_position_order), if given, adds the Spearman
    metric to that dict, same convention as train_candidate/final_test.

    Registration ALSO exports a frozen serving bundle
    (src.models.serving_bundle) to bundle_root/alias.lower() — the SAME
    already-fitted in-memory model_obj, no extra MLflow round trip. This is
    what src/models/predict.py's serving-side load_model() reads; the API
    never resolves a live registry alias at request time. bundle_root
    defaults to artifacts/serving/ — TESTS MUST pass an explicit tmp
    bundle_root or they will write into the real project directory.

    NOTE: this export is UNCHECKED — it overwrites
    whatever bundle_root/alias.lower() currently holds with zero validation.
    For a real promotion (as opposed to a routine registration during
    development), use `scripts/promote_model.py` instead, which gates the
    same export_bundle() call behind smoke checks and a metric-regression
    comparison against the currently-served bundle's manifest. Calling
    register_model() directly still works — it's what promote_model.py's
    own candidates are registered with — but bypasses that gate.

    export=False: skip both the bundle export AND the
    features snapshot freeze — only create the MLflow version/alias, touch
    nothing under artifacts/. For automated pipelines (scripts/
    refresh_and_freeze.py's scheduled-workflow path) that route the
    resulting version through promote_model.py afterward: with the default
    export=True, this function would ALREADY have overwritten
    artifacts/serving/ before promote_model.py's gate ever runs, making the
    gate a no-op. Manual/interactive use (the CLI's --register, ad-hoc
    exploratory retrain sessions) keeps the default True — a human is
    already about to inspect the result.

    Registration ALSO freezes a runtime features snapshot
    (src.models.serving_bundle.export_features_snapshot) by copying
    features_source (default: the training pipeline's own
    data/processed/features.parquet — the file `split` was itself built
    from in the normal CLI flow) to artifacts_root/features.parquet. This
    is the file app/config.py's Settings.features_path reads by default —
    the deployed API never reads data/processed/features.parquet directly.
    artifacts_root defaults to the project's artifacts/ — TESTS MUST pass an
    explicit tmp artifacts_root (and features_source) or they will read/write
    the real project's files.

    Returns the registered model version.
    """
    if alias == "Production":
        fit_df = pd.concat([split.train, split.val], ignore_index=True)
    elif alias == "Staging":
        fit_df = split.train
    else:
        raise ValueError(f"Unknown alias '{alias}' — use 'Staging' or 'Production'.")

    with mlflow.start_run(run_name=f"{name}-register-{alias.lower()}") as run:
        _log_common(name, fingerprint, stage="register")
        mlflow.set_tag("register_alias", alias)
        if params:
            mlflow.log_params(params)

        if calibrate:
            from src.models.calibration import fit_calibrated_model
            model_obj = fit_calibrated_model(
                name, train_df=split.train, fit_df=fit_df, params=params,
            )
        else:
            X_fit, y_fit, _ = to_xy(fit_df)
            model_obj = get_model(name, y_fit)
            if params:
                model_obj.set_params(**params)
            model_obj.fit(X_fit, y_fit)
        calibration = getattr(model_obj, "calibration", "none")
        mlflow.set_tag("calibration", calibration)

        mlflow.log_dict(training_schema(model_obj), "training_schema.json")
        log_info = mlflow.sklearn.log_model(
            model_obj, name="model", registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    version = log_info.registered_model_version
    mlflow.MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, alias, version
    )

    if alias == "Staging":
        eval_model = model_obj
    elif calibrate:
        from src.models.calibration import fit_calibrated_model
        eval_model = fit_calibrated_model(name, train_df=split.train, fit_df=split.train, params=params)
    else:
        X_tr, y_tr, _ = to_xy(split.train)
        eval_model = get_model(name, y_tr)
        if params:
            eval_model.set_params(**params)
        eval_model.fit(X_tr, y_tr)

    X_val, y_val, races_val = to_xy(split.val)
    y_prob_val = eval_model.predict_proba(X_val)[:, 1]
    metrics = evaluate_all(
        y_val, y_prob_val, races_val,
        position_order=_align_position_order(split.val, position_order),
    )

    if export:
        from src.models.serving_bundle import (
            ModelInfo,
            export_bundle,
            export_features_snapshot,
        )
        bundle_info = ModelInfo(
            name=REGISTERED_MODEL_NAME,
            version=str(version),
            alias=alias,
            run_id=run_id,
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
            calibration=calibration,
            model_class=type(model_obj).__name__,
            metrics=metrics,
        )
        export_bundle(model_obj, bundle_info, bundle_root=bundle_root)
        export_features_snapshot(features_source, artifacts_root=artifacts_root)

    return str(version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model training orchestration.")
    parser.add_argument("--model", default="all",
                        help=f"Zoo candidate or 'all'. Available: {sorted(MODEL_ZOO)}")
    parser.add_argument("--tune", action="store_true",
                        help="Stage-2 randomized search (single --model only).")
    parser.add_argument("--n-iter", type=int, default=DEFAULT_TUNE_ITER)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--final-test", action="store_true",
                        help="ONE-TIME 2024 evaluation of a single --model.")
    parser.add_argument("--register", choices=["Staging", "Production"],
                        help="Register a single --model under this alias.")
    parser.add_argument("--calibrate", action="store_true",
                        help="With --register: wrap the model in the OOF "
                             "isotonic calibrator before registering.")
    parser.add_argument("--bundle-root", type=Path, default=None,
                        help="With --register: where to export the frozen "
                             "serving bundle. "
                             "Default: artifacts/serving/.")
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help="With --register: where to freeze the runtime "
                             "features snapshot. Default: "
                             "artifacts/ (writes artifacts/features.parquet).")
    parser.add_argument("--no-export", action="store_true",
                        help="With --register: create the MLflow version/"
                             "alias only — skip the (unchecked) bundle/"
                             "features-snapshot export. "
                             "For automated pipelines that gate the export "
                             "through scripts/promote_model.py afterward; "
                             "manual use should omit this flag.")
    parser.add_argument("--params", default=None,
                        help="JSON dict of pipeline params (copy from --tune "
                             "output) applied to --final-test / --register / a "
                             "single-model run — e.g. '{\"model__C\": 0.5}'. "
                             "Without this, the selected configuration cannot "
                             "reach the test/registration steps.")
    parser.add_argument("--params-file", type=Path, default=None,
                        help="Read the 'params' dict from a JSON file "
                             "instead of inlining it on the command line — "
                             f"e.g. --params-file {DEFAULT_PARAMS_CONFIG_PATH} "
                             "(the shared retrain-config "
                             "source of truth). Mutually exclusive with "
                             "--params.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = parser.parse_args(argv)

    single_model_actions = args.tune or args.final_test or args.register
    if single_model_actions and args.model == "all":
        parser.error("--tune / --final-test / --register need an explicit --model.")
    if args.params and args.params_file:
        parser.error("--params and --params-file are mutually exclusive.")
    if (args.params or args.params_file) and args.tune:
        parser.error("--params/--params-file cannot be combined with --tune "
                     "(tuning produces params).")
    if (args.params or args.params_file) and args.model == "all":
        parser.error("--params/--params-file needs an explicit --model.")
    if args.calibrate and not args.register:
        parser.error("--calibrate is only valid together with --register "
                     "(the calibrated wrapper is a registration-time artifact).")

    params: dict | None = None
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as exc:
            parser.error(f"--params is not valid JSON: {exc}")
        if not isinstance(params, dict):
            parser.error("--params must be a JSON object of pipeline params.")
    elif args.params_file:
        if not args.params_file.exists():
            parser.error(f"--params-file {args.params_file} not found.")
        try:
            params = json.loads(args.params_file.read_text())["params"]
        except (json.JSONDecodeError, KeyError) as exc:
            parser.error(f"--params-file {args.params_file} must be JSON with "
                         f"a 'params' key: {exc}")
        if not isinstance(params, dict):
            parser.error(f"--params-file {args.params_file}'s 'params' must be a JSON object.")

    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found — run `python -m src.features.pipeline`.",
              file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    split = temporal_split(pd.read_parquet(FEATURES_PATH))
    fingerprint = data_fingerprint()
    print(f"Data: {fingerprint} | train {len(split.train)} / val {len(split.val)} "
          f"/ test {len(split.test)} rows")

    # positionOrder for the Spearman metric — an evaluation-time join from
    # master_dataset.parquet, not a feature. Optional: data/ is gitignored
    # and legitimately absent in some environments (fresh
    # clones, CI), so degrade to no Spearman reporting rather than error.
    position_order = (
        load_position_order() if MASTER_DATASET_PATH.exists() else None
    )
    if position_order is None:
        print(f"NOTE: {MASTER_DATASET_PATH} not found — Spearman metric skipped.")

    if args.final_test:
        metrics = final_test(args.model, split, fingerprint, params=params,
                             position_order=position_order)
        print(f"FINAL TEST {args.model}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if k != "test_n_rows"))
        return 0

    if args.register:
        # features_source is passed explicitly (rather than relying on
        # register_model's own default) so it resolves the SAME FEATURES_PATH
        # module global `split` was just built from above — including under
        # a test's monkeypatch.setattr("src.models.train.FEATURES_PATH", ...),
        # which a bound default parameter would not see.
        version = register_model(args.model, split, args.register, fingerprint,
                                 params=params, calibrate=args.calibrate,
                                 bundle_root=args.bundle_root,
                                 features_source=FEATURES_PATH,
                                 artifacts_root=args.artifacts_root,
                                 position_order=position_order,
                                 export=not args.no_export)
        suffix = " (isotonic-oof calibrated)" if args.calibrate else ""
        print(f"Registered {REGISTERED_MODEL_NAME} v{version} as "
              f"@{args.register}{suffix}.")
        if args.no_export:
            print("--no-export: bundle/features snapshot NOT touched — "
                  f"promote v{version} via `python scripts/promote_model.py "
                  f"--alias {args.register} --version {version}`.")
        else:
            from src.models.serving_bundle import (
                DEFAULT_FEATURES_ARTIFACT,
                bundle_dir_for_alias,
            )
            bundle_dir = bundle_dir_for_alias(args.register, args.bundle_root)
            features_artifact = (
                (args.artifacts_root / "features.parquet")
                if args.artifacts_root else DEFAULT_FEATURES_ARTIFACT
            )
            print(f"Serving bundle exported to {bundle_dir}")
            print(f"Runtime features snapshot frozen to {features_artifact}")
        return 0

    if args.tune:
        best_params, best_metrics = tune_candidate(
            args.model, split.train, fingerprint,
            n_iter=args.n_iter, n_folds=args.n_folds,
        )
        # Sampled values arrive as numpy scalars — convert so the printed
        # JSON is directly reusable via --params.
        reusable = {k: (v.item() if hasattr(v, "item") else v)
                    for k, v in best_params.items()}
        print(f"Best {args.model} config (cv_top1={best_metrics['cv_top1_accuracy_mean']:.4f}, "
              f"cv_logloss={best_metrics['cv_log_loss_mean']:.4f}):")
        print(f"  --params '{json.dumps(reusable)}'")
        # Score the tuned finalist on validation via the standard path.
        train_candidate(args.model, split.train, split.val, fingerprint,
                        n_folds=args.n_folds, params=best_params, stage="finalist",
                        position_order=position_order)
        return 0

    names = sorted(MODEL_ZOO) if args.model == "all" else [args.model]
    summary = {}
    for name in names:
        metrics = train_candidate(name, split.train, split.val, fingerprint,
                                  n_folds=args.n_folds, params=params,
                                  position_order=position_order)
        summary[name] = metrics
        print(f"{name:<15} cv_top1={metrics['cv_top1_accuracy_mean']:.4f} "
              f"val_top1={metrics['val_top1_accuracy']:.4f} "
              f"val_top3={metrics['val_top3_recall']:.4f} "
              f"val_logloss={metrics['val_log_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
