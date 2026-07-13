"""
src/models/analysis.py

Post-training explainability and cost analysis.

    python -m src.models.analysis --model random_forest                # finalist analysis
    python -m src.models.analysis --model random_forest --params '{...}'
    python -m src.models.analysis --timing                             # all-zoo timing table

Produces, for one candidate (fit on train, analyzed on validation — never
test):
- native feature-importance CSV + bar plot, grouped by era-robustness class
- permutation importance scored by PER-RACE TOP-1 — what
  actually moves the metric we care about, not row AUC
- SHAP global summary (beeswarm + bar), dependence plots for the top
  features, and per-race waterfall plots for 2022-2023 case-study races
  (TreeExplainer for tree families, LinearExplainer for logreg)
- an all-zoo timing table (fit wall-clock on the train split, predict_proba
  latency on the validation split)

Everything is saved under reports/phase4_analysis/ AND logged to MLflow as a
run tagged stage=analysis, so the artifacts are attributable to the same
data fingerprint as the training runs.

Leakage note: analysis uses train (fit) and validation (explain) only. The
test split is not an input to anything here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.features.metadata import FEATURE_CLASSIFICATION
from src.features.pipeline import FEATURES_PATH
from src.models.evaluate import top1_accuracy
from src.models.registry import MODEL_ZOO, get_model
from src.models.splits import temporal_split, to_xy
from src.models.train import (
    DEFAULT_TRACKING_URI,
    EXPERIMENT_NAME,
    SEED,
    data_fingerprint,
    feature_importance_frame,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "phase4_analysis"
N_DEPENDENCE_PLOTS = 6          # top ~6 features
N_PERMUTATION_REPEATS = 5
N_CASE_STUDY_RACES = 3


# ---------------------------------------------------------------------------
# Timing (training cost / inference latency — a tiebreak input for model selection)
# ---------------------------------------------------------------------------

def measure_timing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params_by_model: dict[str, dict] | None = None,
    n_predict_repeats: int = 5,
) -> pd.DataFrame:
    """Fit wall-clock + predict_proba latency for every zoo candidate.

    predict latency is the median of n_predict_repeats runs over the full
    validation split (880 rows ~ 44 races), reported per call and per row.
    """
    params_by_model = params_by_model or {}
    X_tr, y_tr, _ = to_xy(train_df)
    X_val, _, _ = to_xy(val_df)

    records = []
    for name in sorted(MODEL_ZOO):
        pipeline = get_model(name, y_tr)
        if params_by_model.get(name):
            pipeline.set_params(**params_by_model[name])
        t0 = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - t0

        latencies = []
        for _ in range(n_predict_repeats):
            t0 = time.perf_counter()
            pipeline.predict_proba(X_val)
            latencies.append(time.perf_counter() - t0)
        predict_seconds = float(np.median(latencies))

        records.append({
            "model": name,
            "params": json.dumps(params_by_model.get(name, {})),
            "fit_seconds_train_split": round(fit_seconds, 4),
            "predict_seconds_val_split": round(predict_seconds, 4),
            "predict_ms_per_row": round(predict_seconds / len(X_val) * 1e3, 4),
        })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Permutation importance scored by per-race top-1
# ---------------------------------------------------------------------------

def permutation_importance_top1(
    pipeline,
    val_df: pd.DataFrame,
    n_repeats: int = N_PERMUTATION_REPEATS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Drop in per-race top-1 when each feature column is shuffled on val.

    Positive delta = the model NEEDS the feature to rank winners; ~0 = the
    metric does not depend on it (even if native importance is nonzero).
    """
    X_val, y_val, races_val = to_xy(val_df)
    baseline = top1_accuracy(y_val, pipeline.predict_proba(X_val)[:, 1], races_val)

    rng = np.random.default_rng(seed)
    records = []
    for col in X_val.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            score = top1_accuracy(
                y_val, pipeline.predict_proba(X_perm)[:, 1], races_val
            )
            drops.append(baseline - score)
        records.append({
            "feature": col,
            "top1_drop_mean": float(np.mean(drops)),
            "top1_drop_std": float(np.std(drops)),
            "feature_class": FEATURE_CLASSIFICATION.get(col, "derived"),
        })
    out = pd.DataFrame.from_records(records).sort_values(
        "top1_drop_mean", ascending=False
    ).reset_index(drop=True)
    out.attrs["baseline_top1"] = baseline
    return out


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _transformed_matrices(pipeline, X_tr: pd.DataFrame, X_val: pd.DataFrame):
    """Run both splits through the pipeline's preprocessing (all steps but
    the estimator) and return (Xt_tr, Xt_val, names, estimator) as frames."""
    pre = pipeline[:-1]
    names = list(pre.get_feature_names_out())
    Xt_tr = pd.DataFrame(np.asarray(pre.transform(X_tr), dtype=float), columns=names)
    Xt_val = pd.DataFrame(np.asarray(pre.transform(X_val), dtype=float), columns=names)
    return Xt_tr, Xt_val, names, pipeline.named_steps["model"]


def shap_analysis(
    pipeline,
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """SHAP on the validation split; saves plots; returns |SHAP| summary frame.

    TreeExplainer for tree families, LinearExplainer for logreg; the pole
    heuristic has nothing to explain and raises.
    """
    import shap

    family = MODEL_ZOO[name].family
    if family == "heuristic":
        raise ValueError("The pole baseline is a deterministic rule — no SHAP.")

    X_tr, _, _ = to_xy(train_df)
    X_val, y_val, races_val = to_xy(val_df)
    Xt_tr, Xt_val, names, estimator = _transformed_matrices(pipeline, X_tr, X_val)

    if family == "linear":
        masker = shap.maskers.Independent(Xt_tr, max_samples=500)
        explainer = shap.LinearExplainer(estimator, masker)
        shap_values = explainer.shap_values(Xt_val)
    else:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(Xt_val)
        # sklearn RF returns per-class arrays; take the positive class.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 1]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Global summary: beeswarm + mean-|SHAP| bar.
    shap.summary_plot(shap_values, Xt_val, feature_names=names, show=False)
    plt.gcf().suptitle(f"SHAP summary — {name} (validation 2022-2023)")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    shap.summary_plot(shap_values, Xt_val, feature_names=names,
                      plot_type="bar", show=False)
    plt.gcf().suptitle(f"mean |SHAP| — {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
    base = summary["feature"].str.replace("missingindicator_", "", regex=False)
    summary["feature_class"] = base.map(FEATURE_CLASSIFICATION).fillna("derived")
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Dependence plots for the top features.
    for feat in summary["feature"].head(N_DEPENDENCE_PLOTS):
        shap.dependence_plot(
            feat, shap_values, Xt_val, feature_names=names,
            interaction_index=None, show=False,
        )
        plt.title(f"SHAP dependence — {feat} ({name})")
        plt.tight_layout()
        safe = feat.replace("/", "_")
        plt.savefig(out_dir / f"shap_dependence_{name}_{safe}.png",
                    dpi=150, bbox_inches="tight")
        plt.close("all")

    # Per-race case studies: highest-confidence hit, worst miss, and the
    # first 2022 race (first post-regulation-reset event) — waterfalls.
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    case_rows = _case_study_rows(val_df, y_val, val_probs, races_val)
    base_value = (
        explainer.expected_value if np.ndim(explainer.expected_value) == 0
        else np.asarray(explainer.expected_value).ravel()[-1]
    )
    for label, idx in case_rows.items():
        expl = shap.Explanation(
            values=shap_values[idx],
            base_values=base_value,
            data=Xt_val.iloc[idx].to_numpy(),
            feature_names=names,
        )
        shap.plots.waterfall(expl, max_display=12, show=False)
        plt.title(f"{name} — {label}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_waterfall_{name}_{label}.png",
                    dpi=150, bbox_inches="tight")
        plt.close("all")

    return summary


def _case_study_rows(val_df, y_val, val_probs, races_val) -> dict[str, int]:
    """Positional indices (into the val design matrix) of case-study rows."""
    frame = pd.DataFrame({
        "race_id": np.asarray(races_val),
        "y": np.asarray(y_val),
        "prob": val_probs,
        "year": val_df["year"].to_numpy(),
        "round": val_df["round"].to_numpy(),
    })
    winners = frame[frame["y"] == 1]
    cases: dict[str, int] = {}
    cases["winner_highest_confidence"] = int(winners["prob"].idxmax())
    cases["winner_lowest_confidence"] = int(winners["prob"].idxmin())
    first_2022 = winners[(winners["year"] == 2022) & (winners["round"] == 1)]
    if not first_2022.empty:
        cases["winner_2022_round1_reset_race"] = int(first_2022.index[0])
    return cases


# ---------------------------------------------------------------------------
# Plot helpers for native importances
# ---------------------------------------------------------------------------

def _importance_plots(importances: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    top = importances.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Native feature importance (top 20) — {name}")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_dir / f"feature_importance_{name}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    by_class = importances.groupby("feature_class")["importance"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(by_class.index, by_class.values)
    ax.set_title(f"Importance by feature class — {name}")
    ax.set_xlabel("summed importance")
    fig.tight_layout()
    fig.savefig(out_dir / f"importance_by_class_{name}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-training analysis.")
    parser.add_argument("--model", default=None,
                        help="Zoo candidate to analyze (SHAP + importances).")
    parser.add_argument("--params", default=None,
                        help="JSON dict of pipeline params (tuned config).")
    parser.add_argument("--timing", action="store_true",
                        help="Measure fit/predict timing for the whole zoo.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = parser.parse_args(argv)

    if not args.model and not args.timing:
        parser.error("Nothing to do — pass --model and/or --timing.")

    params: dict = {}
    if args.params:
        params = json.loads(args.params)
        if not isinstance(params, dict):
            parser.error("--params must be a JSON object.")

    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found.", file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    split = temporal_split(pd.read_parquet(FEATURES_PATH))
    fingerprint = data_fingerprint()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.timing:
        with mlflow.start_run(run_name="zoo-timing"):
            mlflow.set_tags({"stage": "analysis", "data_fingerprint": fingerprint})
            timing = measure_timing(split.train, split.val,
                                    params_by_model={args.model: params} if args.model else None)
            timing.to_csv(REPORTS_DIR / "zoo_timing.csv", index=False)
            mlflow.log_text(timing.to_csv(index=False), "analysis/zoo_timing.csv")
        print(timing.to_string(index=False))

    if args.model:
        name = args.model
        X_tr, y_tr, _ = to_xy(split.train)
        pipeline = get_model(name, y_tr)
        if params:
            pipeline.set_params(**params)
        pipeline.fit(X_tr, y_tr)

        with mlflow.start_run(run_name=f"{name}-analysis"):
            mlflow.set_tags({"stage": "analysis", "model_family": MODEL_ZOO[name].family,
                             "data_fingerprint": fingerprint})
            if params:
                mlflow.log_params(params)

            importances = feature_importance_frame(pipeline)
            if importances is not None:
                importances.to_csv(REPORTS_DIR / f"feature_importance_{name}.csv",
                                   index=False)
                mlflow.log_text(importances.to_csv(index=False),
                                f"analysis/feature_importance_{name}.csv")
                _importance_plots(importances, name, REPORTS_DIR)

            perm = permutation_importance_top1(pipeline, split.val)
            perm.to_csv(REPORTS_DIR / f"permutation_importance_{name}.csv", index=False)
            mlflow.log_text(perm.to_csv(index=False),
                            f"analysis/permutation_importance_{name}.csv")
            mlflow.log_metric("perm_baseline_val_top1", perm.attrs["baseline_top1"])

            shap_summary = shap_analysis(pipeline, name, split.train, split.val,
                                         REPORTS_DIR)
            shap_summary.to_csv(REPORTS_DIR / f"shap_summary_{name}.csv", index=False)
            mlflow.log_text(shap_summary.to_csv(index=False),
                            f"analysis/shap_summary_{name}.csv")
            for png in REPORTS_DIR.glob(f"*_{name}*.png"):
                mlflow.log_artifact(str(png), artifact_path="analysis")

        print(f"Analysis artifacts for '{name}' written to {REPORTS_DIR}")
        print("\nTop 10 by mean |SHAP|:")
        print(shap_summary.head(10).to_string(index=False))
        print("\nTop 10 by permutation top-1 drop "
              f"(baseline val top-1 = {perm.attrs['baseline_top1']:.4f}):")
        print(perm.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
