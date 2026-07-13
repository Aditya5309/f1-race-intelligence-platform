"""Model Insights page — importance/SHAP/calibration artifacts.

Renders the static analysis figures from reports/phase4_analysis/;
no model computation happens here. This is the ADVANCED page: the technical
detail intentionally kept off the other pages
— calibration method, run id, algorithm choice — lives
here, for recruiters/engineers who want to see how the model actually works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import Settings
from app.views.common import sidebar_model_panel
from app.views.components import page_header, stat_row

_ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "reports" / "phase4_analysis"
_COMPARISON_CSV = _ANALYSIS_DIR.parent / "model_comparison.csv"
_REPORT = "reports/model_selection_report.md"

# Feature classification, mirrored here as display copy since the
# dashboard cannot import src/ — the code-level source of truth is
# src/features/metadata.py; keep the two in sync on reclassification.
_STABLE_FEATURES = [
    "qualifying_position", "qualifying_gap_to_pole_pct", "reached_q2",
    "reached_q3", "pit_lane_start", "grid_adjusted", "grid_position_norm",
    "driver_experience_races", "driver_avg_finish_last_5",
    "driver_dnf_rate_last_5", "driver_standing_position_prev",
    "constructor_standing_position_prev",
]
_ERA_SENSITIVE_FEATURES = [
    "driver_wins_last_3", "driver_wins_last_5", "driver_wins_last_10",
    "driver_podiums_last_5", "driver_points_last_5", "constructor_wins_last_3",
    "constructor_wins_last_5", "constructor_podiums_last_5",
    "constructor_dnf_rate_last_5", "driver_standing_points_prev",
    "driver_standing_wins_prev", "constructor_standing_points_prev",
]
_EXPERIMENTAL_FEATURES = [
    "q1_sec", "q2_sec", "q3_sec", "driver_circuit_starts",
    "driver_circuit_wins", "driver_circuit_avg_finish",
    "constructor_circuit_wins",
]


def _figure(name: str, caption: str) -> None:
    path = _ANALYSIS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"_{name} not found — run `python -m src.models.analysis`._")


def _developer_console() -> None:
    """?dev=true extra: raw serving-bundle artifacts that exist on disk but
    are never otherwise rendered anywhere -- the actual manifest, recorded
    training schema, and dependency versions the deployed bundle was frozen
    with. Distinct from the curated charts above it, additive only (no nav
    change, no removal of anything public)."""
    bundle_dir = Settings().serving_bundle_path
    manifest_path = bundle_dir / "manifest.json"
    schema_path = bundle_dir / "feature_schema.json"
    env_path = bundle_dir / "model" / "python_env.yaml"
    req_path = bundle_dir / "model" / "requirements.txt"

    with st.expander("Serving bundle manifest"):
        if manifest_path.exists():
            st.json(json.loads(manifest_path.read_text()))
        else:
            st.caption(f"_{manifest_path} not found._")

    with st.expander("Recorded training schema (feature_schema.json)"):
        if schema_path.exists():
            st.json(json.loads(schema_path.read_text()))
        else:
            st.caption(f"_{schema_path} not found._")

    with st.expander("Environment / dependency versions"):
        if env_path.exists():
            st.code(env_path.read_text(), language="yaml")
        if req_path.exists():
            st.code(req_path.read_text(), language="text")
        if not env_path.exists() and not req_path.exists():
            st.caption("_Environment files not found._")


def render() -> None:
    page_header("Model Insights", "🤖", "Advanced — the ML internals behind the predictions.")
    st.badge("Advanced / technical", icon=":material/science:", color="violet")
    health = sidebar_model_panel()

    model = (health or {}).get("model")
    if model:
        st.subheader("🔧 Model card")
        stat_row([
            {"label": "Algorithm", "value": model["model_class"]},
            {"label": "Calibration", "value": model["calibration"]},
            {"label": "Registry alias", "value": f"v{model['version']} @ {model['alias']}"},
            {"label": "Trained", "value": model["trained_at"][:10]},
        ])
        st.caption(f"Run id: `{model['run_id']}`")

    st.subheader("📐 Validation results")
    stat_row([
        {"label": "Top-1 · validation 2022–23", "value": "68.2%",
         "help": "Pole-sitter baseline on the same races: 54.5%"},
        {"label": "Top-3 recall · validation", "value": "88.6%"},
        {"label": "Top-1 · final test 2024", "value": "45.8%",
         "help": "Equal to the pole baseline — the edge is dominance-season "
                 "concentrated"},
        {"label": "ECE after calibration", "value": "0.012",
         "delta": "-0.141 vs raw", "delta_color": "inverse",
         "help": "Expected calibration error on validation, isotonic "
                 "calibration fit on out-of-fold training predictions"},
    ])

    st.markdown(
        f"""
The serving model is a **calibrated logistic regression** chosen over random
forest, XGBoost and LightGBM on validation performance, consistency, and
simplicity. Full evidence: `{_REPORT}`.
        """
    )

    st.subheader("⚖️ Model comparison")
    if _COMPARISON_CSV.exists():
        st.dataframe(pd.read_csv(_COMPARISON_CSV), hide_index=True,
                     width="stretch")
    else:
        st.caption("_model_comparison.csv not found — the comparison table "
                   "lives in reports/ (local-only) and in MLflow "
                   "(`mlflow ui`)._")

    st.subheader("🏷 Feature classes")
    st.caption("All 31 model features, classified by era-robustness. A "
               "healthy model concentrates its signal in Stable features — "
               "this one draws ~59% of its importance from them, led by grid "
               "position and qualifying results.")
    col_s, col_e, col_x = st.columns(3)
    with col_s:
        st.markdown(f"**Stable ({len(_STABLE_FEATURES)})**")
        for feature in _STABLE_FEATURES:
            st.badge(feature, color="green")
    with col_e:
        st.markdown(f"**Era-sensitive ({len(_ERA_SENSITIVE_FEATURES)})**")
        for feature in _ERA_SENSITIVE_FEATURES:
            st.badge(feature, color="orange")
    with col_x:
        st.markdown(f"**Experimental ({len(_EXPERIMENTAL_FEATURES)})**")
        for feature in _EXPERIMENTAL_FEATURES:
            st.badge(feature, color="violet")

    tab_imp, tab_shap, tab_cal, tab_diag = st.tabs(
        ["Feature importance", "SHAP analysis", "Calibration", "Diagnostics"]
    )
    with tab_imp:
        _figure("feature_importance_logreg.png",
                "Native importance (|coefficient|), top 20 — grid/qualifying dominate")
        _figure("importance_by_class_logreg.png",
                "Summed importance by feature class")
    with tab_shap:
        _figure("shap_summary_logreg.png",
                "SHAP beeswarm on validation races — direction and spread per feature")
        _figure("shap_bar_logreg.png", "Mean |SHAP| ranking")
        _figure("shap_waterfall_logreg_winner_highest_confidence.png",
                "Case study: the winner the model was most confident about")
    with tab_cal:
        st.markdown(
            """
Raw class-weighted probabilities are deliberately inflated during training;
an **isotonic calibrator** (fit only on out-of-fold training predictions)
restores honest probabilities: expected calibration error drops from
**0.153 to 0.012** on validation with ranking quality unchanged.
            """
        )
        _figure("calibration_logreg.png", "Reliability diagram")
        st.caption("Per-run calibration plots are logged with every MLflow "
                   "training run (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).")
    with tab_diag:
        st.caption("Classifier diagnostics. Note: per-race top-1/top-3 (above) "
                   "are this project's primary metrics — row-level curves are "
                   "secondary at a 4.7% positive rate.")
        _figure("confusion_matrix.png", "Confusion matrix")
        _figure("roc_curve.png", "ROC curve")
        _figure("precision_recall.png", "Precision–recall curve")

    if st.query_params.get("dev") == "true":
        st.divider()
        st.subheader("🛠 Developer Console")
        st.caption("Extra technical detail, shown only with `?dev=true` in the URL.")
        _developer_console()
    else:
        st.caption("Engineers: append `?dev=true` to this page's URL for the raw "
                   "serving bundle manifest, training schema, and dependency versions.")
