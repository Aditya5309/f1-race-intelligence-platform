"""Model Insights page — importance/SHAP/calibration artifacts (design §3/§14).

Renders the static Phase-4 analysis figures from reports/phase4_analysis/;
no model computation happens here.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.views.common import sidebar_model_panel

_ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "reports" / "phase4_analysis"
_REPORT = "reports/model_selection_report.md"


def _figure(name: str, caption: str) -> None:
    path = _ANALYSIS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"_{name} not found — run `python -m src.models.analysis`._")


def render() -> None:
    st.title("Model Insights")
    sidebar_model_panel()

    st.markdown(
        f"""
The serving model is a **calibrated logistic regression** chosen over random
forest, XGBoost and LightGBM on validation performance, consistency, and
simplicity. Full evidence: `{_REPORT}`.

**Feature classes (Decision 013):** each of the 31 features is labeled
**Stable** (era-robust: normalized grid/qualifying, experience, lagged
standings positions), **Era-sensitive** (raw form counts — strongest in
dominance eras), or **Experimental** (sparse circuit history, raw lap
seconds). A healthy model concentrates its signal in Stable features — this
one draws ~59% of its importance from them, led by grid position and
qualifying results.
        """
    )

    tab_imp, tab_shap, tab_cal = st.tabs(
        ["Feature importance", "SHAP analysis", "Calibration"]
    )
    with tab_imp:
        _figure("feature_importance_logreg.png",
                "Native importance (|coefficient|), top 20 — grid/qualifying dominate")
        _figure("importance_by_class_logreg.png",
                "Summed importance by Decision-013 feature class")
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
