"""Overview page — what the system is and how to read it (design §3/§14)."""

from __future__ import annotations

import streamlit as st

from app.views.common import ERA_CAVEAT, sidebar_model_panel


def render() -> None:
    st.title("F1 Race Winner Prediction")
    health = sidebar_model_panel()

    st.markdown(
        """
Predicts the winner of a Formula 1 Grand Prix **before lights out**, using
only information available once the starting grid is set: qualifying
results, rolling driver/constructor form, circuit history, and lagged
championship standings.

**How it works** — one probability per driver per race, from an
isotonic-calibrated logistic regression trained on 2010–2021 seasons and
selected against a five-model zoo (pole-sitter heuristic, logistic
regression, random forest, XGBoost, LightGBM). Probabilities are normalized
within each race, so they read as *share of win chance* and sum to 100%.
        """
    )

    st.subheader("Headline evaluation (never trained on)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Top-1 accuracy · validation 2022–23", "68.2%",
                help="Pole-sitter baseline on the same races: 54.5%")
    col2.metric("Top-3 recall · validation 2022–23", "88.6%")
    col3.metric("Top-1 accuracy · test 2024", "45.8%",
                help="Equal to the pole baseline in 2024 — see the era note")
    st.info(ERA_CAVEAT)

    st.subheader("What to explore")
    st.markdown(
        """
- **Predictions** — pick any 2010–2024 race and see the field's win shares
  next to the actual outcome.
- **Model Insights** — which features drive predictions (grid position and
  qualifying dominate), SHAP analysis, and calibration quality.
        """
    )
    if health and health.get("status") == "ok":
        st.caption(
            f"API healthy — serving {health['model']['name']} "
            f"v{health['model']['version']} @{health['model']['alias']}."
        )
