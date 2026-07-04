"""
app/dashboard.py

Streamlit dashboard entry point (Decision 016; application_design.md §3/§14).

    streamlit run app/dashboard.py

Three pages (st.navigation): Overview, Predictions, Model Insights.
All data comes from the FastAPI service (F1_API_URL, default
http://localhost:8000) — start it first: `uvicorn app.api:app`.
"""

from __future__ import annotations

import streamlit as st

from app.views import insights, overview, predictions

st.set_page_config(
    page_title="F1 Race Winner Prediction",
    page_icon="🏎️",
    layout="wide",
)

pages = st.navigation([
    st.Page(overview.render, title="Overview", icon="🏁", default=True),
    st.Page(predictions.render, title="Predictions", icon="📊"),
    st.Page(insights.render, title="Model Insights", icon="🔍"),
])
pages.run()
