"""
app/dashboard.py

Streamlit dashboard entry point (Decision 016; UI/UX redesign, Decision
023/024; Phase 2 new pages). Six pages (st.navigation), user-facing first
-- Compare Drivers and Team land in their own commits and will be added
here alongside them:

    streamlit run app/dashboard.py

    Dashboard          system status + headline metrics (app/views/home.py)
    Race Center         per-race storytelling view (app/views/race_center.py)
    Driver Explorer      one driver's races (app/views/driver_explorer.py)
    Circuit Explorer     one circuit's history (app/views/circuit_explorer.py)
    Season Analytics    season-level trends (app/views/season_analytics.py)
    Model Insights       advanced/technical (app/views/insights.py)

All data comes from the FastAPI service (F1_API_URL, default
http://localhost:8000) — start it first: `uvicorn app.api:app`.
"""

from __future__ import annotations

import streamlit as st

from app.views import (
    circuit_explorer,
    driver_explorer,
    home,
    insights,
    race_center,
    season_analytics,
)

st.set_page_config(
    page_title="F1 Race Winner Prediction",
    page_icon="🏎️",
    layout="wide",
)

dashboard_page = st.Page(home.render, title="Dashboard", icon="🏠",
                          url_path="dashboard", default=True)
race_center_page = st.Page(race_center.render, title="Race Center", icon="🏎",
                            url_path="race-center")
driver_explorer_page = st.Page(driver_explorer.render, title="Driver Explorer",
                                icon="👤", url_path="driver-explorer")
circuit_explorer_page = st.Page(circuit_explorer.render, title="Circuit Explorer",
                                 icon="🏟", url_path="circuit-explorer")
season_analytics_page = st.Page(season_analytics.render, title="Season Analytics",
                                 icon="📊", url_path="season-analytics")
insights_page = st.Page(insights.render, title="Model Insights", icon="🤖",
                         url_path="model-insights")

# st.page_link() needs the actual StreamlitPage object for callable-based
# pages (a file-path string only resolves for file-based pages) — published
# here so app/views/home.py's "Explore" cards can link to the other pages
# without importing this module (which would create dashboard -> home ->
# dashboard import cycle, since dashboard.py already imports every view).
st.session_state["_dashboard_pages"] = {
    "race_center": race_center_page,
    "driver_explorer": driver_explorer_page,
    "circuit_explorer": circuit_explorer_page,
    "season_analytics": season_analytics_page,
    "insights": insights_page,
}

pages = st.navigation([
    dashboard_page, race_center_page, driver_explorer_page,
    circuit_explorer_page, season_analytics_page, insights_page,
])
pages.run()
