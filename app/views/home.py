"""Dashboard landing page — system status at a glance (UI/UX redesign).

Answers "is this thing working and what is it" in one screen: serving
status, headline evaluation numbers, and a one-line pointer to each other
page. Technical model internals (calibration method, run id, algorithm)
are deliberately NOT here — see the Model Insights page.
"""

from __future__ import annotations

import streamlit as st

from app.views import metadata
from app.views.common import list_races_or_empty, served_seasons, sidebar_model_panel
from app.views.components import badge_card, page_header, stat_row

_ALIAS_COLOR = {"production": "green", "staging": "blue"}


def _alias_color(alias: str) -> str:
    return _ALIAS_COLOR.get(alias.lower(), "gray")


def render() -> None:
    page_header(
        "Dashboard", "🏠",
        "Predicts the Formula 1 race winner before lights out, from grid "
        "position, form, and championship standings.",
    )
    health = sidebar_model_panel()

    model = (health or {}).get("model")
    api_ok = bool(health and health.get("status") == "ok")
    races = list_races_or_empty()
    seasons = served_seasons(races)

    col1, col2, col3 = st.columns(3)
    with col1:
        if model:
            badge_card("Model Status", model["alias"], color=_alias_color(model["alias"]))
        else:
            badge_card("Model Status", "Unavailable", color="red")
    with col2:
        badge_card("API Status", "Healthy" if api_ok else "Degraded",
                   color="green" if api_ok else "red")
    with col3:
        st.metric("Latest Model", f"v{model['version']}" if model else "—", border=True)

    stat_row([
        {"label": "Last Updated",
         "value": model["trained_at"][:10] if model else "—"},
        {"label": "Supported Seasons",
         "value": f"{seasons[0]}–{seasons[1]}" if seasons else "—"},
        {"label": "Predictions Available",
         "value": f"{len(races)} races" if races else "0"},
    ])

    if races:
        latest = max(races, key=lambda r: (r["year"], r["round"]))
        label = metadata.race_label(latest["race_id"],
                                    fallback_year=latest["year"],
                                    fallback_round=latest["round"])
        facts = metadata.race_facts(latest["race_id"])
        date = f" · {facts['date']}" if "date" in facts else ""
        st.caption(f"Latest served race: **{label}**{date}")

    st.divider()
    st.subheader("📈 Model performance")
    st.caption(
        "Never trained on — held-out validation and test seasons. Figures "
        "below are recorded at promotion time, not recomputed live — see "
        "Model Insights for what the currently-serving model actually does."
    )
    stat_row([
        {"label": "Top-1 accuracy · 2022–23", "value": "68.2%",
         "help": "Pole-sitter baseline on the same races: 54.5%"},
        {"label": "Top-3 recall · 2022–23", "value": "90.9%"},
        {"label": "Top-1 accuracy · 2024 test", "value": "45.8%",
         "help": "Equal to the pole baseline in 2024 — biggest edge shows up "
                 "in dominance seasons, see Season Analytics"},
    ])

    st.divider()
    st.subheader("🧭 Explore")
    # app/dashboard.py publishes its st.Page() objects into session_state —
    # st.page_link() needs the actual StreamlitPage object for callable-based
    # pages, and importing dashboard.py here would create an import cycle
    # (dashboard.py already imports this module).
    nav_pages: dict = st.session_state.get("_dashboard_pages", {})
    cols = st.columns(4)
    # icon carried separately from title: st.page_link() already renders the
    # target st.Page's own configured icon, so repeating it in the label
    # doubled every icon on screen — the icon is only needed in the plain-
    # caption fallback below, which has no other way to show it.
    links = [
        ("race_center", "🏎", "Race Center", "Pick a race, see the favorite and the field"),
        ("driver_explorer", "👤", "Driver Explorer", "A driver's season, race by race"),
        ("season_analytics", "📊", "Season Analytics", "Trends and who's on the rise"),
        ("insights", "🤖", "Model Insights", "How the model actually works"),
    ]
    for col, (key, icon, title, caption) in zip(cols, links, strict=True):
        with col, st.container(border=True):
            page = nav_pages.get(key)
            if page is not None:
                st.page_link(page, label=title)
            else:
                st.caption(f"{icon} {title}")
            st.caption(caption)
