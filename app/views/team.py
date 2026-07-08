"""Team (Constructor) page — one constructor's season or career (Phase 2).

Mirrors Driver Explorer's season/full-career toggle pattern at constructor
grain. Predictions come only from the HTTP API; roster, standings, and
qualifying/finishing trends are display metadata (app/views/metadata.py)
and degrade gracefully when data/ is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views import metadata
from app.views.charts import trend_line
from app.views.common import (
    career_predictions_or_stop,
    list_races_or_stop,
    season_predictions_or_stop,
    sidebar_model_panel,
)
from app.views.components import (
    constructor_color,
    constructor_dot,
    empty_state,
    page_header,
    stat_row,
)


def render() -> None:
    page_header("Team", "🏭",
                "A constructor's season or career — roster, form, and model accuracy.")
    sidebar_model_panel()

    races = list_races_or_stop()
    if not races:
        empty_state("No races available from the API.")
        return
    years = sorted({r["year"] for r in races})

    catalog = metadata.constructor_catalog()
    if catalog.empty:
        empty_state("Constructor metadata unavailable (display data not loaded).")
        return

    with st.sidebar:
        st.divider()
        full_career = st.checkbox(
            "Full career (all seasons)", value=False,
            help="Aggregates every served season — slower on first load, "
                 "instant afterward (cached per season).",
        )
        if full_career:
            frame = career_predictions_or_stop(years)
            scope_label = f"{years[0]}–{years[-1]}"
            year = None
        else:
            year = st.selectbox("Season", years[::-1], index=0)
            frame = season_predictions_or_stop(year)
            scope_label = str(year)

    constructor_ids = catalog["constructorId"].tolist()
    label_by_id = dict(zip(catalog["constructorId"], catalog["name"], strict=True))
    constructor_id = st.selectbox(
        "Team", constructor_ids, format_func=lambda cid: label_by_id[cid])
    team_name = label_by_id[constructor_id]
    team_color = constructor_color(team_name)
    nationality = catalog.loc[catalog["constructorId"] == constructor_id, "nationality"]

    # --- profile card --------------------------------------------------------
    with st.container(border=True):
        col_dot, col_bio = st.columns([1, 5])
        with col_dot:
            st.markdown(f"# {constructor_dot(team_name)}")
        with col_bio:
            st.subheader(team_name)
            nat = nationality.iloc[0] if len(nationality) else "Unknown"
            st.caption(f"{nat} · {scope_label}")
            if year is not None:
                standings = metadata.season_constructor_standings(year)
                if not standings.empty:
                    mine = standings[standings["constructorId"] == constructor_id]
                    if len(mine):
                        row = mine.iloc[0]
                        st.markdown(
                            f"🏆 **Championship P{int(row['position'])}** · "
                            f"{float(row['points']):.0f} pts ({year})"
                        )
                drivers = metadata.constructor_current_drivers(constructor_id, year)
                if drivers:
                    st.caption(f"Drivers this season: {', '.join(drivers)}")

    # --- historical outcome stats (display metadata) -------------------------
    stats = metadata.constructor_season_stats(constructor_id, year)
    if stats:
        stat_row([
            {"label": "Races", "value": str(stats.get("races", 0))},
            {"label": "Wins", "value": str(stats.get("wins", 0))},
            {"label": "Podiums", "value": str(stats.get("podiums", 0))},
            {"label": "Points", "value": f"{stats.get('points', 0):.0f}"},
            {"label": "Avg qualifying",
             "value": f"P{stats['avg_quali']:.1f}" if "avg_quali" in stats else "—"},
        ])
    else:
        empty_state("No historical stats for this team in this scope.")

    # --- model-view: prediction accuracy for this team's drivers -------------
    if not frame.empty:
        team_frame = frame[frame["constructor_name"] == team_name]
        picks = team_frame[team_frame["predicted_rank"] == 1]
        if len(picks):
            decided = picks["actual_winner_driver_id"].notna().sum()
            hits = int((picks["driver_id"] == picks["actual_winner_driver_id"]).sum())
            if decided:
                st.caption(
                    f"🎯 The model picked a {team_name} driver as its #1 favorite "
                    f"in {len(picks)} race(s) this scope; that pick won {hits} of "
                    f"{int(decided)} decided race(s) ({hits / decided:.0%})."
                )
            else:
                st.caption(
                    f"🎯 The model picked a {team_name} driver as its #1 favorite "
                    f"in {len(picks)} race(s) this scope; no results decided yet."
                )
        else:
            st.caption(
                f"The model hasn't picked a {team_name} driver as its #1 "
                "favorite in this scope."
            )

    # --- trends ----------------------------------------------------------------
    st.subheader("📈 Trends")
    hist = metadata.constructor_race_results(constructor_id, year)
    if not hist.empty:
        hist = hist.assign(
            race=lambda x: x["year"].astype(str) + " R" + x["round"].astype(str))
        col_q, col_f = st.columns(2)
        with col_q:
            quali = hist.dropna(subset=["quali_position"]) \
                if "quali_position" in hist.columns else pd.DataFrame()
            if len(quali):
                trend_line(quali, "race", "quali_position",
                           title="Qualifying trend (avg. both cars)",
                           y_label="Position", invert_y=True, color=team_color)
            else:
                empty_state("No qualifying history in this scope.")
        with col_f:
            finish = hist.dropna(subset=["finish"])
            if len(finish):
                trend_line(finish, "race", "finish",
                           title="Best finish per race", y_label="Position",
                           invert_y=True, color=team_color)
            else:
                empty_state("No finishing history in this scope.")
    else:
        empty_state("Qualifying/finishing history unavailable "
                    "(display metadata not loaded).")
