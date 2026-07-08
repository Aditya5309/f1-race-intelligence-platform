"""Driver Explorer — one driver's race-by-race story (UI/UX redesign v2).

Season-scoped by default (fast: one season's worth of /predictions calls);
an optional full-career toggle aggregates every served season, reusing the
same per-season cache Season Analytics uses. Predictions come only from the
HTTP API; championship standings, career stats, and qualifying/finishing
trends are display metadata (app/views/metadata.py, Decision 024) and every
metadata block degrades gracefully when data/ is absent.
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


def _driver_labels(frame: pd.DataFrame) -> dict[int, str]:
    labels = (
        frame[["driver_id", "driver_name"]]
        .drop_duplicates()
        .assign(label=lambda d: d["driver_name"].fillna(d["driver_id"].astype(str)))
        .sort_values("label")
    )
    return dict(zip(labels["driver_id"], labels["label"], strict=True))


def render() -> None:
    page_header("Driver Explorer", "👤",
                "A driver's races, one season or a whole career.")
    sidebar_model_panel()

    races = list_races_or_stop()
    if not races:
        empty_state("No races available from the API.")
        return
    years = sorted({r["year"] for r in races})

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

    if frame.empty:
        empty_state("No prediction data for this scope.")
        return

    label_by_id = _driver_labels(frame)
    driver_ids = list(label_by_id)
    # One-shot preselect from a Race Center card click (app/views/components.
    # py::driver_explorer_link) — consumed immediately so it doesn't pin
    # later, unrelated visits to this page.
    preselect = st.session_state.pop("_preselect_driver_id", None)
    default_index = driver_ids.index(preselect) if preselect in driver_ids else 0
    driver_id = st.selectbox(
        "Driver", driver_ids, index=default_index,
        format_func=lambda did: label_by_id[did],
    )

    d = frame[frame["driver_id"] == driver_id].sort_values(["year", "round"])
    latest_team = d["constructor_name"].dropna()
    team = latest_team.iloc[-1] if len(latest_team) else None
    team_color = constructor_color(team)

    # --- profile card ------------------------------------------------------
    with st.container(border=True):
        col_avatar, col_bio = st.columns([1, 5])
        with col_avatar:
            st.markdown(f"# {constructor_dot(team)}")
        with col_bio:
            st.subheader(label_by_id[driver_id])
            st.caption(f"{team or 'Unknown team'} · {scope_label} · {len(d)} race(s)")
            if year is not None:
                standings = metadata.season_driver_standings(year)
                if not standings.empty:
                    mine = standings[standings["driverId"] == driver_id]
                    if len(mine):
                        row = mine.iloc[0]
                        st.markdown(
                            f"🏆 **Championship P{int(row['position'])}** · "
                            f"{float(row['points']):.0f} pts ({year})"
                        )

    # --- historical outcome stats (display metadata) -----------------------
    stats = metadata.driver_season_stats(driver_id, year)
    if stats:
        stat_row([
            {"label": "Wins", "value": str(stats.get("wins", 0))},
            {"label": "Podiums", "value": str(stats.get("podiums", 0))},
            {"label": "Poles", "value": str(stats.get("poles", 0))},
            {"label": "Points", "value": f"{stats.get('points', 0):.0f}"},
            {"label": "Avg qualifying",
             "value": f"P{stats['avg_quali']:.1f}" if "avg_quali" in stats else "—"},
            {"label": "Avg finish",
             "value": f"P{stats['avg_finish']:.1f}" if "avg_finish" in stats else "—"},
        ])

    # --- model-view stats (predictions) ------------------------------------
    wins_predicted_scope = int((d["driver_id"] == d["actual_winner_driver_id"]).sum())
    stat_items = [
        {"label": "Races (scope)", "value": str(len(d))},
        {"label": "Avg predicted rank", "value": f"{d['predicted_rank'].mean():.1f}"},
        {"label": "Best rank", "value": str(int(d["predicted_rank"].min()))},
        {"label": "Actual wins (scope)", "value": str(wins_predicted_scope)},
    ]
    mid = len(d) // 2
    if mid >= 1 and len(d) - mid >= 1:
        early_avg = d.iloc[:mid]["predicted_rank"].mean()
        recent_avg = d.iloc[mid:]["predicted_rank"].mean()
        stat_items.append({
            "label": "Recent avg rank", "value": f"{recent_avg:.1f}",
            "delta": f"{recent_avg - early_avg:+.1f} vs early {scope_label}",
            "delta_color": "inverse",   # lower rank number = better
        })
    stat_row(stat_items)

    # --- trends -------------------------------------------------------------
    st.subheader("📈 Trends")
    hist = metadata.driver_race_results(driver_id, year)
    if not hist.empty:
        hist = hist.assign(
            race=lambda x: x["year"].astype(str) + " R" + x["round"].astype(str))
        col_q, col_f = st.columns(2)
        with col_q:
            quali = hist.dropna(subset=["quali_position"]) \
                if "quali_position" in hist.columns else pd.DataFrame()
            if len(quali):
                trend_line(quali, "race", "quali_position",
                           title="Qualifying trend", y_label="Position",
                           invert_y=True, color=team_color)
            else:
                empty_state("No qualifying history in this scope.")
        with col_f:
            finish = hist.dropna(subset=["finish"])
            if len(finish):
                trend_line(finish, "race", "finish",
                           title="Finishing trend", y_label="Position",
                           invert_y=True, color=team_color)
            else:
                empty_state("No finishing history in this scope.")
    else:
        empty_state("Qualifying/finishing history unavailable "
                    "(display metadata not loaded).")

    col_pts, col_prob = st.columns(2)
    with col_pts:
        if year is not None:
            prog = metadata.driver_standings_progression(driver_id, year)
            if not prog.empty:
                trend_line(prog, "round", "points",
                           title=f"Championship points · {year}",
                           x_label="Round", y_label="Points", color=team_color)
            else:
                empty_state("Standings progression unavailable.")
        else:
            st.caption("Championship points chart is season-scoped — "
                       "pick a single season to see it.")
    with col_prob:
        chart = d.assign(
            race=lambda x: x["year"].astype(str) + " R" + x["round"].astype(str))
        trend_line(chart, "race", "win_probability",
                   title="Win share through the season",
                   y_label="Win share", color=team_color)

    # --- race log ------------------------------------------------------------
    st.subheader("🗓 Race log")
    log = d[["year", "round", "predicted_rank", "win_probability"]].copy()
    log["result"] = [
        "🏆 Won" if row.driver_id == row.actual_winner_driver_id
        else ("—" if pd.isna(row.actual_winner_driver_id) else "")
        for row in d.itertuples()
    ]
    st.dataframe(log, hide_index=True, width="stretch")
