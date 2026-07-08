"""Compare Drivers — two-driver side-by-side, season-scoped (Phase 2).

Season-scoped only, deliberately: comparing across regulation eras (or a
driver's own career-spanning seasons) would mix car-performance changes
into what should be a driver-skill comparison — the same era confound
already flagged for Circuit Explorer's fastest-lap stat. Predictions come
only from the HTTP API; qualifying/finish/consistency stats are display
metadata (app/views/metadata.py) and degrade gracefully when data/ is
absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views import metadata
from app.views.common import (
    list_races_or_stop,
    season_predictions_or_stop,
    sidebar_model_panel,
)
from app.views.components import constructor_dot, empty_state, page_header


def _driver_labels(frame: pd.DataFrame) -> dict[int, str]:
    labels = (
        frame[["driver_id", "driver_name"]]
        .drop_duplicates()
        .assign(label=lambda d: d["driver_name"].fillna(d["driver_id"].astype(str)))
        .sort_values("label")
    )
    return dict(zip(labels["driver_id"], labels["label"], strict=True))


def _stat_column(driver_id: int, label: str, frame: pd.DataFrame, year: int) -> None:
    stats = metadata.driver_season_stats(driver_id, year)
    team = frame.loc[frame["driver_id"] == driver_id, "constructor_name"].dropna()
    team_name = team.iloc[-1] if len(team) else None
    st.subheader(f"{constructor_dot(team_name)} {label}")
    st.caption(team_name or "Unknown team")
    if not stats:
        empty_state("No stats for this driver this season.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg qualifying",
                  f"P{stats['avg_quali']:.1f}" if "avg_quali" in stats else "—")
    with col2:
        st.metric("Avg finish",
                  f"P{stats['avg_finish']:.1f}" if "avg_finish" in stats else "—")
    if "consistency_std" in stats:
        st.metric("Consistency (std dev of finish position)",
                  f"±{stats['consistency_std']:.1f}",
                  help="Lower = more consistent. Classified finishes only "
                       "(DNFs excluded, so a crash doesn't read as 'bad "
                       "consistency').")
        st.caption("Based on 3+ classified races this season — small "
                   "samples can swing this number.")
    else:
        st.caption("Not enough classified races this season for a "
                   "consistency figure.")


def render() -> None:
    page_header("Compare Drivers", "⚖️",
                "Two drivers, one season — side by side.")
    sidebar_model_panel()

    races = list_races_or_stop()
    if not races:
        empty_state("No races available from the API.")
        return
    years = sorted({r["year"] for r in races}, reverse=True)
    year = st.selectbox("Season", years, index=0)

    with st.spinner(f"Loading {year} predictions…"):
        frame = season_predictions_or_stop(year)
    if frame.empty:
        empty_state(f"No prediction data for {year}.")
        return

    label_by_id = _driver_labels(frame)
    driver_ids = list(label_by_id)
    if len(driver_ids) < 2:
        empty_state("Not enough drivers this season to compare.")
        return

    col_pick_a, col_pick_b = st.columns(2)
    with col_pick_a:
        driver_a = st.selectbox("Driver A", driver_ids, index=0,
                                format_func=lambda did: label_by_id[did])
    with col_pick_b:
        driver_b = st.selectbox("Driver B", driver_ids, index=1,
                                format_func=lambda did: label_by_id[did])

    if driver_a == driver_b:
        st.warning("Pick two different drivers to compare.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        _stat_column(driver_a, label_by_id[driver_a], frame, year)
    with col_b:
        _stat_column(driver_b, label_by_id[driver_b], frame, year)

    st.divider()
    st.subheader("🏁 Predicted win chance for a race this season")
    round_by_race = dict(zip(frame["race_id"], frame["round"], strict=False))
    race_ids = sorted(round_by_race, key=lambda rid: round_by_race[rid])
    race_id = st.selectbox(
        "Race", race_ids,
        format_func=lambda rid: metadata.race_label(
            rid, fallback_round=round_by_race[rid]),
    )
    col_prob_a, col_prob_b = st.columns(2)
    for col, driver_id in ((col_prob_a, driver_a), (col_prob_b, driver_b)):
        row = frame[(frame["race_id"] == race_id) & (frame["driver_id"] == driver_id)]
        with col:
            if len(row):
                st.metric(label_by_id[driver_id],
                         f"{row.iloc[0]['win_probability']:.1%}")
            else:
                st.caption(f"{label_by_id[driver_id]} didn't enter this race.")
