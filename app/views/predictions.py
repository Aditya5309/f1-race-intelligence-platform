"""Predictions page — race selector + win-share chart (design §3/§14)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.views.common import api_get_or_stop, sidebar_model_panel

WINNER_COLOR = "#E10600"     # F1 red for the actual winner
OTHER_COLOR = "#4C78A8"


def _label(p: dict) -> str:
    name = p["driver_name"] or f"driver {p['driver_id']}"
    team = p["constructor_name"]
    return f"{name} ({team})" if team else name


def render() -> None:
    st.title("Race Predictions")
    sidebar_model_panel()

    races = api_get_or_stop("/races")["races"]
    if not races:
        st.warning("No races available from the API.")
        return

    years = sorted({r["year"] for r in races}, reverse=True)
    with st.sidebar:
        year = st.selectbox("Season", years, index=0)
        year_races = [r for r in races if r["year"] == year]
        race = st.selectbox(
            "Race", year_races,
            format_func=lambda r: f"Round {r['round']} (raceId {r['race_id']})",
        )

    body = api_get_or_stop(f"/predictions/{race['race_id']}")
    preds = body["predictions"]
    winner_id = body["actual_winner_driver_id"]

    st.subheader(f"{body['year']} — Round {body['round']}")
    if body["model_top1_hit"] is not None:
        top = preds[0]
        if body["model_top1_hit"]:
            st.success(f"Model picked the winner: **{_label(top)}** "
                       f"({top['win_probability']:.1%} win share)")
        else:
            actual = next((p for p in preds if p["driver_id"] == winner_id), None)
            rank = actual["predicted_rank"] if actual else "?"
            name = _label(actual) if actual else f"driver {winner_id}"
            st.warning(f"Winner **{name}** was the model's **#{rank}** pick "
                       f"(model's #1: {_label(top)}).")

    frame = pd.DataFrame(preds)
    frame["label"] = [_label(p) for p in preds]
    frame["is_winner"] = frame["driver_id"] == winner_id
    top10, rest = frame.head(10), frame.iloc[10:]

    fig = go.Figure(go.Bar(
        x=top10["win_probability"][::-1],
        y=top10["label"][::-1],
        orientation="h",
        marker_color=[WINNER_COLOR if w else OTHER_COLOR
                      for w in top10["is_winner"][::-1]],
        text=[f"{v:.1%}" for v in top10["win_probability"][::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Win share (normalized within race)",
        xaxis_tickformat=".0%", height=420,
        margin=dict(l=10, r=40, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bar = actual winner. Tied shares are normal — the "
               "calibrator maps similar cars to the same probability step; "
               "ranking uses a deterministic tiebreak.")

    if len(rest):
        with st.expander(f"Rest of the field ({len(rest)} drivers)"):
            st.dataframe(
                rest[["predicted_rank", "label", "win_probability",
                      "win_probability_raw"]],
                hide_index=True, use_container_width=True,
            )

    with st.expander("Full field table"):
        st.dataframe(
            frame[["predicted_rank", "label", "win_probability",
                   "win_probability_raw", "is_winner"]],
            hide_index=True, use_container_width=True,
        )
    st.caption(f"prediction_id: `{body['prediction_id']}` · generated "
               f"{body['generated_at']} · model v{body['model']['version']}")
