"""Season Analytics — season-level trends (UI/UX redesign v2).

"What's happening this season" and "who's trending" — model views built from
client-side aggregation over GET /races + GET /predictions/{race_id} (the
same season_predictions() helper Driver Explorer uses); championship
standings come from display metadata (app/views/metadata.py)
and degrade gracefully when data/ is absent. No new endpoints, no
ML-pipeline changes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views import metadata
from app.views.charts import histogram, standings_bar, trend_line
from app.views.common import (
    list_races_or_stop,
    season_predictions_or_stop,
    sidebar_model_panel,
)
from app.views.components import empty_state, page_header, stat_row

_MIN_RACES_FOR_TREND = 4   # >= 2 races per half, for a meaningful early/recent split


def _driver_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-driver early-vs-recent average predicted rank, for the season's
    'Rising & fading' board. Drivers with too few races are excluded rather
    than shown with a noisy two-point trend."""
    rows = []
    for driver_id, g in frame.groupby("driver_id"):
        g = g.sort_values("round")
        if len(g) < _MIN_RACES_FOR_TREND:
            continue
        mid = len(g) // 2
        early_avg = g.iloc[:mid]["predicted_rank"].mean()
        recent_avg = g.iloc[mid:]["predicted_rank"].mean()
        names = g["driver_name"].dropna()
        label = names.iloc[-1] if len(names) else str(driver_id)
        rows.append({
            "driver_id": driver_id, "label": label,
            "early_avg": early_avg, "recent_avg": recent_avg,
            "delta": recent_avg - early_avg,   # negative = improving
        })
    return pd.DataFrame(rows)


_MIN_DOMINANCE_PICKS = 2   # times as the favorite, before calling someone "dominant"


def _season_insights(top_picks: pd.DataFrame, surprises: pd.DataFrame,
                     trend: pd.DataFrame) -> list[str]:
    """3-4 deterministic, templated one-line insights from aggregates this
    page already computes (top_picks, surprises, _driver_trend) — no LLM,
    no new data, just re-narrating numbers already on the page."""
    insights: list[str] = []

    dominance = (
        top_picks.groupby("label")["win_probability"].agg(["mean", "size"])
        .query(f"size >= {_MIN_DOMINANCE_PICKS}")
        .sort_values("mean", ascending=False)
    )
    if len(dominance):
        name = dominance.index[0]
        row = dominance.iloc[0]
        insights.append(
            f"🏆 **Most dominant:** {name} averaged {row['mean']:.1%} win "
            f"share across {int(row['size'])} races as the model's favorite."
        )

    if len(surprises):
        row = surprises.iloc[0]
        name = row["driver_name"] or f"driver {row['driver_id']}"
        insights.append(
            f"😲 **Biggest surprise:** {name} won from model rank "
            f"#{int(row['predicted_rank'])}."
        )

    if len(trend):
        best = trend.nsmallest(1, "delta").iloc[0]
        insights.append(
            f"📈 **Most improved:** {best['label']}, average predicted rank "
            f"{best['early_avg']:.1f} → {best['recent_avg']:.1f} this season."
        )

    if len(top_picks):
        most_conf = top_picks.loc[top_picks["win_probability"].idxmax()]
        least_conf = top_picks.loc[top_picks["win_probability"].idxmin()]
        insights.append(
            f"🔒 **Most confident pick:** {most_conf['label']} at "
            f"{most_conf['win_probability']:.1%} (round {int(most_conf['round'])}). "
            f"**Least confident:** {least_conf['label']} at "
            f"{least_conf['win_probability']:.1%} (round {int(least_conf['round'])})."
        )
    return insights


def render() -> None:
    page_header("Season Analytics", "📊",
                "How the season is unfolding, round by round.")
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

    by_race = frame.drop_duplicates("race_id")
    decided = by_race["model_top1_hit"].dropna()
    hit_rate = decided.mean() if len(decided) else None

    # Computed once, up front, and reused both by the Insights block below
    # and by their own regular sections further down the page.
    top_picks = frame[frame["predicted_rank"] == 1].copy()
    top_picks["label"] = top_picks["driver_name"].fillna(
        top_picks["driver_id"].astype(str))
    winners = frame[frame["driver_id"] == frame["actual_winner_driver_id"]]
    surprises = winners.nlargest(3, "predicted_rank")
    trend = _driver_trend(frame)

    stat_row([
        {"label": "Races", "value": str(by_race["race_id"].nunique())},
        {"label": "Model hit rate",
         "value": f"{hit_rate:.0%}" if hit_rate is not None else "—",
         "help": "Share of races where the model's #1 pick actually won"},
        {"label": "Avg field size",
         "value": f"{frame.groupby('race_id').size().mean():.0f}"},
    ])

    st.subheader("🔎 Insights")
    insights = _season_insights(top_picks, surprises, trend)
    if insights:
        for insight in insights:
            st.caption(insight)
    else:
        empty_state("Not enough decided races yet this season for insights.")

    st.subheader("📈 Model accuracy over the season")
    hits = (
        by_race.sort_values("round")
        .assign(hit=lambda d: d["model_top1_hit"].map({True: 1, False: 0}))
        .dropna(subset=["hit"])
    )
    col_bar, col_cum = st.columns(2)
    with col_bar:
        st.bar_chart(hits.set_index("round")["hit"])
        st.caption("1 = the model's favorite won that round, 0 = it didn't.")
    with col_cum:
        cum = hits.assign(cumulative=lambda d: d["hit"].expanding().mean())
        trend_line(cum, "round", "cumulative",
                   title="Cumulative hit rate", x_label="Round",
                   y_label="Hit rate")

    st.subheader("🏆 Championship standings")
    ds = metadata.season_driver_standings(year)
    cs = metadata.season_constructor_standings(year)
    if ds.empty and cs.empty:
        empty_state("Standings unavailable (display metadata not loaded).")
    else:
        team_by_driver = dict(zip(
            frame["driver_id"],
            frame["constructor_name"].fillna(""), strict=False))
        col_d, col_c = st.columns(2)
        with col_d:
            st.markdown("**Drivers**")
            if not ds.empty and "driver" in ds.columns:
                top = ds.head(10).assign(
                    team=lambda x: x["driverId"].map(team_by_driver))
                standings_bar(top, "driver", "points", color_col="team")
            else:
                empty_state("Driver standings unavailable.")
        with col_c:
            st.markdown("**Constructors**")
            if not cs.empty and "constructor" in cs.columns:
                standings_bar(cs.head(10), "constructor", "points",
                              color_col="constructor")
            else:
                empty_state("Constructor standings unavailable.")

    st.subheader("🎯 Most predicted winners")
    picks = (
        top_picks.groupby("label")
        .agg(picks=("race_id", "size"), team=("constructor_name", "last"))
        .sort_values("picks", ascending=False)
        .reset_index()
    )
    standings_bar(picks, "label", "picks", color_col="team",
                  height=max(220, 40 * len(picks)))
    st.caption("How often each driver was the model's #1 pick this season.")

    st.subheader("😲 Most surprising races")
    if surprises.empty:
        empty_state("No decided races in this season's data.")
    else:
        cols = st.columns(len(surprises))
        for col, row in zip(cols, surprises.itertuples(), strict=True):
            with col, st.container(border=True):
                st.markdown(
                    f"**{metadata.race_label(row.race_id, fallback_round=row.round)}**")
                winner_name = row.driver_name or f"driver {row.driver_id}"
                st.caption(f"Won by **{winner_name}**")
                st.caption(f"Model rank **#{int(row.predicted_rank)}** · "
                           f"{row.win_probability:.1%} win share")

    st.subheader("📊 Win-share distribution")
    histogram(top_picks["win_probability"], "Favorite's win share",
              percent_axis=True)
    st.caption("How dominant the model's favorite looked, race by race — a "
               "right-shifted distribution means one-sided races.")

    st.subheader("📶 Rising & fading")
    if trend.empty:
        empty_state("Not enough multi-race history yet this season for a trend.")
    else:
        rising = trend.nsmallest(3, "delta")
        fading = trend.nlargest(3, "delta")
        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("**🔺 Rising**")
            for row in rising.itertuples():
                st.metric(row.label, f"{row.recent_avg:.1f} avg rank",
                          delta=f"{row.delta:+.1f}", delta_color="inverse",
                          border=True)
        with col_down:
            st.markdown("**🔻 Fading**")
            for row in fading.itertuples():
                st.metric(row.label, f"{row.recent_avg:.1f} avg rank",
                          delta=f"{row.delta:+.1f}", delta_color="inverse",
                          border=True)

    st.subheader("🏭 Constructor outlook")
    by_constructor = (
        frame.dropna(subset=["constructor_name"])
        .groupby("constructor_name")["win_probability"].mean()
        .sort_values(ascending=False).head(8)
        .round(3)
        .reset_index()
    )
    if by_constructor.empty:
        empty_state("Constructor names unavailable (display-name lookup not loaded).")
    else:
        standings_bar(by_constructor, "constructor_name", "win_probability",
                      color_col="constructor_name", height=320,
                      percent=True, axis_title="Win share")
        st.caption("Average predicted win share per race this season — higher "
                   "means the model favors that team more often.")
