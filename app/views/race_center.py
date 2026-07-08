"""Race Center — storytelling-first race view (UI/UX redesign v2).

Leads with the Grand Prix identity, the model's favorite and how confident
it is, the top contenders as cards, and a plain-language "why" — the full
field chart and tables follow as supporting detail. Predictions come only
from GET /races + GET /predictions/{race_id}; Grand Prix names, grids, and
race facts are display metadata (app/views/metadata.py, Decision 024) and
every one of them degrades gracefully when data/ is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.views import metadata
from app.views.charts import win_share_bar
from app.views.common import (
    ERA_CAVEAT_SHORT,
    api_get_or_stop,
    list_races_or_stop,
    season_predictions_or_stop,
    sidebar_model_panel,
)
from app.views.components import (
    driver_card,
    driver_label,
    empty_state,
    favorite_card,
    hit_miss_badge,
    page_header,
    podium_row,
    race_header,
    stat_row,
)

_FORM_WIN_SHARE = 0.20          # mean share over last 3 rounds -> "excellent form"
_CONSTRUCTOR_WIN_SHARE = 0.15   # season mean share -> "strong constructor"
_CLEAR_SEPARATION_GAP = 0.15    # #1 vs #2 win-share gap -> "clear separation"
_TIGHT_RACE_GAP = 0.05          # #1 vs #2 win-share gap -> "tight race"


def _confidence_reasons(top: dict, second: dict | None, grid: int | None) -> list[str]:
    """2-3 plain-language reasons behind the star rating — genuine per-race
    signals only (there's no per-prediction SHAP output to draw a "feature
    stability" reason from): the probability gap to the runner-up, the
    favorite's grid position, and the dominance/competitive era caveat."""
    reasons: list[str] = []
    if second is not None:
        gap = top["win_probability"] - second["win_probability"]
        if gap >= _CLEAR_SEPARATION_GAP:
            reasons.append(f"Clear separation from the field (+{gap:.0%} over 2nd)")
        elif gap < _TIGHT_RACE_GAP:
            reasons.append("Tight race — several close contenders")
    if grid == 1:
        reasons.append("Favorite started from pole")
    elif grid is not None and grid > 5:
        reasons.append(f"Favorite started outside the top 5 (P{grid})")
    reasons.append(ERA_CAVEAT_SHORT)
    return reasons


def _reasons_for_favorite(top: dict, grid: int | None, quali: int | None,
                          season: pd.DataFrame, current_round: int) -> list[str]:
    """Plain-language pre-race factors for the #1 pick, derived from display
    data (grid/quali/season win shares) — NOT a SHAP readout; the caption
    below the badges points at Model Insights for the real analysis."""
    reasons: list[str] = []
    if grid == 1:
        reasons.append("Pole position")
    elif grid is not None and 0 < grid <= 3:
        reasons.append("Front-row start")
    if quali is not None and quali <= 3 and grid != 1:
        reasons.append("Strong qualifying pace")
    if not season.empty and current_round > 1:
        prior = season[(season["driver_id"] == top["driver_id"])
                       & (season["round"] < current_round)]
        recent = prior.sort_values("round").tail(3)
        if len(recent) and recent["win_probability"].mean() > _FORM_WIN_SHARE:
            reasons.append("Excellent recent form")
        team = top.get("constructor_name")
        if team:
            team_rows = season[(season["constructor_name"] == team)
                               & (season["round"] < current_round)]
            if len(team_rows) and team_rows["win_probability"].mean() > _CONSTRUCTOR_WIN_SHARE:
                reasons.append("Strong constructor")
    return reasons


def render() -> None:
    page_header("Race Center", "🏎",
                "Pick a race — see who the model backs, and who could spoil it.")
    sidebar_model_panel()

    races = list_races_or_stop()
    if not races:
        empty_state("No races available from the API.")
        return

    years = sorted({r["year"] for r in races}, reverse=True)
    with st.sidebar:
        st.divider()
        year = st.selectbox("Season", years, index=0)
        year_races = [r for r in races if r["year"] == year]
        race = st.selectbox(
            "Race", year_races,
            format_func=lambda r: metadata.race_label(
                r["race_id"], fallback_round=r["round"]),
        )

    body = api_get_or_stop(f"/predictions/{race['race_id']}")
    preds = body["predictions"]
    winner_id = body["actual_winner_driver_id"]

    facts = metadata.race_facts(race["race_id"])
    race_header(
        metadata.race_label(race["race_id"], fallback_round=body["round"]),
        body["year"], body["round"],
        circuit=facts.get("circuit"), date=facts.get("date"),
    )

    # Season context: rank trends vs the previous round + form/constructor
    # reasons. Cached per season (common.season_predictions), so only the
    # first race of a season pays the fetch.
    with st.spinner("Loading season context…"):
        season = season_predictions_or_stop(year)

    gq = metadata.grid_and_quali(race["race_id"])
    grid_by: dict[int, int] = {}
    quali_by: dict[int, int] = {}
    if not gq.empty:
        for row in gq.itertuples():
            if "grid" in gq.columns and pd.notna(row.grid):
                grid_by[int(row.driverId)] = int(row.grid)
            if "quali_position" in gq.columns and pd.notna(row.quali_position):
                quali_by[int(row.driverId)] = int(row.quali_position)

    trend_by: dict[int, int] = {}
    prob_trend_by: dict[int, float] = {}
    prev_round = body["round"] - 1
    if prev_round >= 1 and not season.empty:
        prev = season[season["round"] == prev_round]
        prev_rank = dict(zip(prev["driver_id"], prev["predicted_rank"], strict=True))
        prev_prob = dict(zip(prev["driver_id"], prev["win_probability"], strict=True))
        for p in preds:
            rank_before = prev_rank.get(p["driver_id"])
            if rank_before is not None:
                trend_by[p["driver_id"]] = int(rank_before) - int(p["predicted_rank"])
            prob_before = prev_prob.get(p["driver_id"])
            if prob_before is not None:
                prob_trend_by[p["driver_id"]] = p["win_probability"] - float(prob_before)

    # --- Hero: headline answer, dominant on page load, no scrolling needed.
    col_hero, col_outcome = st.columns([3, 1])
    with col_hero:
        favorite_card(preds[0], body["year"], body["round"], subtitle="Model favorite")
    with col_outcome:
        st.write("")
        st.write("")
        hit_miss_badge(body["model_top1_hit"])

    st.caption("🥇🥈🥉 Predicted podium")
    podium_row(preds, winner_id)

    st.caption("🔍 Why this confidence level")
    for reason in _confidence_reasons(
        preds[0], preds[1] if len(preds) > 1 else None,
        grid_by.get(preds[0]["driver_id"]),
    ):
        st.caption(f"• {reason}")

    st.subheader("🏁 Top contenders")
    cols = st.columns(min(5, len(preds)))
    for col, p in zip(cols, preds[:5], strict=False):
        with col:
            driver_card(
                p, grid=grid_by.get(p["driver_id"]),
                quali=quali_by.get(p["driver_id"]),
                trend=trend_by.get(p["driver_id"]),
                prob_trend=prob_trend_by.get(p["driver_id"]),
                is_winner=(winner_id is not None and p["driver_id"] == winner_id),
            )

    st.subheader("🧠 Why the model favors these drivers")
    top_n = preds[:3]
    any_reasons = False
    cols = st.columns(len(top_n))
    for col, p in zip(cols, top_n, strict=True):
        reasons = _reasons_for_favorite(
            p, grid_by.get(p["driver_id"]), quali_by.get(p["driver_id"]),
            season, body["round"],
        )
        any_reasons = any_reasons or bool(reasons)
        with col:
            st.markdown(f"**{driver_label(p)}**")
            if reasons:
                for reason in reasons:
                    st.caption(f"✅ {reason}")
            else:
                st.caption("No pre-race factor breakdown available.")
    if not any_reasons:
        st.caption("Display metadata not loaded — pre-race factor "
                   "breakdown is unavailable for this race.")
    st.caption("Indicative factors from pre-race data — full SHAP analysis "
               "on the Model Insights page.")

    st.subheader("⚠️ Risk factors")
    dnf = metadata.circuit_dnf_rate(race["race_id"])
    if dnf:
        st.caption(
            f"**DNF rate at this circuit: {dnf['dnf_rate']:.0%}** — "
            f"{dnf['n_races']} races, {dnf['years']}. The only risk signal "
            "in this dataset with real underlying data (no tire-degradation "
            "or safety-car data exists in this project's source)."
        )
    else:
        st.caption("Not enough circuit history yet for a DNF-rate risk signal.")

    fact_items = [
        {"label": label, "value": str(facts[key])}
        for label, key in (("Laps", "laps"), ("Pole time", "pole_time"),
                           ("Fastest lap", "fastest_lap"),
                           ("Circuit", "circuit"), ("Country", "country"))
        if key in facts
    ]
    if fact_items:
        st.subheader("📋 Race facts")
        stat_row(fact_items[:5])

    st.subheader("📊 Full field")
    frame = pd.DataFrame(preds)
    frame["label"] = [driver_label(p) for p in preds]
    frame["is_winner"] = frame["driver_id"] == winner_id
    top10, rest = frame.head(10), frame.iloc[10:]

    win_share_bar(top10, winner_id=winner_id)
    st.caption("Red outline = actual winner; bars use constructor colors. "
               "Tied shares are normal — similar cars land on the same "
               "probability step; ranking uses a deterministic tiebreak.")

    if len(rest):
        with st.expander(f"Rest of the field ({len(rest)} drivers)"):
            st.dataframe(
                rest[["predicted_rank", "label", "win_probability",
                      "win_probability_raw"]],
                hide_index=True, width="stretch",
            )

    with st.expander("Full field table"):
        st.dataframe(
            frame[["predicted_rank", "label", "win_probability",
                   "win_probability_raw", "is_winner"]],
            hide_index=True, width="stretch",
        )
    st.caption(f"prediction_id: `{body['prediction_id']}` · generated "
               f"{body['generated_at']} · model v{body['model']['version']}")
