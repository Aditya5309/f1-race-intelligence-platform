"""Race Center — storytelling-first race view (UI/UX redesign v2).

Leads with the Grand Prix identity, the model's favorite and how confident
it is, the top contenders as cards, and a plain-language "why" — the full
field chart and tables follow as supporting detail. Predictions come only
from GET /races + GET /predictions/{race_id}; Grand Prix names, grids, and
race facts are display metadata (app/views/metadata.py) and
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
    predict_upcoming_or_stop,
    season_predictions_or_stop,
    sidebar_model_panel,
    upcoming_race_or_none,
)
from app.views.components import (
    confidence_tier,
    driver_card,
    driver_label,
    empty_state,
    favorite_card,
    hit_miss_detail,
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


def _normalize_upcoming_response(raw: dict, race_id: int) -> dict:
    """View-model adapter: UpcomingPredictionResponse (POST
    /predict) -> the same shape PredictionResponse (GET /predictions/
    {race_id}) already provides, PLUS the upcoming-only extras
    (materialization_status, missing_inputs, caveats, provenance) — so
    every existing rendering function below keeps reading `body[...]`
    exactly as it does for a historical race, completely unmodified.
    `actual_winner_driver_id`/`model_top1_hit` are always None here: a
    race with no result has no winner to compare against by definition —
    the correct value, not a degraded placeholder. `race_id` isn't part
    of UpcomingPredictionResponse's own schema (deliberately kept out of
    the API contract, since every route already knows which race it's
    scoring) but the picker already knows it, so it's threaded
    through here rather than re-derived."""
    return {
        **raw,
        "race_id": race_id,
        "actual_winner_driver_id": None,
        "model_top1_hit": None,
    }


def _upcoming_status_banner(status: str, missing_inputs: list[str],
                            caveats: list[str], provenance: dict) -> None:
    """Materialization status + prediction caveats + ETL-snapshot
    freshness for the upcoming-race view. Built from existing Streamlit
    primitives (st.info/st.warning/st.caption, the same ones empty_state()
    and every other caveat elsewhere on this page already use) — a
    single-use panel didn't warrant a new shared components.py entry."""
    if status == "pre_qualifying":
        missing = ", ".join(missing_inputs) if missing_inputs else "qualifying data"
        st.warning(
            "⚠️ **Provisional prediction** — qualifying hasn't fully "
            f"completed for this race yet ({missing} still missing). The "
            "model's own missing-value handling covers the gap, but treat "
            "this as early, not final."
        )
    else:
        st.info(
            "🔮 **Pre-race prediction** for the next Grand Prix — the grid "
            "is set, but this race hasn't been run yet."
        )
    for caveat in caveats:
        st.caption(f"• {caveat}")
    st.caption(f"Data as of {provenance['etl_snapshot_version']}.")


def _provenance_expander(provenance: dict) -> None:
    """Full structured provenance detail — every prediction must be
    reconstructable later from its own recorded metadata alone. The
    compact footer caption (existing, unchanged) covers the casual reader;
    this is for anyone who wants to audit exactly what produced this
    prediction. Reuses st.expander — already used for "Full field table"/
    "Rest of the field" below."""
    with st.expander("🔎 Prediction provenance"):
        st.caption(f"Model: {provenance['model_version']} @ {provenance['model_alias']}")
        st.caption(f"Feature schema: `{provenance['feature_schema_version']}`")
        st.caption(f"ETL snapshot: {provenance['etl_snapshot_version']}")
        st.caption(f"Data as of: {provenance['data_as_of']}")
        st.caption(f"Materialized: {provenance['materialized_at']}")
        st.caption(f"Predicted: {provenance['predicted_at']}")
        st.caption(f"Qualifying status: {provenance['qualifying_status']}")
        st.caption(f"Completeness: {provenance['completeness_status']}")


def render() -> None:
    page_header("Race Center", "🏎",
                "Pick a race — see who the model backs, and who could spoil it.")
    sidebar_model_panel()

    races = list_races_or_stop()
    if not races:
        empty_state("No races available from the API.")
        return

    # Default landing race (first load, no ?race_id= param): the latest
    # COMPLETED race — computed BEFORE the upcoming race (if any) is
    # appended below, so this stays byte-for-byte what it was before
    # upcoming-race prediction existed, even though the upcoming race is
    # now also selectable.
    historical_sorted = sorted(races, key=lambda r: (r["year"], r["round"]))
    default_race_id = historical_sorted[-1]["race_id"]

    # The single next race with no result yet, if resolvable — additive
    # only. upcoming_race_or_none() soft-fails (never st.stop()s) so an
    # unavailable lookup (e.g. this deployment has no data/ tree) leaves
    # everything below byte-for-byte what it is today.
    upcoming = upcoming_race_or_none()
    upcoming_race_id = upcoming["race_id"] if upcoming else None
    if upcoming is not None:
        races = [*races, {
            "race_id": upcoming["race_id"], "year": upcoming["year"],
            "round": upcoming["round"], "n_drivers": None,
        }]

    years = sorted({r["year"] for r in races}, reverse=True)
    race_by_id = {r["race_id"]: r for r in races}
    # Chronological across every served season, for Prev/Next to roll
    # cleanly from one season's last race into the next season's round 1
    # (and, now, into the upcoming race if one was resolved).
    all_sorted = sorted(races, key=lambda r: (r["year"], r["round"]))
    ordered_ids = [r["race_id"] for r in all_sorted]

    if "race_id" not in st.session_state:
        # Shareable/bookmarkable: a ?race_id=<id> link pre-selects that race
        # on first load (including the upcoming race's real raceId) —
        # falls back to the latest COMPLETED race when absent/invalid.
        param_race_id = st.query_params.get("race_id")
        if param_race_id is not None:
            try:
                candidate = int(param_race_id)
            except ValueError:
                candidate = None
            if candidate in race_by_id:
                default_race_id = candidate
        st.session_state["race_id"] = default_race_id
    elif st.session_state["race_id"] not in race_by_id:
        # A previously-valid selection is no longer resolvable THIS run —
        # in practice always the transient upcoming slot (it finished, or
        # the resolved upcoming race rotated to a different one; every
        # historical entry is permanent once added, so this can't happen
        # for one of those). An invalid transient selection, not an
        # exception to raise: fall back to the current sensible default
        # rather than let ordered_ids.index() below raise ValueError.
        st.session_state["race_id"] = default_race_id

    with st.sidebar:
        st.divider()
        pos = ordered_ids.index(st.session_state["race_id"])
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("◀ Prev", disabled=(pos == 0), width="stretch"):
                st.session_state["race_id"] = ordered_ids[pos - 1]
        with col_next:
            if st.button("Next ▶", disabled=(pos == len(ordered_ids) - 1),
                        width="stretch"):
                st.session_state["race_id"] = ordered_ids[pos + 1]

        current = race_by_id[st.session_state["race_id"]]
        year = st.selectbox("Season", years, index=years.index(current["year"]))
        year_races = [r for r in races if r["year"] == year]
        if year != current["year"]:
            # User changed the season dropdown directly — jump to that
            # season's first race rather than trying to preserve a round
            # number that may not exist in the new season.
            st.session_state["race_id"] = year_races[0]["race_id"]
        race_index = next(
            (i for i, r in enumerate(year_races)
             if r["race_id"] == st.session_state["race_id"]), 0,
        )

        def _race_label(r: dict) -> str:
            label = metadata.race_label(r["race_id"], fallback_round=r["round"])
            return f"🔮 {label} (upcoming)" if r["race_id"] == upcoming_race_id else label

        race = st.selectbox(
            "Race", year_races, index=race_index, format_func=_race_label,
        )
        st.session_state["race_id"] = race["race_id"]
        st.query_params["race_id"] = str(race["race_id"])

    is_upcoming = race["race_id"] == upcoming_race_id
    if is_upcoming:
        raw = predict_upcoming_or_stop(race["year"], race["round"])
        body = _normalize_upcoming_response(raw, race["race_id"])
    else:
        body = api_get_or_stop(f"/predictions/{race['race_id']}")
    preds = body["predictions"]
    winner_id = body["actual_winner_driver_id"]

    facts = metadata.race_facts(race["race_id"])
    race_header(
        metadata.race_label(race["race_id"], fallback_round=body["round"]),
        body["year"], body["round"],
        circuit=facts.get("circuit"), date=facts.get("date"),
    )
    if is_upcoming:
        _upcoming_status_banner(
            body["materialization_status"], body["missing_inputs"],
            body["caveats"], body["provenance"])

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

    top = preds[0]
    top_result = (
        metadata.actual_result(race["race_id"], top["driver_id"])
        if winner_id is not None else {}
    )

    # --- Hero: headline answer, dominant on page load, no scrolling needed.
    col_hero, col_outcome = st.columns([3, 1])
    with col_hero:
        favorite_card(top, body["year"], body["round"], subtitle="Model favorite")
    with col_outcome:
        st.write("")
        st.write("")
        hit_miss_detail(body["model_top1_hit"], top_result)

    st.caption("🥇🥈🥉 Predicted podium")
    podium_row(preds, winner_id)

    st.caption("🔍 Why this confidence level")
    confidence_reasons = _confidence_reasons(
        preds[0], preds[1] if len(preds) > 1 else None,
        grid_by.get(preds[0]["driver_id"]),
    )
    if is_upcoming and body["materialization_status"] == "pre_qualifying":
        # _confidence_reasons() itself is unmodified (shared with the
        # historical path) — the provisional note is prepended at the
        # call site only, so historical rendering can't be affected.
        confidence_reasons = ["⚠️ Provisional — qualifying not yet complete",
                               *confidence_reasons]
    for reason in confidence_reasons:
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
            f"🛞 **DNF rate at this circuit: {dnf['dnf_rate']:.0%}** — "
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

    if winner_id is not None:
        st.subheader("🔁 Historical Replay")
        st.caption(f"**Predicted:** {driver_label(top)} — "
                   f"{top['win_probability']:.1%} win share (model favorite)")
        if not top_result:
            st.caption("**Actual:** result unavailable (display metadata not loaded).")
        elif top_result["dnf"]:
            status = top_result.get("status", "did not finish")
            st.caption(f"**Actual:** retired — {status}")
            st.caption(
                f"**Error analysis:** the model's favorite retired ({status}) — "
                "a DNF is a genuine outcome no pre-race signal in this model "
                "predicts, not a ranking error."
            )
        else:
            st.caption(f"**Actual:** finished P{top_result['finish_position']}")
            if body["model_top1_hit"]:
                st.caption("**Result:** the model's favorite won as predicted.")
            else:
                st.caption(
                    f"**Error analysis:** {top['win_probability']:.1%} win share "
                    f"→ P{top_result['finish_position']} — the gap between predicted "
                    "confidence and actual result (see Top contenders above for "
                    "who did win)."
                )

    st.subheader("📊 Full field")
    frame = pd.DataFrame(preds)
    frame["label"] = [driver_label(p) for p in preds]
    frame["is_winner"] = frame["driver_id"] == winner_id
    frame["grid"] = frame["driver_id"].map(grid_by)
    # Zero-padded so the interactive table's column-header click-to-sort
    # (which sorts these display strings lexicographically, not numerically)
    # still produces correct grid order: "02" < "10" as strings, unlike
    # unpadded "2" > "10". PL/missing sort after every real grid slot since
    # digit characters are lexicographically below both.
    frame["grid_display"] = frame["grid"].map(
        lambda g: "PL" if g == 0 else (f"{int(g):02d}" if pd.notna(g) else "—"))
    # Pit-lane starts (grid 0) and missing grids aren't a meaningful "gained
    # N places" delta — only compute it for a real numbered grid slot.
    frame["grid_delta"] = frame.apply(
        lambda r: int(r["grid"] - r["predicted_rank"])
        if pd.notna(r["grid"]) and r["grid"] != 0 else None,
        axis=1,
    )
    frame["confidence_tier"] = frame["win_probability"].map(
        lambda p: confidence_tier(p)[0])
    top10, rest = frame.head(10), frame.iloc[10:]

    win_share_bar(top10, winner_id=winner_id)
    st.caption("Red outline = actual winner; bars use constructor colors. "
               "Tied shares are normal — similar cars land on the same "
               "probability step; ranking uses a deterministic tiebreak.")

    table_columns = ["predicted_rank", "label", "win_probability", "grid_display",
                     "grid_delta", "confidence_tier"]
    column_config = {
        "predicted_rank": "Rank", "label": "Driver", "grid_display": "Grid",
        "grid_delta": st.column_config.NumberColumn(
            "Grid → Rank", help="Grid position minus predicted rank; "
            "positive means the model expects them to gain places."),
        "confidence_tier": "Confidence",
        "win_probability": st.column_config.NumberColumn(
            "Win share", format="percent", step=0.001),
    }

    if len(rest):
        with st.expander(f"Rest of the field ({len(rest)} drivers)"):
            st.dataframe(
                rest[table_columns], hide_index=True, width="stretch",
                column_config=column_config,
            )

    with st.expander("Full field table"):
        st.dataframe(
            frame[[*table_columns, "win_probability_raw", "is_winner"]],
            hide_index=True, width="stretch", column_config=column_config,
        )

    if is_upcoming:
        # Both sections below need a completed race's frozen features row
        # (GET /predictions/{race_id}/vs-baseline and .../simulate/{id}
        # both 404 for a race with no result yet) — explained, not
        # silently broken or omitted without a word.
        st.divider()
        st.caption(
            "🧮 Qualifying Impact and 🎛️ the Grid Simulator need a completed "
            "race to compare against — check back after this one runs."
        )
    else:
        _qualifying_impact_section(race["race_id"], winner_id)
        _grid_simulator_section(race["race_id"], preds, grid_by)

    st.caption(f"prediction_id: `{body['prediction_id']}` · generated "
               f"{body['generated_at']} · model v{body['model']['version']}")
    if is_upcoming:
        _provenance_expander(body["provenance"])


def _qualifying_impact_section(race_id: int, winner_id: int | None) -> None:
    """How much of the model's pick is 'just pole position'
    vs. the rest of the model (form/circuit history/standings). Grounded
    entirely in real, already-registered artifacts: the calibrated model and
    MODEL_ZOO['pole_baseline'] (a grid-only heuristic) —
    no fabricated FP1-FP3 progression."""
    st.divider()
    st.subheader("🧮 Qualifying Impact: model vs. pole-only baseline")
    st.caption(
        "Left: the full model (recent form + circuit history + championship "
        "standings + qualifying). Right: a heuristic that knows nothing but "
        "who's on pole — the bar every trained model has to beat. Comparing "
        "them shows how much of the model's pick is really just qualifying."
    )
    vs_body = api_get_or_stop(f"/predictions/{race_id}/vs-baseline")

    model_field = vs_body["model_predictions"]
    baseline_field = vs_body["baseline_predictions"]
    model_frame = pd.DataFrame(model_field)
    model_frame["label"] = [driver_label(p) for p in model_field]
    baseline_frame = pd.DataFrame(baseline_field)
    baseline_frame["label"] = [driver_label(p) for p in baseline_field]

    col_model, col_base = st.columns(2)
    with col_model:
        st.markdown("**Full model**")
        win_share_bar(model_frame.sort_values("predicted_rank").head(8),
                     winner_id=winner_id, height=320)
    with col_base:
        st.markdown("**Pole-only baseline**")
        win_share_bar(baseline_frame.sort_values("predicted_rank").head(8),
                     winner_id=winner_id, height=320)
        st.caption("100% for the pole sitter, 0% for everyone else — that's "
                   "the whole rule.")

    agree = model_field[0]["driver_id"] == baseline_field[0]["driver_id"]
    if agree:
        st.caption("✅ Same top pick as the pole-only baseline this race — "
                   f"no extra edge from form/history/standings at the very "
                   f"top here. {ERA_CAVEAT_SHORT}")
    else:
        st.caption(
            f"🔀 The full model backs **{driver_label(model_field[0])}** "
            f"over the pole-only pick **{driver_label(baseline_field[0])}** "
            "— form, circuit history, or standings are doing real work here."
        )
    if vs_body["actual_winner_driver_id"] is not None:
        model_mark = "✅ hit" if vs_body["model_top1_hit"] else "❌ missed"
        baseline_mark = "✅ hit" if vs_body["baseline_top1_hit"] else "❌ missed"
        st.caption(f"Result: model {model_mark} · pole-only baseline {baseline_mark}")


def _grid_simulator_section(race_id: int, preds: list[dict],
                            grid_by: dict[int, int]) -> None:
    """Prediction Simulator, grid/qualifying group only.

    Only the starting grid slot is adjustable; the driver's real qualifying
    times/gap-to-pole and all 21 historical form/circuit/standings features
    stay locked to what actually happened for this race (backend enforces
    this — see app/api.py's ADJUSTABLE_GRID_FEATURES). This is deliberately
    NOT a general what-if tool."""
    st.divider()
    st.subheader("🎛️ What if? Grid Position Simulator")
    st.caption(
        "Move one driver's starting grid slot and see their win share "
        "change. Everything else — recent form, circuit history, "
        "championship standing, even the rest of that driver's real "
        "qualifying session — stays locked to what actually happened. "
        "Grid position can genuinely differ from qualifying pace in real F1 "
        "(penalties, reshuffles) — that's exactly the scenario simulated "
        "here, not a fabricated qualifying lap."
    )

    driver_ids = [p["driver_id"] for p in preds]
    label_by_id = {p["driver_id"]: driver_label(p) for p in preds}
    sim_driver = st.selectbox(
        "Driver", driver_ids, index=0, format_func=lambda did: label_by_id[did],
        key="sim_driver_select",
    )
    field_size = len(preds)
    default_grid = grid_by.get(sim_driver) or 1
    default_grid = min(max(int(default_grid), 1), field_size)

    col_slider, col_pit = st.columns([3, 1])
    with col_slider:
        sim_grid_position = st.slider(
            "Starting grid position", min_value=1, max_value=field_size,
            value=default_grid, key="sim_grid_slider",
        )
    with col_pit:
        st.write("")
        sim_pit_lane = st.checkbox("Pit lane start", key="sim_pit_lane")

    params = {"pit_lane": True} if sim_pit_lane else {"grid_position": sim_grid_position}
    sim_body = api_get_or_stop(
        f"/predictions/{race_id}/simulate/{sim_driver}", params=params)

    col_real, col_sim, col_grid = st.columns(3)
    with col_real:
        st.metric("Real win share", f"{sim_body['real_win_probability']:.1%}")
    with col_sim:
        sim_label = "Pit lane" if sim_pit_lane else f"P{sim_grid_position}"
        delta = sim_body["simulated_win_probability"] - sim_body["real_win_probability"]
        st.metric(f"Simulated win share ({sim_label})",
                  f"{sim_body['simulated_win_probability']:.1%}", delta=f"{delta:+.1%}")
    with col_grid:
        real_grid = sim_body["real_grid_position"]
        st.metric("Real starting grid",
                  f"P{int(real_grid)}" if real_grid is not None else "—")

    st.caption(f"🔒 Locked for this simulation: {sim_body['driver_name'] or 'this driver'}'s "
              f"real qualifying times/gap-to-pole ({len(sim_body['locked_qualifying_features'])} "
              f"fields) and all {len(sim_body['locked_features'])} historical "
              "form/circuit-history/standings features — only the grid slot moves.")

    sim_frame = pd.DataFrame(sim_body["field"])
    sim_frame["label"] = [driver_label(p) for p in sim_body["field"]]
    win_share_bar(sim_frame.sort_values("predicted_rank").head(10), winner_id=None)
    st.caption("Only the simulated driver's underlying model output actually "
              "changes — everyone else's bar shifts a little because shares "
              "are renormalized to sum to 100% within the race.")
