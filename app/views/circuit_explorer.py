"""Circuit Explorer — one circuit's history.

Picker by circuit name; every stat here is derived from results.csv /
qualifying.csv / circuits.csv — no lap-by-lap telemetry exists anywhere in
this project's source data, so there's no "average speed" or invented
difficulty rating, only the proxies computed in
app/views/metadata.py::circuit_stats(), each captioned with what it
actually measures and doesn't.

Track-outline geometry (scripts/backfill_circuit_layouts.py)
is a separate, best-effort OpenStreetMap enrichment — not every circuit has
one (see metadata.circuit_layout()'s docstring), so the outline section
renders only when available and stays silent otherwise.
"""

from __future__ import annotations

import streamlit as st

from app.views import metadata
from app.views.charts import circuit_layout_map, standings_bar
from app.views.common import sidebar_model_panel
from app.views.components import empty_state, page_header, stat_row


def render() -> None:
    page_header("Circuit Explorer", "🏟",
                "One circuit's history — winners, risk, and pace over time.")
    sidebar_model_panel()

    catalog = metadata.circuit_catalog()
    if catalog.empty:
        empty_state("Circuit metadata unavailable (display data not loaded).")
        return

    circuit_ids = catalog["circuitId"].tolist()
    label_by_id = dict(zip(
        catalog["circuitId"],
        catalog["name"] + " (" + catalog["country"].fillna("—") + ")",
        strict=True,
    ))
    circuit_id = st.selectbox(
        "Circuit", circuit_ids, format_func=lambda cid: label_by_id[cid])

    row = catalog[catalog["circuitId"] == circuit_id].iloc[0]
    st.subheader(f"📍 {row['name']}")
    st.caption(f"{row['location']}, {row['country']}")

    layout = metadata.circuit_layout(circuit_id)
    if layout is not None:
        circuit_layout_map(layout)
        st.caption(f"Track outline: {layout['properties']['attribution']} "
                   "(ODbL 1.0) — via scripts/backfill_circuit_layouts.py.")

    stats = metadata.circuit_stats(circuit_id)
    if not stats:
        empty_state("No race history for this circuit in the served data.")
        return

    stat_row([
        {"label": "Races held", "value": str(stats["races_held"])},
        {"label": "Active", "value": f"{stats['first_year']}–{stats['last_year']}"},
        {"label": "DNF rate", "value": f"{stats['dnf_rate']:.0%}" if "dnf_rate" in stats else "—",
         "help": "The only risk signal backed by real data in this project's "
                 "source — no tire-degradation or safety-car data exists here."},
    ])

    if "avg_position_change" in stats:
        st.caption(
            f"🔀 **Average grid → finish change: {stats['avg_position_change']:+.1f} "
            "places** — a proxy shaped by strategy, reliability, and grid "
            "penalties as much as genuine overtaking difficulty, not a pure "
            "\"how hard is this track to pass on\" number."
        )

    if "fastest_lap" in stats:
        fl = stats["fastest_lap"]
        st.caption(
            f"🏁 **Fastest recorded lap: {fl['time']}** by {fl['driver']} "
            f"({fl['year']}) — reflects car performance evolution since "
            f"{stats['first_year']}, not a fixed measure of circuit "
            "difficulty; Ergast also doesn't track mid-history layout changes."
        )

    most_wins = stats.get("most_wins")
    if most_wins is not None and len(most_wins):
        st.subheader("🏆 Most successful drivers here")
        standings_bar(most_wins, "driver", "wins",
                     height=max(220, 40 * len(most_wins)))

    winners = stats.get("winners_by_year")
    if winners is not None and len(winners):
        with st.expander(f"Winners by year ({len(winners)} races)"):
            st.dataframe(
                winners, hide_index=True, width="stretch",
                column_config={"year": "Year", "driver": "Driver",
                                "constructor": "Constructor"},
            )
