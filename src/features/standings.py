"""
src/features/standings.py

Lagged championship standings — the highest-severity leakage risk in the
whole design (reports/master_dataset_design.md Section 5.7 / 6.2 failure
mode 2).

Why the lag is mandatory: a row in driver_standings.csv / constructor_
standings.csv keyed by raceId X reflects the standing AFTER race X was run.
Joining it directly onto race X's feature row bakes race X's own result into
its own features (a driver who won race X shows that win in their "current"
standing). Standings must therefore be joined at the PREVIOUS race.

Implementation: build a chronological calendar of the races present in the
modeling frame, sorted by (year, round), and take `prev_raceId = shift(1)`.
This single shift implements BOTH required rules at once:
  - mid-season: race at round N joins standings as of round N-1;
  - round 1 of a season: the previous calendar row is the FINAL race of the
    prior season, i.e. the prior season's final standings — the explicit
    round-1 rule the design doc demands;
  - the first-ever race (or a driver's/constructor's first appearance) has no
    prior standings row and correctly gets nulls, not zeros.

Per Decision 010, standings are read directly from the raw CSVs
(driver_standings.csv / constructor_standings.csv) — they are deliberately
NOT in master_dataset.parquet, and this module must never expect them there.

Standing POSITION (rank) is the primary feature, not raw points — the 2010
points-system change breaks raw-point comparability across seasons
(Decision 008 rationale, design doc Section 5.7); points are kept as a
secondary, same-lag value.
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import load_csv

STANDINGS_FEATURES: tuple[str, ...] = (
    "driver_standing_position_prev",
    "driver_standing_points_prev",
    "driver_standing_wins_prev",
    "constructor_standing_position_prev",
    "constructor_standing_points_prev",
)

_DRIVER_STANDINGS_RENAME = {
    "position": "driver_standing_position_prev",
    "points": "driver_standing_points_prev",
    "wins": "driver_standing_wins_prev",
}
_CONSTRUCTOR_STANDINGS_RENAME = {
    "position": "constructor_standing_position_prev",
    "points": "constructor_standing_points_prev",
}


def load_standings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (driver_standings, constructor_standings) from the raw CSVs."""
    return load_csv("driver_standings.csv"), load_csv("constructor_standings.csv")


def build_prev_race_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map each raceId to the raceId of the chronologically previous race.

    Built from the (raceId, year, round) triples present in `df`, sorted by
    (year, round). The first race in the frame maps to <NA>. Raises if the
    same (year, round) appears under two raceIds — chronological order would
    be ambiguous and every lag downstream would be suspect.
    """
    calendar = (
        df[["raceId", "year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"], kind="mergesort")
        .reset_index(drop=True)
    )
    if calendar["raceId"].duplicated().any():
        raise ValueError(
            "A raceId appears with more than one (year, round) — "
            "the race calendar is inconsistent."
        )
    dup_slots = calendar.duplicated(subset=["year", "round"])
    if dup_slots.any():
        sample = calendar.loc[dup_slots, ["year", "round"]].head(5).to_dict("records")
        raise ValueError(
            f"Multiple raceIds share the same (year, round) slot (sample: {sample}) — "
            "chronological ordering is ambiguous, cannot lag standings safely."
        )
    calendar["raceId"] = calendar["raceId"].astype("Int64")
    calendar["prev_raceId"] = calendar["raceId"].shift(1)
    return calendar[["raceId", "prev_raceId"]]


def _lagged_join(
    out: pd.DataFrame,
    standings: pd.DataFrame,
    entity_key: str,
    columns_rename: dict[str, str],
    table_name: str,
) -> pd.DataFrame:
    """Left-join standings columns at (prev_raceId, entity_key)."""
    cols = ["raceId", entity_key] + list(columns_rename)
    lagged = standings[cols].rename(columns={"raceId": "prev_raceId", **columns_rename})
    # Normalize key dtypes to nullable Int64 so the <NA> prev_raceId of the
    # first race merges cleanly (an NA key simply matches nothing).
    for key in ("prev_raceId", entity_key):
        lagged[key] = lagged[key].astype("Int64")
    merged = out.merge(
        lagged, on=["prev_raceId", entity_key], how="left", validate="many_to_one"
    )
    if len(merged) != len(out):
        raise ValueError(
            f"Standings join against '{table_name}' changed row count "
            f"({len(out):,} -> {len(merged):,})."
        )
    return merged


def add_standings_features(
    df: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add round-(N-1)-lagged standings features to a (raceId, driverId)-grain
    frame.

    Requires df columns: raceId, driverId, constructorId, year, round.
    `driver_standings` / `constructor_standings` are the raw Ergast CSV frames
    (see load_standings()); they are keyed by the raceId AFTER which the
    standing applies. Returns a copy with STANDINGS_FEATURES added; row count
    and order unchanged.
    """
    prev_map = build_prev_race_map(df)

    out = df.copy()
    out["raceId"] = out["raceId"].astype("Int64")
    out = out.merge(prev_map, on="raceId", how="left", validate="many_to_one")
    for key in ("driverId", "constructorId"):
        out[key] = out[key].astype("Int64")

    out = _lagged_join(
        out, driver_standings, "driverId", _DRIVER_STANDINGS_RENAME,
        "driver_standings",
    )
    out = _lagged_join(
        out, constructor_standings, "constructorId", _CONSTRUCTOR_STANDINGS_RENAME,
        "constructor_standings",
    )

    return out.drop(columns=["prev_raceId"])
