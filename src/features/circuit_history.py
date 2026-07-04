"""
src/features/circuit_history.py

Circuit history — how a driver/constructor has performed at THIS circuit on
prior visits only, any season before the current race
(reports/master_dataset_design.md Section 5.6).

Leakage rule: "prior visits only" is implemented with the cumulative-minus-
current pattern (`cumsum() - current_value`, `cumcount()`) over rows sorted by
(year, round) — algebraically identical to shift(1)+expanding, and never
includes the current race's own outcome.

Constructor circuit wins are aggregated to (constructorId, raceId) grain
FIRST, for the same reason as constructor_form.py: a row-level cumulative sum
would leak the teammate's result from the current race.

Known limitation (documented in the design doc, not worked around): many
(driver, circuit) pairs have 0-2 prior visits, so these features are sparse
and noisy for rookies and at new circuits. First visit ever = 0 starts /
0 wins / NaN average finish.
"""

from __future__ import annotations

import pandas as pd

CIRCUIT_HISTORY_FEATURES: tuple[str, ...] = (
    "driver_circuit_starts",
    "driver_circuit_wins",
    "driver_circuit_avg_finish",
    "constructor_circuit_wins",
)


def add_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add circuit-history features to a (raceId, driverId)-grain frame.

    Requires columns: raceId, driverId, constructorId, circuitId, year, round,
    winner, positionOrder. Returns a copy sorted chronologically by
    (year, round); row count unchanged.
    """
    out = df.sort_values(["year", "round", "raceId"], kind="mergesort").copy()

    # --- Driver-level: (raceId, driverId) is unique per master-dataset grain,
    # so cumulative-minus-current at row level cannot see same-race rows.
    grp = out.groupby(["driverId", "circuitId"], sort=False)
    starts = grp.cumcount()
    out["driver_circuit_starts"] = starts

    winner = out["winner"].astype(float)
    out["driver_circuit_wins"] = grp["winner"].cumsum().astype(float) - winner

    finish = out["positionOrder"].astype(float)
    prior_finish_sum = grp["positionOrder"].cumsum().astype(float) - finish
    out["driver_circuit_avg_finish"] = prior_finish_sum / starts.where(starts > 0)

    # --- Constructor-level: aggregate to one row per (constructorId, raceId)
    # before the cumulative sum, so the teammate's same-race result is
    # excluded (see module docstring).
    race_level = (
        out.assign(_win=winner)
        .groupby(["constructorId", "raceId"], as_index=False)
        .agg(
            circuitId=("circuitId", "first"),
            year=("year", "first"),
            round=("round", "first"),
            _race_win=("_win", "max"),
        )
        .sort_values(["year", "round", "raceId"], kind="mergesort")
    )
    circuit_grp = race_level.groupby(["constructorId", "circuitId"], sort=False)
    race_level["constructor_circuit_wins"] = (
        circuit_grp["_race_win"].cumsum() - race_level["_race_win"]
    )

    merged = out.merge(
        race_level[["constructorId", "raceId", "constructor_circuit_wins"]],
        on=["constructorId", "raceId"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(out):
        raise ValueError(
            "Circuit-history merge changed row count "
            f"({len(out):,} -> {len(merged):,}) — (constructorId, raceId) "
            "aggregation is no longer unique."
        )
    return merged
