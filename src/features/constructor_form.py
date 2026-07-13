"""
src/features/constructor_form.py

Rolling constructor form.

A constructor fields (usually) two cars per race, so the master dataset has
two rows per (constructorId, raceId). Rolling directly over driver-grain rows
would create TWO bugs:

1. A "last 5" window would cover 5 rows ≈ 2.5 races, not 5 races.
2. shift(1) at row level would let a row see its TEAMMATE's result from the
   SAME race — an outcome of the race being predicted, i.e. leakage.

So this module first aggregates outcomes to (constructorId, raceId) grain —
one row per constructor per race — then applies the shift-before-rolling
pattern over races, and finally left-joins the race-level features back onto
the driver-grain rows (both teammates get the same constructor-form values,
which is correct: they share the car).

Like driver form, ordering is strictly (year, round) and a constructor's
first-ever race gets NaN, not 0.
"""

from __future__ import annotations

import pandas as pd

CONSTRUCTOR_FORM_WINDOWS: tuple[int, ...] = (3, 5)

CONSTRUCTOR_FORM_FEATURES: tuple[str, ...] = (
    "constructor_wins_last_3", "constructor_wins_last_5",
    "constructor_podiums_last_5",
    "constructor_dnf_rate_last_5",
)


def _prior_rolling(
    values: pd.Series, group_key: pd.Series, window: int, agg: str,
) -> pd.Series:
    """shift(1)-then-roll within each group — see driver_form._prior_rolling."""
    return values.groupby(group_key, sort=False).transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), agg)()
    )


def add_constructor_form_features(
    df: pd.DataFrame, windows: tuple[int, ...] = CONSTRUCTOR_FORM_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling constructor-form features to a (raceId, driverId)-grain frame.

    Requires columns: raceId, driverId, constructorId, year, round, winner,
    positionOrder, finished. Returns a copy with CONSTRUCTOR_FORM_FEATURES
    added; row count and order unchanged. `windows` applies to the win counts;
    podiums and DNF rate use a fixed 5-race window.
    """
    out = df.copy()

    dnf = (~out["finished"].astype("boolean").fillna(False)).astype(float)
    race_level = (
        out.assign(
            _win=out["winner"].astype(float),
            _podium=(out["positionOrder"] <= 3).astype(float),
            _dnf=dnf,
        )
        .groupby(["constructorId", "raceId"], as_index=False)
        .agg(
            year=("year", "first"),
            round=("round", "first"),
            _race_win=("_win", "max"),        # constructor won this race (either car)
            _race_podiums=("_podium", "sum"),  # podium finishes across its cars
            _race_dnf_rate=("_dnf", "mean"),   # reliability proxy across its cars
        )
        .sort_values(["year", "round", "raceId"], kind="mergesort")
    )

    constructor = race_level["constructorId"]
    for w in windows:
        race_level[f"constructor_wins_last_{w}"] = _prior_rolling(
            race_level["_race_win"], constructor, w, "sum"
        )
    race_level["constructor_podiums_last_5"] = _prior_rolling(
        race_level["_race_podiums"], constructor, 5, "sum"
    )
    race_level["constructor_dnf_rate_last_5"] = _prior_rolling(
        race_level["_race_dnf_rate"], constructor, 5, "mean"
    )

    feature_cols = [f"constructor_wins_last_{w}" for w in windows] + [
        "constructor_podiums_last_5", "constructor_dnf_rate_last_5",
    ]
    merged = out.merge(
        race_level[["constructorId", "raceId"] + feature_cols],
        on=["constructorId", "raceId"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(out):
        raise ValueError(
            "Constructor-form merge changed row count "
            f"({len(out):,} -> {len(merged):,}) — (constructorId, raceId) "
            "aggregation is no longer unique."
        )
    return merged
