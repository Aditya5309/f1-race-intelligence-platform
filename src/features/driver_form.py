"""
src/features/driver_form.py

Rolling driver form — trailing-window statistics over each driver's PRIOR
races only.

Leakage rule: every rolling window is
computed on `shift(1)` within the driver's chronologically ordered race
history, so the current race's own result is never inside its own window.
Ordering is by (year, round) — never raceId, which is not guaranteed to sort
chronologically across eras — and the shift naturally spans season boundaries
(round 1 of a season looks back at the previous season's races).

A driver's first-ever race has no prior history: its rolling features are NaN
(not 0 — "no information" and "0 wins in the last 5" are different signals).

Mid-season constructor changes are intentionally ignored here: these are
DRIVER features and correctly span a mid-season team switch — do not "fix"
this by grouping on constructorId.
"""

from __future__ import annotations

import pandas as pd

DRIVER_FORM_WINDOWS: tuple[int, ...] = (3, 5, 10)

DRIVER_FORM_FEATURES: tuple[str, ...] = (
    "driver_wins_last_3", "driver_wins_last_5", "driver_wins_last_10",
    "driver_podiums_last_5",
    "driver_avg_finish_last_5",
    "driver_dnf_rate_last_5",
    "driver_points_last_5",
    "driver_experience_races",
)


def _prior_rolling(
    values: pd.Series, group_key: pd.Series, window: int, agg: str,
) -> pd.Series:
    """Rolling `agg` over the prior `window` rows within each group.

    `shift(1)` BEFORE windowing is the leakage guard: row i's window covers
    rows [i-window, i-1], never row i itself. min_periods=1 allows partial
    windows early in a career; a first-ever row yields NaN.
    """
    return values.groupby(group_key, sort=False).transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), agg)()
    )


def add_driver_form_features(
    df: pd.DataFrame, windows: tuple[int, ...] = DRIVER_FORM_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling driver-form features to a (raceId, driverId)-grain frame.

    Requires columns: raceId, driverId, year, round, winner, positionOrder,
    points, finished. Returns a copy sorted chronologically by (year, round);
    row count unchanged. `windows` applies to the win counts; the remaining
    features use a fixed 5-race window.

    `driver_avg_finish_last_5` uses positionOrder for ALL prior entries,
    including DNFs — Ergast's positionOrder ranks non-finishers at the back of
    the field, which acts as a built-in DNF penalty and is the deliberate
    choice here, not an oversight.
    """
    out = df.sort_values(["year", "round", "raceId"], kind="mergesort").copy()
    driver = out["driverId"]

    winner = out["winner"].astype(float)
    podium = (out["positionOrder"] <= 3).astype(float)
    finish = out["positionOrder"].astype(float)
    dnf = (~out["finished"].astype("boolean").fillna(False)).astype(float)
    points = out["points"].astype(float)

    for w in windows:
        out[f"driver_wins_last_{w}"] = _prior_rolling(winner, driver, w, "sum")
    out["driver_podiums_last_5"] = _prior_rolling(podium, driver, 5, "sum")
    out["driver_avg_finish_last_5"] = _prior_rolling(finish, driver, 5, "mean")
    out["driver_dnf_rate_last_5"] = _prior_rolling(dnf, driver, 5, "mean")
    out["driver_points_last_5"] = _prior_rolling(points, driver, 5, "sum")

    # Career-to-date entry count BEFORE this race (cumcount is 0 for the
    # first row in each group, i.e. already prior-only).
    out["driver_experience_races"] = out.groupby("driverId", sort=False).cumcount()

    return out
