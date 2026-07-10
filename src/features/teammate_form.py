"""
src/features/teammate_form.py

Teammate-relative features (Phase 4, Tranche A) — how a driver compares to
their own teammate, both this weekend and over recent form.

Same fan-out hazard as constructor_form.py: a constructor fields (usually)
two cars per race, so a naive row-level comparison would either miss the
teammate entirely or, worse, leak the teammate's SAME-RACE result into a
feature for the row being predicted. Both features here are built from a
same-race, self-excluding teammate average:

    teammate_avg = (group_sum - self_value) / (group_count - 1)

computed via `groupby(["raceId", "constructorId"])`, which generalizes to
constructors fielding more than two cars (rare, historical) and yields NaN by
construction (division by zero) for the ~2 of 3,219 constructor-race groups
with only one car that race — no bespoke fallback code. That NaN flows into
the same `SimpleImputer(strategy="median", add_indicator=True)` step every
other "no information" feature in this pipeline already relies on
(src/models/registry.py); a missing teammate is treated exactly like a
driver's first-ever race or a constructor's first-ever race elsewhere in this
codebase.

Deliberate asymmetry between the two features (do not "fix" this to make
them symmetric):
- The qualifying-gap raw delta is PRE-RACE-SAFE — quali is set before the
  race, the same precedent as `qualifying_gap_to_pole_pct`
  (src/features/qualifying.py) — so it is exposed BOTH raw
  (`qualifying_gap_to_teammate_current`, this weekend's actual gap) AND
  rolled (`qualifying_gap_to_teammate`, recent-form trend). These answer
  different questions and neither should be dropped for the other.
- The race-pace raw delta is derived from `positionOrder`, a same-race
  OUTCOME column — exposing it raw would leak the race being predicted into
  its own feature. Only the rolled form, `race_pace_delta_to_teammate`, is
  ever exposed as a feature.

Rolling uses the standard shift(1)-then-roll pattern (identical to
driver_form._prior_rolling / constructor_form._prior_rolling), grouped by
driverId and ordered by (year, round), over a single 5-race window — matching
the sibling single-window features (constructor_podiums_last_5,
constructor_dnf_rate_last_5) rather than the dual (3, 5) treatment reserved
for win counts.

Sign convention: positive = self ahead of teammate (a lower qualifying
position / positionOrder is better).
"""

from __future__ import annotations

import pandas as pd

TEAMMATE_FORM_WINDOW = 5

TEAMMATE_FORM_FEATURES: tuple[str, ...] = (
    "qualifying_gap_to_teammate_current",
    "qualifying_gap_to_teammate",
    "race_pace_delta_to_teammate",
)


def _teammate_average(df: pd.DataFrame, col: str) -> pd.Series:
    """Same-race, self-excluding average of `col` across teammates.

    NaN when there is no other valid value to compare against this race —
    either the constructor fielded only one car, or the only other car's
    value is itself null (e.g. a teammate with no qualifying time).
    """
    group = df.groupby(["raceId", "constructorId"])[col]
    total = group.transform("sum")
    count = group.transform("count")
    return (total - df[col]) / (count - 1)


def _prior_rolling(
    values: pd.Series, group_key: pd.Series, window: int, agg: str,
) -> pd.Series:
    """shift(1)-then-roll within each group — see driver_form._prior_rolling."""
    return values.groupby(group_key, sort=False).transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), agg)()
    )


def add_teammate_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add teammate-relative features to a (raceId, driverId)-grain frame.

    Requires columns: raceId, driverId, constructorId, year, round,
    qualifying_position, positionOrder. Returns a copy sorted chronologically
    by (year, round); row count unchanged.
    """
    out = df.sort_values(["year", "round", "raceId"], kind="mergesort").copy()

    quali_teammate_avg = _teammate_average(out, "qualifying_position")
    pace_teammate_avg = _teammate_average(out, "positionOrder")

    quali_gap_raw = quali_teammate_avg - out["qualifying_position"].astype(float)
    pace_gap_raw = pace_teammate_avg - out["positionOrder"].astype(float)

    out["qualifying_gap_to_teammate_current"] = quali_gap_raw

    driver = out["driverId"]
    out["qualifying_gap_to_teammate"] = _prior_rolling(
        quali_gap_raw, driver, TEAMMATE_FORM_WINDOW, "mean"
    )
    out["race_pace_delta_to_teammate"] = _prior_rolling(
        pace_gap_raw, driver, TEAMMATE_FORM_WINDOW, "mean"
    )

    return out
