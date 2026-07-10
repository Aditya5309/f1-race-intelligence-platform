"""
src/features/wet_form.py

Driver/constructor wet-weather form (Phase 4 Tranche B item 2): does a
driver/constructor perform differently in wet races vs. dry races, and by
how much?

Depends on src/features/weather.py's `race_precip_mm` (full 2010-2024
coverage) to classify each race as wet or dry, via WET_THRESHOLD_MM — the
SAME cutoff scripts/backfill_weather.py itself uses for
`conditions_changed`. Deliberately NOT built from `conditions_changed`,
which only covers 68 of 305 races (2022-2024 qualifying-session data) and
answers a different question (did conditions change between sessions) —
far too sparse a history to build a driver's CAREER wet-vs-dry delta from.
This module must run AFTER add_weather_features in the pipeline (it reads
`race_precip_mm` off the already-merged frame).

Leakage-safe by the same discipline every other form feature in this
pipeline uses, just via the cumsum-minus-current idiom (circuit_history.py)
rather than shift+rolling, because the "prior races of this TYPE only"
restriction doesn't fit a fixed rolling window cleanly: `_prior_masked_avg`
computes, for each row, the mean of `values` over strictly-prior rows in
the same group where `mask` is true (a cumulative sum/count of the
mask-restricted series, each minus its own row's contribution — algebraically
identical to shift(1)+expanding on the type-restricted subsequence, see
circuit_history.py's module docstring for the same trick applied to wins).
Missing `race_precip_mm` (shouldn't happen within the backfilled window,
but this is third-party data) is treated as neither wet nor dry — excluded
from both averages, not silently folded into "dry".

Shrinkage: a driver's raw wet-dry delta is estimated from a career sample
of wet races — averaging ~20% of the calendar, often still single digits
even a few seasons in. A driver with 1-2 wet races could show an extreme
delta from pure noise (a single spin, a single inspired drive in the rain).
Standard empirical-Bayes shrinkage toward a FIELD-WIDE wet-dry delta (the
cross-sectional mean, at each race, of every OTHER driver's own
already-prior-only raw delta — never using the current race's own result,
so no leakage) fixes this:

    weight = n_wet / (n_wet + SHRINKAGE_K)
    shrunk_delta = weight * driver_raw_delta + (1 - weight) * field_wide_delta

SHRINKAGE_K = 8 prior wet races: chosen so a driver needs roughly 2 full
seasons of wet-race exposure (wet races are ~20% of a ~20-24 race calendar)
before their own sample outweighs the field-wide prior — deliberately more
conservative than this codebase's typical 3-5-race "form" windows, because
a wet-dry delta is estimated from a much rarer, noisier event type than
routine race-to-race form.

Zero-wet-history fallback: at n_wet=0, weight=0 and the formula reduces
exactly to field_wide_delta (a real, computed "what does an average driver
do differently in the wet" prior) — not zero (a specific, false claim of
"no difference"), and not a silent drop. The raw delta's NaN (undefined
with 0 wet races) is filled with a 0.0 placeholder ONLY inside the
shrinkage blend, since its contribution is already zeroed by weight=0 — the
placeholder never actually influences the result, it just prevents 0*NaN
from propagating as NaN through the whole expression.

Constructor-level mirrors driver-level exactly, aggregated to
(constructorId, raceId) grain FIRST — same reason as constructor_form.py:
a constructor fields two cars, and using the mean of both cars' finishes
per race is the representative "how did the team execute this weekend"
value; the raw per-row driver value would otherwise let a teammate's
SAME-RACE result leak in.
"""

from __future__ import annotations

import pandas as pd

WET_THRESHOLD_MM = 0.2   # matches scripts/backfill_weather.py's own cutoff
SHRINKAGE_K = 8.0

WET_FORM_FEATURES: tuple[str, ...] = (
    "driver_wet_dry_delta",
    "constructor_wet_dry_delta",
)


def _prior_masked_avg(
    values: pd.Series, mask: pd.Series, group_key: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Mean (and count) of `values` over strictly-prior, mask-true rows within
    each group. Rows must already be sorted chronologically by the caller —
    cumsum operates in row order, not by any date column.

    cumsum-minus-current (circuit_history.py's pattern, generalized to an
    arbitrary boolean mask rather than "every prior row"): a row's own
    masked contribution is always subtracted back out, so a mask-true row
    never sees its own value, and a mask-false row is unaffected (its own
    masked contribution is already 0).
    """
    masked_values = values.where(mask, 0.0)
    prior_sum = masked_values.groupby(group_key, sort=False).cumsum() - masked_values
    mask_f = mask.astype(float)
    prior_count = mask_f.groupby(group_key, sort=False).cumsum() - mask_f
    avg = prior_sum / prior_count.where(prior_count > 0)
    return avg, prior_count


def _wet_dry_delta(
    finish: pd.Series, is_wet: pd.Series, is_dry: pd.Series,
    group_key: pd.Series, race_key: pd.Series,
) -> pd.Series:
    """Shrunk wet-minus-dry average-finish delta — see module docstring."""
    wet_avg, wet_n = _prior_masked_avg(finish, is_wet, group_key)
    dry_avg, _ = _prior_masked_avg(finish, is_dry, group_key)
    raw_delta = wet_avg - dry_avg

    field_wide_delta = raw_delta.groupby(race_key, sort=False).transform("mean")

    weight = wet_n / (wet_n + SHRINKAGE_K)
    return weight * raw_delta.fillna(0.0) + (1 - weight) * field_wide_delta


def add_wet_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add driver/constructor wet-dry form deltas to a (raceId, driverId)-grain
    frame. Requires: raceId, driverId, constructorId, year, round,
    positionOrder, race_precip_mm (added by add_weather_features — this
    module must run after it). Returns a copy sorted chronologically by
    (year, round); row count unchanged.
    """
    out = df.sort_values(["year", "round", "raceId"], kind="mergesort").copy()

    has_precip = out["race_precip_mm"].notna()
    is_wet = has_precip & (out["race_precip_mm"] > WET_THRESHOLD_MM)
    is_dry = has_precip & (out["race_precip_mm"] <= WET_THRESHOLD_MM)
    finish = out["positionOrder"].astype(float)

    out["driver_wet_dry_delta"] = _wet_dry_delta(
        finish, is_wet, is_dry, out["driverId"], out["raceId"]
    )

    # Constructor level: aggregate to one row per (constructorId, raceId)
    # first (mean of both cars' finishes), so a teammate's same-race result
    # never leaks in — same hazard and fix as constructor_form.py.
    race_level = (
        out.assign(_is_wet=is_wet, _is_dry=is_dry, _finish=finish)
        .groupby(["constructorId", "raceId"], as_index=False)
        .agg(
            year=("year", "first"),
            round=("round", "first"),
            _is_wet=("_is_wet", "first"),
            _is_dry=("_is_dry", "first"),
            _finish=("_finish", "mean"),
        )
        .sort_values(["year", "round", "raceId"], kind="mergesort")
    )
    race_level["constructor_wet_dry_delta"] = _wet_dry_delta(
        race_level["_finish"], race_level["_is_wet"], race_level["_is_dry"],
        race_level["constructorId"], race_level["raceId"],
    )

    merged = out.merge(
        race_level[["constructorId", "raceId", "constructor_wet_dry_delta"]],
        on=["constructorId", "raceId"], how="left", validate="many_to_one",
    )
    if len(merged) != len(out):
        raise ValueError(
            "Wet-form constructor merge changed row count "
            f"({len(out):,} -> {len(merged):,}) — (constructorId, raceId) "
            "aggregation is no longer unique."
        )
    return merged
