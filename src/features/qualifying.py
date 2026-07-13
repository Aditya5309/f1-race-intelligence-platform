"""
src/features/qualifying.py

Grid and qualifying feature group — pre-race information from the race weekend
itself.

Everything here is known once the grid is set (post-qualifying, pre-race), so
none of it is temporal leakage. The two hazards handled explicitly:

- `grid == 0` is Ergast's pit-lane-start sentinel, NOT a real grid slot.
  It is exposed as a boolean `pit_lane_start` and
  remapped to worst-case `field_size + 1` in `grid_adjusted` /
  `grid_position_norm`; the raw `grid` column is never a model feature.
- `q2`/`q3` nulls are informative (driver eliminated earlier), not
  missing-at-random. They are NOT imputed; instead
  `reached_q2`/`reached_q3` booleans make the knockout stage explicit and the
  `*_sec` columns keep their nulls for tree models' native NaN handling.
  Note: `reached_*` is also False for rows with no qualifying data at all
  (pre-knockout-era races, DNQ) — `qualifying_position` being null
  distinguishes that case.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Feature columns this module adds (plus `qualifying_position`, which already
# exists on the master dataset and passes through as a feature).
QUALIFYING_FEATURES: tuple[str, ...] = (
    "qualifying_position",
    "q1_sec", "q2_sec", "q3_sec",
    "reached_q2", "reached_q3",
    "qualifying_gap_to_pole_pct",
    "pit_lane_start", "grid_adjusted", "grid_position_norm",
    "grid_penalty_applied",
)

# Smallest standard FIA grid-penalty tier (e.g. an unscheduled gearbox
# change) is a 3-place drop; 1-2 place effects are dominated by ordinary
# qualifying-to-grid noise (another driver's own penalty reshuffling the
# grid), so 3+ isolates genuine penalty events rather than incidental
# reshuffling.
GRID_PENALTY_THRESHOLD = 3

# Ergast qualifying times are "M:SS.sss" strings; a bare "SS.sss" (no minutes)
# is accepted defensively. Anything else parses to NaN rather than raising —
# a malformed historical time must not kill the pipeline.
_QUALI_TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}(?:\.\d+)?)\s*$")


def parse_qualifying_time(value: object) -> float:
    """Parse an Ergast qualifying time string ("1:25.846") into seconds."""
    if pd.isna(value):
        return np.nan
    match = _QUALI_TIME_RE.match(str(value))
    if match is None:
        return np.nan
    minutes = int(match.group(1)) if match.group(1) else 0
    return minutes * 60 + float(match.group(2))


def add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add grid + qualifying features to a (raceId, driverId)-grain frame.

    Requires columns: raceId, driverId, grid, q1, q2, q3.
    Returns a copy with the QUALIFYING_FEATURES columns added (row count and
    order unchanged).
    """
    out = df.copy()

    for raw_col, sec_col in (("q1", "q1_sec"), ("q2", "q2_sec"), ("q3", "q3_sec")):
        out[sec_col] = out[raw_col].map(parse_qualifying_time)

    out["reached_q2"] = out["q2"].notna()
    out["reached_q3"] = out["q3"].notna()

    # Best available time per driver: Q3 if present, else Q2, else Q1.
    # Gap to the session's best such time, as a
    # percentage — absolute lap times are not comparable across circuits,
    # the relative gap is.
    best_time = out["q3_sec"].fillna(out["q2_sec"]).fillna(out["q1_sec"])
    pole_time = best_time.groupby(out["raceId"]).transform("min")
    out["qualifying_gap_to_pole_pct"] = (best_time - pole_time) / pole_time * 100.0

    # Grid: remap the pit-lane sentinel before any numeric use.
    # field_size = entries in this race, so normalization is per-race and
    # era-safe within the modeling window.
    field_size = out.groupby("raceId")["driverId"].transform("count")
    grid = out["grid"].astype("Float64")
    pit_lane = grid.eq(0).fillna(False)
    out["pit_lane_start"] = pit_lane.astype(bool)
    adjusted = grid.mask(pit_lane, field_size + 1)
    out["grid_adjusted"] = adjusted
    out["grid_position_norm"] = adjusted / field_size

    quali_pos = out["qualifying_position"].astype("Float64")
    penalty = (adjusted - quali_pos) > GRID_PENALTY_THRESHOLD
    out["grid_penalty_applied"] = penalty.fillna(False).astype(bool)

    return out
