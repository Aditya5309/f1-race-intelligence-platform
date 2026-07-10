"""
src/features/weather.py

Race-weekend weather features — `race_precip_mm`/`race_temp_c` (full
2010-2024 coverage) and `quali_precip_mm`/`conditions_changed` (2022-2024
only, 68 of 305 races). Backfilled by scripts/backfill_weather.py from
Open-Meteo's Historical Weather API into data/interim/race_weather.csv
(training-side only, gitignored like the rest of data/ — never a
runtime/serving dependency, the same status as master_dataset.parquet).

Leakage note: this is ACTUAL, post-hoc observed weather during each
session — not a same-race OUTCOME column (POST_RACE_OUTCOME_COLUMNS);
weather is not a function of who wins. That makes it a legitimate pre-
result feature for THIS project's current historical-prediction/evaluation
framing. See scripts/backfill_weather.py's module docstring for why this
same data would NOT be usable as-is for a future live "predict an upcoming
race" feature (no forecast substitute exists here — a live feature needs a
different data source entirely, with its own uncertainty).

Missingness: `quali_precip_mm`/`conditions_changed` are null for the ~78%
of races before Ergast recorded qualifying session start times (pre-2022)
— left as NaN, no same-day approximation for the years missing session
time (that would misrepresent precision this dataset doesn't have).
`conditions_changed` is stored as float64 (1.0/0.0/NaN), not a nullable
boolean dtype, so it flows through the standard numeric
SimpleImputer(strategy="median", add_indicator=True) pipeline exactly like
every other feature with missing history in this codebase (e.g.
q1_sec/q2_sec/q3_sec) — no new imputation code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_CSV_PATH = _PROJECT_ROOT / "data" / "interim" / "race_weather.csv"

WEATHER_FEATURES: tuple[str, ...] = (
    "race_precip_mm", "race_temp_c", "quali_precip_mm", "conditions_changed",
)

_BOOL_STRINGS = {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}


def load_race_weather(path: Path = WEATHER_CSV_PATH) -> pd.DataFrame:
    """Load the backfilled per-race weather CSV (scripts/backfill_weather.py).

    `conditions_changed` round-trips through CSV as "True"/"False"/empty
    strings; normalized here to float64 (1.0/0.0/NaN) for the model pipeline.
    """
    df = pd.read_csv(path)
    df["conditions_changed"] = df["conditions_changed"].map(_BOOL_STRINGS).astype(float)
    return df


def add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-race weather features to a (raceId, driverId)-grain frame.

    Requires df column: raceId. `weather` is race_weather.csv's frame (one
    row per raceId, see load_race_weather()). Left-join broadcasts each
    race's weather onto every driver row for that race — weather is a
    property of the session, identical for every driver in it. Returns a
    copy with WEATHER_FEATURES added; row count and order unchanged.
    """
    out = df.merge(
        weather[["raceId"] + list(WEATHER_FEATURES)],
        on="raceId", how="left", validate="many_to_one",
    )
    if len(out) != len(df):
        raise ValueError(
            f"Weather merge changed row count ({len(df):,} -> {len(out):,}) — "
            "race_weather.csv likely has duplicate raceId rows."
        )
    return out
