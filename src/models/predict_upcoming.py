"""
src/models/predict_upcoming.py

Row assembly + serving integration for the pre-race materialization
pipeline (see `docs/pre_race_materialization.md` for the full
architecture): feed the Materializer's validated rows through the
unchanged ColumnGuard -> predict_race() chain. Deliberately thin; all the
real work already happened in `materialize_features()` and `predict_race()`.

This module is exactly that: pure composition of two already-complete,
already-tested components — `src.models.materialize.materialize_features`
and `src.models.predict.predict_race` (existing, unmodified,
the same function the historical prediction path already uses) — with NO
feature-engineering, validation, or prediction logic of its own. Neither
`predict.py` nor `registry.py` is touched: the served model's ColumnGuard
re-validates the materialized row's schema exactly as it already does for
every historical prediction, and `predict_race()`'s own NaN/normalization/
ranking behavior is unchanged and untested here beyond proving it still
works against a materialized (rather than batch-built) row.

Public API
----------
`predict_upcoming_race(model, race, entry_list, dimension_inputs,
historical_master, driver_standings, constructor_standings, weather) ->
pd.DataFrame` — the ONLY function here. Same inputs as
`materialize_features()` (see its own docstring for the full contract),
plus `model` — an already-loaded serving bundle, e.g. from
`src.models.predict.load_model()` — the same object `predict_race()`
already takes.

Returns exactly what `predict_race()` returns: one row per entrant, with
carried identifier columns, `win_probability_raw`, `win_probability`
(sum-normalized within the race), and `predicted_rank`. See
`src.models.predict.predict_race`'s own docstring for the full contract —
nothing about that contract changes here.

The historical prediction path (`GET /predictions/{race_id}` and
everything it depends on) is completely unaffected: this module is
additive only, calls nothing that path doesn't already call, and neither
`predict.py` nor `registry.py` gained a single line.
"""

from __future__ import annotations

import pandas as pd

from src.features.upcoming import EntryListEntry, UpcomingRace
from src.models.materialize import materialize_features
from src.models.predict import predict_race


def predict_upcoming_race(
    model,
    race: UpcomingRace,
    entry_list: list[EntryListEntry],
    dimension_inputs: dict[str, pd.DataFrame],
    historical_master: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize `race`'s feature rows and score them with `model`.

    `materialize_features(...)` -> `predict_race(model, ...)`. Both reused
    verbatim; see their own docstrings for the full contract (inputs,
    invariants, exceptions). This function adds nothing beyond the call
    sequence itself.
    """
    materialized = materialize_features(
        race, entry_list, dimension_inputs, historical_master,
        driver_standings, constructor_standings, weather,
    )
    return predict_race(model, materialized)
