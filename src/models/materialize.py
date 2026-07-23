"""
src/models/materialize.py

The Materializer (Decision 049 Refinement 2; Phase 3 of
`.ai/pre_race_materialization_design.md` §7): the ONLY place
feature-construction logic for a race that has not yet happened lives.
Wires the EXISTING, unmodified feature-engineering pipeline
(`src.features.pipeline.build_features`/`validate_features`) against a
single synthetic row per entrant. The synthetic row is always the
chronologically LAST row fed into that pipeline (Decision 050's horizon=1:
there is never a real race after it in the frame), so every rolling/lag
feature's own shift(1)/cumsum-minus-current leakage guard treats it exactly
like any real historical row — no additional temporal-safety logic is
needed or added here.

Public API
----------
`materialize_features(race, entry_list, dimension_inputs, historical_master,
driver_standings, constructor_standings, weather) -> pd.DataFrame`
— the ONLY function other code should call.

Inputs
------
- `race: UpcomingRace` — from `src.features.upcoming.next_race()` (Phase 1).
- `entry_list: list[EntryListEntry]` — from
  `src.features.upcoming.resolve_entry_list()` (Phase 1).
- `dimension_inputs: dict[str, pd.DataFrame]` — keys "races", "circuits",
  "drivers", "constructors", "qualifying". Same shape as
  `src.integration.build_master_dataset.load_inputs()`'s return value MINUS
  "results" (which cannot exist for a race that hasn't run).
- `historical_master: pd.DataFrame` — a real, `MASTER_DATASET_COLUMNS`-shaped
  frame (genuine `build_master_dataset()` output) covering ONLY
  already-completed races strictly before `race`. Supplies the prior-race
  context every rolling/lag feature needs.
- `driver_standings`, `constructor_standings`, `weather` — passed through
  UNCHANGED to `build_features()`; the exact same tables the real batch
  pipeline uses (`src.features.standings.load_standings()`,
  `src.features.weather.load_race_weather()` — this module never calls
  either loader itself; see Invariants).

Outputs
-------
One `pd.DataFrame`, one row per `entry_list` entry, columns =
`src.features.pipeline.ID_COLUMNS + FEATURE_COLUMNS` (`FEATURES_DATASET_COLUMNS`
minus the target column, `MATERIALIZED_COLUMNS` below) — matching
`features.parquet`'s real schema exactly, minus the target, so a future
golden-row-parity check (Phase 4) can diff column-for-column. NEVER
contains a target/outcome column: a real race's `winner` is unknown here
by definition; the placeholder value used internally for feature
computation (see Invariants) is discarded before returning, never exposed.

Invariants
----------
- Pure orchestration: no network requests, no file reads/writes, no model
  loading, no prediction call. Every input is an already-loaded DataFrame;
  the only output is an in-memory DataFrame.
- Reuses `src.features.pipeline.build_features()`/`validate_features()`
  AND `build_master_dataset.validate_output()` UNMODIFIED for every
  rolling/lag/lookup feature and every integrity check — this module
  duplicates none of that logic. Its own new logic is confined to
  assembling one valid synthetic row per entrant in `MASTER_DATASET_COLUMNS`
  shape (`_build_synthetic_master_rows`, private) and appending it to real
  history before calling the existing pipeline.
- Referential integrity (every `driverId`/`constructorId`/`circuitId` in
  `entry_list` must actually resolve to a real dimension row) is enforced
  by running the assembled synthetic rows through
  `build_master_dataset.validate_output()` — the SAME check
  (`_ref_checks`) the real batch pipeline already runs, not a new one. A
  driverId absent from `dimension_inputs["drivers"]` (a stale snapshot, a
  typo) raises instead of silently carrying a null `driver_ref` through to
  the feature pipeline — the "structural, always-available identity"
  hard-fail the design doc (§3) requires.
- The qualifying join is scoped to `race.race_id` BEFORE calling
  `_join_and_check(..., validate="one_to_one")` — that helper and its
  validate spec are unchanged from `build_master_dataset()`'s own usage;
  only the `qualifying` table fed into it is pre-filtered. Necessary
  because qualifying.csv is a fact-like table (one row per (raceId,
  driverId) across ALL races), not a one-row-per-key dimension table:
  unscoped, pandas' `validate="one_to_one"` checks the right side's key
  uniqueness GLOBALLY, so a duplicate pair for a completely unrelated
  historical race would raise here and block materializing THIS race for
  a reason that has nothing to do with it.
- `grid` (final starting position) is set equal to `qualifying_position`
  for every entrant — the documented interim proxy (design doc §1/§3) for
  the still-unresolved lack of any pre-race grid-penalty data source.
  Structural consequence, inherited not introduced here: `pit_lane_start`
  and `grid_penalty_applied` can never read true for a materialized race —
  both reflect "nothing special happened" regardless of what actually
  happens on the day. Not a bug; a known limitation of this proxy.
- A driver with no qualifying row yet gets null qualifying/grid-derived
  features — never fabricated (`context/domain_knowledge.md` §8).
- `historical_master` is trusted, not re-validated, to contain only
  already-completed races strictly before `race` — this function only
  checks it does not ALREADY contain a row for `race.race_id`.
- The combined (historical + synthetic) frame is run through
  `validate_features()` before extraction — the SAME schema/row-count/
  duplicate/null/leakage checks the real batch pipeline enforces, applied
  to the whole frame, not a lighter check invented for this path.

Exceptions
----------
- `ValueError` — `entry_list` is empty (nothing to materialize).
- `ValueError` — `historical_master` already contains a row for
  `race.race_id` (would corrupt rolling history with a duplicate).
- `ValueError` — propagated from `build_master_dataset._join_and_check`
  for a duplicate dimension key (e.g. a malformed `dimension_inputs`
  table), from `validate_output()`'s own errors on the synthetic rows
  (referential integrity, duplicate/null identifiers — joined into the
  message), from `build_features()` for a row-count mismatch, or from
  `validate_features()`'s own errors on the combined frame (joined into
  the message) — the SAME exceptions the real batch pipeline already
  raises for real data, none invented here.

Reuse note (flagged, not hidden): `_build_synthetic_master_rows` imports
four PRIVATE (underscore-prefixed) column/rename constants and the
`_join_and_check` helper directly from `src.integration.build_master_dataset`
— the exact mapping and merge-and-validate logic the real
`build_master_dataset()` uses for races/circuits/drivers/constructors/
qualifying. `build_master_dataset()` itself cannot be called directly for
this purpose: it is structurally anchored on `results` (`df =
results.copy()`), which does not exist for a race that hasn't run — the
one genuine architectural constraint this phase runs into (verified by
reading that module, not assumed). Reusing its private constants/helper
was judged the lower-duplication option versus re-declaring the same
column/rename mapping a second time; `src/integration/
build_master_dataset.py` itself is NOT modified.
"""

from __future__ import annotations

import pandas as pd

from src.features.pipeline import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    build_features,
    validate_features,
)
from src.features.upcoming import EntryListEntry, UpcomingRace
from src.integration.build_master_dataset import (
    _CIRCUITS_COLUMNS,
    _CIRCUITS_RENAME,
    _CONSTRUCTORS_COLUMNS,
    _CONSTRUCTORS_RENAME,
    _DRIVERS_COLUMNS,
    _DRIVERS_RENAME,
    _QUALIFYING_COLUMNS,
    _QUALIFYING_RENAME,
    _RACES_COLUMNS,
    _RACES_RENAME,
    MASTER_DATASET_COLUMNS,
    POST_RACE_OUTCOME_COLUMNS,
    _join_and_check,
    validate_output,
)

#: Output schema: features.parquet's shape minus the target column — a
#: real outcome for an upcoming race is unknown by definition.
MATERIALIZED_COLUMNS: tuple[str, ...] = ID_COLUMNS + FEATURE_COLUMNS


def _build_synthetic_master_rows(
    race: UpcomingRace,
    entry_list: list[EntryListEntry],
    dimension_inputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """One `MASTER_DATASET_COLUMNS`-shaped row per `entry_list` entry.

    Anchored on `entry_list`, not `results` (which doesn't exist yet) — see
    module docstring's Reuse note. Reuses `build_master_dataset`'s own
    column/rename constants and `_join_and_check` helper so races/circuits/
    drivers/constructors/qualifying are joined identically to the real
    pipeline; a driver with no qualifying row yet simply gets nulls there
    (a left join with no match), never a fabricated value.
    """
    if not entry_list:
        raise ValueError("entry_list is empty — nothing to materialize.")

    base_row_count = len(entry_list)
    df = pd.DataFrame({
        "raceId": [race.race_id] * base_row_count,
        "driverId": [e.driver_id for e in entry_list],
        "constructorId": [e.constructor_id for e in entry_list],
    })

    races = dimension_inputs["races"][list(_RACES_COLUMNS)].rename(columns=_RACES_RENAME)
    df = _join_and_check(df, races, on="raceId", validate="many_to_one",
                         step_name="races", expected_row_count=base_row_count)

    circuits = dimension_inputs["circuits"][list(_CIRCUITS_COLUMNS)].rename(columns=_CIRCUITS_RENAME)
    df = _join_and_check(df, circuits, on="circuitId", validate="many_to_one",
                         step_name="circuits", expected_row_count=base_row_count)

    drivers = dimension_inputs["drivers"][list(_DRIVERS_COLUMNS)].rename(columns=_DRIVERS_RENAME)
    df = _join_and_check(df, drivers, on="driverId", validate="many_to_one",
                         step_name="drivers", expected_row_count=base_row_count)

    constructors = dimension_inputs["constructors"][list(_CONSTRUCTORS_COLUMNS)].rename(
        columns=_CONSTRUCTORS_RENAME
    )
    df = _join_and_check(df, constructors, on="constructorId", validate="many_to_one",
                         step_name="constructors", expected_row_count=base_row_count)

    # Scoped to THIS race before the join (unlike build_master_dataset()'s
    # own equivalent join, which legitimately spans full history because
    # its left side does too). qualifying.csv is a fact-like table — one
    # row per (raceId, driverId) across ALL races, not a one-row-per-key
    # dimension table — so validate="one_to_one" checks uniqueness of the
    # right side GLOBALLY, not just the rows that match `df`. Left
    # unscoped, an unrelated duplicate (raceId, driverId) pair anywhere
    # else in qualifying history would raise here and block materializing
    # THIS race for a reason that has nothing to do with it (verified: a
    # MergeError reproduces from a duplicate pair on a different raceId).
    # Pre-filtering to `race.race_id` keeps `_join_and_check` and its
    # validate="one_to_one" semantics completely unchanged — only the data
    # fed into it is narrowed to what's actually relevant.
    qualifying = dimension_inputs["qualifying"]
    qualifying = qualifying[qualifying["raceId"] == race.race_id]
    qualifying = qualifying[list(_QUALIFYING_COLUMNS)].rename(columns=_QUALIFYING_RENAME)
    df = _join_and_check(df, qualifying, on=["raceId", "driverId"], validate="one_to_one",
                         step_name="qualifying", expected_row_count=base_row_count)

    # Interim grid-penalty proxy (design doc §1/§3, still unresolved): no
    # pre-race grid-penalty data source exists, so `grid` (final starting
    # position) is set equal to `qualifying_position`. See module
    # docstring's Invariants for the structural consequence.
    df["grid"] = df["qualifying_position"]

    # Placeholder post-race-outcome columns: never real, always discarded
    # (build_features() itself excludes POST_RACE_OUTCOME_COLUMNS from its
    # selected output; this function's own caller strips the placeholder
    # "winner" too). Any consistent, non-null value is correct here: of
    # these columns, only positionOrder/points/finished are ever READ by
    # the 8 feature functions (verified by reading each one), and each
    # reads this row's own value via shift(1) (never sees it) or an
    # inclusive cumsum minus this row's own value (cancels exactly,
    # regardless of what that value is) — never via same-race leakage.
    for col in POST_RACE_OUTCOME_COLUMNS:
        df[col] = 0
    df["winner"] = 0

    return df[list(MASTER_DATASET_COLUMNS)]


def materialize_features(
    race: UpcomingRace,
    entry_list: list[EntryListEntry],
    dimension_inputs: dict[str, pd.DataFrame],
    historical_master: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize one pre-race feature row per `entry_list` entry for `race`.

    See module docstring for the full contract (inputs/outputs/invariants/
    exceptions). Reuses `build_master_dataset.validate_output()`,
    `build_features()`, and `validate_features()` UNMODIFIED; this
    function's own logic is confined to assembling the synthetic row(s),
    appending them to real history, and extracting the result.
    """
    if race.race_id in set(historical_master["raceId"].unique()):
        raise ValueError(
            f"historical_master already contains raceId {race.race_id} — "
            "it must cover only already-completed races strictly before "
            "the race being materialized."
        )

    synthetic_rows = _build_synthetic_master_rows(race, entry_list, dimension_inputs)

    # Align placeholder post-race-outcome column dtypes to historical_
    # master's REAL dtypes (pandas nullable Int64/boolean/string extension
    # types in the actual pipeline, not plain int/bool/str) before
    # concatenating. Found via golden-row parity testing against real data:
    # pd.concat silently produces an incompatible "object" dtype otherwise,
    # which breaks a downstream .astype("boolean") call in
    # driver_form.py/constructor_form.py. Derived from historical_master
    # itself rather than a hardcoded dtype map, so this never drifts out of
    # sync with whatever src/data/cleaner.py actually produces.
    for col in (*POST_RACE_OUTCOME_COLUMNS, "winner"):
        synthetic_rows[col] = synthetic_rows[col].astype(historical_master[col].dtype)

    # Same referential-integrity/row-count/duplicate-pair/identifier-null
    # checks build_master_dataset()'s own real output goes through — the
    # entry list is exactly the kind of "structural, always-available"
    # identity input the design doc (§3) says must hard-fail loudly on any
    # gap (e.g. a driverId absent from the drivers dimension table), not
    # silently carry a null reference through to the feature pipeline.
    synthetic_validation = validate_output(synthetic_rows, expected_row_count=len(entry_list))
    if not synthetic_validation.passed:
        raise ValueError(
            "Synthetic row assembly failed build_master_dataset.validate_output(): "
            + "; ".join(synthetic_validation.errors)
        )

    combined = pd.concat([historical_master, synthetic_rows], ignore_index=True)

    features = build_features(combined, driver_standings, constructor_standings, weather)

    result = validate_features(features, expected_row_count=len(combined))
    if not result.passed:
        raise ValueError(
            "Materialized feature frame failed validate_features(): "
            + "; ".join(result.errors)
        )

    # Boolean-mask selection already returns a new frame, and sort_values()
    # (non-inplace) returns another — an explicit .copy() here would be a
    # third, redundant materialization of the same data for no benefit.
    materialized = (
        features[features["raceId"] == race.race_id]
        .sort_values("driverId", kind="mergesort")
        .reset_index(drop=True)
    )
    return materialized[list(MATERIALIZED_COLUMNS)]
