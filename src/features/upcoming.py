"""
src/features/upcoming.py

Calendar & entry-list resolution for the pre-race materialization pipeline
(Decisions 049/050; Phase 1 of `.ai/pre_race_materialization_design.md` §7).

This module answers two questions ONLY — "which race is next?" and "who is
entered in it?" — against already-loaded `races.csv`/`results.csv`-shaped
DataFrames (typically produced by `src.data.loader.load_csv`, though this
module never calls it itself). It builds no feature values, fits nothing,
and makes no network call. It is intentionally NOT imported by
`src/features/pipeline.py`, `src/models/predict.py`, or `app/api.py` — the
historical prediction path is completely unchanged by this module's
existence (Decision 049's own discipline: feature engineering and serving
stay decoupled). A later phase's `Materializer` (Phase 3) is this module's
only intended caller.

**Deliberately storage-agnostic (no loader helpers):** unlike, e.g.,
`standings.py`'s paired `load_standings()`, this module takes DataFrames
in and never reads a CSV path itself. Phase 1 is resolution logic only;
Phase 3's `Materializer` is where real data sourcing gets wired in (per
this design's own phase split — see `.ai/pre_race_materialization_design.md`
§7). Adding an I/O helper now would presume that wiring before it exists.
Revisit only if a concrete future phase needs one.

Public API
----------
- `UpcomingRace` (frozen dataclass): identity of one resolved race.
- `EntryListEntry` (frozen dataclass): one (driverId, constructorId) pairing.
- `next_race(races, results) -> UpcomingRace | None`
- `resolve_entry_list(race_id, races, results, *, override=None) -> list[EntryListEntry]`

Invariants
----------
- Materialization horizon = 1 (Decision 050): `next_race()` always resolves
  the SINGLE earliest race with no `results.csv` row yet, in chronological
  (year, round) order — never a race further out, and never today's wall
  clock (the resolution is derived entirely from the two input frames, so
  it is deterministic and reproducible for the same data snapshot).
- **This assumes every already-completed historical race has a matching
  `results.csv` row** (Decision 050). "First race with no result" is only
  equivalent to "first *future* race" because of this — there is no
  separate check restricting the search to the current/future season. If
  that assumption were ever violated (a historical data gap — an
  abandoned race, an ingestion hole), `next_race()` would misidentify that
  old race as "next" rather than skip it as a historical anomaly. No such
  gap exists in the current dataset; hardening against one is explicitly
  deferred until a concrete instance appears or the horizon policy changes
  (Decision 050's Future Work) — not built speculatively here.
- Calendar integrity is validated by reusing
  `src.features.standings.build_prev_race_map`'s existing checks (no
  duplicate raceId, no two raceIds sharing a (year, round) slot) — the
  same discipline the historical feature pipeline already relies on, not a
  parallel implementation of it.
- `resolve_entry_list()` only ever looks BACKWARD from `race_id` (the most
  recently completed race strictly before it) — never forward, and never
  at `race_id`'s own results (which do not exist yet by definition for a
  genuine upcoming race). This is the same "provably earlier than the
  race" discipline `context/domain_knowledge.md` §6 requires of every
  feature in this project.
- Roster inference is a best-effort FALLBACK, not a claim of correctness:
  it cannot see a rookie's debut (no prior result exists for them) or a
  mid-season substitution announced after the most recent completed race.
  The `override` parameter exists precisely for these cases (design doc
  §3, "Entry-list uncertainty") — callers with a confirmed entry list must
  pass it; inference should be treated as a placeholder, not authoritative.

Exceptions
----------
- `ValueError` — ambiguous calendar (duplicate raceId, or two raceIds
  sharing one (year, round) slot); propagated from `build_prev_race_map`.
- `ValueError` — `resolve_entry_list()` called with a `race_id` absent from
  `races`.
- `ValueError` — `resolve_entry_list()` called with no `override` for a
  `race_id` that has no completed race before it in the calendar (nothing
  to infer a roster from).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.standings import build_prev_race_map


@dataclass(frozen=True)
class UpcomingRace:
    """Identity of one resolved not-yet-run race (`races.csv` row shape)."""

    race_id: int
    year: int
    round: int
    circuit_id: int
    name: str
    date: str


@dataclass(frozen=True)
class EntryListEntry:
    """One (driverId, constructorId) pairing for a race's entry list."""

    driver_id: int
    constructor_id: int


def _chronological_race_ids(races: pd.DataFrame) -> list[int]:
    """Validated raceId order, earliest first.

    Reuses `standings.build_prev_race_map`'s calendar-integrity checks
    (raises `ValueError` on a duplicate raceId or a shared (year, round)
    slot) rather than re-implementing them.
    """
    calendar = build_prev_race_map(races[["raceId", "year", "round"]])
    return [int(race_id) for race_id in calendar["raceId"]]


def next_race(races: pd.DataFrame, results: pd.DataFrame) -> UpcomingRace | None:
    """
    Resolve the single next race with no `results.csv` row yet.

    Materialization horizon = 1 (Decision 050): this is always the EARLIEST
    such race in (year, round) order, never a race further out. Returns
    `None` if every race in `races` already has a result (no upcoming race
    scheduled in this data snapshot).

    Assumes every already-completed historical race has a matching
    `results.csv` row (see module docstring's Invariants) — there is no
    independent current/future-season filter, only this assumption. Not
    violated anywhere in the current dataset; not hardened against here by
    design (Decision 050's Future Work).

    Requires `races` columns: raceId, year, round, circuitId, name, date.
    Requires `results` columns: raceId.
    """
    completed = set(results["raceId"].unique())
    for race_id in _chronological_race_ids(races):
        if race_id in completed:
            continue
        row = races.loc[races["raceId"] == race_id].iloc[0]
        return UpcomingRace(
            race_id=race_id,
            year=int(row["year"]),
            round=int(row["round"]),
            circuit_id=int(row["circuitId"]),
            name=str(row["name"]),
            date=str(row["date"]),
        )
    return None


def resolve_entry_list(
    race_id: int,
    races: pd.DataFrame,
    results: pd.DataFrame,
    *,
    override: list[EntryListEntry] | None = None,
) -> list[EntryListEntry]:
    """
    Resolve the (driverId, constructorId) entry list for `race_id`.

    If `override` is given, it is returned verbatim — no inference happens.
    This is the required path whenever the entry list is known to differ
    from historical roster (mid-season substitution, rookie debut, a
    confirmed reserve-driver appearance).

    Otherwise, falls back to the distinct (driverId, constructorId) pairs
    of the most recently COMPLETED race strictly before `race_id` in the
    calendar. This inference is a placeholder only (see module docstring's
    Invariants) — it cannot see a driver who has never raced before, and it
    cannot see a substitution announced after that most-recent race.

    Requires `races` columns: raceId, year, round.
    Requires `results` columns: raceId, driverId, constructorId.

    Raises `ValueError` if `race_id` is not in `races`, or if no completed
    race exists before it (nothing to infer from) and no `override` was
    given.
    """
    if override is not None:
        return list(override)

    order = _chronological_race_ids(races)
    if race_id not in order:
        raise ValueError(f"raceId {race_id} not found in the races calendar.")

    completed = set(results["raceId"].unique())
    idx = order.index(race_id)
    prior_completed = [r for r in order[:idx] if r in completed]
    if not prior_completed:
        raise ValueError(
            f"No completed race exists before raceId {race_id} in the calendar — "
            "cannot infer an entry list; pass an explicit `override` instead."
        )
    most_recent_race_id = prior_completed[-1]

    roster = (
        results.loc[
            results["raceId"] == most_recent_race_id, ["driverId", "constructorId"]
        ]
        .drop_duplicates()
    )
    return [
        EntryListEntry(driver_id=int(row.driverId), constructor_id=int(row.constructorId))
        for row in roster.itertuples()
    ]
