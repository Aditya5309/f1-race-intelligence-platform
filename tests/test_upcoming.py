"""
Tests for src/features/upcoming.py — calendar and entry-list resolution
for the pre-race materialization pipeline (see
docs/pre_race_materialization.md).

Coverage:
  - next_race resolves the single earliest race with no results row yet
    (materialization horizon = 1: always the next race, never further out)
  - next_race returns None once every race has a result
  - next_race/resolve_entry_list propagate the shared calendar-ambiguity
    check (duplicate raceId, duplicate (year, round) slot)
  - resolve_entry_list's default inference matches the most recent
    completed race's roster in the ordinary case
  - two roster edge cases inference alone can't handle: a mid-season
    driver swap (inference is stale) and a rookie debut (inference omits
    the rookie) — both require `override`, which is verified to bypass
    inference entirely
  - resolve_entry_list's error cases: unknown raceId, no prior completed
    race and no override
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.upcoming import EntryListEntry, next_race, resolve_entry_list


def _races(rows: list[tuple[int, int, int, int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["raceId", "year", "round", "circuitId", "name", "date"]
    )


def _results(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["raceId", "driverId", "constructorId"])


def test_next_race_resolves_earliest_unrun_race():
    races = _races([
        (1, 2026, 1, 10, "Race A", "2026-03-01"),
        (2, 2026, 2, 11, "Race B", "2026-03-08"),
        (3, 2026, 3, 12, "Race C", "2026-03-15"),
    ])
    results = _results([(1, 1, 1), (1, 2, 2)])  # only race 1 completed

    resolved = next_race(races, results)

    assert resolved is not None
    assert resolved.race_id == 2
    assert resolved.year == 2026
    assert resolved.round == 2
    assert resolved.circuit_id == 11
    assert resolved.name == "Race B"


def test_next_race_never_looks_past_the_earliest_unrun_race():
    """Horizon = 1: race 3 has no result either, but race 2
    is the one returned — never race 3."""
    races = _races([
        (1, 2026, 1, 10, "Race A", "2026-03-01"),
        (2, 2026, 2, 11, "Race B", "2026-03-08"),
        (3, 2026, 3, 12, "Race C", "2026-03-15"),
    ])
    results = _results([(1, 1, 1)])

    resolved = next_race(races, results)

    assert resolved.race_id == 2


def test_next_race_returns_none_when_every_race_has_a_result():
    races = _races([(1, 2026, 1, 10, "Race A", "2026-03-01")])
    results = _results([(1, 1, 1)])

    assert next_race(races, results) is None


def test_next_race_rejects_ambiguous_calendar():
    races = _races([
        (1, 2026, 1, 10, "Race A", "2026-03-01"),
        (2, 2026, 1, 11, "Race B duplicate round", "2026-03-01"),
    ])
    results = _results([])

    with pytest.raises(ValueError, match="same \\(year, round\\)"):
        next_race(races, results)


def test_resolve_entry_list_infers_from_most_recent_completed_race():
    races = _races([
        (1, 2026, 1, 10, "Race A", "2026-03-01"),
        (2, 2026, 2, 11, "Race B", "2026-03-08"),
    ])
    results = _results([(1, 1, 100), (1, 2, 200)])

    entries = resolve_entry_list(2, races, results)

    assert set(entries) == {
        EntryListEntry(driver_id=1, constructor_id=100),
        EntryListEntry(driver_id=2, constructor_id=200),
    }


def test_resolve_entry_list_inference_is_stale_after_a_mid_season_swap():
    """Driver 1 is replaced by driver 3 at the SAME constructor for the
    upcoming race. Inference (no override) cannot see this — it is
    documented as a known limitation, not silently correct — and an
    explicit override is required to reflect the real lineup."""
    races = _races([
        (1, 2026, 1, 10, "Race A", "2026-03-01"),
        (2, 2026, 2, 11, "Race B", "2026-03-08"),
    ])
    results = _results([(1, 1, 100), (1, 2, 200)])

    inferred = resolve_entry_list(2, races, results)
    assert EntryListEntry(driver_id=1, constructor_id=100) in inferred
    assert EntryListEntry(driver_id=3, constructor_id=100) not in inferred

    confirmed = resolve_entry_list(
        2, races, results,
        override=[
            EntryListEntry(driver_id=3, constructor_id=100),
            EntryListEntry(driver_id=2, constructor_id=200),
        ],
    )
    assert confirmed == [
        EntryListEntry(driver_id=3, constructor_id=100),
        EntryListEntry(driver_id=2, constructor_id=200),
    ]


def test_resolve_entry_list_inference_omits_a_debuting_rookie():
    """A rookie with zero prior results cannot appear in the inferred
    roster by construction — override is the only way to add them."""
    races = _races([
        (1, 2026, 1, 10, "Race A", "2026-03-01"),
        (2, 2026, 2, 11, "Race B", "2026-03-08"),
    ])
    results = _results([(1, 1, 100)])  # driverId 1 only — no rookie (99) yet

    inferred = resolve_entry_list(2, races, results)
    assert all(entry.driver_id != 99 for entry in inferred)

    confirmed = resolve_entry_list(
        2, races, results,
        override=[
            EntryListEntry(driver_id=1, constructor_id=100),
            EntryListEntry(driver_id=99, constructor_id=200),
        ],
    )
    assert EntryListEntry(driver_id=99, constructor_id=200) in confirmed


def test_resolve_entry_list_override_bypasses_inference_entirely():
    races = _races([(1, 2026, 1, 10, "Race A", "2026-03-01")])
    results = _results([])  # no history at all

    override = [EntryListEntry(driver_id=1, constructor_id=1)]
    assert resolve_entry_list(1, races, results, override=override) == override


def test_resolve_entry_list_rejects_unknown_race_id():
    races = _races([(1, 2026, 1, 10, "Race A", "2026-03-01")])
    results = _results([])

    with pytest.raises(ValueError, match="not found in the races calendar"):
        resolve_entry_list(999, races, results)


def test_resolve_entry_list_requires_override_with_no_prior_completed_race():
    races = _races([(1, 2026, 1, 10, "Race A", "2026-03-01")])
    results = _results([])

    with pytest.raises(ValueError, match="No completed race exists before"):
        resolve_entry_list(1, races, results)
