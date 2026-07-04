"""
Tests for src/models/eras.py (Decision 019 — regulation-era domain model).

The era table is the code-level single source of truth for F1 regulation
boundaries (prose source: context/domain_knowledge.md Section 1). These
tests pin the structural invariants future edits must preserve:

  - table starts at the Decision-008 modeling window (2010)
  - eras are contiguous and non-overlapping; only the final era is ongoing
  - year lookup (era_of) and name resolution (get_era) behave at boundaries
  - closed-range access is loud for ongoing eras
"""

import pytest

from src.models.eras import (
    FUTURE_ENGINE,
    GROUND_EFFECT,
    HYBRID,
    MODELING_WINDOW_START,
    REGULATION_ERAS,
    V8,
    RegulationEra,
    era_of,
    get_era,
)


def test_table_starts_at_modeling_window():
    assert REGULATION_ERAS[0].start_year == MODELING_WINDOW_START == 2010


def test_eras_contiguous_and_only_last_ongoing():
    for prev, nxt in zip(REGULATION_ERAS, REGULATION_ERAS[1:]):
        assert prev.end_year is not None          # only the final era is open
        assert nxt.start_year == prev.end_year + 1
    assert REGULATION_ERAS[-1].is_ongoing
    assert not any(e.is_ongoing for e in REGULATION_ERAS[:-1])


def test_expected_era_boundaries():
    # Pin the domain facts (domain_knowledge.md Section 1 / Decision 013's
    # era segmentation): editing these is a domain decision, not a refactor.
    assert (V8.start_year, V8.end_year) == (2010, 2013)
    assert (HYBRID.start_year, HYBRID.end_year) == (2014, 2021)
    assert (GROUND_EFFECT.start_year, GROUND_EFFECT.end_year) == (2022, 2025)
    assert (FUTURE_ENGINE.start_year, FUTURE_ENGINE.end_year) == (2026, None)


def test_era_of_boundary_years():
    assert era_of(2010) is V8
    assert era_of(2013) is V8
    assert era_of(2014) is HYBRID
    assert era_of(2021) is HYBRID
    assert era_of(2022) is GROUND_EFFECT
    assert era_of(2025) is GROUND_EFFECT
    assert era_of(2026) is FUTURE_ENGINE
    assert era_of(2030) is FUTURE_ENGINE          # ongoing era is open-ended
    assert era_of(2009) is None                   # pre-modeling-window


def test_get_era_resolution():
    assert get_era("hybrid") is HYBRID
    assert get_era(GROUND_EFFECT) is GROUND_EFFECT
    with pytest.raises(KeyError, match="Unknown regulation era"):
        get_era("v10")


def test_year_range_closed_vs_ongoing():
    assert HYBRID.year_range == (2014, 2021)
    with pytest.raises(ValueError, match="ongoing"):
        _ = FUTURE_ENGINE.year_range


def test_contains():
    assert HYBRID.contains(2014) and HYBRID.contains(2021)
    assert not HYBRID.contains(2013) and not HYBRID.contains(2022)
    assert FUTURE_ENGINE.contains(2100)           # no upper bound while open


def test_frozen():
    with pytest.raises(AttributeError):
        HYBRID.end_year = 2022                    # type: ignore[misc]
