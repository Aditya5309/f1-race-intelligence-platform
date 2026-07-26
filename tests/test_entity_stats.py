"""Tests for app/entity_stats.py — the entity-stats route family's backing
logic (Frontend Overhaul Phase 5, Decision 064). Verified against small,
hand-built Ergast-shaped CSVs; every function degrades gracefully (empty
dict/list/DataFrame) rather than raising when its source data is absent."""

import pandas as pd
import pytest

from app import entity_stats as es


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "display"
    d.mkdir()

    pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "year": [2023, 2023, 2024, 2024],
        "round": [1, 2, 1, 2],
    }).to_csv(d / "races.csv", index=False)

    pd.DataFrame({
        "driverId": [1, 2],
        "forename": ["Max", "Lewis"],
        "surname": ["Verstappen", "Hamilton"],
    }).to_csv(d / "drivers.csv", index=False)

    pd.DataFrame({
        "constructorId": [1, 2],
        "name": ["Red Bull", "Mercedes"],
        "nationality": ["Austrian", "German"],
    }).to_csv(d / "constructors.csv", index=False)

    # driver 1 (constructor 1): races 1-4, wins race 1 & 3, P2 elsewhere.
    # driver 2 (constructor 2): races 1-4, opposite finishes.
    pd.DataFrame({
        "raceId": [1, 1, 2, 2, 3, 3, 4, 4],
        "driverId": [1, 2, 1, 2, 1, 2, 1, 2],
        "constructorId": [1, 2, 1, 2, 1, 2, 1, 2],
        "position": ["1", "2", "2", "1", "1", "2", "2", "1"],
        "positionOrder": [1, 2, 2, 1, 1, 2, 2, 1],
        "points": [25, 18, 18, 25, 25, 18, 18, 25],
    }).to_csv(d / "results.csv", index=False)

    pd.DataFrame({
        "raceId": [1, 1, 2, 2, 3, 3, 4, 4],
        "driverId": [1, 2, 1, 2, 1, 2, 1, 2],
        "constructorId": [1, 2, 1, 2, 1, 2, 1, 2],
        "position": [1, 2, 2, 1, 1, 2, 2, 1],
    }).to_csv(d / "qualifying.csv", index=False)

    # driver_standings final rows per race (Ergast convention: cumulative).
    pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "driverId": [1, 1, 1, 1],
        "points": [25, 43, 68, 86],
        "position": [1, 1, 1, 1],
        "wins": [1, 1, 2, 2],
    }).to_csv(d / "driver_standings.csv", index=False)
    pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "constructorId": [1, 1, 1, 1],
        "points": [25, 43, 68, 86],
        "position": [1, 1, 1, 1],
        "wins": [1, 1, 2, 2],
    }).to_csv(d / "constructor_standings.csv", index=False)

    return d


def test_driver_season_stats_uses_authoritative_standings_points(data_dir):
    stats = es.driver_season_stats(data_dir, driver_id=1, year=2023)
    assert stats["races"] == 2
    assert stats["wins"] == 1
    assert stats["podiums"] == 2
    assert stats["points"] == 43.0  # final-round-of-2023 standings, not results.csv sum
    assert stats["poles"] == 1
    assert stats["avg_quali"] == 1.5


def test_driver_season_stats_career_scope_sums_across_seasons(data_dir):
    stats = es.driver_season_stats(data_dir, driver_id=1, year=None)
    assert stats["races"] == 4
    assert stats["points"] == 43.0 + 86.0  # final row of 2023 + final row of 2024


def test_driver_season_stats_unknown_driver_returns_empty(data_dir):
    assert es.driver_season_stats(data_dir, driver_id=999, year=2023) == {}


def test_driver_season_stats_missing_data_dir_degrades_to_empty(tmp_path):
    assert es.driver_season_stats(tmp_path / "no-such-dir", driver_id=1, year=2023) == {}


def test_driver_race_results_sorted_and_has_quali_and_finish(data_dir):
    df = es.driver_race_results(data_dir, driver_id=1, year=2023)
    assert list(df["round"]) == [1, 2]
    assert list(df["quali_position"]) == [1, 2]
    assert list(df["finish"]) == [1, 2]


def test_driver_standings_progression(data_dir):
    df = es.driver_standings_progression(data_dir, driver_id=1, year=2023)
    assert list(df["round"]) == [1, 2]
    assert list(df["points"]) == [25, 43]


def test_constructor_catalog_sorted_by_name(data_dir):
    df = es.constructor_catalog(data_dir)
    assert list(df["name"]) == ["Mercedes", "Red Bull"]


def test_constructor_current_drivers(data_dir):
    names = es.constructor_current_drivers(data_dir, constructor_id=1, year=2023)
    assert names == ["Max Verstappen"]


def test_constructor_season_stats(data_dir):
    stats = es.constructor_season_stats(data_dir, constructor_id=1, year=2023)
    assert stats["races"] == 2
    assert stats["wins"] == 1
    assert stats["points"] == 43.0


def test_constructor_race_results(data_dir):
    df = es.constructor_race_results(data_dir, constructor_id=1, year=2023)
    assert list(df["round"]) == [1, 2]
    assert list(df["finish"]) == [1, 2]


def test_radar_scores_none_when_not_enough_races(data_dir):
    # Only 2 races per season in this fixture; CONSISTENCY_MIN_RACES=3.
    assert es.radar_scores(data_dir, driver_id=1, year=2023) is None


def test_radar_scores_real_pool(tmp_path):
    """A season with >= 3 races per driver, 2+ drivers — the real
    score-producing path, not just the None-guard branches above."""
    d = tmp_path / "display"
    d.mkdir()
    pd.DataFrame({"raceId": [1, 2, 3], "year": [2023, 2023, 2023], "round": [1, 2, 3]}) \
        .to_csv(d / "races.csv", index=False)
    pd.DataFrame({
        "raceId": [1, 1, 2, 2, 3, 3],
        "driverId": [1, 2, 1, 2, 1, 2],
        "constructorId": [1, 2, 1, 2, 1, 2],
        "position": ["1", "2", "1", "2", "1", "2"],
        "positionOrder": [1, 2, 1, 2, 1, 2],
        "points": [25, 18, 25, 18, 25, 18],
    }).to_csv(d / "results.csv", index=False)
    pd.DataFrame({
        "raceId": [1, 1, 2, 2, 3, 3],
        "driverId": [1, 2, 1, 2, 1, 2],
        "constructorId": [1, 2, 1, 2, 1, 2],
        "position": [1, 2, 1, 2, 1, 2],
    }).to_csv(d / "qualifying.csv", index=False)
    pd.DataFrame({
        "raceId": [1, 2, 3], "driverId": [1, 1, 1],
        "points": [25, 50, 75], "position": [1, 1, 1], "wins": [1, 2, 3],
    }).to_csv(d / "driver_standings.csv", index=False)

    scores = es.radar_scores(d, driver_id=1, year=2023)
    assert scores is not None
    assert set(scores) == set(es.RADAR_AXES)
    assert scores["Qualifying pace"] == 100.0  # driver 1 always P1, best in pool
    assert scores["Race pace"] == 100.0
