"""Tests for app/circuit_stats.py — Circuit Explorer's backing logic
(Frontend Overhaul Phase 6, Decision 065). Degrades gracefully (empty
dict/list/None) rather than raising when source data is absent."""

import json

import pandas as pd
import pytest

from app import circuit_stats as cs


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "display"
    d.mkdir()

    pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "year": [2022, 2023, 2024, 2021],
        "round": [1, 1, 1, 1],
        "circuitId": [10, 10, 10, 10],
    }).to_csv(d / "races.csv", index=False)
    pd.DataFrame({
        "circuitId": [10, 20],
        "name": ["Silverstone", "Monza"],
        "country": ["UK", "Italy"],
        "location": ["Silverstone", "Monza"],
    }).to_csv(d / "circuits.csv", index=False)
    pd.DataFrame({
        "driverId": [1, 2], "forename": ["Max", "Lewis"], "surname": ["Verstappen", "Hamilton"],
    }).to_csv(d / "drivers.csv", index=False)
    pd.DataFrame({
        "constructorId": [1, 2], "name": ["Red Bull", "Mercedes"],
    }).to_csv(d / "constructors.csv", index=False)
    pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "driverId": [1, 2, 1, 2],
        "constructorId": [1, 2, 1, 2],
        "grid": [1, 3, 2, 2],
        "position": ["1", "1", None, "2"],
        "fastestLapTime": ["1:30.500", "1:29.123", None, None],
    }).to_csv(d / "results.csv", index=False)

    return d


def test_circuit_catalog_sorted_with_races_held(data_dir):
    df = cs.circuit_catalog(data_dir)
    assert list(df["name"]) == ["Monza", "Silverstone"]
    silverstone = df[df["name"] == "Silverstone"].iloc[0]
    assert silverstone["races_held"] == 4
    monza = df[df["name"] == "Monza"].iloc[0]
    assert monza["races_held"] == 0


def test_circuit_layout_reads_geojson(tmp_path):
    d = tmp_path / "display" / "circuit_layouts"
    d.mkdir(parents=True)
    feature = {"type": "Feature", "properties": {"attribution": "test"},
               "geometry": {"type": "LineString", "coordinates": [[1.0, 2.0]]}}
    (d / "10.json").write_text(json.dumps(feature))

    result = cs.circuit_layout(tmp_path / "display", circuit_id=10)
    assert result == feature


def test_circuit_layout_missing_returns_none(tmp_path):
    assert cs.circuit_layout(tmp_path / "display", circuit_id=999) is None


def test_circuit_stats_full(data_dir):
    stats = cs.circuit_stats(data_dir, circuit_id=10)
    assert stats["races_held"] == 4
    assert stats["first_year"] == 2021
    assert stats["last_year"] == 2024
    assert stats["dnf_rate"] == pytest.approx(1 / 4)
    # valid (grid>0 & position notna) rows: (1,1)->0, (3,1)->2, (2,2)->0
    assert stats["avg_position_change"] == pytest.approx((0 + 2 + 0) / 3)
    assert stats["fastest_lap"]["driver"] == "Lewis Hamilton"
    assert stats["fastest_lap"]["time"] == "1:29.123"
    assert set(stats["most_wins"]["driver"]) == {"Max Verstappen", "Lewis Hamilton"}
    assert list(stats["winners_by_year"]["year"]) == [2023, 2022]


def test_circuit_stats_unknown_circuit_returns_empty(data_dir):
    assert cs.circuit_stats(data_dir, circuit_id=999) == {}


def test_circuit_stats_missing_data_dir_degrades_to_empty(tmp_path):
    assert cs.circuit_stats(tmp_path / "no-such-dir", circuit_id=10) == {}
