"""
Tests for scripts/backfill_circuit_layouts.py's pure graph/geometry logic —
no network calls (the live Overpass query path is exercised manually, not
in CI).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_circuit_layouts import (  # noqa: E402
    _group_connected_components,
    _longest_cycle,
    _ring_perimeter_m,
    _stitch_segments,
    _to_geojson,
)


def _way(node_ids: list[int], points: list[tuple[float, float]]) -> dict:
    """Build a synthetic Overpass way element: nodes + geometry (lat/lon)."""
    return {
        "type": "way",
        "nodes": node_ids,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in points],
    }


# A unit square loop: node ids 1-2-3-4-1, one way per edge.
_SQUARE_POINTS = {1: (0.0, 0.0), 2: (0.0, 0.001), 3: (0.001, 0.001), 4: (0.001, 0.0)}


def _square_loop(offset: int = 0) -> list[dict]:
    pts = {k: v for k, v in _SQUARE_POINTS.items()}
    ids = [1 + offset, 2 + offset, 3 + offset, 4 + offset]
    ways = []
    for a, b in zip(ids, ids[1:] + ids[:1]):
        ways.append(_way([a, b], [pts[a - offset], pts[b - offset]]))
    return ways


def test_group_connected_components_separates_disjoint_loops():
    loop1 = _square_loop(offset=0)
    loop2 = _square_loop(offset=100)
    components = _group_connected_components(loop1 + loop2)
    assert len(components) == 2
    assert sorted(len(c) for c in components) == [4, 4]


def test_longest_cycle_closes_simple_square():
    ring = _longest_cycle(
        [{"nodes": w["nodes"], "points": [(p["lat"], p["lon"]) for p in w["geometry"]]}
         for w in _square_loop()]
    )
    assert ring is not None
    assert ring[0] == ring[-1]
    # 4 ways x 2 points, each successive way dropping its shared first point
    # -> 4 unique points + the repeated closing point.
    assert len(ring) == 5


def test_longest_cycle_ignores_dead_end_spur():
    loop = _square_loop()
    # A spur off node 1 to a brand-new dead-end node (id 99) — shares node 1
    # with the loop (same connected component) but cannot close a cycle.
    spur = _way([1, 99], [(0.0, 0.0), (0.0, -0.001)])
    segments = [
        {"nodes": w["nodes"], "points": [(p["lat"], p["lon"]) for p in w["geometry"]]}
        for w in loop + [spur]
    ]
    ring = _longest_cycle(segments)
    assert ring is not None
    assert ring[0] == ring[-1]
    assert len(ring) == 5  # the spur contributes no points to the winning cycle


def test_longest_cycle_none_when_no_loop_closes():
    # A simple open chain 1-2-3-4, no edge back to 1.
    segments = [
        {"nodes": [1, 2], "points": [(0.0, 0.0), (0.0, 0.001)]},
        {"nodes": [2, 3], "points": [(0.0, 0.001), (0.001, 0.001)]},
        {"nodes": [3, 4], "points": [(0.001, 0.001), (0.001, 0.0)]},
    ]
    assert _longest_cycle(segments) is None


def test_stitch_segments_picks_largest_perimeter_component():
    small = _square_loop(offset=0)     # ~0.001 deg square
    big_pts = {1: (0.0, 0.0), 2: (0.0, 0.01), 3: (0.01, 0.01), 4: (0.01, 0.0)}
    big = []
    ids = [101, 102, 103, 104]
    for a, b in zip(ids, ids[1:] + ids[:1]):
        big.append(_way([a, b], [big_pts[a - 100], big_pts[b - 100]]))

    ring = _stitch_segments(small + big)
    assert ring is not None
    # The "big" loop (10x the linear scale) must win on physical perimeter.
    assert _ring_perimeter_m(ring) > 1000


def test_stitch_segments_empty_input():
    assert _stitch_segments([]) is None


def test_ring_perimeter_hand_computed():
    # Two points 0.001 deg of latitude apart (~111.32 m) and back, at the
    # equator so longitude scaling doesn't complicate the hand check.
    ring = [(0.0, 0.0), (0.001, 0.0), (0.0, 0.0)]
    perimeter = _ring_perimeter_m(ring)
    assert perimeter == pytest.approx(2 * 111.32, rel=0.01)


def test_to_geojson_structure_and_coordinate_order():
    ring = [(51.5, -1.0), (51.6, -1.1), (51.5, -1.0)]
    feature = _to_geojson(9, "Silverstone Circuit", ring)
    assert feature["type"] == "Feature"
    assert feature["properties"]["circuitId"] == 9
    assert feature["properties"]["name"] == "Silverstone Circuit"
    assert "OpenStreetMap" in feature["properties"]["attribution"]
    assert feature["geometry"]["type"] == "LineString"
    # GeoJSON coordinate order is [lon, lat], the reverse of the (lat, lon)
    # tuples _stitch_segments/_longest_cycle work with internally.
    assert feature["geometry"]["coordinates"][0] == [-1.0, 51.5]
