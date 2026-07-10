"""
scripts/backfill_circuit_layouts.py

One-time backfill of circuit track-outline geometry from OpenStreetMap /
Overpass — enrichment data for the Circuit Explorer dashboard page, in the
same category as artifacts/display/ (committed, not a runtime dependency).

Why OpenStreetMap and not FastF1: licensing consistency with the rest of
this project (Ergast/ODbL-compatible sourcing), per the Phase 4 Tranche A
investigation.

Why GeoJSON and not SVG: Overpass returns raw lat/lon node coordinates.
GeoJSON stores that losslessly as source data — defensible ODbL provenance
(it's the OSM coordinates themselves, not a rasterized/re-projected
derivative) and needs no new dependency: plotly (already a project
dependency) renders `go.Scatter(x=lon, y=lat, mode="lines")` straight from
it. An SVG artifact would instead bake a specific map projection and viewBox
into the committed file — harder to fix or re-render later if that choice
turns out wrong.

Circuits are the 35 unique circuitId values raced 2010-2024 (the modeling
window), derived from races.csv/circuits.csv — there is no existing config
constant enumerating them. For each circuit, queries Overpass for
`way["highway"="raceway"]` (falling back to `way["leisure"="track"]
["sport"="motor"]` if that returns nothing) within a bounding radius of the
circuit's (lat, lng), then stitches the returned way segments into a single
closed outline by matching shared endpoint node IDs (`_stitch_segments`).
Circuits where the geometry can't be cleanly closed are skipped and
reported — not every one of the 35 needs to succeed for this to be worth
running. A ring that DOES close but comes out implausibly short is also
skipped (MIN_PLAUSIBLE_PERIMETER_M). A small number of circuits with
multiple overlapping official layouts are excluded outright
(KNOWN_AMBIGUOUS_CONFIGURATIONS) because the assembly heuristic has no way
to tell which configuration it found — those need manual verification, not
an automated guess.

Paced a few seconds between requests (Overpass usage-policy courtesy; this
is a one-time 35-circuit run, not a recurring job) with a descriptive
User-Agent (also required by that policy).

    python scripts/backfill_circuit_layouts.py                # full run
    python scripts/backfill_circuit_layouts.py --dry-run       # query + stitch only, no writes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import asin, cos, radians, sin
from pathlib import Path

import httpx
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RACES_CSV = _PROJECT_ROOT / "artifacts" / "display" / "races.csv"
DEFAULT_CIRCUITS_CSV = _PROJECT_ROOT / "artifacts" / "display" / "circuits.csv"
DEFAULT_DEST = _PROJECT_ROOT / "artifacts" / "display" / "circuit_layouts"

MODELING_WINDOW = (2010, 2024)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "f1-race-winner-prediction/1.0 (one-time circuit-layout backfill; github.com)"
REQUEST_DELAY_SECONDS = 3.0
SEARCH_RADIUS_M = 3000
REQUEST_TIMEOUT_SECONDS = 60
# The public overpass-api.de mirror routinely answers a fraction of requests
# with a transient 429/504 under normal load — not a rate-limit violation on
# our side (we already pace REQUEST_DELAY_SECONDS apart), just retry it.
MAX_RETRIES = 8
RETRY_BACKOFF_SECONDS = 10.0
# Every F1 circuit in the 2010-2024 window has an official lap length of at
# least ~3.3km (Monaco, the shortest). A "closed loop" under that is not the
# main track — it's a small unrelated raceway-tagged feature the search
# radius also picked up (a paddock access road, a short karting loop, or —
# observed for a purpose-built circuit with only partial raceway tagging in
# OSM — a genuine but incomplete fragment of the real track). Street
# circuits (Albert Park, Monaco, Valencia, Baku, Vegas, Marina Bay) mostly
# run on ordinary public roads tagged as regular highways, not
# highway=raceway, so Overpass often has nothing but such small fragments to
# offer for them. This floor turns all of the above into honest skips
# instead of a malformed "successful" outline.
MIN_PLAUSIBLE_PERIMETER_M = 3200.0

# Circuits where the pipeline DOES assemble a plausible-length closed loop,
# but manual review couldn't confirm it's the specific Grand Prix
# configuration rather than a different official layout sharing the same
# OSM tarmac/tagging (both circuits below have multiple overlapping
# configurations, and the assembled ring came out longer than the real GP
# lap — 6.31km vs ~5.41km for Bahrain, 4.70km vs ~4.30km for Rodríguez). The
# "longest closed cycle" heuristic has no way to tell configurations apart,
# so rather than ship that ambiguity silently, these are excluded pending
# manual verification. Revisit only with a specific fix in hand (e.g. a
# tighter search radius, or manually identifying the GP-specific way IDs) —
# don't just delete this entry to "make it succeed" again.
KNOWN_AMBIGUOUS_CONFIGURATIONS: dict[int, str] = {
    3: "returned geometry likely doesn't match the specific GP "
       "configuration (Bahrain has 4 overlapping official layouts) — "
       "needs manual verification before shipping",
    32: "returned geometry likely doesn't match the specific GP "
        "configuration (Autódromo Hermanos Rodríguez's stadium section has "
        "multiple layouts) — needs manual verification before shipping",
}

QUERY_TEMPLATE = """
[out:json][timeout:25];
way["{key}"="{value}"](around:{radius},{lat},{lng});
out geom;
"""

# (tag key, tag value) pairs tried in order per circuit.
CANDIDATE_TAGS: tuple[tuple[str, str], ...] = (
    ("highway", "raceway"),
    ("leisure", "track"),
)


def modeling_window_circuits(
    races_csv: Path = DEFAULT_RACES_CSV, circuits_csv: Path = DEFAULT_CIRCUITS_CSV,
) -> pd.DataFrame:
    """The circuits raced in MODELING_WINDOW, with circuitId/name/lat/lng.

    No existing repo constant enumerates these — derived fresh each run from
    races.csv joined to circuits.csv, same as the Phase 4 Tranche A
    investigation did.
    """
    races = pd.read_csv(races_csv, na_values=["\\N"])
    circuits = pd.read_csv(circuits_csv, na_values=["\\N"])
    lo, hi = MODELING_WINDOW
    window = races[races["year"].between(lo, hi)]
    ids = sorted(window["circuitId"].unique())
    out = circuits[circuits["circuitId"].isin(ids)][
        ["circuitId", "circuitRef", "name", "location", "country", "lat", "lng"]
    ].sort_values("circuitId")
    return out.reset_index(drop=True)


def _post_with_retry(client: httpx.Client, query: str) -> dict:
    """POST to Overpass, retrying transient errors (the public mirror
    routinely answers a fraction of requests with 429/504 under load)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(OVERPASS_URL, data={"data": query})
            if response.status_code == 200:
                return response.json()
            last_exc = httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase}",
                request=response.request, response=response,
            )
        except httpx.HTTPError as exc:
            last_exc = exc
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"Overpass request failed after {MAX_RETRIES} attempts") from last_exc


def _query_overpass(client: httpx.Client, lat: float, lng: float) -> list[dict]:
    """Try each candidate tag until one returns way(s); [] if none do."""
    for key, value in CANDIDATE_TAGS:
        query = QUERY_TEMPLATE.format(
            key=key, value=value, radius=SEARCH_RADIUS_M, lat=lat, lng=lng
        )
        data = _post_with_retry(client, query)
        elements = data.get("elements", [])
        ways = [e for e in elements if e.get("type") == "way" and e.get("geometry")]
        if ways:
            return ways
        time.sleep(REQUEST_DELAY_SECONDS)
    return []


def _group_connected_components(ways: list[dict]) -> list[list[dict]]:
    """
    Group way segments into connected components by shared endpoint node
    IDs (union-find). A ~3km search radius routinely also catches unrelated
    nearby raceway-tagged features (a support/training circuit, a separate
    pit-lane loop) that never connect to the main track — grouping first,
    rather than assuming every returned way belongs to one track, is what
    lets those be told apart generically (no circuit-specific tag/name
    matching).
    """
    segments = [
        {"nodes": w["nodes"], "points": [(pt["lat"], pt["lon"]) for pt in w["geometry"]]}
        for w in ways
    ]
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    endpoint_owner: dict[int, int] = {}
    for i, seg in enumerate(segments):
        for node_id in (seg["nodes"][0], seg["nodes"][-1]):
            if node_id in endpoint_owner:
                union(i, endpoint_owner[node_id])
            else:
                endpoint_owner[node_id] = i

    components: dict[int, list[dict]] = {}
    for i, seg in enumerate(segments):
        components.setdefault(find(i), []).append(seg)
    return list(components.values())


# Safety valve on the DFS below — a one-time script must not hang on a
# pathological component; real racetrack graphs (a loop plus a handful of
# pit-lane/service-road branches) explore far fewer paths than this.
MAX_CYCLE_SEARCH_STEPS = 200_000
# Components larger than this are skipped for cycle search entirely — not a
# realistic single-circuit graph (more likely several distinct nearby
# features the ~3km radius also picked up).
MAX_CYCLE_SEARCH_EDGES = 60


def _longest_cycle(segments: list[dict]) -> list[tuple[float, float]] | None:
    """
    Find the longest (by physical perimeter) simple cycle within one
    connected component, via DFS over the node/edge graph (nodes = OSM node
    IDs that are a segment endpoint; edges = segments).

    This is deliberately more general than "stitch every segment into one
    chain": real OSM racetrack data has junction nodes (a pit-lane entry/
    exit where 3+ segments meet), which breaks a pure non-branching chain
    assumption. Exploring all simple (non-node-repeating) closed walks and
    keeping the longest one generically recovers the main track loop while
    leaving pit-lane/service-road spurs unused, with no circuit-specific
    tag or name matching.

    Returns None if the component has too many edges to search safely, or
    no cycle closes at all.
    """
    if len(segments) > MAX_CYCLE_SEARCH_EDGES:
        return None

    node_coord: dict[int, tuple[float, float]] = {}
    adjacency: dict[int, list[tuple[int, int, bool]]] = {}
    self_closed: list[list[tuple[float, float]]] = []
    for idx, seg in enumerate(segments):
        a, b = seg["nodes"][0], seg["nodes"][-1]
        node_coord[a] = seg["points"][0]
        node_coord[b] = seg["points"][-1]
        if a == b:
            if len(seg["points"]) >= 4:
                self_closed.append(seg["points"])
            continue
        adjacency.setdefault(a, []).append((b, idx, True))
        adjacency.setdefault(b, []).append((a, idx, False))

    best_ring: list[tuple[float, float]] | None = None
    best_perimeter = 0.0
    steps = 0

    def edge_points(idx: int, forward: bool) -> list[tuple[float, float]]:
        pts = segments[idx]["points"]
        return pts if forward else list(reversed(pts))

    def dfs(start: int, current: int, used_edges: set[int],
            points: list[tuple[float, float]], visited: set[int]) -> None:
        nonlocal best_ring, best_perimeter, steps
        for neighbor, idx, forward in adjacency.get(current, []):
            steps += 1
            if steps > MAX_CYCLE_SEARCH_STEPS or idx in used_edges:
                continue
            new_points = points + edge_points(idx, forward)[1:]
            if neighbor == start and len(used_edges) >= 2:
                perimeter = _ring_perimeter_m(new_points)
                if perimeter > best_perimeter:
                    best_perimeter, best_ring = perimeter, new_points
                continue
            if neighbor in visited:
                continue
            used_edges.add(idx)
            visited.add(neighbor)
            dfs(start, neighbor, used_edges, new_points, visited)
            used_edges.discard(idx)
            visited.discard(neighbor)

    for start in list(adjacency):
        if steps > MAX_CYCLE_SEARCH_STEPS:
            break
        dfs(start, start, set(), [node_coord[start]], {start})

    for ring in self_closed:
        perimeter = _ring_perimeter_m(ring)
        if perimeter > best_perimeter:
            best_perimeter, best_ring = perimeter, ring

    return best_ring


def _ring_perimeter_m(ring: list[tuple[float, float]]) -> float:
    """Haversine perimeter of a closed lat/lon ring, in meters."""
    earth_radius_m = 6_371_000.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(ring, ring[1:]):
        p1, p2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
        total += 2 * earth_radius_m * asin(min(1.0, a ** 0.5))
    return total


def _stitch_segments(ways: list[dict]) -> list[tuple[float, float]] | None:
    """
    Assemble the circuit's main outline from the Overpass result: group into
    connected components, find the longest closed cycle within each (see
    _longest_cycle — handles junction nodes like a pit-lane entry/exit
    generically), and return the ring with the LARGEST physical perimeter
    across all components (the main track loop is reliably the longest
    cycle within the search radius — a support/training circuit or stray
    pit-lane loop nearby is physically much smaller). None if nothing
    closes at all.
    """
    if not ways:
        return None

    best_ring: list[tuple[float, float]] | None = None
    best_perimeter = 0.0
    for component in _group_connected_components(ways):
        ring = _longest_cycle(component)
        if ring is None:
            continue
        perimeter = _ring_perimeter_m(ring)
        if perimeter > best_perimeter:
            best_ring, best_perimeter = ring, perimeter
    return best_ring


def _to_geojson(circuit_id: int, name: str, ring: list[tuple[float, float]]) -> dict:
    """GeoJSON Feature, [lon, lat] coordinate order per the GeoJSON spec."""
    return {
        "type": "Feature",
        "properties": {
            "circuitId": int(circuit_id),
            "name": name,
            "attribution": "© OpenStreetMap contributors",
            "license": "ODbL 1.0 (https://opendatacommons.org/licenses/odbl/)",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in ring],
        },
    }


def backfill_circuit_layouts(
    dest: Path = DEFAULT_DEST,
    races_csv: Path = DEFAULT_RACES_CSV,
    circuits_csv: Path = DEFAULT_CIRCUITS_CSV,
    dry_run: bool = False,
    circuit_ids: list[int] | None = None,
) -> tuple[list[str], list[str]]:
    """Run the backfill. Returns (succeeded, skipped) circuitRef lists.

    `circuit_ids`, if given, restricts the run to those circuitId(s) —
    for a targeted re-run (e.g. after a heuristic change) without
    re-querying the whole modeling window against the flaky public
    Overpass mirror.
    """
    circuits = modeling_window_circuits(races_csv, circuits_csv)
    if circuit_ids is not None:
        circuits = circuits[circuits["circuitId"].isin(circuit_ids)]
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    succeeded: list[str] = []
    skipped: list[str] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS,
    ) as client:
        for row in circuits.itertuples():
            label = f"{row.circuitId} {row.circuitRef} ({row.name})"
            if row.circuitId in KNOWN_AMBIGUOUS_CONFIGURATIONS:
                print(f"SKIP {label}: {KNOWN_AMBIGUOUS_CONFIGURATIONS[row.circuitId]}")
                skipped.append(row.circuitRef)
                continue
            if pd.isna(row.lat) or pd.isna(row.lng):
                print(f"SKIP {label}: no lat/lng in circuits.csv")
                skipped.append(row.circuitRef)
                continue

            ways = _query_overpass(client, row.lat, row.lng)
            time.sleep(REQUEST_DELAY_SECONDS)
            if not ways:
                print(f"SKIP {label}: Overpass returned no raceway/track ways")
                skipped.append(row.circuitRef)
                continue

            ring = _stitch_segments(ways)
            if ring is None:
                print(f"SKIP {label}: {len(ways)} way segment(s) did not stitch "
                      "into one clean closed loop")
                skipped.append(row.circuitRef)
                continue

            perimeter_m = _ring_perimeter_m(ring)
            if perimeter_m < MIN_PLAUSIBLE_PERIMETER_M:
                print(f"SKIP {label}: assembled ring is only {perimeter_m:.0f}m — "
                      "implausibly short for a full F1 lap (likely an unrelated "
                      "small feature near the circuit, not the main track)")
                skipped.append(row.circuitRef)
                continue

            feature = _to_geojson(row.circuitId, row.name, ring)
            print(f"OK   {label}: {len(ways)} segment(s) -> {len(ring)} points, "
                  f"{perimeter_m / 1000:.2f} km")
            succeeded.append(row.circuitRef)
            if not dry_run:
                out_path = dest / f"{row.circuitId}.json"
                out_path.write_text(json.dumps(feature), encoding="utf-8")

    return succeeded, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Output directory for per-circuit GeoJSON (default: {DEFAULT_DEST}).")
    parser.add_argument("--races-csv", type=Path, default=DEFAULT_RACES_CSV)
    parser.add_argument("--circuits-csv", type=Path, default=DEFAULT_CIRCUITS_CSV)
    parser.add_argument("--dry-run", action="store_true",
                        help="Query and stitch but do not write any files.")
    parser.add_argument("--circuit-ids", default=None,
                        help="Comma-separated circuitId(s) to (re-)run, "
                             "instead of the full 2010-2024 modeling window "
                             "— e.g. '1,6' to retry just Albert Park and Monaco.")
    args = parser.parse_args(argv)

    if not args.races_csv.exists() or not args.circuits_csv.exists():
        print(f"ERROR: {args.races_csv} / {args.circuits_csv} not found.", file=sys.stderr)
        return 1

    circuit_ids = (
        [int(c) for c in args.circuit_ids.split(",")] if args.circuit_ids else None
    )
    succeeded, skipped = backfill_circuit_layouts(
        dest=args.dest, races_csv=args.races_csv, circuits_csv=args.circuits_csv,
        dry_run=args.dry_run, circuit_ids=circuit_ids,
    )
    print(f"\n{len(succeeded)} succeeded, {len(skipped)} skipped "
          f"(of {len(succeeded) + len(skipped)} circuits in the modeling window).")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
