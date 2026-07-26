"""
app/circuit_stats.py

Circuit-grain computations for the Frontend Overhaul's Circuit Explorer
page — a standalone data need (per the roadmap), not part of the driver/
constructor entity-stats route family (app/entity_stats.py), but reusing
the same conventions: pure pandas, no Streamlit dependency, a
caller-supplied data_dir, and graceful degradation (empty dict/list/None,
never raises) when source data is absent.

app/views/metadata.py already computes every one of these things but is
Streamlit-coupled (imports streamlit, st.cache_data) and therefore
off-limits for app/api.py to import — this module duplicates that logic,
the same choice Decision 062/064 made for the other route families.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: Below this many races with a valid grid->finish position change, an
#: average is noise, not signal — mirrors app/views/metadata.py's own floor.
POSITION_CHANGE_MIN_RACES = 3


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_dir / name, na_values=["\\N"])
    except Exception:                                            # noqa: BLE001
        return pd.DataFrame()


def _race_catalog(data_dir: Path) -> pd.DataFrame:
    races = _read_csv(data_dir, "races.csv")
    if races.empty:
        return pd.DataFrame()
    cols = [c for c in ("raceId", "year", "round", "circuitId") if c in races.columns]
    return races[cols]


def circuit_catalog(data_dir: Path) -> pd.DataFrame:
    """circuitId, name, country, location, races_held — name-sorted.
    Mirrors app/views/metadata.py::circuit_catalog exactly."""
    circuits = _read_csv(data_dir, "circuits.csv")
    if circuits.empty:
        return pd.DataFrame()
    catalog = _race_catalog(data_dir)
    counts = (
        catalog.groupby("circuitId")["raceId"].nunique().rename("races_held")
        if not catalog.empty else pd.Series(dtype=int, name="races_held")
    )
    out = circuits[["circuitId", "name", "country", "location"]].merge(
        counts, on="circuitId", how="left")
    out["races_held"] = out["races_held"].fillna(0).astype(int)
    return out.sort_values("name")


def circuit_layout(data_dir: Path, circuit_id: int) -> dict | None:
    """Track-outline GeoJSON Feature (OpenStreetMap backfill,
    scripts/backfill_circuit_layouts.py) for one circuit. None when the
    file is absent (most circuits: OSM data didn't cleanly assemble, or
    this circuit isn't one of the backfilled set) — mirrors
    app/views/metadata.py::circuit_layout exactly."""
    path = data_dir / "circuit_layouts" / f"{circuit_id}.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                            # noqa: BLE001
        return None


def _parse_laptime(value: str) -> float | None:
    """'M:SS.sss' fastestLapTime string -> total seconds. Mirrors
    app/views/metadata.py::_parse_laptime exactly."""
    if not isinstance(value, str) or ":" not in value:
        return None
    minutes, _, seconds = value.partition(":")
    try:
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def circuit_stats(data_dir: Path, circuit_id: int) -> dict:
    """All-time stats for one circuit: races_held, first_year, last_year,
    dnf_rate, avg_position_change, fastest_lap, most_wins,
    winners_by_year. {} when there's no race history for this circuit.
    Mirrors app/views/metadata.py::circuit_stats exactly (real, computed
    proxies — not invented "difficulty" scores; every field needs the
    same caveats that module's own docstring states)."""
    catalog = _race_catalog(data_dir)
    if catalog.empty:
        return {}
    races_here = catalog[catalog["circuitId"] == circuit_id]
    if races_here.empty:
        return {}
    stats: dict = {
        "races_held": int(races_here["raceId"].nunique()),
        "first_year": int(races_here["year"].min()),
        "last_year": int(races_here["year"].max()),
    }
    results = _read_csv(data_dir, "results.csv")
    if results.empty:
        return stats
    r = results.merge(races_here[["raceId", "year"]], on="raceId", how="inner")
    if r.empty:
        return stats

    stats["dnf_rate"] = float(r["position"].isna().mean())

    valid = r[(r["grid"] > 0) & r["position"].notna()]
    if len(valid) >= POSITION_CHANGE_MIN_RACES:
        stats["avg_position_change"] = float(
            (valid["grid"] - pd.to_numeric(valid["position"])).mean())

    laps = r.dropna(subset=["fastestLapTime"]).copy()
    if len(laps):
        laps["_seconds"] = laps["fastestLapTime"].map(_parse_laptime)
        laps = laps.dropna(subset=["_seconds"])
        if len(laps):
            row = laps.loc[laps["_seconds"].idxmin()]
            drivers = _read_csv(data_dir, "drivers.csv")
            driver_name = None
            if not drivers.empty:
                d = drivers[drivers["driverId"] == row["driverId"]]
                if len(d):
                    driver_name = f"{d.iloc[0]['forename']} {d.iloc[0]['surname']}"
            stats["fastest_lap"] = {
                "time": row["fastestLapTime"], "year": int(row["year"]),
                "driver": driver_name or f"driver {int(row['driverId'])}",
            }

    winners = r[pd.to_numeric(r["position"], errors="coerce") == 1].copy()
    if len(winners):
        drivers = _read_csv(data_dir, "drivers.csv")
        if not drivers.empty:
            names = drivers.assign(
                driver=drivers["forename"].astype(str) + " " + drivers["surname"].astype(str)
            )[["driverId", "driver"]]
            winners = winners.merge(names, on="driverId", how="left")
        else:
            winners["driver"] = winners["driverId"].astype(str)
        constructors = _read_csv(data_dir, "constructors.csv")
        if not constructors.empty:
            cnames = constructors.rename(columns={"name": "constructor"})[
                ["constructorId", "constructor"]]
            winners = winners.merge(cnames, on="constructorId", how="left")
        stats["winners_by_year"] = winners[
            ["year", "driver", "constructor"] if "constructor" in winners.columns
            else ["year", "driver"]
        ].sort_values("year", ascending=False).reset_index(drop=True)
        stats["most_wins"] = (
            winners.groupby("driver").size().rename("wins")
            .sort_values(ascending=False).head(5).reset_index()
        )
    return stats
