"""
app/entity_stats.py

Driver/constructor entity-stats computations for the Frontend Overhaul's
"entity stats API" route family (Driver Explorer, Compare Drivers, Team —
the trio the roadmap groups together since they all block on this same
missing capability). Pure pandas logic reading Ergast-format CSVs from a
caller-supplied data_dir — deliberately NO Streamlit dependency, so
app/api.py can import this module directly.

app/views/metadata.py already computes every one of these things, but it is
Streamlit-coupled (imports streamlit, every function wrapped in
st.cache_data) and is therefore off-limits for app/api.py to import (the
same "app/views/ and app/api.py don't cross-import Streamlit-coupled code"
boundary that led app/api.py's own /standings/{year} route (Decision 062)
to duplicate app/views/metadata.py::_final_round_standings rather than
share it). This module continues that established choice at a larger
scale, rather than refactoring metadata.py to extract a shared core — that
refactor is explicitly out of scope for a frontend-conversion phase.

Every function degrades gracefully (empty dict/list/DataFrame, never
raises) when the source CSVs are absent, matching every other
data_dir-sourced lookup in this API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Below this many classified races, a std-dev "consistency" figure is
#: noise, not signal — mirrors app/views/metadata.py's own floor.
CONSISTENCY_MIN_RACES = 3


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_dir / name, na_values=["\\N"])
    except Exception:                                            # noqa: BLE001
        return pd.DataFrame()


def _race_catalog(data_dir: Path) -> pd.DataFrame:
    """raceId/year/round only — the subset every function below actually
    needs (unlike metadata.py's race_catalog(), no circuit/country join)."""
    races = _read_csv(data_dir, "races.csv")
    if races.empty:
        return pd.DataFrame()
    cols = [c for c in ("raceId", "year", "round") if c in races.columns]
    return races[cols]


def _final_round_standings(data_dir: Path, standings_csv: str, year: int) -> pd.DataFrame:
    """Standings rows keyed to the season's LAST race that has standings —
    mirrors app/api.py's own _final_round_standings (Decision 062) exactly."""
    standings = _read_csv(data_dir, standings_csv)
    catalog = _race_catalog(data_dir)
    if standings.empty or catalog.empty:
        return pd.DataFrame()
    season = catalog.loc[catalog["year"] == year, ["raceId", "round"]]
    merged = standings.merge(season, on="raceId", how="inner")
    if merged.empty:
        return pd.DataFrame()
    return merged[merged["round"] == merged["round"].max()].copy()


def _authoritative_points(
    data_dir: Path, standings_csv: str, id_col: str, entity_id: int, year: int | None,
) -> float | None:
    """Points from the *_standings.csv tables (includes sprint points) —
    year=None sums each season's own final-standings row across every
    season the entity appears in. None if the entity never appears at all.
    Mirrors app/views/metadata.py::_authoritative_points exactly."""
    if year is not None:
        rows = _final_round_standings(data_dir, standings_csv, year)
        rows = rows[rows[id_col] == entity_id]
        return float(rows["points"].iloc[0]) if len(rows) else None

    standings = _read_csv(data_dir, standings_csv)
    catalog = _race_catalog(data_dir)
    if standings.empty or catalog.empty:
        return None
    entity_years = catalog.merge(
        standings.loc[standings[id_col] == entity_id, ["raceId"]],
        on="raceId", how="inner")["year"].unique()
    if not len(entity_years):
        return None
    total = 0.0
    for yr in entity_years:
        rows = _final_round_standings(data_dir, standings_csv, int(yr))
        rows = rows[rows[id_col] == entity_id]
        if len(rows):
            total += float(rows["points"].iloc[0])
    return total


def driver_season_stats(data_dir: Path, driver_id: int, year: int | None = None) -> dict:
    """races/wins/podiums/poles/points/avg_quali/avg_finish/
    avg_finish_classified/consistency_std — one season, or career when
    year=None. Mirrors app/views/metadata.py::driver_season_stats exactly."""
    results = _read_csv(data_dir, "results.csv")
    catalog = _race_catalog(data_dir)
    if results.empty or catalog.empty:
        return {}
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results.merge(scope[["raceId"]], on="raceId", how="inner")
    r = r[r["driverId"] == driver_id]
    if r.empty:
        return {}
    position = pd.to_numeric(r["position"], errors="coerce")
    points = _authoritative_points(data_dir, "driver_standings.csv", "driverId", driver_id, year)
    stats = {
        "races": int(len(r)),
        "wins": int((position == 1).sum()),
        "podiums": int((position <= 3).sum()),
        "points": points if points is not None else float(r["points"].sum()),
        "avg_finish": float(r["positionOrder"].mean()),
    }
    classified = position.dropna()
    if len(classified):
        stats["avg_finish_classified"] = float(classified.mean())
        if len(classified) >= CONSISTENCY_MIN_RACES:
            stats["consistency_std"] = float(classified.std())
    quali = _read_csv(data_dir, "qualifying.csv")
    if not quali.empty:
        q = quali.merge(scope[["raceId"]], on="raceId", how="inner")
        qpos = pd.to_numeric(q.loc[q["driverId"] == driver_id, "position"],
                             errors="coerce").dropna()
        if len(qpos):
            stats["poles"] = int((qpos == 1).sum())
            stats["avg_quali"] = float(qpos.mean())
    return stats


def driver_race_results(data_dir: Path, driver_id: int, year: int | None = None) -> pd.DataFrame:
    """(year, round)-sorted per-race grid/quali/finish rows for one driver.
    Mirrors app/views/metadata.py::driver_race_results exactly."""
    results = _read_csv(data_dir, "results.csv")
    catalog = _race_catalog(data_dir)
    if results.empty or catalog.empty:
        return pd.DataFrame()
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results[results["driverId"] == driver_id].merge(
        scope[["raceId", "year", "round"]], on="raceId", how="inner")
    if r.empty:
        return pd.DataFrame()
    quali = _read_csv(data_dir, "qualifying.csv")
    if not quali.empty:
        q = quali.loc[quali["driverId"] == driver_id, ["raceId", "position"]].rename(
            columns={"position": "quali_position"})
        r = r.merge(q, on="raceId", how="left")
    r["finish"] = pd.to_numeric(r["positionOrder"], errors="coerce")
    return r.sort_values(["year", "round"])


def driver_standings_progression(data_dir: Path, driver_id: int, year: int) -> pd.DataFrame:
    """Round-by-round championship points/position for one driver-season.
    Mirrors app/views/metadata.py::driver_standings_progression exactly."""
    standings = _read_csv(data_dir, "driver_standings.csv")
    catalog = _race_catalog(data_dir)
    if standings.empty or catalog.empty:
        return pd.DataFrame()
    season = catalog.loc[catalog["year"] == year, ["raceId", "round"]]
    rows = standings.merge(season, on="raceId", how="inner")
    rows = rows[rows["driverId"] == driver_id]
    return rows.sort_values("round")[["round", "points", "position", "wins"]]


def constructor_catalog(data_dir: Path) -> pd.DataFrame:
    """constructorId, name, nationality. Mirrors
    app/views/metadata.py::constructor_catalog exactly."""
    constructors = _read_csv(data_dir, "constructors.csv")
    if constructors.empty:
        return pd.DataFrame()
    return constructors[["constructorId", "name", "nationality"]].sort_values("name")


def constructor_current_drivers(data_dir: Path, constructor_id: int, year: int) -> list[str]:
    """Driver display names who raced for this constructor in this season.
    Mirrors app/views/metadata.py::constructor_current_drivers exactly."""
    results = _read_csv(data_dir, "results.csv")
    catalog = _race_catalog(data_dir)
    drivers = _read_csv(data_dir, "drivers.csv")
    if results.empty or catalog.empty or drivers.empty:
        return []
    season = catalog.loc[catalog["year"] == year, ["raceId"]]
    r = results.merge(season, on="raceId", how="inner")
    r = r[r["constructorId"] == constructor_id]
    if r.empty:
        return []
    names = drivers[drivers["driverId"].isin(r["driverId"].unique())]
    return sorted((names["forename"].astype(str) + " " + names["surname"].astype(str)).tolist())


def constructor_season_stats(
    data_dir: Path, constructor_id: int, year: int | None = None,
) -> dict:
    """races/wins/podiums/points/avg_quali for one constructor. Mirrors
    app/views/metadata.py::constructor_season_stats exactly."""
    results = _read_csv(data_dir, "results.csv")
    catalog = _race_catalog(data_dir)
    if results.empty or catalog.empty:
        return {}
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results.merge(scope[["raceId"]], on="raceId", how="inner")
    r = r[r["constructorId"] == constructor_id]
    if r.empty:
        return {}
    position = pd.to_numeric(r["position"], errors="coerce")
    points = _authoritative_points(
        data_dir, "constructor_standings.csv", "constructorId", constructor_id, year)
    stats = {
        "races": int(r["raceId"].nunique()),
        "wins": int((position == 1).sum()),
        "podiums": int((position <= 3).sum()),
        "points": points if points is not None else float(r["points"].sum()),
    }
    quali = _read_csv(data_dir, "qualifying.csv")
    if not quali.empty:
        q = quali.merge(scope[["raceId"]], on="raceId", how="inner")
        qpos = pd.to_numeric(
            q.loc[q["constructorId"] == constructor_id, "position"],
            errors="coerce").dropna()
        if len(qpos):
            stats["avg_quali"] = float(qpos.mean())
    return stats


def constructor_race_results(
    data_dir: Path, constructor_id: int, year: int | None = None,
) -> pd.DataFrame:
    """(year, round)-sorted per-race rows for one constructor: mean
    qualifying position across both cars, best finish that round. Mirrors
    app/views/metadata.py::constructor_race_results exactly."""
    results = _read_csv(data_dir, "results.csv")
    catalog = _race_catalog(data_dir)
    if results.empty or catalog.empty:
        return pd.DataFrame()
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results[results["constructorId"] == constructor_id].merge(
        scope[["raceId", "year", "round"]], on="raceId", how="inner")
    if r.empty:
        return pd.DataFrame()
    r["finish"] = pd.to_numeric(r["positionOrder"], errors="coerce")
    agg = r.groupby(["raceId", "year", "round"], as_index=False)["finish"].min()
    quali = _read_csv(data_dir, "qualifying.csv")
    if not quali.empty:
        q = quali[quali["constructorId"] == constructor_id]
        quali_by_race = (
            q.groupby("raceId")["position"].mean().rename("quali_position").reset_index()
        )
        agg = agg.merge(quali_by_race, on="raceId", how="left")
    return agg.sort_values(["year", "round"])


RADAR_AXES = ("Qualifying pace", "Race pace", "Consistency")


def _season_driver_ids(data_dir: Path, year: int) -> list[int]:
    results = _read_csv(data_dir, "results.csv")
    catalog = _race_catalog(data_dir)
    if results.empty or catalog.empty:
        return []
    season = catalog.loc[catalog["year"] == year, ["raceId"]]
    r = results.merge(season, on="raceId", how="inner")
    return sorted(r["driverId"].unique().tolist())


def _season_radar_pool(data_dir: Path, year: int) -> pd.DataFrame:
    """avg_quali / avg_finish_classified / consistency_std for every driver
    with enough races this season to have all three — the normalization
    pool radar_scores() measures one driver against."""
    rows = []
    for driver_id in _season_driver_ids(data_dir, year):
        stats = driver_season_stats(data_dir, driver_id, year)
        if {"avg_quali", "avg_finish_classified", "consistency_std"} <= stats.keys():
            rows.append({
                "driver_id": driver_id,
                "avg_quali": stats["avg_quali"],
                "avg_finish_classified": stats["avg_finish_classified"],
                "consistency_std": stats["consistency_std"],
            })
    return pd.DataFrame(rows)


def _radar_axis_score(value: float, pool: pd.Series) -> float:
    """0-100, smaller raw value = better -> bigger score. Degenerate pool
    (every driver tied) scores everyone 100 rather than dividing by zero."""
    lo, hi = pool.min(), pool.max()
    if hi <= lo:
        return 100.0
    return 100.0 * (1 - (value - lo) / (hi - lo))


def radar_scores(data_dir: Path, driver_id: int, year: int) -> dict[str, float] | None:
    """{axis: 0-100 score} for Qualifying pace / Race pace / Consistency,
    normalized within this season's field. None when this driver lacks
    enough races for all three stats, or the field is too small (< 2
    qualifying drivers) to normalize against. Mirrors
    app/views/metadata.py::radar_scores exactly."""
    stats = driver_season_stats(data_dir, driver_id, year)
    if not {"avg_quali", "avg_finish_classified", "consistency_std"} <= stats.keys():
        return None
    pool = _season_radar_pool(data_dir, year)
    if len(pool) < 2:
        return None
    return {
        "Qualifying pace": _radar_axis_score(stats["avg_quali"], pool["avg_quali"]),
        "Race pace": _radar_axis_score(
            stats["avg_finish_classified"], pool["avg_finish_classified"]),
        "Consistency": _radar_axis_score(
            stats["consistency_std"], pool["consistency_std"]),
    }
