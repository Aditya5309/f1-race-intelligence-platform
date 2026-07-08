"""
app/views/metadata.py

Display-metadata loaders for the dashboard (UI/UX redesign v2, Decision 024).

Reads Ergast-format CSVs from Settings().data_dir — the same knob the API's
display-name lookups already use (app/api.py::_load_name_lookups, the
Decision 016 precedent) — to enrich pages with Grand Prix names, circuits,
grids, championship standings, and historical outcome stats.

Scope note (amends Decision 016's HTTP-only rule for DISPLAY data only):
predictions still come exclusively from the HTTP API and nothing here
imports src/ or feeds the model. Post-race columns (wins, podiums, fastest
laps, standings) are read solely to DISPLAY historical outcomes of races
that already happened — the same class of serving-side enrichment as the
API's driver/constructor name lookups.

Every loader is cached (st.cache_data) and degrades gracefully: when the
CSVs are absent (fresh clone, CI, the offline smoke test — data/ is
gitignored) every function returns an empty DataFrame / empty dict / a
fallback string, and the pages render without metadata rather than crash.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config import Settings

_settings = Settings()

COUNTRY_FLAGS = {
    "Italy": "🇮🇹", "Monaco": "🇲🇨", "UK": "🇬🇧", "USA": "🇺🇸",
    "United States": "🇺🇸", "UAE": "🇦🇪", "Bahrain": "🇧🇭",
    "Saudi Arabia": "🇸🇦", "Australia": "🇦🇺", "Japan": "🇯🇵", "China": "🇨🇳",
    "Azerbaijan": "🇦🇿", "Spain": "🇪🇸", "Canada": "🇨🇦", "Austria": "🇦🇹",
    "France": "🇫🇷", "Hungary": "🇭🇺", "Belgium": "🇧🇪", "Netherlands": "🇳🇱",
    "Singapore": "🇸🇬", "Mexico": "🇲🇽", "Brazil": "🇧🇷", "Qatar": "🇶🇦",
    "Germany": "🇩🇪", "Russia": "🇷🇺", "Turkey": "🇹🇷", "Portugal": "🇵🇹",
    "Korea": "🇰🇷", "India": "🇮🇳", "Malaysia": "🇲🇾", "Argentina": "🇦🇷",
    "South Africa": "🇿🇦", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
}
_FALLBACK_FLAG = "🏁"


@st.cache_data(show_spinner=False)
def _read_csv(name: str) -> pd.DataFrame:
    """data_dir CSV -> DataFrame ('\\N' -> null); EMPTY frame when absent."""
    try:
        return pd.read_csv(_settings.data_dir / name, na_values=["\\N"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def race_catalog() -> pd.DataFrame:
    """races.csv joined to circuits.csv: raceId, year, round, name, date,
    circuit, country, location. Empty frame when races.csv is absent."""
    races = _read_csv("races.csv")
    if races.empty:
        return pd.DataFrame()
    cols = [c for c in ("raceId", "year", "round", "name", "date", "circuitId")
            if c in races.columns]
    catalog = races[cols].copy()
    circuits = _read_csv("circuits.csv")
    if not circuits.empty and "circuitId" in catalog.columns:
        keep = circuits.rename(columns={"name": "circuit"})
        keep = keep[[c for c in ("circuitId", "circuit", "country", "location")
                     if c in keep.columns]]
        catalog = catalog.merge(keep, on="circuitId", how="left")
    return catalog


def race_label(race_id: int, fallback_year: int | None = None,
               fallback_round: int | None = None) -> str:
    """'🇮🇹 Italian Grand Prix' from the catalog; a loud '⚠ Round N' fallback
    (not a blank/plain string) when the metadata CSVs are unavailable, so a
    data-loading failure is visible instead of looking like valid data."""
    catalog = race_catalog()
    if not catalog.empty:
        rows = catalog[catalog["raceId"] == race_id]
        if len(rows):
            row = rows.iloc[0]
            name = row.get("name")
            if pd.notna(name):
                flag = COUNTRY_FLAGS.get(str(row.get("country")), _FALLBACK_FLAG)
                return f"{flag} {name}"
    if fallback_round is not None:
        suffix = f" · {fallback_year}" if fallback_year is not None else ""
        return f"⚠ Round {fallback_round}{suffix}"
    return f"⚠ raceId {race_id}"


@st.cache_data(show_spinner=False)
def grid_and_quali(race_id: int) -> pd.DataFrame:
    """driverId -> grid (results.csv) + quali_position (qualifying.csv) for
    one race. Grid 0 is the Ergast pit-lane-start sentinel — callers display
    it as 'PL', never as a numeric grid slot."""
    results = _read_csv("results.csv")
    quali = _read_csv("qualifying.csv")
    out = pd.DataFrame()
    if not results.empty:
        out = results.loc[results["raceId"] == race_id, ["driverId", "grid"]].copy()
    if not quali.empty:
        q = quali.loc[quali["raceId"] == race_id, ["driverId", "position"]].rename(
            columns={"position": "quali_position"})
        out = out.merge(q, on="driverId", how="outer") if not out.empty else q
    return out


@st.cache_data(show_spinner=False)
def actual_result(race_id: int, driver_id: int) -> dict:
    """Actual outcome for one driver in one race (results.csv, joined to
    status.csv for human-readable status text) -- display only, used for
    Historical Replay / "how wrong" framing.

    Returns {} when results.csv is unavailable or this driver/race isn't in
    it. Otherwise: {"dnf": bool, "finish_position": int | None (the
    classified numeric position; None when dnf), "status": str (e.g.
    "Finished", "Accident", "Engine" -- results.csv `position` is null
    exactly for retirements/non-classified finishes, per Ergast convention;
    verified against status.csv)}."""
    results = _read_csv("results.csv")
    if results.empty:
        return {}
    r = results[(results["raceId"] == race_id) & (results["driverId"] == driver_id)]
    if r.empty:
        return {}
    row = r.iloc[0]
    dnf = pd.isna(row.get("position"))
    out: dict = {
        "dnf": bool(dnf),
        "finish_position": None if dnf else int(row["position"]),
    }
    status_id = row.get("statusId")
    if pd.notna(status_id):
        status = _read_csv("status.csv")
        if not status.empty:
            s = status[status["statusId"] == status_id]
            if len(s):
                out["status"] = str(s.iloc[0]["status"])
    return out


_DNF_LOOKBACK_YEARS = 5
_DNF_MIN_RACES = 3   # below this, a rate is noise, not signal


@st.cache_data(show_spinner=False)
def circuit_dnf_rate(race_id: int, lookback_years: int = _DNF_LOOKBACK_YEARS) -> dict:
    """DNF rate at this circuit over the `lookback_years` seasons before this
    race (exclusive). results.csv `position` is null exactly for
    retirements/non-classified finishes (Ergast convention: positionText
    'R'/'D'/etc.; verified against status.csv) — a genuine data-backed risk
    signal, unlike tire-degradation or safety-car frequency which this
    project's source data doesn't contain. {} when there isn't enough
    history (< _DNF_MIN_RACES prior races at this circuit) to be a signal
    rather than noise."""
    catalog = race_catalog()
    if catalog.empty:
        return {}
    rows = catalog[catalog["raceId"] == race_id]
    if not len(rows) or "circuitId" not in rows.columns:
        return {}
    row = rows.iloc[0]
    circuit_id, year = row.get("circuitId"), row.get("year")
    if pd.isna(circuit_id) or pd.isna(year):
        return {}
    year = int(year)
    scope = catalog[
        (catalog["circuitId"] == circuit_id)
        & (catalog["year"] < year)
        & (catalog["year"] >= year - lookback_years)
    ]
    n_races = scope["raceId"].nunique()
    if n_races < _DNF_MIN_RACES:
        return {}
    results = _read_csv("results.csv")
    if results.empty:
        return {}
    r = results.merge(scope[["raceId"]], on="raceId", how="inner")
    if r.empty:
        return {}
    return {
        "dnf_rate": float(r["position"].isna().mean()),
        "n_races": int(n_races),
        "years": f"{int(scope['year'].min())}–{int(scope['year'].max())}",
    }


def _parse_laptime(value: str) -> float | None:
    """'M:SS.sss' fastestLapTime string -> total seconds. None on anything
    that doesn't match (every fastestLapTime value in this dataset matches
    'M:SS.sss', checked directly; this guard is for genuinely malformed
    input, not an expected case)."""
    if not isinstance(value, str) or ":" not in value:
        return None
    minutes, _, seconds = value.partition(":")
    try:
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def circuit_catalog() -> pd.DataFrame:
    """circuitId, name, country, location + races_held count, for the
    Circuit Explorer picker. Empty frame when circuits.csv is absent."""
    circuits = _read_csv("circuits.csv")
    if circuits.empty:
        return pd.DataFrame()
    catalog = race_catalog()
    counts = (
        catalog.groupby("circuitId")["raceId"].nunique().rename("races_held")
        if not catalog.empty else pd.Series(dtype=int, name="races_held")
    )
    out = circuits[["circuitId", "name", "country", "location"]].merge(
        counts, on="circuitId", how="left")
    out["races_held"] = out["races_held"].fillna(0).astype(int)
    return out.sort_values("name")


_POSITION_CHANGE_MIN_RACES = 3   # below this, an average is noise, not signal


@st.cache_data(show_spinner=False)
def circuit_stats(circuit_id: int) -> dict:
    """All-time stats for one circuit -- races held, years active, DNF rate,
    average grid->finish position change, fastest recorded race lap, and
    win-count leaders. Every number here needs the caveats shown alongside
    it in the UI (see app/views/circuit_explorer.py): comparing across
    every year a circuit has run conflates car-performance evolution and
    (for position change) strategy/reliability/grid-penalty effects with
    genuine circuit character -- these are real, computed numbers, not
    invented "difficulty" scores, but they are proxies, not direct
    measurements. {} when there's no race history for this circuit."""
    catalog = race_catalog()
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
    results = _read_csv("results.csv")
    if results.empty:
        return stats
    r = results.merge(races_here[["raceId", "year"]], on="raceId", how="inner")
    if r.empty:
        return stats

    stats["dnf_rate"] = float(r["position"].isna().mean())

    valid = r[(r["grid"] > 0) & r["position"].notna()]
    if len(valid) >= _POSITION_CHANGE_MIN_RACES:
        stats["avg_position_change"] = float(
            (valid["grid"] - pd.to_numeric(valid["position"])).mean())

    laps = r.dropna(subset=["fastestLapTime"]).copy()
    if len(laps):
        laps["_seconds"] = laps["fastestLapTime"].map(_parse_laptime)
        laps = laps.dropna(subset=["_seconds"])
        if len(laps):
            row = laps.loc[laps["_seconds"].idxmin()]
            drivers = _read_csv("drivers.csv")
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
        drivers = _read_csv("drivers.csv")
        if not drivers.empty:
            names = drivers.assign(
                driver=drivers["forename"].astype(str) + " " + drivers["surname"].astype(str)
            )[["driverId", "driver"]]
            winners = winners.merge(names, on="driverId", how="left")
        else:
            winners["driver"] = winners["driverId"].astype(str)
        constructors = _read_csv("constructors.csv")
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


@st.cache_data(show_spinner=False)
def race_facts(race_id: int) -> dict:
    """Whatever display facts exist for a race: grand_prix, circuit, country,
    location, date, laps, pole_time, fastest_lap. Missing keys are omitted.
    (No track length or weather — not in the Ergast schema.)"""
    facts: dict = {}
    catalog = race_catalog()
    if not catalog.empty:
        rows = catalog[catalog["raceId"] == race_id]
        if len(rows):
            row = rows.iloc[0]
            for key, col in (("grand_prix", "name"), ("circuit", "circuit"),
                             ("country", "country"), ("location", "location"),
                             ("date", "date")):
                if col in rows.columns and pd.notna(row.get(col)):
                    facts[key] = str(row[col])
    results = _read_csv("results.csv")
    if not results.empty:
        r = results[results["raceId"] == race_id]
        if len(r) and r["laps"].notna().any():
            facts["laps"] = int(r["laps"].max())
        if "rank" in r.columns and "fastestLapTime" in r.columns:
            fl = r.loc[pd.to_numeric(r["rank"], errors="coerce") == 1,
                       "fastestLapTime"].dropna()
            if len(fl):
                facts["fastest_lap"] = str(fl.iloc[0])
    quali = _read_csv("qualifying.csv")
    if not quali.empty:
        q = quali[quali["raceId"] == race_id]
        pole = q[pd.to_numeric(q["position"], errors="coerce") == 1]
        if len(pole):
            for col in ("q3", "q2", "q1"):     # pole time = best of final session run
                if col in pole.columns and pd.notna(pole.iloc[0].get(col)):
                    facts["pole_time"] = str(pole.iloc[0][col])
                    break
    return facts


def _final_round_standings(standings_csv: str, year: int) -> pd.DataFrame:
    """Standings rows keyed to the LAST season race that has standings —
    Ergast standings rows apply after their raceId, so this is the season's
    final classification."""
    standings = _read_csv(standings_csv)
    catalog = race_catalog()
    if standings.empty or catalog.empty:
        return pd.DataFrame()
    season = catalog.loc[catalog["year"] == year, ["raceId", "round"]]
    merged = standings.merge(season, on="raceId", how="inner")
    if merged.empty:
        return pd.DataFrame()
    return merged[merged["round"] == merged["round"].max()].copy()


def _authoritative_points(standings_csv: str, id_col: str, entity_id: int,
                          year: int | None) -> float | None:
    """Points from the *_standings.csv tables (Ergast's own running total,
    includes sprint points) rather than summing results.csv's `points`
    column, which is main-Grand-Prix-only and silently undercounts any
    season with sprint races (checked against real 2024 data: summing
    results.csv gave McLaren 609 / Norris 344 vs. the authoritative 666 /
    374 -- sprint_results.csv, the missing points source, isn't part of
    this project's exported display data and isn't needed once this
    reads the standings tables instead). year=None sums each season's
    final standings row across every season the entity appears in (a
    fresh points table per season, so this is not a running total, and
    IS the correct way to combine seasons). None if the entity never
    appears in the standings at all."""
    if year is not None:
        rows = _final_round_standings(standings_csv, year)
        rows = rows[rows[id_col] == entity_id]
        return float(rows["points"].iloc[0]) if len(rows) else None

    standings = _read_csv(standings_csv)
    catalog = race_catalog()
    if standings.empty or catalog.empty:
        return None
    entity_years = catalog.merge(
        standings.loc[standings[id_col] == entity_id, ["raceId"]],
        on="raceId", how="inner")["year"].unique()
    if not len(entity_years):
        return None
    total = 0.0
    for yr in entity_years:
        rows = _final_round_standings(standings_csv, int(yr))
        rows = rows[rows[id_col] == entity_id]
        if len(rows):
            total += float(rows["points"].iloc[0])
    return total


@st.cache_data(show_spinner=False)
def season_driver_standings(year: int) -> pd.DataFrame:
    """Final driver standings of a season, with display names."""
    rows = _final_round_standings("driver_standings.csv", year)
    if rows.empty:
        return rows
    drivers = _read_csv("drivers.csv")
    if not drivers.empty:
        names = drivers.assign(
            driver=drivers["forename"].astype(str) + " " + drivers["surname"].astype(str)
        )[["driverId", "driver"]]
        rows = rows.merge(names, on="driverId", how="left")
    return rows.sort_values("position")


@st.cache_data(show_spinner=False)
def season_constructor_standings(year: int) -> pd.DataFrame:
    """Final constructor standings of a season, with display names."""
    rows = _final_round_standings("constructor_standings.csv", year)
    if rows.empty:
        return rows
    constructors = _read_csv("constructors.csv")
    if not constructors.empty:
        names = constructors.rename(columns={"name": "constructor"})
        rows = rows.merge(names[["constructorId", "constructor"]],
                          on="constructorId", how="left")
    return rows.sort_values("position")


@st.cache_data(show_spinner=False)
def driver_standings_progression(driver_id: int, year: int) -> pd.DataFrame:
    """Round-by-round championship points/position for one driver-season."""
    standings = _read_csv("driver_standings.csv")
    catalog = race_catalog()
    if standings.empty or catalog.empty:
        return pd.DataFrame()
    season = catalog.loc[catalog["year"] == year, ["raceId", "round"]]
    rows = standings.merge(season, on="raceId", how="inner")
    rows = rows[rows["driverId"] == driver_id]
    return rows.sort_values("round")[["round", "points", "position", "wins"]]


_CONSISTENCY_MIN_RACES = 3   # below this, a std dev is noise, not signal


@st.cache_data(show_spinner=False)
def driver_season_stats(driver_id: int, year: int | None = None) -> dict:
    """Historical outcome stats (display only): races, wins, podiums, poles,
    points, avg_quali, avg_finish, avg_finish_classified, consistency_std
    — one season, or career when year=None. `points` comes from
    driver_standings.csv (see _authoritative_points), not summed from
    results.csv, which undercounts sprint-race points.

    avg_finish uses positionOrder (existing behavior, includes DNFs ranked
    at the back — unchanged, other pages already rely on this meaning).
    avg_finish_classified/consistency_std are separate, NEW keys using the
    classified `position` column with DNFs dropped instead — conflating
    "crashed out" with "finished poorly" would make a consistency number
    dishonest. consistency_std needs >= _CONSISTENCY_MIN_RACES classified
    races to appear at all; small samples swing a std dev wildly."""
    results = _read_csv("results.csv")
    catalog = race_catalog()
    if results.empty or catalog.empty:
        return {}
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results.merge(scope[["raceId"]], on="raceId", how="inner")
    r = r[r["driverId"] == driver_id]
    if r.empty:
        return {}
    position = pd.to_numeric(r["position"], errors="coerce")
    points = _authoritative_points("driver_standings.csv", "driverId", driver_id, year)
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
        if len(classified) >= _CONSISTENCY_MIN_RACES:
            stats["consistency_std"] = float(classified.std())
    quali = _read_csv("qualifying.csv")
    if not quali.empty:
        q = quali.merge(scope[["raceId"]], on="raceId", how="inner")
        qpos = pd.to_numeric(q.loc[q["driverId"] == driver_id, "position"],
                             errors="coerce").dropna()
        if len(qpos):
            stats["poles"] = int((qpos == 1).sum())
            stats["avg_quali"] = float(qpos.mean())
    return stats


@st.cache_data(show_spinner=False)
def constructor_catalog() -> pd.DataFrame:
    """constructorId, name, nationality, for the Team page picker."""
    constructors = _read_csv("constructors.csv")
    if constructors.empty:
        return pd.DataFrame()
    return constructors[["constructorId", "name", "nationality"]].sort_values("name")


@st.cache_data(show_spinner=False)
def constructor_current_drivers(constructor_id: int, year: int) -> list[str]:
    """Driver display names who raced for this constructor in this season."""
    results = _read_csv("results.csv")
    catalog = race_catalog()
    drivers = _read_csv("drivers.csv")
    if results.empty or catalog.empty or drivers.empty:
        return []
    season = catalog.loc[catalog["year"] == year, ["raceId"]]
    r = results.merge(season, on="raceId", how="inner")
    r = r[r["constructorId"] == constructor_id]
    if r.empty:
        return []
    names = drivers[drivers["driverId"].isin(r["driverId"].unique())]
    return sorted((names["forename"].astype(str) + " " + names["surname"].astype(str)).tolist())


@st.cache_data(show_spinner=False)
def constructor_season_stats(constructor_id: int, year: int | None = None) -> dict:
    """Historical outcome stats (display only) for one constructor: races,
    wins, podiums, points, avg_quali -- one season, or career when
    year=None. Mirrors driver_season_stats() at constructor grain; "races"
    is distinct raceId count (a 2-car entry doesn't double-count). `points`
    comes from constructor_standings.csv (see _authoritative_points), not
    summed from results.csv, which undercounts sprint-race points."""
    results = _read_csv("results.csv")
    catalog = race_catalog()
    if results.empty or catalog.empty:
        return {}
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results.merge(scope[["raceId"]], on="raceId", how="inner")
    r = r[r["constructorId"] == constructor_id]
    if r.empty:
        return {}
    position = pd.to_numeric(r["position"], errors="coerce")
    points = _authoritative_points(
        "constructor_standings.csv", "constructorId", constructor_id, year)
    stats = {
        "races": int(r["raceId"].nunique()),
        "wins": int((position == 1).sum()),
        "podiums": int((position <= 3).sum()),
        "points": points if points is not None else float(r["points"].sum()),
    }
    quali = _read_csv("qualifying.csv")
    if not quali.empty:
        q = quali.merge(scope[["raceId"]], on="raceId", how="inner")
        qpos = pd.to_numeric(
            q.loc[q["constructorId"] == constructor_id, "position"],
            errors="coerce").dropna()
        if len(qpos):
            stats["avg_quali"] = float(qpos.mean())
    return stats


@st.cache_data(show_spinner=False)
def constructor_race_results(constructor_id: int, year: int | None = None) -> pd.DataFrame:
    """(year, round)-sorted per-race rows for one constructor: mean
    qualifying position across both cars and the team's best finish that
    round -- mirrors driver_race_results() at constructor grain, the Team
    page's qualifying-trend chart source. Best finish uses positionOrder's
    min (always populated, DNFs sort to the back), so one car's DNF doesn't
    hide the other car's genuine result that round."""
    results = _read_csv("results.csv")
    catalog = race_catalog()
    if results.empty or catalog.empty:
        return pd.DataFrame()
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results[results["constructorId"] == constructor_id].merge(
        scope[["raceId", "year", "round"]], on="raceId", how="inner")
    if r.empty:
        return pd.DataFrame()
    r["finish"] = pd.to_numeric(r["positionOrder"], errors="coerce")
    agg = r.groupby(["raceId", "year", "round"], as_index=False)["finish"].min()
    quali = _read_csv("qualifying.csv")
    if not quali.empty:
        q = quali[quali["constructorId"] == constructor_id]
        quali_by_race = (
            q.groupby("raceId")["position"].mean().rename("quali_position").reset_index()
        )
        agg = agg.merge(quali_by_race, on="raceId", how="left")
    return agg.sort_values(["year", "round"])


@st.cache_data(show_spinner=False)
def driver_race_results(driver_id: int, year: int | None = None) -> pd.DataFrame:
    """(year, round)-sorted per-race grid/quali/finish rows for one driver —
    the Driver Explorer trend-chart source."""
    results = _read_csv("results.csv")
    catalog = race_catalog()
    if results.empty or catalog.empty:
        return pd.DataFrame()
    scope = catalog if year is None else catalog[catalog["year"] == year]
    r = results[results["driverId"] == driver_id].merge(
        scope[["raceId", "year", "round"]], on="raceId", how="inner")
    if r.empty:
        return pd.DataFrame()
    quali = _read_csv("qualifying.csv")
    if not quali.empty:
        q = quali.loc[quali["driverId"] == driver_id, ["raceId", "position"]].rename(
            columns={"position": "quali_position"})
        r = r.merge(q, on="raceId", how="left")
    r["finish"] = pd.to_numeric(r["positionOrder"], errors="coerce")
    return r.sort_values(["year", "round"])
