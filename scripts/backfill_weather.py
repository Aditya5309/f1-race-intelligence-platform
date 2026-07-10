"""
scripts/backfill_weather.py

One-time backfill of historical race-weekend weather from Open-Meteo's
Historical Weather API (archive-api.open-meteo.com/v1/archive) — training-
side enrichment data for src/features/weather.py, written to
data/interim/race_weather.csv (gitignored, like the rest of data/ — this is
consumed only when rebuilding data/processed/features.parquet, never a
runtime/serving dependency).

IMPORTANT CAVEAT — READ BEFORE REUSING THIS DATA ELSEWHERE: `race_precip_mm`
and `race_temp_c` are ACTUAL, POST-HOC observed weather (Open-Meteo's
reanalysis archive), not a forecast. That is fine for this project's current
scope — historical prediction/evaluation of races that have already
happened, where the real weather is exactly what a hypothetical
race-morning observer would have known. It is NOT usable as-is for a future
live "predict an upcoming race" feature: at prediction time before a race
that hasn't happened yet, this exact data would not exist. A live feature
would need a forecast-based substitute (a genuinely different data source,
with its own uncertainty/error characteristics) — do not silently wire this
same column in for that use case later.

For each of the 305 races in the 2010-2024 modeling window:
- `race_precip_mm` / `race_temp_c`: summed precipitation / mean temperature
  over a SESSION_WINDOW_HOURS-hour window starting at the race's official
  start (races.csv `date`+`time`, confirmed UTC — cross-checked against the
  British GP's well-documented local start times). Full coverage, all 305
  races.
- `quali_precip_mm` / `conditions_changed`: same mechanism against
  `quali_date`+`quali_time`, but ONLY where both are present — 68 of 305
  races (2022-2024; Ergast doesn't record qualifying session start times
  before then). Left null everywhere else — NOT backfilled with a same-day
  approximation, which would misrepresent precision this dataset doesn't
  have. `conditions_changed` = whether the race session crossed the
  WET_THRESHOLD_MM wet/dry boundary in the opposite direction from
  qualifying (dry quali -> wet race, or vice versa).

SESSION_WINDOW_HOURS = 2: a Grand Prix is capped at 2 hours (rarely extended
to 3 under a red flag); a qualifying session (Q1+Q2+Q3 plus breaks) runs
close to but under an hour. Open-Meteo only offers hourly bins and Ergast
doesn't record session duration, so a 2-hour window from the recorded start
comfortably covers either session without per-race duration data.

WET_THRESHOLD_MM = 0.2: a commonly used meteorological cutoff for
"measurable precipitation" vs. trace/dew noise.

For races with both a race and a qualifying session date, ONE Open-Meteo
request spans [quali_date, race_date] and both sessions are sliced out of
its hourly response — halving the request count vs. querying each session
separately (237 requests instead of 305+68=373).

Paced a few seconds between requests (politeness, same discipline as
scripts/backfill_circuit_layouts.py's treatment of the Overpass API) with
retries for transient failures.

    python scripts/backfill_weather.py                # full run
    python scripts/backfill_weather.py --dry-run       # fetch + aggregate, no write
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RACES_CSV = _PROJECT_ROOT / "data" / "races.csv"
DEFAULT_CIRCUITS_CSV = _PROJECT_ROOT / "data" / "circuits.csv"
DEFAULT_DEST = _PROJECT_ROOT / "data" / "interim" / "race_weather.csv"

MODELING_WINDOW = (2010, 2024)
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "f1-race-winner-prediction/1.0 (one-time weather backfill; github.com)"
SESSION_WINDOW_HOURS = 2
WET_THRESHOLD_MM = 0.2
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 8
RETRY_BACKOFF_SECONDS = 10.0

WEATHER_COLUMNS: tuple[str, ...] = (
    "raceId", "race_precip_mm", "race_temp_c",
    "quali_precip_mm", "conditions_changed",
)


def modeling_window_races(
    races_csv: Path = DEFAULT_RACES_CSV, circuits_csv: Path = DEFAULT_CIRCUITS_CSV,
) -> pd.DataFrame:
    """Races in MODELING_WINDOW with circuit lat/lng and session date/times."""
    races = pd.read_csv(races_csv, na_values=["\\N"])
    circuits = pd.read_csv(circuits_csv, na_values=["\\N"])
    lo, hi = MODELING_WINDOW
    window = races[races["year"].between(lo, hi)].copy()
    out = window.merge(
        circuits[["circuitId", "lat", "lng"]], on="circuitId", how="left",
    )
    return out.sort_values(["year", "round"]).reset_index(drop=True)


def _get_with_retry(client: httpx.Client, params: dict) -> dict:
    """GET the Open-Meteo archive API, retrying transient errors."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(OPEN_METEO_URL, params=params)
            if response.status_code == 200:
                return response.json()
            last_exc = httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase}",
                request=response.request, response=response,
            )
        except httpx.HTTPError as exc:
            last_exc = exc
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed after {MAX_RETRIES} attempts") from last_exc


def _fetch_hourly(
    client: httpx.Client, lat: float, lng: float, start_date: date, end_date: date,
) -> dict[str, tuple[float, float]]:
    """One archive-API call across [start_date, end_date]; {iso_hour: (precip_mm, temp_c)}."""
    params = {
        "latitude": lat, "longitude": lng,
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "hourly": "precipitation,temperature_2m",
        "timezone": "UTC",
    }
    data = _get_with_retry(client, params)
    hourly = data["hourly"]
    return dict(zip(hourly["time"], zip(hourly["precipitation"], hourly["temperature_2m"]), strict=True))


def _session_aggregate(
    hourly_by_hour: dict[str, tuple[float, float]], session_date: str, session_time: str,
) -> tuple[float, float] | tuple[None, None]:
    """Sum precip / mean temp over SESSION_WINDOW_HOURS from session start.

    Missing hourly bins (shouldn't happen within Open-Meteo's archive
    coverage, but the API is a third party) are skipped rather than raising;
    returns (None, None) only if NONE of the window's hours were found.
    """
    start = datetime.fromisoformat(f"{session_date}T{session_time}")
    keys = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(SESSION_WINDOW_HOURS)]
    precip = [hourly_by_hour[k][0] for k in keys if k in hourly_by_hour]
    temp = [hourly_by_hour[k][1] for k in keys if k in hourly_by_hour]
    if not precip:
        return None, None
    return sum(precip), sum(temp) / len(temp)


def _is_wet(precip_mm: float | None) -> bool | None:
    return None if precip_mm is None else precip_mm > WET_THRESHOLD_MM


def backfill_weather(
    races_csv: Path = DEFAULT_RACES_CSV,
    circuits_csv: Path = DEFAULT_CIRCUITS_CSV,
    dest: Path = DEFAULT_DEST,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the backfill. Returns the built DataFrame (also written to `dest`
    unless dry_run)."""
    races = modeling_window_races(races_csv, circuits_csv)
    records: list[dict] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS,
    ) as client:
        for row in races.itertuples():
            has_quali = pd.notna(row.quali_date) and pd.notna(row.quali_time)
            start_date = (
                datetime.fromisoformat(row.quali_date).date() if has_quali
                else datetime.fromisoformat(row.date).date()
            )
            end_date = datetime.fromisoformat(row.date).date()

            hourly = _fetch_hourly(client, row.lat, row.lng, start_date, end_date)
            time.sleep(REQUEST_DELAY_SECONDS)

            race_precip, race_temp = _session_aggregate(hourly, row.date, row.time)
            quali_precip = None
            conditions_changed = None
            if has_quali:
                quali_precip, _ = _session_aggregate(hourly, row.quali_date, row.quali_time)
                if quali_precip is not None and race_precip is not None:
                    conditions_changed = _is_wet(quali_precip) != _is_wet(race_precip)

            records.append({
                "raceId": row.raceId,
                "race_precip_mm": race_precip,
                "race_temp_c": race_temp,
                "quali_precip_mm": quali_precip,
                "conditions_changed": conditions_changed,
            })
            print(f"{row.raceId} {row.year} round {row.round}: "
                  f"race_precip={race_precip}, quali_precip={quali_precip}")

    out = pd.DataFrame.from_records(records)[list(WEATHER_COLUMNS)]
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races-csv", type=Path, default=DEFAULT_RACES_CSV)
    parser.add_argument("--circuits-csv", type=Path, default=DEFAULT_CIRCUITS_CSV)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and aggregate but do not write the CSV.")
    args = parser.parse_args(argv)

    if not args.races_csv.exists() or not args.circuits_csv.exists():
        print(f"ERROR: {args.races_csv} / {args.circuits_csv} not found.", file=sys.stderr)
        return 1

    out = backfill_weather(
        races_csv=args.races_csv, circuits_csv=args.circuits_csv,
        dest=args.dest, dry_run=args.dry_run,
    )
    n_quali = out["quali_precip_mm"].notna().sum()
    print(f"\n{len(out)} races processed, {n_quali} with qualifying-session weather.")
    if not args.dry_run:
        print(f"Wrote {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
