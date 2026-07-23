"""
scripts/ingest_jolpica.py

Backfills recently-completed race weekends from jolpica-f1
(api.jolpi.ca/ergast/f1/) — the actively-maintained, Ergast-schema-compatible
successor to the deprecated Ergast API — into this project's
existing training-side data/ tree: results.csv, qualifying.csv,
driver_standings.csv, constructor_standings.csv, plus drivers.csv/
constructors.csv for any genuinely new entities (rookies, new teams).

Scope: only races ALREADY IN races.csv's schedule (matched by year+round)
that have zero results.csv rows yet — i.e. races that have happened since
the last ingestion. races.csv's calendar is populated far in advance and
rarely changes; this script does not fetch new schedule rows (a schedule
refresh, e.g. after a postponement, is a separate, rare concern — not
built here; use --dry-run to see what a run would touch first).

    python scripts/ingest_jolpica.py                       # all missing completed races
    python scripts/ingest_jolpica.py --dry-run              # fetch + report, no write
    python scripts/ingest_jolpica.py --year 2026 --round 7  # one race only

PHASE 2 EXTENSION (Decisions 049/050, `.ai/pre_race_materialization_design.md`
§7 Phase 2): in the default (no --year/--round) mode, this script ALSO
attempts to ingest `qualifying.csv` rows for the single upcoming race with
no result yet (Decision 050's horizon=1, resolved via
`src.features.upcoming.next_race` — the same function Phase 1 built,
reused rather than re-implemented here). This is the "grid-penalty proxy"
groundwork §1/§3 of the design call for: qualifying position is the only
pre-race starting-grid signal this project has ANY sanctioned source for
— jolpica/Ergast's schema exposes `grid` only inside the results
endpoint, which does not exist before the race is run, and no second,
independently-sourced grid-penalty feed is named anywhere in this
project's decisions (Decision 049 Refinement 1 makes jolpica the sole
authoritative live source). This extension does not invent one; it only
lands the qualifying-position proxy the design doc already accepts as the
documented gap. **Never fetches or writes results/standings for this
race** — those endpoints have nothing to return before the race is run,
and this project's build_master_dataset.py anchors its entire join on
results.csv, so a qualifying-only row for an as-yet-unrun race is
structurally inert to the historical pipeline (verified: the qualifying
join is LEFT-FROM-results, keyed on (raceId, driverId) — a race with no
results row can never be pulled in). Idempotent: skipped once that race's
qualifying already has rows on file, or once it's been fully ingested as
a completed race (checked against `ingested` from the SAME run, to avoid
double-fetching a race that turns out to have just finished).

PHASE 2 OPERATIONAL POLICY (documentation only — no behavior described
here is new; this section makes explicit what was previously only
implicit in the idempotency logic above):

  Expected cadence: this script (via `scripts/refresh_and_freeze.py`,
  which invokes it with no --year/--round, i.e. default mode) runs on
  `.github/workflows/retrain.yml`'s existing weekly schedule (Decision
  037) — the upcoming-qualifying step rides that SAME cadence with zero
  new scheduling. The design doc offered an "added lighter-weight
  cadence" as an alternative; it was not built — the weekly schedule was
  judged sufficient, consistent with Decision 049's own Future Work
  ("assumes the existing weekly cadence is sufficient at this project's
  current scale"). Separately, this script is a plain CLI entry point,
  not gated to CI: `python scripts/ingest_jolpica.py` (or the workflow's
  `workflow_dispatch` trigger) can be run manually at any time — "cadence"
  describes the AUTOMATED trigger frequency, not a restriction on when
  the script may run.

  `data_as_of`: NOT implemented by this script — that is Phase 7's
  provenance-block responsibility (Decision 049 Refinement 6), deliberately
  out of scope here. What this phase produces that a future `data_as_of`
  computation can draw on: `ingest_report/summary.json`'s `generated_at`
  (UTC, ISO-8601, the moment THIS RUN executed) and the `upcoming_qualifying`
  block recording whether that race's qualifying was fetched THIS run.
  Neither `qualifying.csv` nor any other data/ CSV carries a per-row
  ingestion timestamp — today, "as of" for the upcoming race's qualifying
  can only be approximated as "as of the most recent successful run,"
  never pinned to an individual row. Phase 7 will need to decide whether
  a per-row timestamp is worth adding; not decided here.

  Stale data handling: once a qualifying row exists for the upcoming race,
  this script does not periodically re-validate or refresh it. This is
  structurally justified, not merely an omission: Q1/Q2/Q3 times and
  qualifying classification are fixed once a session ends (Ergast/jolpica
  convention) — a post-session grid PENALTY changes the STARTING GRID, a
  separate, deliberately-unresolved data point (the "interim proxy" gap,
  §1/§3), and does NOT retroactively change the qualifying classification
  this row stores. The one edge case this doesn't cover — a rare
  post-session disqualification/correction to the qualifying result
  itself — is not handled specially, consistent with (not a new gap
  relative to) this project's existing posture: there is no live
  correction path for ANY already-ingested row, historical or otherwise
  (Decision 007's repair pipeline runs at build_interim time, not against
  freshly-ingested data/ rows).

  Manual refresh policy: on-demand TRIGGERING (running sooner than the
  weekly schedule) is already possible with no code change, as above.
  FORCING A RE-FETCH of an already-ingested upcoming race's qualifying
  is INTENTIONALLY NOT SUPPORTED at this stage — no --force/override flag
  exists; `resolve_upcoming_qualifying_target()`'s idempotency check is
  unconditional. This is deliberate: (1) not required by Phase 2's own
  scope, which only asks to land the qualifying-position proxy, not build
  a correction mechanism; (2) consistent with Decision 049's Future Work,
  which already defers a targeted/on-demand ETL trigger for the same
  "no concrete need yet" reason; (3) the data this would protect against
  staleness in is structurally stable once a session ends (see Stale data
  handling above), reducing the actual need. If a genuine correction is
  ever needed, the only path today is manual: delete the affected
  `qualifying.csv` row(s) for that raceId, then re-run this script — the
  idempotency check will re-fetch, since the row is no longer on file.
  This is a manual fallback, not a supported feature, and should not be
  treated as routine operation. Revisit only if a real, observed need
  arises.

ID reconciliation: this project's raceId/driverId/
constructorId/circuitId are CSV-dump surrogate keys, NOT what jolpica's
live API returns (it uses string refs like "max_verstappen", matching
Ergast's own driverRef/constructorRef/circuitRef convention). Existing
drivers.csv/constructors.csv already carry these *Ref columns — jolpica
rows reconcile onto them directly. A genuinely new entity (no existing
*Ref match) gets a new numeric ID minted off the current max in that CSV.
raceId has no natural ref at all; this script never mints one — it only
ever writes results/qualifying/standings for a raceId races.csv already
has (see Scope above).

KNOWN, PERMANENT LIMITATION — read before treating 2025+ ingested rows as
equivalent in detail to older rows: jolpica consolidates
granular DNF reasons ("Engine", "Collision", "Gearbox", etc. — 141 distinct
values in this project's historical status.csv) into a generic "Retired"
bucket from 2025 onward, and it also does not preserve Ergast's historical
"N" (Did Not Start) vs "R" (Retired) positionText distinction — non-
classified rows just use "R". This is NOT a bug in this script and NOT
something the normalization below can recover: the detail is gone at the
source. Ingested rows get positionText/status taken at face value from
jolpica (already Ergast-compatible single-letter codes) and, for
non-classified rows, statusId=31 ("Retired", a generic bucket that already
exists in this project's status.csv — not invented here). This does not
affect `finished`/`result_status` classification (src/data/cleaner.py's
numeric-positionText-first rule already handles classified-but-lapped rows
correctly regardless of jolpica's coarser `status` field) — only the
specific DNF *reason* loses granularity for 2025+ races.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from src.features.upcoming import UpcomingRace, next_race

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
#: Weekly-run inspection artifact (Part 2, retrain visibility gap): a
#: structured summary + the exact new rows this run fetched, written
#: regardless of --dry-run and regardless of whether the calling workflow
#: goes on to promote anything — this is "what did ingestion actually do",
#: entirely separate from any later step's outcome. NOT under artifacts/
#: (that tree is the committed-runtime-artifact convention) —
#: this is an ephemeral, gitignored, per-run diagnostic meant to be
#: `actions/upload-artifact`-ed from CI, not committed.
DEFAULT_REPORT_DIRNAME = "ingest_report"

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1-race-winner-prediction/1.0 (weekly retrain ingestion; github.com)"
REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5.0

#: jolpica's generic non-classified-finish bucket. Already exists in this
#: project's status.csv (statusId=31, "Retired") — not invented here. See
#: the module docstring's "KNOWN, PERMANENT LIMITATION" section.
GENERIC_RETIRED_STATUS_ID = 31
#: statusId used for every classified (numeric positionText) row. Loses the
#: historical "+N Laps" granularity Ergast's own statusId sometimes carried
#: for lapped-but-classified cars — inert for src/data/cleaner.py's own
#: classification (numeric positionText already wins there, unconditionally
#: on statusId), so this is a display-only simplification, not a modeling one.
GENERIC_FINISHED_STATUS_ID = 1

NA_TOKEN = "\\N"  # this project's CSVs use the Ergast MySQL-dump NULL convention


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get_with_retry(client: httpx.Client, url: str) -> dict:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(url)
            if response.status_code == 200:
                return response.json()
            last_exc = httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase}",
                request=response.request, response=response,
            )
        except httpx.HTTPError as exc:
            last_exc = exc
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"jolpica-f1 request failed after {MAX_RETRIES} attempts: {url}") from last_exc


def _fetch(client: httpx.Client, path: str) -> dict:
    data = _get_with_retry(client, f"{JOLPICA_BASE_URL}/{path}?format=json")
    time.sleep(REQUEST_DELAY_SECONDS)  # politeness — 200 req/hour unauthenticated limit
    return data["MRData"]


def fetch_results(client: httpx.Client, year: int, round_: int) -> list[dict] | None:
    mrdata = _fetch(client, f"{year}/{round_}/results/")
    races = mrdata["RaceTable"]["Races"]
    return races[0]["Results"] if races else None


def fetch_qualifying(client: httpx.Client, year: int, round_: int) -> list[dict] | None:
    mrdata = _fetch(client, f"{year}/{round_}/qualifying/")
    races = mrdata["RaceTable"]["Races"]
    return races[0]["QualifyingResults"] if races else None


def fetch_driver_standings(client: httpx.Client, year: int, round_: int) -> list[dict] | None:
    mrdata = _fetch(client, f"{year}/{round_}/driverStandings/")
    lists = mrdata["StandingsTable"]["StandingsLists"]
    return lists[0]["DriverStandings"] if lists else None


def fetch_constructor_standings(client: httpx.Client, year: int, round_: int) -> list[dict] | None:
    mrdata = _fetch(client, f"{year}/{round_}/constructorStandings/")
    lists = mrdata["StandingsTable"]["StandingsLists"]
    return lists[0]["ConstructorStandings"] if lists else None


# ---------------------------------------------------------------------------
# ID reconciliation (see module docstring)
# ---------------------------------------------------------------------------

class IdReconciler:
    """Resolves a jolpica *Ref slug to this project's numeric surrogate key,
    minting a new one (max existing + 1) for a genuinely new entity."""

    def __init__(self, df: pd.DataFrame, id_col: str, ref_col: str):
        self.df = df
        self.id_col = id_col
        self.ref_col = ref_col
        self._ref_to_id: dict[str, int] = dict(zip(df[ref_col], df[id_col], strict=True))
        self._next_id = int(df[id_col].max()) + 1
        self.new_rows: list[dict] = []

    def resolve(self, ref: str, new_row_fields: dict) -> int:
        if ref in self._ref_to_id:
            return self._ref_to_id[ref]
        new_id = self._next_id
        self._next_id += 1
        self._ref_to_id[ref] = new_id
        self.new_rows.append({self.id_col: new_id, self.ref_col: ref, **new_row_fields})
        return new_id


def _driver_new_row(d: dict) -> dict:
    return {
        "number": d.get("permanentNumber", NA_TOKEN),
        "code": d.get("code", NA_TOKEN),
        "forename": d["givenName"],
        "surname": d["familyName"],
        "dob": d.get("dateOfBirth", NA_TOKEN),
        "nationality": d.get("nationality", NA_TOKEN),
        "url": d.get("url", NA_TOKEN),
    }


def _constructor_new_row(c: dict) -> dict:
    return {
        "name": c["name"],
        "nationality": c.get("nationality", NA_TOKEN),
        "url": c.get("url", NA_TOKEN),
    }


# ---------------------------------------------------------------------------
# positionText/status normalization (see the module docstring's KNOWN LIMITATION note)
# ---------------------------------------------------------------------------

def normalize_finish(position_text: str) -> tuple[str | float, str, int]:
    """(position, positionText, statusId) for one jolpica result row.

    jolpica already uses Ergast-compatible single-letter non-finish codes
    (verified directly against the live API: "R" for every non-classified
    row) — no code TRANSLATION happens here, only picking a representative
    statusId, since jolpica's own `status` field ("Finished"/"Lapped"/
    "Retired"/...) is coarser than this project's historical statusId
    taxonomy and must not be blindly re-mapped onto it (see the module
    docstring's permanent-limitation note).
    """
    if position_text.isdigit():
        return position_text, position_text, GENERIC_FINISHED_STATUS_ID
    return NA_TOKEN, position_text, GENERIC_RETIRED_STATUS_ID


# ---------------------------------------------------------------------------
# Per-race ingestion
# ---------------------------------------------------------------------------

def missing_completed_races(races: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """races.csv rows with zero results.csv rows — races that have happened
    since the last ingestion (or simply haven't been backfilled yet)."""
    has_results = set(results["raceId"].unique())
    return races[~races["raceId"].isin(has_results)].sort_values(["year", "round"])


def resolve_upcoming_qualifying_target(
    races: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    *,
    already_ingested_race_ids: set[int] | None = None,
) -> UpcomingRace | None:
    """The single upcoming race (Decision 050 horizon=1, via
    `src.features.upcoming.next_race`) whose qualifying is worth attempting
    to fetch this run — or `None` if there is nothing to do.

    `None` cases, all equally "nothing new to fetch":
      - there is no upcoming race at all (every scheduled race has results);
      - that race's qualifying is already on file (idempotent — a prior run
        already landed it);
      - that race was ALSO just fully ingested as a completed race earlier
        in THIS SAME run (`already_ingested_race_ids`, populated from this
        run's own `ingested` list) — its qualifying was already fetched
        there; re-fetching here would duplicate rows.
    """
    race = next_race(races, results)
    if race is None:
        return None
    if already_ingested_race_ids and race.race_id in already_ingested_race_ids:
        return None
    if race.race_id in set(qualifying["raceId"].unique()):
        return None
    return race


def build_results_rows(
    raw_results: list[dict], race_id: int, next_result_id: int,
    drivers: IdReconciler, constructors: IdReconciler,
) -> list[dict]:
    rows = []
    for order, r in enumerate(raw_results, start=1):  # jolpica returns finish order already
        driver_id = drivers.resolve(r["Driver"]["driverId"], _driver_new_row(r["Driver"]))
        constructor_id = constructors.resolve(
            r["Constructor"]["constructorId"], _constructor_new_row(r["Constructor"])
        )
        position, position_text, status_id = normalize_finish(r["positionText"])
        time_block = r.get("Time", {})
        fastest = r.get("FastestLap", {})
        fastest_time = fastest.get("Time", {})
        rows.append({
            "resultId": next_result_id + len(rows),
            "raceId": race_id,
            "driverId": driver_id,
            "constructorId": constructor_id,
            "number": r.get("number", NA_TOKEN),
            "grid": r.get("grid", NA_TOKEN),
            "position": position,
            "positionText": position_text,
            "positionOrder": order,
            "points": r.get("points", "0"),
            "laps": r.get("laps", NA_TOKEN),
            "time": time_block.get("time", NA_TOKEN),
            "milliseconds": time_block.get("millis", NA_TOKEN),
            "fastestLap": fastest.get("lap", NA_TOKEN),
            "rank": fastest.get("rank", NA_TOKEN),
            "fastestLapTime": fastest_time.get("time", NA_TOKEN),
            "fastestLapSpeed": NA_TOKEN,  # jolpica does not provide AverageSpeed
            "statusId": status_id,
        })
    return rows


def build_qualifying_rows(
    raw_qualifying: list[dict], race_id: int, next_qualify_id: int,
    drivers: IdReconciler, constructors: IdReconciler,
) -> list[dict]:
    rows = []
    for q in raw_qualifying:
        driver_id = drivers.resolve(q["Driver"]["driverId"], _driver_new_row(q["Driver"]))
        constructor_id = constructors.resolve(
            q["Constructor"]["constructorId"], _constructor_new_row(q["Constructor"])
        )
        rows.append({
            "qualifyId": next_qualify_id + len(rows),
            "raceId": race_id,
            "driverId": driver_id,
            "constructorId": constructor_id,
            "number": q.get("number", NA_TOKEN),
            "position": q["position"],
            "q1": q.get("Q1", NA_TOKEN),
            "q2": q.get("Q2", NA_TOKEN),
            "q3": q.get("Q3", NA_TOKEN),
        })
    return rows


def build_driver_standings_rows(
    raw_standings: list[dict], race_id: int, next_id: int, drivers: IdReconciler,
) -> list[dict]:
    rows = []
    for s in raw_standings:
        driver_id = drivers.resolve(s["Driver"]["driverId"], _driver_new_row(s["Driver"]))
        rows.append({
            "driverStandingsId": next_id + len(rows),
            "raceId": race_id,
            "driverId": driver_id,
            "points": s["points"],
            "position": s["position"],
            "positionText": s["positionText"],
            "wins": s["wins"],
        })
    return rows


def build_constructor_standings_rows(
    raw_standings: list[dict], race_id: int, next_id: int, constructors: IdReconciler,
) -> list[dict]:
    rows = []
    for s in raw_standings:
        constructor_id = constructors.resolve(
            s["Constructor"]["constructorId"], _constructor_new_row(s["Constructor"])
        )
        rows.append({
            "constructorStandingsId": next_id + len(rows),
            "raceId": race_id,
            "constructorId": constructor_id,
            "points": s["points"],
            "position": s["position"],
            "positionText": s["positionText"],
            "wins": s["wins"],
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _append(csv_path: Path, new_rows: list[dict], columns: tuple[str, ...]) -> None:
    if not new_rows:
        return
    frame = pd.DataFrame(new_rows)[list(columns)]
    frame.to_csv(csv_path, mode="a", header=False, index=False, na_rep=NA_TOKEN)


def write_ingest_report(
    report_dir: Path,
    *,
    dry_run: bool,
    ingested: list[dict],
    skipped: list[dict],
    new_drivers: list[dict],
    new_constructors: list[dict],
    new_results_rows: list[dict],
    new_qualifying_rows: list[dict],
    new_driver_standings_rows: list[dict],
    new_constructor_standings_rows: list[dict],
    upcoming_qualifying: dict | None = None,
) -> Path:
    """Write summary.json + one CSV per endpoint of EXACTLY this run's new
    rows (never the accumulated data/ tree) to report_dir, creating it if
    needed. Always called — dry-run or not, whatever a later pipeline step
    does — so this answers "what did ingestion actually fetch this run",
    inspectable on its own via `gh run download` (no trust-the-log-line
    required). A CSV is omitted (not written empty) when its list has zero
    rows; summary.json's counts already say so.

    `upcoming_qualifying` (Phase 2, Decision 049/050): optional
    `{"year", "round", "name", "raceId", "n_qualifying_rows"}` describing
    the single upcoming race this run attempted qualifying-only ingestion
    for, or `None` when nothing was attempted (explicit --year/--round
    mode, no upcoming race, or its qualifying was already on file). `None`
    by default so existing callers/tests are unaffected.

    Returns report_dir.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "ingested_races": ingested,
        "skipped_races": skipped,
        "upcoming_qualifying": upcoming_qualifying,
        "totals": {
            "races_ingested": len(ingested),
            "races_skipped": len(skipped),
            "results_rows": len(new_results_rows),
            "qualifying_rows": len(new_qualifying_rows),
            "driver_standings_rows": len(new_driver_standings_rows),
            "constructor_standings_rows": len(new_constructor_standings_rows),
            "new_drivers": len(new_drivers),
            "new_constructors": len(new_constructors),
        },
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for filename, rows in (
        ("new_results.csv", new_results_rows),
        ("new_qualifying.csv", new_qualifying_rows),
        ("new_driver_standings.csv", new_driver_standings_rows),
        ("new_constructor_standings.csv", new_constructor_standings_rows),
        ("new_drivers.csv", new_drivers),
        ("new_constructors.csv", new_constructors),
    ):
        if rows:
            pd.DataFrame(rows).to_csv(report_dir / filename, index=False, na_rep=NA_TOKEN)

    return report_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--year", type=int, default=None,
                        help="Ingest one specific race only (requires --round).")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report what would be ingested; write nothing.")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Where to write this run's inspection report "
                             "(summary.json + new-rows CSVs) — see "
                             "write_ingest_report(). Default: "
                             f"<--data-dir's parent>/{DEFAULT_REPORT_DIRNAME} "
                             "(so hermetic runs with a tmp --data-dir get a "
                             "hermetic report location for free). Always "
                             "written, --dry-run or not.")
    args = parser.parse_args(argv)
    if (args.year is None) != (args.round is None):
        parser.error("--year and --round must be given together.")

    data_dir = args.data_dir
    report_dir = args.report_dir or (data_dir.parent / DEFAULT_REPORT_DIRNAME)
    races = pd.read_csv(data_dir / "races.csv", na_values=[NA_TOKEN])
    results = pd.read_csv(data_dir / "results.csv", na_values=[NA_TOKEN])
    qualifying = pd.read_csv(data_dir / "qualifying.csv", na_values=[NA_TOKEN])
    driver_standings = pd.read_csv(data_dir / "driver_standings.csv", na_values=[NA_TOKEN])
    constructor_standings = pd.read_csv(data_dir / "constructor_standings.csv", na_values=[NA_TOKEN])
    drivers_df = pd.read_csv(data_dir / "drivers.csv", na_values=[NA_TOKEN])
    constructors_df = pd.read_csv(data_dir / "constructors.csv", na_values=[NA_TOKEN])

    if args.year is not None:
        targets = races[(races["year"] == args.year) & (races["round"] == args.round)]
        if targets.empty:
            print(f"ERROR: {args.year} round {args.round} not found in {data_dir / 'races.csv'} "
                  "— schedule rows are not created by this script.", file=sys.stderr)
            return 1
    else:
        targets = missing_completed_races(races, results)

    if targets.empty:
        print("Nothing to backfill — every scheduled race already has results.")
        # Falls through rather than returning early: the Phase 2 upcoming-
        # qualifying step below still needs to run in the default (no
        # --year/--round) mode even when there's nothing to backfill — the
        # common case, since races happen far less often than this script runs.

    drivers = IdReconciler(drivers_df, "driverId", "driverRef")
    constructors = IdReconciler(constructors_df, "constructorId", "constructorRef")
    next_result_id = int(results["resultId"].max()) + 1
    next_qualify_id = int(qualifying["qualifyId"].max()) + 1
    next_driver_standings_id = int(driver_standings["driverStandingsId"].max()) + 1
    next_constructor_standings_id = int(constructor_standings["constructorStandingsId"].max()) + 1

    all_results_rows, all_qualifying_rows = [], []
    all_driver_standings_rows, all_constructor_standings_rows = [], []
    ingested, skipped = [], []

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for race in targets.itertuples():
            year, round_, race_id = race.year, race.round, race.raceId
            raw_results = fetch_results(client, year, round_)
            if raw_results is None:
                skipped.append({"year": year, "round": round_, "name": race.name})
                continue

            result_rows = build_results_rows(raw_results, race_id, next_result_id, drivers, constructors)
            next_result_id += len(result_rows)
            all_results_rows += result_rows

            raw_qualifying = fetch_qualifying(client, year, round_) or []
            qualifying_rows = build_qualifying_rows(
                raw_qualifying, race_id, next_qualify_id, drivers, constructors
            )
            next_qualify_id += len(qualifying_rows)
            all_qualifying_rows += qualifying_rows

            raw_driver_standings = fetch_driver_standings(client, year, round_) or []
            ds_rows = build_driver_standings_rows(
                raw_driver_standings, race_id, next_driver_standings_id, drivers
            )
            next_driver_standings_id += len(ds_rows)
            all_driver_standings_rows += ds_rows

            raw_constructor_standings = fetch_constructor_standings(client, year, round_) or []
            cs_rows = build_constructor_standings_rows(
                raw_constructor_standings, race_id, next_constructor_standings_id, constructors
            )
            next_constructor_standings_id += len(cs_rows)
            all_constructor_standings_rows += cs_rows

            ingested.append({
                "year": year, "round": round_, "name": race.name, "raceId": race_id,
                "n_results": len(result_rows), "n_qualifying": len(qualifying_rows),
                "n_driver_standings": len(ds_rows), "n_constructor_standings": len(cs_rows),
            })
            print(f"Fetched {year} round {round_} ({race.name}): "
                  f"{len(result_rows)} results, {len(qualifying_rows)} qualifying, "
                  f"{len(ds_rows)} driver standings, {len(cs_rows)} constructor standings")

        # --- Phase 2 (Decisions 049/050): upcoming-race qualifying only.
        # Default mode only — an explicit --year/--round backfill request
        # targets one specific race and should not also reach for a
        # different, unrelated "next race". Never touches results/standings
        # for this race (see module docstring) — qualifying.csv only.
        upcoming_target = None
        upcoming_qualifying_rows: list[dict] = []
        if args.year is None:
            upcoming_target = resolve_upcoming_qualifying_target(
                races, results, qualifying,
                already_ingested_race_ids={r["raceId"] for r in ingested},
            )
            if upcoming_target is not None:
                raw_upcoming_qualifying = fetch_qualifying(
                    client, upcoming_target.year, upcoming_target.round
                ) or []
                if raw_upcoming_qualifying:
                    upcoming_qualifying_rows = build_qualifying_rows(
                        raw_upcoming_qualifying, upcoming_target.race_id,
                        next_qualify_id, drivers, constructors,
                    )
                    next_qualify_id += len(upcoming_qualifying_rows)
                    all_qualifying_rows += upcoming_qualifying_rows
                    print(f"Upcoming race qualifying: {upcoming_target.year} round "
                          f"{upcoming_target.round} ({upcoming_target.name}) — "
                          f"{len(upcoming_qualifying_rows)} rows")
                else:
                    print(f"Upcoming race ({upcoming_target.year} round "
                          f"{upcoming_target.round}, {upcoming_target.name}): "
                          "qualifying not posted yet — nothing to ingest.")

    print(f"\n{len(ingested)} race(s) ingested, {len(skipped)} skipped (no data yet at jolpica-f1):")
    for s in skipped:
        print(f"  SKIP {s['year']} round {s['round']} ({s['name']}) — not run yet")
    print(f"New drivers: {len(drivers.new_rows)}, new constructors: {len(constructors.new_rows)}")

    upcoming_qualifying_summary = None
    if upcoming_target is not None:
        upcoming_qualifying_summary = {
            "year": upcoming_target.year, "round": upcoming_target.round,
            "name": upcoming_target.name, "raceId": upcoming_target.race_id,
            "n_qualifying_rows": len(upcoming_qualifying_rows),
        }

    report_dir = write_ingest_report(
        report_dir, dry_run=args.dry_run, ingested=ingested, skipped=skipped,
        new_drivers=drivers.new_rows, new_constructors=constructors.new_rows,
        new_results_rows=all_results_rows, new_qualifying_rows=all_qualifying_rows,
        new_driver_standings_rows=all_driver_standings_rows,
        new_constructor_standings_rows=all_constructor_standings_rows,
        upcoming_qualifying=upcoming_qualifying_summary,
    )
    print(f"Ingestion report: {report_dir}")

    if args.dry_run:
        print("\n--dry-run: nothing written to data/.")
        return 0

    _append(data_dir / "drivers.csv", drivers.new_rows,
            tuple(drivers_df.columns))
    _append(data_dir / "constructors.csv", constructors.new_rows,
            tuple(constructors_df.columns))
    _append(data_dir / "results.csv", all_results_rows, tuple(results.columns))
    _append(data_dir / "qualifying.csv", all_qualifying_rows, tuple(qualifying.columns))
    _append(data_dir / "driver_standings.csv", all_driver_standings_rows,
            tuple(driver_standings.columns))
    _append(data_dir / "constructor_standings.csv", all_constructor_standings_rows,
            tuple(constructor_standings.columns))
    print(f"\nWrote {len(all_results_rows)} results, {len(all_qualifying_rows)} qualifying, "
          f"{len(all_driver_standings_rows)} driver standings, "
          f"{len(all_constructor_standings_rows)} constructor standings rows to {data_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
