# Master Modeling Dataset — Design Document

_Status: ACCEPTED AND IMPLEMENTED (Decision 009 as amended by Decisions 010/011;
audited 2026-07-04). Historical planning detail is retained below._
_Author: AI agent, Phase 3 planning session._
_Depends on: `context/decisions.md` (Decisions 003, 006, 007, 008), `reports/eda_summary.md`._

---

## 1. Objective

This document defined the integration and feature contracts before implementation.
The as-built system separates the join-only master table
(`src/integration/build_master_dataset.py`, orchestrated by
`src/pipelines/build_dataset.py`) from modular feature construction in `src/features/`.
The master table intentionally contains historical outcome columns; only the derived
feature matrix is restricted to pre-race features. See `context/architecture.md` for
the authoritative as-built flow.

---

## 2. Dataset Grain

**One row = one (raceId, driverId) pair** — a single driver's entry in a single race.

Rationale:
- Matches Decision 003 (binary classification per driver, target = winner).
- Matches the natural grain of `results.csv` after the dedup already performed in
  `build_interim.py` (Decision 007: 91 duplicate rows dropped so `(raceId, driverId)`
  is unique).
- Rejected alternatives:
  - **One row per race** (multiclass, 20 driver-columns) — doesn't generalize across
    seasons with different grid sizes, complicates the join graph for no benefit.
  - **One row per lap** — wrong grain entirely; this project predicts pre-race
    outcomes, not in-race dynamics.

Row count at this grain, restricted to the Decision-008 modeling window
(2010–2024 inclusive): **305 races**, and (from `current_status.md`) 5,077 + 880 +
479 ≈ **6,436 driver-race rows** across train/val/test.

Every row must have a non-null `raceId` and `driverId` (already enforced by
`cleaner._validate`). A row exists for every driver who was **entered** in a race,
including non-finishers and non-starters — dropping them would bias the feature
distributions toward finishers only (this rationale already lives in
`cleaner.py`'s module docstring and applies identically here).

---

## 3. Required Tables

| Table | Role | Included? |
|---|---|---|
| `results.csv` | Grain source (raceId+driverId), grid position, target (via position), outcome history for rolling stats | **Yes** — core |
| `races.csv` | Race dimension: year, round, circuitId, date, sprint flag | **Yes** — core |
| `qualifying.csv` | Pre-race qualifying position + Q1/Q2/Q3 times | **Yes** — core |
| `driver_standings.csv` | Championship standing, lagged 1 round | **Yes** — core |
| `constructor_standings.csv` | Championship standing, lagged 1 round | **Yes** — core |
| `drivers.csv` | Driver dimension: dob (age), nationality | **Yes** — core |
| `constructors.csv` | Constructor dimension: nationality | **Yes** — core |
| `circuits.csv` | Circuit dimension: country (home-circuit), lat/lng/alt | **Yes** — core |
| `status.csv` | statusId → status description | **Indirect** — already fully consumed into `result_status` by `cleaner.py`; not joined again |
| `sprint_results.csv` | Sprint quali/race result for the same weekend | **Optional enrichment** — see §6.4 |
| `constructor_results.csv` | Constructor points per race | **No** — fully derivable from `results.csv` via groupby; adds a join with no new information |
| `lap_times.csv` | Lap-by-lap timing | **No** — pure in-race telemetry, see §6.1 |
| `pit_stops.csv` | Pit stop timing | **No** — pure in-race telemetry, see §6.1 |
| `seasons.csv` | year → Wikipedia URL | **No** — no modeling signal |

---

## 4. Join Keys

```
races.circuitId        → circuits.circuitId                    (N:1)
results.raceId          → races.raceId                          (N:1)
results.driverId        → drivers.driverId                      (N:1)
results.constructorId   → constructors.constructorId             (N:1)
qualifying.(raceId, driverId)          → results.(raceId, driverId)   (1:1*, left join)
driver_standings.(raceId, driverId)    → results.(raceId, driverId)   (1:1, LAG by 1 round — see §6.2)
constructor_standings.(raceId, constructorId) → results.(raceId, constructorId) (N:1, LAG by 1 round)
sprint_results.(raceId, driverId)      → results.(raceId, driverId)   (1:1*, left join, sparse)
```

`*` = not guaranteed 1:1 in practice: a driver can appear in `qualifying.csv` and be
absent from `results.csv` if they failed to start (or vice versa in rare data-entry
cases). All joins to the base table must be **left joins from `results.csv`**, never
inner joins — an inner join would silently drop DNS/DNQ rows and reintroduce the
finisher-only bias `cleaner.py` was written to avoid.

**Base table:** `results.csv` (post `clean_results()` + `build_interim.py` repairs).
Every other table is joined onto it, keyed by `raceId` (+ `driverId` and/or
`constructorId` as appropriate). `races.csv` is joined first to get `year`/`round`,
since almost every downstream join (standings lag, rolling windows, circuit history)
needs temporal ordering by `(year, round)`, not `raceId` (raceId is not guaranteed
to sort chronologically across eras — this should be verified in code, not assumed).

---

## 5. Required Columns (Draft Schema)

### 5.1 Identifiers (kept for joins / grouping, not fed to the model as raw features except where noted)

| Column | Source | Notes |
|---|---|---|
| `raceId` | results | row/group key |
| `driverId` | results | row/group key |
| `constructorId` | results | row/group key |
| `year` | races | used for the fixed train/val/test split (Decision 008); also a candidate ordinal feature |
| `round` | races | needed for standings lag and rolling-window ordering |
| `circuitId` | races | join key for circuit history |

### 5.2 Target

| Column | Definition | Notes |
|---|---|---|
| `winner` | `1` if `result_status == "Finished"` and `position == 1` (or `positionOrder == 1`), else `0` | Use `positionOrder`, not `positionText`/`position`, to avoid string-vs-int ambiguity already resolved once by `cleaner.py`. Grouping check: exactly one `winner == 1` per raceId (add as a test assertion). |

### 5.3 Pre-race features — grid & qualifying (safe: known before lights out)

| Column | Source | Notes |
|---|---|---|
| `grid` | results.grid | Known once the grid is set (post-qualifying, pre-race). `grid == 0` in Ergast means started from pit lane — treat as its own category or worst-case grid, not literal 0. |
| `grid_position_norm` | engineered: `grid / field_size_this_race` | Normalizes across variable grid sizes (era effect noted in EDA). |
| `qualifying_position` | qualifying.position | Left join; null for drivers absent from `qualifying.csv`. |
| `q1_time_sec`, `q2_time_sec`, `q3_time_sec` | qualifying.q1/q2/q3, parsed `"M:SS.sss"` → seconds | **54% of `q3` is null even in 2010+** — this is *informative* (didn't reach Q3), not missing-at-random. Do not mean/median-impute; either leave null for tree models (native NaN handling) or add an explicit `reached_q2`/`reached_q3` boolean. |
| `qualifying_gap_to_pole_pct` | engineered: `(driver_best_q_time - pole_time) / pole_time` | Use the best available time per driver (Q3 if present, else Q2, else Q1) vs. the session pole time. |

### 5.4 Rolling driver form (strict prior-race window — see §6.2 for the leakage rule)

| Column | Window | Notes |
|---|---|---|
| `driver_wins_last_3` / `_5` / `_10` | trailing N races, driver | count of `winner==1` in prior races only |
| `driver_podiums_last_5` | trailing 5 | `positionOrder <= 3` |
| `driver_avg_finish_last_5` | trailing 5 | mean `positionOrder`, finishers only or with a DNF penalty value — decide in implementation, document the choice |
| `driver_dnf_rate_last_5` | trailing 5 | `finished == False` rate |
| `driver_points_last_5` | trailing 5 | sum or mean of `points` |
| `driver_experience_races` | all prior races (career-to-date count) | cumulative, prior-only |

### 5.5 Rolling constructor form (same temporal rule, grouped by constructorId)

| Column | Window |
|---|---|
| `constructor_wins_last_5` | trailing 5 |
| `constructor_podiums_last_5` | trailing 5 |
| `constructor_dnf_rate_last_5` | trailing 5 (reliability proxy) |

### 5.6 Circuit history (prior visits to this circuitId only, any season before the current race)

| Column | Notes |
|---|---|
| `driver_circuit_starts` | count of prior entries at this circuit |
| `driver_circuit_wins` | count of prior wins at this circuit |
| `driver_circuit_avg_finish` | mean prior `positionOrder` at this circuit |
| `constructor_circuit_wins` | same, constructor-level |

Low-sample-size warning: many (driver, circuit) pairs will have 0–2 prior visits —
these features will be sparse/noisy for less experienced drivers and at newer
circuits; document as a known limitation rather than trying to backfill.

### 5.7 Standings (LAGGED — see §6.2, this is the highest-severity leakage risk in the whole design)

| Column | Notes |
|---|---|
| `driver_standing_position_prev` | standings **position** (rank), not raw points — Decision-008 rationale (2010 points-system change breaks raw point comparability) applies here too |
| `driver_standing_points_prev` | secondary, same-round-N-1 value |
| `driver_standing_wins_prev` | wins-to-date as of round N-1 |
| `constructor_standing_position_prev` | same, constructor level |
| `constructor_standing_points_prev` | same |

For **round 1 of any season**, there is no "previous round" within that season —
use the **final standing of the prior season** (or a null/neutral sentinel for a
driver's/constructor's first-ever season). This must be an explicit, tested rule,
not an accidental null.

### 5.8 Driver/constructor bio and context (static or slow-moving facts, safe)

| Column | Source | Notes |
|---|---|---|
| `driver_age_at_race` | `drivers.dob` vs `races.date` | static fact, safe |
| `is_home_circuit` | `drivers.nationality` vs `circuits.country` | **Requires a manual mapping table** — see §6.3, this is not a direct join |
| `is_sprint_weekend` | `races.sprint_date is not null` | known before the race, safe |
| `season_races_completed` | `round - 1` | proxy for "how much prior-season data exists," safe |

---

## 6. Leakage Risks

Ranked by severity. Every one of these must have a corresponding unit test in
`tests/test_features.py` before Phase 3 is considered done — this document is not
a substitute for that test suite.

### 6.1 — CRITICAL: In-race and post-race columns from `results.csv` must never be features

`position`, `positionText`, `positionOrder`, `points`, `laps`, `time`,
`milliseconds`, `rank`, `fastestLap`, `fastestLapTime`, `fastestLapSpeed`,
`statusId`, `result_status`, `finished` all describe the **outcome of the race being
predicted**. They may be used to:
- compute the target (`winner`), and
- compute rolling/circuit-history features **for prior races only**

but must **never** appear as a feature value for the current race's own row.
`lap_times.csv` and `pit_stops.csv` are excluded from the master dataset entirely —
they are generated *during* the race and have no pre-race analog.

### 6.2 — CRITICAL: Rolling windows and standings must use strict `(year, round)` ordering, exclusive of the current race

Two distinct failure modes to guard against:
1. **Off-by-one inclusion**: a rolling-window `groupby().rolling()` that isn't
   shifted will include the current race's own result in its own "last 5 races"
   average. Every rolling feature must be computed on `shift(1)` (or an explicit
   `round < current_round` filter within `year`, correctly spanning season
   boundaries) before windowing.
2. **Standings lag**: `driver_standings.csv` / `constructor_standings.csv` rows for
   `raceId` X reflect the standing **after** race X is run — joining them directly
   onto race X's feature row leaks that race's own result (a driver who won race X
   will show race X's win baked into their post-race standing). Standings must be
   joined at **round N−1** (Decision 008 already mandated this at the architecture
   level; this document operationalizes it at the column level).

Recommended implementation pattern: build a chronological index of
`(year, round)` first, and implement lagging/windowing as a join against a
`round - 1` (or `shift(1)` within a driver/constructor group) key — never a
same-row transform on the raceId the row already carries.

### 6.3 — MEDIUM: `is_home_circuit` has no direct join key

`drivers.nationality` uses demonym adjectives ("British", "American-Italian",
"Argentinian " — note inconsistent trailing whitespace found in the raw data)
while `circuits.country` uses country names ("United Kingdom" would be expected but
the raw data uses forms like "Australia", "Italy" etc.). There is **no shared
column** — this requires a hand-built nationality → country mapping table
(~30–40 entries), which is itself a maintenance burden if new driver
nationalities appear in future data. Mark this feature as best-effort; a mapping
gap should degrade gracefully to `is_home_circuit = False`/null rather than raise.

### 6.4 — MEDIUM: Sprint results are sparse and time-ordered *within* the same race weekend

Sprint races exist for only 27 of 305 races in the 2010–2024 window, all from 2021
onward. Sprint sessions run on Saturday, before Sunday's Grand Prix — so a sprint
result for the *same* `raceId` is technically pre-race information relative to the
Grand Prix and is **not leakage** if included. However:
- It creates a large structural-null block for ~91% of rows (pre-2021 and
  non-sprint weekends), which the model must handle explicitly (not impute as 0).
- It only overlaps the tail of the Decision-008 windows (val: 2022–2023 has
  sprints; test: 2024 has sprints; train: 2010–2021 has sprints only in 2021).
- Recommendation: treat as an **optional enrichment column set** gated by
  `is_sprint_weekend`, not a required v1 feature — revisit once the core model is
  working.

### 6.5 — LOW: `grid == 0` is a coded sentinel, not a real front-row-adjacent grid slot

Ergast encodes a pit-lane start as `grid = 0`. Treating it as numerically "better
than grid 1" (if signed the wrong way) or silently averaging it into rolling grid
stats would corrupt any grid-based feature. Must be handled as a distinct category
or remapped to `field_size + 1` (worst-case) before any numeric grid feature is
derived — resolved explicitly across the modules in `src/features/`; do not leave implicit.

### 6.6 — LOW: Mid-season constructor changes

A driver who changes constructors mid-season (e.g., a mid-season replacement) will
have `driver_*` rolling stats spanning two different cars. This is **intentional**
(the feature is about the driver, not the team), but `constructor_*` rolling stats
computed for that constructor will correctly reflect the team's form regardless of
which driver was in the seat — no leakage, but worth a one-line comment in
the relevant module under `src/features/` so a future reader does not “fix” it into a bug.

### 6.7 — LOW: Field-size and era normalization already flagged in EDA

Per `reports/eda_summary.md`, field size and finish rates shifted materially around
2010 (Decision 008 restricts training data to 2010+ for this reason). Any feature
normalized by "current field size" (e.g., `grid_position_norm`) is safe *within* the
modeling window but must not be naively compared against pre-2010 data if that data
is ever reintroduced for auxiliary analysis.

---

## 7. Explicit Non-Goals (Out of Scope for the Master Dataset v1)

- Weather features — no weather columns exist in any current CSV; would require a
  new external data source (already noted in `project_overview.md` roadmap).
- Lap-by-lap or pit-stop-derived features — excluded per §6.1; could support a
  *separate* in-race or strategy model later, but that is a different grain
  (lap-level) and a different prediction window (mid-race), not this dataset.
- Multiclass "predict the field order" framing — out of scope per Decision 003.

---

## 8. Summary Table — Column Inclusion Decision

| Category | Include in v1? |
|---|---|
| Grid + qualifying (pre-race) | Yes |
| Rolling driver/constructor form (lagged) | Yes |
| Circuit history (lagged) | Yes |
| Standings (lagged to round N−1) | Yes |
| Driver age, home circuit | Yes (home circuit best-effort) |
| Sprint weekend results | Optional enrichment, not required v1 |
| Any in-race/post-race results column as a feature | **No — leakage** |
| Lap times, pit stops | **No — leakage, wrong grain** |
| Weather | Not available — future data source |

---

## 9. Next Steps (implementation — not started)

1. Confirm this design (this document) before writing any code.
2. Extend `build_interim.py`: `clean_qualifying()` → `qualifying.parquet`;
   standings lag → `standings.parquet`.
3. Implemented modular feature files plus `src/features/pipeline.py`, enforcing §6
   through executable checks (Decision 011 replaced the proposed monolithic file).
4. Write `tests/test_features.py` with explicit temporal-leakage assertions
   (one test per §6 risk, minimum).
5. Only then: build `data/processed/features.parquet`.
