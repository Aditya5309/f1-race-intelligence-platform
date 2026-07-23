# Pre-Race Materialization Pipeline

How `POST /api/v1/predict` scores a race that hasn't happened yet, using
the same served model and the same feature pipeline as every historical
prediction — no separate model, no third-party call at request time.

## The problem this solves

Every other prediction route (`GET /predictions/{race_id}`) looks up an
already-computed feature row from the committed `artifacts/features.parquet`
snapshot. That snapshot only has rows for races that have already been
through the full batch pipeline (`src/integration` → `src/features`) —
which requires a finished result. There is no row to look up for a race
that hasn't run yet.

The materialization pipeline builds that row on demand: it resolves which
race is "next," assembles a synthetic entry for each driver, and runs it
through the **same, unmodified** feature-engineering code the batch
pipeline uses, so the served model sees a row indistinguishable in shape
from the ones it was trained on.

## Architecture

```
src/features/upcoming.py        Calendar & entry-list resolution
  next_race()                   "which race is next?" — the single earliest
                                 race with no results.csv row yet (horizon=1)
  resolve_entry_list()          "who's entered?" — inferred from the most
                                 recently completed race, backward-looking only

src/models/materialize.py       The Materializer
  materialize_features()        Builds one synthetic row per entrant and
                                 feeds it through src.features.pipeline
                                 (build_features/validate_features)
                                 UNMODIFIED — no parallel feature logic

app/upcoming_prediction_service.py   Orchestration (keeps app/api.py thin)
  ensure_materialization_data()      Lazy-loads training-side data on first
                                     request, caches success/failure/not-yet-tried
  resolve_upcoming_race()            Backs GET /races/upcoming (identity only)
  resolve_upcoming_prediction()      Backs POST /predict (materialize + score)

app/api.py                      GET /races/upcoming, POST /predict — thin
                                 transport, all orchestration delegated above

app/views/race_center.py        Dashboard integration — picker entry for the
                                 upcoming race, prediction rendering, provenance
                                 and caveat display
```

Reused, not duplicated: `materialize_features()` calls the exact same
`src.features.pipeline.build_features()`/`validate_features()` and
`src.integration.build_master_dataset.validate_output()` the real batch
pipeline uses. The only new logic is assembling one valid synthetic row
per entrant before handing it to that existing pipeline.

## Known limitations (disclosed, not oversights)

**Qualifying position stands in for final grid.** `grid` is set equal to
`qualifying_position` for every entrant. This project has no sanctioned
source for grid penalties before a race is run — jolpica-f1/Ergast's
`grid` field only exists inside the results endpoint, which has nothing to
return before the race happens. Consequence: `pit_lane_start` and
`grid_penalty_applied` can never read true for a materialized race. Every
`POST /predict` response includes this as a caveat.

**`POST /predict` needs training-side data with no `artifacts/`-tree
equivalent.** Every other route serves from the committed `artifacts/`
tree alone. This route additionally needs `historical_master` (a full
`master_dataset.parquet`-shaped frame) and the raw dimension tables it's
built from — `artifacts/features.parquet` only carries final computed
features, not the raw columns the Materializer re-derives them from. These
paths default to the local `data/` tree (present on a dev checkout, or a
container with `data/` bind-mounted — see `docker-compose.override.yml`);
if missing, `POST /predict` alone degrades to `503` — every other route is
unaffected, the same pattern as the pole-baseline's own degraded-start
handling. This data is lazily loaded on the first `POST /predict` (or
`GET /races/upcoming`) request, not at startup, and cached thereafter.

## Design discipline this pipeline follows

Every feature in this project must be **provably computable at the moment
the starting grid is known** — the platform's #1 correctness constraint.
Two specific rules the Materializer and its tests enforce:

- **Backward-only resolution.** `resolve_entry_list()` only ever looks at
  the most recently completed race strictly before the target race — never
  forward, and never at the target race's own (nonexistent) results.
- **No fabrication.** A driver with no qualifying row yet gets a null
  qualifying/grid-derived feature, never a synthetic zero — "no
  information" and "an actual zero" are different signals, and the model
  is trained to treat them differently (missingness is informative, not
  noise, throughout this project's feature set).

## Acceptance gates

Two mandatory checks passed before `POST /predict` was enabled — both
still run as part of the regular test suite so a future change can't
silently regress them:

- **Golden-row parity** (`tests/test_materialize_golden_row_parity.py`) —
  runs the Materializer in "pretend this hasn't happened yet" mode against
  every race in the served model's own validation/test windows, plus a
  stratified sample of training-window races, and diffs the result
  column-for-column against that race's real row in
  `features.parquet`. The rule: no phase of this pipeline may ship until
  this passes.
- **Historical backtest** (`tests/test_materialize_historical_backtest.py`)
  — re-scores the same historical sample through the materialized path
  and confirms per-race predictions and aggregate metrics land within a
  documented noise band of the real, already-served results (this
  project's dataset is small enough that validation-metric variance
  between individual races is itself non-trivial — roughly ±1 race ≈
  ±2.3 percentage points on the validation split — so the bar is set to
  catch a real regression, not this ordinary noise).

## Cache invalidation

`POST /predict` responses are cached, keyed by six values: `model_version`,
`year`, `round`, `feature_schema_version`, `etl_snapshot_version`, and an
`entry_list_hash`. Any of the six changing (a new model promoted, a new
ETL snapshot ingested, the feature schema changing, or the entry list
itself changing — a late roster confirmation) invalidates the cached
response for that race rather than serving a stale prediction.

## Full request/response contract

See [docs/api_reference.md](api_reference.md#post-apiv1predict) for the
exact `POST /predict` and `GET /races/upcoming` request/response shapes,
and [docs/user_guide.md](user_guide.md) for how the dashboard surfaces
this in Race Center.
