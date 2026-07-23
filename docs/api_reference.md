# F1 Race Winner Prediction — API Reference

Base URL (local default): `http://localhost:8000`. Every route below is
versioned under `/api/v1` — e.g. `http://localhost:8000/api/v1/health`.
Interactive docs: `/docs` (Swagger UI), `/redoc` — both only ever list the
versioned paths. Every route is ALSO reachable at its pre-versioning path
with no `/api/v1` prefix (same handler, same response, not shown in
`/docs`) for anything already hardcoded to it; new integrations should use
the versioned paths shown here. All responses are JSON.

No authentication is required or implemented — this is a public, read-only
demonstration deployment, not a multi-user service. Every route is `GET`
except `POST /predict`, which accepts a small identity payload (year/round,
optional entry-list override) — never feature values, which stay
server-derived on every route.

---

## GET /api/v1/health

Liveness plus the identity of the serving model.

```json
{
  "status": "ok",
  "api_version": "1.4.0",
  "model": {
    "name": "f1-winner", "version": "4", "alias": "Staging",
    "run_id": "9fd5c220dc2548bda6286ae27f5d31ed",
    "trained_at": "2026-07-11T21:13:12+00:00",
    "calibration": "isotonic-oof", "model_class": "CalibratedModel"
  },
  "detail": null
}
```

`status` is `"degraded"` (with `detail`) if the model or feature data failed
to load — prediction routes then return **503**.

## GET /api/v1/model

The `model` object above, alone. **503** when degraded.

## GET /api/v1/races

Races available for scoring (seasons up to `F1_SERVE_MAX_YEAR`; the
2025–2026 forward holdout is never listed).

Query parameters: `year` (optional int) — filter to one season.

```json
{ "races": [ { "race_id": 1098, "year": 2023, "round": 1, "n_drivers": 20 } ] }
```

## GET /api/v1/races/upcoming

Identity only — year, round, name, circuit, date — of the single next race
with no result yet. **Not a prediction**: no `materialization_status`,
`caveats`, or `provenance`; those only appear once you actually request a
scored prediction from `POST /predict`. Backs the dashboard's upcoming-race
picker entry.

```json
{ "race_id": 1141, "year": 2026, "round": 13, "name": "Belgian Grand Prix",
  "circuit_id": 13, "date": "2026-07-26" }
```

Errors: **404** if every calendar race already has a result (nothing
upcoming to show) · **503** if the training-side data this lookup needs
isn't available (see [docs/pre_race_materialization.md](pre_race_materialization.md)
for why this route needs more than the committed `artifacts/` tree).

## GET /api/v1/predictions/{race_id}

Score the full field of one historical race.

```json
{
  "prediction_id": "1b0ee534-3c73-45bb-829e-92b9d7cb8be6",
  "race_id": 1120, "year": 2023, "round": 22,
  "generated_at": "2026-07-04T00:45:12+00:00",
  "model": { "...": "as in /health" },
  "predictions": [
    {
      "driver_id": 830, "driver_name": "Max Verstappen",
      "constructor_id": 9, "constructor_name": "Red Bull",
      "predicted_rank": 1,
      "win_probability": 0.7634,
      "win_probability_raw": 0.9999
    }
  ],
  "actual_winner_driver_id": 830,
  "model_top1_hit": true
}
```

Field notes:
- `win_probability` — per-race sum-normalized share (the user-facing
  number; sums to 1.0 across the race).
- `win_probability_raw` — the calibrated model output before normalization.
- `predicted_rank` — 1 = most likely winner; deterministic tiebreak, so use
  this (not probability equality) for ordering. Tied probabilities are
  normal — the isotonic calibrator is a step function.
- `driver_name` / `constructor_name` — `null` if the display-name lookup
  files are unavailable.
- `actual_winner_driver_id` / `model_top1_hit` — `null` when the outcome is
  unknown.
- `prediction_id` — matches the API's structured log line for this request.

Errors: **404** unknown raceId · **409** forward-holdout race (config
`F1_SERVE_MAX_YEAR`) · **503** degraded.

Responses are cached per `(model_version, race_id)` — repeated calls return
the identical body.

## GET /api/v1/predictions/{race_id}/simulate/{driver_id}

Re-scores one driver under a hypothetical grid position or a pit-lane
start, holding every other feature at its real value for the race — the
dashboard's "Prediction Simulator." Only the grid-position-derived
features move; the rest of the qualifying group (lap times, Q2/Q3
progression, gap to pole) stays frozen at what actually happened, since
fabricating a lap the driver never set would misrepresent the model.

Query parameters: `grid_position` (int, 1..field size) or `pit_lane=true`.

```json
{
  "race_id": 1120, "driver_id": 830, "driver_name": "Max Verstappen",
  "field_size": 20,
  "real_grid_position": 1.0,
  "simulated_grid_position": 5.0,
  "pit_lane_start": false,
  "real_win_probability": 0.7634,
  "simulated_win_probability": 0.4108,
  "field": [ "...same shape as GET /predictions/{race_id}'s predictions array, re-normalized under the override" ],
  "locked_qualifying_features": ["qualifying_position", "q1_sec", "..."],
  "locked_features": ["driver_form_5race", "constructor_form_5race", "..."],
  "model": { "...": "as in /health" }
}
```

Requesting the driver's own real grid position exactly reproduces their
real `win_probability` — this is a regression-tested guarantee, not a
coincidence.

Errors: **404** unknown race or driver · **422** missing/out-of-range
`grid_position` · **409** forward-holdout race.

## GET /api/v1/predictions/{race_id}/vs-baseline

The full model's predictions next to a simple pole-only baseline (predicts
the pole sitter wins with certainty) for the same race and driver set —
the dashboard's "Qualifying Impact" view, illustrating what the model adds
beyond grid position alone.

```json
{
  "race_id": 1120, "year": 2023, "round": 22,
  "model": { "...": "as in /health" },
  "baseline_name": "pole_baseline",
  "baseline_description": "Predicts the pole sitter wins with certainty.",
  "model_predictions": [ "...same shape as GET /predictions/{race_id}'s predictions array" ],
  "baseline_predictions": [ "...same shape, always ranking the pole sitter first" ],
  "actual_winner_driver_id": 830,
  "model_top1_hit": true,
  "baseline_top1_hit": true
}
```

Errors: **404** unknown raceId · **409** forward-holdout race · **503**
if the baseline itself failed to initialize (degrades independently of the
main model).

## GET /api/v1/debug/features/{race_id} — development only

Disabled by default (returns 404). Enable with `F1_DEBUG_ENDPOINTS=true`.
Returns the exact feature vectors fed to the model for a race:

```json
{
  "race_id": 1120,
  "model": { "...": "as in /health" },
  "feature_names": ["qualifying_position", "..."],
  "rows": [ { "driver_id": 830, "features": { "qualifying_position": 1.0, "q3_sec": null } } ]
}
```

`null` feature values are informative missingness (e.g. eliminated before
Q3, no prior history at this circuit) — the model consumes them as signals.

## POST /api/v1/predict

Score the single next race with no result yet (the same race
`GET /races/upcoming` identifies) with the served model — reusing the
exact same feature pipeline and calibration as every historical
prediction. See [docs/pre_race_materialization.md](pre_race_materialization.md)
for how the feature row is built and the limitations that come with that.

Request body:

```json
{
  "year": 2026, "round": 13,
  "entry_list": null,
  "as_of": null
}
```

- `entry_list` — optional list of `{"driver_id": ..., "constructor_id": ...}`
  pairs. Omit to use the roster inferred from the most recently completed
  race. Never a feature payload — driver/constructor identity only;
  everything feature-shaped is still built server-side.
- `as_of` — optional, reserved for a future historical-cutoff override.
  If present, must be an ISO-8601 timestamp with an explicit UTC offset
  (e.g. `"2026-07-23T10:00:00Z"`) — a naive timestamp is rejected with
  `422` rather than silently assumed to be UTC.

Response:

```json
{
  "prediction_id": "…", "year": 2026, "round": 13,
  "materialization_status": "post_qualifying",
  "missing_inputs": [],
  "generated_at": "2026-07-23T10:05:00+00:00",
  "model": { "...": "as in /health" },
  "predictions": [ "...same shape as GET /predictions/{race_id}'s predictions array" ],
  "caveats": [
    "grid_adjusted/grid_position_norm are sourced from qualifying_position — a post-qualifying grid penalty or pit-lane start is not yet reflected."
  ],
  "provenance": {
    "model_version": "4", "model_alias": "Staging",
    "feature_schema_version": "…", "etl_snapshot_version": "…",
    "data_as_of": "…", "materialized_at": "…", "predicted_at": "…",
    "qualifying_status": "complete", "completeness_status": "post_qualifying"
  }
}
```

Field notes:
- `materialization_status` / `completeness_status` — `"post_qualifying"`
  (qualifying fully recorded) or `"pre_qualifying"` (not yet — an
  additional caveat is included in that case). `missing_inputs` names any
  gaps.
- `provenance` — every prediction is reconstructable later from its own
  recorded metadata alone: which model, which feature schema, which ETL
  snapshot, and when each step ran.
- Cached per `(model_version, year, round, feature_schema_version,
  etl_snapshot_version, entry_list_hash)` — any of those six changing
  (a new model promoted, a fresh ETL snapshot, a late roster confirmation)
  invalidates the cached response rather than serving it stale.

Errors: **409** the race already has a real result (not "upcoming"
anymore) · **422** invalid `entry_list`/`as_of` · **503** the training-side
data this route needs isn't available (see
[docs/pre_race_materialization.md](pre_race_materialization.md)).

## Errors

Every error response is a JSON body of the shape `{"detail": "..."}`, with
the exception of an unexpected server error, which always returns a
generic `{"detail": "Internal server error."}` at `500` — no internal
detail (message, type, or traceback) is ever included in the response body;
the real cause is logged server-side only.
