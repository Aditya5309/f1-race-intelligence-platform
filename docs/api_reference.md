# F1 Race Winner Prediction — API Reference

Base URL (local default): `http://localhost:8000`. Interactive docs:
`/docs` (Swagger UI), `/redoc`. All responses are JSON. No authentication
in v1 (local use; see `reports/application_design.md` §13 for the planned
API-key path).

---

## GET /health

Liveness plus the identity of the serving model.

```json
{
  "status": "ok",
  "api_version": "1.0.0",
  "model": {
    "name": "f1-winner", "version": "2", "alias": "Staging",
    "run_id": "0c16d584af9047fda616e0fa473b4dd9",
    "trained_at": "2026-07-03T18:25:30+00:00",
    "calibration": "isotonic-oof", "model_class": "CalibratedModel"
  },
  "detail": null
}
```

`status` is `"degraded"` (with `detail`) if the model or feature data failed
to load — prediction routes then return **503**.

## GET /model

The `model` object above, alone. **503** when degraded.

## GET /races

Races available for scoring (seasons up to `F1_SERVE_MAX_YEAR`; the
2025–2026 forward holdout is never listed).

Query parameters: `year` (optional int) — filter to one season.

```json
{ "races": [ { "race_id": 1098, "year": 2023, "round": 1, "n_drivers": 20 } ] }
```

## GET /predictions/{race_id}

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

## GET /debug/features/{race_id} — development only

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

## POST /predict — reserved

Always returns **501 Not Implemented** in v1. The route is reserved as the
future entry point for *upcoming-race* predictions (races that haven't run
yet), which require the feature pipeline to materialize pre-race feature
rows first. See `reports/application_design.md` §5/§12.
