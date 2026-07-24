"""
Tests for app/api.py.

TestClient against create_app() with an isolated stack: tmp sqlite registry
holding a calibrated model + a tiny synthetic features parquet, wired in via
Settings. Covers: health (ok + degraded), model metadata, race listing +
year filter, prediction happy path (normalization,
ordering, winner comparison, prediction_id), 404 unknown race,
debug endpoint gating, reserved POST /predict (501),
and cache behavior across calls.
"""

import hashlib
import json
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import API_V1_PREFIX, Settings
from src.features.metadata import active_feature_columns
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.registry import training_schema
from src.models.serving_bundle import DEFAULT_FEATURES_ARTIFACT, bundle_dir_for_alias
from src.models.splits import temporal_split
from src.models.train import register_model
from tests.conftest import set_tmp_experiment


def _synthetic_features(years, races_per_year=3, n_drivers=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for rnd in range(1, races_per_year + 1):
            race_id = year * 100 + rnd
            grid = rng.permutation(n_drivers) + 1
            for driver in range(n_drivers):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row["grid_adjusted"] = float(grid[driver])
                row["grid_position_norm"] = float(grid[driver]) / n_drivers
                row.update({
                    "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                    "circuitId": 1, "year": year, "round": rnd,
                    TARGET_COLUMN: int(grid[driver] == 1),
                })
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def serving_stack(tmp_path_factory):
    """Registry (calibrated model @Staging) + exported bundle + features
    parquet + Settings. bundle_root/features_source/artifacts_root are all
    explicit tmp paths — register_model would otherwise read/write the real
    project's data/processed/features.parquet and artifacts/serving/."""
    root = tmp_path_factory.mktemp("serving")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    bundle_root = root / "bundle"
    # 2010-2025 rows; 2025 included so tests can assert it serves like any
    # other completed season now that the verified-seasons gate is removed.
    frame = _synthetic_features(range(2010, 2026))
    features_path = root / "features.parquet"
    frame.to_parquet(features_path)

    mlflow.set_tracking_uri(uri)
    set_tmp_experiment("test-experiment", root)
    split = temporal_split(frame)
    register_model("logreg", split, alias="Staging", calibrate=True,
                   bundle_root=bundle_root, features_source=features_path,
                   artifacts_root=root / "artifacts")
    mlflow.set_tracking_uri(None)

    settings = Settings(
        serving_bundle_path=bundle_dir_for_alias("Staging", bundle_root),
        features_path=features_path,
        data_dir=root / "no-such-dir",     # name lookups degrade to null
        debug_endpoints=False,
    )
    return settings, frame


@pytest.fixture(scope="module")
def client(serving_stack):
    settings, _ = serving_stack
    with TestClient(create_app(settings)) as c:    # context manager runs lifespan
        yield c


@pytest.fixture(scope="module")
def debug_client(serving_stack):
    settings, _ = serving_stack
    debug_settings = settings.model_copy(update={"debug_endpoints": True})
    with TestClient(create_app(debug_settings)) as c:
        yield c


IN_WINDOW_RACE = 202301       # 2023 round 1
RECENT_SEASON_RACE = 202501  # 2025 round 1 — just another servable race, not a holdout


# ---------------------------------------------------------------------------
# Health and model metadata
# ---------------------------------------------------------------------------

def test_health_ok_with_model_metadata(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["name"] == "f1-winner"
    assert body["model"]["alias"] == "Staging"
    assert body["model"]["calibration"] == "isotonic-oof"
    assert body["model"]["model_class"] == "CalibratedModel"


def test_health_degraded_when_registry_missing(tmp_path):
    settings = Settings(
        serving_bundle_path=tmp_path / "no-such-bundle",
        features_path=tmp_path / "missing.parquet",
    )
    with TestClient(create_app(settings)) as c:
        body = c.get("/health").json()
        assert body["status"] == "degraded"
        assert body["detail"]
        assert c.get("/model").status_code == 503
        assert c.get("/predictions/1").status_code == 503


def test_model_endpoint(client):
    body = client.get("/model").json()
    assert body["version"] == "1"
    assert body["calibration"] == "isotonic-oof"
    assert body["trained_at"].startswith("20")


# ---------------------------------------------------------------------------
# Race listing
# ---------------------------------------------------------------------------

def test_races_listed_includes_every_season(client, serving_stack):
    _, frame = serving_stack
    body = client.get("/races").json()
    years = {r["year"] for r in body["races"]}
    assert max(years) == 2025                      # no season-based filtering
    assert len(body["races"]) == frame["raceId"].nunique()
    first = body["races"][0]
    assert set(first) == {"race_id", "year", "round", "n_drivers"}
    assert first["n_drivers"] == 5


def test_races_year_filter(client):
    body = client.get("/races", params={"year": 2023}).json()
    assert body["races"]
    assert all(r["year"] == 2023 for r in body["races"])


def test_races_membership_exactly_matches_features_parquet(client, serving_stack):
    """The architectural replacement for the removed verified-seasons gate
    (Decision 057): a race is servable if and only if it has rows in the
    features snapshot — nothing else gates it, and nothing exempts it."""
    _, frame = serving_stack
    listed_ids = {r["race_id"] for r in client.get("/races").json()["races"]}
    assert listed_ids == set(frame["raceId"].unique())

    absent_race_id = int(frame["raceId"].max()) + 1000
    assert absent_race_id not in listed_ids
    assert client.get(f"/predictions/{absent_race_id}").status_code == 404


# ---------------------------------------------------------------------------
# GET /races/upcoming — identity-only lookup, NOT a prediction.
# Uses the isolated `predict_client` fixture (defined further down, near the
# other POST /predict tests) rather than the module's `serving_stack`, since
# it needs a races.csv/master_dataset.parquet pair with a genuine result-less
# race — reused here via a forward reference resolved at collection time.
# ---------------------------------------------------------------------------

def test_races_upcoming_returns_identity(predict_client):
    resp = predict_client.get(f"{API_V1_PREFIX}/races/upcoming")
    assert resp.status_code == 200
    body = resp.json()
    assert body["race_id"] == 2
    assert body["year"] == 2026
    assert body["round"] == 2
    assert set(body) == {"race_id", "year", "round", "name", "circuit_id", "date"}


def test_races_upcoming_404_when_every_race_has_a_result(serving_stack, tmp_path):
    settings, _ = serving_stack
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame([
        {"raceId": 1, "year": 2026, "round": 1, "circuitId": 1, "name": "R1", "date": "2026-01-01"},
    ]).to_csv(raw_dir / "races.csv", index=False)
    master_path = tmp_path / "master_dataset.parquet"
    pd.DataFrame({"raceId": [1]}).to_parquet(master_path)
    qualifying_path = tmp_path / "qualifying.parquet"
    pd.DataFrame(columns=["raceId"]).to_parquet(qualifying_path)

    complete_settings = settings.model_copy(update={
        "raw_data_dir": raw_dir, "master_dataset_path": master_path,
        "qualifying_interim_path": qualifying_path,
        "weather_csv_path": raw_dir / "no-such-weather.csv",
    })
    # weather_csv_path deliberately points nowhere -- load_race_weather()
    # would fail on a missing file; races/master (the only tables this
    # route reads) are all that matters here, but ensure_materialization_
    # data() loads everything, so give it a real (empty, valid) weather CSV.
    pd.DataFrame(columns=["raceId", "race_precip_mm", "race_temp_c",
                          "quali_precip_mm", "conditions_changed"]).to_csv(
        raw_dir / "no-such-weather.csv", index=False)
    pd.DataFrame([{"driverId": 1, "driverRef": "d1", "code": "D1", "forename": "F",
                   "surname": "L", "dob": "1990-01-01", "nationality": "N"}]).to_csv(
        raw_dir / "drivers.csv", index=False)
    pd.DataFrame([{"constructorId": 1, "constructorRef": "c1", "name": "C1",
                   "nationality": "N"}]).to_csv(raw_dir / "constructors.csv", index=False)
    pd.DataFrame([{"circuitId": 1, "circuitRef": "circ1", "name": "Circuit",
                   "location": "L", "country": "C", "lat": 0.0, "lng": 0.0, "alt": 0}]).to_csv(
        raw_dir / "circuits.csv", index=False)
    pd.DataFrame([{"raceId": 1, "driverId": 1, "constructorId": 1, "points": 0,
                   "position": 1, "positionText": "1", "wins": 0}]).to_csv(
        raw_dir / "driver_standings.csv", index=False)
    pd.DataFrame([{"raceId": 1, "constructorId": 1, "points": 0, "position": 1,
                   "positionText": "1", "wins": 0}]).to_csv(
        raw_dir / "constructor_standings.csv", index=False)

    with TestClient(create_app(complete_settings)) as c:
        resp = c.get(f"{API_V1_PREFIX}/races/upcoming")
        assert resp.status_code == 404


def test_races_upcoming_503_when_materialization_data_unavailable(serving_stack, tmp_path):
    settings, _ = serving_stack
    missing_dir = tmp_path / "no-such-raw"
    broken_settings = settings.model_copy(update={
        "raw_data_dir": missing_dir,
        "master_dataset_path": missing_dir / "master_dataset.parquet",
        "qualifying_interim_path": missing_dir / "qualifying.parquet",
        "weather_csv_path": missing_dir / "race_weather.csv",
    })
    with TestClient(create_app(broken_settings)) as c:
        resp = c.get(f"{API_V1_PREFIX}/races/upcoming")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def test_prediction_happy_path(client, serving_stack):
    _, frame = serving_stack
    body = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    assert body["race_id"] == IN_WINDOW_RACE
    assert body["year"] == 2023 and body["round"] == 1
    assert body["prediction_id"]
    assert body["model"]["calibration"] == "isotonic-oof"

    preds = body["predictions"]
    assert len(preds) == 5
    assert [p["predicted_rank"] for p in preds] == [1, 2, 3, 4, 5]
    shares = [p["win_probability"] for p in preds]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0)
    assert all(p["driver_name"] is None for p in preds)   # lookups degraded

    winner = int(frame[(frame["raceId"] == IN_WINDOW_RACE)
                       & (frame[TARGET_COLUMN] == 1)]["driverId"].iloc[0])
    assert body["actual_winner_driver_id"] == winner
    assert body["model_top1_hit"] == (preds[0]["driver_id"] == winner)


def test_prediction_unknown_race_404(client):
    resp = client.get("/predictions/999999")
    assert resp.status_code == 404
    assert "999999" in resp.json()["detail"]


def test_prediction_recent_season_served(client):
    resp = client.get(f"/predictions/{RECENT_SEASON_RACE}")
    assert resp.status_code == 200
    assert resp.json()["race_id"] == RECENT_SEASON_RACE


def test_prediction_cache_returns_identical_body(client):
    a = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    b = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    # Cached: everything identical including the first call's prediction_id.
    assert a == b


# ---------------------------------------------------------------------------
# Debug endpoint gating
# ---------------------------------------------------------------------------

def test_debug_features_hidden_by_default(client):
    assert client.get(f"/debug/features/{IN_WINDOW_RACE}").status_code == 404


def test_debug_features_when_enabled(debug_client):
    # The serving_stack fixture registers via the default
    # (exclusion-applied) feature set, not the raw full FEATURE_COLUMNS.
    body = debug_client.get(f"/debug/features/{IN_WINDOW_RACE}").json()
    assert body["race_id"] == IN_WINDOW_RACE
    assert body["feature_names"] == list(active_feature_columns())
    assert len(body["rows"]) == 5
    row = body["rows"][0]
    assert set(row["features"]) == set(active_feature_columns())
    assert all(v is None or isinstance(v, float) for v in row["features"].values())


def test_debug_features_recent_season_served(debug_client):
    body = debug_client.get(f"/debug/features/{RECENT_SEASON_RACE}").json()
    assert body["race_id"] == RECENT_SEASON_RACE


# ---------------------------------------------------------------------------
# Prediction Simulator (/predictions/{race_id}/simulate/{driver_id})
#
# _synthetic_features bakes a deterministic relationship into the fixture:
# winner == 1 iff grid_adjusted == 1 (see the module docstring above), with
# every other feature pure noise. The fitted logreg therefore keys almost
# entirely off grid_adjusted, which makes "simulated P1 clearly beats
# simulated last" a real assertion about the model's behavior, not a
# tautology about the endpoint's plumbing.
# ---------------------------------------------------------------------------

def test_simulate_grid_p1_beats_back_of_grid(client):
    p1 = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                    params={"grid_position": 1}).json()
    last = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                      params={"grid_position": 5}).json()
    assert p1["simulated_grid_position"] == 1.0
    assert last["simulated_grid_position"] == 5.0
    assert p1["simulated_win_probability"] > last["simulated_win_probability"]
    assert p1["field_size"] == 5
    assert p1["driver_id"] == 1


def test_simulate_grid_pit_lane(client):
    body = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                      params={"pit_lane": True}).json()
    assert body["pit_lane_start"] is True
    assert body["simulated_grid_position"] == 6.0        # field_size + 1
    p1 = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                    params={"grid_position": 1}).json()
    assert body["simulated_win_probability"] < p1["simulated_win_probability"]


def test_simulate_grid_locked_feature_lists(client):
    body = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                      params={"grid_position": 2}).json()
    # The 3 literally-overridden fields are never reported as locked.
    for adjustable in ("grid_adjusted", "grid_position_norm", "pit_lane_start"):
        assert adjustable not in body["locked_qualifying_features"]
        assert adjustable not in body["locked_features"]
    # The rest of the qualifying group is frozen (not fabricated), not adjustable.
    assert set(body["locked_qualifying_features"]) == {
        "qualifying_position", "q1_sec", "q2_sec", "q3_sec",
        "reached_q2", "reached_q3", "qualifying_gap_to_pole_pct",
        "grid_penalty_applied",
    }
    # 28 historical/standings/teammate/weather aggregates are locked — the
    # served model no longer trains on the wet_form group
    # (driver_wet_dry_delta/constructor_wet_dry_delta), so those 2 are
    # simply absent from the schema entirely, not merely unlocked.
    assert len(body["locked_features"]) == 28
    assert "driver_wins_last_5" in body["locked_features"]
    assert "constructor_standing_position_prev" in body["locked_features"]
    assert "driver_wet_dry_delta" not in body["locked_features"]


def test_simulate_grid_field_renormalizes_but_others_raw_unchanged(client):
    real = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    sim = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                     params={"grid_position": 5}).json()
    real_raw = {p["driver_id"]: p["win_probability_raw"] for p in real["predictions"]}
    sim_raw = {p["driver_id"]: p["win_probability_raw"] for p in sim["field"]}
    # Only the overridden driver's raw model output changes.
    for driver_id, raw in sim_raw.items():
        if driver_id == 1:
            continue
        assert raw == pytest.approx(real_raw[driver_id])
    assert sum(p["win_probability"] for p in sim["field"]) == pytest.approx(1.0)


def test_simulate_grid_unknown_driver_404(client):
    resp = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/999999",
                      params={"grid_position": 1})
    assert resp.status_code == 404


def test_simulate_grid_out_of_range_422(client):
    resp = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1",
                      params={"grid_position": 99})
    assert resp.status_code == 422


def test_simulate_grid_missing_input_422(client):
    resp = client.get(f"/predictions/{IN_WINDOW_RACE}/simulate/1")
    assert resp.status_code == 422


def test_simulate_grid_recent_season_served(client):
    resp = client.get(f"/predictions/{RECENT_SEASON_RACE}/simulate/1",
                      params={"grid_position": 1})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Regression: simulate() at a driver's REAL grid position must exactly
# reproduce the real prediction — the whole "freeze the rest of the
# qualifying group" design (app/api.py's ADJUSTABLE_GRID_FEATURES) depends
# on the override being a true no-op when it matches what actually happened.
# Runs against the ACTUAL committed serving bundle + features snapshot
# (artifacts/serving/staging, artifacts/features.parquet —
# committed to git, unlike the gitignored data/ tree), not the synthetic
# fixture used by every other test in this file, so this is a genuine
# regression check against real 2024 races. Two races are covered: one
# where the model's favorite actually started on pole (real grid == 1,
# the trivial case) and one where it didn't (real grid != 1, the case
# where grid position and predicted rank diverge) — both are asserted
# explicitly below so a data change would fail loudly instead of silently
# testing the wrong scenario.
# ---------------------------------------------------------------------------

_REAL_GRID_REGRESSION_CASES = {
    1121: 1.0,   # 2024 Bahrain (round 1): favorite started on pole
    1129: 2.0,   # 2024 round 9: favorite did NOT start on pole
}


@pytest.fixture(scope="module")
def real_client():
    with TestClient(create_app()) as c:      # default Settings -> real artifacts/
        yield c


@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    return pd.read_parquet(DEFAULT_FEATURES_ARTIFACT)


@pytest.mark.parametrize("race_id", sorted(_REAL_GRID_REGRESSION_CASES))
def test_simulate_at_real_grid_exactly_reproduces_real_prediction(
    real_client, real_features, race_id,
):
    real = real_client.get(f"/predictions/{race_id}")
    assert real.status_code == 200, real.text
    real_body = real.json()
    top = real_body["predictions"][0]
    driver_id = top["driver_id"]

    row = real_features.loc[
        (real_features["raceId"] == race_id) & (real_features["driverId"] == driver_id)
    ].iloc[0]
    real_grid = float(row["grid_adjusted"])
    assert real_grid == _REAL_GRID_REGRESSION_CASES[race_id], (
        f"raceId {race_id}'s favorite's real grid position moved — update "
        "_REAL_GRID_REGRESSION_CASES (the pole-vs-not-pole split this test "
        "covers depends on it)."
    )
    assert not bool(row["pit_lane_start"])

    sim = real_client.get(
        f"/predictions/{race_id}/simulate/{driver_id}",
        params={"grid_position": int(real_grid)},
    )
    assert sim.status_code == 200, sim.text
    sim_body = sim.json()

    assert sim_body["real_grid_position"] == real_grid
    assert sim_body["simulated_grid_position"] == real_grid
    # Hard equality, not pytest.approx: the route is supposed to reproduce
    # the real row exactly (same features in, same predict_proba call out)
    # when the override matches what actually happened.
    assert sim_body["simulated_win_probability"] == sim_body["real_win_probability"]
    assert sim_body["simulated_win_probability"] == top["win_probability"]


# ---------------------------------------------------------------------------
# Qualifying Impact (/predictions/{race_id}/vs-baseline)
# ---------------------------------------------------------------------------

def test_vs_baseline_picks_the_pole_sitter(client, serving_stack):
    _, frame = serving_stack
    body = client.get(f"/predictions/{IN_WINDOW_RACE}/vs-baseline").json()
    race_rows = frame[frame["raceId"] == IN_WINDOW_RACE]
    pole_driver = int(
        race_rows.loc[race_rows["grid_adjusted"] == 1, "driverId"].iloc[0])
    assert body["baseline_name"] == "pole_baseline"
    assert "pole" in body["baseline_description"].lower()
    assert body["baseline_predictions"][0]["driver_id"] == pole_driver
    assert body["baseline_predictions"][0]["predicted_rank"] == 1


def test_vs_baseline_model_predictions_match_plain_endpoint(client):
    plain = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    vs = client.get(f"/predictions/{IN_WINDOW_RACE}/vs-baseline").json()
    plain_probs = {p["driver_id"]: p["win_probability"] for p in plain["predictions"]}
    vs_probs = {p["driver_id"]: p["win_probability"] for p in vs["model_predictions"]}
    assert plain_probs == pytest.approx(vs_probs)
    assert vs["model_top1_hit"] == plain["model_top1_hit"]
    assert vs["actual_winner_driver_id"] == plain["actual_winner_driver_id"]


def test_vs_baseline_unknown_race_404(client):
    assert client.get("/predictions/999999/vs-baseline").status_code == 404


def test_vs_baseline_recent_season_served(client):
    resp = client.get(f"/predictions/{RECENT_SEASON_RACE}/vs-baseline")
    assert resp.status_code == 200


def test_pole_baseline_startup_fit_explicitly_uses_full_feature_columns(client):
    """This call site is worth its own explicit test: get_model()'s
    default (active_feature_columns(), wet_form excluded) would otherwise
    silently mismatch the FULL features.parquet snapshot this route scores
    against (app/api.py builds X via features.loc[:, list(FEATURE_COLUMNS)]
    immediately after). Verify the fix directly — not just via the
    vs-baseline route happening to still work — by checking the actual
    fitted baseline model's own recorded schema."""
    baseline_model = client.app.state.baseline_model
    assert baseline_model is not None, (
        f"pole_baseline failed to load at startup: "
        f"{client.app.state.baseline_load_error}"
    )
    schema = training_schema(baseline_model)["feature_names"]
    assert schema == list(FEATURE_COLUMNS)
    assert "driver_wet_dry_delta" in schema


# ---------------------------------------------------------------------------
# POST /predict — request validation, and the exception-to-HTTP
# mapping app/upcoming_prediction_service.py's plain-Python exceptions go
# through in app/api.py (RaceAlreadyHasResult -> 409, ValueError -> 422,
# RuntimeError -> 503). The exception TYPES themselves are covered in
# tests/test_upcoming_prediction_service.py; these tests prove the real
# route maps them to the right status codes end-to-end. Uses its own
# isolated raw-data fixture (`predict_client`) rather than the module's
# `serving_stack` (which only builds a synthetic features.parquet, not the
# raw dimension tables POST /predict materializes from) — deliberately NOT
# the real project's data/ tree, for test isolation.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predict_client(serving_stack, tmp_path_factory):
    settings, _ = serving_stack
    root = tmp_path_factory.mktemp("predict-service")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    pd.DataFrame([
        {"raceId": 1, "year": 2026, "round": 1, "circuitId": 1, "name": "R1", "date": "2026-01-01"},
        {"raceId": 2, "year": 2026, "round": 2, "circuitId": 1, "name": "R2", "date": "2026-01-08"},
    ]).to_csv(raw_dir / "races.csv", index=False)
    pd.DataFrame([{"driverId": 1, "driverRef": "d1", "code": "D1", "forename": "F",
                   "surname": "L", "dob": "1990-01-01", "nationality": "N"}]).to_csv(
        raw_dir / "drivers.csv", index=False)
    pd.DataFrame([{"constructorId": 1, "constructorRef": "c1", "name": "C1",
                   "nationality": "N"}]).to_csv(raw_dir / "constructors.csv", index=False)
    pd.DataFrame([{"circuitId": 1, "circuitRef": "circ1", "name": "Circuit",
                   "location": "L", "country": "C", "lat": 0.0, "lng": 0.0, "alt": 0}]).to_csv(
        raw_dir / "circuits.csv", index=False)
    pd.DataFrame([{"raceId": 1, "driverId": 1, "constructorId": 1, "points": 0,
                   "position": 1, "positionText": "1", "wins": 0}]).to_csv(
        raw_dir / "driver_standings.csv", index=False)
    pd.DataFrame([{"raceId": 1, "constructorId": 1, "points": 0, "position": 1,
                   "positionText": "1", "wins": 0}]).to_csv(
        raw_dir / "constructor_standings.csv", index=False)
    pd.DataFrame(columns=["raceId", "race_precip_mm", "race_temp_c",
                          "quali_precip_mm", "conditions_changed"]).to_csv(
        raw_dir / "race_weather.csv", index=False)

    processed_dir = root / "processed"
    processed_dir.mkdir()
    master_path = processed_dir / "master_dataset.parquet"
    pd.DataFrame({"raceId": [1]}).to_parquet(master_path)

    interim_dir = root / "interim"
    interim_dir.mkdir()
    qualifying_path = interim_dir / "qualifying.parquet"
    pd.DataFrame(columns=["raceId", "driverId", "constructorId", "number",
                          "position", "q1", "q2", "q3"]).to_parquet(qualifying_path)

    predict_settings = settings.model_copy(update={
        "raw_data_dir": raw_dir, "master_dataset_path": master_path,
        "qualifying_interim_path": qualifying_path,
        "weather_csv_path": raw_dir / "race_weather.csv",
    })
    with TestClient(create_app(predict_settings)) as c:
        yield c


def test_post_predict_requires_body(client):
    """FastAPI's own request validation rejects a bodiless POST with 422
    before the route — and therefore before any lazy materialization-data
    load — ever runs, proving this is real request handling now, not the
    previous unconditional 501."""
    resp = client.post("/predict")
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["msg"] == "Field required"


def test_post_predict_race_already_has_result_409(predict_client):
    resp = predict_client.post(f"{API_V1_PREFIX}/predict", json={"year": 2026, "round": 1})
    assert resp.status_code == 409


def test_post_predict_unknown_race_422(predict_client):
    resp = predict_client.post(f"{API_V1_PREFIX}/predict", json={"year": 2026, "round": 99})
    assert resp.status_code == 422
    assert "not on the races calendar" in resp.json()["detail"]


def test_post_predict_naive_as_of_422(predict_client):
    """Regression test: an offset-less as_of must produce a
    clear 422, not an unhandled TypeError surfacing as a generic 500. The
    companion case (a valid, timezone-aware as_of must NOT be rejected) is
    covered at the service layer —
    tests/test_upcoming_prediction_service.py::test_timezone_aware_as_of_is_accepted
    — where a mocked materializer isolates the as_of check from needing a
    fully realistic historical_master."""
    resp = predict_client.post(
        f"{API_V1_PREFIX}/predict",
        json={"year": 2026, "round": 2, "as_of": "2026-01-01T00:00:00"},
    )
    assert resp.status_code == 422
    assert "UTC offset" in resp.json()["detail"]


def test_post_predict_materialization_data_unavailable_503(serving_stack, tmp_path):
    """A Settings whose Phase-7 paths point nowhere degrades ONLY this
    route to 503 (via the lazy loader), rather than crashing the process."""
    settings, _ = serving_stack
    missing_dir = tmp_path / "no-such-raw"
    broken_settings = settings.model_copy(update={
        "raw_data_dir": missing_dir,
        "master_dataset_path": missing_dir / "master_dataset.parquet",
        "qualifying_interim_path": missing_dir / "qualifying.parquet",
        "weather_csv_path": missing_dir / "race_weather.csv",
    })
    with TestClient(create_app(broken_settings)) as c:
        resp = c.post(f"{API_V1_PREFIX}/predict", json={"year": 2026, "round": 1})
        assert resp.status_code == 503


def _historical_row(**overrides) -> dict:
    """One real MASTER_DATASET_COLUMNS-shaped row (same shape
    tests/test_predict_upcoming.py's fixture uses) -- a genuinely
    completed race for a real, successful materialization."""
    base = {
        "raceId": 1, "driverId": 1, "constructorId": 10, "circuitId": 100,
        "year": 2026, "round": 1,
        "race_name": "Race", "race_date": "2026-01-01",
        "circuit_ref": "circA", "circuit_name": "Circuit A", "circuit_location": "Loc",
        "circuit_country": "Country", "circuit_lat": 1.0, "circuit_lng": 1.0, "circuit_alt": 1,
        "driver_ref": "driver1", "driver_code": "D1", "driver_forename": "First",
        "driver_surname": "Last", "driver_dob": "1990-01-01", "driver_nationality": "Nat",
        "constructor_ref": "team10", "constructor_name": "Team 10", "constructor_nationality": "Nat",
        "grid": 1, "qualifying_position": 1, "q1": "1:20.000", "q2": "1:19.000", "q3": "1:18.000",
        "position": 1, "positionText": "1", "positionOrder": 1, "points": 25.0, "laps": 50,
        "milliseconds": 1000000, "rank": 1, "fastestLap": 30, "fastestLapTime": "1:20.000",
        "fastestLapSpeed": 200.0, "statusId": 1, "result_status": "Finished", "finished": True,
        "winner": 1,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def provenance_client(serving_stack, tmp_path_factory):
    """A fully valid, isolated Phase-7 scenario — one completed race
    (raceId=1) feeding a real materialization + scoring of the upcoming
    race (raceId=2) — for the provenance round-trip test below. Distinct
    from `predict_client` (deliberately just enough data to exercise error
    paths, never a real successful materialization)."""
    settings, _ = serving_stack
    root = tmp_path_factory.mktemp("provenance-service")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    pd.DataFrame([
        {"raceId": 1, "year": 2026, "round": 1, "circuitId": 100, "name": "Race 1", "date": "2026-01-01"},
        {"raceId": 2, "year": 2026, "round": 2, "circuitId": 100, "name": "Race 2", "date": "2026-01-08"},
    ]).to_csv(raw_dir / "races.csv", index=False)
    pd.DataFrame([{"driverId": 1, "driverRef": "driver1", "code": "D1", "forename": "First",
                   "surname": "Last", "dob": "1990-01-01", "nationality": "Nat"}]).to_csv(
        raw_dir / "drivers.csv", index=False)
    pd.DataFrame([{"constructorId": 10, "constructorRef": "team10", "name": "Team 10",
                   "nationality": "Nat"}]).to_csv(raw_dir / "constructors.csv", index=False)
    pd.DataFrame([{"circuitId": 100, "circuitRef": "circA", "name": "Circuit A",
                   "location": "Loc", "country": "Country", "lat": 1.0, "lng": 1.0, "alt": 1}]).to_csv(
        raw_dir / "circuits.csv", index=False)
    pd.DataFrame([{"raceId": 1, "driverId": 1, "constructorId": 10, "points": 25,
                   "position": 1, "positionText": "1", "wins": 1}]).to_csv(
        raw_dir / "driver_standings.csv", index=False)
    pd.DataFrame([{"raceId": 1, "constructorId": 10, "points": 25, "position": 1,
                   "positionText": "1", "wins": 1}]).to_csv(
        raw_dir / "constructor_standings.csv", index=False)
    pd.DataFrame(columns=["raceId", "race_precip_mm", "race_temp_c",
                          "quali_precip_mm", "conditions_changed"]).to_csv(
        raw_dir / "race_weather.csv", index=False)

    processed_dir = root / "processed"
    processed_dir.mkdir()
    master_path = processed_dir / "master_dataset.parquet"
    pd.DataFrame([_historical_row()]).to_parquet(master_path)

    interim_dir = root / "interim"
    interim_dir.mkdir()
    qualifying_path = interim_dir / "qualifying.parquet"
    pd.DataFrame([
        {"raceId": 2, "driverId": 1, "constructorId": 10, "number": 1, "position": 1,
         "q1": "1:19.5", "q2": "1:18.5", "q3": "1:17.5"},
    ]).to_parquet(qualifying_path)

    provenance_settings = settings.model_copy(update={
        "raw_data_dir": raw_dir, "master_dataset_path": master_path,
        "qualifying_interim_path": qualifying_path,
        "weather_csv_path": raw_dir / "race_weather.csv",
    })
    with TestClient(create_app(provenance_settings)) as c:
        yield c


def test_post_predict_provenance_round_trips_to_real_state(provenance_client):
    """Every field in the response's
    provenance block must be reconstructable from already-known state —
    not just internally consistent with itself. Each field is recomputed
    HERE, directly, from the same public state a real client could reach
    (the served model's own training schema, the real files' mtimes),
    rather than by calling app.upcoming_prediction_service's own private
    helpers — calling the same function that produced the value would only
    prove it's consistent with itself, not that it's reconstructable."""
    resp = provenance_client.post(f"{API_V1_PREFIX}/predict", json={"year": 2026, "round": 2})
    assert resp.status_code == 200
    body = resp.json()
    provenance = body["provenance"]

    app_state = provenance_client.app.state
    model_info = app_state.model_info
    assert provenance["model_version"] == model_info.version
    assert provenance["model_alias"] == model_info.alias

    expected_schema_hash = hashlib.sha256(
        json.dumps(training_schema(app_state.model), sort_keys=True).encode()
    ).hexdigest()[:16]
    assert provenance["feature_schema_version"] == expected_schema_hash

    settings = app_state.settings
    watched_paths = (
        settings.master_dataset_path, settings.qualifying_interim_path, settings.weather_csv_path,
    )
    expected_mtime = max(p.stat().st_mtime for p in watched_paths if p.exists())
    expected_etl_snapshot = datetime.fromtimestamp(expected_mtime, tz=UTC).isoformat()
    assert provenance["etl_snapshot_version"] == expected_etl_snapshot
    assert provenance["data_as_of"] == expected_etl_snapshot     # no as_of override given

    materialized_at = datetime.fromisoformat(provenance["materialized_at"])
    predicted_at = datetime.fromisoformat(provenance["predicted_at"])
    assert materialized_at <= predicted_at

    assert provenance["qualifying_status"] == "complete"        # driver 1 has a real quali row
    assert provenance["completeness_status"] == body["materialization_status"] == "post_qualifying"


# ---------------------------------------------------------------------------
# API versioning — the versioned mount is the
# canonical, documented one; every test above already exercises the legacy
# unversioned alias, so its continued passing IS the back-compat regression
# check. These tests cover what's genuinely new: the versioned mount itself.
# ---------------------------------------------------------------------------

def test_versioned_health_matches_legacy(client):
    versioned = client.get(f"{API_V1_PREFIX}/health")
    legacy = client.get("/health")
    assert versioned.status_code == legacy.status_code == 200
    assert versioned.json() == legacy.json()


def test_versioned_prediction_matches_legacy_same_cache_entry(client):
    """Both mounts call the same closures over the same app.state — a cache
    hit on one path returns the SAME cached object (including prediction_id)
    as a hit via the other path, proving there's no second, drifted copy of
    server state per mount."""
    versioned = client.get(f"{API_V1_PREFIX}/predictions/{IN_WINDOW_RACE}")
    legacy = client.get(f"/predictions/{IN_WINDOW_RACE}")
    assert versioned.status_code == legacy.status_code == 200
    assert versioned.json() == legacy.json()


def test_versioned_error_paths_match_legacy(client):
    assert (client.get(f"{API_V1_PREFIX}/predictions/999999999").status_code
            == client.get("/predictions/999999999").status_code == 404)
    # POST /predict is real request handling now, not a 501 stub —
    # the invariant this test actually cares about (versioned and legacy
    # mounts behave identically) still holds for whatever status a bodiless
    # POST gets (422, from FastAPI's own request validation).
    assert (client.post(f"{API_V1_PREFIX}/predict").status_code
            == client.post("/predict").status_code == 422)


def test_openapi_schema_lists_only_versioned_paths(client):
    """The legacy back-compat mount is deliberately include_in_schema=False
    — /docs/OpenAPI should show exactly one contract per route, not two."""
    schema_paths = client.get("/openapi.json").json()["paths"]
    assert schema_paths, "expected at least one documented route"
    assert all(p.startswith(API_V1_PREFIX) for p in schema_paths), schema_paths
    assert "/health" not in schema_paths
    assert f"{API_V1_PREFIX}/health" in schema_paths
