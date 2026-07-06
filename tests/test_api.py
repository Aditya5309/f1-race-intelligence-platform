"""
Tests for app/api.py (Decision 016; reports/application_design.md §15).

TestClient against create_app() with an isolated stack: tmp sqlite registry
holding a calibrated model + a tiny synthetic features parquet, wired in via
Settings. Covers: health (ok + degraded), model metadata, race listing +
year filter + holdout exclusion, prediction happy path (normalization,
ordering, winner comparison, prediction_id), 404 unknown race, 409
forward-holdout guard, debug endpoint gating, reserved POST /predict (501),
and cache behavior across calls.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
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
    """Registry (calibrated model @Staging) + features parquet + Settings."""
    root = tmp_path_factory.mktemp("serving")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    # 2010–2024 in-window plus 2025 forward-holdout rows for the guard tests.
    frame = _synthetic_features(range(2010, 2026))
    features_path = root / "features.parquet"
    frame.to_parquet(features_path)

    mlflow.set_tracking_uri(uri)
    set_tmp_experiment("test-experiment", root)
    split = temporal_split(frame)
    register_model("logreg", split, alias="Staging", calibrate=True)
    mlflow.set_tracking_uri(None)

    settings = Settings(
        tracking_uri=uri,
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
HOLDOUT_RACE = 202501         # 2025 round 1 (forward holdout)


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
        tracking_uri=f"sqlite:///{tmp_path / 'empty.db'}",
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

def test_races_listed_and_holdout_excluded(client, serving_stack):
    _, frame = serving_stack
    body = client.get("/races").json()
    years = {r["year"] for r in body["races"]}
    assert max(years) == 2024                      # 2025 rows exist but are hidden
    in_window = frame[frame["year"] <= 2024]
    assert len(body["races"]) == in_window["raceId"].nunique()
    first = body["races"][0]
    assert set(first) == {"race_id", "year", "round", "n_drivers"}
    assert first["n_drivers"] == 5


def test_races_year_filter(client):
    body = client.get("/races", params={"year": 2023}).json()
    assert body["races"]
    assert all(r["year"] == 2023 for r in body["races"])


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


def test_prediction_forward_holdout_409(client):
    resp = client.get(f"/predictions/{HOLDOUT_RACE}")
    assert resp.status_code == 409
    assert "forward holdout" in resp.json()["detail"]


def test_prediction_cache_returns_identical_body(client):
    a = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    b = client.get(f"/predictions/{IN_WINDOW_RACE}").json()
    # Cached: everything identical including the first call's prediction_id.
    assert a == b


# ---------------------------------------------------------------------------
# Debug endpoint gating (design §5 amendment)
# ---------------------------------------------------------------------------

def test_debug_features_hidden_by_default(client):
    assert client.get(f"/debug/features/{IN_WINDOW_RACE}").status_code == 404


def test_debug_features_when_enabled(debug_client):
    body = debug_client.get(f"/debug/features/{IN_WINDOW_RACE}").json()
    assert body["race_id"] == IN_WINDOW_RACE
    assert body["feature_names"] == list(FEATURE_COLUMNS)
    assert len(body["rows"]) == 5
    row = body["rows"][0]
    assert set(row["features"]) == set(FEATURE_COLUMNS)
    assert all(v is None or isinstance(v, float) for v in row["features"].values())


def test_debug_features_respects_holdout(debug_client):
    assert debug_client.get(f"/debug/features/{HOLDOUT_RACE}").status_code == 409


# ---------------------------------------------------------------------------
# Reserved POST /predict (design §5 amendment)
# ---------------------------------------------------------------------------

def test_post_predict_reserved_501(client):
    resp = client.post("/predict")
    assert resp.status_code == 501
    assert "Phase 8" in resp.json()["detail"]
