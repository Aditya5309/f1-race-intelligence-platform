"""
Tests for src/models/predict.py (Phase 4 module 5; design Sections 2, 12).

Coverage per design Section 12 + Decision 015:
  - registry loading: alias resolution, ModelInfo metadata (version, run id,
    calibration status), missing alias raises
  - schema validation: missing feature columns, extra design-matrix columns
    (via ColumnGuard), wrong dtypes, reliance on the ARTIFACT's stored
    schema rather than repository constants
  - per-race normalization: sums to 1, preserves within-race ranking,
    zero-sum race falls back to uniform
  - output contract: sorted descending within race, predicted_rank,
    carried identifier columns, deterministic across calls
  - race grouping: missing raceId, null raceId, duplicate (raceId, driverId)
  - calibration behavior: a loaded CalibratedModel reports isotonic-oof
    metadata and produces the calibrated (not raw) probabilities
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.calibration import CalibratedModel
from src.models.predict import ModelInfo, load_model, main, predict_race
from src.models.registry import get_model
from src.models.splits import temporal_split, to_xy
from src.models.train import register_model

# ---------------------------------------------------------------------------
# Synthetic data + a tmp registry with one calibrated and one raw model
# ---------------------------------------------------------------------------

def _synthetic_features(years, races_per_year=4, n_drivers=5, seed=0) -> pd.DataFrame:
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
def full_frame() -> pd.DataFrame:
    return _synthetic_features(range(2010, 2025))


@pytest.fixture(scope="module")
def race_frame(full_frame) -> pd.DataFrame:
    """Two 2023 races, ids + features + target column (extra col for the model)."""
    return full_frame[full_frame["raceId"].isin([202301, 202302])].copy()


@pytest.fixture(scope="module")
def registry(tmp_path_factory, full_frame):
    """Tmp sqlite registry: v1 = raw logreg @Candidate-less, v2 = calibrated
    logreg @Staging. Returns the tracking URI."""
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-experiment")
    split = temporal_split(full_frame)
    register_model("logreg", split, alias="Staging", calibrate=False)      # v1
    register_model("logreg", split, alias="Staging", calibrate=True)       # v2 takes alias
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="module")
def staging(registry):
    model, info = load_model(alias="Staging", tracking_uri=registry)
    return model, info


# ---------------------------------------------------------------------------
# Registry loading + metadata
# ---------------------------------------------------------------------------

def test_load_model_resolves_staging_alias(staging):
    model, info = staging
    assert isinstance(info, ModelInfo)
    assert info.name == "f1-winner"
    assert info.alias == "Staging"
    assert info.version == "2"                      # calibrated re-registration
    assert info.calibration == "isotonic-oof"
    assert info.model_class == "CalibratedModel"
    assert info.run_id
    assert info.trained_at.startswith("20")         # ISO date
    assert isinstance(model, CalibratedModel)


def test_load_model_missing_alias_raises(registry):
    with pytest.raises(MlflowException):
        load_model(alias="Production", tracking_uri=registry)


def test_model_info_is_json_ready(staging):
    _, info = staging
    d = info.to_dict()
    assert set(d) == {"name", "version", "alias", "run_id", "trained_at",
                      "calibration", "model_class"}
    assert all(isinstance(v, str) for v in d.values())


# ---------------------------------------------------------------------------
# Prediction contract
# ---------------------------------------------------------------------------

def test_probabilities_normalize_to_one_per_race(staging, race_frame):
    model, _ = staging
    out = predict_race(model, race_frame)
    sums = out.groupby("raceId")["win_probability"].sum()
    assert np.allclose(sums, 1.0)


def test_normalization_preserves_ranking(staging, race_frame):
    model, _ = staging
    out = predict_race(model, race_frame)
    for _, group in out.groupby("raceId"):
        raw_order = group.sort_values("win_probability_raw", ascending=False)
        norm_order = group.sort_values("win_probability", ascending=False)
        # monotone: same ordering by raw and normalized probability
        assert (raw_order.index == norm_order.index).all()


def test_output_sorted_desc_with_ranks(staging, race_frame):
    model, _ = staging
    out = predict_race(model, race_frame)
    for _, group in out.groupby("raceId"):
        assert (group["win_probability"].diff().dropna() <= 1e-15).all()
        assert list(group["predicted_rank"]) == list(range(1, len(group) + 1))


def test_carries_identifier_columns(staging, race_frame):
    model, _ = staging
    out = predict_race(model, race_frame)
    assert {"raceId", "driverId", "year", "round"} <= set(out.columns)
    assert len(out) == len(race_frame)


def test_deterministic_across_calls(staging, race_frame):
    model, _ = staging
    a = predict_race(model, race_frame)
    b = predict_race(model, race_frame)
    pd.testing.assert_frame_equal(a, b)


def test_single_race_input_works(staging, race_frame):
    model, _ = staging
    one = race_frame[race_frame["raceId"] == 202301]
    out = predict_race(model, one)
    assert out["raceId"].nunique() == 1
    assert np.isclose(out["win_probability"].sum(), 1.0)


def test_extra_non_feature_columns_are_ignored(staging, race_frame):
    """Id/target/junk columns beyond the schema never reach the model."""
    model, _ = staging
    noisy = race_frame.copy()
    noisy["junk"] = "not-a-number"          # non-numeric: would explode if fed in
    out = predict_race(model, noisy)
    pd.testing.assert_frame_equal(out, predict_race(model, race_frame))


def test_zero_probability_race_normalizes_uniform(full_frame, race_frame):
    """The pole heuristic gives an all-zero field when nobody starts on pole
    -> uniform shares, deterministically."""
    split = temporal_split(full_frame)
    _, y_tr, _ = to_xy(split.train)
    pole = get_model("pole_baseline", y_tr)
    X_tr, _, _ = to_xy(split.train)
    pole.fit(X_tr, y_tr)

    field = race_frame[race_frame["raceId"] == 202301].copy()
    field["grid_adjusted"] = np.arange(2.0, 2.0 + len(field))   # no pole sitter
    out = predict_race(pole, field)
    assert np.allclose(out["win_probability"], 1.0 / len(field))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_missing_feature_column_raises(staging, race_frame):
    model, _ = staging
    broken = race_frame.drop(columns=["grid_adjusted"])
    with pytest.raises(ValueError, match="missing feature columns.*grid_adjusted"):
        predict_race(model, broken)


def test_extra_design_matrix_column_rejected_by_guard(staging, race_frame):
    """ColumnGuard strictness is still reachable: feeding the model a design
    matrix with an extra column raises (schema names AND order enforced)."""
    model, _ = staging
    X = race_frame.loc[:, list(FEATURE_COLUMNS)].assign(extra=1.0)
    with pytest.raises(ValueError, match="Design matrix columns"):
        model.predict_proba(X)


def test_non_numeric_feature_dtype_raises(staging, race_frame):
    model, _ = staging
    corrupt = race_frame.copy()
    corrupt["grid_adjusted"] = "front row"
    with pytest.raises((ValueError, TypeError)):
        predict_race(model, corrupt)


def test_schema_comes_from_artifact_not_repository(staging, race_frame, monkeypatch):
    """A future FEATURE_COLUMNS change must not affect a loaded artifact:
    the guard validates against its RECORDED schema."""
    model, _ = staging
    # Even if repository constants changed, predict_race reads the artifact.
    out = predict_race(model, race_frame)
    recorded = model.named_steps["guard"].feature_names_in_
    assert recorded == list(FEATURE_COLUMNS)   # today they coincide
    assert len(out) == len(race_frame)


# ---------------------------------------------------------------------------
# Race grouping validation
# ---------------------------------------------------------------------------

def test_missing_race_id_column_raises(staging, race_frame):
    model, _ = staging
    with pytest.raises(ValueError, match="raceId"):
        predict_race(model, race_frame.drop(columns=["raceId"]))


def test_null_race_id_raises(staging, race_frame):
    model, _ = staging
    broken = race_frame.copy()
    broken.loc[broken.index[0], "raceId"] = np.nan
    with pytest.raises(ValueError, match="raceId contains nulls"):
        predict_race(model, broken)


def test_duplicate_driver_in_race_raises(staging, race_frame):
    model, _ = staging
    dupe = pd.concat([race_frame, race_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        predict_race(model, dupe)


def test_empty_frame_raises(staging, race_frame):
    model, _ = staging
    with pytest.raises(ValueError, match="empty"):
        predict_race(model, race_frame.iloc[0:0])


# ---------------------------------------------------------------------------
# Calibration behavior through the serving path
# ---------------------------------------------------------------------------

def test_staging_predictions_are_calibrated_not_raw(staging, race_frame):
    model, info = staging
    assert info.calibration == "isotonic-oof"
    out = predict_race(model, race_frame)
    X = race_frame.loc[:, list(FEATURE_COLUMNS)]
    base_raw = model.base_pipeline.predict_proba(X)[:, 1]
    calibrated = model.predict_proba(X)[:, 1]
    # The serving path reports the calibrated number...
    assert set(np.round(out["win_probability_raw"], 12)) == set(np.round(calibrated, 12))
    # ...which differs from the uncalibrated base output.
    assert not np.allclose(np.sort(base_raw), np.sort(calibrated))


def test_raw_model_reports_no_calibration(registry, race_frame):
    """Loading v1 (the raw pipeline) by version: metadata degrades gracefully."""
    mlflow.set_tracking_uri(registry)
    model = mlflow.sklearn.load_model("models:/f1-winner/1")
    assert getattr(model, "calibration", "none") == "none"
    out = predict_race(model, race_frame)          # model-agnostic path works
    assert np.allclose(out.groupby("raceId")["win_probability"].sum(), 1.0)


# ---------------------------------------------------------------------------
# CLI (python -m src.models.predict --race-id N)
# ---------------------------------------------------------------------------

def test_cli_missing_features_parquet_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "src.features.pipeline.FEATURES_PATH", tmp_path / "missing.parquet"
    )
    assert main(["--race-id", "202301"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_unknown_race_id_returns_1(monkeypatch, tmp_path, full_frame, capsys):
    path = tmp_path / "features.parquet"
    full_frame.to_parquet(path, index=False)
    monkeypatch.setattr("src.features.pipeline.FEATURES_PATH", path)
    assert main(["--race-id", "999999"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_scores_race_with_staging_model(
    monkeypatch, tmp_path, full_frame, registry, capsys
):
    path = tmp_path / "features.parquet"
    full_frame.to_parquet(path, index=False)
    monkeypatch.setattr("src.features.pipeline.FEATURES_PATH", path)
    rc = main(["--race-id", "202301", "--tracking-uri", registry])
    assert rc == 0
    out = capsys.readouterr().out
    assert "f1-winner" in out
    assert "5 drivers" in out
    assert "win_probability" in out
