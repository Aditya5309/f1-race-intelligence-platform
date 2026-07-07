"""
Tests for src/models/calibration.py (Decision 015; design Sections 5, 9.5).

Coverage:
  - OOF protocol: pairs come from held-out fold seasons only; season_folds'
    own guard keeps val/test/forward-holdout years out of the calibrator
  - CalibratedModel mechanics: probability validity/clipping, monotonicity
    (within-race ranking preserved vs the base model), predict threshold,
    named_steps delegation (training_schema works unchanged), fit() refusal
  - Calibration effect: on a class-weight-inflated base model, calibrated
    probabilities have lower ECE and log-loss than raw ones while per-race
    top-1 is unchanged
  - MLflow serialization round trip incl. register_model(calibrate=True) +
    alias resolution; CLI flag validation
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.calibration import (
    PROBABILITY_EPS,
    CalibratedModel,
    fit_calibrated_model,
    fit_isotonic,
    oof_predictions,
)
from src.models.evaluate import (
    expected_calibration_error,
    log_loss_score,
    top1_accuracy,
)
from src.models.registry import training_schema
from src.models.splits import temporal_split, to_xy
from src.models.train import main, register_model
from tests.conftest import set_tmp_experiment

# ---------------------------------------------------------------------------
# Synthetic frame: pole (grid_adjusted == 1) wins with high but not perfect
# probability, so the class-weighted base model has real calibration error.
# ---------------------------------------------------------------------------

def _synthetic_features(years, races_per_year=6, n_drivers=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for rnd in range(1, races_per_year + 1):
            race_id = year * 100 + rnd   # unique across frames built separately
            grid = rng.permutation(n_drivers) + 1
            # pole wins 70% of the time; otherwise a random non-pole driver
            pole_wins = rng.random() < 0.7
            winner_grid = 1 if pole_wins else int(rng.integers(2, n_drivers + 1))
            for driver in range(n_drivers):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row["grid_adjusted"] = float(grid[driver])
                row["grid_position_norm"] = float(grid[driver]) / n_drivers
                row.update({
                    "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                    "circuitId": 1, "year": year, "round": rnd,
                    TARGET_COLUMN: int(grid[driver] == winner_grid),
                })
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return _synthetic_features(range(2010, 2022))


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    return _synthetic_features(range(2022, 2024), seed=99)


@pytest.fixture(scope="module")
def calibrated(train_df) -> CalibratedModel:
    return fit_calibrated_model("logreg", train_df, n_folds=2)


@pytest.fixture()
def tmp_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    set_tmp_experiment("test-experiment", tmp_path)
    yield
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------------------------
# OOF protocol
# ---------------------------------------------------------------------------

def test_oof_predictions_cover_fold_val_seasons_only(train_df):
    oof_prob, oof_y = oof_predictions("logreg", train_df, n_folds=2)
    # 2 folds -> the last 2 training seasons are held out once each
    expected_rows = len(train_df[train_df["year"].isin((2020, 2021))])
    assert len(oof_prob) == len(oof_y) == expected_rows
    assert ((oof_prob >= 0) & (oof_prob <= 1)).all()
    assert set(np.unique(oof_y)) <= {0, 1}


def test_oof_rejects_frames_with_out_of_window_years(train_df, val_df):
    # season_folds' guard propagates: the calibrator can never be fed
    # validation/test/forward-holdout rows.
    with pytest.raises(ValueError):
        oof_predictions("logreg", pd.concat([train_df, val_df]), n_folds=2)


def test_fit_isotonic_is_monotone():
    rng = np.random.default_rng(0)
    prob = rng.random(500)
    y = (rng.random(500) < prob**2).astype(int)   # overconfident base
    iso = fit_isotonic(prob, y)
    grid = np.linspace(0, 1, 101)
    mapped = iso.predict(grid)
    assert (np.diff(mapped) >= -1e-12).all()


# ---------------------------------------------------------------------------
# CalibratedModel mechanics
# ---------------------------------------------------------------------------

def test_calibrated_probabilities_valid_and_clipped(calibrated, val_df):
    X, _, _ = to_xy(val_df)
    proba = calibrated.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba[:, 1] >= PROBABILITY_EPS).all()
    assert (proba[:, 1] <= 1.0 - PROBABILITY_EPS).all()


def test_calibration_preserves_within_race_ranking(calibrated, val_df):
    X, _, _ = to_xy(val_df)
    raw = calibrated.base_pipeline.predict_proba(X)[:, 1]
    cal = calibrated.predict_proba(X)[:, 1]
    # Isotonic is monotone non-decreasing: a strictly higher raw probability
    # can never map to a lower calibrated probability (ties are allowed).
    order = np.argsort(raw)
    assert (np.diff(cal[order]) >= -1e-12).all()


def test_predict_thresholds_at_half(calibrated, val_df):
    X, _, _ = to_xy(val_df)
    pred = calibrated.predict(X)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == (calibrated.predict_proba(X)[:, 1] >= 0.5)).all()


def test_named_steps_delegation_keeps_training_schema_working(calibrated):
    schema = training_schema(calibrated)
    assert schema["feature_names"] == list(FEATURE_COLUMNS)
    assert schema["n_features"] == len(FEATURE_COLUMNS)


def test_columnguard_still_enforced_through_wrapper(calibrated, val_df):
    X, _, _ = to_xy(val_df)
    with pytest.raises(ValueError, match="Design matrix columns"):
        calibrated.predict_proba(X.drop(columns=["grid_adjusted"]))
    with pytest.raises(ValueError, match="Design matrix columns"):
        calibrated.predict_proba(X.assign(extra_col=1.0))


def test_wrapper_refuses_fit(calibrated, val_df):
    X, y, _ = to_xy(val_df)
    with pytest.raises(NotImplementedError, match="fit_calibrated_model"):
        calibrated.fit(X, y)


def test_calibration_marker(calibrated):
    assert calibrated.calibration == "isotonic-oof"


def test_fit_df_defaults_to_train_but_accepts_refit_frame(train_df, val_df):
    # Production-refit path: base fit on train+val, calibrator still train-OOF.
    refit = fit_calibrated_model(
        "logreg", train_df, fit_df=pd.concat([train_df, val_df]), n_folds=2,
    )
    X, _, _ = to_xy(val_df)
    proba = refit.predict_proba(X)[:, 1]
    assert ((proba > 0) & (proba < 1)).all()


# ---------------------------------------------------------------------------
# Calibration effect (the reason this module exists)
# ---------------------------------------------------------------------------

def test_calibration_improves_probability_quality_without_reordering(
    calibrated, train_df, val_df,
):
    X, y, races = to_xy(val_df)
    raw = calibrated.base_pipeline.predict_proba(X)[:, 1]
    cal = calibrated.predict_proba(X)[:, 1]

    assert expected_calibration_error(y, cal) < expected_calibration_error(y, raw)
    assert log_loss_score(np.asarray(y), cal) < log_loss_score(np.asarray(y), raw)
    # Isotonic is monotone: it can never REORDER drivers, so top-1 can never
    # improve. It CAN lose races to tie plateaus under the pessimistic tie
    # policy (evaluate.py) — the exact-equality claim on the real validation
    # split is verified at execution time, not asserted on synthetic data
    # whose weak base model produces many plateau collisions.
    assert top1_accuracy(y, cal, races) <= top1_accuracy(y, raw, races)


def test_deterministic(train_df, val_df):
    X, _, _ = to_xy(val_df)
    a = fit_calibrated_model("logreg", train_df, n_folds=2).predict_proba(X)
    b = fit_calibrated_model("logreg", train_df, n_folds=2).predict_proba(X)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Serialization and registration
# ---------------------------------------------------------------------------

def test_mlflow_round_trip(tmp_mlflow, calibrated, val_df):
    X, _, _ = to_xy(val_df)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(calibrated, name="model")
    loaded = mlflow.sklearn.load_model(info.model_uri)
    assert isinstance(loaded, CalibratedModel)
    assert loaded.calibration == "isotonic-oof"
    assert np.array_equal(loaded.predict_proba(X), calibrated.predict_proba(X))


def test_register_model_calibrated_sets_alias_and_tag(tmp_mlflow, tmp_path, train_df, val_df):
    split = temporal_split(
        pd.concat([train_df, val_df, _synthetic_features([2024], seed=5)])
    )
    version = register_model("logreg", split, alias="Staging", calibrate=True,
                             bundle_root=tmp_path / "bundle")
    client = mlflow.MlflowClient()
    resolved = client.get_model_version_by_alias("f1-winner", "Staging")
    assert str(resolved.version) == str(version)
    run = client.get_run(resolved.run_id)
    assert run.data.tags["calibration"] == "isotonic-oof"

    loaded = mlflow.sklearn.load_model("models:/f1-winner@Staging")
    assert isinstance(loaded, CalibratedModel)


def test_cli_rejects_calibrate_without_register():
    with pytest.raises(SystemExit) as excinfo:
        main(["--model", "logreg", "--calibrate"])
    assert excinfo.value.code == 2
