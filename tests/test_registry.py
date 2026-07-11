"""
Tests for src/models/registry.py and src/features/metadata.py
(Phase 4 module 2 — Decision 012/013).

Design-doc Section 12 requirements for test_registry:
  - every zoo entry builds a fit-able sklearn pipeline on synthetic data
  - design-matrix columns == FEATURE_COLUMNS (Section 11.1 ColumnGuard)
  - pole-baseline heuristic produces valid probabilities
  - class weights computed from data, not hardcoded

Plus Decision-013 metadata integrity (single source of truth in
src/features/metadata.py).
"""

import sys

import numpy as np
import pandas as pd
import pytest

from src.features.metadata import (
    ERA_SENSITIVE_FEATURES,
    EXPERIMENTAL_FEATURES,
    FEATURE_CLASSIFICATION,
    FEATURE_GROUPS,
    STABLE_FEATURES,
    features_in_class,
)
from src.features.pipeline import FEATURE_COLUMNS
from src.models.registry import (
    MODEL_ZOO,
    ColumnGuard,
    PoleSitterBaseline,
    compute_scale_pos_weight,
    get_model,
    training_schema,
)

# ---------------------------------------------------------------------------
# Synthetic training data — realistic dtypes: floats with informative NaNs,
# booleans, ~5% positive rate, 2 drivers x 100 races.
# ---------------------------------------------------------------------------

def _training_data(n_rows: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS)
    )
    # Informative NaNs in the columns that carry them in real data.
    for col in ("q3_sec", "driver_circuit_avg_finish", "driver_standing_position_prev",
                "driver_wins_last_5"):
        X.loc[rng.random(n_rows) < 0.4, col] = np.nan
    # Boolean features as real booleans; grid columns as plausible values.
    X["reached_q2"] = rng.random(n_rows) < 0.7
    X["reached_q3"] = rng.random(n_rows) < 0.5
    X["pit_lane_start"] = rng.random(n_rows) < 0.05
    X["grid_adjusted"] = rng.integers(1, 21, n_rows).astype(float)
    X["grid_position_norm"] = X["grid_adjusted"] / 20.0
    y = pd.Series((rng.random(n_rows) < 0.05).astype(int), name="winner")
    y.iloc[:5] = 1   # guarantee both classes
    return X, y


# ---------------------------------------------------------------------------
# Zoo entries build, fit, and predict valid probabilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODEL_ZOO))
def test_zoo_entry_fits_and_predicts(name):
    X, y = _training_data()
    pipeline = get_model(name, y)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_zoo_has_exactly_the_designed_candidates():
    assert sorted(MODEL_ZOO) == [
        "lightgbm", "logreg", "pole_baseline", "random_forest", "xgboost",
    ]


def test_unknown_model_raises():
    _, y = _training_data()
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("catboost", y)   # not in the approved design (Decision 012)


def test_tuned_candidates_declare_distributions():
    for name in ("xgboost", "lightgbm", "logreg", "random_forest"):
        assert MODEL_ZOO[name].param_distributions, name
    assert MODEL_ZOO["pole_baseline"].param_distributions == {}


# ---------------------------------------------------------------------------
# Phase 4 Tranche D Item 1a — xgboost/lightgbm are imported lazily, only by
# the specific candidate that needs them. Serving a non-boosted-trees model
# (the currently-registered logreg) must not require either installed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blocked", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("name", ["logreg", "random_forest", "pole_baseline"])
def test_get_model_unaffected_by_blocking_unrelated_booster(monkeypatch, blocked, name):
    """sys.modules[blocked] = None makes `import blocked` raise ImportError —
    the standard way to simulate "not installed" without actually
    uninstalling anything. Building a candidate that never imports xgboost/
    lightgbm must succeed regardless."""
    monkeypatch.setitem(sys.modules, blocked, None)
    X, y = _training_data()
    pipeline = get_model(name, y)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)


@pytest.mark.parametrize("blocked,candidate", [("xgboost", "xgboost"), ("lightgbm", "lightgbm")])
def test_get_model_raises_import_error_when_its_own_booster_is_blocked(monkeypatch, blocked, candidate):
    """The flip side: actually requesting xgboost/lightgbm still needs it
    installed — a clean ImportError, not a confusing failure elsewhere."""
    monkeypatch.setitem(sys.modules, blocked, None)
    _, y = _training_data()
    with pytest.raises(ImportError):
        get_model(candidate, y)


# ---------------------------------------------------------------------------
# Section 11.1 — ColumnGuard: design matrix must be exactly FEATURE_COLUMNS
# ---------------------------------------------------------------------------

def test_guard_rejects_missing_column():
    X, y = _training_data()
    with pytest.raises(ValueError, match="grid_adjusted"):
        ColumnGuard().fit(X.drop(columns=["grid_adjusted"]), y)


def test_guard_rejects_extra_column():
    X, y = _training_data()
    poisoned = X.assign(position=1)   # post-race outcome column
    with pytest.raises(ValueError, match="position"):
        ColumnGuard().fit(poisoned, y)


def test_guard_rejects_reordered_columns():
    X, y = _training_data()
    reordered = X[list(X.columns[::-1])]
    with pytest.raises(ValueError, match="FEATURE_COLUMNS"):
        ColumnGuard().fit(reordered, y)


def test_guard_rejects_raw_arrays():
    X, y = _training_data()
    with pytest.raises(TypeError, match="DataFrame"):
        ColumnGuard().fit(X.to_numpy(), y)


def test_guard_checks_at_predict_time_too():
    X, y = _training_data()
    pipeline = get_model("xgboost", y).fit(X, y)
    with pytest.raises(ValueError):
        pipeline.predict_proba(X.drop(columns=["q1_sec"]))


def test_guard_normalizes_dtypes():
    X, y = _training_data()
    out = ColumnGuard().fit(X, y).transform(X)
    assert (out.dtypes == np.float64).all()
    # Booleans became 0/1 floats; NaNs survived the cast.
    assert set(out["reached_q2"].unique()) <= {0.0, 1.0}
    assert out["q3_sec"].isna().sum() == X["q3_sec"].isna().sum()


# ---------------------------------------------------------------------------
# Pole-sitter baseline
# ---------------------------------------------------------------------------

def test_pole_baseline_probabilities():
    X, y = _training_data()
    X.loc[X.index[:3], "grid_adjusted"] = 1.0
    X.loc[X.index[3:], "grid_adjusted"] = 5.0
    model = PoleSitterBaseline().fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert (proba[:3] == 1.0).all()
    assert (proba[3:] == 0.0).all()


def test_pole_baseline_through_pipeline():
    X, y = _training_data()
    pipeline = get_model("pole_baseline", y).fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    # Guard casts to float; heuristic must still key on grid_adjusted == 1.
    assert set(np.unique(proba)) <= {0.0, 1.0}
    assert (proba == (X["grid_adjusted"] == 1).astype(float).to_numpy()).all()


# ---------------------------------------------------------------------------
# Section 5 — class weighting computed from data, never hardcoded
# ---------------------------------------------------------------------------

def test_scale_pos_weight_computed_from_target():
    y_19 = pd.Series([1] + [0] * 19)
    y_9 = pd.Series([1] + [0] * 9)
    assert compute_scale_pos_weight(y_19) == pytest.approx(19.0)
    assert compute_scale_pos_weight(y_9) == pytest.approx(9.0)
    with pytest.raises(ValueError, match="no positive"):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_boosters_receive_data_derived_weight(name):
    X, y = _training_data()
    expected = compute_scale_pos_weight(y)
    pipeline = get_model(name, y)
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_sklearn_models_use_balanced_class_weight():
    _, y = _training_data()
    for name in ("logreg", "random_forest"):
        model = get_model(name, y).named_steps["model"]
        assert model.get_params()["class_weight"] == "balanced"


def test_logreg_keeps_missingness_visible():
    # The imputer must add missing-indicator columns so "no history" stays a
    # signal (design Section 3; domain_knowledge Section 2).
    _, y = _training_data()
    imputer = get_model("logreg", y).named_steps["imputer"]
    assert imputer.get_params()["add_indicator"] is True


# ---------------------------------------------------------------------------
# Training-schema capture — inference validates against the MODEL's schema
# ---------------------------------------------------------------------------

def test_guard_records_training_schema():
    X, y = _training_data()
    guard = ColumnGuard().fit(X, y)
    schema = guard.schema_dict()
    assert schema["n_features"] == len(FEATURE_COLUMNS)
    assert schema["feature_names"] == list(FEATURE_COLUMNS)
    assert schema["feature_dtypes"]["reached_q2"] == "bool"


def test_transform_validates_against_fitted_schema_not_repo_state(monkeypatch):
    # The core guarantee: a fitted (serialized) model keeps validating against
    # what it was TRAINED on, even if the repository's FEATURE_COLUMNS evolves.
    import src.models.registry as registry_module

    X, y = _training_data()
    guard = ColumnGuard().fit(X, y)

    # Simulate a future v2 feature set landing in the repo.
    monkeypatch.setattr(
        registry_module, "FEATURE_COLUMNS", tuple(FEATURE_COLUMNS) + ("new_v2_feature",)
    )
    # The already-fitted guard still accepts its training schema...
    out = guard.transform(X)
    assert list(out.columns) == list(FEATURE_COLUMNS)
    # ...and still rejects input matching the NEW repo state but not its own.
    with pytest.raises(ValueError, match="training schema"):
        guard.transform(X.assign(new_v2_feature=0.0))
    # A FRESH guard, by contrast, now demands the new repo contract.
    with pytest.raises(ValueError, match="FEATURE_COLUMNS"):
        ColumnGuard().fit(X, y)


def test_guard_transform_before_fit_raises():
    X, _ = _training_data()
    with pytest.raises(RuntimeError, match="before fit"):
        ColumnGuard().transform(X)


def test_training_schema_from_fitted_pipeline():
    X, y = _training_data()
    pipeline = get_model("lightgbm", y).fit(X, y)
    schema = training_schema(pipeline)
    assert schema["feature_names"] == list(FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="guard"):
        training_schema(__import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
            [("model", PoleSitterBaseline())]
        ))


# ---------------------------------------------------------------------------
# ModelSpec descriptive metadata
# ---------------------------------------------------------------------------

def test_model_spec_metadata_values():
    assert MODEL_ZOO["pole_baseline"].family == "heuristic"
    assert MODEL_ZOO["logreg"].family == "linear"
    assert MODEL_ZOO["random_forest"].family == "bagged-trees"
    assert MODEL_ZOO["xgboost"].family == "boosted-trees"
    assert MODEL_ZOO["lightgbm"].family == "boosted-trees"
    # NaN policy matches the built pipelines: boosters have no imputer.
    for name in ("xgboost", "lightgbm"):
        assert MODEL_ZOO[name].handles_nan_natively
        assert "imputer" not in get_model(name, pd.Series([1, 0, 0])).named_steps
    for name in ("logreg", "random_forest"):
        assert not MODEL_ZOO[name].handles_nan_natively
        assert "imputer" in get_model(name, pd.Series([1, 0, 0])).named_steps
    # Scaling requirement matches the pipelines too.
    assert MODEL_ZOO["logreg"].requires_scaling
    assert "scaler" in get_model("logreg", pd.Series([1, 0, 0])).named_steps
    assert not MODEL_ZOO["xgboost"].requires_scaling


def test_model_spec_to_metadata_is_json_ready():
    import json
    for spec in MODEL_ZOO.values():
        meta = spec.to_metadata()
        json.dumps(meta)   # must not raise
        assert meta["name"] == spec.name
        assert meta["tunable"] == bool(spec.param_distributions)
        assert set(meta) == {
            "name", "description", "family", "handles_nan_natively",
            "requires_scaling", "explainability", "training_cost",
            "tunable", "tuned_params",
        }
    assert MODEL_ZOO["pole_baseline"].to_metadata()["tunable"] is False
    assert MODEL_ZOO["pole_baseline"].training_cost == "trivial"


# ---------------------------------------------------------------------------
# Decision 013 — feature metadata single source of truth
# ---------------------------------------------------------------------------

def test_classification_partitions_feature_columns():
    all_classified = set(STABLE_FEATURES) | set(ERA_SENSITIVE_FEATURES) | set(EXPERIMENTAL_FEATURES)
    assert all_classified == set(FEATURE_COLUMNS)
    assert len(STABLE_FEATURES) + len(ERA_SENSITIVE_FEATURES) + len(EXPERIMENTAL_FEATURES) \
        == len(FEATURE_COLUMNS)
    # Decision 013 counts (12, 12, 7) + Phase 4 Tranche A: grid_penalty_applied
    # (item 2) and qualifying_gap_to_teammate_current, qualifying_gap_to_teammate,
    # race_pace_delta_to_teammate (item 1) — all classified stable — + Tranche B
    # item 1: race_precip_mm, race_temp_c, quali_precip_mm, conditions_changed,
    # and item 2: driver_wet_dry_delta, constructor_wet_dry_delta — all
    # classified experimental (new, unvalidated signal).
    assert (len(STABLE_FEATURES), len(ERA_SENSITIVE_FEATURES), len(EXPERIMENTAL_FEATURES)) \
        == (16, 12, 13)


def test_classification_dict_consistent_with_tuples():
    assert set(FEATURE_CLASSIFICATION) == set(FEATURE_COLUMNS)
    for feature in STABLE_FEATURES:
        assert FEATURE_CLASSIFICATION[feature] == "stable"
    assert features_in_class("experimental") == EXPERIMENTAL_FEATURES
    with pytest.raises(ValueError, match="Unknown feature class"):
        features_in_class("volatile")


def test_groups_cover_feature_columns():
    grouped = {f for group in FEATURE_GROUPS.values() for f in group}
    assert grouped == set(FEATURE_COLUMNS)
    assert list(FEATURE_GROUPS) == [
        "qualifying", "driver_form", "constructor_form", "teammate_form",
        "circuit_history", "standings", "weather", "wet_form",
    ]
