"""
src/models/registry.py

Model zoo for Phase 4 (Decision 012; reports/model_development_design.md
Sections 2, 3, 7, 11.1).

Each candidate is a ModelSpec: name -> estimator factory (`build`) +
hyperparameter distributions for the stage-2 randomized search. Adding a
candidate model = one MODEL_ZOO entry; train.py never changes.

Design rules enforced here:
- Every pipeline starts with a ColumnGuard that asserts, at fit AND predict
  time, that the design matrix is exactly FEATURE_COLUMNS in canonical order
  (Section 11.1 — no identifier or post-race column can ever reach an
  estimator), then casts to plain float64 (nullable Float64 -> NaN-bearing
  float, booleans -> 0/1) so every estimator sees one uniform dtype.
- Class imbalance is handled by weighting computed FROM THE TRAINING TARGET
  at build time (`compute_scale_pos_weight`), never hardcoded (Section 5).
  LogReg/RF use class_weight='balanced' (equivalent, sklearn-native).
- Informative NaNs are preserved for the tree boosters (native NaN handling);
  LogReg gets a median imputer WITH missing-indicator flags so "no prior
  history" stays visible as a signal instead of dissolving into the median
  (Section 3 rationale).
- The pole-sitter heuristic (Section 3, candidate 0b) implements the same
  predict_proba interface as the real models so it flows through identical
  CV/evaluation/MLflow plumbing. It is the bar every model must beat
  (~50% per-race top-1 in-window, Section 14.1).
- 0a (always-negative dummy) is deliberately NOT a zoo entry: it exists to
  make a rhetorical point about row-level metrics, which evaluate.py's
  per-race metrics already make structurally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import FEATURE_COLUMNS

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

class ColumnGuard(BaseEstimator, TransformerMixin):
    """First step of every zoo pipeline: schema assertion + dtype normalization.

    At FIT time the guard validates against the feature pipeline's
    FEATURE_COLUMNS (the training-time contract) and RECORDS the schema it
    saw (`feature_names_in_`, `feature_dtypes_in_`). At TRANSFORM time —
    which includes every predict call on the serialized pipeline — it
    validates against its own RECORDED schema, not the repository state.
    A fitted model therefore carries its training schema with it: if
    FEATURE_COLUMNS later evolves (v2 features), old artifacts still
    validate serving input against what they were actually trained on.
    Names and order are enforced strictly; recorded dtypes are contract
    documentation (everything is cast to float64, so Float64-vs-float64
    differences are not errors).
    """

    def _check(self, X: pd.DataFrame, expected: list, contract: str) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ColumnGuard requires a pandas DataFrame so column names are "
                "verifiable — got a raw array."
            )
        if list(X.columns) != list(expected):
            raise ValueError(
                f"Design matrix columns do not match {contract} exactly "
                "(names and order). "
                f"Missing: {[c for c in expected if c not in X.columns]}; "
                f"unexpected: {[c for c in X.columns if c not in expected]}."
            )

    def fit(self, X: pd.DataFrame, y=None) -> ColumnGuard:
        self._check(X, list(FEATURE_COLUMNS), "FEATURE_COLUMNS")
        self.feature_names_in_ = list(X.columns)
        self.feature_dtypes_in_ = {col: str(dtype) for col, dtype in X.dtypes.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("ColumnGuard.transform called before fit.")
        self._check(X, self.feature_names_in_, "the model's training schema")
        # float64 everywhere: nullable Float64 -> NaN floats, bools -> 0/1.
        return X.astype(np.float64)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Pass feature names through (sklearn convention) so
        `pipeline[:-1].get_feature_names_out()` resolves post-preprocessing
        names for importance reporting."""
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("ColumnGuard has not been fitted.")
        return np.asarray(self.feature_names_in_, dtype=object)

    def schema_dict(self) -> dict:
        """The recorded training schema, JSON-ready for an MLflow artifact."""
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("ColumnGuard has not been fitted — no schema recorded.")
        return {
            "n_features": len(self.feature_names_in_),
            "feature_names": list(self.feature_names_in_),
            "feature_dtypes": dict(self.feature_dtypes_in_),
        }


def training_schema(pipeline: Pipeline) -> dict:
    """Extract the training schema from a fitted zoo pipeline's guard.

    train.py logs this as a JSON artifact next to the model; predict.py
    validates serving input against it (via the guard itself) without
    depending on repository constants.
    """
    guard = pipeline.named_steps.get("guard")
    if guard is None:
        raise ValueError("Pipeline has no 'guard' step — not a zoo-built pipeline.")
    return guard.schema_dict()


class PoleSitterBaseline(BaseEstimator, ClassifierMixin):
    """Heuristic baseline: P(win) = 1 if starting from pole, else 0.

    grid_adjusted == 1 identifies the pole sitter (pit-lane starts were
    remapped to field_size + 1 upstream, so 1 is unambiguous).
    """

    def fit(self, X: pd.DataFrame, y) -> PoleSitterBaseline:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pole = np.asarray(X["grid_adjusted"] == 1, dtype=float)
        return np.column_stack([1.0 - pole, pole])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1].round().astype(int)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """negatives/positives from the actual training target (never hardcoded)."""
    positives = float(np.sum(y))
    if positives == 0:
        raise ValueError("Training target has no positive rows — cannot weight classes.")
    return (len(y) - positives) / positives


# ---------------------------------------------------------------------------
# Zoo entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One candidate: build(y_train) -> ready-to-fit sklearn Pipeline.

    The descriptive fields are reusable metadata for MLflow run tags,
    documentation, and future dashboard components — export via
    `to_metadata()` (JSON-ready). They describe the RAW estimator; the
    built pipeline compensates where needed (e.g. logreg's pipeline
    contains the imputer/scaler its estimator requires).
    """
    name: str
    description: str
    family: str                    # "heuristic" | "linear" | "bagged-trees" | "boosted-trees"
    handles_nan_natively: bool     # estimator accepts NaN without imputation
    requires_scaling: bool         # estimator needs standardized inputs
    explainability: str            # how to explain it (design doc Section 10)
    training_cost: str             # "trivial" | "low" | "medium" — relative, this dataset
    build: Callable[[pd.Series], Pipeline]
    # Stage-2 randomized-search distributions, keyed by pipeline param path.
    # Empty for candidates that are not tuned (baseline, default-only models).
    param_distributions: dict = field(default_factory=dict)

    @property
    def tunable(self) -> bool:
        return bool(self.param_distributions)

    def to_metadata(self) -> dict:
        """JSON-ready descriptive metadata (no callables/distributions)."""
        return {
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "handles_nan_natively": self.handles_nan_natively,
            "requires_scaling": self.requires_scaling,
            "explainability": self.explainability,
            "training_cost": self.training_cost,
            "tunable": self.tunable,
            "tuned_params": sorted(self.param_distributions),
        }


def _build_pole_baseline(y_train: pd.Series) -> Pipeline:
    return Pipeline([
        ("guard", ColumnGuard()),
        ("model", PoleSitterBaseline()),
    ])


def _build_logreg(y_train: pd.Series) -> Pipeline:
    return Pipeline([
        ("guard", ColumnGuard()),
        # add_indicator=True: informative missingness (rookies, no Q3, debut
        # standings) survives imputation as explicit boolean flags.
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])


def _build_random_forest(y_train: pd.Series) -> Pipeline:
    return Pipeline([
        ("guard", ColumnGuard()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def _build_xgboost(y_train: pd.Series) -> Pipeline:
    # Imported here, not at module level: serving a non-boosted-trees model
    # (e.g. the registered logreg) must not require xgboost to be
    # installed at all — only actually building/training THIS candidate does.
    from xgboost import XGBClassifier

    return Pipeline([
        ("guard", ColumnGuard()),
        # No imputer: XGBoost handles NaN natively (missing-branch learning),
        # preserving informative missingness without extra columns.
        ("model", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=compute_scale_pos_weight(y_train),
            tree_method="hist", eval_metric="logloss",
            importance_type="gain",     # design Sections 8/10: gain, not split count
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])


def _build_lightgbm(y_train: pd.Series) -> Pipeline:
    # See _build_xgboost's comment — same reasoning for lightgbm.
    from lightgbm import LGBMClassifier

    return Pipeline([
        ("guard", ColumnGuard()),
        ("model", LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=compute_scale_pos_weight(y_train),
            importance_type="gain",     # design Sections 8/10: gain, not split count
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )),
    ])


# Design doc Section 7: distributions for the stage-2 randomized search,
# declared per candidate. Keys use sklearn pipeline param paths.
_XGBOOST_DISTRIBUTIONS = {
    "model__n_estimators": randint(150, 800),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__max_depth": randint(3, 8),
    "model__min_child_weight": randint(1, 8),
    "model__subsample": uniform(0.6, 0.4),          # [0.6, 1.0]
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__reg_alpha": loguniform(1e-4, 1.0),
    "model__reg_lambda": loguniform(1e-2, 10.0),
}

_LIGHTGBM_DISTRIBUTIONS = {
    "model__n_estimators": randint(150, 800),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__num_leaves": randint(15, 63),
    "model__min_child_samples": randint(5, 40),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__reg_alpha": loguniform(1e-4, 1.0),
    "model__reg_lambda": loguniform(1e-2, 10.0),
}

_LOGREG_DISTRIBUTIONS = {
    "model__C": loguniform(1e-3, 10.0),
}

_RANDOM_FOREST_DISTRIBUTIONS = {
    "model__n_estimators": randint(200, 800),
    "model__max_depth": randint(4, 16),
    "model__min_samples_leaf": randint(1, 10),
    "model__max_features": uniform(0.3, 0.7),       # [0.3, 1.0]
}


MODEL_ZOO: dict[str, ModelSpec] = {
    "pole_baseline": ModelSpec(
        name="pole_baseline",
        description="Heuristic: pole sitter wins (~50% per-race top-1 in-window). "
                    "The bar every trained model must beat (design Section 9.1).",
        family="heuristic",
        handles_nan_natively=True,     # reads grid_adjusted only; NaN -> not pole
        requires_scaling=False,
        explainability="deterministic rule (grid_adjusted == 1)",
        training_cost="trivial",
        build=_build_pole_baseline,
    ),
    "logreg": ModelSpec(
        name="logreg",
        description="Logistic Regression: linear baseline; median imputer with "
                    "missing-indicator flags + standard scaler; class_weight balanced.",
        family="linear",
        handles_nan_natively=False,    # pipeline supplies imputer + indicators
        requires_scaling=True,         # pipeline supplies StandardScaler
        explainability="coefficients (+ permutation importance)",
        training_cost="low",
        build=_build_logreg,
        param_distributions=_LOGREG_DISTRIBUTIONS,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        description="Random Forest: non-linear baseline, low tuning sensitivity; "
                    "imputer with missing indicators; class_weight balanced.",
        family="bagged-trees",
        handles_nan_natively=False,    # sklearn RF rejects NaN; pipeline imputes
        requires_scaling=False,
        explainability="impurity importance + permutation + SHAP (tree)",
        training_cost="medium",
        build=_build_random_forest,
        param_distributions=_RANDOM_FOREST_DISTRIBUTIONS,
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        description="XGBoost: expected best family; native NaN handling (no "
                    "imputer); scale_pos_weight computed from the training target.",
        family="boosted-trees",
        handles_nan_natively=True,
        requires_scaling=False,
        explainability="gain importance + permutation + SHAP TreeExplainer",
        training_cost="medium",
        build=_build_xgboost,
        param_distributions=_XGBOOST_DISTRIBUTIONS,
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        description="LightGBM: same class as XGBoost; native NaN handling; "
                    "scale_pos_weight computed from the training target.",
        family="boosted-trees",
        handles_nan_natively=True,
        requires_scaling=False,
        explainability="gain importance + permutation + SHAP TreeExplainer",
        training_cost="medium",
        build=_build_lightgbm,
        param_distributions=_LIGHTGBM_DISTRIBUTIONS,
    ),
}


def get_model(name: str, y_train: pd.Series) -> Pipeline:
    """Build a ready-to-fit pipeline for one zoo candidate."""
    if name not in MODEL_ZOO:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_ZOO)}."
        )
    return MODEL_ZOO[name].build(y_train)
