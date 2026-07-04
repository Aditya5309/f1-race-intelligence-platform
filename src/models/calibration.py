"""
src/models/calibration.py

Out-of-fold isotonic probability calibration for Phase 4 finalists
(Decision 015; design Sections 5 and 9.5).

Why: class-weighted training (class_weight='balanced', scale_pos_weight)
deliberately distorts probability SCALE while preserving probability ORDER.
The Phase 4 finalist (tuned logreg) showed the expected inflation on
validation (ECE 0.153 — e.g. its "55% win chance" bin realizes ~3% winners).
Design Section 5's approved remedy: an isotonic calibrator fit on
CROSS-VALIDATION FOLD PREDICTIONS ONLY — never on validation or test — so
calibration adds no new data exposure beyond the training split.

How it composes (leakage discipline unchanged):
- `oof_predictions()` replays the exact season-fold protocol of
  train.run_cv: a FRESH pipeline is built and fit inside every fold
  (Section 11.4 containment), and each fold's held-out season supplies
  out-of-fold (probability, outcome) pairs.
- `fit_calibrated_model()` fits IsotonicRegression on those OOF pairs, refits
  the base pipeline on the full fit frame, and wraps both in a
  `CalibratedModel`.
- `CalibratedModel` is a plain sklearn-style estimator: predict_proba runs
  the base pipeline (ColumnGuard schema validation INTACT — it is the first
  step of the wrapped pipeline) and passes the positive-class probability
  through the isotonic map. Isotonic regression is monotone non-decreasing,
  so within-race ranking is preserved by construction (ties can appear on
  isotonic plateaus — the pessimistic tie policy in evaluate.py then applies;
  measured cost on validation: top-1 unchanged, top-3 −1 race).
- The wrapper exposes `named_steps` (delegated to the base pipeline) so
  `registry.training_schema()` and predict.py's schema introspection work
  identically for calibrated and raw artifacts, and it serializes through
  mlflow.sklearn (cloudpickle) like any zoo pipeline.

CalibratedModel instances are assembled from parts fit by
`fit_calibrated_model()` — calling .fit() on the wrapper raises, because a
naive refit would silently re-learn the calibrator on non-OOF predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from src.models.registry import get_model
from src.models.splits import DEFAULT_N_FOLDS, season_folds, to_xy

# Keep calibrated probabilities strictly inside (0, 1): isotonic can emit
# exact 0/1, which breaks log-loss and per-race normalization edge cases.
PROBABILITY_EPS = 1e-6


class CalibratedModel(BaseEstimator, ClassifierMixin):
    """A fitted zoo pipeline + a fitted isotonic map over its P(win).

    Duck-types the parts of the sklearn Pipeline interface the rest of the
    project relies on (predict_proba / predict / classes_ / named_steps), so
    train.py registration, registry.training_schema() and predict.py treat
    calibrated and raw artifacts uniformly.
    """

    #: Introspectable calibration marker (predict.py reports it as metadata).
    calibration = "isotonic-oof"

    def __init__(self, base_pipeline: Pipeline, calibrator: IsotonicRegression):
        self.base_pipeline = base_pipeline
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    @property
    def named_steps(self) -> dict:
        """Delegate to the base pipeline so schema introspection
        (registry.training_schema) works unchanged."""
        return self.base_pipeline.named_steps

    def fit(self, X, y=None):
        raise NotImplementedError(
            "CalibratedModel is assembled by fit_calibrated_model() — a plain "
            "refit would re-learn the calibrator on non-out-of-fold "
            "predictions, which is exactly the leakage the OOF protocol exists "
            "to prevent."
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base_pipeline.predict_proba(X)[:, 1]
        cal = np.clip(
            self.calibrator.predict(raw), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
        )
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def oof_predictions(
    name: str,
    train_df: pd.DataFrame,
    params: dict | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (probability, outcome) pairs over the season folds.

    Identical fold protocol to train.run_cv: fresh pipeline per fold, fit on
    the fold's train seasons only (Section 11.4), scored on the fold's
    held-out season. season_folds() itself rejects any input containing
    val/test/forward-holdout years, so the calibrator can only ever see
    training-window data.
    """
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for fold in season_folds(train_df, n_folds=n_folds):
        X_tr, y_tr, _ = to_xy(fold.train)
        pipeline = get_model(name, y_tr)
        if params:
            pipeline.set_params(**params)
        pipeline.fit(X_tr, y_tr)
        X_va, y_va, _ = to_xy(fold.val)
        probs.append(pipeline.predict_proba(X_va)[:, 1])
        ys.append(np.asarray(y_va))
    return np.concatenate(probs), np.concatenate(ys)


def fit_isotonic(oof_prob: np.ndarray, oof_y: np.ndarray) -> IsotonicRegression:
    """Monotone map from raw model probability to observed win frequency."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_prob, oof_y)
    return calibrator


def fit_calibrated_model(
    name: str,
    train_df: pd.DataFrame,
    fit_df: pd.DataFrame | None = None,
    params: dict | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
) -> CalibratedModel:
    """Build the production-ready calibrated artifact for one candidate.

    train_df — the Decision-008 training split; sole source of the
    calibrator's OOF pairs regardless of what the base model is fit on.
    fit_df — what the base pipeline is refit on (defaults to train_df;
    a future Production refit passes train+val here, per design Section 4,
    while the calibrator stays train-OOF).
    """
    oof_prob, oof_y = oof_predictions(name, train_df, params=params, n_folds=n_folds)
    calibrator = fit_isotonic(oof_prob, oof_y)

    fit_df = train_df if fit_df is None else fit_df
    X_fit, y_fit, _ = to_xy(fit_df)
    base = get_model(name, y_fit)
    if params:
        base.set_params(**params)
    base.fit(X_fit, y_fit)

    return CalibratedModel(base_pipeline=base, calibrator=calibrator)
