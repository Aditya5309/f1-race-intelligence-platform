"""
Tests for scripts/promote_model.py (Phase 4 Tranche C, Item 2).

scripts/ is not a package (no __init__.py, mirrors backfill_weather.py/
smoke.py/dev.py — none of which have tests either), so the module is loaded
directly via importlib rather than a normal import.

Coverage:
  - resolve_version: explicit --version vs. defaulting to the highest
    registered version
  - check_schema_and_predictions: passes on a real fit, refuses on a
    degenerate (constant-output) model, refuses on an empty features frame
  - check_regression: refuses on a regression past tolerance, passes within
    tolerance, skips a metric missing on either side
  - end-to-end: a deliberately regressed candidate is REFUSED and the
    currently-served bundle is left byte-for-byte untouched; a legitimate
    candidate is PROMOTED and becomes the new served bundle
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.serving_bundle import load_bundle
from src.models.splits import TemporalSplit, temporal_split
from src.models.train import register_model
from tests.conftest import set_tmp_experiment

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_promote_module():
    spec = importlib.util.spec_from_file_location(
        "promote_model", _PROJECT_ROOT / "scripts" / "promote_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promote_model = _load_promote_module()


# ---------------------------------------------------------------------------
# Synthetic data (mirrors tests/test_train.py's builder)
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


def _full_frame(**kwargs) -> pd.DataFrame:
    return _synthetic_features(range(2010, 2025), **kwargs)


@pytest.fixture()
def tmp_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    set_tmp_experiment("promote-test", tmp_path)
    yield
    mlflow.set_tracking_uri(None)


@pytest.fixture()
def env(tmp_mlflow, tmp_path):
    """Isolated registry + bundle/artifacts roots + features.parquet —
    everything promote_model.py touches, none of it the real project."""
    frame = _full_frame()
    features_path = tmp_path / "features.parquet"
    frame.to_parquet(features_path, index=False)
    return {
        "tmp_path": tmp_path,
        "split": temporal_split(frame),
        "features_path": features_path,
        "bundle_root": tmp_path / "bundle",
        "artifacts_root": tmp_path / "artifacts",
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
    }


def _register(env, split=None, alias="Staging"):
    return register_model(
        "logreg", split or env["split"], alias=alias,
        bundle_root=env["bundle_root"], features_source=env["features_path"],
        artifacts_root=env["artifacts_root"],
    )


def _promote_args(env, alias="Staging", version=None):
    args = [
        "--alias", alias,
        "--tracking-uri", env["tracking_uri"],
        "--bundle-root", str(env["bundle_root"]),
        "--artifacts-root", str(env["artifacts_root"]),
        "--features-source", str(env["features_path"]),
    ]
    if version is not None:
        args += ["--version", str(version)]
    return args


# ---------------------------------------------------------------------------
# resolve_version
# ---------------------------------------------------------------------------

def test_resolve_version_explicit(env):
    _register(env)
    client = mlflow.MlflowClient(tracking_uri=env["tracking_uri"])
    assert promote_model.resolve_version(client, "1") == "1"


def test_resolve_version_defaults_to_highest(env):
    _register(env)
    v2 = _register(env)
    client = mlflow.MlflowClient(tracking_uri=env["tracking_uri"])
    assert promote_model.resolve_version(client, None) == str(v2)


def test_resolve_version_empty_registry_refuses(env):
    client = mlflow.MlflowClient(tracking_uri=env["tracking_uri"])
    with pytest.raises(promote_model.PromotionRefused, match="No versions"):
        promote_model.resolve_version(client, None)


# ---------------------------------------------------------------------------
# check_schema_and_predictions
# ---------------------------------------------------------------------------

def test_check_schema_and_predictions_passes_for_real_model(env):
    version = _register(env)
    model = mlflow.sklearn.load_model(f"models:/f1-winner/{version}")
    promote_model.check_schema_and_predictions(model, env["split"].val)  # no raise


def test_check_schema_and_predictions_refuses_empty_frame(env):
    version = _register(env)
    model = mlflow.sklearn.load_model(f"models:/f1-winner/{version}")
    with pytest.raises(promote_model.PromotionRefused, match="No races"):
        promote_model.check_schema_and_predictions(model, env["split"].val.iloc[:0])


class _ConstantModel:
    """A model whose predict_proba never varies — the degenerate case
    predict_race()'s own checks (no NaN, sums to 1) would let through."""

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def test_check_schema_and_predictions_refuses_degenerate_output(env, monkeypatch):
    monkeypatch.setattr(
        promote_model, "predict_race",
        lambda model, race_df: pd.DataFrame({
            "raceId": race_df["raceId"].to_numpy(),
            "win_probability": np.full(len(race_df), 1.0 / len(race_df)),
        }),
    )
    with pytest.raises(promote_model.PromotionRefused, match="degenerate"):
        promote_model.check_schema_and_predictions(_ConstantModel(), env["split"].val)


# ---------------------------------------------------------------------------
# check_regression
# ---------------------------------------------------------------------------

def test_check_regression_refuses_past_tolerance():
    served = {"top1_accuracy": 0.70, "spearman_corr": 0.75}
    candidate = {"top1_accuracy": 0.60, "spearman_corr": 0.75}
    with pytest.raises(promote_model.PromotionRefused, match="top1_accuracy"):
        promote_model.check_regression(candidate, served, top1_tolerance=0.03, spearman_tolerance=0.015)


def test_check_regression_passes_within_tolerance():
    served = {"top1_accuracy": 0.70, "spearman_corr": 0.75}
    candidate = {"top1_accuracy": 0.68, "spearman_corr": 0.745}
    promote_model.check_regression(candidate, served, top1_tolerance=0.03, spearman_tolerance=0.015)  # no raise


def test_check_regression_skips_missing_metric():
    served = {"top1_accuracy": 0.70}       # no spearman_corr recorded
    candidate = {"top1_accuracy": 0.71, "spearman_corr": 0.10}
    promote_model.check_regression(candidate, served, top1_tolerance=0.03, spearman_tolerance=0.015)  # no raise


# ---------------------------------------------------------------------------
# End-to-end: refuse a regression, promote a legitimate candidate
# ---------------------------------------------------------------------------

@pytest.fixture()
def served(env):
    """A good v1, exported and then snapshotted. Every later register_model()
    call in these tests ALSO auto-exports (its own unchecked behavior,
    unrelated to this gate) — each test restores the snapshot afterward to
    simulate "a candidate was registered but never promoted", which is
    exactly the real-world state this script is meant to operate on."""
    version = _register(env)
    served_dir = env["bundle_root"] / "staging"
    backup_dir = env["tmp_path"] / "served_backup"
    shutil.copytree(served_dir, backup_dir)
    return version, served_dir, backup_dir


def _restore_served(served_dir: Path, backup_dir: Path) -> None:
    shutil.rmtree(served_dir)
    shutil.copytree(backup_dir, served_dir)


def test_promote_refuses_regression_and_leaves_bundle_untouched(env, served):
    good_version, served_dir, backup_dir = served
    good_manifest = (served_dir / "manifest.json").read_text()

    # Deliberately bad candidate: winners shuffled within each race (Section
    # 11.5's canary) collapses the learnable signal toward chance.
    bad_train = env["split"].train.copy()
    rng = np.random.default_rng(1)
    bad_train[TARGET_COLUMN] = (
        bad_train.groupby("raceId")[TARGET_COLUMN]
        .transform(lambda s: s.sample(frac=1.0, random_state=rng.integers(1 << 31)).to_numpy())
    )
    bad_split = TemporalSplit(train=bad_train, val=env["split"].val,
                              test=env["split"].test, strategy=env["split"].strategy)
    bad_version = _register(env, split=bad_split)
    assert bad_version != good_version
    _restore_served(served_dir, backup_dir)   # undo register_model's own auto-export

    rc = promote_model.main(_promote_args(env, version=bad_version))

    assert rc == 1
    assert (served_dir / "manifest.json").read_text() == good_manifest


def test_promote_succeeds_for_legitimate_candidate(env, served):
    good_version, served_dir, backup_dir = served

    next_version = _register(env)   # same data/params — a legitimate candidate
    assert next_version != good_version
    _restore_served(served_dir, backup_dir)

    rc = promote_model.main(_promote_args(env, version=next_version))

    assert rc == 0
    _, info = load_bundle(served_dir)
    assert info.version == str(next_version)
    assert info.metrics   # populated, not the empty-dict legacy default


def test_promote_first_ever_promotion_has_no_baseline_to_compare(env):
    """No bundle exists yet at all — the regression check has nothing to
    compare against and must not block the first promotion."""
    version = _register(env)
    shutil.rmtree(env["bundle_root"] / "staging")   # simulate "never promoted"

    rc = promote_model.main(_promote_args(env, version=version))

    assert rc == 0
    assert (env["bundle_root"] / "staging" / "manifest.json").exists()


def test_promote_missing_features_source_returns_1(env, capsys):
    rc = promote_model.main([
        "--alias", "Staging",
        "--tracking-uri", env["tracking_uri"],
        "--bundle-root", str(env["bundle_root"]),
        "--artifacts-root", str(env["artifacts_root"]),
        "--features-source", str(env["tmp_path"] / "missing.parquet"),
    ])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
