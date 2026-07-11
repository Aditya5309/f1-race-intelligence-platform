"""
Tests for src/models/serving_bundle.py (Decision 026/027/029).

Coverage:
  - export_bundle writes model/ + manifest.json + feature_schema.json
  - export_bundle overwrites a prior bundle at the same alias path
  - load_bundle round-trips a real fitted model + all ModelInfo fields
  - load_bundle raises FileNotFoundError with a clear, actionable message
    when the bundle (or its model/manifest) is missing
  - bundle_dir_for_alias lowercases the alias for the directory name
  - export_features_snapshot copies the source parquet to
    artifacts_root/features.parquet, creating directories as needed
  - default runtime artifact paths are all rooted under artifacts/
    (Decision 029 — the committed runtime tree, contrast with the
    gitignored data/ and models/ training-side trees)
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.registry import get_model
from src.models.serving_bundle import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_FEATURES_ARTIFACT,
    ModelInfo,
    bundle_dir_for_alias,
    export_bundle,
    export_features_snapshot,
    load_bundle,
)
from src.models.splits import to_xy


def _synthetic_frame(n_rows=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
        row[TARGET_COLUMN] = int(i % 10 == 0)
        row["raceId"] = i // 10
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def fitted_model():
    frame = _synthetic_frame()
    X, y, _ = to_xy(frame)
    model = get_model("logreg", y)
    model.fit(X, y)
    return model


@pytest.fixture()
def sample_info() -> ModelInfo:
    return ModelInfo(
        name="f1-winner", version="1", alias="Staging", run_id="abc123",
        trained_at="2026-07-07T00:00:00+00:00", calibration="none",
        model_class="Pipeline",
    )


def test_bundle_dir_for_alias_lowercases(tmp_path):
    assert bundle_dir_for_alias("Staging", tmp_path) == tmp_path / "staging"
    assert bundle_dir_for_alias("Production", tmp_path) == tmp_path / "production"


def test_export_bundle_writes_model_manifest_and_schema(tmp_path, fitted_model, sample_info):
    bundle_dir = export_bundle(fitted_model, sample_info, bundle_root=tmp_path)

    assert bundle_dir == tmp_path / "staging"
    assert (bundle_dir / "model" / "MLmodel").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "feature_schema.json").exists()

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["name"] == "f1-winner"
    assert manifest["version"] == "1"
    assert manifest["alias"] == "Staging"
    assert manifest["run_id"] == "abc123"
    assert manifest["calibration"] == "none"
    assert manifest["model_class"] == "Pipeline"
    assert manifest["metrics"] == {}
    assert manifest["bundle_format_version"] == 1
    assert manifest["exported_at"].startswith("20")

    schema = json.loads((bundle_dir / "feature_schema.json").read_text())
    assert schema["feature_names"] == list(FEATURE_COLUMNS)


def test_export_bundle_overwrites_existing(tmp_path, fitted_model, sample_info):
    export_bundle(fitted_model, sample_info, bundle_root=tmp_path)

    updated_info = ModelInfo(**{**sample_info.to_dict(), "version": "2"})
    bundle_dir = export_bundle(fitted_model, updated_info, bundle_root=tmp_path)

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["version"] == "2"
    # No stray files left from the first export's model/ directory.
    assert (bundle_dir / "model" / "MLmodel").exists()


def test_load_bundle_roundtrip(tmp_path, fitted_model, sample_info):
    bundle_dir = export_bundle(fitted_model, sample_info, bundle_root=tmp_path)
    model, info = load_bundle(bundle_dir)

    assert isinstance(info, ModelInfo)
    assert info == sample_info
    X, y, _ = to_xy(_synthetic_frame(seed=1))
    np.testing.assert_array_equal(
        model.predict_proba(X)[:, 1], fitted_model.predict_proba(X)[:, 1]
    )


def test_load_bundle_does_not_need_mlflow_importable(tmp_path, fitted_model, sample_info, monkeypatch):
    """Phase 4 Tranche D Item 1b: load_bundle() (predict.py/app/api.py's
    entire serving path) must not need mlflow installed at all — only
    export_bundle() (training/registration-side) still does."""
    bundle_dir = export_bundle(fitted_model, sample_info, bundle_root=tmp_path)

    monkeypatch.setitem(sys.modules, "mlflow", None)
    model, info = load_bundle(bundle_dir)

    assert info == sample_info
    X, y, _ = to_xy(_synthetic_frame(seed=1))
    np.testing.assert_array_equal(
        model.predict_proba(X)[:, 1], fitted_model.predict_proba(X)[:, 1]
    )


def test_load_bundle_roundtrips_metrics(tmp_path, fitted_model, sample_info):
    info_with_metrics = ModelInfo(
        **{**sample_info.to_dict(), "metrics": {"top1_accuracy": 0.682, "spearman_corr": 0.749}}
    )
    bundle_dir = export_bundle(fitted_model, info_with_metrics, bundle_root=tmp_path)
    _, info = load_bundle(bundle_dir)
    assert info.metrics == {"top1_accuracy": 0.682, "spearman_corr": 0.749}


def test_load_bundle_defaults_metrics_for_legacy_manifest(tmp_path, fitted_model, sample_info):
    """A manifest.json written before Tranche C has no "metrics" key at all
    — load_bundle must degrade to {} rather than KeyError, so bundles
    already committed to the repo keep loading unchanged."""
    bundle_dir = export_bundle(fitted_model, sample_info, bundle_root=tmp_path)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["metrics"]
    manifest_path.write_text(json.dumps(manifest))

    _, info = load_bundle(bundle_dir)
    assert info.metrics == {}


def test_load_bundle_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No serving bundle"):
        load_bundle(tmp_path / "does-not-exist")


def test_load_bundle_missing_manifest_raises(tmp_path, fitted_model, sample_info):
    bundle_dir = export_bundle(fitted_model, sample_info, bundle_root=tmp_path)
    (bundle_dir / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="No serving bundle"):
        load_bundle(bundle_dir)


# ---------------------------------------------------------------------------
# Runtime artifact layout (Decision 029)
# ---------------------------------------------------------------------------

def test_default_paths_are_rooted_under_artifacts():
    """The committed runtime tree — contrast with gitignored data/models/."""
    assert DEFAULT_ARTIFACTS_ROOT.name == "artifacts"
    assert DEFAULT_BUNDLE_ROOT == DEFAULT_ARTIFACTS_ROOT / "serving"
    assert DEFAULT_FEATURES_ARTIFACT == DEFAULT_ARTIFACTS_ROOT / "features.parquet"


def test_export_features_snapshot_copies_to_artifacts_root(tmp_path):
    source = tmp_path / "source.parquet"
    pd.DataFrame({"raceId": [1, 2], "driverId": [1, 2]}).to_parquet(source)

    dest = export_features_snapshot(source, artifacts_root=tmp_path / "artifacts")

    assert dest == tmp_path / "artifacts" / "features.parquet"
    assert dest.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(dest), pd.read_parquet(source))


def test_export_features_snapshot_creates_missing_directories(tmp_path):
    source = tmp_path / "source.parquet"
    pd.DataFrame({"raceId": [1]}).to_parquet(source)

    dest = export_features_snapshot(
        source, artifacts_root=tmp_path / "nested" / "artifacts"
    )

    assert dest.exists()


def test_export_features_snapshot_overwrites_existing(tmp_path):
    source = tmp_path / "source.parquet"
    artifacts_root = tmp_path / "artifacts"
    pd.DataFrame({"raceId": [1]}).to_parquet(source)
    export_features_snapshot(source, artifacts_root=artifacts_root)

    pd.DataFrame({"raceId": [1, 2, 3]}).to_parquet(source)
    dest = export_features_snapshot(source, artifacts_root=artifacts_root)

    assert len(pd.read_parquet(dest)) == 3
