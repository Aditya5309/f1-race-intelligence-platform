"""
src/models/serving_bundle.py

Frozen serving artifacts (Decision 026/027/029): everything FastAPI actually
reads at request time — a frozen model bundle plus a frozen features
snapshot — produced as a byproduct of registration
(src/models/train.py::register_model). Serving never talks to a live MLflow
tracking server or registry, and never reads from the gitignored `data/`
training tree; it reads a plain local directory that is committed to git.

Runtime artifact layout, rooted at artifacts/ (Decision 029 — separates
committed runtime artifacts from gitignored training artifacts in data/,
mlruns/, mlflow.db, reports/):

    artifacts/
        features.parquet      A frozen snapshot of the serving feature
                              matrix (export_features_snapshot), copied
                              from the training pipeline's
                              data/processed/features.parquet at
                              registration time. Shared across aliases —
                              it answers "which races/drivers exist", which
                              is orthogonal to which model version serves
                              them, so it is NOT nested per-alias like the
                              model bundle below (that would duplicate the
                              same ~1 MB file per registered alias).
        serving/
            <alias>/           One frozen bundle per registered alias
                               (e.g. staging/, production/):
                model/         mlflow.sklearn.save_model() output — the
                               fitted pipeline (ColumnGuard + preprocessing
                               + estimator, or the CalibratedModel wrapper
                               around it) in MLflow's own serialization
                               format. Loaded via a local-path
                               mlflow.sklearn.load_model() call — no
                               tracking URI, no registry client, no network.
                manifest.json  ModelInfo fields, frozen at export time
                               (name, version, alias, run_id, trained_at,
                               calibration, model_class) plus a bundle
                               format version and the export timestamp.
                feature_schema.json
                               A human-readable mirror of the ColumnGuard
                               schema (training_schema(model)) — NOT
                               load-bearing (the schema is already embedded
                               in the pickled model itself and
                               re-validated by ColumnGuard on every call)
                               but cheap and useful for inspecting a
                               deployed bundle without unpickling it.

This module has no opinion about experiments, aliases, or how the model was
produced — export_bundle() takes an already-fitted model + ModelInfo built
by the caller (train.py). Loading (used by predict.py) has no MLflow
tracking/registry imports at all — just a filesystem read.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from src.models.registry import training_schema

BUNDLE_FORMAT_VERSION = 1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Runtime artifact root (Decision 029) — committed to git; contrast with
#: the gitignored data/, mlruns/, mlflow.db, models/ training-side trees.
DEFAULT_ARTIFACTS_ROOT = _PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_ROOT = DEFAULT_ARTIFACTS_ROOT / "serving"
DEFAULT_FEATURES_ARTIFACT = DEFAULT_ARTIFACTS_ROOT / "features.parquet"

_MODEL_SUBDIR = "model"
_MANIFEST_FILENAME = "manifest.json"
_FEATURE_SCHEMA_FILENAME = "feature_schema.json"
_MODEL_INFO_FIELDS = (
    "name", "version", "alias", "run_id", "trained_at", "calibration", "model_class",
)


@dataclass(frozen=True)
class ModelInfo:
    """JSON-ready bundle/model metadata (dashboard/API display)."""
    name: str
    version: str
    alias: str
    run_id: str
    trained_at: str          # ISO-8601 UTC, frozen at export time
    calibration: str         # "isotonic-oof" | "none"
    model_class: str         # e.g. "CalibratedModel", "Pipeline"

    def to_dict(self) -> dict:
        return asdict(self)


def bundle_dir_for_alias(alias: str, bundle_root: Path | None = None) -> Path:
    """artifacts/serving/{alias.lower()} (or bundle_root/{alias.lower()})."""
    return (bundle_root or DEFAULT_BUNDLE_ROOT) / alias.lower()


def export_bundle(model, info: ModelInfo, bundle_root: Path | None = None) -> Path:
    """Freeze an already-fitted model + its metadata to bundle_root/alias/.

    Overwrites any existing bundle at that path — a new registration
    supersedes the old frozen artifact for that alias. Takes the model
    object directly (no MLflow reload): the caller (register_model) already
    has it fitted in memory.
    """
    bundle_dir = bundle_dir_for_alias(info.alias, bundle_root)
    model_path = bundle_dir / _MODEL_SUBDIR
    if model_path.exists():
        shutil.rmtree(model_path)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(model, str(model_path))

    manifest = {
        **info.to_dict(),
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (bundle_dir / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    (bundle_dir / _FEATURE_SCHEMA_FILENAME).write_text(
        json.dumps(training_schema(model), indent=2)
    )
    return bundle_dir


def export_features_snapshot(source: Path | str, artifacts_root: Path | None = None) -> Path:
    """Freeze the current training-side features.parquet as the runtime
    serving snapshot (Decision 029): artifacts_root/features.parquet.

    Copies `source` (normally the training pipeline's
    data/processed/features.parquet, produced by src.features.pipeline)
    into the committed runtime artifact tree. Called by
    train.py::register_model() on every registration, so the servable
    snapshot always matches the features the just-registered model was
    trained/evaluated against. Not alias-scoped: GET /races and
    GET /predictions/{race_id} both look up rows by raceId regardless of
    which model alias serves them, so one shared snapshot (rather than a
    copy per alias) is the correct shape.
    """
    dest = (artifacts_root or DEFAULT_ARTIFACTS_ROOT) / "features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def load_bundle(bundle_dir: Path | str):
    """Load a frozen bundle. Returns (model, ModelInfo).

    No MLflow tracking URI, no registry client, no network — a plain local
    filesystem read. Raises FileNotFoundError with a clear, actionable
    message if the bundle is missing; app/api.py's degraded-start lifespan
    catches this exactly like it caught a missing registry before.
    """
    bundle_dir = Path(bundle_dir)
    model_path = bundle_dir / _MODEL_SUBDIR
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    if not model_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"No serving bundle at {bundle_dir} (expected {_MODEL_SUBDIR}/ and "
            f"{_MANIFEST_FILENAME}) — run `python -m src.models.train --register "
            "<alias> --calibrate` to produce one."
        )
    manifest = json.loads(manifest_path.read_text())
    info = ModelInfo(**{field: manifest[field] for field in _MODEL_INFO_FIELDS})
    model = mlflow.sklearn.load_model(str(model_path))
    return model, info
