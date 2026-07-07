"""
src/models/serving_bundle.py

Frozen serving bundle (Decision 026/027): the artifact FastAPI actually loads
at request time. Produced as a byproduct of registration
(src/models/train.py::register_model) — serving never talks to a live MLflow
tracking server or registry; it reads a plain local directory.

Bundle layout, per alias (e.g. models/serving/staging/):

    model/               mlflow.sklearn.save_model() output — the fitted
                         pipeline (ColumnGuard + preprocessing + estimator,
                         or the CalibratedModel wrapper around it) in
                         MLflow's own serialization format. Loaded via a
                         local-path mlflow.sklearn.load_model() call — no
                         tracking URI, no registry client, no network.
    manifest.json        ModelInfo fields, frozen at export time (name,
                         version, alias, run_id, trained_at, calibration,
                         model_class) plus a bundle format version and the
                         export timestamp.
    feature_schema.json  A human-readable mirror of the ColumnGuard schema
                         (training_schema(model)) — NOT load-bearing (the
                         schema is already embedded in the pickled model
                         itself and re-validated by ColumnGuard on every
                         call) but cheap and useful for inspecting a
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
DEFAULT_BUNDLE_ROOT = _PROJECT_ROOT / "models" / "serving"

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
    """models/serving/{alias.lower()} (or bundle_root/{alias.lower()})."""
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
