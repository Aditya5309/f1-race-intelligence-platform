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
                               format (cloudpickle). Written via
                               mlflow.sklearn.save_model() but loaded via a
                               plain stdlib pickle.load() on
                               model/model.pkl (Phase 4 Tranche D Item 1b)
                               — no MLflow import, tracking URI, registry
                               client, or network needed to serve.
                manifest.json  ModelInfo fields, frozen at export time
                               (name, version, alias, run_id, trained_at,
                               calibration, model_class, metrics — the
                               evaluate_all() dict this bundle's model
                               scored on the validation split, Phase 4
                               Tranche C Item 1 — and baseline_bootstrapped,
                               True only if this bundle was promoted via
                               `promote_model.py --force-baseline` with no
                               prior baseline to compare against, Phase 4
                               Tranche D post-mortem) plus a bundle format
                               version and the export timestamp.
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
by the caller (train.py). Loading (used by predict.py) needs no MLflow
import at all (Item 1b) — plain stdlib json + pickle over a filesystem read.
Only export_bundle() still imports mlflow, since writing still goes through
mlflow.sklearn.save_model() (unchanged — this is a training/registration-side
concern, not a serving one).
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.models.registry import training_schema

BUNDLE_FORMAT_VERSION = 1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Runtime artifact root (Decision 029) — committed to git; contrast with
#: the gitignored data/, mlruns/, mlflow.db, models/ training-side trees.
DEFAULT_ARTIFACTS_ROOT = _PROJECT_ROOT / "artifacts"
DEFAULT_BUNDLE_ROOT = DEFAULT_ARTIFACTS_ROOT / "serving"
DEFAULT_FEATURES_ARTIFACT = DEFAULT_ARTIFACTS_ROOT / "features.parquet"

_MODEL_SUBDIR = "model"
#: mlflow.sklearn.save_model()'s own stable, hardcoded filename for its
#: pickle/cloudpickle-format flavor (mlflow.sklearn.MODEL_PICKLE_FILE_NAME —
#: unchanged across pickle vs cloudpickle serialization; verified directly
#: against the real committed bundle). export_bundle() below is the only
#: writer, via mlflow.sklearn.save_model(); load_bundle() reads this same
#: fixed path with plain stdlib pickle — no mlflow import needed to load
#: (Phase 4 Tranche D Item 1b). A cloudpickle-written stream is a valid
#: standard pickle stream for objects with no closures/dynamic classes —
#: true of every zoo pipeline/CalibratedModel here — confirmed byte-identical
#: predict_proba() output against mlflow.sklearn.load_model() before this
#: change shipped.
_MODEL_PICKLE_FILENAME = "model.pkl"
_MANIFEST_FILENAME = "manifest.json"
_FEATURE_SCHEMA_FILENAME = "feature_schema.json"
_MODEL_INFO_FIELDS = (
    "name", "version", "alias", "run_id", "trained_at", "calibration", "model_class",
    "metrics", "baseline_bootstrapped",
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
    #: evaluate_all() metrics for this bundle's own fitted model, scored on
    #: the Decision-008 validation split (Phase 4 Tranche C, Item 1) — what
    #: this bundle actually scored, not just what it is. {} for bundles
    #: exported before this field existed (load_bundle degrades gracefully;
    #: see its manifest.get("metrics", {})) or when a caller has no honest
    #: held-out metrics to report.
    metrics: dict = field(default_factory=dict)
    #: True only for a bundle promoted via `promote_model.py --force-baseline`
    #: (Phase 4 Tranche D post-mortem) — i.e. this bundle's metrics were
    #: recorded WITHOUT a regression check against a prior baseline, because
    #: none existed yet. Distinguishes "passed a real comparison" from
    #: "bootstrapped the first one" for anyone reading this manifest later.
    #: False (the default) for every normal promotion and for bundles
    #: exported before this field existed.
    baseline_bootstrapped: bool = False

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

    mlflow is imported here, not at module level (Phase 4 Tranche D Item
    1b): this is the ONLY function in this module that still needs it (to
    write via mlflow.sklearn.save_model()) — a training/registration-side
    concern. Serving (load_bundle, and everything predict.py/app/api.py
    call) must be able to import this whole module without mlflow installed
    at all.
    """
    import mlflow

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

    No MLflow tracking URI, no registry client, no network, and (Phase 4
    Tranche D Item 1b) no MLflow import at all — a plain local filesystem
    read plus stdlib pickle. Raises FileNotFoundError with a clear,
    actionable message if the bundle is missing; app/api.py's degraded-start
    lifespan catches this exactly like it caught a missing registry before.
    """
    bundle_dir = Path(bundle_dir)
    model_path = bundle_dir / _MODEL_SUBDIR
    model_pickle_path = model_path / _MODEL_PICKLE_FILENAME
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    if not model_pickle_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"No serving bundle at {bundle_dir} (expected {_MODEL_SUBDIR}/"
            f"{_MODEL_PICKLE_FILENAME} and {_MANIFEST_FILENAME}) — run "
            "`python -m src.models.train --register <alias> --calibrate` "
            "to produce one."
        )
    manifest = json.loads(manifest_path.read_text())
    # Older bundles predate one or both of "metrics"/"baseline_bootstrapped"
    # — degrade to their ModelInfo defaults rather than KeyError so existing
    # committed bundles keep loading unchanged.
    _defaults = {"metrics": {}, "baseline_bootstrapped": False}
    info = ModelInfo(**{
        f: manifest.get(f, _defaults[f]) if f in _defaults else manifest[f]
        for f in _MODEL_INFO_FIELDS
    })
    with model_pickle_path.open("rb") as f:
        model = pickle.load(f)
    return model, info
