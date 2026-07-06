"""
Shared pytest helpers (Phase C1 — CI readiness).

Also importable directly: `from tests.conftest import set_tmp_experiment`
(the tests package has an __init__.py, so this is a normal module import;
pytest additionally auto-loads it for fixtures).
"""

from __future__ import annotations

from pathlib import Path

import mlflow


def set_tmp_experiment(name: str, tmp_dir: Path) -> None:
    """mlflow.set_experiment(), with run ARTIFACTS contained under tmp_dir.

    A bare ``mlflow.set_experiment(name)`` against a fresh sqlite tracking
    backend creates the experiment with the default artifact location
    ``./mlruns`` — relative to the process CWD. Test runs would then write
    model/plot artifacts into the repository checkout (and, because
    experiment ids restart at 1 in every tmp database, intermix them with
    the REAL Phase-4 artifact store in mlruns/). Pinning the artifact
    location into the test's tmp dir keeps every run hermetic — required
    for clean CI workspaces.
    """
    artifact_uri = (tmp_dir / "mlartifacts").as_uri()
    existing = mlflow.get_experiment_by_name(name)
    experiment_id = (
        existing.experiment_id if existing
        else mlflow.create_experiment(name, artifact_location=artifact_uri)
    )
    mlflow.set_experiment(experiment_id=experiment_id)
