"""
Tests for scripts/refresh_features_snapshot.py (Part 1 fix: decouple the
runtime features snapshot refresh from model registration/promotion).

scripts/ is not a package — loaded via importlib, same pattern as
ingest_jolpica.py's/promote_model.py's test files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_features_snapshot", _PROJECT_ROOT / "scripts" / "refresh_features_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refresh_features_snapshot = _load_module()


def test_copies_source_to_artifacts_root(tmp_path):
    source = tmp_path / "features.parquet"
    pd.DataFrame({"raceId": [1, 2]}).to_parquet(source, index=False)
    artifacts_root = tmp_path / "artifacts"

    rc = refresh_features_snapshot.main([
        "--source", str(source), "--artifacts-root", str(artifacts_root),
    ])

    assert rc == 0
    dest = artifacts_root / "features.parquet"
    assert dest.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(dest), pd.read_parquet(source))


def test_overwrites_existing_snapshot(tmp_path):
    source = tmp_path / "features.parquet"
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / "features.parquet").write_bytes(b"stale content")
    pd.DataFrame({"raceId": [9]}).to_parquet(source, index=False)

    refresh_features_snapshot.main([
        "--source", str(source), "--artifacts-root", str(artifacts_root),
    ])

    assert pd.read_parquet(artifacts_root / "features.parquet")["raceId"].tolist() == [9]


def test_missing_source_returns_1(tmp_path, capsys):
    rc = refresh_features_snapshot.main([
        "--source", str(tmp_path / "does-not-exist.parquet"),
        "--artifacts-root", str(tmp_path / "artifacts"),
    ])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
