"""
Tests for the two pipeline orchestration entry points:
  - src/features/pipeline.py  main()          (features.parquet builder CLI)
  - src/pipelines/build_dataset.py build_dataset()  (master-dataset orchestrator)

These are wiring tests: the heavy transform/validation logic is already
covered by tests/test_features.py and tests/test_build_master_dataset.py, so
the transforms are stubbed here and only the orchestration contract is
asserted — exit codes, dry-run semantics, validation-failure propagation,
and the parquet writes.
"""

import pandas as pd
import pytest

import src.features.pipeline as pipeline_mod
import src.pipelines.build_dataset as build_dataset_mod
from src.data.validator import ValidationResult
from src.features.pipeline import main as features_main
from src.pipelines.build_dataset import build_dataset


def _passing(row_count=1, warnings=None) -> ValidationResult:
    return ValidationResult(
        passed=True, errors=[], warnings=warnings or [], row_count=row_count
    )


def _failing(row_count=1) -> ValidationResult:
    return ValidationResult(
        passed=False, errors=["synthetic failure"], warnings=[], row_count=row_count
    )


# ---------------------------------------------------------------------------
# src/features/pipeline.py main()
# ---------------------------------------------------------------------------

class TestFeaturesPipelineCLI:
    @pytest.fixture
    def cli_env(self, monkeypatch, tmp_path):
        """Stub the transforms and point every path constant at tmp_path."""
        master_path = tmp_path / "master_dataset.parquet"
        master = pd.DataFrame({"raceId": [1, 1], "driverId": [1, 2]})
        master.to_parquet(master_path, index=False)

        features = pd.DataFrame({"raceId": [1, 1], "driverId": [1, 2]})
        features_path = tmp_path / "features.parquet"

        monkeypatch.setattr(pipeline_mod, "MASTER_DATASET_PATH", master_path)
        monkeypatch.setattr(pipeline_mod, "FEATURES_PATH", features_path)
        monkeypatch.setattr(pipeline_mod, "_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            pipeline_mod, "load_standings",
            lambda: (pd.DataFrame(), pd.DataFrame()),
        )
        monkeypatch.setattr(
            pipeline_mod, "load_race_weather", lambda: pd.DataFrame(),
        )
        monkeypatch.setattr(
            pipeline_mod, "build_features", lambda m, ds, cs, weather: features
        )
        monkeypatch.setattr(
            pipeline_mod, "validate_features",
            lambda df, expected_row_count: _passing(row_count=len(df)),
        )
        return {"features_path": features_path, "features": features}

    def test_missing_master_dataset_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            pipeline_mod, "MASTER_DATASET_PATH", tmp_path / "missing.parquet"
        )
        assert features_main([]) == 1
        assert "not found" in capsys.readouterr().err

    def test_happy_path_writes_features_parquet(self, cli_env):
        assert features_main([]) == 0
        assert cli_env["features_path"].exists()
        written = pd.read_parquet(cli_env["features_path"])
        pd.testing.assert_frame_equal(written, cli_env["features"])

    def test_dry_run_skips_write(self, cli_env):
        assert features_main(["--dry-run"]) == 0
        assert not cli_env["features_path"].exists()

    def test_validation_failure_returns_1_and_no_write(
        self, cli_env, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            pipeline_mod, "validate_features",
            lambda df, expected_row_count: _failing(row_count=len(df)),
        )
        assert features_main([]) == 1
        assert not cli_env["features_path"].exists()
        assert "synthetic failure" in capsys.readouterr().err

    def test_validation_warnings_are_printed(self, cli_env, monkeypatch, capsys):
        monkeypatch.setattr(
            pipeline_mod, "validate_features",
            lambda df, expected_row_count: _passing(
                row_count=len(df), warnings=["mostly null"]
            ),
        )
        assert features_main(["--dry-run"]) == 0
        assert "WARNING: mostly null" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# src/pipelines/build_dataset.py build_dataset()
# ---------------------------------------------------------------------------

class TestBuildDatasetOrchestrator:
    @pytest.fixture
    def cli_env(self, monkeypatch, tmp_path):
        results = pd.DataFrame({"raceId": [1, 1], "driverId": [1, 2]})
        master = pd.DataFrame(
            {"raceId": [1, 1], "driverId": [1, 2], "winner": [1, 0]}
        )
        monkeypatch.setattr(
            build_dataset_mod, "load_inputs", lambda: {"results": results}
        )
        monkeypatch.setattr(
            build_dataset_mod, "validate_inputs", lambda inputs: _passing()
        )
        monkeypatch.setattr(
            build_dataset_mod, "build_master_dataset", lambda inputs: master
        )
        monkeypatch.setattr(
            build_dataset_mod, "validate_output",
            lambda df, expected_row_count: _passing(row_count=len(df)),
        )
        return {"master": master, "out": tmp_path / "master_dataset.parquet"}

    def test_happy_path_writes_parquet(self, cli_env):
        returned = build_dataset(output_path=cli_env["out"])
        assert cli_env["out"].exists()
        pd.testing.assert_frame_equal(returned, cli_env["master"])
        written = pd.read_parquet(cli_env["out"])
        pd.testing.assert_frame_equal(written, cli_env["master"])

    def test_dry_run_skips_write(self, cli_env):
        returned = build_dataset(dry_run=True, output_path=cli_env["out"])
        assert not cli_env["out"].exists()
        pd.testing.assert_frame_equal(returned, cli_env["master"])

    def test_input_validation_failure_raises(self, cli_env, monkeypatch):
        monkeypatch.setattr(
            build_dataset_mod, "validate_inputs", lambda inputs: _failing()
        )
        with pytest.raises(ValueError, match="Input validation failed"):
            build_dataset(output_path=cli_env["out"])
        assert not cli_env["out"].exists()

    def test_output_validation_failure_raises(self, cli_env, monkeypatch):
        monkeypatch.setattr(
            build_dataset_mod, "validate_output",
            lambda df, expected_row_count: _failing(),
        )
        with pytest.raises(ValueError, match="Output validation failed"):
            build_dataset(output_path=cli_env["out"])
        assert not cli_env["out"].exists()

    def test_happy_path_never_touches_module_processed_dir(
        self, cli_env, monkeypatch, tmp_path
    ):
        """Regression: build_dataset() used to unconditionally
        `_PROCESSED_DIR.mkdir()` (the module's own hardcoded data/processed/
        default) regardless of `output_path`, leaking an empty directory into
        the real project checkout on every test run. It must create only
        output_path's own parent."""
        sentinel = tmp_path / "should-never-be-created"
        monkeypatch.setattr(build_dataset_mod, "_PROCESSED_DIR", sentinel)
        build_dataset(output_path=cli_env["out"])
        assert not sentinel.exists()
        assert cli_env["out"].exists()
