"""
Tests for scripts/refresh_and_freeze.py (Phase 4 Tranche D, Part 2).

Unit-level: mocks subprocess.run to verify sequencing, stop-on-first-failure,
and the --automated/manual export-mode wiring — NOT a real pipeline run (see
the real end-to-end verification performed manually against this project's
actual data/ tree and reported in Decision 035 / the session's own report).

scripts/ is not a package — loaded via importlib, same pattern as
promote_model.py's and ingest_jolpica.py's test files.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_and_freeze", _PROJECT_ROOT / "scripts" / "refresh_and_freeze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refresh_and_freeze = _load_module()


class _FakeResult:
    def __init__(self, returncode: int):
        self.returncode = returncode


def _params_file(tmp_path, model="logreg", calibrate=True) -> Path:
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"model": model, "calibrate": calibrate, "params": {"model__C": 0.5}}))
    return path


def test_runs_all_eight_steps_in_order_manual_mode(tmp_path):
    params_file = _params_file(tmp_path)
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main(["--params-file", str(params_file)])

    assert rc == 0
    commands = [call.args[0] for call in mock_run.call_args_list]
    assert len(commands) == 8
    assert "ingest_jolpica.py" in commands[0][1]
    assert commands[1][2:4] == ["src.data.build_interim", "--target"]
    assert commands[2][2] == "src.pipelines.build_dataset"
    assert commands[3][2] == "src.features.pipeline"
    assert commands[4][2] == "src.models.season_tracking"
    assert "refresh_features_snapshot.py" in commands[5][1]
    assert "export_display_data.py" in commands[6][1]
    assert commands[7][2] == "src.models.train"
    assert "--no-export" not in commands[7]   # manual mode: immediate export


def test_display_dest_passed_through_to_export_display_data_step(tmp_path):
    """The real gap found during manual verification: without this,
    --automated mode still overwrites the real committed artifacts/display/
    — display refresh is deliberately never gated (no "good vs bad" concept
    for it), so it needs its OWN hermetic-testing override, separate from
    --bundle-root/--artifacts-root which only protect the registration step."""
    params_file = _params_file(tmp_path)
    display_dest = tmp_path / "display"
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main([
            "--automated", "--params-file", str(params_file),
            "--display-dest", str(display_dest),
        ])

    assert rc == 0
    display_cmd = mock_run.call_args_list[6].args[0]
    assert "export_display_data.py" in display_cmd[1]
    assert "--dest" in display_cmd
    assert str(display_dest) in display_cmd


def test_artifacts_root_passed_through_to_both_features_snapshot_and_register_steps(tmp_path):
    """Part 1 fix: --artifacts-root must redirect BOTH writers of
    artifacts/features.parquet — the always-on refresh_features_snapshot.py
    step AND train.py --register — not just the gated one. Missing this
    would mean a hermetic test (or a real --automated run) still overwrites
    the real committed artifacts/features.parquet via the always-on step
    even when --artifacts-root is set."""
    params_file = _params_file(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main([
            "--automated", "--params-file", str(params_file),
            "--artifacts-root", str(artifacts_root),
        ])

    assert rc == 0
    features_snapshot_cmd = mock_run.call_args_list[5].args[0]
    assert "refresh_features_snapshot.py" in features_snapshot_cmd[1]
    assert "--artifacts-root" in features_snapshot_cmd
    assert str(artifacts_root) in features_snapshot_cmd

    register_cmd = mock_run.call_args_list[-1].args[0]
    assert "--artifacts-root" in register_cmd
    assert str(artifacts_root) in register_cmd


def test_features_snapshot_step_always_runs_even_without_artifacts_root_override(tmp_path):
    """Never gated on --automated, same reasoning as display refresh —
    verify the step fires with no --artifacts-root flag at all (its own
    script default then applies)."""
    params_file = _params_file(tmp_path)
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main(["--automated", "--params-file", str(params_file)])

    assert rc == 0
    features_snapshot_cmd = mock_run.call_args_list[5].args[0]
    assert "refresh_features_snapshot.py" in features_snapshot_cmd[1]
    assert "--artifacts-root" not in features_snapshot_cmd


def test_tracking_step_always_runs_and_forwards_bundle_root_and_tracking_dir(tmp_path):
    """Tracking must run even in --automated mode (never gated, same as
    display refresh) and must read --bundle-root (the CURRENTLY served
    bundle) rather than only the register step's write target."""
    params_file = _params_file(tmp_path)
    bundle_root = tmp_path / "serving"
    tracking_dir = tmp_path / "tracking"
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main([
            "--automated", "--params-file", str(params_file),
            "--bundle-root", str(bundle_root),
            "--tracking-dir", str(tracking_dir),
        ])

    assert rc == 0
    tracking_cmd = mock_run.call_args_list[4].args[0]
    assert tracking_cmd[2] == "src.models.season_tracking"
    assert "--bundle-root" in tracking_cmd
    assert str(bundle_root) in tracking_cmd
    assert "--tracking-dir" in tracking_cmd
    assert str(tracking_dir) in tracking_cmd


def test_automated_mode_registers_with_no_export(tmp_path):
    params_file = _params_file(tmp_path)
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main(["--automated", "--params-file", str(params_file)])

    assert rc == 0
    register_cmd = mock_run.call_args_list[-1].args[0]
    assert "--no-export" in register_cmd


def test_register_command_reads_model_and_calibrate_from_config(tmp_path):
    """The real bug this guards against: hardcoding --model would silently
    apply a different candidate's hyperparameters to the wrong model class."""
    params_file = _params_file(tmp_path, model="random_forest", calibrate=False)
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        refresh_and_freeze.main(["--params-file", str(params_file)])

    register_cmd = mock_run.call_args_list[-1].args[0]
    assert "random_forest" in register_cmd
    assert "logreg" not in register_cmd
    assert "--calibrate" not in register_cmd   # config said calibrate: false


def test_stops_at_first_failing_step_and_runs_nothing_after(tmp_path):
    params_file = _params_file(tmp_path)
    results = [_FakeResult(0), _FakeResult(0), _FakeResult(1)]   # step 3 (build_dataset) fails
    with patch.object(refresh_and_freeze.subprocess, "run", side_effect=results) as mock_run:
        rc = refresh_and_freeze.main(["--params-file", str(params_file)])

    assert rc == 1
    assert mock_run.call_count == 3   # steps 4/5/6/7/8 never ran


def test_skip_ingest_runs_seven_steps_not_eight(tmp_path):
    params_file = _params_file(tmp_path)
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main(["--skip-ingest", "--params-file", str(params_file)])

    assert rc == 0
    assert mock_run.call_count == 7


def test_dry_run_stops_after_ingest_step(tmp_path):
    params_file = _params_file(tmp_path)
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main(["--dry-run", "--params-file", str(params_file)])

    assert rc == 0
    assert mock_run.call_count == 1
    assert "--dry-run" in mock_run.call_args_list[0].args[0]


def test_missing_params_file_returns_1_before_running_any_step(tmp_path):
    """Checked up front — a config typo must not waste time on steps 1-7
    before failing at registration."""
    with patch.object(refresh_and_freeze.subprocess, "run", return_value=_FakeResult(0)) as mock_run:
        rc = refresh_and_freeze.main(["--params-file", str(tmp_path / "missing.json")])
    assert rc == 1
    assert mock_run.call_count == 0
