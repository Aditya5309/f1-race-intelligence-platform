"""Tests for scripts/export_analysis_data.py."""

import pandas as pd
import pytest

from scripts.export_analysis_data import ANALYSIS_FILES, export_analysis_data, main


def _write_source(root):
    (root / "phase4_analysis").mkdir(parents=True)
    pd.DataFrame({"model": ["logreg"], "selected": ["Yes"]}).to_csv(
        root / "model_comparison.csv", index=False)
    pd.DataFrame({"feature": ["grid_adjusted"], "importance": [0.5],
                  "feature_class": ["stable"]}).to_csv(
        root / "phase4_analysis" / "feature_importance_logreg.csv", index=False)
    pd.DataFrame({"feature": ["grid_adjusted"], "mean_abs_shap": [0.4],
                  "feature_class": ["stable"]}).to_csv(
        root / "phase4_analysis" / "shap_summary_logreg.csv", index=False)


def test_export_copies_exactly_the_three_files(tmp_path):
    source = tmp_path / "reports"
    dest = tmp_path / "artifacts" / "analysis"
    _write_source(source)

    copied = export_analysis_data(source, dest)

    assert set(copied) == {name for _, name in ANALYSIS_FILES}
    for _, name in ANALYSIS_FILES:
        assert (dest / name).exists()
    # Not exported: nothing beyond the three named files.
    assert sorted(p.name for p in dest.iterdir()) == sorted(copied)


def test_export_raises_when_source_incomplete(tmp_path):
    source = tmp_path / "reports"
    dest = tmp_path / "artifacts" / "analysis"
    source.mkdir()
    (source / "phase4_analysis").mkdir()
    # Only model_comparison.csv present — the other two are missing.
    pd.DataFrame({"model": ["logreg"]}).to_csv(source / "model_comparison.csv", index=False)

    with pytest.raises(FileNotFoundError, match="feature_importance_logreg.csv"):
        export_analysis_data(source, dest)


def test_cli_success(tmp_path, capsys):
    source = tmp_path / "reports"
    dest = tmp_path / "artifacts" / "analysis"
    _write_source(source)

    exit_code = main(["--source", str(source), "--dest", str(dest)])

    assert exit_code == 0
    assert "Exported 3 files" in capsys.readouterr().out
    assert (dest / "model_comparison.csv").exists()


def test_cli_missing_source_returns_1(tmp_path, capsys):
    exit_code = main(["--source", str(tmp_path / "no-such-dir"), "--dest", str(tmp_path / "dest")])

    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().err
