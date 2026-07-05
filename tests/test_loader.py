"""
Tests for src/data/loader.py

Covers:
  - Ergast null-token mapping (\\N, empty string, NA, N/A) to NaN
  - Ordinary values are loaded unchanged
  - FileNotFoundError for a missing CSV (with the path in the message)
  - load_all: every CSV in a directory, keyed by filename; empty directory
  - profile: shape/dtype/numeric-summary/missing-value report sections
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DATA_DIR, load_all, load_csv, profile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(directory: Path, name: str, text: str) -> Path:
    """Write raw CSV text to directory/name and return the path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_csv — null-token mapping
# ---------------------------------------------------------------------------

class TestLoadCsvNullTokens:
    def test_backslash_n_maps_to_nan(self, tmp_path):
        r"""Ergast MySQL dumps encode NULL as \N — must load as NaN."""
        _write_csv(tmp_path, "results.csv", "raceId,position\n1,\\N\n2,3\n")
        df = load_csv("results.csv", data_dir=tmp_path)
        assert pd.isna(df.loc[0, "position"])
        assert df.loc[1, "position"] == 3

    def test_empty_string_maps_to_nan(self, tmp_path):
        _write_csv(tmp_path, "t.csv", "a,b\n1,\n2,x\n")
        df = load_csv("t.csv", data_dir=tmp_path)
        assert pd.isna(df.loc[0, "b"])
        assert df.loc[1, "b"] == "x"

    def test_na_token_maps_to_nan(self, tmp_path):
        _write_csv(tmp_path, "t.csv", "a,b\n1,NA\n")
        df = load_csv("t.csv", data_dir=tmp_path)
        assert pd.isna(df.loc[0, "b"])

    def test_n_slash_a_token_maps_to_nan(self, tmp_path):
        _write_csv(tmp_path, "t.csv", "a,b\n1,N/A\n")
        df = load_csv("t.csv", data_dir=tmp_path)
        assert pd.isna(df.loc[0, "b"])

    def test_null_tokens_do_not_swallow_real_values(self, tmp_path):
        """Strings that merely contain N are not nulls."""
        _write_csv(
            tmp_path, "t.csv",
            "code,name\nNOR,Norris\nN1,Nico\n",
        )
        df = load_csv("t.csv", data_dir=tmp_path)
        assert df["code"].tolist() == ["NOR", "N1"]
        assert df["name"].tolist() == ["Norris", "Nico"]

    def test_all_null_column_is_fully_nan(self, tmp_path):
        r"""A column of only \N tokens (e.g. q3 for eliminated drivers)."""
        _write_csv(tmp_path, "t.csv", "raceId,q3\n1,\\N\n2,\\N\n")
        df = load_csv("t.csv", data_dir=tmp_path)
        assert df["q3"].isna().all()


# ---------------------------------------------------------------------------
# load_csv — basic behavior and errors
# ---------------------------------------------------------------------------

class TestLoadCsvBasics:
    def test_loads_shape_and_values(self, tmp_path):
        _write_csv(
            tmp_path, "races.csv",
            "raceId,year,round\n1,2024,1\n2,2024,2\n3,2024,3\n",
        )
        df = load_csv("races.csv", data_dir=tmp_path)
        assert df.shape == (3, 3)
        assert df["year"].tolist() == [2024, 2024, 2024]
        assert df["round"].tolist() == [1, 2, 3]

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_csv("nonexistent.csv", data_dir=tmp_path)
        assert "nonexistent.csv" in str(excinfo.value)

    def test_default_data_dir_is_project_data(self):
        """DATA_DIR points at the repository's data/ directory."""
        assert DATA_DIR.name == "data"


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------

class TestLoadAll:
    def test_loads_every_csv_keyed_by_filename(self, tmp_path):
        _write_csv(tmp_path, "drivers.csv", "driverId\n1\n2\n")
        _write_csv(tmp_path, "races.csv", "raceId\n1\n")
        datasets = load_all(data_dir=tmp_path)
        assert set(datasets.keys()) == {"drivers.csv", "races.csv"}
        assert len(datasets["drivers.csv"]) == 2
        assert len(datasets["races.csv"]) == 1

    def test_values_are_dataframes(self, tmp_path):
        _write_csv(tmp_path, "a.csv", "x\n1\n")
        datasets = load_all(data_dir=tmp_path)
        assert all(isinstance(v, pd.DataFrame) for v in datasets.values())

    def test_null_tokens_apply_through_load_all(self, tmp_path):
        _write_csv(tmp_path, "a.csv", "x,y\n1,\\N\n")
        datasets = load_all(data_dir=tmp_path)
        assert pd.isna(datasets["a.csv"].loc[0, "y"])

    def test_ignores_non_csv_files(self, tmp_path):
        _write_csv(tmp_path, "a.csv", "x\n1\n")
        (tmp_path / "notes.txt").write_text("not a csv", encoding="utf-8")
        (tmp_path / "b.parquet").write_bytes(b"\x00")
        datasets = load_all(data_dir=tmp_path)
        assert set(datasets.keys()) == {"a.csv"}

    def test_empty_directory_returns_empty_dict(self, tmp_path, capsys):
        datasets = load_all(data_dir=tmp_path)
        assert datasets == {}
        assert "No CSV files found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

class TestProfile:
    def test_reports_shape_and_dtypes(self, capsys):
        df = pd.DataFrame({"raceId": [1, 2], "name": ["a", "b"]})
        profile(df, "races.csv")
        out = capsys.readouterr().out
        assert "races.csv" in out
        assert "2 rows" in out
        assert "raceId" in out
        assert "name" in out

    def test_reports_numeric_summary(self, capsys):
        df = pd.DataFrame({"points": [10.0, 20.0]})
        profile(df, "t")
        out = capsys.readouterr().out
        assert "Numeric summary:" in out
        assert "points" in out

    def test_no_numeric_summary_for_all_string_frame(self, capsys):
        df = pd.DataFrame({"name": ["a", "b"]})
        profile(df, "t")
        out = capsys.readouterr().out
        assert "Numeric summary:" not in out

    def test_reports_no_missing_values(self, capsys):
        df = pd.DataFrame({"a": [1, 2]})
        profile(df, "t")
        assert "Missing values: none" in capsys.readouterr().out

    def test_reports_missing_value_counts(self, capsys):
        df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
        profile(df, "t")
        out = capsys.readouterr().out
        assert "Missing values:" in out
        assert "Missing values: none" not in out
        # Only the column with nulls appears in the report block
        assert out.rstrip().splitlines()[-1].strip().startswith("a")
