"""
Tests for src/models/analysis.py (Phase 4 explainability/timing — Decision 014).

Covers:
  - measure_timing: one row per zoo candidate, timing columns, tuned-params
    override recorded in the output
  - permutation_importance_top1: one row per feature, Decision-013 class
    mapping, baseline attr, signal feature shows a positive top-1 drop
  - _case_study_rows: highest/lowest-confidence winner and the 2022-round-1
    regulation-reset race selection
  - _importance_plots: writes the two importance PNGs
  - shap_analysis: linear (logreg) and tree (random_forest) explainer paths
    produce the summary frame and plot artifacts; the pole heuristic raises
  - main CLI: nothing-to-do error, missing features.parquet, --timing run,
    and the full --model analysis run against a tmp MLflow store

All tests run on a small synthetic feature frame (pole always wins — a
perfectly learnable signal); no repository data files are required.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.features.metadata import FEATURE_CLASSIFICATION, active_feature_columns
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.analysis import (
    _case_study_rows,
    _importance_plots,
    main,
    measure_timing,
    permutation_importance_top1,
    shap_analysis,
)
from src.models.registry import MODEL_ZOO, get_model
from src.models.splits import temporal_split, to_xy

# ---------------------------------------------------------------------------
# Synthetic feature frame: driver on pole (grid_adjusted == 1) always wins.
# ---------------------------------------------------------------------------

def _synthetic_features(years, races_per_year=2, n_drivers=4, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, race_id = [], 0
    for year in years:
        for rnd in range(1, races_per_year + 1):
            race_id += 1
            grid = rng.permutation(n_drivers) + 1
            for driver in range(n_drivers):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row["grid_adjusted"] = float(grid[driver])
                row["grid_position_norm"] = float(grid[driver]) / n_drivers
                row.update({
                    "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                    "circuitId": 1, "year": year, "round": rnd,
                    TARGET_COLUMN: int(grid[driver] == 1),
                })
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def full_frame() -> pd.DataFrame:
    return _synthetic_features(range(2010, 2025))


@pytest.fixture(scope="module")
def split(full_frame):
    return temporal_split(full_frame)


@pytest.fixture(scope="module")
def fitted_logreg(split):
    X_tr, y_tr, _ = to_xy(split.train)
    pipeline = get_model("logreg", y_tr)
    pipeline.fit(X_tr, y_tr)
    return pipeline


# ---------------------------------------------------------------------------
# measure_timing
# ---------------------------------------------------------------------------

class TestMeasureTiming:
    def test_one_row_per_zoo_candidate(self, split):
        timing = measure_timing(split.train, split.val, n_predict_repeats=1)
        assert sorted(timing["model"]) == sorted(MODEL_ZOO)
        assert {"fit_seconds_train_split", "predict_seconds_val_split",
                "predict_ms_per_row", "params"} <= set(timing.columns)
        assert (timing["fit_seconds_train_split"] >= 0).all()
        assert (timing["predict_seconds_val_split"] >= 0).all()

    def test_params_override_is_recorded(self, split):
        timing = measure_timing(
            split.train, split.val,
            params_by_model={"logreg": {"model__C": 0.5}},
            n_predict_repeats=1,
        )
        logreg_row = timing[timing["model"] == "logreg"].iloc[0]
        assert "0.5" in logreg_row["params"]
        other = timing[timing["model"] != "logreg"]
        assert (other["params"] == "{}").all()


# ---------------------------------------------------------------------------
# permutation_importance_top1
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    @pytest.fixture(scope="class")
    def perm(self, fitted_logreg, split):
        return permutation_importance_top1(fitted_logreg, split.val, n_repeats=2)

    def test_one_row_per_feature(self, perm):
        # Decision 041: fitted_logreg was fit via the default (exclusion-
        # applied) feature set, not the raw full FEATURE_COLUMNS.
        assert sorted(perm["feature"]) == sorted(active_feature_columns())

    def test_baseline_attr_and_sorting(self, perm):
        assert 0.0 <= perm.attrs["baseline_top1"] <= 1.0
        assert perm["top1_drop_mean"].is_monotonic_decreasing

    def test_signal_feature_has_positive_drop(self, perm):
        """grid_adjusted IS the winner signal — shuffling it must hurt top-1."""
        drop = perm.loc[perm["feature"] == "grid_adjusted", "top1_drop_mean"]
        assert float(drop.iloc[0]) > 0

    def test_feature_classes_come_from_decision_013(self, perm):
        allowed = set(FEATURE_CLASSIFICATION.values()) | {"derived"}
        assert set(perm["feature_class"]) <= allowed


# ---------------------------------------------------------------------------
# _case_study_rows
# ---------------------------------------------------------------------------

class TestCaseStudyRows:
    def _val_frame(self, include_2022_r1=True):
        """Three 2-driver races; winner probs 0.9 / 0.2 / 0.5."""
        year1 = 2022 if include_2022_r1 else 2023
        val_df = pd.DataFrame({
            "year":  [year1, year1, 2022, 2022, 2023, 2023],
            "round": [1, 1, 2, 2, 3, 3],
        })
        y_val = np.array([1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.5])
        races = np.array([101, 101, 102, 102, 103, 103])
        return val_df, y_val, probs, races

    def test_selects_highest_and_lowest_confidence_winners(self):
        val_df, y_val, probs, races = self._val_frame()
        cases = _case_study_rows(val_df, y_val, probs, races)
        assert cases["winner_highest_confidence"] == 0    # prob 0.9
        assert cases["winner_lowest_confidence"] == 2     # prob 0.2

    def test_selects_2022_round1_reset_race_when_present(self):
        val_df, y_val, probs, races = self._val_frame(include_2022_r1=True)
        cases = _case_study_rows(val_df, y_val, probs, races)
        assert cases["winner_2022_round1_reset_race"] == 0

    def test_reset_race_key_absent_without_2022_round1(self):
        val_df, y_val, probs, races = self._val_frame(include_2022_r1=False)
        cases = _case_study_rows(val_df, y_val, probs, races)
        assert "winner_2022_round1_reset_race" not in cases


# ---------------------------------------------------------------------------
# _importance_plots
# ---------------------------------------------------------------------------

class TestImportancePlots:
    def test_writes_both_pngs(self, tmp_path):
        importances = pd.DataFrame({
            "feature": ["grid_adjusted", "reached_q3", "q1_sec"],
            "importance": [0.5, 0.3, 0.2],
            "feature_class": ["Stable", "Stable", "Experimental"],
        })
        _importance_plots(importances, "logreg", tmp_path)
        assert (tmp_path / "feature_importance_logreg.png").exists()
        assert (tmp_path / "importance_by_class_logreg.png").exists()


# ---------------------------------------------------------------------------
# shap_analysis
# ---------------------------------------------------------------------------

class TestShapAnalysis:
    def test_linear_explainer_path(self, fitted_logreg, split, tmp_path):
        summary = shap_analysis(
            fitted_logreg, "logreg", split.train, split.val, tmp_path
        )
        assert {"feature", "mean_abs_shap", "feature_class"} <= set(summary.columns)
        assert summary["mean_abs_shap"].is_monotonic_decreasing
        assert (summary["mean_abs_shap"] >= 0).all()
        assert (tmp_path / "shap_summary_logreg.png").exists()
        assert (tmp_path / "shap_bar_logreg.png").exists()
        assert list(tmp_path.glob("shap_dependence_logreg_*.png"))
        assert list(tmp_path.glob("shap_waterfall_logreg_*.png"))

    def test_tree_explainer_path(self, split, tmp_path):
        X_tr, y_tr, _ = to_xy(split.train)
        rf = get_model("random_forest", y_tr)
        rf.fit(X_tr, y_tr)
        summary = shap_analysis(rf, "random_forest", split.train, split.val, tmp_path)
        assert not summary.empty
        assert (tmp_path / "shap_summary_random_forest.png").exists()

    def test_pole_baseline_raises(self, split, tmp_path):
        X_tr, y_tr, _ = to_xy(split.train)
        pole = get_model("pole_baseline", y_tr)
        pole.fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="pole baseline"):
            shap_analysis(pole, "pole_baseline", split.train, split.val, tmp_path)


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    @pytest.fixture
    def cli_env(self, monkeypatch, tmp_path, full_frame):
        """Point the CLI's module globals at tmp copies; return the env."""
        features_path = tmp_path / "features.parquet"
        full_frame.to_parquet(features_path, index=False)
        reports_dir = tmp_path / "reports"
        monkeypatch.setattr("src.models.analysis.FEATURES_PATH", features_path)
        monkeypatch.setattr("src.models.analysis.REPORTS_DIR", reports_dir)
        monkeypatch.setattr(
            "src.models.analysis.data_fingerprint", lambda: "test-fp"
        )
        # main() creates its --experiment with the DEFAULT ./mlruns artifact
        # location — chdir so those artifacts land in tmp, not the checkout.
        monkeypatch.chdir(tmp_path)
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        yield {"reports_dir": reports_dir, "tracking_uri": tracking_uri}
        mlflow.set_tracking_uri(None)

    def test_nothing_to_do_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            main([])

    def test_missing_features_parquet_returns_1(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "src.models.analysis.FEATURES_PATH", tmp_path / "missing.parquet"
        )
        assert main(["--timing"]) == 1

    def test_timing_run_writes_csv(self, cli_env, capsys):
        rc = main(["--timing", "--tracking-uri", cli_env["tracking_uri"],
                   "--experiment", "test-analysis"])
        assert rc == 0
        assert (cli_env["reports_dir"] / "zoo_timing.csv").exists()
        timing = pd.read_csv(cli_env["reports_dir"] / "zoo_timing.csv")
        assert sorted(timing["model"]) == sorted(MODEL_ZOO)

    def test_model_analysis_run_writes_artifacts(self, cli_env):
        rc = main(["--model", "logreg", "--tracking-uri", cli_env["tracking_uri"],
                   "--experiment", "test-analysis"])
        assert rc == 0
        reports = cli_env["reports_dir"]
        assert (reports / "feature_importance_logreg.csv").exists()
        assert (reports / "permutation_importance_logreg.csv").exists()
        assert (reports / "shap_summary_logreg.csv").exists()
        assert (reports / "shap_summary_logreg.png").exists()
