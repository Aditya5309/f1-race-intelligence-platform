"""
Tests for src/models/season_tracking.py (2026 continuous tracking set).

Coverage:
  - the ongoing-era resolver + the report path it names (shifts with the
    era table, never hardcoded to a literal year)
  - STRUCTURAL isolation from the training/registration path: a fresh
    subprocess import must never pull in src.models.train or
    src.models.splits (the module docstring's guarantee, verified here
    rather than trusted)
  - scoring: new completed races get one row each, appended and persisted
  - idempotency: re-running with nothing new leaves the report unchanged
  - a race with no confirmed single winner yet is skipped, not scored
  - summarize()'s small-sample flag and race count
  - the CLI end to end against tmp paths only (never the real project's
    committed artifacts/)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models import eras, season_tracking
from src.models.evaluate import evaluate_all
from src.models.predict import load_model, predict_race
from src.models.registry import get_model
from src.models.serving_bundle import ModelInfo, export_bundle
from src.models.splits import to_xy

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ONGOING = eras.REGULATION_ERAS[-1]
assert _ONGOING.is_ongoing, "Test assumes the era table's final entry is open-ended."


# ---------------------------------------------------------------------------
# Synthetic data: pole (grid_adjusted == 1) always wins, so a fitted model
# separates cleanly from chance — same pattern as test_train.py/test_predict.py.
# ---------------------------------------------------------------------------

def _synthetic_races(
    year: int, race_ids: list[int], n_drivers: int = 4, complete: bool = True, seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year)
    rows = []
    for rnd, race_id in enumerate(race_ids, start=1):
        grid = rng.permutation(n_drivers) + 1
        for driver in range(n_drivers):
            row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            row["grid_adjusted"] = float(grid[driver])
            row["grid_position_norm"] = float(grid[driver]) / n_drivers
            row.update({
                "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                "circuitId": 1, "year": year, "round": rnd,
                TARGET_COLUMN: (int(grid[driver] == 1) if complete else 0),
            })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def fitted_bundle(tmp_path):
    """A logreg fit on pre-era (2010-2021) synthetic rows, exported as a
    plain serving bundle (no MLflow tracking store needed — mirrors
    test_serving_bundle.py's direct export_bundle() pattern)."""
    train_frame = pd.concat(
        [_synthetic_races(y, [y * 10 + 1, y * 10 + 2]) for y in range(2010, 2022)],
        ignore_index=True,
    )
    X, y, _ = to_xy(train_frame)
    model = get_model("logreg", y)
    model.fit(X, y)

    bundle_root = tmp_path / "serving"
    info = ModelInfo(
        name="f1-winner", version="1", alias="Staging", run_id="test-run",
        trained_at="2026-01-01T00:00:00+00:00", calibration="none",
        model_class="Pipeline",
    )
    export_bundle(model, info, bundle_root=bundle_root)
    return bundle_root


# ---------------------------------------------------------------------------
# Ongoing-era resolution + report naming
# ---------------------------------------------------------------------------

def test_ongoing_era_is_the_final_open_ended_entry():
    assert season_tracking._ongoing_era() is _ONGOING


def test_default_report_path_named_after_ongoing_era_start_year(tmp_path):
    path = season_tracking.default_report_path(tracking_dir=tmp_path)
    assert path == tmp_path / f"{_ONGOING.start_year}_running_eval.csv"


# ---------------------------------------------------------------------------
# Structural isolation from the training/registration path
# ---------------------------------------------------------------------------

def test_never_imports_train_or_splits_in_a_clean_interpreter():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import src.models.season_tracking; "
         "assert 'src.models.train' not in sys.modules, 'imported train.py'; "
         "assert 'src.models.splits' not in sys.modules, 'imported splits.py'; "
         "print('OK')"],
        cwd=_PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Scoring + persistence
# ---------------------------------------------------------------------------

def test_scores_new_completed_races_one_row_each(fitted_bundle, tmp_path):
    era_year = _ONGOING.start_year
    features = pd.concat([
        _synthetic_races(2015, [150001]),                        # pre-era: must be excluded
        _synthetic_races(era_year, [era_year * 10 + 1, era_year * 10 + 2]),
    ], ignore_index=True)
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    master = features[["raceId", "driverId", TARGET_COLUMN]].copy()
    # positionOrder: winner=1, everyone else in some fixed order after it.
    master["positionOrder"] = master.groupby("raceId")[TARGET_COLUMN].rank(
        ascending=False, method="first"
    ).astype(int)
    master_path = tmp_path / "master_dataset.parquet"
    master.to_parquet(master_path, index=False)

    report_path = tmp_path / "report.csv"
    result = season_tracking.score_new_races(
        alias="Staging", bundle_root=fitted_bundle,
        features_path=features_path, master_path=master_path,
        report_path=report_path,
    )

    assert len(result) == 2   # only the two era-year races, never the 2015 one
    assert set(result["race_id"]) == {era_year * 10 + 1, era_year * 10 + 2}
    assert (result["year"] == era_year).all()
    assert report_path.exists()
    assert "top1_accuracy" in result.columns
    assert "spearman_corr" in result.columns

    # Cross-check against an independent, direct evaluate_all() call on the
    # same races/model — proves score_new_races wires the real prediction
    # through correctly, without assuming anything about the model's actual
    # skill (a tiny synthetic fit is not guaranteed to separate perfectly).
    era_rows = features[features["year"] == era_year]
    model, _ = load_model(fitted_bundle / "staging")
    predictions = predict_race(model, era_rows)
    merged = predictions.merge(
        era_rows[["raceId", "driverId", TARGET_COLUMN]], on=["raceId", "driverId"],
    )
    expected = evaluate_all(
        merged[TARGET_COLUMN].to_numpy(), merged["win_probability_raw"].to_numpy(),
        merged["raceId"].to_numpy(),
    )
    assert result["top1_accuracy"].mean() == pytest.approx(expected["top1_accuracy"])


def test_idempotent_rerun_does_not_rescore_or_duplicate(fitted_bundle, tmp_path):
    era_year = _ONGOING.start_year
    features = _synthetic_races(era_year, [era_year * 10 + 1])
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)
    report_path = tmp_path / "report.csv"

    first = season_tracking.score_new_races(
        alias="Staging", bundle_root=fitted_bundle,
        features_path=features_path, master_path=tmp_path / "no-such-file.parquet",
        report_path=report_path,
    )
    second = season_tracking.score_new_races(
        alias="Staging", bundle_root=fitted_bundle,
        features_path=features_path, master_path=tmp_path / "no-such-file.parquet",
        report_path=report_path,
    )

    assert len(first) == 1
    pd.testing.assert_frame_equal(first, second)


def test_incomplete_race_is_skipped_not_scored(fitted_bundle, tmp_path, capsys):
    era_year = _ONGOING.start_year
    features = _synthetic_races(era_year, [era_year * 10 + 1], complete=False)   # no confirmed winner
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)
    report_path = tmp_path / "report.csv"

    result = season_tracking.score_new_races(
        alias="Staging", bundle_root=fitted_bundle,
        features_path=features_path, master_path=tmp_path / "no-such-file.parquet",
        report_path=report_path,
    )

    assert result.empty
    assert not report_path.exists()
    assert "skipping" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------

def test_summarize_no_report_yet(tmp_path):
    summary = season_tracking.summarize(report_path=tmp_path / "missing.csv", era=_ONGOING)
    assert summary["n_races"] == 0
    assert summary["small_sample"] is True


def test_summarize_small_sample_flag_and_race_count(tmp_path):
    report_path = tmp_path / "report.csv"
    # Fabricate a report directly — the threshold logic doesn't need a real
    # scoring pass to verify, just enough rows either side of the boundary.
    columns = [
        "race_id", "year", "round", "n_drivers", "top1_accuracy", "top3_recall",
        "winner_mrr", "avg_winner_probability", "median_winner_rank",
        "log_loss", "brier_score", "ece", "n_races", "n_rows",
    ]
    small = pd.DataFrame(
        [{**{c: 1.0 for c in columns}, "race_id": i} for i in range(10)]
    )
    small.to_csv(report_path, index=False)
    summary = season_tracking.summarize(report_path=report_path, era=_ONGOING)
    assert summary["n_races"] == 10
    assert summary["small_sample"] is True

    large = pd.DataFrame(
        [{**{c: 1.0 for c in columns}, "race_id": i} for i in range(30)]
    )
    large.to_csv(report_path, index=False)
    summary = season_tracking.summarize(report_path=report_path, era=_ONGOING)
    assert summary["n_races"] == 30
    assert summary["small_sample"] is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_scores_and_prints_race_count_next_to_every_metric(fitted_bundle, tmp_path, capsys):
    era_year = _ONGOING.start_year
    features = _synthetic_races(era_year, [era_year * 10 + 1, era_year * 10 + 2])
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)
    tracking_dir = tmp_path / "tracking"

    rc = season_tracking.main([
        "--alias", "Staging",
        "--bundle-root", str(fitted_bundle),
        "--features-path", str(features_path),
        "--master-path", str(tmp_path / "no-such-file.parquet"),
        "--tracking-dir", str(tracking_dir),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "n=2 race(s) scored" in out
    assert "(n=2 races)" in out   # every metric line carries the count
    assert (tracking_dir / f"{era_year}_running_eval.csv").exists()
