"""
Tests for src/models/splits.py (Phase 4 module 1 — Decision 012).

Covers the design-doc Section 11.2 split-integrity leakage checks plus the
Section 14.4 forward-holdout guard:

  - Decision-008 year boundaries exact per split
  - Disjoint raceIds across train/val/test
  - Forward holdout (2025+) rows never appear in ANY split
  - Empty-window and missing-column failures are loud
  - season_folds: expanding-window monotonicity, val years 2016..2021,
    every train season strictly precedes its validation season, race
    integrity, rejection of non-training-split input, determinism
  - to_xy: design matrix is exactly FEATURE_COLUMNS (no identifiers, no
    post-race columns), y is the target, groups are raceIds
  - Real-data smoke test: measured Decision-008 split sizes
"""

import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.integration.build_master_dataset import POST_RACE_OUTCOME_COLUMNS
from src.models.splits import (
    DEFAULT_N_FOLDS,
    FORWARD_HOLDOUT_MIN_YEAR,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    season_folds,
    temporal_split,
    to_xy,
)


# ---------------------------------------------------------------------------
# Fixture builder — minimal feature-frame rows (2 drivers per race).
# ---------------------------------------------------------------------------

def _frame(years: list[int], races_per_year: int = 2) -> pd.DataFrame:
    rows = []
    race_id = 0
    for year in years:
        for rnd in range(1, races_per_year + 1):
            race_id += 1
            for driver in (1, 2):
                row = {
                    "raceId": race_id, "driverId": driver, "constructorId": 1,
                    "circuitId": 1, "year": year, "round": rnd,
                    TARGET_COLUMN: int(driver == 1),
                }
                row.update({c: 0.0 for c in FEATURE_COLUMNS})
                rows.append(row)
    return pd.DataFrame(rows)


def _full_range_frame() -> pd.DataFrame:
    return _frame(list(range(2010, 2027)))   # includes forward-holdout years


# ---------------------------------------------------------------------------
# temporal_split — Decision-008 boundaries and integrity
# ---------------------------------------------------------------------------

def test_split_year_boundaries_exact():
    split = temporal_split(_full_range_frame())
    assert sorted(split.train["year"].unique()) == list(range(2010, 2022))
    assert sorted(split.val["year"].unique()) == [2022, 2023]
    assert sorted(split.test["year"].unique()) == [2024]


def test_split_raceids_disjoint():
    split = temporal_split(_full_range_frame())
    train, val, test = (set(s["raceId"]) for s in (split.train, split.val, split.test))
    assert not (train & val) and not (train & test) and not (val & test)


def test_forward_holdout_never_enters_any_split():
    # THE Section 14.4 guard: 2025-2026 rows exist in the input and must be
    # absent from every split — a max(year)-derived split would swallow them.
    df = _full_range_frame()
    assert (df["year"] >= FORWARD_HOLDOUT_MIN_YEAR).any()   # fixture sanity
    split = temporal_split(df)
    for part in (split.train, split.val, split.test):
        assert int(part["year"].max()) < FORWARD_HOLDOUT_MIN_YEAR
    total_used = len(split.train) + len(split.val) + len(split.test)
    n_holdout = int((df["year"] >= FORWARD_HOLDOUT_MIN_YEAR).sum())
    n_pre_window = int((df["year"] < TRAIN_YEARS[0]).sum())
    assert total_used == len(df) - n_holdout - n_pre_window


def test_split_races_never_cut():
    split = temporal_split(_full_range_frame())
    df = _full_range_frame()
    for part in (split.train, split.val, split.test):
        for race_id, group in part.groupby("raceId"):
            assert len(group) == len(df[df["raceId"] == race_id])


def test_empty_window_raises():
    with pytest.raises(ValueError, match="empty"):
        temporal_split(_frame([2010, 2011]))     # no val/test years


def test_missing_year_column_raises():
    with pytest.raises(KeyError, match="year"):
        temporal_split(pd.DataFrame({"raceId": [1]}))


# ---------------------------------------------------------------------------
# season_folds — expanding-window CV
# ---------------------------------------------------------------------------

def test_folds_validate_last_six_seasons():
    split = temporal_split(_full_range_frame())
    folds = season_folds(split.train)
    assert len(folds) == DEFAULT_N_FOLDS
    assert [f.val_year for f in folds] == [2016, 2017, 2018, 2019, 2020, 2021]


def test_folds_expand_monotonically_and_precede_val():
    folds = season_folds(temporal_split(_full_range_frame()).train)
    prev_train_rows = 0
    for fold in folds:
        assert len(fold.train) > prev_train_rows          # expanding window
        prev_train_rows = len(fold.train)
        assert int(fold.train["year"].max()) < fold.val_year   # strictly prior
        assert set(fold.val["year"].unique()) == {fold.val_year}
        assert fold.train_years == (2010, fold.val_year - 1)
        # Race integrity across the fold boundary.
        assert not set(fold.train["raceId"]) & set(fold.val["raceId"])


def test_folds_reject_non_training_input():
    # Passing val/test years into the fold generator would leak them into
    # model selection — must be loud (design doc Section 11.2).
    with pytest.raises(ValueError, match="training split only"):
        season_folds(_frame([2010, 2011, 2022]))


def test_folds_reject_invalid_n_folds():
    train = temporal_split(_full_range_frame()).train
    with pytest.raises(ValueError, match="n_folds"):
        season_folds(train, n_folds=12)   # 12 seasons -> no train season left
    with pytest.raises(ValueError, match="n_folds"):
        season_folds(train, n_folds=0)


def test_folds_deterministic():
    train = temporal_split(_full_range_frame()).train
    a, b = season_folds(train), season_folds(train)
    for fa, fb in zip(a, b):
        assert fa.val_year == fb.val_year
        pd.testing.assert_frame_equal(fa.train, fb.train)
        pd.testing.assert_frame_equal(fa.val, fb.val)


# ---------------------------------------------------------------------------
# to_xy — design-matrix integrity (Section 11.1)
# ---------------------------------------------------------------------------

def test_to_xy_matrix_is_exactly_feature_columns():
    split = temporal_split(_full_range_frame())
    X, y, race_ids = to_xy(split.train)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    # No identifiers, no post-race outcome columns in the design matrix.
    assert not {"raceId", "driverId", "constructorId", "circuitId", "year", "round"} & set(X.columns)
    assert not POST_RACE_OUTCOME_COLUMNS & set(X.columns)
    assert (y == split.train[TARGET_COLUMN]).all()
    assert (race_ids == split.train["raceId"]).all()
    assert len(X) == len(y) == len(race_ids) == len(split.train)


def test_to_xy_missing_feature_raises():
    df = temporal_split(_full_range_frame()).train.drop(columns=["grid_adjusted"])
    with pytest.raises(KeyError, match="grid_adjusted"):
        to_xy(df)


# ---------------------------------------------------------------------------
# Real-data smoke test — measured Decision-008 sizes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/features.parquet").exists(),
    reason="features.parquet not built",
)
def test_real_data_split_sizes():
    df = pd.read_parquet("data/processed/features.parquet")
    split = temporal_split(df)
    assert (len(split.train), len(split.val), len(split.test)) == (5077, 880, 479)
    assert (split.train["raceId"].nunique(), split.val["raceId"].nunique(),
            split.test["raceId"].nunique()) == (237, 44, 24)
    # Winner counts: one per race (modeling window has no shared drives).
    assert int(split.train[TARGET_COLUMN].sum()) == 237
    assert int(split.test[TARGET_COLUMN].sum()) == 24
    folds = season_folds(split.train)
    assert [f.val_year for f in folds] == list(range(2016, 2022))
    # First fold trains on 2010-2015.
    assert sorted(folds[0].train["year"].unique()) == list(range(2010, 2016))
