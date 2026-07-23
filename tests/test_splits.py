"""
Tests for src/models/splits.py.

Covers the split-integrity leakage checks plus the forward-holdout guard:

  - Year boundaries exact per split (historical default)
  - Disjoint raceIds across train/val/test
  - Forward holdout (2025+) rows never appear in ANY split by default
  - Empty-window and missing-column failures are loud
  - season_folds: expanding-window monotonicity, val years 2016..2021,
    every train season strictly precedes its validation season, race
    integrity, rejection of non-training-split input, determinism
  - to_xy: design matrix is exactly FEATURE_COLUMNS (no identifiers, no
    post-race columns), y is the target, groups are raceIds
  - Real-data smoke test: measured split sizes
  - SplitStrategy presets: preset window boundaries, construction-time
    validation (inverted/overlapping windows, holdout opt-in), unknown-name
    resolution, rolling-window factory arithmetic, ground_effect's clear
    error when 2025 data is absent, season_folds strategy awareness, and
    exact backward compatibility of the historical default
  - Regulation-era domain model: every preset carries the
    correct EvaluationObjective; within-era presets stay entirely inside
    one regulation era; within_era_strategy arithmetic, ongoing-era and
    too-short-era rejection; historical preset provably crosses the 2022
    era boundary (its stated purpose)
"""

import pandas as pd
import pytest

from src.features.metadata import active_feature_columns
from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.integration.build_master_dataset import POST_RACE_OUTCOME_COLUMNS
from src.models import eras
from src.models.splits import (
    DEFAULT_N_FOLDS,
    FORWARD_HOLDOUT_MIN_YEAR,
    GROUND_EFFECT,
    HISTORICAL,
    HYBRID_ERA,
    STRATEGIES,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    EvaluationObjective,
    SplitStrategy,
    get_strategy,
    rolling_window_strategy,
    season_folds,
    temporal_split,
    to_xy,
    within_era_strategy,
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
# temporal_split — boundaries and integrity
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
    # model selection — must be loud, not a silent no-op.
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
# to_xy — design-matrix integrity
# ---------------------------------------------------------------------------

def test_to_xy_matrix_is_exactly_active_feature_columns_by_default():
    # The DEFAULT contract is active_feature_columns() (the
    # training-exclusion-applied set, currently FEATURE_COLUMNS minus
    # wet_form) — never the raw FEATURE_COLUMNS unless explicitly
    # requested (see test_to_xy_explicit_override_uses_full_feature_columns
    # below).
    split = temporal_split(_full_range_frame())
    X, y, race_ids = to_xy(split.train)
    assert list(X.columns) == list(active_feature_columns())
    assert "driver_wet_dry_delta" not in X.columns
    # No identifiers, no post-race outcome columns in the design matrix.
    assert not {"raceId", "driverId", "constructorId", "circuitId", "year", "round"} & set(X.columns)
    assert not POST_RACE_OUTCOME_COLUMNS & set(X.columns)
    assert (y == split.train[TARGET_COLUMN]).all()
    assert (race_ids == split.train["raceId"]).all()
    assert len(X) == len(y) == len(race_ids) == len(split.train)


def test_to_xy_explicit_override_uses_full_feature_columns():
    """An explicit feature_columns= override is how research/
    ablation work deliberately opts into the full, unexcluded set —
    never the silent default."""
    split = temporal_split(_full_range_frame())
    X, y, race_ids = to_xy(split.train, feature_columns=FEATURE_COLUMNS)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert "driver_wet_dry_delta" in X.columns


def test_to_xy_missing_feature_raises():
    df = temporal_split(_full_range_frame()).train.drop(columns=["grid_adjusted"])
    with pytest.raises(KeyError, match="grid_adjusted"):
        to_xy(df)


# ---------------------------------------------------------------------------
# SplitStrategy presets and validation
# ---------------------------------------------------------------------------

def test_historical_default_is_backward_compatible():
    # No-argument call must be byte-identical to the explicit historical
    # preset — every existing caller depends on this.
    df = _full_range_frame()
    default_split = temporal_split(df)
    explicit_split = temporal_split(df, strategy="historical")
    assert default_split.strategy is HISTORICAL
    for a, b in zip(
        (default_split.train, default_split.val, default_split.test),
        (explicit_split.train, explicit_split.val, explicit_split.test),
    ):
        pd.testing.assert_frame_equal(a, b)
    assert (TRAIN_YEARS, VAL_YEARS, TEST_YEARS) == (
        HISTORICAL.train_years, HISTORICAL.val_years, HISTORICAL.test_years)


def test_hybrid_era_boundaries_exact():
    # Entirely within the hybrid regulations (2014-2021) — an earlier
    # definition tested on 2022, which was cross-era.
    split = temporal_split(_full_range_frame(), strategy="hybrid_era")
    assert sorted(split.train["year"].unique()) == list(range(2014, 2020))
    assert sorted(split.val["year"].unique()) == [2020]
    assert sorted(split.test["year"].unique()) == [2021]
    assert split.strategy is HYBRID_ERA


def test_ground_effect_boundaries_and_holdout_opt_in():
    # ground_effect legitimately tests on 2025 — but only because the
    # preset declares allow_forward_holdout=True.
    split = temporal_split(_full_range_frame(), strategy="ground_effect")
    assert sorted(split.train["year"].unique()) == [2022, 2023]
    assert sorted(split.val["year"].unique()) == [2024]
    assert sorted(split.test["year"].unique()) == [2025]
    assert GROUND_EFFECT.allow_forward_holdout


def test_ground_effect_missing_2025_raises_clear_error():
    with pytest.raises(ValueError, match=r"empty 'test' set for years 2025-2025"):
        temporal_split(_frame(list(range(2010, 2025))), strategy="ground_effect")


def test_strategy_raceids_disjoint_for_all_presets():
    df = _full_range_frame()
    for name in STRATEGIES:
        split = temporal_split(df, strategy=name)
        train, val, test = (
            set(s["raceId"]) for s in (split.train, split.val, split.test))
        assert not (train & val) and not (train & test) and not (val & test)


def test_strategy_rejects_inverted_window():
    with pytest.raises(ValueError, match="inverted"):
        SplitStrategy("bad", (2021, 2010), (2022, 2023), (2024, 2024))


def test_strategy_rejects_overlapping_windows():
    with pytest.raises(ValueError, match="ordered"):
        SplitStrategy("bad", (2010, 2022), (2022, 2023), (2024, 2024))
    with pytest.raises(ValueError, match="ordered"):
        SplitStrategy("bad", (2010, 2021), (2022, 2024), (2024, 2024))


def test_strategy_holdout_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="allow_forward_holdout"):
        SplitStrategy("bad", (2011, 2022), (2023, 2024), (2025, 2025))
    # Identical windows WITH the opt-in construct fine.
    SplitStrategy("ok", (2011, 2022), (2023, 2024), (2025, 2025),
                  allow_forward_holdout=True)


def test_get_strategy_resolution():
    assert get_strategy("historical") is HISTORICAL
    assert get_strategy(HYBRID_ERA) is HYBRID_ERA
    with pytest.raises(KeyError, match="Unknown split strategy"):
        get_strategy("season_2027")


def test_rolling_window_arithmetic():
    # Docstring example: previous 5 seasons train, 1 val, 1 test.
    s = rolling_window_strategy(2024)
    assert s.train_years == (2018, 2022)
    assert s.val_years == (2023, 2023)
    assert s.test_years == (2024, 2024)
    assert s.default_n_folds == min(DEFAULT_N_FOLDS, 4)
    # Configurable window lengths.
    s = rolling_window_strategy(2023, train_seasons=3, val_seasons=2, test_seasons=2,
                                allow_forward_holdout=True)
    assert s.train_years == (2018, 2020)
    assert s.val_years == (2021, 2022)
    assert s.test_years == (2023, 2024)


def test_rolling_window_splits_frame():
    split = temporal_split(_full_range_frame(), strategy=rolling_window_strategy(2024))
    assert sorted(split.train["year"].unique()) == list(range(2018, 2023))
    assert sorted(split.val["year"].unique()) == [2023]
    assert sorted(split.test["year"].unique()) == [2024]


def test_rolling_window_guards():
    # Reaching the holdout still demands the explicit opt-in.
    with pytest.raises(ValueError, match="allow_forward_holdout"):
        rolling_window_strategy(2025)
    rolling_window_strategy(2025, allow_forward_holdout=True)   # opt-in OK
    with pytest.raises(ValueError, match="train_seasons"):
        rolling_window_strategy(2024, train_seasons=1)
    with pytest.raises(ValueError, match="val_seasons"):
        rolling_window_strategy(2024, val_seasons=0)


def test_season_folds_respect_strategy_window():
    split = temporal_split(_full_range_frame(), strategy="hybrid_era")
    folds = season_folds(split.train, strategy="hybrid_era")
    assert len(folds) == HYBRID_ERA.default_n_folds == 3
    assert [f.val_year for f in folds] == [2017, 2018, 2019]
    for fold in folds:
        assert int(fold.train["year"].max()) < fold.val_year
        assert not set(fold.train["raceId"]) & set(fold.val["raceId"])
    # The guard is strategy-relative: historical training years are
    # out-of-window for hybrid_era and must be rejected loudly.
    historical_train = temporal_split(_full_range_frame()).train
    with pytest.raises(ValueError, match="training split only"):
        season_folds(historical_train, strategy="hybrid_era")


def test_season_folds_default_n_folds_unchanged_for_historical():
    train = temporal_split(_full_range_frame()).train
    assert [f.val_year for f in season_folds(train)] == \
        [f.val_year for f in season_folds(train, n_folds=DEFAULT_N_FOLDS)]


# ---------------------------------------------------------------------------
# Regulation-era domain model
# ---------------------------------------------------------------------------

def test_preset_objectives_are_explicit():
    assert HISTORICAL.objective is EvaluationObjective.CROSS_ERA_GENERALIZATION
    assert HYBRID_ERA.objective is EvaluationObjective.WITHIN_ERA_VALIDATION
    assert GROUND_EFFECT.objective is EvaluationObjective.WITHIN_ERA_VALIDATION
    assert rolling_window_strategy(2024).objective is \
        EvaluationObjective.PRODUCTION_FORECASTING
    # Ad-hoc strategies default to CUSTOM.
    ad_hoc = SplitStrategy("adhoc", (2010, 2019), (2020, 2021), (2022, 2022))
    assert ad_hoc.objective is EvaluationObjective.CUSTOM
    # str-valued for MLflow logging.
    assert HISTORICAL.objective == "cross_era_generalization"


def test_within_era_presets_stay_inside_one_era():
    # The core property: every window of a within-era preset
    # sits under the same technical regulations.
    for preset, era in ((HYBRID_ERA, eras.HYBRID), (GROUND_EFFECT, eras.GROUND_EFFECT)):
        for lo, hi in (preset.train_years, preset.val_years, preset.test_years):
            assert era.contains(lo) and era.contains(hi), (
                f"{preset.name} window ({lo}, {hi}) leaves the {era.name} era")
        # The windows exactly tile the closed era — derived, not hand-typed.
        assert preset.train_years[0] == era.start_year
        assert preset.test_years[1] == era.end_year


def test_historical_preset_crosses_the_2022_era_boundary():
    # Its stated purpose IS cross-era generalization: training ends in the
    # hybrid era, evaluation happens entirely in the ground-effect era.
    assert eras.era_of(HISTORICAL.train_years[1]) is eras.HYBRID
    assert eras.era_of(HISTORICAL.val_years[0]) is eras.GROUND_EFFECT
    assert eras.era_of(HISTORICAL.test_years[0]) is eras.GROUND_EFFECT
    # And it is a frozen literal contract, era-table-independent.
    assert (HISTORICAL.train_years, HISTORICAL.val_years, HISTORICAL.test_years) \
        == ((2010, 2021), (2022, 2023), (2024, 2024))


def test_within_era_strategy_arithmetic():
    s = within_era_strategy("hybrid")
    assert s.train_years == (2014, 2019)
    assert s.val_years == (2020, 2020)
    assert s.test_years == (2021, 2021)
    assert s.name == "hybrid_within_era"
    assert s.objective is EvaluationObjective.WITHIN_ERA_VALIDATION
    # Configurable window lengths.
    s = within_era_strategy("hybrid", val_seasons=2, test_seasons=1)
    assert s.train_years == (2014, 2018)
    assert s.val_years == (2019, 2020)
    assert s.test_years == (2021, 2021)


def test_within_era_strategy_rejects_ongoing_era():
    with pytest.raises(ValueError, match="ongoing"):
        within_era_strategy("future_engine")


def test_within_era_strategy_rejects_too_short_era():
    # A 3-season closed era leaves only 1 training season for val=1/test=1.
    short_era = eras.RegulationEra("short", "Short era", 2022, 2024, "test-only")
    with pytest.raises(ValueError, match="too short"):
        within_era_strategy(short_era)
    with pytest.raises(ValueError, match="val_seasons"):
        within_era_strategy("hybrid", val_seasons=0)
    with pytest.raises(KeyError, match="Unknown regulation era"):
        within_era_strategy("turbo_1980s")


def test_ground_effect_preset_is_era_derived_with_holdout_opt_in():
    derived = within_era_strategy(
        eras.GROUND_EFFECT, allow_forward_holdout=True, name="ground_effect")
    assert (derived.train_years, derived.val_years, derived.test_years) == (
        GROUND_EFFECT.train_years, GROUND_EFFECT.val_years, GROUND_EFFECT.test_years)
    # Without the opt-in the same era is rejected — the SplitStrategy guard
    # composes with the within_era_strategy factory.
    with pytest.raises(ValueError, match="allow_forward_holdout"):
        within_era_strategy(eras.GROUND_EFFECT)


# ---------------------------------------------------------------------------
# Real-data smoke test — measured split sizes
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
