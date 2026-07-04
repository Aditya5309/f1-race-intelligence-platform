"""
src/models/splits.py

Temporal data splitting for model development (Decisions 008 and 012;
reports/model_development_design.md Sections 2 and 4).

Two responsibilities, both purely row selection — no fitting, no metrics:

1. `temporal_split(df)` — the fixed outer split (Decision 008):
       train 2010-2021 / validation 2022-2023 / test 2024.
   Year ranges are EXPLICIT constants, never derived from the data's max
   year: features.parquet contains 2025-2026 rows (the forward holdout,
   Decision 012 Section 13.1) that must never enter any split. Design doc
   Section 14.4 mandates this guard.

2. `season_folds(train_df)` — season-grouped expanding-window CV folds
   within the training split (design doc Section 4):
       fold 1: train 2010-2015 -> validate 2016
       ...
       fold 6: train 2010-2020 -> validate 2021
   Season granularity (not sklearn's row-level TimeSeriesSplit) because a
   row-level split can cut a race in half, corrupting per-race metrics, and
   rows within a race are not exchangeable.

`to_xy(df)` extracts the design matrix from a split: exactly FEATURE_COLUMNS
(imported from the feature pipeline — the single source of truth), the
target, and raceId groups for per-race evaluation. Identifier columns are
never features (design doc Section 11.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN

# Decision 008 — fixed outer split, inclusive year ranges.
TRAIN_YEARS: tuple[int, int] = (2010, 2021)
VAL_YEARS: tuple[int, int] = (2022, 2023)
TEST_YEARS: tuple[int, int] = (2024, 2024)

# Decision 012 Section 13.1 — 2025+ is the forward holdout, reserved for the
# Phase 8 retraining/monitoring rehearsal. No Phase 4 code may consume it.
FORWARD_HOLDOUT_MIN_YEAR: int = 2025

# Design doc Section 4 — six expanding-window folds validating 2016..2021.
DEFAULT_N_FOLDS: int = 6


@dataclass(frozen=True)
class TemporalSplit:
    """The Decision-008 outer split. Frames are copies; mutate freely."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class SeasonFold:
    """One expanding-window CV fold: train on seasons strictly before val_year."""
    fold: int
    train_years: tuple[int, int]
    val_year: int
    train: pd.DataFrame
    val: pd.DataFrame


def _select_years(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    lo, hi = years
    return df.loc[df["year"].between(lo, hi)].copy()


def temporal_split(df: pd.DataFrame) -> TemporalSplit:
    """
    Split a feature frame into the fixed Decision-008 train/val/test windows.

    Raises
    ------
    KeyError
        If `df` has no 'year' column.
    ValueError
        If any split comes back empty (wrong input frame — e.g. an already
        year-filtered subset), or if any split contains forward-holdout
        years or overlapping raceIds (both indicate a bug in this module or
        corrupted input, and must be loud).
    """
    if "year" not in df.columns:
        raise KeyError("temporal_split requires a 'year' column.")

    split = TemporalSplit(
        train=_select_years(df, TRAIN_YEARS),
        val=_select_years(df, VAL_YEARS),
        test=_select_years(df, TEST_YEARS),
    )

    parts = {"train": split.train, "val": split.val, "test": split.test}
    for name, part in parts.items():
        if part.empty:
            raise ValueError(
                f"Temporal split produced an empty '{name}' set — the input "
                "frame does not cover the Decision-008 year windows."
            )
        # Defense in depth: explicit ranges make this structurally
        # impossible, but the forward holdout must never leak silently
        # (design doc Section 14.4).
        if int(part["year"].max()) >= FORWARD_HOLDOUT_MIN_YEAR:
            raise ValueError(
                f"'{name}' split contains forward-holdout year(s) "
                f">= {FORWARD_HOLDOUT_MIN_YEAR} — Decision 012 Section 13.1 violation."
            )

    train_races = set(split.train["raceId"])
    val_races = set(split.val["raceId"])
    test_races = set(split.test["raceId"])
    if (train_races & val_races) or (train_races & test_races) or (val_races & test_races):
        raise ValueError(
            "Splits share raceIds — a race appears in more than one split."
        )

    return split


def season_folds(
    train_df: pd.DataFrame, n_folds: int = DEFAULT_N_FOLDS,
) -> list[SeasonFold]:
    """
    Expanding-window CV folds over the seasons of the training split.

    The last `n_folds` seasons each serve once as a validation season; each
    fold trains on every season strictly before its validation season. All
    rows of a race land on one side of every boundary (year-based selection —
    a race belongs to exactly one season).

    Raises
    ------
    ValueError
        If `train_df` contains years outside TRAIN_YEARS (this function must
        only ever see the training split — passing val/test data here would
        leak it into model selection), or if `n_folds` leaves no training
        seasons for the first fold.
    """
    if "year" not in train_df.columns:
        raise KeyError("season_folds requires a 'year' column.")

    lo, hi = TRAIN_YEARS
    if not train_df["year"].between(lo, hi).all():
        raise ValueError(
            f"season_folds expects the training split only ({lo}-{hi}); "
            f"got years {sorted(train_df.loc[~train_df['year'].between(lo, hi), 'year'].unique())}."
        )

    seasons = sorted(int(y) for y in train_df["year"].unique())
    if n_folds < 1 or n_folds >= len(seasons):
        raise ValueError(
            f"n_folds={n_folds} is invalid for {len(seasons)} training seasons — "
            "the first fold needs at least one full season to train on."
        )

    folds: list[SeasonFold] = []
    for i, val_year in enumerate(seasons[-n_folds:], start=1):
        prev_season = seasons[seasons.index(val_year) - 1]
        folds.append(
            SeasonFold(
                fold=i,
                train_years=(seasons[0], prev_season),
                val_year=val_year,
                train=train_df.loc[train_df["year"] < val_year].copy(),
                val=train_df.loc[train_df["year"] == val_year].copy(),
            )
        )
    return folds


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract (X, y, race_ids) from a split frame.

    X contains exactly FEATURE_COLUMNS in canonical order — identifiers and
    post-race columns are structurally excluded because FEATURE_COLUMNS is
    the feature pipeline's guarded constant. race_ids carries the per-race
    grouping every evaluation metric needs.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Frame is missing feature column(s): {missing}")
    return df[list(FEATURE_COLUMNS)], df[TARGET_COLUMN], df["raceId"]
