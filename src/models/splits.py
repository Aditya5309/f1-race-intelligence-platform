"""
src/models/splits.py

Temporal data splitting for model development.

WHY SPLITS ARE REGULATION-AWARE (concept drift)
-----------------------------------------------
F1 regulation rewrites reset the competitive order (see src/models/eras.py):
constructor form, dominance
concentration, and qualifying-to-race relationships learned under one
ruleset weaken or break under the next. A temporal split therefore does not
just prevent leakage — it CHOOSES which drift question the evaluation
answers. Each predefined strategy answers exactly one:

  EvaluationObjective.CROSS_ERA_GENERALIZATION
      "Can a model trained before a regulation change generalize to a new
       era?"  -> `historical` (the original default; the research setting
      behind every registered artifact).

  EvaluationObjective.WITHIN_ERA_VALIDATION
      "How well does the model perform when trained and evaluated under the
       same technical regulations?"  -> `hybrid_era`, `ground_effect`
      (built from the era table by `within_era_strategy`).

  EvaluationObjective.PRODUCTION_FORECASTING
      "What training window would a real F1 analytics team use to predict
       the next season?"  -> `rolling_window_strategy(...)` (recent seasons
      regardless of era labels — the shape a scheduled retraining job would
      use; windows that span a reset carry weakened constructor signal, a
      documented caveat, not an error).

Preset summary (`STRATEGIES`):

    historical    train 2010-2021 / val 2022-2023 / test 2024
                  (LITERAL years — never era-derived, so era-table edits
                  can never move the baseline contract)
    hybrid_era    train 2014-2019 / val 2020 / test 2021
                  (entirely inside the hybrid era — chosen so validation
                  and test never cross into a different ruleset)
    ground_effect train 2022-2023 / val 2024 / test 2025
                  (entirely inside the ground-effect era; touches the
                  forward holdout — see the guard note below)

MECHANICS
---------
1. `SplitStrategy` — a frozen, validated definition of train/validation/test
   year windows, tagged with its `EvaluationObjective`.
   `within_era_strategy(era)` carves a CLOSED regulation era's final seasons
   into val/test; adding a future era (e.g. 2026) to src/models/eras.py makes
   its within-era preset a one-liner with no changes to splitting logic.

2. `temporal_split(df, strategy=...)` — the outer split for the selected
   strategy. Year ranges are EXPLICIT per-strategy constants, never derived
   from the data's max year: features.parquet contains 2025-2026 rows (the
   forward holdout — data deliberately reserved to evaluate the system on
   genuinely unseen seasons later) that must never enter a split silently.

   Forward-holdout policy: a strategy may only include years >=
   FORWARD_HOLDOUT_MIN_YEAR if it declares `allow_forward_holdout=True` at
   construction — and actually running such a strategy on real data
   additionally requires resolving the provenance of the 2025-2026 rows
   already present in the source data. The default remains a hard
   rejection, so no existing caller can leak the holdout by accident.

3. `season_folds(train_df, ...)` — season-grouped expanding-window CV folds
   within the selected strategy's training window, e.g. for the historical
   default:
       fold 1: train 2010-2015 -> validate 2016
       ...
       fold 6: train 2010-2020 -> validate 2021
   Season granularity (not sklearn's row-level TimeSeriesSplit) because a
   row-level split can cut a race in half, corrupting per-race metrics, and
   rows within a race are not exchangeable.

`to_xy(df)` extracts the design matrix from a split: by default a curated
subset of columns returned by `active_feature_columns()` (currently
`FEATURE_COLUMNS` minus one experimental group found not to generalize —
see src/features/metadata.py), never the raw `FEATURE_COLUMNS` unless
explicitly requested via `feature_columns=`, the target, and raceId groups
for per-race evaluation. Identifier columns are never features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.features.metadata import active_feature_columns
from src.features.pipeline import TARGET_COLUMN
from src.models import eras

# 2025+ is the forward holdout, reserved for evaluating the system on
# genuinely unseen future seasons. Only strategies that explicitly declare
# `allow_forward_holdout=True` may reach it, and running such a strategy on
# real data is additionally gated on resolving 2025-2026 data provenance.
FORWARD_HOLDOUT_MIN_YEAR: int = 2025

# Six expanding-window folds for the historical strategy, validating
# 2016..2021. Other strategies carry their own default.
DEFAULT_N_FOLDS: int = 6


class EvaluationObjective(str, Enum):
    """The drift question a split strategy answers.

    str-valued so it logs cleanly to MLflow tags and reports.
    """

    CROSS_ERA_GENERALIZATION = "cross_era_generalization"
    WITHIN_ERA_VALIDATION = "within_era_validation"
    PRODUCTION_FORECASTING = "production_forecasting"
    CUSTOM = "custom"


@dataclass(frozen=True)
class SplitStrategy:
    """
    A validated train/validation/test year-window definition.

    All windows are inclusive `(lo, hi)` year ranges and must be strictly
    ordered without overlap: train < val < test. Gaps between windows are
    permitted (none of the shipped presets have any).

    Attributes
    ----------
    name : registry key and MLflow-loggable identifier.
    train_years, val_years, test_years : inclusive year windows.
    default_n_folds : `season_folds` fold count when the caller passes none;
        sized so every fold keeps at least one full training season.
    allow_forward_holdout : opt-in required for any window that reaches
        FORWARD_HOLDOUT_MIN_YEAR. Never set this on a new strategy without
        also resolving the 2025-2026 data-provenance question and the
        forward-holdout gate below.
    description : one-line purpose, for reports and logs.
    objective : which drift question this strategy answers; directly
        constructed ad-hoc strategies default to CUSTOM.
    """

    name: str
    train_years: tuple[int, int]
    val_years: tuple[int, int]
    test_years: tuple[int, int]
    default_n_folds: int = 1
    allow_forward_holdout: bool = False
    description: str = ""
    objective: EvaluationObjective = EvaluationObjective.CUSTOM

    def __post_init__(self) -> None:
        windows = {
            "train": self.train_years,
            "val": self.val_years,
            "test": self.test_years,
        }
        for label, (lo, hi) in windows.items():
            if lo > hi:
                raise ValueError(
                    f"Strategy '{self.name}': {label} window ({lo}, {hi}) is "
                    "inverted — expected (lo, hi) with lo <= hi."
                )
        if not (self.train_years[1] < self.val_years[0]
                and self.val_years[1] < self.test_years[0]):
            raise ValueError(
                f"Strategy '{self.name}': windows must be strictly ordered "
                f"train < val < test without overlap; got train={self.train_years}, "
                f"val={self.val_years}, test={self.test_years}."
            )
        if not self.allow_forward_holdout and self.test_years[1] >= FORWARD_HOLDOUT_MIN_YEAR:
            raise ValueError(
                f"Strategy '{self.name}' reaches forward-holdout year(s) "
                f">= {FORWARD_HOLDOUT_MIN_YEAR} without allow_forward_holdout=True "
                "— the forward holdout requires an explicit opt-in."
            )
        if self.default_n_folds < 1:
            raise ValueError(
                f"Strategy '{self.name}': default_n_folds must be >= 1."
            )


def within_era_strategy(
    era: eras.RegulationEra | str,
    *,
    val_seasons: int = 1,
    test_seasons: int = 1,
    default_n_folds: int | None = None,
    allow_forward_holdout: bool = False,
    name: str | None = None,
) -> SplitStrategy:
    """
    Build a WITHIN_ERA_VALIDATION strategy from a closed regulation era:
    the era's final `test_seasons` seasons form the test
    window, the `val_seasons` before them the validation window, and every
    earlier era season the training window. All windows share one ruleset,
    so the evaluation measures skill under stable regulations rather than
    cross-era generalization.

    This is the future-proofing hook: when a new era is closed in
    src/models/eras.py, its within-era preset is one call — no changes to
    splitting logic.

    Raises
    ------
    ValueError
        If the era is ongoing (no closed year range to carve — via
        RegulationEra.year_range), if the era is too short to leave at
        least two training seasons (season_folds needs one full season per
        fold side), or — via SplitStrategy validation — if the era reaches
        the forward holdout without `allow_forward_holdout=True`.
    KeyError
        If `era` names no known regulation era.
    """
    era = eras.get_era(era)
    if val_seasons < 1 or test_seasons < 1:
        raise ValueError(
            "within_era_strategy requires val_seasons >= 1 and test_seasons >= 1."
        )
    start, end = era.year_range   # loud for ongoing eras
    train_hi = end - val_seasons - test_seasons
    train_seasons = train_hi - start + 1
    if train_seasons < 2:
        raise ValueError(
            f"Era '{era.name}' ({start}-{end}) is too short for "
            f"val_seasons={val_seasons} + test_seasons={test_seasons}: only "
            f"{max(train_seasons, 0)} training season(s) would remain, and "
            "season_folds needs at least two."
        )
    return SplitStrategy(
        name=name or f"{era.name}_within_era",
        train_years=(start, train_hi),
        val_years=(train_hi + 1, train_hi + val_seasons),
        test_years=(end - test_seasons + 1, end),
        default_n_folds=(default_n_folds if default_n_folds is not None
                         else min(DEFAULT_N_FOLDS, train_seasons - 1)),
        allow_forward_holdout=allow_forward_holdout,
        description=(
            f"Within-era validation inside the {era.label} "
            f"({start}-{end}): same technical regulations for train, "
            "validation, and test."
        ),
        objective=EvaluationObjective.WITHIN_ERA_VALIDATION,
    )


# ---------------------------------------------------------------------------
# Preset strategies. Era boundaries live in src/models/eras.py; the ONLY
# literal years below are the original baseline split, which is a frozen
# contract and must never move with era-table edits.
# ---------------------------------------------------------------------------

HISTORICAL = SplitStrategy(
    name="historical",
    train_years=(2010, 2021),   # literal on purpose — never era-derived
    val_years=(2022, 2023),
    test_years=(2024, 2024),
    default_n_folds=DEFAULT_N_FOLDS,
    description=(
        "Baseline split (CROSS-ERA GENERALIZATION): trains on the "
        "pooled V8 + hybrid eras (2010-2021) and evaluates entirely across "
        "the 2022 ground-effect reset — the research setting behind every "
        "registered artifact. Measured consequence: the "
        "model's top-1 edge concentrates in dominance seasons."
    ),
    objective=EvaluationObjective.CROSS_ERA_GENERALIZATION,
)

# This preset stays entirely inside the hybrid regulations by design — an
# earlier definition tested on 2022 (the first ground-effect season), which
# made it a second cross-era experiment mislabeled as within-era. Caveat:
# its validation season (2020) ran a COVID-shortened calendar (same
# ruleset, fewer races).
HYBRID_ERA = within_era_strategy(
    eras.HYBRID,
    default_n_folds=3,
    name="hybrid_era",
)

GROUND_EFFECT = within_era_strategy(
    eras.GROUND_EFFECT,
    allow_forward_holdout=True,   # test window is 2025 — explicit opt-in;
    name="ground_effect",         # real-data use gated on provenance
)

STRATEGIES: dict[str, SplitStrategy] = {
    s.name: s for s in (HISTORICAL, HYBRID_ERA, GROUND_EFFECT)
}

DEFAULT_STRATEGY: SplitStrategy = HISTORICAL

# Backward-compatibility aliases — the historical preset IS the fixed outer
# split every existing caller relies on. Do not remove.
TRAIN_YEARS: tuple[int, int] = HISTORICAL.train_years
VAL_YEARS: tuple[int, int] = HISTORICAL.val_years
TEST_YEARS: tuple[int, int] = HISTORICAL.test_years


def rolling_window_strategy(
    test_start_year: int,
    *,
    train_seasons: int = 5,
    val_seasons: int = 1,
    test_seasons: int = 1,
    allow_forward_holdout: bool = False,
    name: str | None = None,
) -> SplitStrategy:
    """
    Build a PRODUCTION_FORECASTING strategy anchored on the test season —
    the shape a scheduled automated-retraining job would use: train on
    the most recent completed seasons to predict the next one, the window a
    real analytics team would use.

    Windows are contiguous and counted backwards from `test_start_year`:
    validation covers the `val_seasons` seasons immediately before the test
    window, training the `train_seasons` seasons before that. Example:
    `rolling_window_strategy(2024)` -> train 2018-2022, val 2023, test 2024.

    Era caveat (deliberate): rolling windows ignore era boundaries. A window
    spanning a regulation reset carries weakened constructor-form signal for
    post-reset seasons — expected domain behavior to report, not an error to
    prevent. Use `eras.era_of()` if a caller wants to flag boundary-spanning
    windows.

    Raises
    ------
    ValueError
        If any window length is < 1, if `train_seasons` < 2 (season_folds
        needs at least one train season per fold), or — via SplitStrategy
        validation — if the test window reaches the forward holdout without
        `allow_forward_holdout=True`.
    """
    if train_seasons < 2:
        raise ValueError(
            "rolling_window_strategy requires train_seasons >= 2 so that "
            "season_folds keeps at least one full training season per fold."
        )
    if val_seasons < 1 or test_seasons < 1:
        raise ValueError(
            "rolling_window_strategy requires val_seasons >= 1 and test_seasons >= 1."
        )

    val_start = test_start_year - val_seasons
    train_start = val_start - train_seasons
    return SplitStrategy(
        name=name or f"rolling_{train_seasons}_{val_seasons}_{test_seasons}_test{test_start_year}",
        train_years=(train_start, val_start - 1),
        val_years=(val_start, test_start_year - 1),
        test_years=(test_start_year, test_start_year + test_seasons - 1),
        default_n_folds=min(DEFAULT_N_FOLDS, train_seasons - 1),
        allow_forward_holdout=allow_forward_holdout,
        description=(
            f"Production forecasting: {train_seasons} train / {val_seasons} "
            f"val / {test_seasons} test most-recent seasons, testing from "
            f"{test_start_year}."
        ),
        objective=EvaluationObjective.PRODUCTION_FORECASTING,
    )


def get_strategy(strategy: SplitStrategy | str) -> SplitStrategy:
    """Resolve a strategy object or preset name; loud on unknown names."""
    if isinstance(strategy, SplitStrategy):
        return strategy
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise KeyError(
            f"Unknown split strategy '{strategy}'. Available presets: "
            f"{sorted(STRATEGIES)}; or pass a SplitStrategy (e.g. from "
            "within_era_strategy() or rolling_window_strategy())."
        ) from None


@dataclass(frozen=True)
class TemporalSplit:
    """An outer split. Frames are copies; mutate freely."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: SplitStrategy = DEFAULT_STRATEGY


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


def temporal_split(
    df: pd.DataFrame, strategy: SplitStrategy | str = DEFAULT_STRATEGY,
) -> TemporalSplit:
    """
    Split a feature frame into a strategy's train/val/test windows.

    Defaults to the `historical` preset — the original, fixed baseline
    split — so existing callers are unchanged.

    Raises
    ------
    KeyError
        If `df` has no 'year' column, or `strategy` names no known preset.
    ValueError
        If any split comes back empty (wrong input frame, or — for
        forward-era strategies like `ground_effect` — a season whose data
        is not available yet), or if any split contains forward-holdout
        years without the strategy's opt-in, or overlapping raceIds (both
        indicate a bug in this module or corrupted input, and must be loud).
    """
    strategy = get_strategy(strategy)
    if "year" not in df.columns:
        raise KeyError("temporal_split requires a 'year' column.")

    split = TemporalSplit(
        train=_select_years(df, strategy.train_years),
        val=_select_years(df, strategy.val_years),
        test=_select_years(df, strategy.test_years),
        strategy=strategy,
    )

    windows = {
        "train": (split.train, strategy.train_years),
        "val": (split.val, strategy.val_years),
        "test": (split.test, strategy.test_years),
    }
    for name, (part, (lo, hi)) in windows.items():
        if part.empty:
            raise ValueError(
                f"Split strategy '{strategy.name}' produced an empty '{name}' "
                f"set for years {lo}-{hi} — the input frame has no rows in "
                "that window (wrong input frame, or the season's data is not "
                "available yet)."
            )
        # Defense in depth: explicit ranges make this structurally
        # impossible, but the forward holdout must never leak silently.
        if (not strategy.allow_forward_holdout
                and int(part["year"].max()) >= FORWARD_HOLDOUT_MIN_YEAR):
            raise ValueError(
                f"'{name}' split contains forward-holdout year(s) "
                f">= {FORWARD_HOLDOUT_MIN_YEAR} without an explicit opt-in."
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
    train_df: pd.DataFrame,
    n_folds: int | None = None,
    strategy: SplitStrategy | str = DEFAULT_STRATEGY,
) -> list[SeasonFold]:
    """
    Expanding-window CV folds over the seasons of a strategy's training split.

    The last `n_folds` seasons each serve once as a validation season; each
    fold trains on every season strictly before its validation season. All
    rows of a race land on one side of every boundary (year-based selection —
    a race belongs to exactly one season). `n_folds=None` uses the strategy's
    `default_n_folds` (6 for the historical preset — unchanged behavior).

    Pass the same `strategy` used for `temporal_split`: the training-window
    guard below is what keeps out-of-split data (and therefore the OOF
    calibrator, which is fit only on these folds) from ever seeing
    validation/test seasons.

    Raises
    ------
    ValueError
        If `train_df` contains years outside the strategy's training window
        (this function must only ever see the training split — passing
        val/test data here would leak it into model selection), or if
        `n_folds` leaves no training seasons for the first fold.
    """
    strategy = get_strategy(strategy)
    if "year" not in train_df.columns:
        raise KeyError("season_folds requires a 'year' column.")

    lo, hi = strategy.train_years
    if not train_df["year"].between(lo, hi).all():
        raise ValueError(
            f"season_folds expects the '{strategy.name}' training split only ({lo}-{hi}); "
            f"got years {sorted(train_df.loc[~train_df['year'].between(lo, hi), 'year'].unique())}."
        )

    if n_folds is None:
        n_folds = strategy.default_n_folds

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


def to_xy(
    df: pd.DataFrame, feature_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract (X, y, race_ids) from a split frame.

    `feature_columns` defaults — when NOT given — to
    `active_feature_columns()` (a curated subset, currently `FEATURE_COLUMNS`
    minus one experimental group found not to generalize — see
    src/features/metadata.py), never to the raw, full `FEATURE_COLUMNS`.
    This is the same safe-by-default inversion `registry.get_model()`
    applies, for the same reason: an exclusion enforced only by manual
    discipline is invisible to automated retraining, which has no way to
    know about it — so the default must be the safe, curated set, and
    reaching the full, unexcluded set requires a caller to explicitly pass
    `feature_columns=FEATURE_COLUMNS` (or another explicit tuple) on
    purpose. race_ids carries the per-race grouping every evaluation metric
    needs.
    """
    columns = feature_columns if feature_columns is not None else active_feature_columns()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Frame is missing feature column(s): {missing}")
    return df[list(columns)], df[TARGET_COLUMN], df["raceId"]
