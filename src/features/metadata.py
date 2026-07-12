"""
src/features/metadata.py

Feature metadata — the single source of truth for feature grouping, the
Decision-013 classification (Stable / Era-sensitive / Experimental), AND
(Decision 041) which feature groups are excluded from training by default.

Consumers: model training (importance reporting grouped by class; `to_xy()`/
`get_model()`'s default feature-column resolution), evaluation, SHAP analysis,
ETL/drift monitoring (Phase 8: Stable-feature drift suggests a data problem;
Era-sensitive drift at regulation boundaries is expected domain behavior),
and future dashboard components. Import from here — never re-type feature
lists, classifications, or exclusions elsewhere.

Rationale for each class lives in Decision 013 and the feature-set review in
reports/model_development_design.md §14:
- Stable: era-robust relative/normalized measures, rank-based values, or
  structural facts. Expected to survive regulation resets.
- Era-sensitive: predictive power depends on dominance concentration,
  regulation continuity, or points-system stability. Expected to weaken at
  era boundaries (2026) and under cost-cap convergence.
- Experimental: weak, noisy, or proxy signal; explicit keep-or-drop decision
  after v1 SHAP/error analysis (e.g., raw q times are ~99% circuit-identity
  variance).

Reclassifying a feature requires a new decision entry (Decision 013
consequence) — do not edit these tuples casually.

TRAINING-TIME EXCLUSIONS (Decision 041, resolving Decisions 036/040): see
`EXCLUDED_FROM_TRAINING`/`active_feature_columns()` below. A real automated
retrain (Decision 036/PR #1) regressed because it trained on the full,
current `FEATURE_COLUMNS` — including `wet_form`, which Decision 040's
ablation study isolated as the actual cause (weather/qualifying_raw_times
are inert; teammate_form/grid_penalty_applied are net-positive and must
stay included — NOT a blanket "exclude all experimental features" rule).
`to_xy()`/`get_model()` now default to `active_feature_columns()` (never
the raw `FEATURE_COLUMNS`) precisely so that regression can't silently
recur: getting the full, unexcluded set requires an explicit
`feature_columns=FEATURE_COLUMNS` override, never the default.
"""

from __future__ import annotations

from src.features.circuit_history import CIRCUIT_HISTORY_FEATURES
from src.features.constructor_form import CONSTRUCTOR_FORM_FEATURES
from src.features.driver_form import DRIVER_FORM_FEATURES
from src.features.pipeline import FEATURE_COLUMNS
from src.features.qualifying import QUALIFYING_FEATURES
from src.features.standings import STANDINGS_FEATURES
from src.features.teammate_form import TEAMMATE_FORM_FEATURES
from src.features.weather import WEATHER_FEATURES
from src.features.wet_form import WET_FORM_FEATURES

# ---------------------------------------------------------------------------
# Grouping by source module (execution order of the feature pipeline).
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "qualifying": QUALIFYING_FEATURES,
    "driver_form": DRIVER_FORM_FEATURES,
    "constructor_form": CONSTRUCTOR_FORM_FEATURES,
    "teammate_form": TEAMMATE_FORM_FEATURES,
    "circuit_history": CIRCUIT_HISTORY_FEATURES,
    "standings": STANDINGS_FEATURES,
    "weather": WEATHER_FEATURES,
    "wet_form": WET_FORM_FEATURES,
}

# ---------------------------------------------------------------------------
# Decision-013 classification.
# ---------------------------------------------------------------------------

STABLE_FEATURES: tuple[str, ...] = (
    "qualifying_position",
    "qualifying_gap_to_pole_pct",
    "reached_q2",
    "reached_q3",
    "pit_lane_start",
    "grid_adjusted",
    "grid_position_norm",
    "grid_penalty_applied",
    "driver_experience_races",
    "driver_avg_finish_last_5",
    "driver_dnf_rate_last_5",
    "driver_standing_position_prev",
    "constructor_standing_position_prev",
    "qualifying_gap_to_teammate_current",
    "qualifying_gap_to_teammate",
    "race_pace_delta_to_teammate",
)

ERA_SENSITIVE_FEATURES: tuple[str, ...] = (
    "driver_wins_last_3",
    "driver_wins_last_5",
    "driver_wins_last_10",
    "driver_podiums_last_5",
    "driver_points_last_5",
    "constructor_wins_last_3",
    "constructor_wins_last_5",
    "constructor_podiums_last_5",
    "constructor_dnf_rate_last_5",
    "driver_standing_points_prev",
    "driver_standing_wins_prev",
    "constructor_standing_points_prev",
)

EXPERIMENTAL_FEATURES: tuple[str, ...] = (
    "q1_sec",
    "q2_sec",
    "q3_sec",
    "driver_circuit_starts",
    "driver_circuit_wins",
    "driver_circuit_avg_finish",
    "constructor_circuit_wins",
    # Phase 4 Tranche B: brand-new, never-before-assessed weather signal —
    # explicit keep-or-drop decision pending, per this class's own criteria,
    # after the Tranche B retrain's feature-importance/ablation check.
    "race_precip_mm",
    "race_temp_c",
    "quali_precip_mm",
    "conditions_changed",
    # Same rationale — driver_wet_dry_delta/constructor_wet_dry_delta are
    # brand-new, never-before-assessed signals derived from the weather
    # features above; explicit keep-or-drop decision pending too.
    "driver_wet_dry_delta",
    "constructor_wet_dry_delta",
)

FEATURE_CLASSES: tuple[str, ...] = ("stable", "era_sensitive", "experimental")

FEATURE_CLASSIFICATION: dict[str, str] = {
    **{f: "stable" for f in STABLE_FEATURES},
    **{f: "era_sensitive" for f in ERA_SENSITIVE_FEATURES},
    **{f: "experimental" for f in EXPERIMENTAL_FEATURES},
}


def features_in_class(feature_class: str) -> tuple[str, ...]:
    """Return the features of one Decision-013 class ('stable', 'era_sensitive', 'experimental')."""
    if feature_class not in FEATURE_CLASSES:
        raise ValueError(
            f"Unknown feature class '{feature_class}' — expected one of {FEATURE_CLASSES}."
        )
    return tuple(f for f, c in FEATURE_CLASSIFICATION.items() if c == feature_class)


# ---------------------------------------------------------------------------
# Training-time exclusions (Decision 041 — the minimal path-(b) mechanism
# resolving Decision 036/040's regression). NOT a Feature Profile System:
# a single denylist of FEATURE_GROUPS names, applied as the training DEFAULT
# everywhere (see active_feature_columns()) so an exclusion can never be
# silently bypassed the way Decision 036's manual, uncommitted exclusion
# was. Referenced by GROUP NAME, never by Decision-013 classification —
# Decision 040's own ablation showed classification-level exclusion is too
# coarse (it would incorrectly exclude teammate_form/grid_penalty_applied,
# which are `stable`-classified and must stay included).
#
# Changing this list (excluding or re-including a group) requires a new
# decision entry in context/decisions.md — same convention this file
# already applies to the STABLE/ERA_SENSITIVE/EXPERIMENTAL tuples above.
# Do not edit casually.
EXCLUDED_FROM_TRAINING: tuple[str, ...] = ("wet_form",)


def active_feature_columns(
    excluded_groups: tuple[str, ...] = EXCLUDED_FROM_TRAINING,
) -> tuple[str, ...]:
    """FEATURE_COLUMNS minus every feature in `excluded_groups` (looked up
    by FEATURE_GROUPS name), preserving FEATURE_COLUMNS' original order.

    THIS is the training default (Decision 041): `to_xy()`/`get_model()`
    resolve to this when no explicit `feature_columns` override is given,
    so excluding a group cannot be silently bypassed the way Decision 036's
    manual, uncommitted exclusion was. Pass the raw `FEATURE_COLUMNS`
    explicitly (never this function) to deliberately opt into the full,
    unexcluded set for research/ablation purposes — see Decision 040's own
    investigation for the precedent this generalizes.
    """
    unknown = set(excluded_groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown feature group(s) in excluded_groups: {sorted(unknown)}. "
            f"Known groups: {sorted(FEATURE_GROUPS)}."
        )
    excluded_features = {f for g in excluded_groups for f in FEATURE_GROUPS[g]}
    return tuple(f for f in FEATURE_COLUMNS if f not in excluded_features)


_unknown_excluded_default = set(EXCLUDED_FROM_TRAINING) - set(FEATURE_GROUPS)
assert not _unknown_excluded_default, (
    f"EXCLUDED_FROM_TRAINING references unknown feature group(s): "
    f"{sorted(_unknown_excluded_default)}. Known groups: {sorted(FEATURE_GROUPS)}."
)


# ---------------------------------------------------------------------------
# Import-time integrity — this module cannot be imported in a state where the
# classification or grouping disagrees with the pipeline's FEATURE_COLUMNS.
# ---------------------------------------------------------------------------

_classified = set(FEATURE_CLASSIFICATION)
assert _classified == set(FEATURE_COLUMNS), (
    "FEATURE_CLASSIFICATION must cover FEATURE_COLUMNS exactly. "
    f"Missing: {sorted(set(FEATURE_COLUMNS) - _classified)}; "
    f"extra: {sorted(_classified - set(FEATURE_COLUMNS))}."
)
assert len(STABLE_FEATURES) + len(ERA_SENSITIVE_FEATURES) + len(EXPERIMENTAL_FEATURES) == len(FEATURE_COLUMNS), \
    "Feature classes must partition FEATURE_COLUMNS (a feature appears in two classes)."

_grouped = {f for group in FEATURE_GROUPS.values() for f in group}
assert _grouped == set(FEATURE_COLUMNS), (
    "FEATURE_GROUPS must cover FEATURE_COLUMNS exactly. "
    f"Missing: {sorted(set(FEATURE_COLUMNS) - _grouped)}; "
    f"extra: {sorted(_grouped - set(FEATURE_COLUMNS))}."
)
