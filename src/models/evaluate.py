"""
src/models/evaluate.py

Model-agnostic evaluation for Phase 4 (Decision 012;
reports/model_development_design.md Section 6).

Everything here is a pure function of `(y_true, y_prob, race_ids[, years])` —
no model objects, no MLflow, no I/O — so any classifier in MODEL_ZOO (or a
future one) evaluates through identical code, and tests can hand-compute
every number.

Why per-race metrics are the only honest ones (design Section 6): at a 4.7%
positive rate, row-level accuracy is meaningless (predicting "nobody wins" is
95% accurate). The prediction task is "pick the winner within each race", so
the headline metrics group by raceId.

Tie policy (deliberate, pessimistic): the winner's predicted rank uses
worst-case competition ranking — rank = count of drivers with probability
>= the winner's probability. A model that outputs equal probabilities for
several drivers gets NO credit for the winner being "among" them: with all
20 drivers tied, the winner's rank is 20, not 1. This prevents degenerate
constant-probability models from scoring free top-1 hits and makes the
pole-sitter baseline's 0/1 probabilities behave sensibly (a non-pole winner
ranks last, tied with the other non-pole drivers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss

# Bins for calibration curve / ECE — uniform width over [0, 1].
DEFAULT_CALIBRATION_BINS = 10


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate(y_true, y_prob, race_ids) -> pd.DataFrame:
    """Align inputs into one frame; enforce the evaluation contract."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    race_ids = np.asarray(race_ids)

    if not (len(y_true) == len(y_prob) == len(race_ids)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}, "
            f"race_ids={len(race_ids)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate zero rows.")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN.")
    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob contains values outside [0, 1].")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must be binary (0/1).")

    df = pd.DataFrame({"race_id": race_ids, "y_true": y_true, "y_prob": y_prob})

    winners_per_race = df.groupby("race_id")["y_true"].sum()
    bad = winners_per_race[winners_per_race != 1]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} race(s) do not have exactly one winner "
            f"(sample raceIds: {bad.index[:5].tolist()}) — evaluation requires "
            "the (modeling-window) one-winner-per-race invariant."
        )
    return df


# ---------------------------------------------------------------------------
# Per-race table — the core primitive every ranking metric aggregates from
# ---------------------------------------------------------------------------

def per_race_table(y_true, y_prob, race_ids, position_order=None) -> pd.DataFrame:
    """
    One row per race:
      race_id, n_drivers, winner_prob (probability the model gave the actual
      winner), winner_rank (pessimistic tie policy — see module docstring),
      top1_hit, top3_hit, reciprocal_rank, max_prob (the model's most
      confident driver in that race).

    If `position_order` is supplied (the full field's actual Ergast finishing
    order — NOT a FEATURE_COLUMNS member, joined in at evaluation time from
    master_dataset.parquet), also adds `spearman_corr`: the per-race Spearman
    correlation between the model's probability ranking and the actual
    finishing order (+1.0 = the model's ranking exactly matches the finish
    order; see spearman_rank_correlation for the sign convention). NaN for a
    degenerate race (fewer than 2 distinct predicted probabilities or
    finishing positions) rather than raising — same discipline as
    calibration_table omitting empty bins.
    """
    df = _validate(y_true, y_prob, race_ids)
    if position_order is not None:
        position_order = np.asarray(position_order, dtype=float)
        if len(position_order) != len(df):
            raise ValueError(
                f"Length mismatch: position_order={len(position_order)}, rows={len(df)}."
            )
        df = df.assign(position_order=position_order)

    records = []
    for race_id, group in df.groupby("race_id", sort=True):
        winner_prob = float(group.loc[group["y_true"] == 1, "y_prob"].iloc[0])
        rank = int((group["y_prob"] >= winner_prob).sum())   # worst-case ties
        record = {
            "race_id": race_id,
            "n_drivers": len(group),
            "winner_prob": winner_prob,
            "winner_rank": rank,
            "top1_hit": rank == 1,
            "top3_hit": rank <= 3,
            "reciprocal_rank": 1.0 / rank,
            "max_prob": float(group["y_prob"].max()),
        }
        if position_order is not None:
            if group["position_order"].nunique() >= 2 and group["y_prob"].nunique() >= 2:
                # Negate y_prob so both series share a "1 = best" direction —
                # a perfect model then correlates at +1.0, not -1.0.
                corr, _ = spearmanr(-group["y_prob"].to_numpy(), group["position_order"].to_numpy())
                record["spearman_corr"] = float(corr) if not np.isnan(corr) else np.nan
            else:
                record["spearman_corr"] = np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Headline ranking metrics (design Section 6)
# ---------------------------------------------------------------------------

def top1_accuracy(y_true, y_prob, race_ids) -> float:
    """Fraction of races whose highest-probability driver is the winner."""
    return float(per_race_table(y_true, y_prob, race_ids)["top1_hit"].mean())


def top3_recall(y_true, y_prob, race_ids) -> float:
    """Fraction of races whose winner is among the 3 highest probabilities."""
    return float(per_race_table(y_true, y_prob, race_ids)["top3_hit"].mean())


def winner_mrr(y_true, y_prob, race_ids) -> float:
    """Mean reciprocal rank of the actual winner in the predicted ordering."""
    return float(per_race_table(y_true, y_prob, race_ids)["reciprocal_rank"].mean())


def spearman_rank_correlation(y_true, y_prob, race_ids, position_order) -> float:
    """
    Mean per-race Spearman correlation between the model's full-field
    probability ranking and the actual Ergast finishing order (`positionOrder`
    — a post-race outcome column, never a FEATURE_COLUMNS member; join it in
    from master_dataset.parquet at evaluation time only).

    Unlike top1_accuracy/top3_recall/winner_mrr, this scores agreement across
    the WHOLE field, not just whether the winner was identified. +1.0 = the
    model's ranking exactly matches the finish order; 0.0 = no rank
    relationship; -1.0 = exactly inverted. Races with a degenerate field
    (see per_race_table) are excluded from the mean.
    """
    table = per_race_table(y_true, y_prob, race_ids, position_order=position_order)
    return float(table["spearman_corr"].mean())


def winner_rank_distribution(y_true, y_prob, race_ids) -> pd.Series:
    """Count of races by the winner's predicted rank (index=rank, sorted)."""
    table = per_race_table(y_true, y_prob, race_ids)
    return table["winner_rank"].value_counts().sort_index()


def average_winner_probability(y_true, y_prob, race_ids) -> float:
    """Mean probability the model assigned to the actual winners.

    A sharpness/quality hybrid: random guessing gives ~1/field (~0.05);
    a confident, correct model pushes this toward 1.
    """
    return float(per_race_table(y_true, y_prob, race_ids)["winner_prob"].mean())


# ---------------------------------------------------------------------------
# Probability-quality metrics (row-level, raw probabilities)
# ---------------------------------------------------------------------------

def log_loss_score(y_true, y_prob) -> float:
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def brier_score(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))


def calibration_table(
    y_true, y_prob, n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """
    Reliability-diagram data: uniform-width probability bins with
      bin_lower, bin_upper, mean_predicted, fraction_positive, count.
    Empty bins are omitted. Feeds both the saved calibration plot artifact
    and expected_calibration_error().
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins; probability exactly 0 goes in the first bin
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    records = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        records.append({
            "bin_lower": edges[b],
            "bin_upper": edges[b + 1],
            "mean_predicted": float(y_prob[mask].mean()),
            "fraction_positive": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame.from_records(records)


def expected_calibration_error(
    y_true, y_prob, n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> float:
    """ECE: count-weighted mean |fraction_positive − mean_predicted| over bins."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    weights = table["count"] / table["count"].sum()
    gaps = (table["fraction_positive"] - table["mean_predicted"]).abs()
    return float((weights * gaps).sum())


# ---------------------------------------------------------------------------
# Aggregates for MLflow logging and per-season monitoring
# ---------------------------------------------------------------------------

def evaluate_all(y_true, y_prob, race_ids, position_order=None) -> dict[str, float]:
    """
    The scalar metric dict train.py logs to MLflow. All plain floats.

    `position_order` is optional (default None, backward-compatible with
    every existing call site) — when supplied, adds "spearman_corr" (see
    spearman_rank_correlation). It is a same-race-full-field evaluation-time
    join from master_dataset.parquet, never a feature.
    """
    table = per_race_table(y_true, y_prob, race_ids, position_order=position_order)
    result = {
        "top1_accuracy": float(table["top1_hit"].mean()),
        "top3_recall": float(table["top3_hit"].mean()),
        "winner_mrr": float(table["reciprocal_rank"].mean()),
        "avg_winner_probability": float(table["winner_prob"].mean()),
        "median_winner_rank": float(table["winner_rank"].median()),
        "log_loss": log_loss_score(np.asarray(y_true), np.asarray(y_prob, dtype=float)),
        "brier_score": brier_score(np.asarray(y_true), np.asarray(y_prob, dtype=float)),
        "ece": expected_calibration_error(y_true, y_prob),
        "n_races": float(len(table)),
        "n_rows": float(len(np.asarray(y_true))),
    }
    if position_order is not None:
        result["spearman_corr"] = float(table["spearman_corr"].mean())
    return result


def evaluate_by_season(y_true, y_prob, race_ids, years, position_order=None) -> pd.DataFrame:
    """
    evaluate_all() per season — the regulation-era monitoring view mandated
    by design Section 9 ("report metrics per season, not only pooled"): era
    effects and form-feature nonstationarity are invisible in pooled numbers.

    Returns a DataFrame indexed by year, one column per evaluate_all metric.
    `position_order`, if supplied, is sliced by season the same way as the
    other arrays and threaded into each season's evaluate_all call.
    """
    years = np.asarray(years)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    race_ids = np.asarray(race_ids)
    if len(years) != len(y_true):
        raise ValueError(f"Length mismatch: years={len(years)}, y_true={len(y_true)}.")
    if position_order is not None:
        position_order = np.asarray(position_order, dtype=float)
        if len(position_order) != len(y_true):
            raise ValueError(
                f"Length mismatch: position_order={len(position_order)}, y_true={len(y_true)}."
            )

    # A race belongs to exactly one season; enforce rather than assume.
    season_check = pd.DataFrame({"race_id": race_ids, "year": years})
    if (season_check.groupby("race_id")["year"].nunique() > 1).any():
        raise ValueError("A raceId spans multiple years — corrupted input.")

    rows = {}
    for year in sorted(np.unique(years)):
        mask = years == year
        po_slice = position_order[mask] if position_order is not None else None
        rows[int(year)] = evaluate_all(
            y_true[mask], y_prob[mask], race_ids[mask], position_order=po_slice
        )
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "year"
    return out
