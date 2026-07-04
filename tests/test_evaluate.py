"""
Tests for src/models/evaluate.py (Phase 4 module 3 — Decision 012).

Design-doc Section 12 requirements: per-race top-1/top-3/MRR on
hand-computable synthetic races (including tie probabilities, a race
violating the one-winner invariant raising, single-driver race edge);
log-loss/Brier against the sklearn reference; calibration binning.

Plus the module-2-review additions: per-season summaries, winner rank
distribution, average winner probability, ECE, and model-agnosticism
(pole baseline from MODEL_ZOO evaluated through the same functions).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from src.models.evaluate import (
    average_winner_probability,
    brier_score,
    calibration_table,
    evaluate_all,
    evaluate_by_season,
    expected_calibration_error,
    log_loss_score,
    per_race_table,
    top1_accuracy,
    top3_recall,
    winner_mrr,
    winner_rank_distribution,
)

# ---------------------------------------------------------------------------
# Hand-computable fixture: 3 races x 4 drivers.
#   Race 1: winner has the top probability            -> rank 1
#   Race 2: winner has the 2nd probability            -> rank 2
#   Race 3: winner has the lowest probability         -> rank 4
# Expected: top1 = 1/3, top3 = 2/3, MRR = (1 + 1/2 + 1/4)/3 = 7/12
# ---------------------------------------------------------------------------

def _three_races():
    race_ids = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    y_true =   [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    y_prob =   [0.7, 0.2, 0.06, 0.04,
                0.5, 0.3, 0.15, 0.05,
                0.4, 0.3, 0.2, 0.1]
    return y_true, y_prob, race_ids


def test_headline_metrics_hand_computed():
    y_true, y_prob, race_ids = _three_races()
    assert top1_accuracy(y_true, y_prob, race_ids) == pytest.approx(1 / 3)
    assert top3_recall(y_true, y_prob, race_ids) == pytest.approx(2 / 3)
    assert winner_mrr(y_true, y_prob, race_ids) == pytest.approx(7 / 12)
    assert average_winner_probability(y_true, y_prob, race_ids) == pytest.approx(
        (0.7 + 0.3 + 0.1) / 3
    )


def test_per_race_table_contents():
    y_true, y_prob, race_ids = _three_races()
    table = per_race_table(y_true, y_prob, race_ids).set_index("race_id")
    assert table.loc[1, "winner_rank"] == 1 and bool(table.loc[1, "top1_hit"])
    assert table.loc[2, "winner_rank"] == 2 and bool(table.loc[2, "top3_hit"])
    assert table.loc[3, "winner_rank"] == 4 and not bool(table.loc[3, "top3_hit"])
    assert (table["n_drivers"] == 4).all()
    assert table.loc[2, "max_prob"] == pytest.approx(0.5)


def test_winner_rank_distribution():
    y_true, y_prob, race_ids = _three_races()
    dist = winner_rank_distribution(y_true, y_prob, race_ids)
    assert dist.to_dict() == {1: 1, 2: 1, 4: 1}


# ---------------------------------------------------------------------------
# Tie policy — pessimistic competition ranking
# ---------------------------------------------------------------------------

def test_all_tied_probabilities_get_no_credit():
    # 4 drivers, all 0.25: winner's pessimistic rank is 4 — a constant model
    # must not collect free top-1 hits.
    y_true, y_prob, race_ids = [0, 1, 0, 0], [0.25] * 4, [1] * 4
    table = per_race_table(y_true, y_prob, race_ids)
    assert table.loc[0, "winner_rank"] == 4
    assert top1_accuracy(y_true, y_prob, race_ids) == 0.0


def test_two_way_tie_at_top_counts_as_rank_two():
    y_true = [1, 0, 0]
    y_prob = [0.4, 0.4, 0.2]
    table = per_race_table(y_true, y_prob, [1, 1, 1])
    assert table.loc[0, "winner_rank"] == 2
    assert not bool(table.loc[0, "top1_hit"])
    assert bool(table.loc[0, "top3_hit"])


def test_single_driver_race():
    # Degenerate but legal: the sole entrant wins -> rank 1, MRR 1.
    assert top1_accuracy([1], [0.9], [7]) == 1.0
    assert winner_mrr([1], [0.9], [7]) == 1.0


# ---------------------------------------------------------------------------
# Contract violations are loud
# ---------------------------------------------------------------------------

def test_race_without_winner_raises():
    with pytest.raises(ValueError, match="exactly one winner"):
        per_race_table([0, 0], [0.5, 0.5], [1, 1])


def test_race_with_two_winners_raises():
    with pytest.raises(ValueError, match="exactly one winner"):
        per_race_table([1, 1], [0.5, 0.5], [1, 1])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        per_race_table([1, 0], [0.5], [1, 1])


def test_probability_out_of_range_raises():
    with pytest.raises(ValueError, match="outside"):
        per_race_table([1, 0], [1.2, 0.1], [1, 1])


def test_nan_probability_raises():
    with pytest.raises(ValueError, match="NaN"):
        per_race_table([1, 0], [np.nan, 0.1], [1, 1])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="zero rows"):
        per_race_table([], [], [])


# ---------------------------------------------------------------------------
# Probability-quality metrics vs the sklearn reference
# ---------------------------------------------------------------------------

def test_log_loss_and_brier_match_sklearn():
    rng = np.random.default_rng(3)
    y_true = (rng.random(500) < 0.05).astype(int)
    y_true[:3] = 1
    y_prob = rng.random(500)
    assert log_loss_score(y_true, y_prob) == pytest.approx(
        log_loss(y_true, y_prob, labels=[0, 1])
    )
    assert brier_score(y_true, y_prob) == pytest.approx(
        brier_score_loss(y_true, y_prob)
    )


def test_calibration_table_perfectly_calibrated():
    # 100 rows at p=0.25 with exactly 25 positives -> bin gap 0, ECE ~ 0.
    y_prob = np.full(100, 0.25)
    y_true = np.array([1] * 25 + [0] * 75)
    table = calibration_table(y_true, y_prob, n_bins=4)
    assert len(table) == 1
    assert table.iloc[0]["mean_predicted"] == pytest.approx(0.25)
    assert table.iloc[0]["fraction_positive"] == pytest.approx(0.25)
    assert expected_calibration_error(y_true, y_prob, n_bins=4) == pytest.approx(0.0)


def test_ece_hand_computed():
    # Two half-weight bins: |0.0 - 0.1| and |1.0 - 0.9| -> ECE = 0.1.
    y_prob = np.array([0.1] * 50 + [0.9] * 50)
    y_true = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.1)


def test_calibration_bins_cover_edge_probabilities():
    # p=0.0 and p=1.0 must land in the first/last bins, not vanish.
    table = calibration_table([0, 1], [0.0, 1.0], n_bins=10)
    assert table["count"].sum() == 2


# ---------------------------------------------------------------------------
# Aggregates: evaluate_all and per-season summaries
# ---------------------------------------------------------------------------

def test_evaluate_all_keys_and_consistency():
    y_true, y_prob, race_ids = _three_races()
    metrics = evaluate_all(y_true, y_prob, race_ids)
    assert set(metrics) == {
        "top1_accuracy", "top3_recall", "winner_mrr", "avg_winner_probability",
        "median_winner_rank", "log_loss", "brier_score", "ece",
        "n_races", "n_rows",
    }
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["top1_accuracy"] == pytest.approx(top1_accuracy(y_true, y_prob, race_ids))
    assert metrics["n_races"] == 3.0 and metrics["n_rows"] == 12.0
    assert metrics["median_winner_rank"] == 2.0


def test_evaluate_by_season_separates_eras():
    # Season 2020: model perfect (winner top). Season 2021: model wrong.
    y_true = [1, 0, 0, 1]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    race_ids = [1, 1, 2, 2]
    years = [2020, 2020, 2021, 2021]
    by_season = evaluate_by_season(y_true, y_prob, race_ids, years)
    assert list(by_season.index) == [2020, 2021]
    assert by_season.loc[2020, "top1_accuracy"] == 1.0
    assert by_season.loc[2021, "top1_accuracy"] == 0.0
    assert (by_season["n_races"] == 1.0).all()


def test_evaluate_by_season_rejects_race_spanning_years():
    with pytest.raises(ValueError, match="spans multiple years"):
        evaluate_by_season([1, 0], [0.9, 0.1], [1, 1], [2020, 2021])


def test_evaluate_by_season_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_by_season([1, 0], [0.9, 0.1], [1, 1], [2020])


# ---------------------------------------------------------------------------
# Model-agnosticism: any MODEL_ZOO classifier flows through unchanged
# ---------------------------------------------------------------------------

def test_pole_baseline_through_evaluation():
    from src.features.pipeline import FEATURE_COLUMNS
    from src.models.registry import get_model

    # 3 races x 4 drivers; pole (grid_adjusted == 1) wins races 1 and 2.
    rows, y_true, race_ids = [], [], []
    for race in (1, 2, 3):
        for pos in (1, 2, 3, 4):
            row = {c: 0.0 for c in FEATURE_COLUMNS}
            row["grid_adjusted"] = float(pos)
            rows.append(row)
            race_ids.append(race)
            winner_pos = 1 if race in (1, 2) else 3
            y_true.append(int(pos == winner_pos))
    X = pd.DataFrame(rows)[list(FEATURE_COLUMNS)]
    y = pd.Series(y_true)

    pipeline = get_model("pole_baseline", y).fit(X, y)
    y_prob = pipeline.predict_proba(X)[:, 1]

    metrics = evaluate_all(y_true, y_prob, race_ids)
    # Pole wins 2 of 3 races -> top-1 = 2/3. In race 3 the winner (P3) is
    # tied at probability 0 with the two other non-pole drivers ->
    # pessimistic rank 4 -> no top-3 credit: top-3 recall also 2/3.
    assert metrics["top1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["top3_recall"] == pytest.approx(2 / 3)
    assert metrics["avg_winner_probability"] == pytest.approx(2 / 3)
