# Phase 4 — Model Selection Report

_Date: 2026-07-03; milestone update 2026-07-04. Author: AI agent._
_Protocol: Decision 012 (`reports/model_development_design.md`), Decision 008 split,
Decision 013 feature classes. Data fingerprint: `27279rows-80744c5053e2`
(train 5,077 / val 880 / test 479 rows; 237 / 44 / 24 races)._
_All runs logged to MLflow experiment `f1-winner-prediction` (sqlite:///mlflow.db)._

---

## 1. Executive Summary

**Serving model: OOF-isotonic-calibrated tuned Logistic Regression**
(`logreg`, `model__C = 0.01653693718282442`), registered as **`f1-winner` v2
@Staging**. Version 1 is the raw tuned pipeline retained as registry history.
No `Production` alias exists.

- **Validation (2022–2023, 44 races):** per-race top-1 **68.2%** (30/44) vs the
  pole-sitter baseline's 54.5% — a 6-race edge; top-3 recall **88.6%**;
  winner MRR 0.796.
- **Final test (2024, 24 races, scored exactly once):** top-1 **45.8%** (11/24),
  top-3 recall **75.0%**, MRR 0.643 — the model **ties the pole baseline on
  top-1** (pole also won 11/24 races in 2024) while adding substantial ranking
  value beyond P1 (winner ranked top-2 in 17/24 races, median rank 2).
- Both project success bars are met on both evaluation splits: beats-or-ties
  the pole baseline on top-1 (beats on val, ties on test) and top-3 recall
  ≥ 70% (88.6% val / 75.0% test).
- The winning margin over pole is **era-dependent**: large in the 2023
  dominance season (90.9% vs 63.6%), zero in the competitive 2022/2024
  seasons (~46% for both model and pole). This confirms the design's §14.3
  era-nonstationarity warning and is documented as the model's primary
  limitation, not a defect.
- Raw probabilities were miscalibrated (val ECE 0.153) from balanced class
  weighting. The validated OOF isotonic remedy is implemented in
  `src/models/calibration.py` and registered in v2 (ECE 0.153 → 0.012,
  log-loss 0.268 → 0.088, top-1 unchanged). See Decision 015.
- No leakage indicators: no tripwire warnings, explainability profile matches
  the domain prior (grid/qualifying features dominate), shuffled-target canary
  and all §11 guards green (235/235 tests passing before execution).

---

## 2. Model Comparison

Machine-readable version: `reports/model_comparison.csv`.

| Model | Config | CV top-1 (mean ± std) | Val top-1 | Val top-3 | Val MRR | Val log-loss | Val ECE | Fit (s) | Predict (ms/row) |
|---|---|---|---|---|---|---|---|---|---|
| pole_baseline | rule | 0.519 ± 0.079 | 0.545 | 0.545 | 0.568 | 1.638 | 0.045 | 0.001 | 0.0003 |
| **logreg** | **tuned C=0.0165** | **0.560 ± 0.063** | **0.682** | **0.886** | **0.796** | 0.268 | 0.153 | 0.034 | 0.0022 |
| logreg | default | 0.482 ± 0.093 | 0.636 | 0.886 | 0.764 | 0.264 | 0.137 | ~0.03 | ~0.002 |
| random_forest | default | 0.534 ± 0.097 | 0.614 | 0.886 | 0.759 | **0.102** | **0.029** | 0.518 | 0.0845 |
| random_forest | tuned (rejected) | 0.566 ± 0.086 | 0.523 | 0.864 | 0.708 | 0.163 | 0.075 | — | — |
| xgboost | default | 0.459 ± 0.115 | 0.568 | 0.818 | 0.716 | 0.132 | 0.029 | 0.442 | 0.0052 |
| lightgbm | default | 0.446 ± 0.097 | 0.545 | 0.864 | 0.704 | 0.165 | 0.038 | 0.530 | 0.0048 |

Selection noise band (design §9.3): 1 race on val = 2.3 p.p.; the band is
~4.5 p.p. (2 races).

## 3. Cross-Validation Results

Six season-grouped expanding-window folds inside 2010–2021 (fold k trains
2010..2014+k, validates season 2015+k).

- **Tuned logreg fold-by-fold top-1:** 2016: 0.619, 2017: 0.500, 2018: 0.476,
  2019: 0.524, 2020: 0.647, 2021: 0.591 → mean 0.560, std 0.063 — the
  **most consistent** trained model (lowest CV std in the zoo).
- The pole baseline's CV mean is 0.519 — tuned logreg is the only candidate
  whose CV mean beats pole (RF-tuned did too, at 0.566, but failed on val;
  see §11).
- Stage-1 defaults: only random_forest (0.534) exceeded pole on CV; both
  boosters landed under 0.46 with the largest fold variance (XGBoost std
  0.115) — with only 17–22 races per fold, boosted trees' variance dominates
  at this data size.

## 4. Hyperparameter Tuning (stage 2)

Randomized search, 40 configs × 6 folds each, on the two best stage-1
families (design §7: "best one or two"); the boosters were not tuned
(both trailed on CV *and* val — documented as a limitation in §13).

| Family | Best config | CV top-1 before → after | Val top-1 before → after |
|---|---|---|---|
| logreg | `C = 0.0165` (strong L2) | 0.482 → 0.560 | 0.636 → **0.682** |
| random_forest | `max_depth=6, max_features=0.776, min_samples_leaf=9, n_estimators=366` | 0.534 → 0.566 | 0.614 → **0.523** ⚠ |

The strong regularization winning for logreg is coherent with a 31-feature,
237-positive-row problem: shrink coefficients hard, trust the few dominant
signals. The RF result is the opposite story — see §11 (suspicious results).

## 5. Validation Results (2022–2023, 44 races)

Tuned logreg: top-1 **0.682**, top-3 **0.886**, MRR 0.796, avg winner
probability 0.865, median winner rank 1, log-loss 0.268, Brier 0.088,
ECE 0.153. Winner-rank distribution: rank 1 in 30 races, rank 2 in 5,
rank 3 in 4, rank 4 in 4, rank 5 in 1 — the winner is never ranked worse
than 5th on validation.

## 6. Final Test Results (2024, 24 races — scored once, guarded)

Run `logreg-FINAL-TEST` (tag `final=true`), fit on train only (pre-refit
protocol, design §4):

| Metric | Value |
|---|---|
| top-1 accuracy | **0.4583** (11/24) |
| top-3 recall | **0.7500** (18/24) |
| winner MRR | 0.6434 |
| avg winner probability | 0.773 |
| median winner rank | 2 |
| log-loss / Brier / ECE | 0.328 / 0.110 / 0.167 |

Winner-rank distribution: rank 1 ×11, rank 2 ×6, rank 3 ×1, rank 4 ×1,
rank 5 ×3, rank 6 ×1, rank 11 ×1. Reference point: the pole sitter won
11/24 (45.8%) of 2024 races — identical top-1. The test does not overturn
the selection (§9.6: it is a report card, not a re-selection input).

## 7. Per-Season Performance (regulation-era view)

| Season | Model top-1 | Pole top-1 | Model top-3 | Context |
|---|---|---|---|---|
| 2022 (val) | 0.455 | 0.455 | 0.818 | 1st ground-effect year; competitive |
| 2023 (val) | **0.909** | 0.636 | 0.955 | historic Red Bull dominance |
| 2024 (test) | 0.458 | 0.458 | 0.750 | cost-cap convergence; 7 different winners |

The pattern is unambiguous: **the model's edge over the pole heuristic
comes almost entirely from dominance seasons**, where rolling form and
lagged standings carry real signal. In competitive seasons the model
degenerates to approximately the pole pick on top-1 — but retains
significant value in ranking depth (top-3 recall 75–82% vs pole's
by-construction 45%) and probability quality. This is exactly the
era-fragility documented in design §14.3 and domain_knowledge.md §1; expect
the effect to persist into 2025–2026 (forward holdout) under cost-cap
convergence and the 2026 regulation reset.

## 8. Calibration Analysis

- All trained models except logreg are reasonably calibrated (val ECE
  0.029–0.038). Logreg's `class_weight='balanced'` inflates probability
  scale (ECE 0.153; e.g. the 0.5–0.6 predicted bin realizes only 3%
  winners). Ranking is unaffected — the reliability curve is monotone.
- **Design §5 remedy validated empirically** (out-of-fold isotonic
  calibrator fit on the 6 CV folds' predictions only, never val/test):
  ECE 0.153 → **0.012**, log-loss 0.268 → **0.088**, Brier 0.088 → 0.026,
  top-1 **unchanged** at 0.682, top-3 −1 race (isotonic plateaus create
  probability ties, penalized by the pessimistic tie policy).
- **Implemented after the original selection session:** Decision 015 added
  `src/models/calibration.py`, its tests, training/registration integration, and
  calibrated registry version 2. Per-race normalization remains a presentation
  transform; it is not a substitute for artifact calibration.

## 9. Feature Importance, Permutation Importance, SHAP

Artifacts: `reports/phase4_analysis/` (plots + CSVs, also logged to MLflow
runs `logreg-analysis` and `random_forest-analysis`, stage=analysis).

- **Native + SHAP agree on the head:** for the finalist, mean-|SHAP| is led
  by `grid_position_norm` (0.66), `grid_adjusted` (0.61),
  `qualifying_position` (0.56) — then a step down to
  `constructor_standing_position_prev` (0.26), `reached_q2`,
  `driver_podiums_last_5`, `driver_standing_position_prev`. The RF
  runner-up's SHAP ordering is materially the same. **This matches the
  domain prior (grid dominance) — the §10 leakage-audit reading of
  importances is clean.**
- **Permutation importance (per-race top-1 scored, validation):** the
  finalist's top-1 depends most on `qualifying_position` (−5.5 p.p. when
  shuffled), then `driver_wins_last_3` and `driver_standing_wins_prev`
  (−4.5 p.p. each). Note the era caveat: permutation on a 2022–2023 val
  split overstates form features for the same reason §7 shows.
- **Raw `q1/q2/q3_sec` are near-noise as predicted** (§14.3): none appears
  in the finalist's SHAP top-10; `q2_sec` shows a marginal 1.8 p.p.
  permutation effect. They remain v2 drop candidates.
- **Missing-indicator flags carry real signal** (e.g.
  `missingindicator_driver_circuit_avg_finish`, mean-|SHAP| 0.18) —
  confirming the informative-NaN design; "no circuit history" is genuinely
  predictive information.
- SHAP case-study waterfalls (highest-confidence winner, lowest-confidence
  winner, 2022 round 1 post-reset race) are in `reports/phase4_analysis/`.

## 10. Decision-013 Feature-Class Analysis

Share of summed native importance (finalist / RF runner-up):

| Class | logreg | random_forest | Reading |
|---|---|---|---|
| Stable (12) | **59.3%** | **60.8%** | Majority of signal — healthy |
| Era-sensitive (12) | 24.5% | 29.2% | Meaningful minority; the §7 era gap lives here |
| Experimental (7 + derived indicators) | 16.2% | 10.0% | Minor; mostly missing-indicator derivatives |

Per Decision 013's suspicion rule, a model dominated by Experimental
features would indicate circuit memorization — **both models pass**: signal
is concentrated in Stable (grid/qualifying/normalized) features. The
era-sensitive quarter of the signal is precisely what §7's season spread
predicts will decay at era boundaries.

## 11. Model Validation Review (pre-recommendation checklist)

| Check | Finding |
|---|---|
| vs pole baseline | Val: +6 races (68.2% vs 54.5%), gate passed. Test: tie (45.8% both). Top-3/MRR/log-loss beat pole everywhere (pole's top-3 = its top-1 by construction). |
| CV consistency | Finalist has the lowest CV std of the zoo (0.063); fold range 0.476–0.647 with no anomalous fold. |
| Val vs test generalization | Pooled val (68.2%) → test (45.8%) looks like a drop, but per-season view (§7) shows a stable pattern: ~46% in competitive seasons (2022, 2024), ~91% in the dominance season (2023). The gap is era composition, not classic overfitting. |
| Regulation eras | Edge over pole is dominance-concentrated (§7). Expected to shrink under 2025–2026 convergence/reset. |
| Calibration | Raw ECE 0.153 (balanced-weighting inflation); monotone curve, ranking unaffected; OOF-isotonic remedy validated (ECE 0.012), deferred to module-5 checkpoint (§8). |
| Feature importance | Grid/qualifying dominate (59–61% Stable-class share) — matches domain prior; no implausible feature at the top. |
| SHAP | Confirms native ordering; raw q*_sec near-zero as predicted; informative-NaN indicators contribute. |
| Training cost | 0.03 s fit on the full training split — cheapest trained candidate by ~15×. |
| Inference latency | 0.0022 ms/row ≈ 44 µs per 20-driver race — negligible; well within any serving budget. |
| Overfitting signs | Tuned logreg: none visible (CV 0.560 < val 0.682, and the val excess is the 2023 era effect). Tuned RF: **yes** — CV 0.566 → val 0.523 (below pole gate); the search overfit the CV protocol; default RF generalized better. |
| Temporal leakage signs | None. No tripwire (>70%) warnings on any pooled metric; shuffled-target canary and all §11 guards in the test suite green; importance profile has no leakage signature; the only >70% figure (2023 per-season slice, 90.9%) is explained by genuine single-team dominance and is mirrored by pole's own 63.6% that year. |
| Surprising results | (1) Linear model beats all tree models — plausible at 237 positive training rows; regularized linear + strong engineered features is the right capacity. (2) Tuned RF regression on val (above). (3) Boosters' defaults underperform even pole on CV — untuned, high fold variance; their true ceiling is unmeasured (limitation, §13). (4) Model ties pole exactly on 2024 top-1 — both got 11/24, largely the same races. |

## 12. Production Model Recommendation

**Registered serving artifact (done): `f1-winner` v2 @Staging = calibrated tuned
logreg.** Version 1 is the uncalibrated historical artifact.

**Original selection artifact: `f1-winner` v1 = tuned logreg
(`model__C = 0.01653693718282442`), fit on train (2010–2021).**
**Production promotion: deliberately NOT performed.** The calibration decision and
module-5 inference work are complete; a Production alias remains a separate approval
and refit-policy decision.

**Why logreg:**
1. Passes the §9.1 pole gate on validation with the largest margin (+6 races).
2. Highest val top-1 (0.682) — no other candidate is within the 2-race noise
   band (RF-default 0.614 is 3 races behind).
3. Best CV consistency (std 0.063) and the only candidate whose CV mean
   beats pole while also beating it on val.
4. Parsimony (§9.4): simplest trained family — interpretable coefficients,
   trivially cheap to fit/serve/retrain, fewest tuned hyperparameters (one).
5. Cleanest explainability story: SHAP/native/permutation all agree and
   match the domain prior.

**Why not the others:**
- **random_forest (default):** 3 races behind on val top-1; better raw
  calibration and log-loss, but the primary metric decides (§9.2) and it is
  outside the noise band. Its tuned variant failed the pole gate on val.
- **random_forest (tuned):** rejected outright — val top-1 0.523 < pole
  0.545 (§9.1 gate failure) despite the best CV mean; a textbook
  CV-overfit.
- **xgboost / lightgbm (defaults):** below pole on CV mean, 5–6 races
  behind the finalist on val, worst fold variance; not tuned per design §7
  (not among the top-two stage-1 families), so eliminated at stage 1.
- **pole_baseline:** the floor, not a candidate — but note it remains
  embarrassingly competitive in convergence seasons; every report should
  keep printing it next to the model.

## 13. Known Limitations and Risks

1. **Era dependence (the big one):** the model's advantage over "pick the
   pole sitter" is concentrated in dominance seasons. In competitive
   seasons (2022, 2024, plausibly 2025–2026) it adds top-3/probability value
   but no top-1 edge. Set stakeholder expectations accordingly.
2. **Raw probability scale is inflated** (ECE 0.15–0.17) until the isotonic
   step is wired in; per-race normalized shares (the predict.py contract)
   mitigate the user-facing impact. Do not consume raw probabilities as
   absolute win chances in the meantime.
3. **Boosters were eliminated untuned.** Their stage-1 defaults were weak,
   but their tuned ceiling on this data is unmeasured — a modest open
   question, acceptable under the §7 budget rule.
4. **Small-sample selection noise:** 44 val races; the logreg-vs-RF gap is
   3 races. The selection is defensible but not statistically airtight.
5. **2024 test verdict is 24 races** — one season, one era condition. The
   45.8% number should be quoted with that context.
6. **Single-feature-set freeze:** v1 deliberately measured only the frozen
   31-feature set (Decision 012 amendment). Ranked v2 features
   (grid−quali delta, teammate deltas, circuit pole-conversion rate) are
   untested upside.
7. **Tripwire proximity:** val top-1 0.682 sits near the 0.70 alarm
   threshold; any future "improvement" that crosses it should be treated
   with suspicion first, celebration second.

## 14. Future Improvements (ranked, consistent with design §14.5)

1. **Production refit/promotion policy** (calibration wrapper is implemented) on
   train+val — module-5 checkpoint items.
2. `grid_minus_qualifying_position` (trivial, pre-race, broad support).
3. Teammate-delta form features (best car/driver de-confounder in schema).
4. `driver_positions_gained_last_5` (racecraft proxy).
5. `circuit_pole_win_rate_prior` (era-stable circuit interaction).
6. Drop/replace raw `q1/q2/q3_sec` (confirmed near-noise).
7. Tune XGBoost/LightGBM once the v2 feature set exists (re-open the
   family question with more signal).
8. Season-weighted or era-aware training when 2026 data enters (Phase 8).

## 15. Reproduction

```bash
python -m pytest tests/                                          # 235 passing
python -m src.models.train --model all                           # stage 1
python -m src.models.train --model logreg --tune                 # stage 2 (logreg)
python -m src.models.train --model random_forest --tune          # stage 2 (RF)
python -m src.models.analysis --timing --model logreg --params '{"model__C": 0.01653693718282442}'
python -m src.models.analysis --model random_forest
python -m src.models.train --model logreg --final-test --params '{"model__C": 0.01653693718282442}'
python -m src.models.train --model logreg --register Staging --calibrate --params '{"model__C": 0.01653693718282442}'
mlflow ui --backend-store-uri sqlite:///mlflow.db                # inspect runs
```

MLflow inventory (experiment `f1-winner-prediction`): 5 stage-1 parent runs
(+ 6 fold children each), 80 tune runs (40 logreg + 40 RF), 2 finalist runs,
2 analysis runs + 1 timing run, 1 final-test run (tag `final=true`),
The milestone registry contains raw v1 history and calibrated v2 with alias
**Staging** only. `Production` remains unset.
