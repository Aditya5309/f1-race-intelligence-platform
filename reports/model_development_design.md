# Phase 4 — Model Development Design

_Status: ACCEPTED AND IMPLEMENTED (Decisions 012–015; audited 2026-07-04).
Historical design rationale is retained; `context/architecture.md` is authoritative
for the as-built runtime._
_Author: AI agent, Phase 4 planning session, 2026-07-03._
_Depends on: `context/decisions.md` (Decisions 003, 004, 008, 010, 011),
`reports/master_dataset_design.md`, `src/features/pipeline.py` (FEATURE_COLUMNS),
`data/processed/features.parquet` (27,279 rows × 38 cols)._

---

## 0. Inputs and Ground Truth

- **Feature matrix:** `data/processed/features.parquet` — one row per
  (raceId, driverId); 6 identifier columns, 31 features, `winner` target.
- **The model's input list is `src.features.pipeline.FEATURE_COLUMNS`** —
  imported, never re-typed. Identifier columns are for grouping/splitting only.
- Measured split sizes (verified against the built parquet):

  | Split | Years | Races | Rows | Winners | Positive rate |
  |---|---|---|---|---|---|
  | Train | 2010–2021 | 237 | 5,077 | 237 | 4.67% |
  | Validation | 2022–2023 | 44 | 880 | 44 | 5.00% |
  | Test | 2024 | 24 | 479 | 24 | 5.01% |

- **Data finding (new, must be resolved — see §13.1):** the dataset now also
  contains complete 2025 (24 races) and partial 2026 (6 races, season in
  progress) — 611 labeled rows *beyond* the Decision-008 test window, which
  was defined on 2026-06-08 when 2024 was the latest complete season.

---

## 1. Overall Training Architecture

```
data/processed/features.parquet
        │
        ▼
src/models/splits.py            temporal_split(df) → train / val / test frames
        │                       season_folds(train) → expanding-window CV folds
        ▼
src/models/registry.py          model zoo: {name → (estimator factory,
        │                       preprocessing factory, param distributions)}
        │                       + heuristic baselines (pole-sitter)
        ▼
src/models/train.py             CLI orchestration:
        │                         for each candidate: CV on train folds →
        │                         refit on full train → score val →
        │                         log run to MLflow (params, metrics, artifacts)
        │                       tuning mode: randomized search over the zoo's
        │                         param distributions, same fold protocol
        ▼
src/models/evaluate.py          pure metric functions (per-race top-1, top-3
        │                       recall, MRR, log-loss, Brier, calibration) —
        │                       consumed by train.py, tests, and later app/
        ▼
MLflow (project-root SQLite tracking/registry, Decision 004/017)
        │  experiment: f1-winner-prediction
        │  registry:   f1-winner (Staging/Production aliases)
        ▼
src/models/predict.py           load registered model → score one race's field
                                → per-race-normalized win probabilities
                                (the exact function app/api.py calls)
```

Principles carried over from earlier phases:
- **Pure logic in importable modules, thin CLI orchestration** (mirrors
  `src/integration/` + `src/pipelines/`). `train.py` is re-runnable and
  idempotent-by-configuration so a Phase 8 scheduler can call it unchanged.
- **Every preprocessing step with fitted state lives inside the model's
  sklearn `Pipeline`** and is fit only on the fold/split it trains on
  (Decision 011 moved imputation/scaling here from Phase 3).
- **The estimator consumes rows; evaluation consumes races.** All headline
  metrics group by `raceId` before scoring (Decision 003 consequence).

## 2. Repository Structure for `src/models/`

| File | Responsibility | Notes |
|---|---|---|
| `splits.py` | Explicit temporal windows, season-grouped expanding CV, `to_xy()` | Implemented |
| `registry.py` | Five-candidate zoo, preprocessing, distributions, `ColumnGuard` | Implemented |
| `evaluate.py` | Per-race ranking and probability metrics | Implemented |
| `train.py` | CV, tuning, guarded final test, MLflow logging and registration | Implemented |
| `analysis.py` | SHAP, per-race permutation importance, timing | Implemented |
| `calibration.py` | OOF isotonic calibration and registered wrapper | Implemented after original design (Decision 015) |
| `predict.py` | Alias loading, artifact-schema validation, race-normalized scoring | Implemented; Phase 5 serving contract |

Tests mirror this: `tests/test_splits.py`, `tests/test_evaluate.py`,
`tests/test_registry.py`, `tests/test_train.py`, `tests/test_predict.py` (§12).

`AI_AGENT.md` §4's table gains rows for `splits.py` / `registry.py` (same
pattern as the Decision-011 feature-module update).

## 3. Baseline and Candidate Models

Ordered by increasing capacity; each must beat the one above it to justify its
complexity (§9 parsimony rule).

| # | Model | Rationale |
|---|---|---|
| 0a | **Always-negative dummy** | Sanity floor for row-level metrics (95.3% row accuracy, useless top-1) — demonstrates why per-race metrics are the only honest ones |
| 0b | **Pole-sitter heuristic** — P(win)=1 for `grid_adjusted == 1`, else 0 | The real bar: pole wins **43.5%** of races (EDA). Any trained model that can't beat "pick the pole sitter" has learned nothing beyond qualifying. Implemented as a zoo entry with the standard predict-proba interface so it appears in MLflow alongside real models |
| 1 | **Logistic Regression** (`class_weight='balanced'`; pipeline: median imputer + missing-indicator flags + standard scaler) | Calibrated-ish linear baseline; coefficients are directly interpretable; fast; establishes whether the problem is mostly linear in the engineered features. Needs imputation — our NaNs are informative, hence the missing-indicator flags rather than silent median fill |
| 2 | **Random Forest** (`class_weight='balanced'`, imputer, no scaler) | Non-linear baseline with minimal tuning sensitivity; strong overfitting contrast probe vs boosted trees |
| 3 | **XGBoost** (`scale_pos_weight ≈ 20`, native NaN handling — no imputer) | Expected best family: tabular, moderate size, informative missingness handled natively (design doc §5.3 explicitly anticipated tree-native NaN) |
| 4 | **LightGBM** (`scale_pos_weight ≈ 20`, native NaN) | Same class as XGBoost; cheap to include; occasionally better calibrated on small-ish tabular data |

Explicitly out of scope for v1 (Icebox/later): learning-to-rank (LambdaRank)
framing — Decision 003 defers it until the binary model works; neural nets —
5,077 training rows is too small to justify them.

## 4. Train / Validation / Test Strategy

- **Fixed outer split: Decision 008 verbatim** — train 2010–2021, val
  2022–2023, test 2024. Split by `year` column; never shuffled.
- **Inner CV (for tuning and variance estimates): season-grouped
  expanding-window folds** within 2010–2021:

  ```
  fold 1: train 2010–2015 → validate 2016
  fold 2: train 2010–2016 → validate 2017
  ...
  fold 6: train 2010–2020 → validate 2021
  ```

  This *operationalizes* Decision 008's "TimeSeriesSplit within the training
  set" at season granularity rather than sklearn's row-index granularity,
  because row-level `TimeSeriesSplit`:
  (a) can cut a race in half (some drivers in-fold, some out — corrupting
  per-race metrics), and (b) ignores that rows within a race are not
  exchangeable. Six folds × one season each gives 17–22 races per validation
  fold — enough for a stable per-race top-1 estimate. (§13.2 — refinement,
  needs approval.)
- **Fold hygiene:** preprocessing pipelines are fit inside each fold's train
  window only; rolling features already only look backward (Phase 3b), so no
  feature recomputation per fold is needed.
- **Test protocol:** 2024 is scored **once**, by the single selected
  configuration, behind an explicit `--final-test` flag (§11.3). Never used
  for tuning, early stopping, threshold choice, or model choice.
- **Refit policy:** after the one-time test evaluation is reported, the
  selected configuration is refit on train+val (2010–2023) and that artifact
  is what gets registered for serving — the registered model should not
  ignore the two most recent completed seasons it was validated on. The test
  metric on record remains the pre-refit one. (§13.3 — needs approval.)

## 5. Class Imbalance Handling

- **Weighting, not resampling.** `scale_pos_weight ≈ (1-0.0467)/0.0467 ≈ 20`
  for XGBoost/LightGBM (computed from the actual train split at runtime, not
  hardcoded); `class_weight='balanced'` for LogReg/RF.
- **SMOTE/over-sampling is explicitly rejected:** synthetic rows would
  fabricate driver-race entries that never happened, break the per-race grain
  (a race with 3 "winners"), and interpolate across temporally ordered
  neighbours — a leakage vector.
- The per-race argmax framing already neutralizes imbalance for top-1
  accuracy (we compare probabilities *within* a race, where exactly 1 of ~20
  is positive); weighting mainly improves probability scale and log-loss.
- Because weighting distorts calibration, reported probabilities get a
  calibration check (§6); if raw probabilities are badly calibrated, an
  isotonic calibrator fit on **CV fold predictions only** (never val/test) is
  the approved remedy — added only if the calibration curve demands it.

## 6. Evaluation Metrics

All computed by `evaluate.py` as pure functions of `(y_true, y_prob, race_ids)`.

| Metric | Definition | Role |
|---|---|---|
| **Per-race top-1 accuracy** | fraction of races where argmax-probability driver is the winner | **Primary.** Success bar ≥ 40% (project_overview.md); pole baseline 43.5% is the real floor to beat |
| **Per-race top-3 recall** | fraction of races where the winner is among the 3 highest-probability drivers | Secondary; success bar ≥ 70% |
| **Winner MRR** | mean reciprocal rank of the actual winner in the predicted ordering | Tiebreaker sensitive to "how wrong" misses are |
| **Log-loss** (row-level, on raw probabilities) | standard | Probability quality; comparison across runs |
| **Brier score** | standard | Complements log-loss; less tail-sensitive |
| **Calibration curve** (deciles, saved as artifact) | reliability diagram | Gate for the §5 calibration decision |

Reporting rule: headline metrics use **raw model probabilities** for ranking;
`predict.py` additionally exposes **per-race sum-normalized probabilities**
(interpretable as "share of win chance") — normalization is monotonic within a
race so it never changes top-1/top-3, but it is the user-facing number for the
dashboard.

## 7. Hyperparameter Tuning Strategy

- **Stage 1 — zoo defaults:** every candidate runs once with sensible fixed
  defaults through the §4 CV protocol. Establishes the family ranking cheaply.
- **Stage 2 — randomized search on the best one or two families only**
  (expected: XGBoost/LightGBM): ~40 samples from the distributions declared in
  `registry.py` (learning_rate, max_depth/num_leaves, min_child_weight,
  subsample, colsample_bytree, n_estimators via early stopping on the fold's
  validation season, reg_alpha/lambda). Custom loop over the season folds —
  **not** `RandomizedSearchCV`, whose scorer API can't do grouped per-race
  top-1 cleanly and whose CV splitter would be row-based.
  _(Implementation note, 2026-07-03 review: `n_estimators` is SAMPLED from
  the search distribution rather than early-stopped per fold — early stopping
  through an sklearn Pipeline's eval_set is awkward and the sampled approach
  is equivalent for selection purposes at this scale. Revisit only if tuning
  cost grows.)_
- **Selection statistic:** mean CV per-race top-1; ties broken by mean CV
  log-loss. The 2022–2023 val split then arbitrates between the tuned
  finalists — val is not searched against repeatedly (≤ a handful of finalist
  evaluations, logged).
- **No Optuna/new dependency for v1** — 40 × 6 folds × ~5k rows is minutes of
  compute; sophistication is not the bottleneck. Revisit only if search cost
  grows.
- Seeds fixed and logged everywhere (`random_state=42` convention).

## 8. MLflow Experiment Structure

- **Experiment:** `f1-winner-prediction` (Decision 004; local `mlruns/` store).
- **One parent run per candidate-model training invocation**, with child runs
  per CV fold (fold metrics visible individually; parent aggregates
  mean ± std). Tuning creates one parent per sampled config (tagged
  `stage=tune`) so the search is fully auditable.
- **Tags:** `model_family`, `stage` (`baseline`/`default`/`tune`/`finalist`),
  `git-less data fingerprint`: row count + file hash of `features.parquet`,
  `feature_count`, `code_phase=phase4`.
- **Params:** full estimator params, preprocessing choice, split definition,
  `scale_pos_weight` actually used, seed.
- **Metrics:** `cv_top1_mean/std`, `cv_logloss_mean`, `val_top1`, `val_top3`,
  `val_mrr`, `val_logloss`, `val_brier`; `test_*` only ever on the single
  `--final-test` run.
- **Artifacts:** serialized sklearn pipeline (mlflow.sklearn flavor), the
  training schema JSON (feature names + dtypes recorded by the fitted
  pipeline's ColumnGuard via `registry.training_schema()` — inference
  validates against the model's own schema, not repository state),
  calibration plot, feature-importance CSV + plot, SHAP summary plot (§10).
- **Registry:** model name `f1-winner`; aliases `Staging` (best finalist,
  pre-test) and `Production` (post-test, refit-on-train+val artifact, §4).
  No standalone `.pkl` files outside MLflow (AI_AGENT rule 2).

## 9. Model Selection Criteria

Applied in order, on validation (2022–2023) metrics:

1. **Gate:** must beat the pole-sitter heuristic's val top-1. (Pole baseline
   on val is computed, not assumed — 2022–2023 ground-effect regs may shift it.)
2. **Primary:** highest val per-race top-1.
3. **Tiebreak within noise** (Δtop-1 ≤ ~2 races ≈ 4.5 p.p. on 44 races —
   selection noise is real at this sample size): higher val top-3 recall,
   then lower val log-loss.
4. **Parsimony:** if a simpler family is within the noise band of a more
   complex one, the simpler family wins (easier to serve, explain, retrain).
5. **Calibration sanity:** a finalist with a pathological reliability curve
   needs the §5 isotonic step before registration, and that calibrated
   variant is what competes.
6. Test (2024) is scored once for the report card; it does **not** overturn
   the selection unless it reveals an outright bug (e.g., top-1 below the
   always-negative floor → investigate, don't re-tune).

## 10. Feature Importance and Explainability

- **Native importances** (gain for boosted trees, coefficients for LogReg) —
  logged as CSV + bar plot per run. Cheap, always on.
- **Permutation importance** on the validation split, scored by per-race
  top-1 (not row AUC) — measures what actually moves the metric we care about.
- **SHAP** (`TreeExplainer` for the tree finalist): global summary (beeswarm)
  + dependence plots for the top ~6 features + per-race force/waterfall for a
  few 2022–2023 case-study races (saved to `reports/`). Adds `shap` to
  `requirements.txt` (§13.4).
- **Leakage tripwire duty:** explainability doubles as a leakage audit. The
  prior is that grid/qualifying features dominate (pole wins 43.5%), followed
  by driver form and lagged standings. If a feature with no plausible causal
  path (e.g., a circuit-history count on rookie-heavy fields) dominates, or
  any importance profile looks "too clean", that triggers §11 investigation
  before anything is registered.
- Dashboard hook (Phase 5): per-race SHAP values are computable at predict
  time from the stored explainer — `predict.py`'s design leaves room for an
  optional `explain=True` flag later, but v1 does not implement it.

## 11. Leakage Checks Specific to the Modeling Stage

Phase 3b guaranteed the features are leakage-safe *per row*; Phase 4 adds the
ways *training procedure* can leak:

1. **Feature-list integrity (executable):** model input columns ==
   `FEATURE_COLUMNS` imported from `src.features.pipeline` — no identifier
   (`raceId`, `driverId`, `year`, …) and no `POST_RACE_OUTCOME_COLUMNS`
   member may enter the design matrix. Asserted in `registry.py` at pipeline
   build time and re-asserted by `predict.py` against the artifact's stored
   column list.
2. **Split/fold integrity (tested):** train ∩ val ∩ test raceIds are empty;
   every race's rows land entirely in one side of every fold boundary; every
   CV train season strictly precedes its validation season; test years touch
   nothing else.
3. **Test-set discipline (enforced, not just promised):** `train.py` only
   computes test metrics behind `--final-test`, which also tags the run
   `final=true`; the selection code path physically cannot see test rows.
4. **Fitted-state containment (tested):** imputer/scaler statistics are fit
   inside each fold's train window (verify: fold-fit imputer's learned median
   differs from the full-data median on a synthetic frame constructed to make
   them differ).
5. **Shuffled-target canary (tested):** training the smallest real model on
   within-race-permuted winners must collapse per-race top-1 toward chance
   (~1/field ≈ 5%) — catches any accidental target-derived signal in the
   plumbing.
6. **Too-good-to-be-true tripwire (runtime warning):** val or CV top-1
   > 70% emits a loud warning demanding investigation before proceeding —
   pole predicts 43.5%, and published F1 winner models rarely exceed ~60–65%.
7. **Duplicate-winner guard:** modeling-window races must have exactly one
   winner (the two shared-drive anomalies are pre-2010 and excluded by the
   split; assert this stays true after any data refresh).

## 12. Required Tests and Validation

| Test file | Coverage |
|---|---|
| `tests/test_splits.py` | Decision-008 year boundaries; disjoint raceIds across splits; race-integrity within folds; expanding-window monotonicity; deterministic output; §11.2 |
| `tests/test_registry.py` | every zoo entry builds a fit-able sklearn pipeline on synthetic data; design-matrix columns == FEATURE_COLUMNS (§11.1); pole-baseline heuristic produces valid probabilities; class-weight/scale_pos_weight computed from data not hardcoded |
| `tests/test_evaluate.py` | per-race top-1/top-3/MRR on hand-computable synthetic races (incl. tie probabilities, a race missing from predictions raises, single-driver race edge); log-loss/Brier against sklearn reference; calibration binning |
| `tests/test_train.py` | end-to-end smoke on a tiny synthetic feature frame with MLflow pointed at a tmp dir: runs, logs expected params/metrics/artifacts, is idempotent on re-run; `--final-test` guard (§11.3); fold-fit containment (§11.4); shuffled-target canary (§11.5) |
| `tests/test_predict.py` | loads a registered tmp model; per-race normalization sums to 1 and preserves ranking; schema mismatch (missing/extra/reordered feature column) raises; output sorted desc; deterministic |

Definition of done for Phase 4 additionally requires: all §11 checks
implemented as code (not prose), 149 existing tests still green, one
`--final-test` run logged, and the selected model registered with its
FEATURE_COLUMNS artifact.

## 13. Approval Record (accepted and implemented)

**STATUS: ALL SEVEN POINTS APPROVED 2026-07-03** (Decision 012 flipped to
Accepted). Point 7 approved with an amendment: the feature freeze applies
only to Phase 4 model comparison and hyperparameter tuning — future feature
additions are evaluated after baseline performance, SHAP, and error analysis.
Feature classification recorded as Decision 013. Implementation proceeds one
module at a time with a user checkpoint after each.

Original proposal text follows (retained for the record):

1. **2025–2026 data policy.** The dataset now contains 2025 (complete) and
   2026 (partial) — 611 labeled rows newer than the Decision-008 test set.
   **Recommendation: keep Decision 008 unchanged** (train ≤2021, val
   2022–2023, test 2024) and designate 2025–2026 as a *forward holdout*: never
   touched in Phase 4, reserved as the first real "new data arrives" scenario
   for Phase 8 monitoring/retraining rehearsal. Alternative (rejected but
   listed): shift the whole split forward (train 2010–2022 / val 2023–2024 /
   test 2025) — more training data, but re-litigates an accepted decision
   mid-phase and consumes the only true out-of-time data before the
   monitoring story needs it.
2. **Season-grouped expanding-window CV** replacing row-level
   `TimeSeriesSplit` as the operationalization of Decision 008's CV clause
   (§4 rationale: row splits cut races in half and break per-race metrics).
3. **Refit-on-train+val for the registered artifact** after the one-time test
   evaluation (§4). The reported test metric stays the pre-refit number.
4. **`src/models/` gains `splits.py` and `registry.py`** beyond the
   backlog's train/evaluate/predict trio; `AI_AGENT.md` table updated
   accordingly.
5. **New dependency: `shap`** (and confirming `scipy` transitively present)
   in `requirements.txt`; also start version-pinning per architecture.md's
   standing note ("pin once confirmed working").
6. **Pole-sitter heuristic added as a formal MLflow-tracked baseline**
   (not in the original backlog, which started at Logistic Regression).
7. **Feature set frozen for Phase 4 v1** (added after the §14 review):
   train and measure on the 31 implemented features first; the §14.5 ranked
   improvements are v2 work gated on v1 importance/error analysis, not
   Phase 4 blockers.

---

## 14. Critical Feature-Set Review (F1 domain perspective)

_Added 2026-07-03 after `context/domain_knowledge.md` was written. Claims
marked "(verified)" were measured on this project's data during this review;
domain-knowledge cross-references use that document's section numbers._

### 14.1 Verified baseline correction

**The pole-sitter bar is ~50%, not 43.5%, inside the modeling window**
(verified: pole win rate by era segment — 2010–2013: 48.1%, 2014–2021:
52.5%, 2022–2024: 51.5%). The EDA's 43.5% figure comes from a wider
denominator. Consequence: the §9.1 selection gate must use the pole
baseline **computed on the same split at runtime** (already designed), and
success expectations shift: a model needs val top-1 comfortably above ~50%
to demonstrate real learning. The project's ≥40% success metric
(project_overview.md) is below the naive baseline — Phase 4 should report
against the pole baseline, and the success-metric wording should be
revisited when Decision 012 is accepted.

### 14.2 Missing high-impact features (ranked, all v2 — none block Phase 4)

1. **`grid_minus_qualifying_position`** — penalty/recovery signal. Verified:
   30.0% of 2010–2024 rows have `qualifying_position != grid`, so support is
   broad. Trivial to implement, fully pre-race, in-schema. (domain doc §4)
2. **Teammate-delta features** (rolling quali-gap-to-teammate,
   finish-vs-teammate) — the only car-controlled driver-skill measure
   available in this schema; best de-confounder of the car-dominance problem
   (domain doc §2). Moderate effort: must reuse the race-grain teammate-
   exclusion pattern.
3. **`driver_positions_gained_last_5`** — racecraft proxy (grid − finish of
   prior races); trivial, leakage-safe under standard shift discipline.
4. **`circuit_pole_win_rate_prior`** — circuit overtaking-difficulty proxy
   interacting with grid features; must be prior-races-only (domain doc §5).
5. **Constructor lineage mapping** — fixes rebrand history resets. Verified:
   constructor-form NaN coverage in-window is only 0.37% of rows, so the
   bias is small and concentrated in a rebranded team's first races —
   real but low priority (domain doc §3).
6. **`driver_age_at_race`** — cheap, low expected gain.
7. **Weather forecast** (external source) — highest-value external addition,
   v3 (domain doc §9).
8. **Sprint features** — v3, must be per-season format-aware (domain doc §4).

### 14.3 Era-fragility findings

- **Form-feature predictiveness is non-stationary (verified):**
  point-biserial correlation with `winner` for `driver_wins_last_10` rises
  0.405 (2010–2013) → 0.464 (2014–2021) → 0.620 (2022–2024), and
  `constructor_wins_last_5` 0.310 → 0.445 → 0.464 — form features look
  strongest in high-dominance eras. `grid_position_norm` is era-stable
  (−0.32 across all three segments). Implication: a model selected on the
  2022–2023 val split (peak dominance) will lean on form features exactly
  when the cost-cap convergence + 2026 reset (domain doc §1) makes them
  least reliable going forward. Mitigations: per-season metric reporting
  (already designed, §9), and expect the 2025–2026 forward holdout to
  punish form-heavy models — that is signal, not pipeline failure.
- **Raw `q1/q2/q3_sec` are ~99% circuit identity (verified):** between-circuit
  variance share of `q1_sec` is 98.99% — absolute qualifying seconds encode
  which track it is, not who is fast. The informative content is already
  captured by `qualifying_gap_to_pole_pct` (relative, era-robust) and
  `reached_q2/q3`. Keep the raw seconds in v1 (trees can ignore them) but
  expect near-zero or spurious importance; **v2 cleanup candidate: drop them
  or replace with within-session deltas.** If SHAP shows them mattering,
  treat it as circuit-memorization, not pace signal.
- **`driver_points_last_5`:** mildly era-drifted within the window (2019
  fastest-lap point, 2021+ sprint points, fastest-lap point removed 2025) —
  position-based features are the robust siblings. Low severity; note for
  importance interpretation.
- **`grid_adjusted` range shifts in 2026** (22-car fields, verified in
  domain doc §1): values 21–23 unseen in training. `grid_position_norm`
  covers this; models should prefer it — check via importance analysis.

### 14.4 Biases and remaining leakage risks

- **Biases (accepted, documented):** rebranded-constructor under-prediction
  (small — §14.2 item 5); dominance-era memorization (the training window
  contains the 2014–2020 Mercedes streak; §14.3 shows the model will
  overrate form persistence); bottom-team drivers get ~0 probabilities
  (correct — no such driver wins in-window); class-weighting distorts raw
  calibration (handled by §5/§6 calibration gate).
- **Feature-level leakage: none found.** Re-audited all 31 features against
  domain doc §7: `driver_experience_races` is cumcount (prior-only);
  `qualifying_gap_to_pole_pct` uses same-weekend pre-race data (legitimate,
  §7 rule 6); field size for `grid_adjusted` derives from the entry list
  (known pre-race). All rolling/lagged features covered by existing tests.
- **One new modeling-stage risk, added to §11 scope:** `features.parquet`
  now contains 2025–2026 rows (forward holdout). `splits.py` must select
  **explicit year ranges** for train/val/test — any "everything up to
  max(year)" logic would silently swallow the forward holdout into
  training. Add a dedicated test: split outputs contain no year > 2024.

### 14.5 Ranked improvement plan

| Rank | Action | Phase | Expected impact |
|---|---|---|---|
| 1 | Train v1 on frozen feature set; per-season error + importance analysis | Phase 4 | Measurement before expansion — decides everything below |
| 2 | `grid_minus_qualifying_position` | v2 | Medium; trivial cost |
| 3 | Teammate-delta form features | v2 | Medium-high; the de-confounder |
| 4 | `driver_positions_gained_last_5` | v2 | Low-medium; trivial cost |
| 5 | `circuit_pole_win_rate_prior` | v2 | Medium at street circuits |
| 6 | Drop/replace raw `q*_sec` if importance confirms noise | v2 | Cleanup; reduces circuit memorization |
| 7 | Constructor lineage table | v2 | Low overall (0.37% coverage), targeted fix |
| 8 | `driver_age_at_race` | v2 | Low |
| 9 | Weather forecast source | v3 | High but external dependency |
| 10 | Sprint features (format-aware) | v3 | Low-medium; 9% of races |
