# Architectural Decisions Log

Decisions are append-only. Do NOT edit past decisions — add new ones if something changes.

---

## Decision 001 — Project folder structure follows production ML layout

**Date:** 2026-06-08
**Status:** Accepted

**Context:** Needed a folder structure that separates concerns cleanly and supports
a production deployment path (API + dashboard), not just a research notebook.

**Decision:** Use `src/data/`, `src/features/`, `src/models/` as importable Python
packages; `app/` for serving layer; `tests/` for pytest; `notebooks/` for EDA only.

**Consequences:**
- `setup.py` with `pip install -e .` required before running any `src.*` imports
- Notebooks should import from `src`, not contain logic directly
- All business logic lives in `src`; notebooks are read-only consumers

---

## Decision 002 — `data/` and `models/` are gitignored

**Date:** 2026-06-08
**Status:** Accepted

**Context:** Data files (parquet, CSV) and model artifacts (.pkl, .json) can be
large and contain potentially licensed content (Ergast API data).

**Decision:** Add `data/` and `models/` to `.gitignore`. Use `.gitkeep` placeholders
so the directories exist locally after cloning.

**Consequences:**
- New contributors must run the data download script before any code works
- Model artifacts must be versioned via MLflow, not git
- A `README` or `context/current_status.md` must always describe how to bootstrap data

---

## Decision 003 — Binary classification framing as baseline

**Date:** 2026-06-08
**Status:** Proposed (not yet implemented)

**Context:** Could frame as multiclass (predict exact P1 from 20 drivers),
binary per driver, or a ranking problem.

**Decision:** Start with binary classification per driver (one row per race+driver,
target = 1 if winner). Select the driver with highest predicted probability at inference.

**Rationale:** Simplest to implement and evaluate; well-supported by XGBoost/LightGBM.
Can extend to ranking later without throwing away the binary model.

**Consequences:**
- Training set is heavily imbalanced (~5% positive rate); must handle with class weights
- Evaluation needs per-race grouping (pick top-1 per race, not overall accuracy)

---

## Decision 004 — MLflow for experiment tracking (not Weights & Biases)

**Date:** 2026-06-08
**Status:** Proposed

**Context:** Need experiment tracking. Options: MLflow (local/self-hosted), W&B (cloud).

**Decision:** Use MLflow with local file store (`mlruns/`) initially. Can migrate to
a remote tracking server later without changing training code.

**Consequences:**
- `mlruns/` added to `.gitignore`
- Run `mlflow ui` locally to inspect experiments
- Model serialization uses MLflow's artifact store, not manual pickle paths

---

## Decision 005 — Use pre-downloaded Ergast CSV dump as primary data source

**Date:** 2026-06-08
**Status:** Accepted

**Context:** The original plan assumed data would be fetched live from FastF1 and the
Ergast REST API. On inspecting `data/`, 14 Ergast-format CSVs were already present
(results, qualifying, races, drivers, constructors, standings, lap times, etc.).
The Ergast API was also officially deprecated at end of 2024.

**Decision:** Use the local CSV files in `data/` as the primary data source.
`loader.load_csv()` reads them directly. No live API calls are needed for historical
results data.

**Consequences:**
- No FastF1 cache setup or API key required for the core pipeline
- FastF1 is still available as an optional source if telemetry or weather features are added later
- The data bootstrap step is now: clone repo + CSVs are already there (no download script needed)
- `data/` remains gitignored; new contributors must obtain the CSVs separately

---

## Decision 006 — Dual-source result status: positionText first, statusId fallback

**Date:** 2026-06-08
**Status:** Accepted

**Context:** `results.csv` has two overlapping columns for race outcome: `positionText`
(single-character codes + numeric strings) and `statusId` (FK into `status.csv` with
142 entries). Neither column alone covers all cases cleanly.

**Decision:** In `clean_results()`, derive `result_status` using `positionText` as
the primary signal (numeric = Finished; R/D/E/N/F/W = known categories), with
`statusId` as fallback for rows where `positionText` is null. All 140+ mechanical
and accident statusIds not in the explicit sets default to "Retired".

**Consequences:**
- The 6-category `result_status` column is the single authoritative outcome field
- `finished` (bool) is derived from `result_status`, not directly from `position`
- Adding a new outcome category requires updating `_POSITION_TEXT_MAP` or the
  status ID frozensets in `cleaner.py` — both are clearly documented

---

## Decision 007 — Repair logic lives in build_interim.py, not cleaner.py

**Date:** 2026-06-08
**Status:** Accepted

**Context:** Running the full pipeline against results.csv revealed two real data
quality issues: (1) 85 duplicate (raceId, driverId) pairs (176 rows), likely sprint
race entries or post-race corrections added as new rows in Ergast; (2) 2 rows in
raceId=71 with null `position` despite a numeric `positionText`.

**Decision:** Repair logic is placed in `build_interim.py`, not `cleaner.py`.
`clean_results()` only transforms a single DataFrame; it has no concept of "choose
between two competing rows for the same race entry". Deduplication and cross-row
arbitration are pipeline concerns, not cleaning concerns.

Repair strategies chosen:
- Duplicates: prefer Finished rows over non-Finished; break ties by highest resultId
  (most recently added = most likely corrected entry). 91 rows dropped.
- Null position on finished row: fill from positionText where it is a numeric string.
  2 rows fixed.

**Consequences:**
- `clean_results()` contract is unchanged (transform only, no row arbitration)
- `validate_results()` correctly catches these issues when run on raw cleaned data
- Re-running `build_interim.py` is idempotent: same repairs applied, same output
- Future data updates to results.csv should re-run `build_interim.py`

---

## Decision 009 — Master modeling dataset design (grain, joins, leakage rules)

**Date:** 2026-07-03
**Status:** Proposed (design only — not yet implemented)

**Context:** Phase 3 (Feature Engineering) needs a fixed contract for the master
modeling table before `src/features/engineer.py` is written, so rolling/standings
leakage risks are caught by design rather than discovered in a trained model's
suspiciously high accuracy.

**Decision:** Full design in `reports/master_dataset_design.md`. Key points:
- **Grain:** one row per `(raceId, driverId)`.
- **Required tables:** `results`, `races`, `qualifying`, `driver_standings`,
  `constructor_standings`, `drivers`, `constructors`, `circuits` (core); `sprint_results`
  optional enrichment; `lap_times`, `pit_stops`, `constructor_results`, `seasons`
  excluded (leakage or redundant).
- **Standings must be lagged to round N−1**, not joined on the current raceId.
- **All post-race `results.csv` columns** (`position`, `points`, `laps`, `milliseconds`,
  `statusId`, etc.) are usable only for computing the target and *prior*-race rolling
  history — never as a feature for the row's own race.
- `is_home_circuit` requires a hand-built nationality→country mapping table; no
  direct join key exists between `drivers.nationality` and `circuits.country`.
- `grid == 0` (pit-lane start) must be treated as a coded sentinel, not a numeric grid value.

**Consequences:**
- `tests/test_features.py` must include one explicit leakage-assertion test per
  risk listed in `reports/master_dataset_design.md` §6.
- Sprint-race features are deferred to a post-v1 enrichment pass — do not block
  the first working model on sprint data (only 27/305 races have it).
- This decision will be updated to "Accepted" once `engineer.py` and the leakage
  tests are implemented and passing.

---

## Decision 010 — Master Dataset Integration layer implemented as its own stage; standings excluded

**Date:** 2026-07-03
**Status:** Accepted

**Context:** Implementing Decision 009's design surfaced two points requiring
resolution, confirmed with the user before coding:

1. Decision 009's join list bundled the driver/constructor standings joins
   together with their round-N−1 lag. Lagging is a temporal transform, not a
   plain key join, and doesn't belong in an "integration only" phase.
2. `qualifying.csv` had no `clean_qualifying()` step or interim parquet yet
   (a separate pending Phase 3 backlog item), but the design treats qualifying
   as a core join.

**Decision:**
- **Standings are excluded entirely** from `data/processed/master_dataset.parquet`.
  `driver_standings.csv` / `constructor_standings.csv` are deferred in full
  (join + lag together) to the future feature-engineering phase, which will
  read the standings CSVs directly rather than via the master dataset. This
  narrows Decision 009 §4's join list — the master dataset join graph no
  longer includes standings tables.
- **`clean_qualifying()` was implemented first** (in `src/data/cleaner.py`,
  paired with `validate_qualifying()` in `src/data/validator.py` and
  `build_qualifying_interim()` in `src/data/build_interim.py`), producing
  `data/interim/qualifying.parquet` (11,102 rows). Only dtype casting is
  applied — `q1`/`q2`/`q3` remain raw time strings; parsing to seconds is
  feature engineering, still out of scope.
- **The integration layer is its own architectural stage**, split from
  feature engineering as originally sketched in
  `reports/master_dataset_design.md` §9: `src/integration/build_master_dataset.py`
  holds reusable join/validation logic; `src/pipelines/build_dataset.py` is a
  thin orchestration entry point. This fits the existing target-architecture
  diagram's `DATASET LAYERS` stage in `project_overview.md` and keeps future
  incremental-sync work able to call the integration logic without going
  through feature code.
- The master dataset is **not year-filtered** — it covers the full available
  race history (1950–2026, 27,279 rows) so future rolling/circuit-history
  features can use pre-2010 races as context. The Decision-008 train/val/test
  split is applied later, at training time.
- Post-race outcome columns from `results.csv` (`position`, `points`, `laps`,
  `statusId`, etc.) are retained in the master dataset for target derivation
  and future rolling-history computation, clearly separated from pre-race-safe
  columns via `POST_RACE_OUTCOME_COLUMNS` in `build_master_dataset.py`.

**Consequences:**
- `data/processed/master_dataset.parquet` built and validated: 27,279 rows,
  43 columns, one row per `(raceId, driverId)`.
- Discovered and confirmed as a genuine historical data quirk (not a bug):
  two races — 1957 British GP (raceId 780) and 1956 Argentine GP (raceId
  784) — have two drivers each with `positionOrder == 1`, from the era's
  shared-drive rule. Both are pre-2010, outside the modeling window;
  `validate_output()` surfaces this as a warning, not an error.
- Future `src/features/engineer.py` must join driver/constructor standings
  itself (with the round-N−1 lag) rather than expecting them in
  `master_dataset.parquet`.
- 38 new tests added (`tests/test_build_master_dataset.py`,
  plus `clean_qualifying`/`validate_qualifying` tests) — 124/124 passing.

---

## Decision 011 — Feature layer: modular per-group modules; functional pipeline (no fitted sklearn Pipeline in Phase 3b)

**Date:** 2026-07-03
**Status:** Accepted

**Context:** The backlog sketched Phase 3b as a single `src/features/engineer.py`
plus an sklearn `Pipeline` in `src/features/pipeline.py` "fit on 2010–2021 only".
Implementing it surfaced two design mismatches: (1) one file holding five unrelated
feature groups resists independent testing and reuse; (2) every Phase 3b transform
is stateless — rolling windows, lags, and string parsing computed deterministically
from history — so a fit/transform API would be an empty ceremony with nothing to fit.

**Decision:**
- One module per feature group, each exposing a pure `add_*_features(df) -> df`
  function and a `*_FEATURES` column-name constant:
  `src/features/qualifying.py`, `driver_form.py`, `constructor_form.py`,
  `circuit_history.py`, `standings.py`. This supersedes the planned monolithic
  `engineer.py` (which was never created).
- `src/features/pipeline.py` is a functional composition + CLI (mirroring
  `src/pipelines/build_dataset.py`): applies the groups in order, validates,
  writes `data/processed/features.parquet`. It defines `FEATURE_COLUMNS` /
  `FEATURES_DATASET_COLUMNS` and asserts **at import time** that no
  `POST_RACE_OUTCOME_COLUMNS` member (imported from the integration layer) is a
  feature.
- Fitted preprocessing (imputation, scaling — fit on the 2010–2021 training window
  only, per Decision 008) moves to Phase 4's model pipeline, where the estimator
  lives. The "fit on train only" constraint is unchanged; it just applies to the
  stage that actually has fitted state.
- Constructor-level rolling features are computed at **(constructorId, raceId)
  grain first**, then joined back — a row-level window would both mis-count
  ("last 5 rows" ≈ 2.5 races) and leak the teammate's same-race result.
- Standings lag is implemented as `prev_raceId = shift(1)` over the
  (year, round)-sorted race calendar — one rule that yields round N−1 mid-season,
  the prior season's final standings at round 1, and null for a first-ever
  appearance.
- Deferred, deliberately absent from `FEATURE_COLUMNS` (guarded by tests):
  `is_home_circuit` (design doc §6.3, needs a hand-built mapping) and sprint
  enrichment (§6.4). Raw `grid` and raw `q1/q2/q3` are also excluded — only their
  engineered forms (`grid_adjusted`, `grid_position_norm`, `pit_lane_start`,
  `*_sec`, `reached_q2/q3`, `qualifying_gap_to_pole_pct`) are features.

**Consequences:**
- `data/processed/features.parquet` built: 27,279 rows × 38 columns
  (6 ids + 31 features + `winner`), full history; Decision-008 split still applied
  at training time.
- `tests/test_features.py` (25 tests) covers every §6 leakage risk explicitly;
  149/149 tests passing.
- Phase 4 must handle informative nulls (no prior history, didn't reach Q3) —
  they are deliberately NOT imputed here; tree models' native NaN handling or a
  Phase 4 imputer are the intended consumers.
- `AI_AGENT.md` §4's "where to write code" table entry for `engineer.py` is
  superseded by the per-group modules.

---

## Decision 012 — Model development layer design (Phase 4)

**Date:** 2026-07-03
**Status:** Accepted (2026-07-03, all seven §13 points approved — point 7 with an
amendment, see the approval note at the end of this entry)

**Context:** Phase 4 (model training) needs its contract fixed before
`src/models/` is written, per the same design-before-code discipline as
Decisions 009/011. Designing against the built `features.parquet` also
surfaced that the data now extends beyond the Decision-008 test window
(2025 complete + 2026 partial: 611 labeled rows newer than the 2024 test set).

**Decision:** Full design in `reports/model_development_design.md`. Key points:
- `src/models/`: `splits.py` (temporal split + season-grouped expanding-window
  CV), `registry.py` (model zoo incl. pole-sitter heuristic baseline),
  `evaluate.py` (pure per-race metrics), `train.py` (CLI orchestration +
  MLflow logging), `predict.py` (registry load + per-race-normalized scoring —
  the Phase 5 serving contract).
- Candidates: pole-sitter heuristic (43.5% top-1 floor) → Logistic Regression
  → Random Forest → XGBoost → LightGBM; weighting (scale_pos_weight ≈ 20 /
  class_weight='balanced') for imbalance, resampling explicitly rejected.
- Metrics: per-race top-1 (primary), top-3 recall, winner MRR, log-loss,
  Brier, calibration curve. Selection on validation with a parsimony rule;
  2024 test scored exactly once behind a guarded `--final-test` flag.
- Tuning: stage-1 zoo defaults, stage-2 randomized search (~40 configs) on
  the best family over the season folds; no new tuning dependency.
- MLflow: experiment `f1-winner-prediction`, parent run per candidate with
  per-fold child runs, data fingerprint tags, FEATURE_COLUMNS JSON artifact;
  registry `f1-winner` with Staging/Production aliases.
- Explainability: native importances + permutation importance (per-race
  top-1 scored) + SHAP for the tree finalist; doubles as a leakage audit.
- Modeling-stage leakage checks (§11) and required tests (§12) enumerated —
  all executable, not prose.

**Six points awaiting approval (design doc §13):** (1) keep Decision-008
split and reserve 2025–2026 as an untouched forward holdout for Phase 8;
(2) season-grouped expanding-window CV as the operationalization of
Decision 008's TimeSeriesSplit clause; (3) refit selected config on
train+val for the registered artifact after the one-time test eval;
(4) add `splits.py`/`registry.py` to `src/models/`; (5) add `shap` dependency
and start pinning versions; (6) pole-sitter heuristic as a formal tracked
baseline.

_Addendum (same day, still Proposed): a critical feature-set review was
appended to the design doc as §14, adding a 7th approval point — freeze the
31-feature set for Phase 4 v1; ranked v2 improvements are gated on v1
importance/error analysis. §14 also corrects the baseline expectation (pole
wins ~50% in the 2010–2024 window, verified — not the EDA's 43.5%), documents
verified era-nonstationarity of form features, flags raw q1/q2/q3_sec as
~99% circuit-identity proxies (v2 cleanup candidates), and adds one
modeling-stage guard: splits must use explicit year ranges so the 2025–2026
forward holdout can never leak into training._

**Consequences:**
- Once approved, implementation proceeds in separate steps (splits →
  registry/evaluate → train → predict), each with its tests.
- This decision will be updated to "Accepted" (or amended) once the §13
  points are resolved by the user.

_Approval note (2026-07-03): the user approved all seven §13 points, with
point 7 amended as follows — the 31-feature set is frozen **only for Phase 4
model comparison and hyperparameter tuning**, not indefinitely; future
feature additions will be evaluated after baseline performance, SHAP
analysis, and error analysis are available. Implementation proceeds one
module at a time (splits.py → registry.py → evaluate.py → train.py →
predict.py), with tests, memory updates, and a user checkpoint after each
module. Feature classification recorded separately as Decision 013._

---

## Decision 013 — Feature classification: Stable / Era-sensitive / Experimental

**Date:** 2026-07-03
**Status:** Accepted

**Context:** The §14 feature-set review (reports/model_development_design.md)
measured that feature predictiveness is not uniform across regulation eras:
form-feature correlation with winning varies 0.41→0.62 across era segments,
raw qualifying seconds are ~99% circuit-identity variance, and circuit-history
features are structurally sparse. Phase 4 evaluation, Phase 8 drift
monitoring, and future v2 pruning decisions all need a shared vocabulary for
"how much should we trust this feature to keep working."

**Decision:** Every feature in the frozen Phase 4 set (Decision 012) is
assigned one of three classes. The classification is interpretive metadata —
it does NOT change the Phase 4 model input list (all 31 features are used).

**Stable (12)** — era-robust: relative/normalized measures, rank-based
values, or structural facts. Expected to survive regulation resets; a drift
alarm on these suggests a data problem, not a domain shift:
`qualifying_position`, `qualifying_gap_to_pole_pct`, `reached_q2`,
`reached_q3`, `pit_lane_start`, `grid_adjusted`, `grid_position_norm`,
`driver_experience_races`, `driver_avg_finish_last_5`,
`driver_dnf_rate_last_5`, `driver_standing_position_prev`,
`constructor_standing_position_prev`.

**Era-sensitive (12)** — predictive power depends on dominance concentration,
regulation continuity, or points-system stability. Expected to weaken at era
boundaries (2026) and under cost-cap convergence; drift here is domain
behavior, not necessarily a bug:
`driver_wins_last_3/_5/_10`, `driver_podiums_last_5`, `driver_points_last_5`,
`constructor_wins_last_3/_5`, `constructor_podiums_last_5`,
`constructor_dnf_rate_last_5` (also resets on constructor rebrands),
`driver_standing_points_prev`, `driver_standing_wins_prev`,
`constructor_standing_points_prev`.

**Experimental (7)** — weak, noisy, or proxy signal; explicit keep-or-drop
decision after v1 SHAP/error analysis:
`q1_sec`, `q2_sec`, `q3_sec` (~99% circuit-identity variance — verified;
drop candidates), `driver_circuit_starts`, `driver_circuit_wins`,
`driver_circuit_avg_finish`, `constructor_circuit_wins` (structurally sparse:
most (driver, circuit) pairs have 0–2 prior visits).

**Consequences:**
- Phase 4 evaluation artifacts should report feature importance grouped by
  class; a model whose top importances are Experimental features is suspect
  (circuit memorization risk per §14.3).
- Phase 8 drift monitoring treats the classes differently: Stable-feature
  drift → investigate data; Era-sensitive drift at era boundaries → expected.
- The classification materializes in code as a `FEATURE_CLASSIFICATION`
  constant when first consumed (expected: train.py's importance reporting) —
  no dead code before then.
  _Addendum (same day, user-directed): materialized immediately instead, as
  `src/features/metadata.py` — a reusable single source of truth
  (`FEATURE_CLASSIFICATION`, `FEATURE_GROUPS`, `STABLE_FEATURES`,
  `ERA_SENSITIVE_FEATURES`, `EXPERIMENTAL_FEATURES`, `features_in_class()`)
  with import-time integrity asserts against `FEATURE_COLUMNS`, so training,
  evaluation, SHAP analysis, ETL/drift monitoring, and future dashboard
  components all import one module rather than train.py owning it._
- Reclassifying a feature (e.g., promoting an Experimental feature after v1
  evidence) requires updating this decision's classification via a new
  decision entry, plus `context/domain_knowledge.md` §10/§11 if applicable.

---

## Decision 014 — Phase 4 model selection: tuned Logistic Regression registered as Staging

**Date:** 2026-07-03
**Status:** Accepted

**Context:** Phase 4 execution (user-approved) ran the full Decision-012
workflow: stage-1 zoo comparison (pole baseline, LogReg, RF, XGBoost,
LightGBM), stage-2 randomized search (40 configs × 6 season folds) on the two
best stage-1 families (logreg, random_forest), the one-time guarded 2024
final test, and the §10 explainability/timing analysis. Full evidence:
`reports/model_selection_report.md` + `reports/model_comparison.csv`;
all runs in MLflow (`sqlite:///mlflow.db`, fingerprint 27279rows-80744c5053e2).

**Decision:**
- **Selected model: tuned LogReg (`model__C ≈ 0.01654`)** — val top-1 68.2%
  (vs pole 54.5%, RF-default 61.4%), val top-3 88.6%, lowest CV std (0.063),
  cheapest to fit/serve, cleanest explainability. Registered as
  **`f1-winner` v1 @Staging** (fit on train 2010–2021). **NOT promoted to
  Production** — the Production refit (train+val) happens at the module-5
  checkpoint per design §4.
- **Tuned RF rejected despite best CV mean (0.566):** its val top-1 (52.3%)
  fell below the pole gate — a CV-overfit; the default RF (61.4%) was the
  honest RF entry and still lost to logreg outside the noise band.
- **Boosters eliminated at stage 1 untuned** (below pole on CV, 5–6 races
  behind on val) per design §7's tune-only-top-families budget rule. Their
  tuned ceiling is unmeasured — acceptable open question.
- **Calibration remedy validated but deferred:** the finalist's raw
  probabilities carry the expected class-weight inflation (val ECE 0.153).
  An out-of-fold isotonic calibrator (design §5's approved remedy) was
  validated empirically — ECE → 0.012, log-loss 0.268 → 0.088, top-1
  unchanged — but wiring it into the artifact means changing the reviewed
  registry.py, so it is deferred to the module-5 checkpoint. Until then the
  registered model's raw probabilities must not be read as absolute win
  chances (per-race normalization in predict.py is the serving contract).
- **`src/models/analysis.py` added** (design §10 implementation): SHAP
  (Tree/Linear explainers), permutation importance scored by per-race top-1,
  Decision-013 class summaries, zoo timing; artifacts to
  `reports/phase4_analysis/` + MLflow runs tagged stage=analysis.

**Consequences:**
- Final test (2024, scored once, tag final=true): top-1 45.8% (11/24) —
  ties the pole baseline exactly (pole also 11/24 in 2024) — top-3 75.0%,
  MRR 0.643. Both success bars met (beats/ties pole; top-3 ≥ 70%).
- **The model's top-1 edge over pole is dominance-season-concentrated**
  (2023: 90.9% vs 63.6%; 2022/2024: parity at ~46%) — confirms design §14.3
  era-nonstationarity; primary documented limitation for 2025–2026.
- Decision-013 class shares are healthy (Stable ≈ 59–61% of importance;
  Experimental ≈ 10–16%); raw q1/q2/q3_sec confirmed near-noise (v2 drop
  candidates); no leakage indicators anywhere (§11 guards, tripwire,
  canary, importance audit all clean).
- v2 feature work (design §14.5) is now unblocked by measurement; the
  boosters may be re-tuned after the v2 feature set exists.

---

## Decision 015 — OOF isotonic calibration as the production-artifact standard; predict.py serving contract

**Date:** 2026-07-04
**Status:** Accepted

**Context:** Decision 014's finalist (tuned LogReg) carries the expected
class-weight probability inflation (val ECE 0.153; monotone curve, ranking
unaffected). Design §5/§9.5 pre-approved an isotonic remedy fit on CV fold
predictions only; the module-5 checkpoint (user-approved) authorized
implementing it before the prediction layer.

**Decision:**
- **`src/models/calibration.py` (new):** `oof_predictions()` replays the
  exact season-fold protocol (fresh pipeline per fold, §11.4 containment;
  `season_folds` itself rejects out-of-window years, so the calibrator can
  only ever see training-split data); `fit_isotonic()`;
  `fit_calibrated_model(name, train_df, fit_df=None, params)` returns a
  **`CalibratedModel`** wrapper: fitted base pipeline + fitted
  IsotonicRegression, predict_proba clipped to (1e-6, 1−1e-6),
  `named_steps` delegated to the base pipeline so ColumnGuard schema
  introspection (`registry.training_schema`) works unchanged, `calibration
  = "isotonic-oof"` marker, `.fit()` raises (assembly only via
  `fit_calibrated_model` — a naive refit would re-learn the calibrator on
  non-OOF predictions). Serializes through mlflow.sklearn like any pipeline.
- **`train.py --calibrate`** (valid only with `--register`):
  `register_model(..., calibrate=True)` registers the calibrated wrapper
  and tags the run `calibration=isotonic-oof`. For a future Production
  refit (base on train+val), the calibrator STAYS train-OOF.
- **Registered: `f1-winner` v2 @Staging** = calibrated tuned LogReg
  (C≈0.01654, base fit on 2010–2021). v1 (raw) retained as history.
  Production still deliberately unset.
- **`src/models/predict.py` (module 5/5, design §2):** `load_model(alias)`
  → (model, `ModelInfo`) with name/version/alias/run_id/trained_at/
  calibration status/model class (JSON-ready); `predict_race(model,
  race_df)` → per-race SUM-NORMALIZED probabilities sorted desc with
  `predicted_rank`, identifiers carried through. The design matrix is built
  from **the artifact's own stored schema** (ColumnGuard-recorded), never
  repository constants — old artifacts keep validating against what they
  were trained on. Zero-probability races normalize to a uniform share
  (deterministic). Validates: raceId present/non-null, no duplicate
  (raceId, driverId), missing feature columns, non-numeric dtypes (via
  ColumnGuard cast). CLI: `python -m src.models.predict --race-id N`.

**Consequences:**
- Verified on the real validation split (production path, not just the
  experiment): top-1 unchanged 68.2%, ECE 0.153 → **0.012**, log-loss
  0.268 → **0.088**, Brier 0.088 → 0.026; top-3 −1 race (0.886 → 0.864,
  isotonic tie-plateaus under the pessimistic tie policy — accepted cost).
- Isotonic output is a 19-step function: several drivers per race can share
  a calibrated probability; consumers must use `predicted_rank` (already
  tie-broken deterministically), not equality comparisons.
- +37 tests (16 `tests/test_calibration.py`, 21 `tests/test_predict.py`);
  272/272 passing.
- Phase 4 is now fully complete (all 5 modules); Phase 5 (FastAPI/
  Streamlit) unblocked — `predict_race` + `ModelInfo` are the API's
  intended dependency surface.

---

## Decision 016 — Phase 5 application-layer design (FastAPI + Streamlit serving architecture)

**Date:** 2026-07-04
**Status:** Accepted (2026-07-04 — all six §16 points approved, with six user
amendments folded into the design doc: reserved POST /predict 501 stub for
Phase 8; dev-only GET /debug/features/{race_id} gated by F1_DEBUG_ENDPOINTS;
cache key explicitly (model_version, race_id); structured prediction-log
fields incl. prediction_id; three-page dashboard Overview/Predictions/Model
Insights; new `docs/` directory for user-facing documentation while
`context/` stays internal AI memory)

**Context:** Phase 4 is complete; `src/models/predict.py` is the serving
contract (`load_model(alias)` + `predict_race()`). Per the project's
design-before-code discipline (Decisions 009/012), the application layer
needs its contract fixed before `app/` is written.

**Decision:** Full design in `reports/application_design.md`. Key points:
- `app/` stays logic-free: FastAPI translates HTTP to exactly two predict.py
  calls; Streamlit consumes the API over HTTP (not `src/` imports) so the
  API contract is exercised by a real client and Phase 7 can deploy the two
  processes independently.
- **Server-side feature lookup by raceId** (API reads features.parquet;
  clients never send feature payloads — features are derived artifacts of
  the leakage-audited pipeline). This SUPERSEDES the original
  architecture.md API sketch (POST body with driver feature dicts), which
  is deferred to Phase 8 upcoming-race scoring.
- Endpoints: `GET /health`, `GET /model`, `GET /races`,
  `GET /predictions/{race_id}`; pydantic schemas carry ModelInfo (version +
  calibration status) on every prediction response.
- **Forward-holdout serving guard:** years > 2024 return 409 by default
  (config-overridable) so the dashboard can't informally burn the 2025–2026
  holdout before Phase 8.
- Model + features load once at startup (restart-to-deploy in v1; reload
  endpoint deferred to Phase 8b with auth); per-race prediction cache keyed
  by (race_id, model_version); pydantic-settings config, env-prefixed `F1_`,
  no hardcoded paths; stdout logging; no auth in v1 (localhost), API-key
  path documented.
- New dependencies on approval: fastapi, uvicorn, streamlit, plotly, httpx,
  pydantic-settings.
- Dashboard: single page — season/race selector, probability bar chart with
  actual-winner highlight and hit/miss badge, model metadata panel, and
  MANDATORY era-caveat copy (Decision 014's measured limitation surfaced in
  the UI).

**Consequences:**
- Once approved, implementation proceeds in checkpointed steps: config+API+
  tests → dashboard+smoke → docs sync.
- architecture.md's "API Design" section must be rewritten to match §5/§6 of
  the design doc when implementation lands.
- The 409 guard couples serving policy to the Decision-012 holdout policy —
  Phase 8 must flip `F1_SERVE_MAX_YEAR` deliberately, not accidentally.

---

_Add new decisions below this line using the same format._

---

## Decision 008 — Train/validation/test split: 2010–2021 / 2022–2023 / 2024

**Date:** 2026-06-08
**Status:** Accepted

**Context:** EDA in `notebooks/01_eda_raw_data.ipynb` revealed three structural reasons to restrict
training data to 2010+: (1) field size stabilised at 20 drivers; (2) finish rate jumped from ~50% to
~84%, making pre-2010 reliability patterns misleading for modern predictions; (3) the 2010 points
system change (max 10 → 25) creates a hard discontinuity in raw championship points.

**Decision:** Split strictly by year with no random shuffle:
- **Train:** 2010–2021 (237 races, 5,077 entries, 237 winners, 4.67% positive rate)
- **Validation:** 2022–2023 (44 races, 880 entries) — new ground-effect regulations provide a
  moderate intentional distribution shift to test robustness
- **Test:** 2024 (24 races, 479 entries) — fully held out; used only for final model selection

**Consequences:**
- `scale_pos_weight ≈ 20` for XGBoost/LightGBM to handle class imbalance
- Rolling features must be computed using only prior-race data (strict temporal window)
- Championship standings must be lagged by one round (use round N-1 values)
- Pre-2010 data is not used in any split; it is not deleted — may be revisited for pre-training or
  auxiliary analysis
- `TimeSeriesSplit` cross-validation within the training set; do NOT use standard k-fold

---

## Decision 017 — Core ML Platform milestone baseline and decision-status reconciliation

**Date:** 2026-07-04

**Status:** Accepted

**Context:** Phases 0–5 are implemented, but several early entries retained their
historical “Proposed” wording and some designs still described built modules as future.
The append-only decision log needs a reconciliation without rewriting its history.

**Decision:** Treat the audited implementation as the baseline for future work.

- Decision 003 is **Accepted and implemented** (binary per-driver classification).
- Decision 004 is **Accepted and implemented** with a project-root SQLite MLflow
  tracking/registry backend, replacing its initial file-store detail.
- Decision 009 is **Accepted and implemented**, as amended by Decisions 010 and 011.
- Decisions 001–002, 005–008, and 010–016 remain **Accepted**.
- No decision is currently classified Superseded or Deferred.
- Decision 008's generic `TimeSeriesSplit` wording is implemented more precisely as
  season-grouped expanding folds in `src/models/splits.py`.

The implemented baseline ends at local historical serving: reproducible batch data
builds, features, model development, MLflow registry, calibrated inference, FastAPI,
and an API-only Streamlit dashboard. Authentication, CI/CD, containers, deployment,
maintained ingestion, upcoming-race prediction, reload, scheduling, and monitoring
remain future work and require new decisions.

The next recommended milestone is the Quality Baseline: loader tests, measured ≥80%
`src/` coverage, and Git tracking. ETL design must wait for resolution of the
2025–2026 data provenance and update-semantics question.

**Consequences:** Historical entries stay immutable while their effective status is
unambiguous. Roadmap prose is not an approved implementation contract. Future agents
start from this milestone rather than re-interpreting Phase 0–5 planning documents.

---

## Decision 018 — Configurable split strategies (regulation-aware temporal splitting)

**Date:** 2026-07-04
**Status:** Accepted

**Context:** `src/models/splits.py` hardcoded a single outer split (Decision 008:
2010–2021 / 2022–2023 / 2024). That split deliberately measures cross-era
generalization over the 2022 ground-effect reset — but era-aware evaluation
(within-era training) and the Phase 8 rolling-retraining rehearsal need other
windows, and `domain_knowledge.md` §1 documents that regulation resets make
cross-era pooling a modeling choice, not a given. A refactor was requested to make
the split configurable without breaking the Decision-008 default or the
forward-holdout guard.

**Decision:**
- **`SplitStrategy` frozen dataclass** in `src/models/splits.py`: named, validated
  train/val/test year windows (`(lo, hi)` inclusive) plus `default_n_folds` and an
  `allow_forward_holdout` opt-in flag. Construction rejects inverted or overlapping
  windows and — without the opt-in — any window reaching
  `FORWARD_HOLDOUT_MIN_YEAR` (2025).
- **Presets** in the `STRATEGIES` registry:
  - `historical` — 2010–2021 / 2022–2023 / 2024 (Decision 008; the DEFAULT;
    `default_n_folds=6`).
  - `hybrid_era` — 2014–2019 / 2020–2021 / 2022 (`default_n_folds=3`).
  - `ground_effect` — 2022–2023 / 2024 / 2025 (`default_n_folds=1`;
    `allow_forward_holdout=True` baked in; raises a clear empty-split error if
    2025 rows are absent).
- **`rolling_window_strategy(test_start_year, train_seasons=5, val_seasons=1,
  test_seasons=1, allow_forward_holdout=False)`** factory for window-length-based
  strategies (the Phase 8 automated-retraining shape). Requires `train_seasons >= 2`.
- **`temporal_split(df, strategy=...)`** and **`season_folds(train_df, n_folds=None,
  strategy=...)`** accept a preset name or `SplitStrategy`; both default to
  `historical`, so all existing callers (train.py, calibration.py, analysis.py,
  tests) are unchanged. `season_folds` validates the input against the SELECTED
  strategy's training window (`n_folds=None` → the strategy's default), preserving
  the Decision-015 guarantee that the OOF calibrator can never see
  validation/test seasons. `TemporalSplit` now carries its `strategy`.
  `TRAIN_YEARS`/`VAL_YEARS`/`TEST_YEARS` remain as aliases of the historical preset.
- **Forward-holdout policy is mechanism-plus-gate:** the `ground_effect` preset and
  holdout-reaching rolling strategies EXIST, but Decision 012 §13.1 remains the
  default runtime behavior, and actually evaluating on 2025–2026 rows stays gated
  on the unresolved provenance question (`domain_knowledge.md` §8 milestone note).
  Nothing in Phase 4–6 code selects a non-historical strategy.

**Consequences:**
- Era-aware evaluation and the Phase 8 rolling-retraining design are unblocked at
  the splitting layer; wiring a `--strategy` CLI flag into `train.py` is a separate,
  deliberate future step (not done — training remains historical-only).
- All preserved invariants are tested: disjoint raceIds, empty-split detection,
  holdout rejection without opt-in, strategy-relative `season_folds` containment.
  `tests/test_splits.py` grew 14 → 28 tests; full suite 285 → 299 passing.
- Feature engineering, model pipeline, registry artifacts, and the API are
  untouched; the registered `f1-winner` v2 artifact is unaffected.
- Anyone adding a new preset must justify its windows against
  `domain_knowledge.md` §1 era boundaries and needs the explicit opt-in (plus the
  provenance resolution) to touch 2025+.

---

## Decision 019 — Regulation-era domain model; within-era preset correction

**Date:** 2026-07-04
**Status:** Accepted (amends the Decision-018 preset definitions; the Decision-018
framework — `SplitStrategy`, guards, backward-compatible defaults — is unchanged)

**Context:** A domain review of the Decision-018 presets found `hybrid_era`
(train 2014–2019 / val 2020–2021 / test 2022) mislabeled: testing on the first
ground-effect season makes it a second CROSS-era experiment, redundant with
`historical`'s objective and contradicting its within-era name. Era years were
also hand-typed per preset, so a future regulation change would touch multiple
definitions. The evaluation objective of each preset (within-era vs cross-era vs
production forecasting) existed only as prose.

**Decision:**
- **`src/models/eras.py` (new):** frozen `RegulationEra` dataclass +
  `REGULATION_ERAS` table — `v8` 2010–2013, `hybrid` 2014–2021, `ground_effect`
  2022–2025, `future_engine` 2026–ongoing (`end_year=None`) — matching
  `domain_knowledge.md` §1 and Decision 013's era segmentation. Import-time
  integrity asserts (starts at the 2010 modeling window, contiguous,
  non-overlapping, only the final era open-ended); helpers `era_of(year)` /
  `get_era(name)`. Follows the Decision-013 `src/features/metadata.py`
  precedent: one code-level source of domain truth, reusable by Phase 8 drift
  monitoring without importing split logic. Pre-2010 years map to no era by
  design (Decision 008).
- **`EvaluationObjective` str-enum** on `SplitStrategy` (MLflow-loggable):
  `cross_era_generalization`, `within_era_validation`,
  `production_forecasting`, `custom` (default for ad-hoc constructions).
- **`within_era_strategy(era, ...)` factory:** carves a CLOSED era's final
  seasons into val/test, training on the rest; rejects ongoing eras and eras
  too short to leave ≥2 training seasons; composes with the Decision-018
  forward-holdout opt-in. This is the future-proofing hook: a new era added to
  `eras.py` gets a within-era preset in one line, zero splitting-logic changes.
- **Preset corrections:**
  - `hybrid_era` REDEFINED to train 2014–2019 / val 2020 / test 2021 —
    entirely inside the hybrid regulations (derived from `eras.HYBRID`;
    `default_n_folds=3`). Documented caveat: 2020 ran a COVID-shortened
    calendar under the same ruleset.
  - `ground_effect` now DERIVED from `eras.GROUND_EFFECT` (same windows as
    before: 2022–2023 / 2024 / 2025; holdout opt-in and provenance gate
    unchanged).
  - `historical` windows stay LITERAL Decision-008 years on purpose — the
    baseline contract behind every registered artifact must never move with
    era-table edits; tagged `cross_era_generalization`.
  - `rolling_window_strategy` tagged `production_forecasting`; documented as
    deliberately era-agnostic (a real team trains on recent seasons; windows
    spanning a reset carry weakened constructor signal — a reporting caveat,
    not an error).

**Consequences:**
- The three preset questions are now explicit and machine-readable: historical
  = cross-era research, hybrid_era/ground_effect = within-era validation,
  rolling = production forecasting.
- BREAKING (contained): any consumer that relied on the pre-019 `hybrid_era`
  windows sees different val/test years. No production code selects
  non-historical strategies, so the blast radius was one test, updated.
- 2026 era handling is now a data edit: close `ground_effect` if its end year
  changes, and `future_engine` gains a within-era preset automatically once
  closed (or long enough) — no splitting-logic changes.
- Tests: `tests/test_eras.py` added (8 tests); `tests/test_splits.py` 28 → 35
  (objective tags, era containment of within-era presets, factory arithmetic
  and rejections, historical's boundary-crossing property pinned). Full suite
  299 → 314 passing. Feature engineering, training, evaluation, calibration,
  prediction, API, and dashboard untouched.
- Era boundaries are public years in advance — era metadata is not leakage
  (`domain_knowledge.md` §1); the §11 exclusion of regulation-era *model
  features* still stands (eras inform SPLITS here, not the feature matrix).

---

## Decision 020 — Packaging on pyproject.toml; single-source version; repo hygiene baseline

**Date:** 2026-07-05
**Status:** Accepted

**Context:** The Phase-A engineering audit found: `setup.py` declared
`python_requires>=3.9` while the pinned stack (numpy 2.x) requires ≥3.11 and the
dev environment runs 3.14; three disagreeing version sources (`setup.py` 0.1.0,
git tag v1.0.0, hardcoded `API_VERSION` 1.0.0); dev tools (pytest, notebook)
shipped as runtime dependencies; no LICENSE on a public GitHub repo; no
line-ending normalization (Windows dev, future Linux CI); and gitignore gaps.

**Decision:**
- **Packaging is PEP 621 `pyproject.toml`; `setup.py` is deleted.** Distribution
  name stays `f1-race-winner-prediction`; only `src/` is packaged (parity with
  the old setup.py — `app/` remains a serving-entry directory run from the
  project root, not a library). `requires-python = ">=3.11"`.
- **Version 1.1.0, defined once** in pyproject.toml. `app/api.py` reads it via
  `importlib.metadata` (fallback `0.0.0+uninstalled`). Release tags must match
  the pyproject version going forward (v1.0.0 tag = the pre-migration release).
- **Dependency split:** runtime deps (pipeline, training/analysis incl.
  shap/matplotlib, serving, dashboard) in `[project.dependencies]` with the
  Decision-012 `~=` pinning policy; `[project.optional-dependencies].dev` holds
  pytest, pytest-cov, ruff, notebook (Phase B/C consumers — declared, not yet
  wired into CI). `requirements.txt` becomes a `-e .[dev]` shim so every
  documented install command keeps working.
- **Repo hygiene:** MIT LICENSE (owner-approved); `.gitattributes` normalizes
  all text to LF with binary exemptions (png/parquet/zip/db/pkl); `.gitignore`
  adds tooling caches and AI working dirs. **Memory-versioning policy:**
  `context/` (decisions, architecture, domain knowledge, overview) is tracked —
  durable knowledge; `.ai/` (status, handoff, backlog, agent manual) is
  machine-local operational memory and stays untracked.

**Consequences:**
- Docker base images and CI matrices can read a truthful Python floor; one
  version string governs releases, the API, and the package.
- `pip install -e .` gives a runtime-only environment (future serving images);
  `pip install -r requirements.txt` gives the full dev environment.
- Contributors on any OS produce byte-identical text files.
- `.ai/` files do not travel with the repo — a fresh clone bootstraps agent
  context from `context/` + README; if collaboration later needs shared
  operational state, that reversal needs a new decision.
