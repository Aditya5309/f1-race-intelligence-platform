# Formula 1 Domain Knowledge — Engineering Reference

_Status: LIVING DOCUMENT — first-class project document alongside
`project_overview.md`, `architecture.md`, `decisions.md`, `current_status.md`,
`AI_AGENT.md`. Read it before making any data, feature, model, or evaluation
decision._

_Purpose: this document does NOT explain Formula 1. It documents the F1
concepts that materially change engineering decisions in this project — and
what those decisions should be. General motorsport knowledge that does not
change an engineering decision is deliberately omitted._

_Compiled: 2026-07-03, by an AI agent with training knowledge through early
2026. Statements about the in-progress 2026 season are flagged; verify against
current sources before relying on them. Facts marked "(verified in data)" were
checked against this project's actual datasets._

**Confidence scale used throughout:**
- **[High]** — well-documented regulation/fact, or verified directly in this project's data.
- **[Medium]** — widely accepted but with debated magnitude, team-dependent variation, or my knowledge may lag current events.
- **[Low]** — plausible, weakly evidenced, or genuinely contested; treat as hypothesis, not fact.

---

## How to use this document

| You are about to… | Read sections |
|---|---|
| Ingest/refresh data (ETL) | 1, 8, and the validation rules in every section |
| Add or change a feature | 2–7, 10, 11 (check it isn't intentionally excluded) |
| Design/train a model | 1, 6, 7, 10 |
| Evaluate a model | 1, 4, 7, 10 |
| Build prediction/serving logic | 4, 7, 9 |
| Plan future data sources | 9, 11 |

Maintenance protocol is at the end of the document.

---

## 1. Regulation Eras

### Background
The FIA periodically rewrites the technical regulations (chassis,
aerodynamics, power unit) and sporting regulations (points, formats, budget).
Major rewrites reset the competitive order: a dominant constructor can become
midfield overnight, because the *car* — not the driver — is the primary
performance differentiator (see §3). This project's entire split strategy
(Decision 008) is built on one such boundary (2010) and validated across
another (2022).

### Confidence Level
High for the existence, dates, and direction of the resets; Medium for their
magnitudes; Medium for 2026-season specifics (announced pre-2026, early-season
reality not fully known to the author).

### Established Domain Knowledge
- **2009–2010 modern-era boundary** [High]: 2009 aero overhaul, 2010
  refueling ban and points change (win: 10 → 25), field stabilization at ~20
  cars, finish rate jump to ~84% (verified in data — EDA fig05). This is why
  the modeling window starts at 2010 (Decision 008).
- **2011** [High]: Pirelli becomes sole tire supplier with a deliberately
  high-degradation philosophy; DRS (drag reduction overtaking aid) introduced.
  Increases overtaking → slightly weakens grid→finish coupling vs 2010.
- **2014 Hybrid Era** [High]: 1.6L V6 turbo-hybrid power units. Mercedes won
  8 consecutive constructors' titles (2014–2021) — the most extreme
  dominance streak in the dataset (constructor HHI 0.594 in the hybrid era,
  verified in EDA). Power-unit manufacturer became a major performance axis.
- **2017** [High]: wider cars/tires, faster cornering; moderate reset.
- **2021 Cost Cap** [High]: budget cap introduced (~$145M, stepping down),
  plus success-based sliding scale of permitted aerodynamic testing. Both are
  explicitly *convergence mechanisms*: over multiple seasons they compress
  the top-to-bottom performance spread. Expect era-over-era narrowing of
  dominance — historical dominance patterns overstate future dominance
  [Medium for magnitude].
- **2022 Ground Effect Era** [High]: venturi-floor aerodynamics, 18" wheels.
  Reset the order (Mercedes fell from dominance; Red Bull dominant
  2022–2023, historically dominant 2023). Decision 008 deliberately uses
  2022–2023 as validation to test robustness across this reset.
- **2024** [High]: within-era convergence — McLaren rose to win the
  constructors' title; multiple teams won races. Evidence that late-era
  seasons are more competitive than early-era seasons.
- **2026 Regulations** [Medium — announced pre-2026; verify details]: the
  largest combined chassis + power-unit reset since 2014: new PU with ~50%
  electric power share and no MGU-H, 100% sustainable fuel, smaller/lighter
  cars, active aerodynamics (straight-line low-drag mode). New entrants:
  Audi (absorbing Sauber) and Cadillac as an 11th team — **the field grew
  from 20 to 22 cars (verified in data: 2026 races have 22 entries and 11
  constructors)**.
- **Technical directives (TDs)** [High that they exist and can shift
  competitiveness mid-season; Medium for any specific case]: mid-season FIA
  clarifications (e.g., the 2022 anti-porpoising directive, flexi-wing/floor
  clampdowns) can change the order *within* a season without any visible
  event in this project's data. They are unobservable in the Ergast schema.

### Reasonable Modeling Assumptions
- Within a regulation era, constructor form is strongly autocorrelated;
  across an era boundary, constructor form carries over weakly and driver
  skill carries over well [Medium — consistent with 2009, 2014, 2022 resets].
- Rolling windows of 3–10 races are short enough that they self-correct
  within ~half a season after a reset; no explicit era handling is required
  for v1 [Medium — this is the current implicit assumption; test it].
- The 2010+ window is internally consistent enough to pool for training
  [Medium — Decision 008's premise; the 2014 and 2022 resets inside the
  training window are absorbed by rolling features].

### Future Research Opportunities
- Quantify how much validation error concentrates in the first ~6 races
  after a reset (2022 rounds 1–6 are in the val split — measurable today).
- Era-aware sample weighting (downweight cross-era history at era starts).

### Measured era effect on model skill (verified in modeling, 2026-07-03 — Decision 014)
The Phase 4 selected model's (tuned LogReg) top-1 edge over the pole-sitter
heuristic is **entirely dominance-season-concentrated** [High — measured]:
2023 (peak Red Bull dominance): model 90.9% vs pole 63.6%; 2022 (reset year)
and 2024 (convergence year): exact parity at ~46% top-1 for both. In
competitive seasons the model still adds top-3 recall (75–82% vs pole's
~46%) and probability quality, but no top-1 advantage. Expect the same
pattern — or worse — on the 2025–2026 forward holdout (convergence + 2026
reset); this is domain behavior, not pipeline failure.

### Era boundaries materialized in code (2026-07-04, Decision 019)
The era segmentation of this section is now a code-level table:
`src/models/eras.py` (`REGULATION_ERAS`: v8 2010–2013, hybrid 2014–2021,
ground_effect 2022–2025, future_engine 2026–ongoing), consumed by the split
strategies in `src/models/splits.py` (within-era vs cross-era vs production-
forecasting objectives). Keep this section and that table in sync — the
Maintenance Protocol applies to both. NOTE: this materializes eras for SPLIT
definitions only; regulation-era *model features* remain excluded (§11).

### Candidate Features (none implemented — see §11)
- `regulation_era` categorical (2010–2013, 2014–2021, 2022–2025, 2026–) [High value for era-aware analysis; Medium for direct model gain].
- `races_since_regulation_reset` (constructor-level counter) [Medium].
- Era-interaction: zero out / downweight constructor rolling stats that span
  a reset boundary [Low — added complexity, unproven gain].

### Validation Rules
- **Never hardcode a field size of 20** — it is 22 from 2026 (verified in
  data) and varied historically. `grid_position_norm` already normalizes by
  per-race field size; keep any new feature per-race-normalized the same way.
- On data refresh: assert the set of constructorIds per season changes only
  at season boundaries; a new constructorId mid-season is a data error or a
  genuine mid-season rebrand — investigate, don't assume.
- Any evaluation report should break metrics out by season, not only pooled —
  a model that wins on 2022–2023 pooled may be carried entirely by 2023.

### Leakage Considerations
- Era boundaries are known in advance (regulations are published years
  ahead) — era features are NOT leakage [High].
- The **forward holdout (2025–2026, Decision 012 §13.1) straddles the 2026
  reset**: any Phase 8 retraining rehearsal on it is simultaneously a
  regulation-reset stress test. Do not interpret 2026 performance decay as
  pipeline failure — it is expected domain behavior.

### Recommendations
- Keep v1 era-free (rolling windows self-correct), but make "metrics by
  season" a standard evaluation artifact so era effects are visible.
- When Phase 8 reaches 2026 data: expect degraded accuracy for early-2026
  races; consider `races_since_regulation_reset` as the first era feature.

---

## 2. Driver Performance

### Background
F1 outcomes are dominated by the car, with the driver as a significant but
secondary factor. Academic decompositions attribute roughly 80–90% of results
variance to team/car and 10–20% to the driver [Medium — estimates vary by
methodology and era; treat the split as directional, not precise]. This
inverts the naive intuition that "the best driver wins" and is the single
most important calibration for feature design: driver-identity effects are
mostly *car effects in disguise*.

### Confidence Level
High for the car-dominance direction; Medium for magnitudes and most
driver-trait claims; Low for anything about individual driver psychology.

### Established Domain Knowledge
- **Driver rolling form conflates driver and car** [High]: a driver's recent
  wins measure "driver + current car". This is acceptable — the confound is
  itself predictive — but it means driver features do NOT transfer cleanly
  across team moves.
- **Team moves cause step changes in driver results** [High]: a driver
  switching from a midfield to a top car (or vice versa) breaks their rolling
  form. The features intentionally span team changes (design doc §6.6:
  driver features follow the driver), so post-transfer predictions lag
  reality for a few races [High that this lag exists].
- **Rookie wins are effectively nonexistent in the modeling window** [High]:
  no driver has won in their debut season in 2010+; debut-race wins are
  unheard of in the modern era. Rookies' NaN feature rows (no history) are
  therefore *informative* — NaN ≈ very low win probability. Tree models can
  learn this natively; do not impute rookie NaNs to population means, which
  would make rookies look mid-field [High].
- **Wet-weather skill differentials are real** [Medium — widely accepted,
  hard to quantify]: some drivers systematically outperform in rain. This
  project has NO weather data (§8, §11), so the signal is unreachable in v1.
- **Qualifying and race pace are distinct skills** [Medium]: some drivers
  over-qualify relative to race results and vice versa (tire management,
  racecraft, starts). The gap between `qualifying_position` and prior
  finishing positions carries this residual.
- **Experience and age** [Medium]: performance follows an experience curve
  (rapid early gains) and an age curve (peak somewhere in the late 20s to
  early 30s — magnitude debated, superstars defy it). `driver_experience_races`
  exists; `driver_age_at_race` is designed but not implemented (§11).
- **Mid-season driver swaps happen** [High]: promotions/demotions (notably
  within the Red Bull program) mid-season. Pipelines must not assume a
  driver↔constructor mapping is constant within a season.

### Reasonable Modeling Assumptions
- Driver form windows of 3/5/10 races capture short/medium-term driver+car
  form adequately for v1 [Medium].
- Driver identity itself (driverId) must never be a raw feature — win rates
  are extremely right-skewed (EDA) and an ID feature would memorize eras, not
  generalize [High — already the project rule].

### Future Research Opportunities
- **Teammate-delta features**: performance relative to the same-car teammate
  is the cleanest car-controlled driver-skill measure available in this data
  (e.g., rolling qualifying-gap-to-teammate, race-finish-vs-teammate). Must
  use prior races only. This is likely the highest-value unimplemented
  driver feature [Medium].
- Positions-gained-from-grid (grid − finish) rolling stat as a racecraft
  proxy — uses post-race data of *prior* races only, so leakage-safe under
  the standard shift(1) discipline [Medium].
- Wet-race form split, once weather data exists (§9).

### Candidate Features
- `driver_quali_gap_to_teammate_last_5` (car-controlled skill) [Medium].
- `driver_positions_gained_last_5` (racecraft) [Medium].
- `driver_age_at_race` (dob exists in drivers.csv) [Medium, cheap].
- `driver_races_with_current_constructor` (integration/adaptation proxy;
  resets on team change, complements form that spans the change) [Medium].

### Validation Rules
- A driverId appearing for two constructors in the same race is a data error;
  in the same *season* is normal (mid-season swap) — validators must allow it.
- New driverIds should appear only at (or near) season starts or at known
  swap points; a never-before-seen driverId in a data refresh is expected
  (rookies) — but a new driverId with *prior-season standings rows* is a key
  mismatch bug.

### Leakage Considerations
- Teammate-delta features must exclude the current race's teammate result
  (same-race outcome = leakage), exactly like constructor form
  (Decision 011's race-grain aggregation pattern applies).
- "Driver of the day", post-race penalties, and stewards' decisions are
  post-race information — never features.

### Recommendations
- Prioritize teammate-delta features in v2 — best available de-confounding of
  driver vs car in this schema.
- Keep NaN-as-signal for rookies; document in Phase 4 that LogReg's imputer
  must pair with missing-indicator flags so "rookie" stays visible (already
  in the Phase 4 design, §3).

---

## 3. Constructor Performance

### Background
The constructor (team + chassis + power unit) is the dominant performance
factor (§2). Constructor competitiveness evolves on three timescales: within
a season (development race), across seasons within an era (convergence under
cost cap), and discontinuously at era boundaries (§1). This differs
fundamentally from driver skill, which is smooth and slow-moving.

### Confidence Level
High for the mechanisms; Medium for rates of change.

### Established Domain Knowledge
- **In-season development moves the order** [High]: teams bring upgrade
  packages continuously; a car that starts the season P4-competitive can end
  it P1-competitive (e.g., McLaren 2023–2024 trajectory). Short rolling
  windows (3/5 races) capture this; season-long averages would not — this is
  why constructor windows are shorter than driver windows (3/5 vs 3/5/10).
- **Reliability is a constructor property** [High]: mechanical DNFs cluster
  by team and by era; `constructor_dnf_rate_last_5` is the implemented proxy.
  Modern finish rates ~84% (verified in EDA) mean DNF features are sparse
  signals in-window.
- **Power-unit supplier changes cause step changes** [High]: switching
  engine suppliers between seasons (works vs customer status matters) can
  shift a team's baseline sharply. Not directly observable in this schema —
  constructorId does not encode the PU supplier [High that the schema lacks
  it].
- **Constructor identity fragments across rebrands** [High — verified in
  data]: Ergast assigns NEW constructorIds on renames. Known lineage chains
  include Force India→Racing Point→Aston Martin, Renault→Lotus→Renault→
  Alpine, Toro Rosso→AlphaTauri→RB, Sauber→Alfa Romeo→Sauber→**Audi (2026,
  verified in data)**. Consequence: constructor rolling stats and circuit
  history **silently reset to NaN at every rebrand** even though the
  organization (factory, staff, car lineage) is continuous. The current
  features treat Aston Martin 2021 like a brand-new team.
- **Two cars per race** [High]: every constructor-level statistic must be
  computed at (constructorId, raceId) grain first — already load-bearing in
  the implementation (Decision 011) — both for correct window semantics and
  to avoid teammate same-race leakage.
- **Customer/junior team correlations exist** [Medium]: teams sharing a PU
  (or ownership, e.g., Red Bull/Racing Bulls) have correlated performance;
  not modeled and not encoded in the schema.

### Reasonable Modeling Assumptions
- Constructor form over the last 3–5 races is the best available proxy for
  current car performance [High — pole/grid features aside].
- Rebrand-induced history resets are rare enough (roughly one team per 2–3
  seasons) that v1 tolerates them [Medium — acceptable, but a known bias
  against recently rebranded teams].

### Future Research Opportunities
- **Hand-built constructor lineage table** (`constructorId → lineageId`)
  so rolling stats and circuit history survive rebrands. Same maintenance
  profile as the nationality mapping (design doc §6.3): small, manual,
  best-effort. Likely moderate value, concentrated on a few teams [Medium].
- PU-supplier table (constructorId, season → engine manufacturer) for
  supplier-change features [Medium; requires external data].

### Candidate Features
- `constructor_races_since_lineage_start` (with lineage table) [Medium].
- `constructor_wins_last_10` under lineage continuity [Medium].
- `pu_supplier_changed_this_season` [Medium; external data needed].

### Validation Rules
- On data refresh: a constructorId's rows must be temporally contiguous
  (a team that vanishes for 3 seasons and returns is a rename/lineage event
  or an error — flag it).
- Constructor standings rows must exist for every (raceId, constructorId)
  that has results rows in the modern era; gaps indicate a join-key problem.

### Leakage Considerations
- Upgrade announcements are pre-race public information and would be
  legitimate features — but they are not in any current data source (§9).
- Never aggregate constructor stats at row level: the teammate same-race
  exclusion (Decision 011) is the canonical guard; every future
  constructor-level feature must reuse the race-grain-first pattern.

### Recommendations
- Add the lineage mapping table before extending constructor windows beyond
  5 races — long windows amplify the rebrand-reset bias.
- Treat `constructor_dnf_rate_last_5` as weak-signal; do not be surprised if
  its importance is near zero in-window.

---

## 4. Race Weekend Structure

### Background
An F1 weekend is a time-ordered sequence of sessions. The prediction point
for this project is fixed: **after the grid is set, before lights out**
(project_overview.md). Every piece of weekend information is therefore
classifiable as before/after that instant — the classification below is the
canonical reference.

### Confidence Level
High for session structure and the information timeline; Medium for
sprint-format details (they changed nearly every year).

### Information timeline (canonical)

**Known BEFORE race start (legitimate features):**
- Grid positions, incl. penalties applied — Ergast's `grid` column is the
  actual race-start grid [High].
- Qualifying session results and times (q1/q2/q3) [High].
- Sprint race results *for the same raceId* — sprints run Saturday, before
  Sunday's Grand Prix, so they are technically pre-race information (design
  doc §6.4) [High]. Deferred, not leakage.
- Practice session times [High] — not in this schema (§9).
- Weather *forecast* [High] — not in this schema (§9).
- Grid penalties and pit-lane-start decisions, parc fermé breaches [High] —
  visible only via their effects: `grid` vs `qualifying_position` divergence
  and the `grid == 0` sentinel.
- Championship standings as of the previous round [High].

**Generated DURING the race (never features for that race; usable in rolling
history for later races):**
- Finishing positions, points, laps, race time, fastest lap, status/DNF
  [High] — this is exactly `POST_RACE_OUTCOME_COLUMNS`.
- Safety cars, virtual safety cars, red flags, in-race weather changes, pit
  strategy, lap times [High] — excluded from the master dataset entirely
  (design doc §6.1); they also make race outcomes irreducibly noisy: a large
  fraction of upsets trace to SC timing and first-lap incidents [Medium for
  the fraction; High that it caps achievable accuracy].

**Never usable in any form:**
- Post-race stewards' reclassifications for the *current* race,
  disqualifications, post-race penalties [High]. Note: Ergast results encode
  the FINAL classification (post-penalty), which is correct for computing
  the target and prior-race history, but means the recorded winner can
  differ from the on-road winner — no action needed, just awareness.

### Established Domain Knowledge
- **Knockout qualifying (Q1/Q2/Q3) since 2006** [High]: q2/q3 nulls encode
  elimination stage — informative missingness (54% q3 nulls verified in
  data; handled via `reached_q2`/`reached_q3`).
- **`grid == 0` means pit-lane start** [High — Ergast convention; 1,635 rows
  verified in data; handled via `pit_lane_start`/`grid_adjusted`].
- **`qualifying_position` ≠ `grid` implies penalties or parc-fermé breaches**
  [High]: the signed delta is an unimplemented but legitimate pre-race
  feature (a big engine-penalty drop means a fast car starting deep in the
  field — historically a strong recovery-drive setup) [Medium for value].
- **Sprint weekends (2021–) changed format repeatedly** [High that they
  changed; Medium on per-year details]: which session sets the Sunday grid
  and how sprint points are awarded differ by season (2021–22: sprint set
  the grid; 2023+: standalone shootout, Friday quali sets Sunday grid). Any
  future sprint feature MUST be implemented per-season-format-aware — this
  complexity is why sprints were deferred (design doc §6.4).
- **Pole-sitter wins ~50% of races inside the 2010–2024 modeling window**
  [High — verified in data, per era segment: 48.1% (2010–2013), 52.5%
  (2014–2021), 51.5% (2022–2024); design doc §14.1]. The earlier EDA figure
  of 43.5% used a wider denominator — corrected here 2026-07-03 per this
  document's maintenance protocol. Top-3 grid ≈ 79% of wins [High — EDA].
  Grid position is the strongest single pre-race predictor; every model must
  beat the pole-sitter heuristic, computed on the same split at runtime
  (Phase 4 design §3/§9.1), to demonstrate any learning.

### Reasonable Modeling Assumptions
- The grid column fully absorbs pre-race penalty information for v1 [Medium
  — penalty *reasons* (fast car vs slow car penalized) are lost].
- In-race randomness (SC, first-lap chaos) is irreducible noise for a
  pre-race model; top-1 accuracy has a domain ceiling well below 100% —
  treat ~60–65% as a realistic excellent score, and >70% as a leakage alarm
  (Phase 4 design §11.6) [Medium].

### Future Research Opportunities
- `grid_vs_quali_delta` feature (penalty recovery setups) [Medium].
- Practice-pace features via FastF1 (§9) — long-run pace from FP2 is the
  classic missing predictor [Medium].

### Candidate Features
- `grid_minus_qualifying_position` [Medium, cheap, leakage-safe].
- `front_row_start` boolean [Low — trees find this threshold themselves].

### Validation Rules
- `grid` must be in [0, field_size]; `qualifying_position` in [1, ~24].
- Exactly one `positionOrder == 1` per race in the modeling window (shared
  drives are pre-1958 only — §8).
- On refresh: every 2021+ season should contain some sprint-weekend races;
  none before 2021.

### Leakage Considerations
This section IS the leakage reference for weekend data; see the information
timeline above. The single subtlest point: **same-raceId sprint results are
pre-race, not leakage** — future agents repeatedly get this wrong in both
directions (excluding them as "same race = leakage" is wrong; including
Sunday results is catastrophically wrong).

### Recommendations
- Implement `grid_minus_qualifying_position` in v2 — cheapest untapped
  pre-race signal in the current schema.
- Keep the >70% top-1 tripwire wired into every future evaluation.

---

## 5. Circuit Characteristics

### Background
Circuits differ in how much they reward car characteristics vs driver skill
and in how hard overtaking is — which modulates how deterministic the
grid→finish mapping is. Circuit knowledge mostly enters this project through
(a) circuit-history features and (b) the understanding that grid position's
predictive power is circuit-dependent.

### Confidence Level
High for overtaking-difficulty differences; Medium for most specific-circuit
claims; Low for home-advantage effects.

### Established Domain Knowledge
- **Overtaking difficulty varies enormously by circuit** [High]: street and
  narrow circuits (Monaco archetypally, Singapore, Hungary among permanent
  tracks) convert qualifying position to finishing position far more
  deterministically than high-overtaking circuits (Spa, Monza, Bahrain).
  Consequence: grid-based features are *more* predictive at some circuits —
  a circuit-level "pole conversion rate" (computed from prior seasons only)
  is a legitimate interaction feature [Medium for gain].
- **Circuit-specific driver/constructor affinity exists but is sparse**
  [Medium]: repeat winners at the same circuit are common, but most
  (driver, circuit) pairs have 0–2 prior visits (design doc §5.6). The
  implemented features (starts/wins/avg finish at circuit) are NaN-heavy by
  design; do not backfill.
- **Layouts change under a constant circuitId** [Medium]: circuits get
  reprofiled (e.g., Melbourne 2022, Abu Dhabi 2021, Barcelona chicane
  removal). Circuit-history features silently span layout changes. Impact is
  believed small; documented, not corrected.
- **New circuits enter the calendar regularly** [High]: several additions in
  2020–2026. All circuit-history features are NaN there for everyone —
  the model must degrade gracefully to form+grid features (it does, via
  tree-native NaN).
- **Track evolution within a weekend** (grip improves session to session)
  [High] — in-weekend phenomenon, irrelevant at this project's grain.
- **Home-race advantage** [Low]: popularly asserted, weakly evidenced in F1
  (unlike e.g. football); the effect, if any, is small and confounded.
  Combined with the join problem (nationality demonyms vs country names,
  design doc §6.3), this is why `is_home_circuit` is deferred and LOW
  priority.

### Reasonable Modeling Assumptions
- Circuit history with prior-visits-only computation is unbiased, just
  sparse [High].
- Treating all circuits identically (no type feature) costs little in v1
  because grid features implicitly carry most circuit-difficulty signal
  through qualifying [Medium].

### Future Research Opportunities
- `circuit_pole_win_rate_prior` — rolling, prior-seasons-only pole
  conversion per circuit; interacts naturally with `grid_adjusted` [Medium,
  probably the best unimplemented circuit feature].
- Hand-built `circuit_type` categorical (street/permanent/hybrid) — small
  static table, low maintenance [Medium].
- Layout-change registry (circuitId, year of reprofile) [Low].

### Candidate Features
As above; plus `driver_circuit_podiums` (already derivable) [Low — highly
correlated with existing circuit wins/avg-finish].

### Validation Rules
- circuitId must resolve against circuits.csv (already enforced by
  integration referential-integrity checks).
- A circuitId appearing in a season after ≥10 years of absence is suspicious
  but legitimate (returning venues); warn, don't fail.

### Leakage Considerations
- Circuit-level aggregate features (pole conversion rate etc.) must be
  computed from PRIOR races only, exactly like driver circuit history — a
  circuit's "career" statistics computed over the full dataset would leak
  future information into early rows [High — same shift-discipline as all
  rolling features].

### Recommendations
- If adding one circuit feature in v2, make it `circuit_pole_win_rate_prior`.
- Keep `is_home_circuit` deferred until someone demonstrates the effect
  exists in this data at all.

---

## 6. Championship Context

### Background
Championship standings summarize season-to-date performance and are the only
implemented feature with an *explicit* temporal lag rule. Late-season
championship dynamics (pressure, team orders, clinched titles) can decouple
finishing order from raw pace.

### Confidence Level
High for the standings-lag mechanics (verified in data/schema); Medium-Low
for behavioral effects (pressure, orders).

### Established Domain Knowledge
- **Standings rows are keyed by the race AFTER which they apply** [High —
  Ergast schema fact, verified]: `driver_standings.csv` row for raceId X is
  the standing *including* race X's result. Joining it to race X's feature
  row bakes the race's own outcome into its features — the highest-severity
  leakage risk in the project (design doc §6.2). The implemented rule:
  join at `prev_raceId` via the (year, round)-sorted calendar shift, which
  yields round N−1 mid-season, the prior season's final standings at round 1,
  and NaN on debut (Decision 011).
- **Points systems change; positions don't** [High]: 2010 points overhaul
  (10→25 per win), fastest-lap bonus point added 2019 and removed for 2025
  [Medium on the removal — verify], sprint points from 2021. This is why
  standing POSITION is the primary feature and raw points secondary
  (design doc §5.7); never compare raw points across seasons.
- **Team orders are legal (since 2011) and used** [High]: within-team
  position swaps happen for championship reasons, mildly distorting
  finishing order between teammates. Effect on WINNER prediction is small —
  orders rarely decide the win itself [Medium].
- **Season progression changes standings informativeness** [High]: standings
  after round 2 are nearly noise; after round 15 they are strong summaries.
  `round` is available; the interaction is not modeled in v1.
- **Clinched championships / dead-rubber races** [Medium]: after a title is
  decided, incentives change (experimentation, rookie FP outings) — a minor
  late-season noise source; not modeled.

### Reasonable Modeling Assumptions
- Lagged standings position is a smooth, low-noise summary of season-long
  form that complements short rolling windows [High].
- Behavioral late-season effects are noise at this project's grain [Medium].

### Future Research Opportunities
- `standings_gap_to_leader_prev` (points behind leader, era-normalized by
  points-per-win) [Medium].
- `round / season_length` progress fraction as an interaction input [Low].

### Candidate Features
As above; plus `championship_leader_prev` boolean [Low — subsumed by
position].

### Validation Rules
- Every raceId in standings CSVs must exist in races.csv (referential).
- For 2010+: standings after round N must contain every driver who scored
  points by round N (completeness check on refresh).
- The prev-race calendar map must reject duplicate (year, round) slots —
  already implemented and tested (`build_prev_race_map`).

### Leakage Considerations
**The standings lag is the canonical example of F1-specific leakage** — a
plain "join standings on raceId" is syntactically natural and semantically
catastrophic (a race winner's row would show a standing that already includes
the win). Any future standings-derived feature must go through
`build_prev_race_map`, never a direct raceId join. This includes seemingly
innocent aggregates like "points scored this season so far."

### Recommendations
- Reuse `src/features/standings.py`'s prev-race mechanism for ALL future
  season-to-date features; never write a second lag implementation.

---

## 7. Temporal Considerations (canonical leakage reference)

### Background
This project's #1 correctness constraint (project_overview.md). The domain
structure of F1 creates leakage vectors beyond generic time-series ML — this
section consolidates all of them.

### Confidence Level
High throughout — these are schema facts and implemented, tested rules.

### The rules (all implemented; cite this list before adding any feature)

1. **Chronological order is (year, round), never raceId** — raceId does not
   sort chronologically across eras [High, verified]. Every sort in
   `src/features/` uses (year, round).
2. **shift(1) before any rolling window** — a race's own result must never
   be in its own window (design doc §6.2.1). Windows span season boundaries
   naturally under rule 1.
3. **Standings must be lagged via the prev-race calendar map** (§6 above) —
   with the round-1 → prior-season-final rule and NaN on debut.
4. **`POST_RACE_OUTCOME_COLUMNS` (src/integration/build_master_dataset.py)
   is the single source of truth** for same-race-outcome columns. Import it;
   never re-derive the list. Enforced at import time in
   `src/features/pipeline.py` and re-checked in Phase 4's registry/predict.
5. **Constructor-level anything: aggregate to (constructorId, raceId) grain
   first** — row-level windows leak the teammate's same-race result and
   mis-count window lengths (Decision 011).
6. **Within-weekend ordering matters**: qualifying and sprint precede the
   Grand Prix — same-raceId qualifying/sprint data is pre-race (legitimate);
   same-raceId race data is not (§4 timeline).
7. **Preprocessing state is temporal too**: imputers/scalers fit on data
   that includes future races leak distributional information. All fitted
   state fits on the training window only (Decision 011/012).
8. **Evaluation is grouped by race**: row-level accuracy is meaningless at
   4.7% positive rate; per-race top-1/top-3 with strictly temporal splits
   (Decision 008), season-grouped CV folds (Phase 4 design §4).
9. **The test set is read once** (Phase 4 design §11.3); 2025–2026 is a
   forward holdout that no Phase 4 code may touch (Decision 012 §13.1,
   pending approval).

### F1-specific leakage vectors (why generic time-series discipline is not enough)
- Post-race-keyed standings tables (rule 3) — unique to this schema family.
- Two rows per constructor per race (rule 5).
- Post-race penalty reclassification baked into results (§4) — fine for
  targets/history, but means "winner" is the *classified* winner.
- Shared drives creating two winner rows pre-1958 (§8) — breaks the
  "exactly one winner" invariant outside the modeling window.
- Sprint weekends: same-raceId data that is legitimately pre-race (rule 6) —
  the one place where "same raceId ⇒ leakage" is WRONG.

### Validation Rules
- `tests/test_features.py` maps one test to every design-doc §6 risk — keep
  that mapping alive; any new feature adds a leakage test in the same PR.
- Phase 4 adds the shuffled-target canary and the >70% top-1 tripwire
  (Phase 4 design §11.5–11.6).

### Recommendations
- When in doubt about a new feature: state exactly what instant the
  information becomes public in the real world, and compare it to "grid is
  set". If it is not provably earlier, it is not a feature.

---

## 8. Data Limitations

### Background
The project runs on a static Ergast-schema CSV dump (Decision 005). Ergast
itself was deprecated at the end of 2024; this repo's data extends through
mid-2026, so the rows past 2024 necessarily come from a community
continuation of the schema [High that Ergast deprecated; the exact
provenance of this repo's 2025–2026 rows is UNVERIFIED — see Validation
Rules].

### Confidence Level
High — most items below are verified directly in this project's data.

### Established Domain Knowledge (all verified in data unless noted)
- **`\N` encodes NULL** throughout the CSVs (MySQL dump convention);
  handled centrally in `loader.py` [High].
- **Informative missingness is the norm, not the exception** [High]:
  q3 null ⇒ eliminated earlier (54% of 2010+ rows); rolling-feature NaN ⇒
  no prior history (rookies, new circuits, rebranded constructors);
  standings NaN ⇒ debut. Mean-imputing any of these destroys signal —
  the standing rule is: NaN is data.
- **Shared drives (pre-1958)** [High]: two drivers credited with the same
  car/position — raceIds 780 and 784 have two `positionOrder == 1` rows
  each. Outside the modeling window; surfaced as a validation warning by
  design. Any "exactly one winner" assertion must scope to 2010+.
- **Scoring systems changed repeatedly** [High]: raw points are not
  comparable across seasons (§6); use positions.
- **Constructor renames fragment history** [High]: §3; new constructorId
  per rebrand, no lineage encoding.
- **Historical duplicates existed and were repaired** [High]: 85 duplicate
  (raceId, driverId) pairs and 2 null-position rows, repaired
  deterministically in `build_interim.py` (Decision 007). Data refreshes
  must re-run the repair pipeline, which is idempotent.
- **Driver nationality strings are dirty** [High]: demonyms with
  inconsistent whitespace ("Argentinian ") — one reason `is_home_circuit`
  is deferred.
- **No weather, tire-compound, practice, or telemetry data exists in this
  schema** [High]: §9 and §11.
- **Pre-1980 data is structurally different** [High]: ~50% finish rates,
  variable field sizes, Indy 500 included in the world championship
  (1950–1960) — all excluded from modeling by the 2010+ window but present
  in rolling-history context (harmless: windows of 3–10 races never reach
  back that far for 2010+ rows).

### Implications for ML
- Prefer models with native NaN handling (XGBoost/LightGBM) or explicit
  missing-indicator pipelines (LogReg) — already the Phase 4 plan.
- Class imbalance (4.67% positive) is structural — ~1 winner per ~20 rows,
  forever; handled by weighting, never resampling (Phase 4 design §5).
- The dataset is SMALL by ML standards (5,077 training rows) — variance in
  validation metrics is large (±1 race ≈ ±2.3 p.p. on the val split);
  selection must respect noise bands (Phase 4 design §9).

### Validation Rules
- **On any data refresh:** re-run the full chain (build_interim →
  build_dataset → features pipeline) and the whole test suite — every
  stage's validators are designed for exactly this.
- **Provenance check (open item):** document where the 2025–2026 rows came
  from (community Ergast-schema continuation) and pin that source before
  Phase 8 automates ingestion. An ingestion source that silently changes
  conventions (e.g., stops using `\N`, renumbers statusIds) is the biggest
  refresh risk.
- Row counts only ever grow; a refresh that shrinks any table is an error.

### Leakage Considerations
- A data refresh mid-experiment changes the feature matrix under the model —
  always log the features.parquet fingerprint (row count + hash) with every
  MLflow run (Phase 4 design §8) so results are attributable to a dataset
  version.

### Recommendations
- Resolve the 2025–2026 provenance question before Phase 8.
- Never hand-edit a CSV; all repairs go through `build_interim.py`
  (reproducibility principle, project_overview.md).

---

## 9. Future Enhancements (data sources)

### Background
Ranked by (expected predictive value) × (integration feasibility) ÷
(leakage risk). None are v1; all must respect the §4 information timeline.

### Confidence Level
Medium throughout — value estimates are informed judgment, not measurement.

| Source | What it adds | Grain | Pre-race availability | Assessment |
|---|---|---|---|---|
| **Weather (forecast + historical)** — e.g., FastF1 session weather, meteorological APIs | Rain probability, temperature (tire behavior) | per session | Forecast: yes. Actuals: only for PAST races (usable in rolling wet-form features) | **Highest value/effort ratio** [Medium]. Enables wet-specialist features (§2). Leakage trap: use *forecast* for the current race, *actuals* only for prior races |
| **Practice pace (FastF1, 2018+)** | Long-run race pace — the classic missing predictor; qualifying measures one lap, FP2 long runs measure race pace | per lap → aggregate to driver-weekend | Yes (Fri/Sat) | High value, high cleaning cost (fuel loads/engine modes unobservable — pace is confounded) [Medium] |
| **Tire compounds & allocation (FastF1)** | Strategy constraints, compound performance | per stint | Allocation: yes. Actual strategy: no (in-race) | Moderate; mostly matters for a strategy model, different grain [Medium] |
| **Sprint results (already in data/)** | Saturday form signal on sprint weekends | per race | Yes (§4 rule 6) | Cheap, already designed (§6.4), format-fragmented; v2 candidate [High feasibility] |
| **Telemetry (FastF1, 2018+)** | Car characteristics (top speed, cornering) | per lap/sample | Prior sessions: yes | Research-grade; heavy; unlikely to beat aggregated proxies for winner prediction [Low] |
| **FIA documents (penalties, TDs)** | Pre-race penalty reasons; mid-season regulation shifts (§1) | per event | Yes (published pre-race) | PDF scraping; brittle; penalty *reasons* would enrich the grid-delta feature [Low-Medium] |
| **Race control messages (FastF1)** | SC/VSF/red-flag history | in-race | Only for past races | Could build "chaos rate by circuit" priors [Low] |
| **News / driver interviews / social sentiment** | Silly-season moves, team morale, upgrade chatter | unstructured | Yes | Noisiest, least reproducible; conflicts with the project's determinism principle; keep out until everything structured is exhausted [Low] |

### Recommendations
- v2 order: weather forecast → sprint features → practice pace.
- Every new source enters through the ingestion/validation/interim pattern
  (Phase 8 architecture) — never joined ad hoc in feature code.

---

## 10. Modeling Implications (consolidated matrix)

_This section intentionally uses a matrix instead of the standard template —
it IS the per-concept template applied across concepts._

| Concept | Why it matters | Candidate features (status) | Validation rules | Leakage rule | Evaluation consideration |
|---|---|---|---|---|---|
| Regulation eras (§1) | Resets break constructor form continuity | `races_since_regulation_reset` (future) | Never hardcode field size | Era boundaries are public in advance — not leakage | Report metrics per season; expect 2026 degradation |
| Car >> driver (§2, §3) | Driver features are car-confounded | Teammate-delta features (future, best de-confounder) | — | Teammate delta must exclude current race | Don't over-interpret "driver skill" importances |
| Rookies/no-history (§2) | NaN is informative | — (NaN discipline implemented) | — | Don't impute to population means | Rookie rows drag metrics; that's correct behavior |
| Grid dominance (§4) | Pole wins ~50% in-window (corrected 2026-07-03, §4); the baseline to beat | `grid_adjusted`, norm, pit-lane (implemented); grid-quali delta (future) | grid ∈ [0, field_size] | grid is pre-race, safe | Pole-sitter heuristic is the mandatory baseline (Phase 4 §3) |
| In-race randomness (§4) | Caps achievable accuracy | — (irreducible) | — | SC/red-flag data is in-race | >70% top-1 = leakage alarm, not success |
| Sprint weekends (§4) | Same-raceId pre-race info | sprint features (deferred, format-aware) | 2021+ only | Pre-race, NOT leakage — the classic false positive | Sparse block: only ~9% of races |
| Circuit affinity (§5) | Sparse but real | circuit history (implemented); pole-conversion rate (future) | prior-visits-only | All circuit aggregates lagged | NaN-heavy at new circuits — expected |
| Standings (§6) | Season-long form summary | position/points/wins prev (implemented) | prev-race map rejects ambiguous calendar | THE canonical lag; never join on own raceId | Position (not points) comparable across eras |
| Class imbalance (§8) | 4.7% positive, structural | — | exactly 1 winner/race in window | Resampling fabricates race entries — forbidden | Per-race metrics only; row accuracy is meaningless |
| Small data (§8) | Metric variance is large | — | — | — | Selection respects noise bands (±2 races on val) |
| Constructor rebrands (§3, §8) | History resets at renames | lineage table (future) | contiguity check | — | Recently rebranded teams under-predicted — known bias |

## 11. Known Simplifications (v1 exclusion registry)

_Purpose: stop future agents from re-introducing intentionally excluded
complexity without a decision. Adding any item below requires a new entry in
`context/decisions.md` and features must pass the §7 leakage rules._

| Excluded from v1 | Why excluded | Where documented | Revisit trigger |
|---|---|---|---|
| Weather (any form) | Not in schema; needs new source | design doc §7; §9 here | First v2 feature-expansion phase |
| Tire compounds/degradation | Not in schema; strategy-model grain | §9 here | Strategy sub-model (Icebox) |
| Live telemetry / lap times / pit stops | In-race data; wrong grain; leakage for own race | design doc §6.1 | In-race model (different project) |
| Team strategy / team orders | Unobservable pre-race; small winner effect | §6 here | Never for winner prediction, likely |
| FIA technical directives | Unobservable in schema | §1 here | If FIA-doc ingestion is ever built |
| Practice pace | Not in schema (FastF1 needed); confounded | §9 here | v2, after weather |
| Sprint features | Format churn per season; 9% sparsity | design doc §6.4 | v2; must be per-season format-aware |
| `is_home_circuit` | No join key; effect probably tiny | design doc §6.3; §5 here | Only if effect demonstrated in data |
| `driver_age_at_race` | Cheap but deferred to keep v1 scope tight | design doc §5.8 | Any feature-expansion pass |
| Constructor lineage mapping | Manual table; rebrand resets tolerated | §3 here | Before extending constructor windows |
| Regulation-era features | Rolling windows self-correct; unproven gain | §1 here | When 2026 data enters modeling |
| Learning-to-rank framing | Decision 003: binary first | decisions.md 003 | After binary model works end-to-end |
| Teammate-delta features | v2 candidate, needs race-grain care | §2 here | v2 feature expansion |
| Neural networks | 5,077 training rows — too small | Phase 4 design §3 | Not foreseeable |

---

## Milestone audit note (2026-07-04)

The Core ML Platform documentation audit found no implemented domain rule that
contradicts this reference. The canonical leakage rules in §7 remain enforced by
the feature and split tests, and the §11 exclusions remain deferred. The unresolved
2025–2026 provenance item in §8 is a **must-resolve precondition** for ETL, forward
evaluation, or serving beyond 2024; those rows must not be treated as an approved
live source merely because they exist in the local files.

## Maintenance Protocol (living document)

Update this document — and note the update in `context/session_handoff.md` —
whenever:
- FIA regulations or race-weekend formats change (new era, new sprint format,
  points changes) → §1, §4, §6.
- A new data source is integrated → §8, §9, and the §4 information timeline.
- A modeling result contradicts a claim here (e.g., an era feature helps, a
  "Low" claim proves true/false) → correct the claim and its confidence tag.
- A new leakage vector is discovered → §7, plus a test, plus a decision entry.
- A "Known Simplification" is implemented → move it out of §11 with a pointer
  to the decision that admitted it.

Rules for edits:
- Never delete a claim silently — correct it and adjust its confidence tag;
  if it was load-bearing for a past decision, note the correction in
  `context/decisions.md`.
- Every new significant claim carries a confidence tag and, where possible,
  a "(verified in data)" check.
- Keep this document about DOMAIN knowledge; implementation details live in
  `architecture.md` and design docs under `reports/`.
