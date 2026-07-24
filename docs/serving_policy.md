# Serving Policy: three separate "which years?" questions

This project has three distinct mechanisms that each answer a different
question about "which race years." They share a similar shape (a year, or a
comparison against one) and used to share a single config value by
coincidence, which made them easy to conflate — this doc exists so that
mistake isn't repeated.

| Question | Mechanism | Can it change, and how | Config / code |
|---|---|---|---|
| Which historical races can the live dashboard/API show? | Structural completeness guarantee | Automatic — a race becomes servable the moment it has rows in the features snapshot, never a config change | `artifacts/features.parquet` row presence, enforced nowhere else needed |
| Which seasons does the model get evaluated on, kept genuinely unseen? | Evaluation holdout | Effectively never — a fixed scientific boundary, not an operational knob | `src/models/splits.py::FORWARD_HOLDOUT_MIN_YEAR` |
| What race gets a live pre-race prediction? | Upcoming-race prediction | Always "the next race with no result," recomputed on every request | `src/features/upcoming.py::next_race()`, `POST /predict`, `GET /races/upcoming` |

## 1. Historical serving: structural completeness guarantee (Decision 057)

`GET /races`, `GET /predictions/{race_id}`, `.../simulate/{driver_id}`, and
`.../vs-baseline` serve every race present in the frozen features snapshot
(`artifacts/features.parquet`) — there is no year-based config gate. A race
appears there if and only if it has a completed result: `build_master_dataset()`
left-joins every other table onto `results.csv`, so a race with zero result
rows produces zero rows anywhere downstream, by construction, not by a
filter. "Don't serve a race with no result yet" therefore needs no runtime
check — it's already true by the time the snapshot is built.

**Why this replaced a config-driven cutoff (`verified_seasons_through`,
removed 2026-07-24):** that field's entire purpose was gating on Decision
050's then-unresolved provenance question for 2025–2026 data. Decision 056
(2026-07-24) resolved that question — the 2025–2026 rows are the same
Ergast/jolpica-f1 lineage as every earlier season, externally verified
against two independent datasets. With the provenance concern gone, a
year-keyed gate had no remaining justification, and it was also always the
wrong shape: this project's real data gaps are race-level (a handful of
specific raceIds), never season-level, so a year cutoff could only ever
express "everything through season N," not the actual shape of any gap this
data has had. A config value would also have to be remembered and bumped by
a human — the structural guarantee requires no maintenance and cannot go
stale by definition. See `reports/serving_policy_revision_proposal.md` and
`reports/serving_gate_removal_validation.md` for the full analysis, and
Decision 057 in `context/decisions.md` for the implementation record.

**Known residual gap (not new, not a regression):** the completeness
guarantee is proven structurally for "no results," "cancelled," "future
scheduled," and "failed feature generation." It is not structurally proven
for a race with *partial* results (some but not all drivers) — nothing
rejects that shape today, only a non-blocking warning. This has never
occurred in this dataset and is unlikely given `ingest_jolpica.py` receives
results atomically per race, but it's a real, narrow, currently-theoretical
gap worth knowing about. The removed gate never protected against this
either (it only ever checked `year`), so this isn't a new risk.

**Ingestion-quality issue, deliberately separate:** 4 races (raceId
1175–1178) show generic "Retired" instead of a specific DNF cause — a
display-only data-quality question, not a reason to withhold serving. See
`reports/ingestion_status_granularity_issue.md`.

## 2. Evaluation holdout

`src/models/splits.py::FORWARD_HOLDOUT_MIN_YEAR = 2025` marks the start of a
genuinely unseen test window for comparing candidate models (Decision
008/012). `SplitStrategy` rejects any split whose test window reaches this
year unless `allow_forward_holdout=True` is passed explicitly — an opt-in,
not a default — so this boundary can't be crossed by accident.

This is answering a different question from serving: it protects
*reproducible model comparison*, not *what a live user can browse*. It is
expected to stay fixed indefinitely; nothing about verified-seasons policy
changes above should ever touch it. (2025–2026 rows *are* already usable as
internal engineering evidence for Phase 4/6 correctness gates — golden-row
parity, historical backtests — since those are checks of the pipeline
against already-known outcomes, not a real-world accuracy claim. Decision
050 draws that line explicitly.)

## 3. Upcoming-race prediction

`POST /predict` and `GET /races/upcoming` always resolve "the single next
scheduled race with no result yet" via `next_race()` (horizon = 1, see
[`pre_race_materialization.md`](pre_race_materialization.md)) and serve a
live materialized prediction for it — completely independent of
`verified_seasons_through`. Provenance-verification doesn't apply to a race
that hasn't happened: there's no historical result to trust or distrust yet.
This is why the dashboard's Race Center can show a 2026 prediction (the next
race) while every other completed 2026 race stays excluded from historical
serving.

## Where each is enforced

```
app/api.py
  GET /races                -> reads app.state.features (artifacts/features.parquet) directly
  _race_rows() (shared by
    /predictions, /simulate,
    /vs-baseline)            -> 404 only if raceId absent from the snapshot

src/models/splits.py
  SplitStrategy.__post_init__ -> rejects test windows >= FORWARD_HOLDOUT_MIN_YEAR
                                  unless allow_forward_holdout=True

app/upcoming_prediction_service.py
  resolve_upcoming_race() /
  resolve_upcoming_prediction() -> next_race(), unrelated to either guard above
```
