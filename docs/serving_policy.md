# Serving Policy: three separate "which years?" questions

This project has three distinct mechanisms that each answer a different
question about "which race years." They share a similar shape (a year, or a
comparison against one) and used to share a single config value by
coincidence, which made them easy to conflate — this doc exists so that
mistake isn't repeated.

| Question | Mechanism | Can it change, and how | Config / code |
|---|---|---|---|
| Which historical seasons can the live dashboard/API show? | Verified-seasons serving policy | Only via a new decision recording how newer seasons were provenance-verified | `app/config.py::Settings.verified_seasons_through`, `Settings.is_season_verified()` |
| Which seasons does the model get evaluated on, kept genuinely unseen? | Evaluation holdout | Effectively never — a fixed scientific boundary, not an operational knob | `src/models/splits.py::FORWARD_HOLDOUT_MIN_YEAR` |
| What race gets a live pre-race prediction? | Upcoming-race prediction | Always "the next race with no result," recomputed on every request | `src/features/upcoming.py::next_race()`, `POST /predict`, `GET /races/upcoming` |

## 1. Historical serving: verified-seasons policy

`GET /races`, `GET /predictions/{race_id}`, `.../simulate/{driver_id}`, and
`.../vs-baseline` only serve races from seasons that
`Settings.is_season_verified(year)` accepts — currently `year <=
verified_seasons_through` (default `2024`, override via
`F1_VERIFIED_SEASONS_THROUGH`).

**Why 2025–2026 are excluded today (Decision 050, 2026-07-22):** the
2025–2026 rows present in `data/`/`features.parquet` are schema-consistent
with the rest of the historical dump, but their upstream origin is a
post-2024 community continuation feed of unverified identity — distinct
from `scripts/ingest_jolpica.py`, the pinned, verified jolpica-f1 source used
for every new race weekend going forward (Decision 035). Decision 050 could
only confirm what these rows *aren't* (not jolpica output — they carry 26
distinct DNF status values against jolpica's single generic one) — not
resolve who *did* produce them. That question was ruled unresolvable from
this repo alone. Until a future decision records how it was resolved,
`verified_seasons_through` stays at `2024` and 2025–2026 stay unserved
historically, regardless of how much wall-clock time passes.

**Why this is a policy value, not a calendar cutoff:** raising it requires a
new decision entry documenting *how* the newer seasons' provenance was
verified — never a routine bump to track "the current year." That's the
whole reason it's named and described in terms of verification rather than
recency: a name like "serve through last year" invites exactly the kind of
silent staleness this value must never have.

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
  GET /races                -> Settings.verified_seasons_through filter
  _race_rows() (shared by
    /predictions, /simulate,
    /vs-baseline)            -> Settings.is_season_verified() -> 409

src/models/splits.py
  SplitStrategy.__post_init__ -> rejects test windows >= FORWARD_HOLDOUT_MIN_YEAR
                                  unless allow_forward_holdout=True

app/upcoming_prediction_service.py
  resolve_upcoming_race() /
  resolve_upcoming_prediction() -> next_race(), unrelated to either guard above
```
