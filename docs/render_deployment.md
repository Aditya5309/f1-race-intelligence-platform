# Render Deployment — API Service Setup

How the API service is deployed to [Render](https://render.com) as a
**native Python web service** (no Dockerfile on this deployment target —
see [Docker](../README.md#9-docker) for the separate, containerized path
used for local/self-hosted deployment).

## Why this needs its own setup, beyond a plain `pip install`

Every route except two serves entirely from the committed `artifacts/`
tree (the frozen model bundle and features snapshot) — a fresh clone with
no `data/` already has everything they need. `POST /predict` and
`GET /races/upcoming` are the disclosed exception: they need the
gitignored, never-committed `data/` tree (a `master_dataset.parquet`-shaped
frame plus several raw dimension CSVs) to materialize a feature row for a
race that hasn't been through the batch pipeline yet — see
[docs/pre_race_materialization.md](pre_race_materialization.md) for the
full architecture and why that gap exists at all.

Render builds this repository directly from source. There is no GitHub
Actions cache here, and no `artifacts/`-tree equivalent for this specific
training-side data — so unlike a local checkout (where `data/` is just
however you last built it) or CI (which restores `data/` from a durable
GitHub Release asset before every retrain run — see
[docs/retrain_workflow_setup.md](retrain_workflow_setup.md)), a fresh
Render build starts with **no `data/` at all**, which is exactly why
`data/processed/master_dataset.parquet` was missing: nothing had ever
provisioned it there.

## The fix: provision `data/` at build time, from the same durable seed

`scripts/render_build.sh` and `scripts/provision_upcoming_race_data.sh`
close this gap by reusing the **exact same** durable `data-seed` GitHub
Release asset the scheduled retrain workflow's own cache-eviction fallback
already depends on (`.github/actions/restore-data-seed`) — not a second,
parallel mechanism. The provisioning script:

1. Downloads and extracts that release asset into `data/` (skipped if
   `data/races.csv` is already present — in practice this never triggers
   on Render itself, since Render's build compute is fully isolated per
   deploy with no access to a prior deploy's disk; see "Performance &
   trade-offs" below. The check still matters for local runs or any other
   environment where `data/` might already be there).
2. Runs `scripts/ingest_jolpica.py` to backfill any races completed since
   that snapshot was taken — keeps "the next race with no result yet"
   accurate on every deploy, not frozen at the seed's own date.
3. Rebuilds `data/interim/*.parquet` and
   `data/processed/master_dataset.parquet` (`src.data.build_interim`,
   `src.pipelines.build_dataset`) from whatever `data/` now holds.

**This never runs model training, registration, or promotion** — Render
serves from the already-frozen, committed `artifacts/` bundle exactly like
every other deployment target; this script only ever touches the
gitignored `data/` tree, never `artifacts/`.

**Nothing here is fatal to the deploy.** A failure anywhere in
provisioning (a GitHub or jolpica-f1 outage, a rate limit) is logged as a
warning and the build still succeeds — `POST /predict` and
`GET /races/upcoming` alone degrade to `503` in that case
(`app/upcoming_prediction_service.py`'s `ensure_materialization_data()`
already implements exactly this contract at the application layer; this
script is the same non-fatal philosophy one layer up, at the deployment
level). Every other route is never affected by anything in this script.

## Render service configuration

**Runtime:** Python (native — not Docker).

**Build Command:**
```
bash scripts/render_build.sh
```

**Start Command:**
```
uvicorn app.api:app --host 0.0.0.0 --port $PORT
```
(`--host 0.0.0.0 --port $PORT` is required by Render regardless of this
data-provisioning fix — Render assigns the port dynamically via `$PORT`
and only proxies to a service listening on all interfaces; the
`uvicorn app.api:app` form used elsewhere in this repo's docs assumes a
fixed local port and is for local/Docker use only.)

**Environment variables:** every `F1_*` setting from
[`.env.example`](../.env.example) is supported the same way here as
anywhere else (see [docs/user_guide.md](user_guide.md#configuration)) — set
them in Render's dashboard under the service's Environment tab, not a
committed `.env` file. Nothing Render-specific is needed beyond the Build
and Start commands above.

## Known limitation this does not remove

`app/upcoming_prediction_service.py`'s `ensure_materialization_data()`
loads `data/interim/race_weather.csv` unconditionally today — even though
the served model excludes the wet-weather feature group by default (see
[README's Data & ML Pipeline section](../README.md#7-data--ml-pipeline)).
If that one file is ever absent, `POST /predict` degrades to `503` the
same as any other missing input, purely as a side effect of that
unconditional load — not because weather is actually required for a
prediction. `scripts/provision_upcoming_race_data.sh` surfaces this
explicitly as a warning if it happens, rather than failing silently.

## Performance & trade-offs

Measured directly (not estimated) against the real `data-seed` release
asset and the real ingestion/rebuild pipeline, on a typical broadband
connection:

| Step | Measured time | Notes |
|---|---|---|
| Download `data-seed.tar.gz` | ~1.2s | 20,531,056 bytes (~19.6 MiB) compressed |
| Extract | ~0.2s | → a ~40 MB `data/` tree |
| `scripts/ingest_jolpica.py` | ~30s | Dominated by per-upcoming-race status-check calls to jolpica-f1 (12 in the observed run), not data volume — scales with how many races have completed since the snapshot was taken, not with total dataset size |
| `src.data.build_interim --target all` | ~0.9s | Pure pandas transforms over already-local CSVs |
| `src.pipelines.build_dataset` | ~1.0s | Join-only, no feature engineering |
| **Data-provisioning total** | **~33s** | The four steps above, combined |
| `pip install -r requirements.txt` (no cache, full `[dev]` extra) | ~2m 7s | Pre-existing cost, unrelated to this fix — dominates total build time |
| **Estimated total build time** | **~2.5–3 min** | Provisioning + install |

**Fit against Render's documented limits** ([Render build pipeline docs](https://render.com/docs/build-pipeline)):
- **Build timeout: 120 minutes.** ~3 minutes used ⇒ roughly **40×** headroom.
- **Build disk cap: 16 GB.** The installed dependency set (~1.5 GB, measured
  in a clean venv with the full `[dev]` extra) plus the ~40 MB `data/` tree
  ⇒ **~1.55 GB used, ~10%** of the cap.

Both fit comfortably, with large margin — this is not a close call.

### Trade-offs worth knowing about

- **Every Render deploy re-does the full download + backfill + rebuild,
  with no cross-deploy caching possible.** Render's own docs state build
  compute is fully isolated from any running instance's disk ("Builds
  don't have access to your running service instance's resources... This
  is because pipeline tasks run on completely separate compute") — this
  holds regardless of whether a persistent disk is attached to the
  service, since a persistent disk mounts to the *runtime* instance, not
  the *build* environment. Concretely: the `data/races.csv`
  already-present check in `scripts/provision_upcoming_race_data.sh` is a
  real optimization for local re-runs, but a no-op on Render — the ~33s
  cost is paid on every single deploy, not just the first one. That's
  still comfortably cheap, so this is a documented characteristic, not a
  problem to fix.
- **Freshness depends on deploy cadence, not a schedule.** Unlike the
  GitHub Actions retrain workflow (which runs weekly regardless), Render
  only rebuilds `data/` when a deploy actually happens. A service left
  running for a long stretch without a redeploy will keep serving
  `POST /predict`/`GET /races/upcoming` against whatever `data/` its *last*
  deploy provisioned — correct, not stale in a way that breaks anything
  (the served model and every other route are entirely unaffected either
  way), but worth knowing if "next race" ever looks behind the real
  calendar: that means it's time to redeploy, not a bug in this script.
- **jolpica-f1's 200 req/hour rate limit is not a concern at normal deploy
  cadence** (a handful of races' worth of catch-up per deploy, per the
  measured ~30s/12-call cost above) but would become one for a service
  redeployed only rarely after a long idle period, or redeployed
  repeatedly in a short window (e.g. rapid iterative deploys while
  debugging something unrelated) — each still pays the same ~30s
  status-check cost regardless of whether anything new is actually found.
- **The seed asset carries ~33 MB this pipeline never reads**
  (`archive.zip` — the original raw download — plus `lap_times.csv`, and a
  few other CSVs outside the 9 tables `src.integration.build_master_dataset`
  actually joins on). Harmless at this size (a ~20MB download either way
  is trivial against the limits above), but a leaner Render-specific seed
  containing only the required tables would shrink both the download and
  the extracted footprint further — a real future optimization, not
  implemented here since there's no measured need for it.
- **`requirements.txt` installs the full `[dev]` extra** (pytest, ruff,
  `notebook`/Jupyter, `pip-audit`) on Render, none of which the running API
  needs — analogous to `docker/Dockerfile.api` using a slim,
  serving-only `docker/requirements-api.txt` instead. This is the dominant
  cost in the total build time above and is unrelated to this fix (it
  predates it); switching Render's Build Command to a runtime-only install
  would cut it, but is a separate, optional optimization, not required for
  correctness or to fit within Render's limits.

## One-time verification

The durable `data-seed` release asset (as of this writing) already
contains a precomputed `data/processed/master_dataset.parquet` and
`data/interim/race_weather.csv` from when it was last packaged — so the
very first Render build after adding this script should succeed without
needing `scripts/ingest_jolpica.py` or the rebuild steps to do anything
beyond a routine incremental refresh. If the seed is ever refreshed
without those two paths (see Part 1 of
[docs/retrain_workflow_setup.md](retrain_workflow_setup.md) for the
packaging commands), re-run `src.data.build_interim`/
`src.pipelines.build_dataset` once against a full local `data/` tree
before re-uploading it, or `POST /predict` will 503 on Render until the
next successful `ingest_jolpica.py` + rebuild pass fills the gap back in.
