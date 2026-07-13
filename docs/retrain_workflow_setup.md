# Scheduled Retrain Workflow — Manual Setup & First-Run Runbook

_Operational runbook for `.github/workflows/retrain.yml`. Covers the
one-time manual steps that workflow cannot do for itself, and what to watch
for on its first real run._

## Why manual steps are needed at all

`data/` is entirely gitignored — a fresh GitHub Actions
checkout has none of it. `scripts/ingest_jolpica.py` only **backfills races
missing from an existing `data/` tree**; it is not a full-history bootstrap
(jolpica-f1's unauthenticated rate limit is 200 requests/hour, and a full
history pull needs ~4,700 requests — over 24 hours, impractical for a
weekly job). `retrain.yml` persists `data/` across runs via a GitHub
Actions cache, but **that cache starts out empty** — it has to be seeded
once, manually, before the first scheduled or dispatched run can find any
data at all.

---

## Part 1 — Seed the `data/` cache (one-time, before the first run)

### Cache key this reads/writes (confirm against the workflow file, don't trust this doc if they ever diverge)

`retrain.yml`'s restore step (`Restore data/ cache`) uses:
```yaml
key: f1-data-${{ github.run_id }}
restore-keys: |
  f1-data-
```
`restore-keys` does a **prefix match** — it finds the most recent cache
entry whose key starts with `f1-data-`, whatever the rest of that key is.
The seeding workflow below saves under `f1-data-seed-<run id>`, which
shares that prefix — so it's exactly what the very first scheduled run's
lookup will find. (Why not one fixed key for everything: GitHub Actions
cache entries are **immutable** once created under an exact key —
`actions/cache/save` silently no-ops if you try to reuse one — so every
save here uses a fresh, run-unique key instead. See the "CACHE KEY SCHEME"
comment at the top of `retrain.yml` for the full reasoning.)

### Step 1 — Package your local `data/` tree

Run this from the repo root. It excludes `archive.zip` (the original raw
download, not read by the pipeline) and `interim/`/`processed/` (rebuilt
fresh by every retrain run regardless — no point seeding stale copies):

```bash
tar -czf data-seed.tar.gz \
  --exclude='data/archive.zip' \
  --exclude='data/interim' \
  --exclude='data/processed' \
  --exclude='data/.gitkeep' \
  data
```

Verified locally: this produces a ~9MB archive containing exactly the 14
top-level Ergast CSVs, and round-trips correctly (`tar -xzf data-seed.tar.gz`
from an empty directory reproduces `data/*.csv` exactly).

### Step 2 — Upload it as a GitHub Release asset

This is CI-only plumbing, not a real versioned release — the tag
`data-seed` is just a stable anchor the seed workflow knows to look for.

First time:
```bash
gh release create data-seed data-seed.tar.gz \
  --repo Aditya5309/f1-race-intelligence-platform \
  --title "Training data snapshot (CI cache seed — not a real release)" \
  --notes "Seed snapshot for .github/workflows/seed-data-cache.yml. data/ is deliberately gitignored and never committed to any branch — this release asset is the only place a full data/ snapshot lives outside your local machine."
```

Re-seeding later (cache evicted, or you want to refresh from a bigger local
`data/`) — same command, but upload to the existing release instead:
```bash
gh release upload data-seed data-seed.tar.gz --clobber \
  --repo Aditya5309/f1-race-intelligence-platform
```

### Step 3 — Run the seed workflow

```bash
gh workflow run seed-data-cache.yml --repo Aditya5309/f1-race-intelligence-platform
```

Or via the UI: **Actions** tab → **Seed data/ cache** (left sidebar) →
**Run workflow** button → branch `main` → **Run workflow**.

This workflow (`.github/workflows/seed-data-cache.yml`):
1. Downloads `data-seed.tar.gz` from the `data-seed` release via `gh release download`.
2. Extracts it.
3. Sanity-checks all 9 CSVs the pipeline actually needs are present (fails loudly, naming the missing file, if not).
4. Saves the result to the `f1-data-` cache under a fresh, run-unique key.

Takes well under a minute — it's a ~9MB download/extract/cache-save, no
Python dependencies installed, no model training.

### Step 4 — Confirm the cache actually populated

```bash
gh cache list --repo Aditya5309/f1-race-intelligence-platform
```

Look for an entry whose key starts with `f1-data-seed-` and check its
size is in the right ballpark (tens of MB, not a few KB — a few KB would
mean the extraction produced an empty or near-empty `data/`). Equivalent
UI check: repo → **Settings** → **Actions** → **Caches**.

**Do not proceed to Part 2 until this shows a real entry.** If it's
missing, re-check the `seed-data-cache.yml` run's logs (Step 3) —
the sanity-check step there will have failed loudly and named exactly
which CSV was missing or empty if the archive was built wrong.

---

## Part 2 — First real `workflow_dispatch` run of `retrain.yml`

### Trigger

```bash
gh workflow run retrain.yml --repo Aditya5309/f1-race-intelligence-platform
```

Or UI: **Actions** → **Scheduled Retrain** → **Run workflow** → branch
`main` → **Run workflow**.

Then watch it:
```bash
gh run watch --repo Aditya5309/f1-race-intelligence-platform
```
(picks the most recent run automatically if you run this right after
dispatching; or use `gh run list --workflow=retrain.yml` to find the run
id first if anything else has run in between).

### What "working correctly" looks like, stage by stage

| Step | What success looks like | What failure looks like, and what it means |
|---|---|---|
| **Restore data/ cache** | Log line showing a cache hit against a key starting `f1-data-` (the seed entry, on this first run) | "Cache not found for input keys" — the seed (Part 1) wasn't completed, or the cache was evicted before this ran. Go back to Part 1. |
| **Confirm data/ was restored** | Passes silently (no output) | `::error::data/races.csv not found...` — the restore step above ran but the archive it restored is somehow empty/wrong. Re-check Part 1 Step 4's cache-list output. |
| **Ingest + rebuild + freeze** (the longest step, several minutes) | Prints each of `refresh_and_freeze.py`'s 8 sub-steps passing: ingestion (reports races fetched or "Nothing to ingest" if fully caught up), `build_interim`, `build_dataset`, `features.pipeline`, current-era season tracking (scores any newly completed race against the bundle that was already served — see `src/models/season_tracking.py`), runtime features-snapshot refresh (`scripts/refresh_features_snapshot.py` — always, independent of promotion), display-data refresh, and finally `Registered f1-winner vN as @Staging` | Any sub-step's own error surfaces here and stops the job (the orchestrator halts at the first failure, per its own design — nothing after a failed step runs). A jolpica-f1 API error here most likely means you've hit its 200 req/hour rate limit — wait and re-dispatch. |
| **Upload ingestion report** | An `ingest-report-<run-id>` artifact appears on the run's summary page — download with `gh run download <run-id> -n ingest-report-<run-id>` for `summary.json` (race/row counts, skipped races) plus one CSV per endpoint of exactly the new rows this run fetched (never the full `data/` tree). Runs even if a later step fails (`if: always()`) — it's independent visibility into ingestion, not a gate. | `if-no-files-found: warn` — a missing artifact means ingestion itself never got far enough to write a report (check the step above for the real error); this step itself does not fail the job. |
| **Save data/ cache** | Runs regardless of the step above (marked `if: always()`) — confirms a new cache entry was saved | Should not fail; if it does, it's a GitHub Actions infrastructure issue, not this project's code. |
| **Open PR: refresh display data, features snapshot, tracking set** | A PR appears against `main`, branch `scheduled-retrain/data-refresh`, scoped to ONLY `artifacts/display/`, `artifacts/features.parquet`, `artifacts/tracking/` — **unconditional**, no `if:` tying it to promotion (Part 1 fix: this is "current vs. stale" ground-truth data, not a quality judgment, so it must not be silently discarded just because that week's model candidate is refused). Still human-reviewed like everything else, not auto-merged. If there's nothing new (e.g. `--skip-ingest` locally, or a week with zero completed races), the action exits silently — no empty PR. | Same permissions note as the model-promotion PR below. If this PR is missing on a week where you know new data was ingested, check this step's own log — it runs unless a REQUIRED step before it (ingestion through display refresh) failed outright. |
| **Promote (gated)** | Either `PROMOTED f1-winner vN to @Staging -> ...` (candidate passed all 3 gate checks: model-class, smoke, metric non-regression) or `PROMOTION REFUSED: <specific reason>` — both are captured, neither fails the job | N/A — this step is designed to never itself be "the failure"; the reason (if refused) is in its own printed output, surfaced again in the next step. |
| **Report refusal and stop** *(only runs if refused)* | Prints `::warning::Promotion refused...` plus the exact reason from `promote_model.py`, then stops cleanly | This IS the "working correctly, but nothing to promote this week" outcome — see Part 3 below. No model-promotion PR opens (the data-refresh PR above still may have). `artifacts/serving/` is untouched. |
| **Build metrics-diff summary** *(only runs if promoted)* | Prints a markdown table comparing previously-served vs. new candidate metrics (top1_accuracy, spearman_corr, etc.) | Would only fail if `artifacts/serving/staging/manifest.json` is somehow malformed after a successful promotion — shouldn't happen if the step above genuinely reported success. |
| **Open PR with the promoted candidate** *(only runs if promoted)* | A separate PR appears against `main`, branch `scheduled-retrain/staging`, scoped to ONLY `artifacts/serving/`, titled "Scheduled retrain: new candidate promoted to Staging", body = the metrics-diff table | A permissions error here (403/token) means the repo's Actions settings don't allow this token to open PRs — check **Settings → Actions → General → Workflow permissions**, needs "Read and write permissions" + "Allow GitHub Actions to create and approve pull requests". |

### If something fails and you're not sure why

Re-run with more detail: `gh run view <run-id> --log --repo Aditya5309/f1-race-intelligence-platform`
finds the exact step and its full output. Every step above is designed to
fail with an explicit, human-readable message (not a bare stack trace) —
if you see a bare Python traceback instead, that's itself worth reporting,
since it means a code path wasn't handling an error case that the design
assumed would be caught.

---

## Part 3 — Does this touch `main` directly, or promote anything automatically?

**No. A "successful" run means, at most, a PR appears for you to review —
nothing merges, and nothing is served, without you clicking merge
yourself.** Concretely, tracing the actual mechanism:

- The workflow's own `permissions:` block grants `contents: write` and
  `pull-requests: write` — write access to open branches/PRs, **not** any
  branch-protection bypass or direct-push special case for `main`.
- `promote_model.py` writes to `artifacts/serving/` inside the **runner's
  own ephemeral checkout** — this is not visible anywhere outside that one
  job until something explicitly commits and pushes it.
- The only step that turns those local file changes into anything
  persistent is `peter-evans/create-pull-request@v6`, which — true to its
  name — **opens a pull request**. It does not merge, does not push
  directly to `main`, and `main` itself is not the branch it commits to
  (it creates/pushes `scheduled-retrain/staging` and opens a PR from
  *that* branch against `main` — `base: main` is set explicitly in the
  workflow, not left to default to whatever ref the run happened to be
  dispatched against).
- If `promote_model.py` refuses (metric regression, disallowed model
  class, or any smoke-check failure), the "Report refusal and stop" step
  runs instead and the job ends there — the PR step's `if:` condition
  (`env.PROMOTE_EXIT_CODE == '0'`) is false, so it's skipped entirely, not
  attempted-and-failed.

So there are exactly three possible outcomes of a real run, and none of
them change `main` or what's served without your review:
1. **Refused** — no PR, no bundle change, a clearly-logged reason.
2. **Promoted, PR opened** — `artifacts/serving/`/`artifacts/features.parquet`/`artifacts/display/`
   changed **only inside that PR's branch**; `main` and the real deployed
   bundle (wherever that's actually hosted) are unaffected until you merge it.
3. **A step failed outright** (ingestion error, cache miss, etc.) — job
   goes red, nothing downstream ran, nothing changed anywhere.
