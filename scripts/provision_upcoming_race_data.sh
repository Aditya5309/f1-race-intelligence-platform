#!/usr/bin/env bash
#
# Provisions the gitignored data/ tree that POST /predict and
# GET /races/upcoming need (settings.raw_data_dir / master_dataset_path /
# qualifying_interim_path / weather_csv_path in app/config.py) on a
# deployment that never runs this project's own CI (e.g. Render). See
# docs/render_deployment.md for how this is wired into the Render build,
# and docs/pre_race_materialization.md for why this data has no
# artifacts/-tree equivalent in the first place.
#
# Reuses the SAME durable source the scheduled retrain workflow's own
# cache-eviction fallback already restores from (see
# .github/actions/restore-data-seed and docs/retrain_workflow_setup.md) —
# not a second, parallel data source. Downloaded here via a plain
# unauthenticated HTTPS request rather than `gh release download`: this
# repository and its releases are public, so no token is needed — `gh` is
# a GitHub-Actions-runner convenience, not a requirement for reading a
# public release asset from any other environment.
#
# Idempotent and safe to re-run on every deploy:
#   - The `data/races.csv` presence check below is a no-op on Render
#     specifically: Render's own docs confirm build compute is fully
#     isolated per deploy (no access to a prior deploy's disk, persistent
#     disk or not — see docs/render_deployment.md's "Performance &
#     trade-offs" section for the citation), so every Render build starts
#     with no data/ and always takes the download branch. The check still
#     matters for local runs / any other environment where data/ might
#     already exist from a previous invocation — harmless either way.
#   - Backfilling and rebuilding are safe to repeat: ingest_jolpica.py only
#     ever adds missing completed races, and build_interim/build_dataset
#     are pure, deterministic rebuilds from whatever data/ already holds.
#
# Exit code reflects whether the DATA IS USABLE afterward — the caller
# (scripts/render_build.sh) still does not let that fail the overall
# build (see that script's own comment).

set -uo pipefail   # deliberately no -e: report and continue past one
                    # failed sub-step (a partially refreshed data/ tree is
                    # still more useful than none) rather than aborting the
                    # whole script on the first problem.

REPO="Aditya5309/f1-race-intelligence-platform"
RELEASE_TAG="data-seed"
ASSET_NAME="data-seed.tar.gz"
ASSET_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"

# 9 raw CSVs src.integration.build_master_dataset joins on (the same list
# .github/actions/restore-data-seed sanity-checks after its own restore).
REQUIRED_CSVS=(races drivers constructors circuits driver_standings constructor_standings status qualifying results)

status=0

if [ ! -f data/races.csv ]; then
    echo "data/ not present — downloading the '${RELEASE_TAG}' release snapshot..."
    if curl -fsSL -o "${ASSET_NAME}" "${ASSET_URL}" && tar -xzf "${ASSET_NAME}"; then
        rm -f "${ASSET_NAME}"
        echo "Extracted ${ASSET_NAME}."
    else
        echo "ERROR: failed to download or extract ${ASSET_URL}" >&2
        status=1
    fi
else
    echo "data/ already present — skipping seed download, refreshing in place."
fi

for f in "${REQUIRED_CSVS[@]}"; do
    if [ ! -f "data/${f}.csv" ]; then
        echo "ERROR: data/${f}.csv missing — the seed snapshot is incomplete or this checkout's data/ is corrupt." >&2
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "Backfilling any races completed since this data snapshot was taken..."
    if ! python scripts/ingest_jolpica.py; then
        echo "WARNING: ingest_jolpica.py failed — continuing with the snapshot as-is (may not reflect the most recent race weekend)." >&2
    fi

    echo "Rebuilding data/interim/*.parquet..."
    if ! python -m src.data.build_interim --target all; then
        echo "WARNING: build_interim failed — data/interim/*.parquet may be stale." >&2
    fi

    echo "Rebuilding data/processed/master_dataset.parquet..."
    if ! python -m src.pipelines.build_dataset; then
        echo "ERROR: build_dataset failed — data/processed/master_dataset.parquet is missing or stale; POST /predict will 503." >&2
        status=1
    fi
fi

if [ ! -f data/interim/race_weather.csv ]; then
    echo "WARNING: data/interim/race_weather.csv missing — the wet-weather feature group won't compute for a materialized row. This alone does not fail provisioning: those features are excluded from the served model by default (see README's Data & ML Pipeline section), but app/upcoming_prediction_service.py's ensure_materialization_data() still loads this file unconditionally today, so its absence WILL 503 POST /predict. See docs/render_deployment.md." >&2
    status=1
fi

exit "$status"
