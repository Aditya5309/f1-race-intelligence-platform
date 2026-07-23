#!/usr/bin/env bash
#
# Render "Build Command" for the API service (a native Python web service —
# no Dockerfile involved on this deployment target). See
# docs/render_deployment.md for the full setup and why this script exists.
#
# Every other route already works from a plain `pip install` alone — the
# committed artifacts/ tree (frozen model bundle + features snapshot) is
# all they need (see app/config.py). POST /predict and GET /races/upcoming
# are the one disclosed exception: they need the gitignored, never-committed
# data/ tree (docs/pre_race_materialization.md). Since Render builds this
# repo directly (no GitHub Actions cache, no artifacts/-tree equivalent for
# training-side data), that tree has to be provisioned here, at build time.
#
# Non-negotiable design constraint: a failure anywhere in the data/
# provisioning below must NOT fail this build. If it did, deploying a
# provisioning hiccup (a transient GitHub/jolpica-f1 outage, a rate limit)
# would take down every route, not just the one that's supposed to degrade
# gracefully to 503 when its data isn't available — the exact contract
# app/upcoming_prediction_service.py's ensure_materialization_data()
# already implements at the application layer. This script is the same
# philosophy at the deployment layer: best-effort, never fatal.

set -euo pipefail

echo "=== Installing dependencies ==="
pip install -r requirements.txt
pip install -e .

echo
echo "=== Provisioning data/ for POST /predict (best-effort) ==="
if bash scripts/provision_upcoming_race_data.sh; then
    echo "data/ provisioned successfully — POST /predict and GET /races/upcoming should serve real predictions."
else
    echo "WARNING: data/ provisioning failed or was incomplete." >&2
    echo "Every other route is unaffected. POST /predict and GET /races/upcoming will" >&2
    echo "return 503 until this is resolved — see docs/render_deployment.md." >&2
fi

echo
echo "=== Build complete ==="
