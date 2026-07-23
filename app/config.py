"""
app/config.py

Application-layer configuration.

Every knob the API or dashboard reads comes through this Settings class —
environment variables prefixed `F1_` (e.g. F1_SERVING_BUNDLE_PATH=...), with
`.env`-file support for local development (.env is gitignored). No hardcoded
paths in app code: the first local run and a container run use identical
code, different environments.

Serving never resolves a live MLflow registry alias — there is deliberately
no `tracking_uri`/`model_alias` setting here. `serving_bundle_path` points
at a frozen bundle (src.models.serving_bundle) instead; the API doesn't
know or care what alias or experiment produced it.

Both `serving_bundle_path` and `features_path` default under `artifacts/`
— a runtime artifact tree that is committed to git (contrast with
`data_dir`, `data/`, `mlruns/`, `mlflow.db`, all gitignored). A freshly
cloned repository (source + `artifacts/` only, no `data/`) has everything
the deployed API requires to serve predictions.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS_ROOT = _PROJECT_ROOT / "artifacts"

#: The API's versioned URL prefix. Not a Settings field — this is internal
#: wiring, not an operator-configurable knob. Shared by app/api.py (mounts
#: every route under this prefix) and app/views/common.py (the dashboard's
#: HTTP client prepends it to every request) so both sides can never drift
#: apart. Lives here, not in app/api.py, because app/views/ must never
#: import app.api (that would pull src/ imports into the dashboard process
#: — the whole point of the dashboard-never-imports-src/ boundary) but both
#: already import app.config for Settings.
API_V1_PREFIX = "/api/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="F1_", env_file=".env", extra="ignore",
    )

    # --- model --------------------------------------------------------------
    #: Frozen serving bundle directory — no MLflow tracking URI or registry
    #: alias needed at request time. Committed to git under artifacts/
    #: (unlike the gitignored data/ and mlruns/ trees).
    serving_bundle_path: Path = _ARTIFACTS_ROOT / "serving" / "staging"

    # --- data --------------------------------------------------------------
    #: Frozen serving feature matrix — a committed snapshot exported by
    #: train.py::register_model(), NOT the live training-side
    #: data/processed/features.parquet (which stays gitignored).
    features_path: Path = _ARTIFACTS_ROOT / "features.parquet"
    #: Directory holding the display-name/metadata CSVs (drivers, races,
    #: standings, etc.) — an OPTIONAL enrichment only: names degrade to a
    #: fallback string if the files are absent, never a hard requirement to
    #: serve predictions. Defaults to the committed frozen snapshot
    #: (artifacts/display/, scripts/export_display_data.py) rather than the
    #: gitignored data/ tree, so a fresh clone resolves real names out of
    #: the box — mirrors how features_path/serving_bundle_path already
    #: default under artifacts/. Point at the full data/ tree locally via
    #: .env (F1_DATA_DIR=data) if wanted; nothing requires it.
    data_dir: Path = _ARTIFACTS_ROOT / "display"

    # --- serving policy ----------------------------------------------------
    #: Forward-holdout guard: races after this year return 409. The
    #: 2025-2026 seasons are held out as an unseen evaluation window — raise
    #: this deliberately, with a real evaluation plan, never by accident.
    serve_max_year: int = 2024
    #: Enables GET /debug/* (development only — keep false in production).
    debug_endpoints: bool = False
    #: Bounded per-race prediction cache, keyed (model_version, race_id).
    prediction_cache_size: int = 512

    # --- ops ----------------------------------------------------------------
    log_level: str = "INFO"
    #: Dashboard-side: where the API lives.
    api_url: str = "http://localhost:8000"

    # --- API hardening -------------------------------------------------------
    #: Comma-separated list of allowed CORS origins, e.g.
    #: "https://example.com,https://foo.example.com". Empty (default)
    #: means no origin is allowed cross-origin access — CORSMiddleware is
    #: still registered either way, but with an empty allow-list this is
    #: behaviorally identical to not having CORS middleware at all (no
    #: browser ever gets an Access-Control-* header back). CORS only
    #: matters for browser JS callers; it has no effect on this project's
    #: own dashboard (a server-side httpx client, never a browser), curl,
    #: or tests. Set "*" to allow any origin — reasonable for this
    #: read-only public demo API; never combine "*" with credentials
    #: (this API never uses cookies/auth headers, so that risk doesn't
    #: apply here regardless).
    cors_allow_origins: str = ""

    # --- pre-race materialization (see docs/pre_race_materialization.md) ----
    #: Training-side data the Materializer (`src.models.materialize`) needs
    #: — NOT the artifacts/ tree every other route reads from. Unlike
    #: `GET /predictions/{race_id}`, `POST /predict` genuinely cannot be
    #: served from a fresh clone with `artifacts/` alone: `historical_master`
    #: (a full `master_dataset.parquet`-shaped frame) and the raw dimension
    #: tables it's built from have no committed, artifacts/-tree equivalent
    #: today — `features.parquet` only carries the final computed features,
    #: not the raw columns needed to feed back through the pipeline. This is
    #: a genuine, disclosed gap (see `docs/pre_race_materialization.md`),
    #: not an oversight: these paths default to the local `data/` tree
    #: (present on a local dev checkout or a container with `data/`
    #: bind-mounted, e.g. `docker-compose.override.yml`) and, if missing,
    #: `POST /predict` alone degrades to 503 — every other route is
    #: unaffected, the same pattern as the existing pole-baseline
    #: degraded-start.
    raw_data_dir: Path = _PROJECT_ROOT / "data"
    master_dataset_path: Path = _PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet"
    qualifying_interim_path: Path = _PROJECT_ROOT / "data" / "interim" / "qualifying.parquet"
    weather_csv_path: Path = _PROJECT_ROOT / "data" / "interim" / "race_weather.csv"
    #: Materialization horizon: the Materializer may only ever build a row
    #: for the single next race with no result yet. Not operationally
    #: adjustable today — fixed at 1 by design — but exposed as a Settings
    #: field for the same transparency reason
    #: `serve_max_year` is, not because changing it is currently supported
    #: (`next_race()`'s own horizon=1 behavior is unaffected by this value).
    materialization_horizon: int = 1
    #: Pre-race cache TTL, in seconds: "minutes, not hours" — a backstop
    #: against a missed invalidation trigger, since pre-race entries (unlike
    #: historical ones) describe a race whose inputs genuinely change over a
    #: race weekend.
    pre_race_cache_ttl_seconds: int = 300
