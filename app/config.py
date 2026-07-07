"""
app/config.py

Application-layer configuration (Decision 016; application_design.md §11).

Every knob the API or dashboard reads comes through this Settings class —
environment variables prefixed `F1_` (e.g. F1_SERVING_BUNDLE_PATH=...), with
`.env`-file support for local development (.env is gitignored). No hardcoded
paths in app code (project guiding principle): the first local run and a
future container run use identical code, different environments.

Decision 026/027: serving no longer resolves a live MLflow registry alias —
there is deliberately no `tracking_uri`/`model_alias` setting here anymore.
`serving_bundle_path` points at a frozen bundle (src.models.serving_bundle);
the API doesn't know what alias or experiment produced it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="F1_", env_file=".env", extra="ignore",
    )

    # --- model --------------------------------------------------------------
    #: Frozen serving bundle directory (Decision 026/027) — no MLflow
    #: tracking URI or registry alias needed at request time.
    serving_bundle_path: Path = _PROJECT_ROOT / "models" / "serving" / "staging"

    # --- data --------------------------------------------------------------
    features_path: Path = _PROJECT_ROOT / "data" / "processed" / "features.parquet"
    #: Directory holding drivers.csv / constructors.csv for display names.
    #: Names degrade to null in responses if the files are absent.
    data_dir: Path = _PROJECT_ROOT / "data"

    # --- serving policy ----------------------------------------------------
    #: Forward-holdout guard (application_design.md §5.1): races after this
    #: year return 409. Raise deliberately in Phase 8, never by accident.
    serve_max_year: int = 2024
    #: Enables GET /debug/* (development only — keep false in production).
    debug_endpoints: bool = False
    #: Bounded per-race prediction cache, keyed (model_version, race_id).
    prediction_cache_size: int = 512

    # --- ops ----------------------------------------------------------------
    log_level: str = "INFO"
    #: Dashboard-side: where the API lives.
    api_url: str = "http://localhost:8000"
