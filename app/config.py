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

Decision 029: both `serving_bundle_path` and `features_path` default under
`artifacts/` — a runtime artifact tree that is committed to git (contrast
with `data_dir`, `data/`, `mlruns/`, `mlflow.db`, all gitignored). A freshly
cloned repository (source + `artifacts/` only, no `data/`) has everything
the deployed API requires to serve predictions.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS_ROOT = _PROJECT_ROOT / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="F1_", env_file=".env", extra="ignore",
    )

    # --- model --------------------------------------------------------------
    #: Frozen serving bundle directory (Decision 026/027/029) — no MLflow
    #: tracking URI or registry alias needed at request time. Committed to
    #: git under artifacts/ (unlike the gitignored data/ and mlruns/ trees).
    serving_bundle_path: Path = _ARTIFACTS_ROOT / "serving" / "staging"

    # --- data --------------------------------------------------------------
    #: Frozen serving feature matrix (Decision 029) — a committed snapshot
    #: exported by train.py::register_model(), NOT the live training-side
    #: data/processed/features.parquet (which stays gitignored).
    features_path: Path = _ARTIFACTS_ROOT / "features.parquet"
    #: Directory holding the display-name/metadata CSVs (drivers, races,
    #: standings, etc.) — an OPTIONAL enrichment only (Decision 016): names
    #: degrade to a fallback string if the files are absent, never a hard
    #: requirement to serve predictions. Defaults to the committed frozen
    #: snapshot (artifacts/display/, scripts/export_display_data.py) rather
    #: than the gitignored data/ tree, so a fresh clone (Streamlit Cloud,
    #: Render, CI) resolves real names out of the box — mirrors how
    #: features_path/serving_bundle_path already default under artifacts/.
    #: Point at the full data/ tree locally via .env (F1_DATA_DIR=data) if
    #: wanted; nothing requires it.
    data_dir: Path = _ARTIFACTS_ROOT / "display"

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
