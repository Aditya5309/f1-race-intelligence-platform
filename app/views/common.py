"""
app/views/common.py

Shared dashboard plumbing (Decision 016; application_design.md §3).
All project logic goes through the API — the dashboard never imports src/.
"""

from __future__ import annotations

import httpx
import streamlit as st

from app.config import Settings

ERA_CAVEAT = (
    "**Reading the numbers:** the model's advantage over simply picking the "
    "pole sitter is concentrated in *dominance* seasons (e.g. 2023). In "
    "competitive seasons (2022, 2024) expect top-1 parity with the pole "
    "pick — but strong top-3 ranking and calibrated probabilities."
)

_settings = Settings()


def api_url() -> str:
    return _settings.api_url.rstrip("/")


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict:
    """GET {api_url}{path} -> parsed JSON. Raises httpx.HTTPError on failure."""
    response = httpx.get(f"{api_url()}{path}", params=params, timeout=10.0)
    response.raise_for_status()
    return response.json()


def api_get_or_stop(path: str, params: dict | None = None) -> dict:
    """api_get with a user-facing error banner instead of a traceback."""
    try:
        return api_get(path, params=params)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"API error ({exc.response.status_code}): {detail}")
        st.stop()
    except httpx.HTTPError as exc:
        st.error(
            f"Cannot reach the API at {api_url()} — is it running? "
            f"Start it with `uvicorn app.api:app`. ({exc})"
        )
        st.stop()


def sidebar_model_panel() -> dict | None:
    """Render the serving-model panel in the sidebar; returns /health JSON."""
    try:
        health = api_get("/health")
    except httpx.HTTPError:
        st.sidebar.error(f"API unreachable at {api_url()}")
        return None
    model = health.get("model")
    with st.sidebar:
        st.divider()
        if health["status"] != "ok" or not model:
            st.error(f"API degraded: {health.get('detail', 'model not loaded')}")
            return health
        st.caption("Serving model")
        st.markdown(
            f"**{model['name']}** v{model['version']} `@{model['alias']}`  \n"
            f"{model['model_class']} · calibration: `{model['calibration']}`  \n"
            f"trained {model['trained_at'][:10]}"
        )
        st.info(ERA_CAVEAT)
    return health
