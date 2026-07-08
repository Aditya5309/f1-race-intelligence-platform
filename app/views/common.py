"""
app/views/common.py

Shared dashboard plumbing (Decision 016; application_design.md §3).
All project logic goes through the API — the dashboard never imports src/.
"""

from __future__ import annotations

import logging
import os

import httpx
import pandas as pd
import streamlit as st

from app.config import Settings

_log = logging.getLogger(__name__)

ERA_CAVEAT = (
    "**Reading the numbers:** the model's advantage over simply picking the "
    "pole sitter is concentrated in *dominance* seasons (e.g. 2023). In "
    "competitive seasons (2022, 2024) expect top-1 parity with the pole "
    "pick — but strong top-3 ranking and calibrated probabilities."
)
ERA_CAVEAT_SHORT = (
    "Edge is biggest in dominance seasons (e.g. 2023) — competitive "
    "seasons run closer to the pole pick."
)

_settings = Settings()

# --- TEMPORARY DIAGNOSTICS (remove after root-causing "API unreachable") ---
_API_URL_SOURCE = "environment" if os.environ.get("F1_API_URL") is not None else "default"
_log.warning(
    "DIAG api_url resolved to %r (source=%s)", _settings.api_url, _API_URL_SOURCE
)
# --- end temporary diagnostics ---


def api_url() -> str:
    return _settings.api_url.rstrip("/")


@st.cache_resource
def _http_client() -> httpx.Client:
    """One pooled connection per Streamlit process instead of the bare
    httpx.get() convenience function opening (and TLS/TCP-handshaking) a
    fresh client per call. Measured impact on this stack: ~2.2s/request with
    a fresh client vs ~0.1s/request pooled — a batch fetch across a 24-race
    season (season_predictions) went from ~55s to ~2s. st.cache_resource is
    the correct Streamlit primitive for a shared, non-serializable resource
    (as opposed to st.cache_data, used below for the serializable responses).
    """
    return httpx.Client(base_url=api_url(), timeout=10.0)


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict:
    """GET {api_url}{path} -> parsed JSON. Raises httpx.HTTPError on failure."""
    requested_url = f"{api_url()}{path}"
    # --- TEMPORARY DIAGNOSTICS (remove after root-causing "API unreachable") ---
    try:
        response = _http_client().get(path, params=params)
    except Exception as exc:
        st.sidebar.error(
            f"DIAG exception_type={type(exc).__name__!r} "
            f"message={exc!r} requested_url={requested_url!r}"
        )
        raise
    st.sidebar.caption(
        f"DIAG status={response.status_code} body={response.text[:500]!r} "
        f"requested_url={requested_url!r}"
    )
    # --- end temporary diagnostics ---
    response.raise_for_status()
    return response.json()


def _or_stop(fetch):
    """Run a zero-arg fetch callable; on API failure, show a user-facing
    error banner and st.stop() instead of a traceback. Shared by every
    *_or_stop() wrapper below so the error copy stays in one place."""
    try:
        return fetch()
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


def api_get_or_stop(path: str, params: dict | None = None) -> dict:
    """api_get with a user-facing error banner instead of a traceback."""
    return _or_stop(lambda: api_get(path, params=params))


@st.cache_data(ttl=300, show_spinner=False)
def list_races(year: int | None = None) -> list[dict]:
    """/races, optionally filtered to one season. Raises httpx.HTTPError."""
    params = {"year": year} if year is not None else None
    return api_get("/races", params=params)["races"]


def list_races_or_stop(year: int | None = None) -> list[dict]:
    return _or_stop(lambda: list_races(year))


def list_races_or_empty(year: int | None = None) -> list[dict]:
    """Soft-fail variant for status displays that must render even when the
    API is unreachable (e.g. the Dashboard landing page showing "Degraded")."""
    try:
        return list_races(year)
    except httpx.HTTPError:
        return []


def served_seasons(races: list[dict]) -> tuple[int, int] | None:
    """(min year, max year) covered by a /races listing, or None if empty."""
    if not races:
        return None
    years = [r["year"] for r in races]
    return min(years), max(years)


@st.cache_data(ttl=300, show_spinner=False)
def season_predictions(year: int) -> pd.DataFrame:
    """Every driver-race prediction row for one season, flattened to one
    DataFrame (one row per driver per race). Built entirely from the
    existing GET /races + GET /predictions/{race_id} contracts — no new
    endpoint — so Season Analytics and Driver Explorer share one fetch path
    instead of duplicating the aggregation loop. Cached per season: the
    first page to touch a season pays the N-races round trip, everyone
    after (any page, same session or not) reads the cached frame for 5 min.
    Raises httpx.HTTPError if the API is unreachable.
    """
    rows = []
    for race in list_races(year):
        body = api_get(f"/predictions/{race['race_id']}")
        for pred in body["predictions"]:
            rows.append({
                **pred,
                "race_id": body["race_id"],
                "year": body["year"],
                "round": body["round"],
                "actual_winner_driver_id": body["actual_winner_driver_id"],
                "model_top1_hit": body["model_top1_hit"],
            })
    return pd.DataFrame(rows)


def season_predictions_or_stop(year: int) -> pd.DataFrame:
    return _or_stop(lambda: season_predictions(year))


def career_predictions_or_stop(years: list[int]) -> pd.DataFrame:
    """Concatenated season_predictions() across every given season, with a
    progress indicator — the "full career" scope in Driver Explorer. Each
    season is cached individually (season_predictions), so only the first
    visit pays the full N-seasons round trip."""
    if not years:
        return pd.DataFrame()
    progress = st.progress(0.0, text="Loading career history…")
    frames = []
    for i, year in enumerate(years, start=1):
        frames.append(season_predictions_or_stop(year))
        progress.progress(i / len(years), text=f"Loaded {year} ({i}/{len(years)})")
    progress.empty()
    return pd.concat(frames, ignore_index=True)


def sidebar_model_panel() -> dict | None:
    """Compact serving-status panel shown in the sidebar on every page.

    User-facing pages get a status badge + one-line caveat; the technical
    detail (calibration method, run id, model class) lives only on the
    Model Insights advanced page — see app/views/insights.py.
    """
    from app.views.components import status_badge  # local: avoid import cycle

    # --- TEMPORARY DIAGNOSTICS (remove after root-causing "API unreachable") ---
    with st.sidebar:
        st.caption(f"DIAG api_url={_settings.api_url!r} source={_API_URL_SOURCE}")
    # --- end temporary diagnostics ---

    try:
        health = api_get("/health")
    except httpx.HTTPError:
        with st.sidebar:
            st.divider()
            status_badge(False, bad_text="API unreachable")
        return None

    model = health.get("model")
    with st.sidebar:
        st.divider()
        if health["status"] != "ok" or not model:
            status_badge(False, bad_text=health.get("detail") or "Model not loaded")
            return health
        status_badge(True)
        st.caption(f"**{model['name']}** v{model['version']} · serving `@{model['alias']}`")
        st.caption(f"🏁 {ERA_CAVEAT_SHORT}")
    return health
