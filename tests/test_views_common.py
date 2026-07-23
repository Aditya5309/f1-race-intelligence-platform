"""
Tests for app/views/common.py's upcoming_race()/upcoming_race_or_none() --
the shared helpers every dashboard view uses to fetch the upcoming race's
identity from GET /races/upcoming.

app/views/*.py has 0% unit coverage by design elsewhere in this project —
presentation-only HTTP consumers, exercised via scripts/smoke.py's headless
AppTest instead (see that file's own docstring). This one function is a
narrow, deliberate exception: it's new, pure (no Streamlit rendering),
directly mockable logic with a real soft-fail branch worth verifying
directly, not a rendering concern.
"""

from __future__ import annotations

import httpx
import pytest

from app.views import common


@pytest.fixture(autouse=True)
def _clear_cache():
    """st.cache_data persists across tests in-process; clear it so each
    test's monkeypatched api_get is actually the one that runs."""
    common.upcoming_race.clear()
    yield
    common.upcoming_race.clear()


def test_upcoming_race_returns_identity_on_success(monkeypatch):
    payload = {"race_id": 2, "year": 2026, "round": 2, "name": "R2",
               "circuit_id": 1, "date": "2026-01-08"}
    monkeypatch.setattr(common, "api_get", lambda path, params=None: payload)

    assert common.upcoming_race_or_none() == payload


def test_upcoming_race_returns_none_on_http_error(monkeypatch):
    def _boom(path, params=None):
        raise httpx.ConnectError("unreachable")
    monkeypatch.setattr(common, "api_get", _boom)

    assert common.upcoming_race_or_none() is None


def test_upcoming_race_returns_none_on_404_status_error(monkeypatch):
    def _boom(path, params=None):
        request = httpx.Request("GET", "http://testserver/api/v1/races/upcoming")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)
    monkeypatch.setattr(common, "api_get", _boom)

    assert common.upcoming_race_or_none() is None
