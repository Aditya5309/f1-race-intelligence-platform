"""
Behavioral tests for app/views/race_center.py's Phase 8 upcoming-race
integration, via streamlit.testing.v1.AppTest (real headless script
execution, no live API — every call is monkeypatched at
app.views.common.api_get/predict_upcoming, the same two chokepoints every
call site in this dashboard already funnels through).

app/views/*.py has 0% *unit* coverage by design elsewhere in this project
(presentation-only HTTP consumers, normally exercised only by
scripts/smoke.py's offline AppTest + manual verification). This file is a
deliberate, narrow exception: Phase 8 added real BRANCHING logic to
race_center.py (is this the upcoming race?) that the "presentation only"
rationale doesn't fully cover, and the single most important guarantee
this phase promised — historical rendering stays byte-for-byte unchanged,
and gracefully falls back to it when the new endpoint is unavailable — is
only checkable by actually running the page.

Covers:
  - the degraded/fallback case (GET /races/upcoming unavailable) renders
    exactly like a historical-only Race Center — no exception, no banner,
    no picker entry for a race that doesn't exist
  - the upcoming-race case renders without exception, shows the
    materialization-status banner + caveats + confidence note + provenance
    expander, and skips Qualifying Impact / Grid Simulator with an
    explanatory caption instead of failing
  - a historical race, with an upcoming race ALSO available, still renders
    Qualifying Impact / Grid Simulator (proving the branch is race-specific,
    not "disable whenever an upcoming race exists")
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.views import common

_APP_SCRIPT = Path(__file__).parent / "_fixtures" / "race_center_app.py"

_MODEL = {
    "name": "f1-winner", "version": "1", "alias": "Staging", "run_id": "r1",
    "trained_at": "2026-01-01T00:00:00", "calibration": "isotonic-oof",
    "model_class": "CalibratedModel",
}
_PRED_A = {"driver_id": 1, "driver_name": "Driver One", "constructor_id": 10,
           "constructor_name": "Team A", "predicted_rank": 1,
           "win_probability": 0.6, "win_probability_raw": 0.6}
_PRED_B = {"driver_id": 2, "driver_name": "Driver Two", "constructor_id": 20,
           "constructor_name": "Team B", "predicted_rank": 2,
           "win_probability": 0.4, "win_probability_raw": 0.4}

_HISTORICAL_RACE_ID = 100
_UPCOMING_RACE_ID = 101

_HISTORICAL_PREDICTION = {
    "prediction_id": "p1", "race_id": _HISTORICAL_RACE_ID, "year": 2026, "round": 1,
    "generated_at": "2026-01-02T00:00:00", "model": _MODEL,
    "predictions": [_PRED_A, _PRED_B],
    "actual_winner_driver_id": 1, "model_top1_hit": True,
}
_BASELINE_PRED_A = {**_PRED_A, "win_probability": 1.0, "win_probability_raw": 1.0}
_BASELINE_PRED_B = {**_PRED_B, "win_probability": 0.0, "win_probability_raw": 0.0}
_VS_BASELINE = {
    # Deliberately DIFFERENT win shares from _PRED_A/_PRED_B above (matching
    # how the real pole-only baseline actually behaves: 100%/0%) -- not
    # just realism, a byte-identical model_frame/baseline_frame plotly
    # figure pair trips Streamlit's own auto-generated-element-ID collision
    # detection (unrelated to Phase 8; a real render always differs here).
    "race_id": _HISTORICAL_RACE_ID, "year": 2026, "round": 1, "model": _MODEL,
    "baseline_name": "pole_baseline", "baseline_description": "grid-only heuristic",
    "model_predictions": [_PRED_A, _PRED_B],
    "baseline_predictions": [_BASELINE_PRED_A, _BASELINE_PRED_B],
    "actual_winner_driver_id": 1, "model_top1_hit": True, "baseline_top1_hit": True,
}
_SIMULATE = {
    "race_id": _HISTORICAL_RACE_ID, "driver_id": 1, "driver_name": "Driver One",
    "field_size": 2, "real_grid_position": 1.0, "simulated_grid_position": 1.0,
    "pit_lane_start": False, "real_win_probability": 0.6, "simulated_win_probability": 0.6,
    "field": [_PRED_A, _PRED_B], "locked_qualifying_features": [], "locked_features": [],
    "model": _MODEL,
}
_UPCOMING_IDENTITY = {
    "race_id": _UPCOMING_RACE_ID, "year": 2026, "round": 2,
    "name": "Upcoming GP", "circuit_id": 1, "date": "2026-02-01",
}
_UPCOMING_PREDICTION = {
    "prediction_id": "p2", "year": 2026, "round": 2,
    "materialization_status": "pre_qualifying",
    "missing_inputs": ["qualifying_position"],
    "generated_at": "2026-01-02T00:00:00", "model": _MODEL,
    "predictions": [_PRED_A, _PRED_B],
    "caveats": ["Grid-derived features use the interim qualifying-position proxy."],
    "provenance": {
        "model_version": "1", "model_alias": "Staging",
        "feature_schema_version": "abc123",
        "etl_snapshot_version": "2026-01-01T00:00:00+00:00",
        "data_as_of": "2026-01-01T00:00:00+00:00",
        "materialized_at": "2026-01-01T00:00:01+00:00",
        "predicted_at": "2026-01-01T00:00:02+00:00",
        "qualifying_status": "not_started", "completeness_status": "pre_qualifying",
    },
}


def _not_found(path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"http://testserver/api/v1{path}")
    response = httpx.Response(404, request=request)
    return httpx.HTTPStatusError("not found", request=request, response=response)


def _make_api_get(*, upcoming_available: bool):
    def _fake_api_get(path: str, params: dict | None = None) -> dict:
        if path == "/health":
            return {"status": "ok", "api_version": "1.0.0", "model": _MODEL}
        if path == "/races":
            year = params.get("year") if params else None
            races = [{"race_id": _HISTORICAL_RACE_ID, "year": 2026, "round": 1, "n_drivers": 2}]
            if year is not None:
                races = [r for r in races if r["year"] == year]
            return {"races": races}
        if path == "/races/upcoming":
            if not upcoming_available:
                raise _not_found(path)
            return _UPCOMING_IDENTITY
        if path == f"/predictions/{_HISTORICAL_RACE_ID}":
            return _HISTORICAL_PREDICTION
        if path == f"/predictions/{_HISTORICAL_RACE_ID}/vs-baseline":
            return _VS_BASELINE
        if path.startswith(f"/predictions/{_HISTORICAL_RACE_ID}/simulate/"):
            return _SIMULATE
        raise AssertionError(f"unexpected api_get path in this test: {path}")
    return _fake_api_get


def _fake_predict_upcoming(year: int, round_: int) -> dict:
    assert (year, round_) == (2026, 2)
    return _UPCOMING_PREDICTION


@pytest.fixture(autouse=True)
def _clear_caches():
    for fn in (common.api_get, common.predict_upcoming, common.list_races,
               common.upcoming_race, common.season_predictions):
        fn.clear()
    yield


def test_fallback_when_upcoming_endpoint_unavailable(monkeypatch):
    """GET /races/upcoming unavailable (404) -> Race Center renders exactly
    like historical-only, no exception, no upcoming-race artifacts."""
    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=False))
    monkeypatch.setattr(common, "predict_upcoming", _fake_predict_upcoming)

    at = AppTest.from_file(str(_APP_SCRIPT), default_timeout=15).run()

    assert not at.exception
    assert not at.warning
    assert not at.info
    assert not any("Upcoming GP" in b.value for b in at.markdown if hasattr(b, "value"))
    assert any("Qualifying Impact" in s.value for s in at.subheader)


def test_upcoming_race_renders_status_banner_and_skips_historical_only_sections(monkeypatch):
    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=True))
    monkeypatch.setattr(common, "predict_upcoming", _fake_predict_upcoming)

    at = AppTest.from_file(str(_APP_SCRIPT), default_timeout=15)
    at.session_state["race_id"] = _UPCOMING_RACE_ID
    at.run()

    assert not at.exception
    assert any("Provisional" in w.value for w in at.warning)
    assert any("Data as of" in c.value for c in at.caption)
    assert any(_UPCOMING_PREDICTION["caveats"][0] in c.value for c in at.caption)
    assert any("qualifying not yet complete" in c.value for c in at.caption)
    assert any("check back after this one runs" in c.value for c in at.caption)
    assert any(e.label == "🔎 Prediction provenance" for e in at.expander)
    # Qualifying Impact / Grid Simulator subheaders must NOT render for the
    # upcoming race — the explanatory caption above replaces them entirely.
    assert not any("Qualifying Impact" in s.value for s in at.subheader)
    assert not any("Grid Position Simulator" in s.value for s in at.subheader)


def test_historical_race_still_renders_qualifying_impact_and_simulator_when_upcoming_exists(monkeypatch):
    """The branch is race-specific, not "disable whenever an upcoming race
    is resolvable" -- selecting a HISTORICAL race must still show both
    sections even though GET /races/upcoming resolves successfully."""
    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=True))
    monkeypatch.setattr(common, "predict_upcoming", _fake_predict_upcoming)

    at = AppTest.from_file(str(_APP_SCRIPT), default_timeout=15)
    at.session_state["race_id"] = _HISTORICAL_RACE_ID
    at.run()

    assert not at.exception
    assert not at.warning
    assert any("Qualifying Impact" in s.value for s in at.subheader)
    assert any("Grid Position Simulator" in s.value for s in at.subheader)


def _race_selectbox(at):
    return next(sb for sb in at.selectbox if sb.label == "Race")


def test_picker_shows_upcoming_race_only_when_discovery_succeeds(monkeypatch):
    """Picker shows the upcoming race only when GET /races/upcoming
    resolves -- and historical entries are unaffected either way (same
    count, same labels) when it doesn't."""
    monkeypatch.setattr(common, "predict_upcoming", _fake_predict_upcoming)

    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=True))
    at = AppTest.from_file(str(_APP_SCRIPT), default_timeout=15)
    at.session_state["race_id"] = _HISTORICAL_RACE_ID
    at.run()
    assert not at.exception
    race_options = _race_selectbox(at).options
    assert len(race_options) == 2
    assert any("(upcoming)" in o for o in race_options)

    # Discovery now fails (e.g. this deployment has no data/ tree) -- the
    # picker must fall back to exactly the historical-only entries: same
    # count, same labels, no "(upcoming)" anywhere.
    common.upcoming_race.clear()
    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=False))
    at2 = AppTest.from_file(str(_APP_SCRIPT), default_timeout=15)
    at2.session_state["race_id"] = _HISTORICAL_RACE_ID
    at2.run()
    assert not at2.exception
    race_options_fallback = _race_selectbox(at2).options
    assert len(race_options_fallback) == 1
    assert not any("(upcoming)" in o for o in race_options_fallback)
    assert race_options_fallback == [o for o in race_options if "(upcoming)" not in o]


def test_prev_next_boundaries_including_when_upcoming_race_disappears(monkeypatch):
    """Prev/Next behave correctly at both ends of the picker, AND when the
    transient upcoming slot vanishes between reruns (the stale-selection
    fix, exercised here via the Prev/Next-adjacent path rather than a
    freshly-seeded session_state)."""
    monkeypatch.setattr(common, "predict_upcoming", _fake_predict_upcoming)
    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=True))

    at = AppTest.from_file(str(_APP_SCRIPT), default_timeout=15)
    at.session_state["race_id"] = _HISTORICAL_RACE_ID
    at.run()
    assert not at.exception

    def _buttons(app_test):
        prev_btn = next(b for b in app_test.button if b.label == "◀ Prev")
        next_btn = next(b for b in app_test.button if b.label == "Next ▶")
        return prev_btn, next_btn

    # At the first (historical) position: Prev disabled, Next enabled.
    prev_btn, next_btn = _buttons(at)
    assert prev_btn.disabled is True
    assert next_btn.disabled is False

    # Next -> lands on the upcoming race (session_state updates within
    # this same run). The clicked button's OWN `disabled=` param is fixed
    # at instantiation time, BEFORE that update -- pre-existing Streamlit
    # behavior, not a Phase 8 change -- so a settle-run with no new
    # interaction is needed before the boundary state re-reads correctly.
    next_btn.click().run()
    assert not at.exception
    assert at.session_state["race_id"] == _UPCOMING_RACE_ID
    at.run()
    prev_btn, next_btn = _buttons(at)
    assert prev_btn.disabled is False
    assert next_btn.disabled is True

    # Prev -> back to the historical race; boundary state matches the
    # very first assertion above (round-trip, nothing drifted).
    prev_btn.click().run()
    assert not at.exception
    assert at.session_state["race_id"] == _HISTORICAL_RACE_ID
    at.run()
    prev_btn, next_btn = _buttons(at)
    assert prev_btn.disabled is True
    assert next_btn.disabled is False

    # Now go back to the upcoming race, then simulate it DISAPPEARING
    # (finished, or resolved to a different race) between reruns, with NO
    # user interaction of its own -- the stale-selection fix must reset to
    # the historical default and the boundary must reflect the NEW
    # (now single-entry) ordering, not crash.
    next_btn.click().run()
    assert at.session_state["race_id"] == _UPCOMING_RACE_ID

    common.upcoming_race.clear()
    monkeypatch.setattr(common, "api_get", _make_api_get(upcoming_available=False))
    at.run()
    assert not at.exception
    assert at.session_state["race_id"] == _HISTORICAL_RACE_ID
    prev_btn, next_btn = _buttons(at)
    assert prev_btn.disabled is True
    assert next_btn.disabled is True
