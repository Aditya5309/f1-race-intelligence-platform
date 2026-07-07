"""
scripts/smoke.py — end-to-end platform smoke test (Phase B quality baseline).

    python scripts/smoke.py

Verifies the core serving workflow on a fully SYNTHETIC stack — no gitignored
data/, no project mlflow.db, no network, no ports, no external services — so
the same command runs identically on a dev machine and in a future GitHub
Actions job:

  1. project configuration     app.config.Settings constructs and exposes the
                               documented knobs
  2. MLflow train + registry   register a calibrated logreg into a THROWAWAY
                               sqlite registry built in a temp dir, which
                               ALSO exports a frozen serving bundle to a
                               temp dir (Decision 026/027)
  3. model loading             load the frozen serving bundle exported in
                               step 2 via src.models.predict.load_model — no
                               MLflow tracking URI or registry call
  4. prediction pipeline       predict_race contract: per-race normalization,
                               dense ranks, deterministic output
  5. FastAPI startup + /health in-process TestClient (lifespan runs; no
                               socket): /health ok + one full prediction
  6. Streamlit dashboard       streamlit imports, every view module imports,
                               dashboard entry script byte-compiles, AND the
                               script actually runs headless via
                               streamlit.testing.v1.AppTest (catches runtime-
                               only failures like st.navigation() duplicate
                               URL pathnames — see step docstring)

The synthetic frame is seeded (pole sitter always wins), so every run is
deterministic. Exit code 0 = all steps passed; 1 = a step failed.

NOTE: training on a perfectly learnable synthetic signal intentionally trips
the >70% top-1 leakage tripwire warning — that warning is EXPECTED here.
"""

from __future__ import annotations

import py_compile
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))          # runnable without pip install

SMOKE_RACE_ID = 202301                          # 2023 round 1, in-window


def _synthetic_features(years, races_per_year=3, n_drivers=5, seed=0) -> pd.DataFrame:
    """Seeded feature frame; the pole sitter (grid_adjusted == 1) always wins.

    Mirrors the builder used across tests/test_{api,predict,train}.py.
    """
    from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN

    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for rnd in range(1, races_per_year + 1):
            race_id = year * 100 + rnd
            grid = rng.permutation(n_drivers) + 1
            for driver in range(n_drivers):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row["grid_adjusted"] = float(grid[driver])
                row["grid_position_norm"] = float(grid[driver]) / n_drivers
                row.update({
                    "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                    "circuitId": 1, "year": year, "round": rnd,
                    TARGET_COLUMN: int(grid[driver] == 1),
                })
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Steps — each takes the shared context dict and raises on failure
# ---------------------------------------------------------------------------

def step_configuration(ctx: dict) -> None:
    """Settings constructs; the documented serving knobs exist and are typed."""
    from app.config import Settings

    defaults = Settings()                       # env/.env may override values
    assert isinstance(defaults.serve_max_year, int)
    assert isinstance(defaults.features_path, Path)
    assert isinstance(defaults.serving_bundle_path, Path)

    # The smoke stack overrides everything explicitly — host env cannot leak in.
    ctx["settings"] = Settings(
        serving_bundle_path=ctx["bundle_dir"],
        features_path=ctx["features_path"],
        data_dir=ctx["tmp_dir"] / "no-such-dir",   # display names degrade to null
        debug_endpoints=False,
    )


def step_mlflow_train_and_register(ctx: dict) -> None:
    """Train + calibrate + register into the throwaway registry (MLflow up),
    which ALSO exports a frozen serving bundle (Decision 026/027) — the
    thing step 3 actually loads, with no MLflow call at all."""
    import mlflow

    from src.models.splits import temporal_split
    from src.models.train import register_model

    frame = _synthetic_features(range(2010, 2025))
    frame.to_parquet(ctx["features_path"], index=False)

    mlflow.set_tracking_uri(ctx["tracking_uri"])
    # Create the experiment with its artifact store INSIDE the temp dir — the
    # default is ./mlruns relative to the CWD, which would leak run artifacts
    # into the repository / CI checkout.
    experiment_id = mlflow.create_experiment(
        "smoke-test", artifact_location=(ctx["tmp_dir"] / "mlartifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    version = register_model(
        "logreg", temporal_split(frame), alias="Staging", calibrate=True,
        bundle_root=ctx["bundle_root"],
    )
    mlflow.set_tracking_uri(None)
    assert version == "1", f"expected fresh registry version 1, got {version}"
    assert ctx["bundle_dir"].exists(), "register_model did not export a bundle"
    ctx["frame"] = frame


def step_model_loading(ctx: dict) -> None:
    """Loads the frozen bundle exported in step 2 — no MLflow tracking URI,
    no registry call (Decision 026/027)."""
    from src.models.predict import load_model

    model, info = load_model(ctx["bundle_dir"])
    assert info.name == "f1-winner"
    assert info.alias == "Staging"
    assert info.calibration == "isotonic-oof"
    assert info.model_class == "CalibratedModel"
    ctx["model"] = model


def step_prediction_pipeline(ctx: dict) -> None:
    """predict_race contract: normalization, dense ranks, determinism."""
    from src.models.predict import predict_race

    race = ctx["frame"][ctx["frame"]["raceId"] == SMOKE_RACE_ID]
    out = predict_race(ctx["model"], race)
    assert len(out) == len(race)
    assert abs(out["win_probability"].sum() - 1.0) < 1e-9
    assert sorted(out["predicted_rank"]) == list(range(1, len(race) + 1))
    pd.testing.assert_frame_equal(out, predict_race(ctx["model"], race))


def step_api_health_and_prediction(ctx: dict) -> None:
    """create_app lifespan + /health + one full prediction, all in-process."""
    from fastapi.testclient import TestClient

    from app.api import create_app

    with TestClient(create_app(ctx["settings"])) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        body = health.json()
        assert body["status"] == "ok", f"API degraded: {body.get('detail')}"
        assert body["model"]["name"] == "f1-winner"

        response = client.get(f"/predictions/{SMOKE_RACE_ID}")
        assert response.status_code == 200, response.text
        predictions = response.json()["predictions"]
        assert len(predictions) == 5
        total = sum(p["win_probability"] for p in predictions)
        assert abs(total - 1.0) < 1e-6
        assert predictions[0]["predicted_rank"] == 1


def step_streamlit_dashboard(ctx: dict) -> None:
    """streamlit + every view module import; the entry script actually runs.

    Import/compile checks alone are NOT enough: app/dashboard.py's
    st.navigation([...]) call only validates that page URL pathnames are
    unique when it detects a live Streamlit ScriptRunContext
    (streamlit/commands/navigation.py::_navigation — see the `if not ctx:
    return default_page` early-out). Outside a real script run (e.g. a bare
    `import` or `py_compile`), that check silently never runs. This is
    exactly how a real bug shipped invisibly: the pre-redesign Overview,
    Predictions, and Model Insights pages all named their entry point
    `render`, so without an explicit `url_path` they collided on the same
    inferred pathname — `st.navigation()` raised StreamlitAPIException the
    moment a user opened the dashboard in a browser, while every existing
    check here (import, compile) stayed green. Fixed by giving each st.Page
    an explicit url_path (app/dashboard.py) — kept for all five pages in the
    UI/UX redesign (home/race_center/driver_explorer/season_analytics/insights).

    streamlit.testing.v1.AppTest runs the script inside a real, headless
    ScriptRunContext (no browser, no socket) — the only way to catch this
    class of bug here. It needs no live API: every page's
    sidebar_model_panel() catches httpx errors and degrades gracefully
    (app/views/common.py), so this step stays fully offline like every
    other step in this file.
    """
    import importlib

    import streamlit
    from streamlit.testing.v1 import AppTest

    assert streamlit.__version__
    for module in ("app.views.common", "app.views.components",
                   "app.views.metadata", "app.views.charts", "app.views.home",
                   "app.views.race_center", "app.views.driver_explorer",
                   "app.views.season_analytics", "app.views.insights"):
        importlib.import_module(module)
    dashboard_path = _PROJECT_ROOT / "app" / "dashboard.py"
    py_compile.compile(str(dashboard_path), doraise=True)

    at = AppTest.from_file(str(dashboard_path), default_timeout=15).run()
    assert not at.exception, (
        "dashboard raised on headless run: "
        f"{[e.value for e in at.exception]}"
    )


STEPS = [
    ("project configuration", step_configuration),
    ("MLflow training + registry", step_mlflow_train_and_register),
    ("model loading (frozen bundle)", step_model_loading),
    ("prediction pipeline contract", step_prediction_pipeline),
    ("FastAPI startup + /health + prediction", step_api_health_and_prediction),
    ("Streamlit dashboard imports", step_streamlit_dashboard),
]


def main() -> int:
    print("=== F1 platform smoke test (synthetic stack; no data/ required) ===\n")
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        bundle_root = tmp_dir / "bundle"
        ctx: dict = {
            "tmp_dir": tmp_dir,
            "tracking_uri": f"sqlite:///{tmp_dir / 'mlflow.db'}",
            "features_path": tmp_dir / "features.parquet",
            # register_model (Decision 026/027) exports a bundle here; this
            # path is explicit so nothing ever touches the real project's
            # models/serving/ during a smoke run.
            "bundle_root": bundle_root,
            "bundle_dir": bundle_root / "staging",
        }
        for i, (name, step) in enumerate(STEPS, start=1):
            t0 = time.perf_counter()
            try:
                step(ctx)
            except Exception:
                print(f"[{i}/{len(STEPS)}] {name:<40} FAIL")
                traceback.print_exc()
                return 1
            print(f"[{i}/{len(STEPS)}] {name:<40} PASS "
                  f"({time.perf_counter() - t0:.1f}s)")

    print(f"\nSMOKE PASSED — {len(STEPS)}/{len(STEPS)} steps "
          f"in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
