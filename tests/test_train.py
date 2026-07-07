"""
Tests for src/models/train.py (Phase 4 module 4 — Decision 012).

Design-doc Section 12 requirements:
  - end-to-end smoke on a tiny synthetic feature frame with MLflow pointed
    at a tmp store: runs, logs expected params/metrics/artifacts, is
    deterministic on re-run
  - test-set discipline (Section 11.3): training runs never emit test_*
    metrics; final_test tags final=true; CLI refuses --final-test/--tune/
    --register without an explicit --model
  - fold-fit containment (Section 11.4): imputer statistics are fit inside
    each fold's train window only
  - shuffled-target canary (Section 11.5): within-race-permuted winners
    collapse per-race top-1 toward chance
  - tripwire (Section 11.6): top-1 above 70% warns loudly
Plus: feature-importance frame with Decision-013 classes and missing-
indicator mapping; tune_candidate selection; model registration + alias.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.evaluate import top1_accuracy
from src.models.registry import get_model
from src.models.splits import temporal_split, to_xy
from src.models.train import (
    check_tripwire,
    feature_importance_frame,
    final_test,
    main,
    register_model,
    run_cv,
    train_candidate,
    tune_candidate,
)
from tests.conftest import set_tmp_experiment

# ---------------------------------------------------------------------------
# Synthetic feature frame: driver on pole (grid_adjusted == 1) always wins —
# a perfectly learnable signal, so models separate cleanly from chance.
# ---------------------------------------------------------------------------

def _synthetic_features(
    years, races_per_year: int = 4, n_drivers: int = 4, signal: bool = True, seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, race_id = [], 0
    for year in years:
        for rnd in range(1, races_per_year + 1):
            race_id += 1
            grid = rng.permutation(n_drivers) + 1
            for driver in range(n_drivers):
                row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
                row["grid_adjusted"] = float(grid[driver])
                row["grid_position_norm"] = float(grid[driver]) / n_drivers
                row.update({
                    "raceId": race_id, "driverId": driver + 1, "constructorId": 1,
                    "circuitId": 1, "year": year, "round": rnd,
                    TARGET_COLUMN: int(grid[driver] == 1) if signal else 0,
                })
                rows.append(row)
            if not signal:
                # a random driver wins instead
                winner_idx = len(rows) - n_drivers + int(rng.integers(n_drivers))
                rows[winner_idx][TARGET_COLUMN] = 1
    return pd.DataFrame(rows)


def _full_frame(**kwargs) -> pd.DataFrame:
    return _synthetic_features(range(2010, 2025), **kwargs)


@pytest.fixture()
def tmp_mlflow(tmp_path, monkeypatch):
    """Isolated MLflow store per test — tracking URI, artifacts, AND cwd.

    chdir must happen BEFORE the first mlflow call: the sqlite store caches
    a default artifact root of ./mlruns resolved against the cwd at store
    creation, and experiments created later inside a CLI main()
    (--experiment) inherit it. Without the chdir those artifacts leak into
    the repository / CI checkout.
    """
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    set_tmp_experiment("test-experiment", tmp_path)
    yield
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------------------------
# Smoke: train_candidate logs the expected run, metrics, artifacts
# ---------------------------------------------------------------------------

def test_train_candidate_smoke(tmp_mlflow):
    split = temporal_split(_full_frame())
    metrics = train_candidate(
        "logreg", split.train, split.val, fingerprint="test-fp", n_folds=2,
    )
    assert metrics["val_top1_accuracy"] == 1.0        # perfectly learnable signal
    assert "cv_top1_accuracy_mean" in metrics

    runs = mlflow.search_runs(search_all_experiments=True)
    parents = runs[runs["tags.stage"] == "default"]
    folds = runs[runs["tags.stage"] == "cv-fold"]
    assert len(parents) == 1 and len(folds) == 2      # 1 parent + 2 fold children
    parent = parents.iloc[0]
    assert parent["tags.data_fingerprint"] == "test-fp"
    assert parent["tags.model_family"] == "linear"
    assert parent["metrics.val_top1_accuracy"] == 1.0

    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(parent["run_id"])}
    assert "training_schema.json" in artifacts
    val_artifacts = {a.path for a in
                     mlflow.MlflowClient().list_artifacts(parent["run_id"], "val")}
    assert {"val/metrics_by_season.csv", "val/per_race_metrics.csv",
            "val/calibration_table.csv", "val/calibration_plot.png",
            "val/feature_importance.csv", "val/importance_by_class.csv"} <= val_artifacts


def test_train_candidate_deterministic(tmp_mlflow):
    split = temporal_split(_full_frame())
    m1 = train_candidate("logreg", split.train, split.val, n_folds=2)
    m2 = train_candidate("logreg", split.train, split.val, n_folds=2)
    for key in m1:
        assert m1[key] == pytest.approx(m2[key]), key


# ---------------------------------------------------------------------------
# Section 11.3 — test-set discipline
# ---------------------------------------------------------------------------

def test_training_never_emits_test_metrics(tmp_mlflow):
    split = temporal_split(_full_frame())
    train_candidate("pole_baseline", split.train, split.val, n_folds=2)
    runs = mlflow.search_runs(search_all_experiments=True)
    test_metric_cols = [c for c in runs.columns if c.startswith("metrics.test_")]
    assert not test_metric_cols


def test_final_test_is_tagged_and_scores_2024(tmp_mlflow):
    split = temporal_split(_full_frame())
    metrics = final_test("pole_baseline", split, fingerprint="test-fp")
    assert metrics["test_top1_accuracy"] == 1.0
    runs = mlflow.search_runs(search_all_experiments=True)
    final_runs = runs[runs.get("tags.final") == "true"]
    assert len(final_runs) == 1
    assert final_runs.iloc[0]["metrics.test_top1_accuracy"] == 1.0


def test_cli_refuses_guarded_actions_without_explicit_model():
    for flags in (["--final-test"], ["--tune"], ["--register", "Staging"]):
        with pytest.raises(SystemExit) as excinfo:
            main(flags)
        assert excinfo.value.code == 2   # argparse error


# ---------------------------------------------------------------------------
# Section 11.4 — fitted state is contained inside each fold
# ---------------------------------------------------------------------------

def test_fold_pipelines_fit_imputer_on_fold_train_only():
    # driver_points_last_5 == year makes fold medians differ from the full
    # median, so containment is detectable.
    train_df = _synthetic_features(range(2010, 2022)).copy()
    train_df["driver_points_last_5"] = train_df["year"].astype(float)

    fold_metrics, fold_pipelines = run_cv("logreg", train_df, n_folds=2)
    col = list(FEATURE_COLUMNS).index("driver_points_last_5")
    full_median = train_df["driver_points_last_5"].median()

    for fm, pipeline in zip(fold_metrics, fold_pipelines):
        fold_train_years = train_df[train_df["year"] < fm["val_year"]]["year"]
        expected = float(fold_train_years.median())
        fitted = float(pipeline.named_steps["imputer"].statistics_[col])
        assert fitted == pytest.approx(expected)
        assert fitted != pytest.approx(float(full_median))


# ---------------------------------------------------------------------------
# Section 11.5 — shuffled-target canary
# ---------------------------------------------------------------------------

def test_shuffled_target_canary():
    df = _full_frame()
    train_df = df[df["year"] <= 2019]
    eval_df = df[(df["year"] >= 2020) & (df["year"] <= 2021)]

    def fit_score(frame):
        X_tr, y_tr, _ = to_xy(frame)
        pipe = get_model("logreg", y_tr).fit(X_tr, y_tr)
        X_ev, y_ev, races = to_xy(eval_df)
        return top1_accuracy(y_ev, pipe.predict_proba(X_ev)[:, 1], races)

    assert fit_score(train_df) >= 0.9      # real signal is learnable

    rng = np.random.default_rng(7)
    shuffled = train_df.copy()
    shuffled[TARGET_COLUMN] = (
        shuffled.groupby("raceId")[TARGET_COLUMN]
        .transform(lambda s: s.sample(frac=1.0, random_state=rng.integers(1 << 31)).to_numpy())
    )
    assert shuffled.groupby("raceId")[TARGET_COLUMN].sum().eq(1).all()  # still 1 winner/race
    # Permuted winners: accuracy collapses toward chance (1/4 field).
    assert fit_score(shuffled) <= 0.5


# ---------------------------------------------------------------------------
# Section 11.6 — tripwire
# ---------------------------------------------------------------------------

def test_tripwire_warns_above_threshold():
    with pytest.warns(UserWarning, match="TRIPWIRE"):
        check_tripwire({"top1_accuracy": 0.75}, context="unit test")


def test_tripwire_silent_below_threshold(recwarn):
    check_tripwire({"top1_accuracy": 0.55}, context="unit test")
    assert not [w for w in recwarn if "TRIPWIRE" in str(w.message)]


# ---------------------------------------------------------------------------
# Feature importance with Decision-013 classes
# ---------------------------------------------------------------------------

def test_feature_importance_frame_boosted_trees():
    df = _full_frame()
    X, y, _ = to_xy(df)
    pipeline = get_model("lightgbm", y).fit(X, y)
    frame = feature_importance_frame(pipeline)
    assert set(frame["feature"]) == set(FEATURE_COLUMNS)      # no imputer -> 1:1
    assert set(frame["feature_class"]) <= {"stable", "era_sensitive", "experimental"}
    # The planted signal must dominate.
    assert frame.iloc[0]["feature"] in ("grid_adjusted", "grid_position_norm")


def test_feature_importance_frame_maps_missing_indicators():
    df = _full_frame()
    df.loc[df.index[::3], "q3_sec"] = np.nan
    X, y, _ = to_xy(df)
    pipeline = get_model("logreg", y).fit(X, y)
    frame = feature_importance_frame(pipeline)
    indicator = frame[frame["is_missing_indicator"]]
    assert not indicator.empty
    row = indicator[indicator["feature"] == "missingindicator_q3_sec"].iloc[0]
    assert row["feature_class"] == "experimental"             # class of q3_sec


def test_feature_importance_none_for_heuristic():
    df = _full_frame()
    X, y, _ = to_xy(df)
    pipeline = get_model("pole_baseline", y).fit(X, y)
    assert feature_importance_frame(pipeline) is None


# ---------------------------------------------------------------------------
# Tuning and registration
# ---------------------------------------------------------------------------

def test_tune_candidate_selects_and_logs(tmp_mlflow):
    split = temporal_split(_full_frame())
    best_params, best_metrics = tune_candidate(
        "logreg", split.train, n_iter=2, n_folds=2,
    )
    assert set(best_params) == {"model__C"}
    assert 0.0 <= best_metrics["cv_top1_accuracy_mean"] <= 1.0
    runs = mlflow.search_runs(search_all_experiments=True)
    assert (runs["tags.stage"] == "tune").sum() == 2


def test_tune_rejects_untunable_candidate(tmp_mlflow):
    split = temporal_split(_full_frame())
    with pytest.raises(ValueError, match="no tunable"):
        tune_candidate("pole_baseline", split.train, n_iter=2, n_folds=2)


def _features_snapshot_source(tmp_path) -> Path:
    """A tmp features.parquet for register_model's features_source param —
    tests must never let register_model default to the real project's
    data/processed/features.parquet (mirrors the bundle_root hermeticity
    discipline)."""
    path = tmp_path / "features-source.parquet"
    _full_frame().to_parquet(path, index=False)
    return path


def test_register_model_sets_alias(tmp_mlflow, tmp_path):
    split = temporal_split(_full_frame())
    version = register_model("pole_baseline", split, alias="Staging",
                             bundle_root=tmp_path / "bundle",
                             features_source=_features_snapshot_source(tmp_path),
                             artifacts_root=tmp_path / "artifacts")
    client = mlflow.MlflowClient()
    resolved = client.get_model_version_by_alias("f1-winner", "Staging")
    assert str(resolved.version) == str(version)


def test_register_rejects_unknown_alias(tmp_mlflow):
    split = temporal_split(_full_frame())
    with pytest.raises(ValueError, match="alias"):
        register_model("pole_baseline", split, alias="Canary")


def test_tune_rejects_zero_iterations(tmp_mlflow):
    split = temporal_split(_full_frame())
    with pytest.raises(ValueError, match="n_iter"):
        tune_candidate("logreg", split.train, n_iter=0, n_folds=2)


# ---------------------------------------------------------------------------
# Selected-configuration plumbing (review fix: tuned params must be able to
# reach final test and registration — design Sections 4/9)
# ---------------------------------------------------------------------------

def test_final_test_applies_and_logs_params(tmp_mlflow):
    split = temporal_split(_full_frame())
    final_test("logreg", split, params={"model__C": 0.123})
    runs = mlflow.search_runs(search_all_experiments=True)
    run = runs[runs.get("tags.final") == "true"].iloc[0]
    assert run["params.model__C"] == "0.123"


def test_register_applies_params(tmp_mlflow, tmp_path):
    split = temporal_split(_full_frame())
    version = register_model("logreg", split, alias="Staging",
                             params={"model__C": 0.123},
                             bundle_root=tmp_path / "bundle",
                             features_source=_features_snapshot_source(tmp_path),
                             artifacts_root=tmp_path / "artifacts")
    model = mlflow.sklearn.load_model(f"models:/f1-winner/{version}")
    assert model.named_steps["model"].get_params()["C"] == 0.123


def test_cli_rejects_bad_params_combinations():
    for flags in (
        ["--model", "logreg", "--tune", "--params", "{}"],       # params + tune
        ["--params", '{"model__C": 1}'],                          # params + all
        ["--model", "logreg", "--params", "not-json"],            # invalid JSON
        ["--model", "logreg", "--params", "[1, 2]"],              # non-dict JSON
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(flags)
        assert excinfo.value.code == 2


def test_data_fingerprint_format():
    features = __import__("pathlib").Path("data/processed/features.parquet")
    if not features.exists():
        pytest.skip("features.parquet not built")
    from src.models.train import data_fingerprint
    fp = data_fingerprint()
    rows, digest = fp.split("rows-")
    assert rows.isdigit() and len(digest) == 12


# ---------------------------------------------------------------------------
# CLI happy paths (python -m src.models.train)
# ---------------------------------------------------------------------------

def _patch_features(monkeypatch, tmp_path) -> None:
    """Point the CLI at a tmp synthetic features.parquet — never data/."""
    path = tmp_path / "features.parquet"
    _full_frame().to_parquet(path, index=False)
    monkeypatch.setattr("src.models.train.FEATURES_PATH", path)
    monkeypatch.setattr("src.models.train.data_fingerprint", lambda: "test-fp")


def test_cli_missing_features_parquet_returns_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "src.models.train.FEATURES_PATH", tmp_path / "missing.parquet"
    )
    assert main(["--model", "pole_baseline"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_single_model_run(tmp_mlflow, monkeypatch, tmp_path, capsys):
    _patch_features(monkeypatch, tmp_path)
    rc = main(["--model", "pole_baseline", "--n-folds", "2",
               "--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}",
               "--experiment", "cli-test"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pole_baseline" in out
    assert "val_top1=" in out


def test_cli_final_test_run(tmp_mlflow, monkeypatch, tmp_path, capsys):
    _patch_features(monkeypatch, tmp_path)
    rc = main(["--model", "pole_baseline", "--final-test",
               "--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}",
               "--experiment", "cli-test"])
    assert rc == 0
    assert "FINAL TEST pole_baseline" in capsys.readouterr().out


def test_cli_register_run(tmp_mlflow, monkeypatch, tmp_path, capsys):
    _patch_features(monkeypatch, tmp_path)
    rc = main(["--model", "logreg", "--register", "Staging",
               "--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}",
               "--experiment", "cli-test",
               "--bundle-root", str(tmp_path / "bundle"),
               "--artifacts-root", str(tmp_path / "artifacts")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Registered f1-winner v1 as @Staging" in out
    assert "Serving bundle exported to" in out
    assert "Runtime features snapshot frozen to" in out
    assert (tmp_path / "bundle" / "staging" / "manifest.json").exists()
    assert (tmp_path / "artifacts" / "features.parquet").exists()
