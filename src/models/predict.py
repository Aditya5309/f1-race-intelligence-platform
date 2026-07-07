"""
src/models/predict.py

Inference for Phase 4/5 (Decision 012 module 5; design Section 2; Decision
026/027) — the serving contract app/api.py will call.

    python -m src.models.predict --race-id 1101                     # frozen bundle (default)
    python -m src.models.predict --race-id 1101 --bundle-dir PATH    # explicit bundle
    python -m src.models.predict --race-id 1101 --tracking-uri URI   # live registry (dev only)

Responsibilities:
- `load_model(bundle_dir)` — load a FROZEN SERVING BUNDLE (see
  src.models.serving_bundle) from a local directory and return
  (model, ModelInfo). No MLflow tracking URI, no registry client, no
  network — this is what app/api.py calls, and it has no concept of
  experiments or aliases (Decision 026/027). `load_from_registry(alias,
  tracking_uri)` is kept alongside it for ad-hoc dev/CLI use directly
  against a live MLflow registry — app/api.py never calls it.
- `predict_race(model, race_df)` — score one or more races' fields and
  return per-race SUM-NORMALIZED win probabilities sorted descending
  (design Section 6: normalization is monotone within a race, so it never
  changes top-1/top-3; the normalized number is the user-facing "share of
  win chance").

Schema discipline (design Section 11.1): the design matrix is built from
THE ARTIFACT'S OWN stored schema (`registry.training_schema`, recorded by
ColumnGuard at fit time) — not from repository constants — so a model
trained on an older FEATURE_COLUMNS keeps validating input against what it
actually saw. The ColumnGuard inside the loaded pipeline then re-asserts
names/order and casts dtypes on every call; anything non-numeric raises.

Model-agnostic by construction: everything the module needs from the
artifact is predict_proba + the guard's recorded schema, both shared by all
zoo pipelines and the CalibratedModel wrapper (Decision 015).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.registry import training_schema
from src.models.serving_bundle import ModelInfo, bundle_dir_for_alias, load_bundle

DEFAULT_ALIAS = "Staging"
#: Identifier columns carried through to the prediction output when present.
CARRIED_ID_COLUMNS = ("raceId", "driverId", "constructorId", "year", "round")


def load_model(bundle_dir: Path | str | None = None):
    """Load the frozen serving bundle FastAPI depends on (Decision 026/027).

    A plain local-filesystem read — no MLflow tracking URI, no registry
    client, no network. Raises FileNotFoundError if the bundle is missing
    (app/api.py's degraded-start lifespan catches this, same as it caught a
    missing registry entry before).

    bundle_dir defaults to models/serving/staging (DEFAULT_ALIAS).
    """
    if bundle_dir is None:
        bundle_dir = bundle_dir_for_alias(DEFAULT_ALIAS)
    return load_bundle(bundle_dir)


def load_from_registry(alias: str = DEFAULT_ALIAS, tracking_uri: str | None = None,
                       name: str | None = None):
    """Resolve `name@alias` directly against a LIVE MLflow Model Registry.

    Dev/CLI convenience only (e.g. scoring a freshly-tuned model before it's
    been exported as a bundle) — app/api.py never calls this; it always
    uses load_model()'s frozen bundle instead (Decision 026/027). Imports of
    mlflow and train.py's constants are deferred here so the serving path
    (load_model/predict_race) never needs them.

    Returns (model, ModelInfo). Raises MlflowException if the model or alias
    does not exist (e.g. Production before anything was promoted).
    """
    from datetime import UTC, datetime

    import mlflow

    from src.models.train import DEFAULT_TRACKING_URI, REGISTERED_MODEL_NAME

    tracking_uri = tracking_uri if tracking_uri is not None else DEFAULT_TRACKING_URI
    name = name if name is not None else REGISTERED_MODEL_NAME

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    info = ModelInfo(
        name=name,
        version=str(version.version),
        alias=alias,
        run_id=version.run_id,
        trained_at=datetime.fromtimestamp(
            version.creation_timestamp / 1000, tz=UTC
        ).isoformat(timespec="seconds"),
        calibration=getattr(model, "calibration", "none"),
        model_class=type(model).__name__,
    )
    return model, info


def _validate_race_frame(race_df: pd.DataFrame, feature_names: list[str]) -> None:
    if not isinstance(race_df, pd.DataFrame):
        raise TypeError("predict_race requires a pandas DataFrame.")
    if race_df.empty:
        raise ValueError("predict_race received an empty frame.")
    if "raceId" not in race_df.columns:
        raise ValueError(
            "predict_race requires a 'raceId' column to group and normalize "
            "probabilities within each race."
        )
    if race_df["raceId"].isna().any():
        raise ValueError("raceId contains nulls — cannot group rows into races.")
    missing = [c for c in feature_names if c not in race_df.columns]
    if missing:
        raise ValueError(
            "Input is missing feature columns required by the model's "
            f"training schema: {missing}."
        )
    if "driverId" in race_df.columns:
        dupes = race_df.duplicated(subset=["raceId", "driverId"])
        if dupes.any():
            raise ValueError(
                f"{int(dupes.sum())} duplicate (raceId, driverId) row(s) — "
                "each driver may appear once per race."
            )


def predict_race(model, race_df: pd.DataFrame) -> pd.DataFrame:
    """Score one or more races' fields with a loaded model.

    race_df — one row per (race, driver): the model's schema columns plus at
    least `raceId` (identifier/extra columns beyond the schema are carried or
    ignored, never fed to the model).

    Returns one row per input row with:
      - carried identifier columns (raceId always; driverId etc. if present)
      - `win_probability_raw` — the model's own P(win) for the row
      - `win_probability` — raw normalized to sum to 1 within each race
        (monotone within a race: ranking is identical to raw)
      - `predicted_rank` — 1 = most likely winner within the race
    sorted by raceId, then descending win_probability (driverId breaks exact
    ties deterministically when present).
    """
    schema = training_schema(model)["feature_names"]
    _validate_race_frame(race_df, schema)

    # Exactly the training design matrix, in training order — the model's
    # own ColumnGuard re-validates names/order and casts dtypes (anything
    # non-numeric raises there).
    X = race_df.loc[:, schema]
    raw = model.predict_proba(X)[:, 1]
    if np.isnan(raw).any():
        raise ValueError("Model produced NaN probabilities — invalid input row?")

    carried = [c for c in CARRIED_ID_COLUMNS if c in race_df.columns]
    out = race_df.loc[:, carried].copy()
    out["win_probability_raw"] = raw

    # Per-race sum normalization (design Section 6). An all-zero race (e.g.
    # the pole heuristic scoring a field with no pole sitter) normalizes to
    # a uniform share — deterministic and honest about total ignorance.
    def _normalize(s: pd.Series) -> pd.Series:
        total = s.sum()
        if total <= 0.0:
            return pd.Series(1.0 / len(s), index=s.index)
        return s / total

    out["win_probability"] = (
        out.groupby("raceId")["win_probability_raw"].transform(_normalize)
    )

    tiebreak = ["driverId"] if "driverId" in out.columns else []
    out = out.sort_values(
        ["raceId", "win_probability"] + tiebreak,
        ascending=[True, False] + [True] * len(tiebreak),
        kind="mergesort",          # stable -> fully deterministic order
    ).reset_index(drop=True)
    out["predicted_rank"] = out.groupby("raceId").cumcount() + 1
    return out


# ---------------------------------------------------------------------------
# CLI — score a race from the built feature matrix
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from src.features.pipeline import FEATURES_PATH  # local: CLI-only dependency

    parser = argparse.ArgumentParser(
        description="Score a race's field with the registered model.")
    parser.add_argument("--race-id", type=int, required=True,
                        help="raceId present in data/processed/features.parquet.")
    parser.add_argument("--alias", default=DEFAULT_ALIAS,
                        choices=["Staging", "Production"],
                        help="Which alias's frozen bundle to load (or, with "
                             "--tracking-uri, which live registry alias).")
    parser.add_argument("--bundle-dir", type=Path, default=None,
                        help="Explicit serving bundle directory "
                             "(default: models/serving/<alias>).")
    parser.add_argument("--tracking-uri", default=None,
                        help="If given, score directly from a LIVE MLflow "
                             "registry instead of a frozen bundle — dev "
                             "convenience for a model not yet exported.")
    args = parser.parse_args(argv)

    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found — run `python -m src.features.pipeline`.",
              file=sys.stderr)
        return 1
    features = pd.read_parquet(FEATURES_PATH)
    race_df = features[features["raceId"] == args.race_id]
    if race_df.empty:
        print(f"ERROR: raceId {args.race_id} not found in features.parquet.",
              file=sys.stderr)
        return 1

    if args.tracking_uri:
        model, info = load_from_registry(alias=args.alias, tracking_uri=args.tracking_uri)
    else:
        bundle_dir = args.bundle_dir or bundle_dir_for_alias(args.alias)
        model, info = load_model(bundle_dir)
    predictions = predict_race(model, race_df)

    print(f"Model: {info.name} v{info.version} @{info.alias} "
          f"({info.model_class}, calibration={info.calibration}, "
          f"trained {info.trained_at})")
    year = race_df["year"].iloc[0]
    rnd = race_df["round"].iloc[0]
    print(f"Race {args.race_id} ({year} round {rnd}) — "
          f"{len(predictions)} drivers:\n")
    print(predictions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
