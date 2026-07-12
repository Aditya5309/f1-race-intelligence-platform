"""
scripts/refresh_and_freeze.py

Orchestrates the full data-refresh-to-registration sequence as ONE atomic
run, so its steps can't drift out of sync the way they already did once
before (Decision 030): the model bundle and the display-data snapshot used
to be two independent manual commands, and a real production bug shipped
because someone ran one without the other. Fixing that structurally, not
just documenting "remember to run both", is the point of this script.

Sequence (stops at the first failing step — never a "best effort, continue
anyway" runner; everything after a failed step is left untouched):

  1. scripts/ingest_jolpica.py          pull newly-completed races (skip
                                        with --skip-ingest to rebuild/
                                        freeze from data/ as it already is)
  2. src.data.build_interim --target all
  3. src.pipelines.build_dataset
  4. src.features.pipeline
  5. src.models.season_tracking         ALWAYS runs, right here — scores any
                                        newly completed race in the current,
                                        ONGOING regulation era (2026's
                                        `future_engine` today; see
                                        src/models/eras.py) against
                                        whichever bundle was ALREADY served
                                        before this run (never the not-yet-
                                        registered candidate from step 7
                                        below). Read-only: never a training
                                        input, never touches split.test or
                                        temporal_split (src/models/
                                        season_tracking.py's own module
                                        docstring is the structural
                                        guarantee, not just this comment).
  6. scripts/export_display_data.py     ALWAYS runs — display data has no
                                        "good vs bad" concept to gate on,
                                        only "current vs stale" (Decision
                                        030); it must not be tied to
                                        whether the model registration
                                        below succeeds or is later promoted
  7. src.models.train --model <from config> --register [--calibrate]
     --params-file config/registered_model_params.json — model family,
     calibrate flag, and hyperparameters ALL read from the shared config
     file (Phase 4 Tranche D's source of truth), not hardcoded here

    python scripts/refresh_and_freeze.py                  # manual mode (default): step 7 exports immediately
    python scripts/refresh_and_freeze.py --automated       # step 7 registers only (export=False); a separate
                                                            # `python scripts/promote_model.py` call is the gate
                                                            # that actually swaps the served bundle (Tranche C/D)
    python scripts/refresh_and_freeze.py --skip-ingest      # rebuild/freeze only, using data/ as-is
    python scripts/refresh_and_freeze.py --dry-run          # runs ingest_jolpica.py --dry-run, stops there

--tracking-uri/--bundle-root/--artifacts-root pass straight through to step
7's `train.py --register` call (same flags, same meaning); --display-dest
passes through to step 6's `export_display_data.py --dest`; --tracking-dir
passes through to step 5's `season_tracking --tracking-dir` (default:
artifacts/tracking/). Point ALL FIVE at tmp locations for a hermetic dry
run of the whole sequence against real data — the default (manual mode, no
override) WILL overwrite the real committed
mlflow.db/artifacts/serving/artifacts/display/artifacts/tracking, same as
`train.py --register`/`export_display_data.py` always have. Steps 5 and 6
are NEVER gated (see above), so --tracking-dir/--display-dest matter even
in --automated mode, unlike --bundle-root/--artifacts-root which
--automated already protects via --no-export. Step 5 also reads
--bundle-root (NOT --bundle-root's step-7 write target — the SAME flag,
since tracking must score whatever bundle is currently sitting at that
root, i.e. last week's served candidate, before step 7 potentially
overwrites it).

--automated is what the scheduled retrain workflow (Phase 4 Tranche D Part
3) uses: it never touches artifacts/serving/ itself, then calls
promote_model.py separately, so promote_model.py's gate is the ONLY thing
that can change what's served — not a no-op behind an already-completed
unchecked export. Manual/interactive use keeps the default (immediate
export) unchanged from how this was documented before Tranche D existed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_FILE = _PROJECT_ROOT / "config" / "registered_model_params.json"


def _run(step_name: str, command: list[str]) -> bool:
    print(f"\n=== {step_name} ===")
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nSTOPPED: '{step_name}' failed (exit {result.returncode}) — "
              "everything after this step was left untouched.", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default="Staging", choices=["Staging", "Production"],
                        help="Alias to register the retrained candidate under.")
    parser.add_argument("--automated", action="store_true",
                        help="Register with --no-export (Phase 4 Tranche D) "
                             "so promote_model.py remains the only path "
                             "that changes what's served. Default (manual "
                             "mode): register with an immediate, unchecked "
                             "export, matching this project's pre-Tranche-D "
                             "documented manual command.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip scripts/ingest_jolpica.py — rebuild/"
                             "freeze from data/ as it already is.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run scripts/ingest_jolpica.py --dry-run and stop.")
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS_FILE,
                        help="Shared retrain-hyperparameter config passed "
                             "to `train.py --register` (Phase 4 Tranche D).")
    parser.add_argument("--tracking-uri", default=None,
                        help="Passed through to `train.py --register` "
                             "(default: train.py's own DEFAULT_TRACKING_URI). "
                             "Point at a tmp store for hermetic testing.")
    parser.add_argument("--bundle-root", type=Path, default=None,
                        help="Passed through to `train.py --register` "
                             "(default: artifacts/serving/). Point at a tmp "
                             "dir for hermetic testing — otherwise a manual-"
                             "mode run WILL overwrite the real committed bundle.")
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help="Passed through to `train.py --register` "
                             "(default: artifacts/). Point at a tmp dir for "
                             "hermetic testing, same caveat as --bundle-root.")
    parser.add_argument("--display-dest", type=Path, default=None,
                        help="Passed through to `scripts/export_display_data.py "
                             "--dest` (default: artifacts/display/). Point at "
                             "a tmp dir for hermetic testing — otherwise step "
                             "6 WILL overwrite the real committed display "
                             "data, even in --automated mode (display refresh "
                             "is never gated, see the module docstring).")
    parser.add_argument("--tracking-dir", type=Path, default=None,
                        help="Passed through to `src.models.season_tracking "
                             "--tracking-dir` (default: artifacts/tracking/). "
                             "Point at a tmp dir for hermetic testing — "
                             "otherwise step 5 WILL append to the real "
                             "committed running-eval CSV, even in "
                             "--automated mode (tracking is never gated, "
                             "same reasoning as display refresh).")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.params_file.exists():
        print(f"ERROR: --params-file {args.params_file} not found — checked "
              "up front so a config typo doesn't waste time on steps 1-6 "
              "before failing at registration.", file=sys.stderr)
        return 1

    py = sys.executable

    if not args.skip_ingest:
        ingest_cmd = [py, "scripts/ingest_jolpica.py"]
        if args.dry_run:
            ingest_cmd.append("--dry-run")
        if not _run("1. Ingest new race weekends (jolpica-f1)", ingest_cmd):
            return 1
        if args.dry_run:
            print("\n--dry-run: stopping after ingestion preview.")
            return 0
    else:
        print("\n=== 1. Ingest new race weekends (jolpica-f1) ===\nSKIPPED (--skip-ingest)")

    if not _run("2. Build interim datasets", [py, "-m", "src.data.build_interim", "--target", "all"]):
        return 1
    if not _run("3. Build master dataset", [py, "-m", "src.pipelines.build_dataset"]):
        return 1
    if not _run("4. Build feature store", [py, "-m", "src.features.pipeline"]):
        return 1

    # Score newly completed races with whatever bundle is CURRENTLY served
    # at --bundle-root, BEFORE step 7 potentially overwrites it — tracks the
    # model that was actually in production, not this run's not-yet-vetted
    # candidate. Always runs, regardless of --automated / whether step 7's
    # candidate ends up promoted (src/models/season_tracking.py: read-only,
    # never a training input).
    tracking_cmd = [py, "-m", "src.models.season_tracking", "--alias", args.alias]
    if args.bundle_root:
        tracking_cmd += ["--bundle-root", str(args.bundle_root)]
    if args.tracking_dir:
        tracking_cmd += ["--tracking-dir", str(args.tracking_dir)]
    if not _run("5. Track current-era races against the currently-served bundle", tracking_cmd):
        return 1

    display_cmd = [py, "scripts/export_display_data.py"]
    if args.display_dest:
        display_cmd += ["--dest", str(args.display_dest)]
    if not _run("6. Refresh display-data snapshot (always, unconditionally)", display_cmd):
        return 1

    config = json.loads(args.params_file.read_text())
    register_cmd = [
        py, "-m", "src.models.train",
        "--model", config["model"],  # read from the file, not hardcoded — must match
                                       # the model family --params-file's params are for
        "--register", args.alias,
        "--params-file", str(args.params_file),
    ]
    if config.get("calibrate"):
        register_cmd.append("--calibrate")
    if args.tracking_uri:
        register_cmd += ["--tracking-uri", args.tracking_uri]
    if args.bundle_root:
        register_cmd += ["--bundle-root", str(args.bundle_root)]
    if args.artifacts_root:
        register_cmd += ["--artifacts-root", str(args.artifacts_root)]
    if args.automated:
        register_cmd.append("--no-export")
    step_label = "7. Register candidate (export=False — gated by promote_model.py next)" \
        if args.automated else "7. Register candidate (immediate export)"
    if not _run(step_label, register_cmd):
        return 1

    print("\nAll steps completed.")
    if args.automated:
        print(f"Run `python scripts/promote_model.py --alias {args.alias}` next "
              "to gate and actually promote this candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
