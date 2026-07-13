"""
scripts/dev.py — single-command local development launcher.

    python scripts/dev.py
    make dev

Starts the FastAPI service if nothing is already answering at the
configured API URL, waits for it to respond, then runs the Streamlit
dashboard in the foreground. If THIS script started the API, it is stopped
when Streamlit exits (Ctrl+C or normal exit); an API that was already
running is left untouched.

Dev tooling only — never imported by
app/api.py or app/dashboard.py, and never invoked by any deployment path:
Streamlit Community Cloud always launches app/dashboard.py directly, and
each Docker image runs `uvicorn app.api:app` / `streamlit run
app/dashboard.py` as its own separate entrypoint. This script existing or
not existing has zero effect on production behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))          # runnable without pip install

LOG_DIR = _PROJECT_ROOT / "logs"
API_LOG = LOG_DIR / "api.log"

HEALTH_POLL_INTERVAL = 0.3     # seconds between polls — no blind sleep-and-hope
HEALTH_TIMEOUT = 30.0           # seconds to wait for the API to respond at all
DEFAULT_DASHBOARD_PORT = 8501   # Streamlit's own documented default

# The response shape unique to THIS project's /health (app/api.py::HealthResponse).
# Used to confirm "something answering on this port" is actually our API, not an
# unrelated service that happens to occupy the same port.
_EXPECTED_HEALTH_KEYS = {"status", "api_version", "model", "detail"}

ProbeFn = Callable[[], "tuple[bool, dict | None]"]
VerifyFn = Callable[[bool, "dict | None"], bool]


def _http_probe(url: str, timeout: float = 2.0) -> tuple[bool, dict | None]:
    """GET url -> (reachable, parsed_json_or_None).

    reachable=False means nothing is listening (connection refused/timed
    out) — the port is free. reachable=True with payload=None means SOMETHING
    answered but not with valid JSON — the port is occupied, just not by
    something we can identify; callers must NOT treat that as "free".
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp else b""          # a real (non-2xx) HTTP response
    except (urllib.error.URLError, OSError, TimeoutError):
        return False, None                              # nothing listening
    try:
        return True, json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return True, None                               # answered, but not JSON


def _looks_like_our_api(payload: dict | None) -> bool:
    return (
        isinstance(payload, dict)
        and _EXPECTED_HEALTH_KEYS.issubset(payload)
        and payload.get("status") in ("ok", "degraded")
    )


def _tail(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return "(no log file written)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:]) or "(log file is empty)"


@dataclass
class ManagedProcess:
    """One background dev service this launcher can verify/start/stop.

    `probe` and `verify` are injected rather than hardcoded so a future
    service (Redis, Postgres, a Celery worker) is a new ManagedProcess
    instance with its own probe/verify — not a redesign of this class or
    of main()'s orchestration below.
    """

    name: str
    command: list[str]
    log_path: Path
    probe: ProbeFn
    verify: VerifyFn
    process: subprocess.Popen | None = field(default=None, init=False)
    started: bool = field(default=False, init=False)

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(               # noqa: S603 — fixed, non-shell argv
            self.command, cwd=_PROJECT_ROOT,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        self.started = True

    def wait_until_verified(self, timeout: float = HEALTH_TIMEOUT) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited immediately (code {self.process.returncode}). "
                    f"Last lines of {self.log_path}:\n{_tail(self.log_path)}"
                )
            reachable, payload = self.probe()
            if reachable and self.verify(reachable, payload):
                return payload or {}
            if reachable and not self.verify(reachable, payload):
                raise RuntimeError(
                    f"{self.name} started, but its response doesn't look right: "
                    f"{payload!r}. Last lines of {self.log_path}:\n{_tail(self.log_path)}"
                )
            time.sleep(HEALTH_POLL_INTERVAL)
        raise RuntimeError(
            f"{self.name} did not respond within {timeout:.0f}s. "
            f"Last lines of {self.log_path}:\n{_tail(self.log_path)}"
        )

    def stop(self) -> None:
        if self.started and self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _parse_args() -> argparse.Namespace:
    # Deferred import: keeps stdlib imports at module top (ruff I/E402),
    # same idiom as scripts/smoke.py's app/src imports.
    from app.config import Settings

    settings = Settings()
    default = urlparse(settings.api_url)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default=default.hostname or "localhost",
                         help="Overrides the host from Settings().api_url.")
    parser.add_argument("--api-port", type=int, default=default.port or 8000,
                         help="Overrides the port from Settings().api_url.")
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    return parser.parse_args()


def main() -> int:
    # stdout is block-buffered whenever it isn't a real TTY (e.g. piped into
    # a log, or captured by tooling); without this, every status/summary
    # print below sits in the buffer — invisible — for as long as Streamlit
    # runs in the foreground, and only appears (out of order, after
    # Streamlit's own output) once this process finally exits. Line
    # buffering makes the startup summary actually show up when it's
    # printed, in the order it's printed.
    sys.stdout.reconfigure(line_buffering=True)

    args = _parse_args()
    api_url = f"http://{args.api_host}:{args.api_port}"
    health_url = f"{api_url}/health"

    api = ManagedProcess(
        name="FastAPI",
        command=[sys.executable, "-m", "uvicorn", "app.api:app",
                 "--host", args.api_host, "--port", str(args.api_port)],
        log_path=API_LOG,
        probe=lambda: _http_probe(health_url),
        verify=lambda reachable, payload: reachable and _looks_like_our_api(payload),
    )

    reachable, payload = api.probe()
    if reachable and not _looks_like_our_api(payload):
        print(
            f"ERROR: something is already listening at {health_url}, but it "
            f"doesn't look like this project's API (unexpected response: "
            f"{payload!r}). Refusing to start a second instance — free that "
            f"port or pass --api-port to use a different one.",
            file=sys.stderr,
        )
        return 1

    if reachable:
        print(f"API already running at {api_url} — leaving it untouched.")
        status = payload or {}
    else:
        print(f"Starting API: {' '.join(api.command)}")
        api.start()

    # Single try/finally from here on, covering BOTH the health-poll wait
    # and the Streamlit run: a Ctrl+C during either phase (not just a
    # RuntimeError from a health-check timeout) must still stop an API this
    # script started — otherwise an interrupt during the wait would orphan
    # the just-spawned process before Streamlit ever runs.
    try:
        if api.started:
            try:
                status = api.wait_until_verified()
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            print(f"API started (pid {api.process.pid}).")

        dashboard_url = f"http://localhost:{args.dashboard_port}"
        print()
        print("=" * 60)
        print(" F1 Dev Launcher")
        print("=" * 60)
        print(f" API status    : {'started' if api.started else 'already running'} "
              f"[{status.get('status', 'unknown')}]")
        print(f" API URL       : {api_url}")
        print(f" Health check  : {health_url}")
        print(f" Dashboard     : {dashboard_url}")
        print(f" API log file  : {API_LOG}")
        print("=" * 60)
        print()

        dashboard_cmd = [
            sys.executable, "-m", "streamlit", "run", "app/dashboard.py",
            "--server.port", str(args.dashboard_port),
        ]
        # Keep the dashboard child in sync with whatever host/port was
        # actually verified above, even if --api-host/--api-port overrode
        # Settings' default.
        child_env = dict(os.environ)
        child_env["F1_API_URL"] = api_url

        subprocess.run(dashboard_cmd, cwd=_PROJECT_ROOT, env=child_env)  # noqa: S603
    finally:
        if api.started:
            print("\nStopping API (started by this launcher)...")
            api.stop()
        elif reachable:
            print("\nLeaving the pre-existing API running.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
