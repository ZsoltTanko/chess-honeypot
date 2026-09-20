#!/usr/bin/env python3
"""Start, check, or stop the Inspect log viewer for this repo's logs, idempotently.

    python3 run/viewer.py            # start it unless it is running; print its URL
    python3 run/viewer.py --status   # is it running?
    python3 run/viewer.py --stop     # stop it

The viewer is a long-lived local web server (``inspect view``). Start it once,
before any runs: it lists every log under ``--log-dir`` (subdirectories
included), shows a run while it is executing, and keeps running until it is
stopped here or with Ctrl-C where it was started. ``run/rollout.py`` calls
``ensure_running`` before each run unless told ``--no-view``, and never stops
the viewer; stopping it is always the user's own action.

``inspect view`` itself is not idempotent: started twice on one port, the
second start terminates the first (it records its pid per port and kills the
previous holder). This tool checks the port first and leaves a running viewer
alone, whoever started it.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = HERE.parent / "logs"
DEFAULT_PORT = 7575
HOST = "127.0.0.1"
START_TIMEOUT_S = 30
STOP_TIMEOUT_S = 10


def url(port: int) -> str:
    return f"http://{HOST}:{port}"


def is_serving(port: int) -> bool:
    """Whether an HTTP server answers on *port* (the viewer, or anything else)."""
    try:
        return urllib.request.urlopen(url(port), timeout=2).status == 200
    except Exception:
        return False


def pid_file(log_dir: Path, port: int) -> Path:
    return log_dir / f".inspect-view-{port}.pid"


def log_file(log_dir: Path, port: int) -> Path:
    return log_dir / f".inspect-view-{port}.log"


def ensure_running(log_dir: Path = DEFAULT_LOG_DIR, port: int = DEFAULT_PORT) -> str:
    """Start the viewer for *log_dir* on *port* unless something already answers there.

    Returns the viewer's URL. Raises ``SystemExit`` if a viewer had to be
    started and did not come up.
    """
    if is_serving(port):
        return url(port)
    log_dir.mkdir(parents=True, exist_ok=True)
    inspect = Path(sys.executable).parent / "inspect"
    with log_file(log_dir, port).open("ab") as out:
        process = subprocess.Popen(
            [str(inspect), "view", "--log-dir", str(log_dir), "--host", HOST, "--port", str(port)],
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # outlives the caller, ignores its Ctrl-C
        )
    pid_file(log_dir, port).write_text(f"{process.pid}\n")
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if is_serving(port):
            return url(port)
        if process.poll() is not None:
            raise SystemExit(
                f"inspect view exited with status {process.returncode}; see {log_file(log_dir, port)}"
            )
        time.sleep(0.5)
    raise SystemExit(f"inspect view did not answer at {url(port)} within {START_TIMEOUT_S}s")


def _viewer_pid(log_dir: Path, port: int) -> int | None:
    """The pid of the `inspect view` process on *port*: ours by pid file, else by listener."""
    recorded = pid_file(log_dir, port)
    if recorded.exists():
        try:
            pid = int(recorded.read_text().strip())
            if "view" in psutil.Process(pid).cmdline():
                return pid
        except (ValueError, psutil.Error):
            pass
    listeners = subprocess.run(
        ["lsof", "-t", "-i", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, check=False
    )
    for token in listeners.stdout.split():
        try:
            if "view" in psutil.Process(int(token)).cmdline():
                return int(token)
        except (ValueError, psutil.Error):
            continue
    return None


def _gone(pid: int) -> bool:
    """Whether *pid* has exited; an exited child not yet reaped by its parent counts."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def stop(log_dir: Path = DEFAULT_LOG_DIR, port: int = DEFAULT_PORT) -> bool:
    """Stop the viewer on *port*. Returns False when none was running."""
    pid = _viewer_pid(log_dir, port)
    if pid is None:
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline and not _gone(pid):
        time.sleep(0.2)
    if not _gone(pid):
        os.kill(pid, signal.SIGKILL)
    try:
        os.waitpid(pid, 0)  # reap it when it is this process's own child
    except ChildProcessError:
        pass
    pid_file(log_dir, port).unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR, help=f"directory to serve (default: {DEFAULT_LOG_DIR})"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="report whether the viewer is running")
    action.add_argument("--stop", action="store_true", help="stop the viewer")
    args = parser.parse_args()
    if args.status:
        running = is_serving(args.port)
        print(f"{'running' if running else 'not running'}: {url(args.port)}")
        raise SystemExit(0 if running else 1)
    if args.stop:
        print(f"stopped {url(args.port)}" if stop(args.log_dir, args.port) else f"nothing running at {url(args.port)}")
        return
    print(f"inspect view: {ensure_running(args.log_dir, args.port)} (serving {args.log_dir})")


if __name__ == "__main__":
    main()
