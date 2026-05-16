"""Dynamic engine-PID resolver for containerized aging runs.

Long-running helper that, every N seconds, identifies the engine worker
PID inside a Docker container and writes it to a pidfile that
``proc_monitor.py`` re-reads on each sample.

Motivation. ``docker top`` captures a *snapshot* of host-side PIDs.
Container engines (vLLM V0 multiprocess, V1 EngineCore, Triton Python
backend) routinely spawn and respawn worker subprocesses, so a PID
captured once at run start is brittle: if the worker exits and respawns
under a new PID, the proc monitor records ``process_alive=False`` for the
rest of the run.

Strategy. At each poll:

  1. Resolve the container's init PID via ``docker inspect``.
  2. Walk its descendants with psutil.
  3. Pick the descendant whose cmdline matches a user-supplied regex; if
     multiple match, prefer the one with the highest accumulated CPU
     time (the worker doing real work, not an idle wrapper).
  4. Atomically write that PID to the target pidfile.
  5. Log to stderr only when the resolved PID changes.

The proc monitor's ``--pidfile`` re-read logic then transparently
follows worker respawns without modification.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError as e:
    raise SystemExit("psutil not installed. Run: pip install psutil") from e


def _log(msg: str) -> None:
    print(f"[find_engine_pid] {msg}", file=sys.stderr, flush=True)


def get_container_pid(container_name: str) -> Optional[int]:
    """Return the host-side PID of the container's init process, or None."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        _log(f"docker inspect failed: {e}")
        return None
    if result.returncode != 0:
        _log(f"docker inspect rc={result.returncode}: {result.stderr.strip()}")
        return None
    pid_str = result.stdout.strip()
    if not pid_str or pid_str == "0":
        return None
    try:
        return int(pid_str)
    except ValueError:
        return None


def find_matching_descendant(
    root_pid: int, pattern: re.Pattern[str]
) -> Optional[psutil.Process]:
    """Among descendants of root_pid, return the one matching the pattern.

    If multiple descendants match, return the one with the highest
    accumulated CPU time (user + system) — that worker is doing real
    work, not an idle parent/wrapper.
    """
    try:
        root = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    candidates: list[tuple[float, psutil.Process]] = []
    try:
        descendants = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    for proc in descendants:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if not pattern.search(cmdline):
            continue
        try:
            times = proc.cpu_times()
            cpu_total = times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            cpu_total = 0.0
        try:
            rss = proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            rss = 0
        candidates.append((rss, cpu_total, proc))

    if not candidates:
        return None
    # Prefer the descendant with the model loaded (multi-GB RSS) over a
    # lightweight wrapper/stub that matches the same regex; use CPU time
    # as a tiebreak when RSS is comparable.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def write_pidfile_atomic(pidfile: Path, pid: int) -> None:
    tmp = pidfile.with_suffix(pidfile.suffix + ".tmp")
    tmp.write_text(f"{pid}\n")
    os.replace(tmp, pidfile)


class _Shutdown:
    def __init__(self) -> None:
        self.flag = False

    def trigger(self, _sig, _frame) -> None:
        self.flag = True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Track engine worker PID inside a Docker container and "
        "publish it to a pidfile for proc_monitor.py."
    )
    p.add_argument("--container-name", type=str, required=True)
    p.add_argument(
        "--process-pattern",
        type=str,
        required=True,
        help="Regex matched against the descendant's joined cmdline. "
        "Examples: 'EngineCore' (vLLM V1), 'spawn_main' (vLLM V0 mp), "
        "'triton_python_backend_stub' (Triton in-process backend).",
    )
    p.add_argument("--pidfile", type=Path, required=True)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument(
        "--duration-seconds",
        type=int,
        default=0,
        help="If > 0, exit after this many seconds. 0 means run until signaled.",
    )
    args = p.parse_args()

    try:
        pattern = re.compile(args.process_pattern)
    except re.error as e:
        raise SystemExit(f"invalid --process-pattern regex: {e}")

    args.pidfile.parent.mkdir(parents=True, exist_ok=True)

    shutdown = _Shutdown()
    signal.signal(signal.SIGTERM, shutdown.trigger)
    signal.signal(signal.SIGINT, shutdown.trigger)

    last_published_pid: Optional[int] = None
    started_at = time.monotonic()
    deadline = started_at + args.duration_seconds if args.duration_seconds > 0 else None

    _log(
        f"starting: container={args.container_name} pattern={args.process_pattern!r} "
        f"pidfile={args.pidfile} poll={args.poll_seconds}s "
        f"duration={args.duration_seconds if args.duration_seconds > 0 else 'forever'}"
    )

    while not shutdown.flag:
        container_pid = get_container_pid(args.container_name)
        if container_pid is None:
            if last_published_pid is not None:
                _log(f"container {args.container_name!r} not running; last pid was {last_published_pid}")
                last_published_pid = None
        else:
            worker = find_matching_descendant(container_pid, pattern)
            if worker is None:
                if last_published_pid is not None:
                    _log(
                        f"no descendant of container pid {container_pid} matches "
                        f"{args.process_pattern!r} (last published pid was {last_published_pid})"
                    )
                    last_published_pid = None
            else:
                try:
                    worker_pid = worker.pid
                    worker_cmdline = " ".join(worker.cmdline())
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    worker_pid = None
                    worker_cmdline = ""
                if worker_pid is not None:
                    if worker_pid != last_published_pid:
                        try:
                            write_pidfile_atomic(args.pidfile, worker_pid)
                        except OSError as e:
                            _log(f"failed to write pidfile {args.pidfile}: {e}")
                        else:
                            _log(f"resolved pid={worker_pid} (prev={last_published_pid}) cmdline={worker_cmdline!r}")
                            last_published_pid = worker_pid
                    else:
                        # Refresh pidfile mtime so health checks can tell a
                        # healthy daemon (stable PID) apart from a stalled one.
                        try:
                            os.utime(args.pidfile, None)
                        except OSError:
                            pass

        if deadline is not None and time.monotonic() >= deadline:
            _log("duration elapsed, exiting")
            break

        # Sleep in small slices so SIGTERM is honored promptly.
        slept = 0.0
        while slept < args.poll_seconds and not shutdown.flag:
            chunk = min(1.0, args.poll_seconds - slept)
            time.sleep(chunk)
            slept += chunk

    _log("shutdown complete")


if __name__ == "__main__":
    main()
