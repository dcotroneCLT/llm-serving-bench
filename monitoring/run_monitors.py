"""Run orchestrator: launches gpu, proc, system monitors for one run.

Usage:

  python run_monitors.py \\
      --run-id pilot_vllm_replicate1 \\
      --runs-root ./runs \\
      --gpu-index 1 \\
      --pidfile ./runs/pilot_vllm_replicate1/engine.pid \\
      --duration-seconds 86400 \\
      --label-engine vllm_standalone

The orchestrator creates a fresh subdirectory under runs-root/run-id and
writes a manifest.json with environment metadata. It then spawns the
three monitors as separate Python processes, all pointed at that
directory. On SIGTERM/SIGINT, or on duration expiry, it sends SIGTERM
to each child and waits up to grace_period_s for them to flush.

Engine PID handling. The orchestrator does NOT start the engine.
That is the caller's responsibility (the engine is started by docker
compose, by a script, or manually). The caller writes the engine's PID
into --pidfile before or shortly after launching this orchestrator;
proc_monitor will pick it up on the first sample.

Manifest fields:

  run_id, started_at, ended_at
  host (hostname, kernel, os release)
  cpu (model, cores)
  memory_total_bytes
  gpu (model, driver, uuid, vram_total_bytes for the monitored GPU)
  python (version, executable)
  args (command-line args this script was invoked with)
  monitors (list with names, pids, cmd, log paths)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_host_info(gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "os": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
    }
    try:
        import psutil

        info["cpu"] = {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        }
        try:
            info["cpu"]["model"] = (
                Path("/proc/cpuinfo")
                .read_text()
                .split("model name")[1]
                .split(":", 1)[1]
                .splitlines()[0]
                .strip()
            )
        except (OSError, IndexError):
            pass
        info["memory_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu"] = {
                "index": gpu_index,
                "name": pynvml.nvmlDeviceGetName(handle),
                "uuid": pynvml.nvmlDeviceGetUUID(handle),
                "driver_version": pynvml.nvmlSystemGetDriverVersion(),
                "vram_total_bytes": mem.total,
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        info["gpu_error"] = str(e)
    return info


def spawn_monitor(name: str, cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_f = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Spawn and supervise monitoring agents for one run.")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--pidfile", type=Path, required=True, help="File where the engine's PID will appear.")
    p.add_argument("--duration-seconds", type=int, default=86400, help="Monitoring duration; 0 means run forever until signaled.")
    p.add_argument("--label-engine", type=str, required=True, help="Engine label for the proc monitor CSV name.")
    p.add_argument("--gpu-period", type=float, default=1.0)
    p.add_argument("--proc-period", type=float, default=5.0)
    p.add_argument("--system-period", type=float, default=5.0)
    p.add_argument("--rotation-seconds", type=int, default=60)
    p.add_argument("--grace-period-s", type=float, default=15.0)
    args = p.parse_args()

    run_dir = args.runs_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    started_at_unix = time.time()
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "started_at": utc_iso(),
        "started_at_unix": started_at_unix,
        "host": collect_host_info(args.gpu_index),
        "args": vars(args).copy(),
        "monitors": [],
    }
    # Path objects need stringification for JSON.
    manifest["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in manifest["args"].items()}

    here = Path(__file__).parent

    monitor_specs = [
        (
            "gpu",
            [
                sys.executable,
                str(here / "gpu_monitor.py"),
                "--gpu-index",
                str(args.gpu_index),
                "--output-dir",
                str(run_dir),
                "--period-seconds",
                str(args.gpu_period),
                "--rotation-seconds",
                str(args.rotation_seconds),
            ],
            log_dir / "gpu_monitor.log",
        ),
        (
            "proc",
            [
                sys.executable,
                str(here / "proc_monitor.py"),
                "--pidfile",
                str(args.pidfile),
                "--label",
                args.label_engine,
                "--output-dir",
                str(run_dir),
                "--period-seconds",
                str(args.proc_period),
                "--rotation-seconds",
                str(args.rotation_seconds),
            ],
            log_dir / "proc_monitor.log",
        ),
        (
            "system",
            [
                sys.executable,
                str(here / "system_monitor.py"),
                "--output-dir",
                str(run_dir),
                "--period-seconds",
                str(args.system_period),
                "--rotation-seconds",
                str(args.rotation_seconds),
            ],
            log_dir / "system_monitor.log",
        ),
    ]

    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for name, cmd, log_path in monitor_specs:
        proc = spawn_monitor(name, cmd, log_path)
        procs.append((name, proc, log_path))
        manifest["monitors"].append({
            "name": name,
            "pid": proc.pid,
            "cmd": cmd,
            "log": str(log_path),
        })

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[run_monitors] run_dir={run_dir}", flush=True)
    print(f"[run_monitors] spawned: " + ", ".join(f"{n}(pid={p.pid})" for n, p, _ in procs), flush=True)

    # Supervise. Stop conditions: signal received, duration elapsed, or any child exits.
    stop = False

    def handle(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    deadline = started_at_unix + args.duration_seconds if args.duration_seconds > 0 else None
    try:
        while not stop:
            time.sleep(1.0)
            if deadline is not None and time.time() >= deadline:
                print("[run_monitors] duration elapsed, shutting down", flush=True)
                break
            for name, proc, _ in procs:
                rc = proc.poll()
                if rc is not None:
                    print(f"[run_monitors] monitor {name} exited unexpectedly with rc={rc}", flush=True)
                    stop = True
                    break
    finally:
        # Send SIGTERM to all monitors and wait for graceful shutdown.
        for name, proc, _ in procs:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline_grace = time.time() + args.grace_period_s
        for name, proc, _ in procs:
            remaining = max(0.0, deadline_grace - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[run_monitors] monitor {name} did not stop in grace period, sending SIGKILL", flush=True)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        ended_at_unix = time.time()
        manifest["ended_at"] = utc_iso()
        manifest["ended_at_unix"] = ended_at_unix
        manifest["duration_seconds_actual"] = ended_at_unix - started_at_unix
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[run_monitors] done. duration={manifest['duration_seconds_actual']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
