"""Single-cell launcher for the WoSAR 2026 replication campaign.

Runs one (cell, replica) pair end-to-end:

  1. Resolve image digest pin and verify the local tag matches.
  2. Tear down any stale container with the same name.
  3. Build the docker run command from the cell yaml + paths and start
     the container in detached mode.
  4. Poll the readiness endpoint until the engine reports ready or the
     timeout elapses.
  5. Sanity-gate: verify the container's host PID appears among the
     compute apps of the expected GPU device. Abort the run if not.
  6. Resolve the engine worker PID and publish it to engine.pid, either
     statically (container_pid1) or via a find_engine_pid.py daemon
     (triton_child).
  7. Spawn monitoring/run_monitors.py with the cell's gpu_device as the
     gpu_index, the resolved pidfile, and the cell-defined sampling
     periods.
  8. Spawn client/run_client.py with the cell's workload overrides.
  9. Wait for duration_s of the cell, OR until any subprocess exits
     unexpectedly.
 10. Graceful shutdown: SIGTERM client, monitors, PID daemon, then
     docker rm -f the container.
 11. Wait for VRAM on gpu_device to return within
     vram_baseline_quiescence_mib of pre-run baseline, OR for
     post_run_cooldown_s to elapse, whichever comes later.

The run directory is fully self-contained after launch_cell.py exits:
  <run_dir>/
    manifest.json            (provenance: image digest, command, args, env, host info)
    engine.pid               (published by container_pid1 hook or find_engine_pid daemon)
    gpu0_*.csv               (gpu monitor, sampled at engine.gpu_device)
    <label>_*.csv            (proc monitor)
    system_*.csv             (system monitor)
    client/requests_*.csv    (per-request client log)
    logs/                    (stdout/stderr of monitors, client, find_engine_pid)
    docker_inspect.json      (post-launch docker inspect of the container)
    image_digest.txt         (verbatim sha256 from image_pin.json)

Invocation:

  python scripts/launch_cell.py \
      --cell-yaml campaigns/wosar2026/cells/e1.yaml \
      --replica 1 \
      --runs-root /home/dcotrone/wosar/runs \
      --repo-root /home/dcotrone/wosar/llm-serving-bench \
      --hf-cache-host /home/dcotrone/wosar/hf_cache \
      --campaign-id wosar2026

The launcher is intentionally synchronous and single-process. Parallelism
across slots (GPU 0/1/2) is the orchestrator's responsibility
(campaign.py spawns one launch_cell.py per slot per round, in separate
tmux sessions).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[launch_cell] {utc_iso()} {msg}", flush=True)


def die(msg: str, rc: int = 1) -> None:
    print(f"[launch_cell] FATAL {msg}", flush=True, file=sys.stderr)
    sys.exit(rc)


def render(template: str, **subs: str) -> str:
    """Substitute {placeholder} tokens in a template string."""
    out = template
    for k, v in subs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def render_in_obj(obj: Any, **subs: str) -> Any:
    """Recursively substitute {placeholders} in strings nested under obj."""
    if isinstance(obj, str):
        return render(obj, **subs)
    if isinstance(obj, list):
        return [render_in_obj(x, **subs) for x in obj]
    if isinstance(obj, dict):
        return {k: render_in_obj(v, **subs) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Image pinning
# ---------------------------------------------------------------------------


def load_image_pin(pin_path: Path) -> dict:
    if not pin_path.exists():
        die(f"image pin file missing: {pin_path}. Run scripts/utils/pin_images.sh first.")
    return json.loads(pin_path.read_text())


def verify_image_present(image_tag: str) -> None:
    rc = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        capture_output=True,
    ).returncode
    if rc != 0:
        die(f"image not present locally: {image_tag}. Run scripts/utils/pin_images.sh.")


# ---------------------------------------------------------------------------
# Docker container management
# ---------------------------------------------------------------------------


def teardown_container(name: str) -> None:
    """Remove any stale container with the same name. No-op if absent."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    if name in result.stdout.split():
        log(f"removing existing container {name}")
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


def build_docker_run_cmd(
    cell: dict,
    container_name: str,
) -> list[str]:
    eng = cell["engine"]
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--gpus", f'"device={eng["gpu_device"]}"',
        "--shm-size", eng["shm_size"],
    ]
    for port_map in eng.get("port_mapping", []):
        cmd += ["-p", port_map]
    for vol in eng.get("volumes", []):
        cmd += ["-v", vol]
    for k, v in eng.get("env", {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(f'{eng["image_repo"]}:{eng["image_tag"]}')
    cmd += [str(x) for x in eng.get("command", [])]
    return cmd


def docker_inspect(container_name: str) -> dict:
    result = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True,
        text=True,
        check=True,
    )
    arr = json.loads(result.stdout)
    return arr[0] if arr else {}


def get_container_pid(container_name: str) -> Optional[int]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    s = result.stdout.strip()
    if not s or s == "0":
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def wait_for_readyz(url: str, timeout_s: int, container_name: str) -> None:
    log(f"waiting up to {timeout_s}s for {url}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    log(f"readyz OK ({int(timeout_s - (deadline - time.monotonic()))}s)")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        # Detect container death so we fail fast instead of waiting for the timeout.
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if container_name not in result.stdout.split():
            die(f"container {container_name} exited during startup", rc=2)
        time.sleep(2)
    die(f"readyz did not come up in {timeout_s}s for {url}", rc=2)


# ---------------------------------------------------------------------------
# GPU sanity gate
# ---------------------------------------------------------------------------


def gpu_sanity_check(container_pid: int, gpu_device: int) -> None:
    """Verify the container PID appears as a compute app on gpu_device."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(gpu_device)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"WARNING: nvidia-smi failed for gpu {gpu_device}: {result.stderr.strip()}")
        return
    pids_on_gpu = [int(x.strip()) for x in result.stdout.split() if x.strip().isdigit()]
    if container_pid in pids_on_gpu:
        log(f"GPU sanity OK: container pid {container_pid} on gpu {gpu_device}")
        return
    # The engine may have spawned children that hold the GPU; check
    # descendants too. We use psutil if available; otherwise skip and warn.
    try:
        import psutil

        root = psutil.Process(container_pid)
        descendants = {p.pid for p in root.children(recursive=True)} | {container_pid}
        if any(p in pids_on_gpu for p in descendants):
            log(f"GPU sanity OK: container descendant on gpu {gpu_device} (root={container_pid})")
            return
    except Exception:
        pass
    die(
        f"GPU sanity FAILED: container pid {container_pid} (and descendants) "
        f"NOT on gpu {gpu_device}. Compute apps on gpu {gpu_device}: {pids_on_gpu}",
        rc=3,
    )


# ---------------------------------------------------------------------------
# PID resolution strategies
# ---------------------------------------------------------------------------


def setup_pid_strategy(
    cell: dict,
    container_name: str,
    pidfile: Path,
    repo_root: Path,
    log_dir: Path,
) -> Optional[subprocess.Popen]:
    """Return None if PID is static (container_pid1), else the daemon proc."""
    strategy = cell["engine"]["pid_strategy"]
    kind = strategy["type"]
    if kind == "container_pid1":
        pid = get_container_pid(container_name)
        if pid is None:
            die("container_pid1 strategy: docker inspect returned no PID", rc=4)
        pidfile.write_text(f"{pid}\n")
        log(f"pid_strategy=container_pid1, engine_pid={pid}")
        return None

    if kind == "triton_child":
        # Spawn find_engine_pid.py as a long-running daemon. It writes
        # the resolved worker PID to the pidfile and updates it on
        # respawn (handles vLLM EngineCore worker churn).
        pattern = strategy["process_pattern"]
        helper = repo_root / "monitoring" / "find_engine_pid.py"
        if not helper.exists():
            die(f"find_engine_pid.py not found at {helper}", rc=4)
        log_path = log_dir / "find_engine_pid.log"
        log_f = log_path.open("ab", buffering=0)
        cmd = [
            sys.executable,
            str(helper),
            "--container-name", container_name,
            "--process-pattern", pattern,
            "--pidfile", str(pidfile),
            "--poll-seconds", "30",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Wait briefly for the first resolution so the proc_monitor's
        # first sample is not 'process_alive=False'.
        first_resolve_deadline = time.monotonic() + 60
        while time.monotonic() < first_resolve_deadline:
            if pidfile.exists() and pidfile.read_text().strip().isdigit():
                log(f"pid_strategy=triton_child, initial pid={pidfile.read_text().strip()}")
                return proc
            if proc.poll() is not None:
                die(f"find_engine_pid daemon exited early rc={proc.returncode}", rc=4)
            time.sleep(1)
        die("triton_child strategy: first PID not resolved within 60s", rc=4)

    die(f"unknown pid_strategy.type: {kind}", rc=4)


# ---------------------------------------------------------------------------
# Monitor and client subprocesses
# ---------------------------------------------------------------------------


def spawn_monitors(
    repo_root: Path,
    run_dir: Path,
    cell: dict,
    pidfile: Path,
    duration_s: int,
    log_dir: Path,
    runs_root: Path,
    run_id: str,
) -> subprocess.Popen:
    monitors = cell["monitors"]
    gpu_device = cell["engine"]["gpu_device"]
    cmd = [
        sys.executable,
        str(repo_root / "monitoring" / "run_monitors.py"),
        "--run-id", run_id,
        "--runs-root", str(runs_root),
        "--gpu-index", str(gpu_device),
        "--pidfile", str(pidfile),
        "--duration-seconds", str(duration_s),
        "--label-engine", monitors["proc"]["label"],
        "--gpu-period", str(monitors["gpu"]["period_s"]),
        "--proc-period", str(monitors["proc"]["period_s"]),
        "--system-period", str(monitors["system"]["period_s"]),
        "--rotation-seconds", str(monitors["rotation_s"]),
    ]
    log_path = log_dir / "run_monitors.log"
    log_f = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def materialize_client_config(
    repo_root: Path,
    run_dir: Path,
    cell: dict,
    replica: int,
) -> Path:
    base_config_path = repo_root / "client" / "config.yaml"
    base_cfg = yaml.safe_load(base_config_path.read_text())
    overrides = render_in_obj(
        cell["workload"]["client_config_overrides"], replica=str(replica)
    )
    # seed_template uses {replica}; resolve into a real seed value.
    if "seed_template" in overrides:
        seed_str = overrides.pop("seed_template")
        overrides["seed"] = int(seed_str)
    merged = dict(base_cfg)
    merged.update(overrides)
    out = run_dir / "client_config.yaml"
    out.write_text(yaml.safe_dump(merged, sort_keys=False))
    return out


def spawn_client(
    repo_root: Path,
    run_dir: Path,
    client_config: Path,
    duration_s: int,
    log_dir: Path,
) -> subprocess.Popen:
    client_out = run_dir / "client"
    client_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "client" / "run_client.py"),
        "--config", str(client_config),
        "--output-dir", str(client_out),
        "--duration-seconds", str(duration_s),
    ]
    log_path = log_dir / "run_client.log"
    log_f = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(repo_root / "client"),
    )


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def host_info(gpu_index: int) -> dict:
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


def git_sha(repo_root: Path) -> Optional[str]:
    if not (repo_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.SubprocessError:
        return None


# ---------------------------------------------------------------------------
# VRAM quiescence
# ---------------------------------------------------------------------------


def vram_used_mib(gpu_index: int) -> Optional[int]:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(mem.used / (1024 * 1024))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def wait_vram_quiescence(
    gpu_index: int, baseline_mib: int, tolerance_mib: int, max_wait_s: int
) -> None:
    log(f"waiting for VRAM on gpu {gpu_index} to return within +/- {tolerance_mib} MiB of {baseline_mib} MiB")
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        current = vram_used_mib(gpu_index)
        if current is None:
            log("WARNING: pynvml VRAM read failed; skipping quiescence wait")
            return
        if abs(current - baseline_mib) <= tolerance_mib:
            log(f"VRAM quiesced at {current} MiB (baseline {baseline_mib} MiB)")
            return
        time.sleep(5)
    final = vram_used_mib(gpu_index)
    log(f"VRAM quiescence timeout: still at {final} MiB after {max_wait_s}s. Proceeding.")


# ---------------------------------------------------------------------------
# Subprocess teardown
# ---------------------------------------------------------------------------


def stop_subprocess(proc: subprocess.Popen, name: str, grace_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    log(f"stopping {name} (pid={proc.pid})")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        log(f"{name} did not exit in {grace_s}s, sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Launch one (cell, replica) of the campaign.")
    p.add_argument("--cell-yaml", type=Path, required=True)
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--hf-cache-host", type=Path, required=True)
    p.add_argument("--campaign-id", type=str, default="wosar2026")
    p.add_argument(
        "--gpu-device-override",
        type=int,
        default=None,
        help="Override engine.gpu_device. Used by sanity_runs.",
    )
    p.add_argument(
        "--duration-s-override",
        type=int,
        default=None,
        help="Override cell.duration_s. Used by sanity_runs.",
    )
    args = p.parse_args()

    # 1. Load and substitute placeholders.
    cell_raw = yaml.safe_load(args.cell_yaml.read_text())
    cell = render_in_obj(
        cell_raw,
        repo_root=str(args.repo_root),
        hf_cache_host=str(args.hf_cache_host),
        replica=str(args.replica),
    )
    if args.gpu_device_override is not None:
        cell["engine"]["gpu_device"] = args.gpu_device_override
    if args.duration_s_override is not None:
        cell["duration_s"] = args.duration_s_override

    cell_id = cell["cell_id"]
    replica = args.replica
    run_id = f"{args.campaign_id}_{cell_id}_r{replica:02d}"
    run_dir = args.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    log(f"run_id={run_id}")
    log(f"run_dir={run_dir}")

    # 2. Verify image pin.
    pin = load_image_pin(args.repo_root / cell["engine"]["digest_pin_file"])
    image_full = f'{cell["engine"]["image_repo"]}:{cell["engine"]["image_tag"]}'
    if pin["image_tag"] != image_full:
        die(f"image pin mismatch: cell expects {image_full}, pin file has {pin['image_tag']}")
    verify_image_present(image_full)
    (run_dir / "image_digest.txt").write_text(pin["digest"] + "\n")
    log(f"image: {image_full}  digest: {pin['digest']}")

    # 3. Container name with replica substituted.
    container_name = render(cell["engine"]["container_name_template"], replica=f"{replica:02d}")
    teardown_container(container_name)

    # 4. Capture pre-run VRAM baseline on the cell's GPU.
    gpu_device = cell["engine"]["gpu_device"]
    baseline_mib = vram_used_mib(gpu_device) or 0
    log(f"pre-run VRAM baseline on gpu {gpu_device}: {baseline_mib} MiB")

    # 5. Start container.
    docker_cmd = build_docker_run_cmd(cell, container_name)
    log("docker run cmd: " + " ".join(docker_cmd))
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        die(f"docker run failed rc={result.returncode}\nstderr: {result.stderr}")
    time.sleep(2)  # let docker assign host PID

    # 6. Persist docker inspect snapshot for provenance.
    inspect = docker_inspect(container_name)
    (run_dir / "docker_inspect.json").write_text(json.dumps(inspect, indent=2))
    container_pid = inspect.get("State", {}).get("Pid", 0)
    if not container_pid:
        die("container has no PID after docker run")

    # 7. Wait for readiness.
    readyz = cell["engine"]["readyz"]
    wait_for_readyz(readyz["url"], int(readyz["timeout_s"]), container_name)

    # 8. GPU sanity gate.
    gpu_sanity_check(container_pid, gpu_device)

    # 9. Resolve engine worker PID.
    pidfile = run_dir / "engine.pid"
    pid_daemon = setup_pid_strategy(cell, container_name, pidfile, args.repo_root, log_dir)

    # 10. Write the run manifest (started_at).
    started_at_unix = time.time()
    manifest = {
        "run_id": run_id,
        "campaign_id": args.campaign_id,
        "cell_id": cell_id,
        "replica": replica,
        "started_at": utc_iso(),
        "started_at_unix": started_at_unix,
        "host": host_info(gpu_device),
        "git_sha": git_sha(args.repo_root),
        "image": {
            "tag": image_full,
            "digest": pin["digest"],
            "source_tag": pin.get("source_tag"),
            "pinned_at": pin.get("pinned_at"),
        },
        "container": {
            "name": container_name,
            "host_pid": container_pid,
            "docker_run_cmd": docker_cmd,
        },
        "engine": cell["engine"],
        "monitors": cell["monitors"],
        "workload": cell["workload"],
        "duration_s": cell["duration_s"],
        "warmup_discard_s": cell["warmup_discard_s"],
        "vram_baseline_mib_pre_run": baseline_mib,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # 11. Spawn monitors and client.
    duration_s = int(cell["duration_s"])
    monitors_proc = spawn_monitors(
        args.repo_root, run_dir, cell, pidfile, duration_s, log_dir, args.runs_root, run_id
    )
    log(f"monitors orchestrator pid={monitors_proc.pid}")

    client_config = materialize_client_config(args.repo_root, run_dir, cell, replica)
    client_proc = spawn_client(args.repo_root, run_dir, client_config, duration_s, log_dir)
    log(f"client pid={client_proc.pid}")

    # 12. Supervise until duration elapses or any subprocess exits.
    deadline = started_at_unix + duration_s
    interrupted = False

    def handle_signal(_sig, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while not interrupted:
            time.sleep(5)
            if time.time() >= deadline:
                log("duration elapsed, beginning shutdown")
                break
            for name, proc in [("monitors", monitors_proc), ("client", client_proc)]:
                if proc.poll() is not None:
                    log(f"WARNING: {name} exited early rc={proc.returncode}")
                    interrupted = True
                    break
            if pid_daemon is not None and pid_daemon.poll() is not None:
                log(f"WARNING: pid_daemon exited early rc={pid_daemon.returncode}")
                interrupted = True
    finally:
        # 13. Graceful teardown.
        stop_subprocess(client_proc, "client")
        stop_subprocess(monitors_proc, "monitors", grace_s=60.0)
        if pid_daemon is not None:
            stop_subprocess(pid_daemon, "pid_daemon")
        log(f"removing container {container_name}")
        subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True)

        # 14. Wait for VRAM quiescence on the cell's GPU.
        cooldown = int(cell.get("post_run_cooldown_s", 600))
        wait_vram_quiescence(gpu_device, baseline_mib, tolerance_mib=200, max_wait_s=cooldown)

        # 15. Finalize manifest.
        ended_at_unix = time.time()
        manifest["ended_at"] = utc_iso()
        manifest["ended_at_unix"] = ended_at_unix
        manifest["duration_seconds_actual"] = ended_at_unix - started_at_unix
        manifest["interrupted_early"] = interrupted
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        log(f"done. duration={manifest['duration_seconds_actual']:.0f}s interrupted={interrupted}")

    sys.exit(0 if not interrupted else 2)


if __name__ == "__main__":
    main()
