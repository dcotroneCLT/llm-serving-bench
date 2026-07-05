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

  python3 scripts/launch_cell.py \
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
import csv
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

# reaper is a sibling module; make it importable whether launch_cell runs as a
# script, is imported by attach_run, or is loaded by the test harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reaper  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


_ABORT_CLEANUP_ENABLED = False
_ABORT_CONTAINER_NAMES: list[str] = []
_ABORT_LOG_DIR: Optional[Path] = None


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[launch_cell] {utc_iso()} {msg}", flush=True)


def enable_abort_cleanup(container_names, log_dir: Path) -> None:
    """Register the container(s) to force-remove (with a best-effort log dump)
    if the launcher aborts before its normal teardown runs. Accepts a single
    name or a list: multi-container stacks (e.g. Dynamo disaggregated) must have
    their WHOLE stack torn down on abort, not just one container."""
    global _ABORT_CLEANUP_ENABLED, _ABORT_CONTAINER_NAMES, _ABORT_LOG_DIR
    if isinstance(container_names, str):
        container_names = [container_names]
    _ABORT_CLEANUP_ENABLED = True
    _ABORT_CONTAINER_NAMES = list(container_names)
    _ABORT_LOG_DIR = log_dir


def disable_abort_cleanup() -> None:
    global _ABORT_CLEANUP_ENABLED
    _ABORT_CLEANUP_ENABLED = False


def cleanup_after_abort() -> None:
    if not _ABORT_CLEANUP_ENABLED or not _ABORT_CONTAINER_NAMES or not _ABORT_LOG_DIR:
        return
    single = len(_ABORT_CONTAINER_NAMES) == 1
    for name in _ABORT_CONTAINER_NAMES:
        try:
            _ABORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            # Single-container runs keep the historical docker_abort.log name.
            log_name = "docker_abort.log" if single else f"docker_abort_{name}.log"
            save_docker_logs(name, _ABORT_LOG_DIR / log_name)
        except Exception:
            pass
        try:
            subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, timeout=60)
        except Exception:
            pass


def die(msg: str, rc: int = 1) -> None:
    print(f"[launch_cell] FATAL {msg}", flush=True, file=sys.stderr)
    cleanup_after_abort()
    sys.exit(rc)


def free_gb(path: Path) -> Optional[float]:
    """Free GiB at `path`, or None if it cannot be stat'd. NEVER returns inf:
    fabricating 'infinite free space' on error would make a disk gate pass when
    it must fail."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return None


def docker_root_dir() -> Optional[Path]:
    """The Docker data-root volume (where images/layers/container logs live)."""
    try:
        r = subprocess.run(["docker", "info", "-f", "{{.DockerRootDir}}"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def require_free_space(paths: list[Optional[Path]], min_gb: float, label: str = "run") -> None:
    """SC-2 #1: pre-run free-space GATE across the distinct filesystems behind
    `paths` (runs-root for CSVs, docker data-root for images/container logs).
    Refuses to start if any is below min_gb. A None path (e.g. the docker
    data-root could not be determined) or a path that cannot be stat'd is a HARD
    FAIL, not a skip: an unknown filesystem is exactly when we must not proceed."""
    seen: set = set()
    for p in paths:
        if p is None:
            die(f"free-space gate: a filesystem to check is unknown (docker data-root "
                f"undetermined?) -- refusing to start {label}. Check `docker info`.", rc=7)
        try:
            key = os.stat(str(p)).st_dev
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        g = free_gb(p)
        if g is None:
            die(f"free-space gate: cannot stat {p} -- refusing to start {label}.", rc=7)
        log(f"free-space gate: {p} has {g:.1f} GB free")
        if g < min_gb:
            die(f"free-space gate: {p} has {g:.1f} GB < {min_gb} GB required to start {label}. "
                f"Free space (docker image prune; move data-root to /home) or lower --min-free-gb.", rc=7)


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


def assert_run_dir_fresh(run_dir: Path) -> None:
    """Refuse to mix a new attempt with old CSVs/manifests/state."""
    if not run_dir.exists():
        return
    allowed_precreated = {"launch_cell.log"}
    entries = [p.name for p in run_dir.iterdir()]
    stale = [name for name in entries if name not in allowed_precreated]
    if stale:
        preview = ", ".join(sorted(stale)[:8])
        die(
            f"run_dir already contains campaign artifacts: {run_dir} ({preview}). "
            "Archive or remove it before relaunching this run.",
            rc=6,
        )


def save_docker_logs(container_name: str, out_path: Path, tail: int = 50000) -> None:
    """Best-effort docker logs capture before the container is removed.

    --tail caps captured volume: a 36 h verbose engine log can reach
    multiple GB, and buffering it in the supervisor would risk OOM.
    """
    try:
        with out_path.open("wb") as f:
            subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_name],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        out_path.write_text(f"docker logs failed: {e}\n")


def summarize_client_csvs(client_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": 0,
        "ok": 0,
        "error": 0,
        "timeout": 0,
        "dropped": 0,
        "status_counts": {},
        "first_ts_unix": None,
        "last_ts_unix": None,
    }
    for path in sorted(client_dir.glob("requests_*.csv")):
        try:
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    summary["total"] += 1
                    status = (row.get("status") or "").strip()
                    counts = summary["status_counts"]
                    counts[status] = counts.get(status, 0) + 1
                    if status == "ok":
                        summary["ok"] += 1
                    elif status == "timeout":
                        summary["timeout"] += 1
                    elif status == "dropped":
                        summary["dropped"] += 1
                    elif status:
                        summary["error"] += 1
                    ts_text = row.get("finished_at_unix") or row.get("submitted_at_unix") or ""
                    try:
                        ts = float(ts_text)
                    except ValueError:
                        continue
                    if summary["first_ts_unix"] is None or ts < summary["first_ts_unix"]:
                        summary["first_ts_unix"] = ts
                    if summary["last_ts_unix"] is None or ts > summary["last_ts_unix"]:
                        summary["last_ts_unix"] = ts
        except OSError as e:
            counts = summary["status_counts"]
            counts[f"csv_read_error:{path.name}"] = str(e)
    first_ts = summary["first_ts_unix"]
    last_ts = summary["last_ts_unix"]
    if first_ts is not None and last_ts is not None and last_ts > first_ts:
        summary["span_s"] = last_ts - first_ts
        summary["issued_rate_rps"] = summary["total"] / summary["span_s"]
    else:
        summary["span_s"] = 0.0
        summary["issued_rate_rps"] = 0.0
    return summary


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
        timeout=30,
    ).returncode
    if rc != 0:
        die(f"image not present locally: {image_tag}. Run scripts/utils/pin_images.sh.")


# ---------------------------------------------------------------------------
# Docker container management
# ---------------------------------------------------------------------------


def teardown_container(name: str, log_path: Optional[Path] = None) -> None:
    """Remove any stale container with the same name. No-op if absent."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    if name in result.stdout.split():
        log(f"removing existing container {name}")
        if log_path is not None:
            save_docker_logs(name, log_path)
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, timeout=60)


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
        # SC-2 #3: cap container json-file logs so a 48h run cannot fill the
        # docker data-root disk.
        "--log-opt", "max-size=50m", "--log-opt", "max-file=3",
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


def all_dyn_containers() -> list[str]:
    """Every container (running or stopped) whose name starts with dyn_ -- the
    Dynamo stack's naming prefix, across any topology/index."""
    try:
        r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=30)
    except subprocess.SubprocessError:
        return []
    return [n for n in r.stdout.split() if n.startswith("dyn_")]


def dynamo_engine_procs_on_host() -> list[int]:
    """Host PIDs whose cmdline names a Dynamo engine worker/frontend -- used to
    fail hard if an engine process survives the container sweep."""
    hits: list[int] = []
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return hits
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if "dynamo.vllm" in cl or "dynamo.frontend" in cl:
            hits.append(p.info["pid"])
    return hits


def docker_inspect(container_name: str) -> dict:
    result = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    arr = json.loads(result.stdout)
    return arr[0] if arr else {}


def get_container_pid(container_name: str) -> Optional[int]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True,
        text=True,
        timeout=30,
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
            timeout=30,
        )
        if container_name not in result.stdout.split():
            die(f"container {container_name} exited during startup", rc=2)
        time.sleep(2)
    die(f"readyz did not come up in {timeout_s}s for {url}", rc=2)


def verify_models_listed(models_url: str, model_name: str, timeout_s: int) -> None:
    """Re-verify that an OpenAI-compatible frontend actually LISTS the model on
    /v1/models before we proceed. /health can report 200 while /v1/models is
    empty (the Dynamo registry root cause we closed); the bring-up script already
    gates on this, so this is a cheap independent confirmation, not the primary
    wait."""
    log(f"verifying {model_name} listed on {models_url} (up to {timeout_s}s)")
    deadline = time.monotonic() + timeout_s
    last_err = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if 200 <= resp.status < 300:
                    try:
                        ids = [m.get("id") for m in json.loads(body).get("data", [])]
                    except (json.JSONDecodeError, AttributeError):
                        ids = []
                    if model_name in ids:
                        log(f"/v1/models lists {model_name}")
                        return
                    last_err = f"model not in {ids}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = str(e)
        time.sleep(3)
    die(f"/v1/models did not list {model_name} within {timeout_s}s ({last_err})", rc=2)


# ---------------------------------------------------------------------------
# Runtime health (supervision-loop liveness of the engine, not just our children)
# ---------------------------------------------------------------------------

# The supervision loop runs a cheap health check on this cadence (inside the
# existing 5 s cycle). Isolated failures are tolerated; only N consecutive
# lifecycle-health failures, or a sustained all-fail client window, trip a death.
HEALTH_CHECK_EVERY_S = 30
HEALTH_FAIL_CONSECUTIVE = 3
ENDPOINT_DEAD_WINDOW_S = 300      # ~5 min rolling window for endpoint-dead detection
ENDPOINT_DEAD_MIN_ROWS = 5        # need at least this many rows before declaring death


def container_running(name: str) -> bool:
    """True if the container is running. A docker error (daemon hiccup) is
    TOLERATED as running so a transient blip does not false-trigger a health kill;
    a container docker definitively reports missing is not running."""
    try:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                           capture_output=True, text=True, timeout=15)
    except subprocess.SubprocessError:
        return True  # can't tell -> tolerate
    if r.returncode != 0:
        err = (r.stderr or "").lower()
        if "no such object" in err or "no such container" in err:
            return False  # definitively absent
        return True  # ambiguous docker error -> tolerate
    return r.stdout.strip().lower() == "true"


def client_all_fail_window(client_dir: Path, window_s: int, now_unix: float,
                           min_rows: int = ENDPOINT_DEAD_MIN_ROWS) -> Optional[int]:
    """Endpoint-dead detector. Return the number of client rows in the last
    window_s IFF there are at least min_rows and NONE of them are status=ok
    (the engine is de facto dead); else None.

    Deliberately NOT a drop/error-RATE threshold: high drop rates are a legitimate
    stress-workload signal (a prior cell ran at 14-16% dropped in a confirmed
    stationary regime). ONLY an all-fail window counts as death. Reads just the two
    newest requests_*.csv (rotation-safe, cheap at 48 h)."""
    cutoff = now_unix - window_s
    try:
        files = sorted(client_dir.glob("requests_*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:2]
    except OSError:
        return None
    total = 0
    ok = 0
    for path in files:
        try:
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    ts_text = row.get("finished_at_unix") or row.get("submitted_at_unix") or ""
                    try:
                        ts = float(ts_text)
                    except ValueError:
                        continue
                    if ts < cutoff:
                        continue
                    total += 1
                    if (row.get("status") or "").strip() == "ok":
                        ok += 1
        except OSError:
            continue
    if total >= min_rows and ok == 0:
        return total
    return None


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
        die(f"nvidia-smi failed for gpu {gpu_device}: {result.stderr.strip()}", rc=3)
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


def gpu_devices_for_cell(cell: dict) -> list[int]:
    """Devices to sample: engine.gpu_devices (list, multi-GPU e.g. Dynamo) if
    present, else the single engine.gpu_device."""
    eng = cell["engine"]
    if eng.get("gpu_devices"):
        return [int(x) for x in eng["gpu_devices"]]
    return [int(eng["gpu_device"])]


def proc_prefix_for_cell(cell: dict) -> str:
    """The proc-series prefix the analysis should read for this cell.

    Multi-process cells expose monitors.components: the analysis reads the
    engine aggregate (agg_<engine_group>). Single-process cells read the
    proc monitor's label. Recorded in the manifest as manifest.proc_prefix.
    """
    monitors = cell["monitors"]
    components = monitors.get("components")
    if components:
        return f"agg_{components.get('engine_group', 'engine')}"
    return monitors["proc"]["label"]


class CalibrationError(Exception):
    """A calibration file cannot be accepted for a production run."""


def resolve_calibrated_rate(calib: dict, allow_lower_bound: bool) -> float:
    """Return the calibrated rate to apply, or raise CalibrationError.

    Only an `ok` calibration is accepted by default: a non-saturated ceiling
    (status did_not_saturate / no_stable_point) makes the DoW 'fraction-of-
    ceiling' rate factor meaningless. allow_lower_bound is the explicit operator
    override (recorded in the manifest)."""
    rate_cal = calib.get("rate_calibrated_rps")
    if rate_cal is None:
        raise CalibrationError(
            f"calibration file has no rate_calibrated_rps (status={calib.get('status')!r})")
    status = (calib.get("status") or "").strip().lower()
    if status != "ok" and not allow_lower_bound:
        raise CalibrationError(
            f"calibration status={status!r} (not 'ok'): the ceiling is not a saturated value, "
            f"so the fraction-of-ceiling rate is meaningless. Re-calibrate, or pass "
            f"--allow-lower-bound-calibration to override deliberately.")
    return float(rate_cal)


def merge_component_identity(cell: dict, identity_path: Path) -> None:
    """Inject the bring-up's recorded PGID identity into the cell's component
    specs, IN PLACE, so the materialized components.json scopes the monitor to
    exactly the recorded process groups (not a host-wide cmdline regex).

    The identity file is produced by deploy/dynamo/record_component_pids.py.
    Every component the cell declares must have a recorded entry, else we fail:
    a missing entry would otherwise silently fall back to regex scoping for that
    component, which is exactly the contamination this batch removes.
    """
    components = cell.get("monitors", {}).get("components")
    if not components:
        die("merge_component_identity called on a cell with no monitors.components")
    comp_list = components["components"]
    if not identity_path.exists():
        die(f"component identity file not found: {identity_path} "
            f"(run the bring-up so it records component PGIDs)")
    identity = json.loads(identity_path.read_text())
    recorded = identity.get("components", {})
    missing = [c["label"] for c in comp_list if c["label"] not in recorded]
    if missing:
        die(f"identity file {identity_path} is missing components {missing}; "
            f"recorded={sorted(recorded)}")
    for c in comp_list:
        entry = recorded[c["label"]]
        c["pgids"] = list(entry["pgids"])
        # The recorded instance count is authoritative; flag a cell/bring-up
        # mismatch rather than silently trusting either side.
        rec_expected = int(entry["expected_count"])
        if c.get("expected_count") is not None and int(c["expected_count"]) != rec_expected:
            die(f"component {c['label']}: cell expected_count={c['expected_count']} "
                f"!= recorded {rec_expected} (topology mismatch)")
        c["expected_count"] = rec_expected
    log(f"merged PGID identity for {len(comp_list)} components from {identity_path}")


def spawn_monitors(
    repo_root: Path,
    run_dir: Path,
    cell: dict,
    pidfile: Optional[Path],
    duration_s: int,
    log_dir: Path,
    runs_root: Path,
    run_id: str,
) -> subprocess.Popen:
    monitors = cell["monitors"]
    components = monitors.get("components")
    gpu_indices = ",".join(str(d) for d in gpu_devices_for_cell(cell))
    cmd = [
        sys.executable,
        str(repo_root / "monitoring" / "run_monitors.py"),
        "--run-id", run_id,
        "--runs-root", str(runs_root),
        "--gpu-indices", gpu_indices,
        "--duration-seconds", str(duration_s),
        "--gpu-period", str(monitors["gpu"]["period_s"]),
        "--proc-period", str(monitors["proc"]["period_s"]),
        "--system-period", str(monitors["system"]["period_s"]),
        "--rotation-seconds", str(monitors["rotation_s"]),
    ]
    if components:
        # Multi-process system (e.g. Dynamo): materialize the component spec and
        # let run_monitors spawn the per-component multiproc monitor.
        comp_file = run_dir / "components.json"
        comp_file.write_text(json.dumps(components, indent=2))
        cmd += ["--components-file", str(comp_file)]
    else:
        cmd += ["--pidfile", str(pidfile), "--label-engine", monitors["proc"]["label"]]
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

    # Path-valued fields in the base config are RELATIVE to the base
    # config's directory (client/). When we materialize the run-specific
    # config in run_dir, the run_client.py resolver would look for those
    # paths relative to run_dir and fail. Pre-resolve to absolute here.
    if "corpus_path" in base_cfg:
        corpus = Path(base_cfg["corpus_path"])
        if not corpus.is_absolute():
            base_cfg["corpus_path"] = str((base_config_path.parent / corpus).resolve())

    overrides = render_in_obj(
        cell["workload"]["client_config_overrides"], replica=str(replica)
    )
    # seed_template uses {replica}; resolve into a real seed value.
    if "seed_template" in overrides:
        seed_str = overrides.pop("seed_template")
        overrides["seed"] = int(seed_str)
    # One-level deep merge: a cell override `prompt_len: {median: 50}` should
    # keep the base config's p95/min/max instead of silently dropping them.
    merged = dict(base_cfg)
    for k, v in overrides.items():
        base_v = merged.get(k)
        if isinstance(base_v, dict) and isinstance(v, dict):
            child = dict(base_v)
            child.update(v)
            merged[k] = child
        else:
            merged[k] = v
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


def stop_subprocess(proc: subprocess.Popen, name: str, grace_s: float = 30.0) -> bool:
    """Terminate a process group. Returns True if SIGKILL was needed."""
    if proc.poll() is not None:
        return False
    log(f"stopping {name} (pid={proc.pid})")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return False
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        log(f"{name} did not exit in {grace_s}s, sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return True
    return False


# ---------------------------------------------------------------------------
# Engine lifecycle abstraction
# ---------------------------------------------------------------------------
#
# A cell's engine is brought up, made ready, identified (PID/PGID), and torn
# down through one of these lifecycles, selected by engine.lifecycle in the cell
# yaml. The shared steps (image pin, monitors, client, supervision loop,
# manifest finalize) live in main(); only the lifecycle-specific steps are
# encapsulated here so main() is engine-agnostic.
#
#   single_container  -- EXACTLY the historical behavior: one docker run, one
#                        container PID, one GPU. Manifests are byte-identical to
#                        pre-refactor runs (the standalone/Triton path must not
#                        change in any way).
#   dynamo_disagg     -- multi-container Dynamo disaggregated stack, brought up
#                        via the COMMITTED deploy/dynamo/*.sh scripts (the single
#                        source of truth for how containers start; no docker run
#                        logic is duplicated in Python).


class EngineLifecycle:
    """Interface main() drives. Implementations own the engine-specific steps."""

    kind = "base"

    def __init__(self, cell: dict, args, run_dir: Path, log_dir: Path,
                 pin: dict, image_full: str) -> None:
        self.cell = cell
        self.args = args
        self.run_dir = run_dir
        self.log_dir = log_dir
        self.pin = pin
        self.image_full = image_full

    # -- bring-up: start engine(s), wait ready, GPU sanity, capture baselines,
    #    register abort cleanup. die() on any failure.
    def bring_up(self) -> None:
        raise NotImplementedError

    # -- resolve monitoring identity: (pidfile, pid_daemon). Either may be None
    #    (a components-scoped cell has no single pidfile / daemon).
    def resolve_pid_identity(self) -> tuple[Optional[Path], Optional[subprocess.Popen]]:
        raise NotImplementedError

    # -- GPU whose host_info() goes into the manifest (the "primary" device).
    def primary_gpu(self) -> int:
        raise NotImplementedError

    # -- manifest sections inserted between "image" and "monitors" (key order
    #    matters: single_container must reproduce {"container", "engine"}).
    def manifest_sections(self) -> dict:
        raise NotImplementedError

    # -- manifest baseline section inserted after "warmup_discard_s".
    def manifest_baseline_sections(self) -> dict:
        raise NotImplementedError

    # -- teardown: save docker logs then remove container(s). Records the log
    #    path(s) internally; finalize_manifest() writes them LAST so the manifest
    #    key order is preserved. Must not raise out of the finally block.
    def teardown(self) -> None:
        raise NotImplementedError

    # -- periodic runtime liveness check (~every HEALTH_CHECK_EVERY_S). Return
    #    None if healthy, else a short reason string. The supervision loop acts on
    #    HEALTH_FAIL_CONSECUTIVE consecutive non-None results (isolated failures
    #    are tolerated). Base default: always healthy.
    def health_check(self) -> Optional[str]:
        return None

    # -- post-teardown VRAM quiescence wait on the relevant GPU(s).
    def wait_quiescence(self) -> None:
        raise NotImplementedError

    # -- add the docker-log-path key(s) to the manifest, called LAST.
    def finalize_manifest(self, manifest: dict) -> None:
        raise NotImplementedError


class SingleContainerLifecycle(EngineLifecycle):
    """The historical single-container path, verbatim. Manifest byte-identical."""

    kind = "single_container"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.container_name = self.cell["engine"]["container_name_template"]
        self.gpu_device = self.cell["engine"]["gpu_device"]
        self.baseline_mib = 0
        self.container_pid = 0
        self.docker_cmd: list[str] = []

    def bring_up(self) -> None:
        # 3. Tear down any stale container with the same name.
        teardown_container(self.container_name, self.log_dir / "docker_stale_before_teardown.log")

        # 4. Capture pre-run VRAM baseline on the cell's GPU.
        self.baseline_mib = vram_used_mib(self.gpu_device) or 0
        log(f"pre-run VRAM baseline on gpu {self.gpu_device}: {self.baseline_mib} MiB")

        # 5. Start container.
        self.docker_cmd = build_docker_run_cmd(self.cell, self.container_name)
        log("docker run cmd: " + " ".join(self.docker_cmd))
        result = subprocess.run(self.docker_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            die(f"docker run failed rc={result.returncode}\nstderr: {result.stderr}")
        enable_abort_cleanup(self.container_name, self.log_dir)
        time.sleep(2)  # let docker assign host PID

        # 6. Persist docker inspect snapshot for provenance.
        inspect = docker_inspect(self.container_name)
        (self.run_dir / "docker_inspect.json").write_text(json.dumps(inspect, indent=2))
        self.container_pid = inspect.get("State", {}).get("Pid", 0)
        if not self.container_pid:
            die("container has no PID after docker run")

        # 7. Wait for readiness.
        readyz = self.cell["engine"]["readyz"]
        wait_for_readyz(readyz["url"], int(readyz["timeout_s"]), self.container_name)

        # 8. GPU sanity gate.
        gpu_sanity_check(self.container_pid, self.gpu_device)

    def resolve_pid_identity(self) -> tuple[Optional[Path], Optional[subprocess.Popen]]:
        pidfile = self.run_dir / "engine.pid"
        if pidfile.exists():
            pidfile.unlink()
        pid_daemon = setup_pid_strategy(
            self.cell, self.container_name, pidfile, self.args.repo_root, self.log_dir
        )
        return pidfile, pid_daemon

    def primary_gpu(self) -> int:
        return self.gpu_device

    def manifest_sections(self) -> dict:
        return {
            "container": {
                "name": self.container_name,
                "host_pid": self.container_pid,
                "docker_run_cmd": self.docker_cmd,
            },
            "engine": self.cell["engine"],
        }

    def manifest_baseline_sections(self) -> dict:
        return {"vram_baseline_mib_pre_run": self.baseline_mib}

    def teardown(self) -> None:
        save_docker_logs(self.container_name, self.log_dir / "docker.log")
        log(f"removing container {self.container_name}")
        subprocess.run(["docker", "rm", "-f", self.container_name],
                       check=False, capture_output=True, timeout=60)

    def health_check(self) -> Optional[str]:
        if not container_running(self.container_name):
            return f"engine container {self.container_name} is not running"
        return None

    def wait_quiescence(self) -> None:
        cooldown = int(self.cell.get("post_run_cooldown_s", 600))
        wait_vram_quiescence(self.gpu_device, self.baseline_mib, tolerance_mib=200, max_wait_s=cooldown)

    def finalize_manifest(self, manifest: dict) -> None:
        manifest["docker_log_path"] = str(self.log_dir / "docker.log")


class DynamoDisaggLifecycle(EngineLifecycle):
    """Multi-container Dynamo disaggregated stack via deploy/dynamo/*.sh.

    The shell scripts are the SINGLE source of truth for how containers start
    (--user, HF cache mount, NIXL ports, frontend cache identity, ...). This
    class only orchestrates them, inherits their fail-hard exit codes, and reads
    the PGID identity file record_component_pids.py writes at the end of
    serve_disaggregated.sh. No docker run logic is duplicated here -- that
    duplication is exactly what caused the registry bug we just closed.
    """

    kind = "dynamo_disagg"

    # Container names, matching the env.sh defaults (deploy/dynamo/env.sh). The
    # launcher does not override them, so the defaults hold.
    FRONTEND = "dyn_frontend"
    PREFILL_PREFIX = "dyn_prefill"
    DECODE_PREFIX = "dyn_decode"
    ETCD = "dyn_etcd"
    NATS = "dyn_nats"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        eng = self.cell["engine"]
        topo = eng.get("topology", {})
        self.n_prefill = int(topo.get("n_prefill", 1))
        self.n_decode = int(topo.get("n_decode", 1))
        self.prefill_gpu = int(topo.get("prefill_gpu", 0))
        self.decode_gpu = int(topo.get("decode_gpu", 1))
        # The two GPUs the workers hold; the monitor samples both.
        self.gpus = [self.prefill_gpu, self.decode_gpu]
        eng["gpu_devices"] = list(self.gpus)
        # Identity file the bring-up records and we merge from (same default as
        # attach_run / env.sh COMPONENT_PIDS_FILE).
        self.component_pids = Path(self.args.component_pids)
        # Bring-up budget for the serve script (model load + settle + poll).
        self.bringup_timeout_s = int(eng.get("readyz", {}).get("timeout_s", 1800))
        # /v1/models readiness re-verify target.
        overrides = self.cell["workload"]["client_config_overrides"]
        self.model_name = overrides["model"]
        base_url = overrides.get("base_url", "http://localhost:8400").rstrip("/")
        self.models_url = f"{base_url}/v1/models"
        self.baselines: dict[int, int] = {}
        self.serve_invocations: list[dict] = []
        self._log_paths: dict[str, str] = {}
        self._identity: dict = {}

    def _worker_names(self) -> tuple[list[str], list[str]]:
        prefill = [f"{self.PREFILL_PREFIX}_{i}" for i in range(1, self.n_prefill + 1)]
        decode = [f"{self.DECODE_PREFIX}_{i}" for i in range(1, self.n_decode + 1)]
        return prefill, decode

    def stack_containers(self) -> list[str]:
        prefill, decode = self._worker_names()
        return [self.FRONTEND, *prefill, *decode, self.ETCD, self.NATS]

    def _script_env(self) -> dict:
        env = os.environ.copy()
        env.update({
            "N_PREFILL": str(self.n_prefill),
            "N_DECODE": str(self.n_decode),
            "PREFILL_GPU": str(self.prefill_gpu),
            "DECODE_GPU": str(self.decode_gpu),
            # Keep the served model in lockstep with the client's model.
            "MODEL": str(self.model_name),
            # record_component_pids.py must write where we later read the identity.
            "WOSAR_COMPONENT_PIDS": str(self.component_pids),
        })
        # Any extra env the cell declares for the scripts.
        for k, v in self.cell["engine"].get("env", {}).items():
            env[k] = str(v)
        return env

    def _run_script(self, script: Path, env: dict, log_name: str,
                    timeout_s: Optional[int] = None) -> None:
        """Run a deploy/dynamo/*.sh script fail-hard: capture its output into
        run_dir/logs/, and die() with the log path on non-zero exit."""
        out_path = self.log_dir / f"{log_name}.log"
        # Record the invocation (argv + the extra env we set) for provenance.
        extra_env = {k: v for k, v in env.items() if k in (
            "N_PREFILL", "N_DECODE", "PREFILL_GPU", "DECODE_GPU", "MODEL",
            "WOSAR_COMPONENT_PIDS") or k in self.cell["engine"].get("env", {})}
        argv = ["bash", str(script)]
        self.serve_invocations.append({"script": script.name, "argv": argv, "env": extra_env})
        log(f"running {script.name} (log -> {out_path})")
        with out_path.open("wb") as f:
            try:
                result = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT,
                                        timeout=timeout_s)
            except subprocess.TimeoutExpired:
                die(f"{script.name} timed out after {timeout_s}s; see {out_path}", rc=2)
        if result.returncode != 0:
            die(f"{script.name} failed rc={result.returncode}; see {out_path}", rc=2)

    def _gpu_sanity_per_worker(self) -> None:
        """Adapt gpu_sanity_check to disaggregation: each worker container's host
        PID must appear on its ASSIGNED GPU (prefill_gpu / decode_gpu)."""
        prefill, decode = self._worker_names()
        for name in prefill:
            pid = get_container_pid(name)
            if pid is None:
                die(f"gpu sanity: prefill worker {name} has no container PID", rc=3)
            gpu_sanity_check(pid, self.prefill_gpu)
        for name in decode:
            pid = get_container_pid(name)
            if pid is None:
                die(f"gpu sanity: decode worker {name} has no container PID", rc=3)
            gpu_sanity_check(pid, self.decode_gpu)

    def bring_up(self) -> None:
        deploy = self.args.repo_root / "deploy" / "dynamo"

        # R1-3: sweep EVERY dyn_* engine container (any name/index/topology), not
        # just this run's, BEFORE the baseline -- a stale container from a
        # differently-named prior run (e.g. dyn_worker, dyn_prefill_2) could
        # otherwise survive, re-register against the fresh etcd/NATS, and serve
        # traffic outside our recorded PGID identity. infra_up.sh recreates
        # etcd/nats afterwards, so sweeping them here is harmless.
        stale = all_dyn_containers()
        if stale:
            log(f"[dynamo] sweeping stale dyn_* containers before bring-up: {stale}")
            subprocess.run(["docker", "rm", "-f", *stale],
                           check=False, capture_output=True, timeout=90)
        # Then fail HARD if any dynamo/vllm engine process is still visible on the
        # host (a container removed but its process lingering, or a non-container
        # engine): proceeding would let it serve outside our identity.
        deadline = time.monotonic() + 10
        stray = dynamo_engine_procs_on_host()
        while stray and time.monotonic() < deadline:
            time.sleep(1)
            stray = dynamo_engine_procs_on_host()
        if stray:
            die(f"stale dynamo/vllm engine process(es) still on host after container "
                f"sweep: {stray}; refusing to bring up (kill them, then retry).", rc=3)

        # Baselines on BOTH GPUs before anything starts.
        for g in self.gpus:
            self.baselines[g] = vram_used_mib(g) or 0
            log(f"pre-run VRAM baseline on gpu {g}: {self.baselines[g]} MiB")

        env = self._script_env()
        # Register whole-stack abort cleanup BEFORE the first container starts, so
        # an abort during bring-up tears the entire stack down (not one name).
        enable_abort_cleanup(self.stack_containers(), self.log_dir)

        self._run_script(deploy / "infra_up.sh", env, "infra_up", timeout_s=180)
        # serve_disaggregated.sh blocks until /v1/models is served or exits
        # non-zero (fail-hard); it also writes the PGID identity file at the end.
        self._run_script(deploy / "serve_disaggregated.sh", env, "serve_disaggregated",
                         timeout_s=self.bringup_timeout_s)

        # Independent re-verify that the model is actually listed.
        verify_models_listed(self.models_url, self.model_name, timeout_s=120)

        # GPU sanity: each worker on its assigned device.
        self._gpu_sanity_per_worker()

        # Provenance: per-container docker inspect snapshot of the whole stack.
        stack_inspect = {}
        for name in self.stack_containers():
            try:
                stack_inspect[name] = docker_inspect(name)
            except Exception as e:  # pragma: no cover - best effort
                stack_inspect[name] = {"inspect_error": str(e)}
        (self.run_dir / "docker_inspect.json").write_text(json.dumps(stack_inspect, indent=2))

    def resolve_pid_identity(self) -> tuple[Optional[Path], Optional[subprocess.Popen]]:
        # Multi-process: scope the monitor to the PGIDs the bring-up recorded.
        # merge_component_identity mutates cell["monitors"]["components"] in place
        # and is fail-hard on a missing component or expected_count mismatch.
        merge_component_identity(self.cell, self.component_pids)
        self._identity = json.loads(self.component_pids.read_text()).get("components", {})
        return None, None

    def primary_gpu(self) -> int:
        return self.prefill_gpu

    def manifest_sections(self) -> dict:
        prefill, decode = self._worker_names()
        return {
            "lifecycle": self.kind,
            "engine": self.cell["engine"],
            "topology": {
                "n_prefill": self.n_prefill,
                "n_decode": self.n_decode,
                "prefill_gpu": self.prefill_gpu,
                "decode_gpu": self.decode_gpu,
            },
            "gpu_devices": list(self.gpus),
            "containers": {
                "frontend": self.FRONTEND,
                "prefill": prefill,
                "decode": decode,
                "etcd": self.ETCD,
                "nats": self.NATS,
            },
            # Per-component containers + merged PGIDs, as the monitor will scope.
            "components": self._identity,
            "serve_invocations": self.serve_invocations,
        }

    def manifest_baseline_sections(self) -> dict:
        return {"vram_baselines_mib_pre_run": {str(g): v for g, v in self.baselines.items()}}

    def teardown(self) -> None:
        deploy = self.args.repo_root / "deploy" / "dynamo"
        env = self._script_env()
        # Save logs for EVERY stack container BEFORE tearing anything down.
        for name in self.stack_containers():
            p = self.log_dir / f"docker_{name}.log"
            save_docker_logs(name, p)
            self._log_paths[name] = str(p)
        # serve_down (engine) then infra_down (etcd + nats).
        for script in ("serve_down.sh", "infra_down.sh"):
            try:
                subprocess.run(["bash", str(deploy / script)], env=env,
                               check=False, capture_output=True, timeout=120)
            except subprocess.SubprocessError as e:
                log(f"WARNING: {script} during teardown failed: {e}")
        # Verify no dyn_* container remains; force-remove any straggler.
        try:
            r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=30)
            remaining = [n for n in r.stdout.split() if n.startswith("dyn_")]
            if remaining:
                log(f"WARNING: dyn_* containers remained after teardown: {remaining}; force-removing")
                subprocess.run(["docker", "rm", "-f", *remaining],
                               check=False, capture_output=True, timeout=60)
        except subprocess.SubprocessError as e:
            log(f"WARNING: could not verify dyn_* teardown: {e}")

    def _models_listed_quick(self) -> bool:
        """Short-timeout single GET of /v1/models; True iff it lists the model."""
        try:
            with urllib.request.urlopen(self.models_url, timeout=5) as resp:
                if not (200 <= resp.status < 300):
                    return False
                body = resp.read().decode("utf-8", errors="replace")
            ids = [m.get("id") for m in json.loads(body).get("data", [])]
            return self.model_name in ids
        except Exception:
            return False

    def health_check(self) -> Optional[str]:
        # All stack containers still running?
        down = [c for c in self.stack_containers() if not container_running(c)]
        if down:
            return f"stack container(s) not running: {down}"
        # Frontend still serving the model? (isolated failures tolerated by the
        # caller's consecutive-failure counter.)
        if not self._models_listed_quick():
            return f"/v1/models no longer lists {self.model_name}"
        return None

    def wait_quiescence(self) -> None:
        cooldown = int(self.cell.get("post_run_cooldown_s", 600))
        for g in self.gpus:
            wait_vram_quiescence(g, self.baselines.get(g, 0), tolerance_mib=200, max_wait_s=cooldown)

    def finalize_manifest(self, manifest: dict) -> None:
        manifest["docker_log_paths"] = self._log_paths


def make_lifecycle(cell: dict, args, run_dir: Path, log_dir: Path,
                   pin: dict, image_full: str) -> EngineLifecycle:
    """Select the lifecycle from engine.lifecycle (default single_container)."""
    kind = cell["engine"].get("lifecycle", "single_container")
    impls = {
        "single_container": SingleContainerLifecycle,
        "dynamo_disagg": DynamoDisaggLifecycle,
    }
    if kind not in impls:
        die(f"unknown engine.lifecycle: {kind!r} (expected one of {sorted(impls)})")
    return impls[kind](cell, args, run_dir, log_dir, pin, image_full)


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
    p.add_argument("--attempt", type=int, default=1, help="Attempt number for provenance.")
    p.add_argument(
        "--component-pids",
        type=Path,
        default=Path(os.environ.get("WOSAR_COMPONENT_PIDS",
                                    str(Path.home() / "wosar" / "dynamo_component_pids.json"))),
        help="Multi-process lifecycles (dynamo_disagg): identity file the bring-up "
             "writes (deploy/dynamo/record_component_pids.py). The monitor is scoped "
             "to its recorded PGIDs. Ignored by single_container cells.",
    )
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
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="SC-2 pre-run free-space gate: refuse to start if free space on the "
             "runs-root or the docker data-root is below this (GB).",
    )
    p.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="JSON from scripts/calibrate_rate.py. If given, its "
             "rate_calibrated_rps overrides the cell's target_rate_rps and the "
             "ceiling/fraction/rate are recorded in the manifest. The rate is "
             "fixed for the whole run (no mid-run re-calibration).",
    )
    p.add_argument(
        "--allow-lower-bound-calibration",
        action="store_true",
        help="Permit a calibration whose status != 'ok' (e.g. did_not_saturate, "
             "where the ceiling is only a lower bound). Off by default: a "
             "non-saturated ceiling makes the 'fraction-of-ceiling' rate factor "
             "meaningless and must be an explicit, recorded operator decision.",
    )
    args = p.parse_args()

    # 1. Load and substitute placeholders.
    #
    # replica is zero-padded HERE (Python side) and the cell yamls use
    # plain {replica} placeholders. Doing the format spec in YAML
    # ({replica:02d}) would require a Python-format-aware renderer; we
    # keep the renderer trivial (literal {key} replacement) and centralize
    # the formatting in this one line.
    cell_raw = yaml.safe_load(args.cell_yaml.read_text())
    replica_padded = f"{args.replica:02d}"
    cell = render_in_obj(
        cell_raw,
        repo_root=str(args.repo_root),
        hf_cache_host=str(args.hf_cache_host),
        replica=replica_padded,
    )
    if args.gpu_device_override is not None:
        cell["engine"]["gpu_device"] = args.gpu_device_override
    if args.duration_s_override is not None:
        cell["duration_s"] = args.duration_s_override

    cell_id = cell["cell_id"]
    replica = args.replica
    run_id = f"{args.campaign_id}_{cell_id}_r{replica:02d}"
    run_dir = args.runs_root / run_id
    assert_run_dir_fresh(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    log(f"run_id={run_id}")
    log(f"run_dir={run_dir}")

    # SC-2 #1: pre-run free-space gate (runs-root for CSVs, docker data-root for
    # images + 48h container logs).
    require_free_space([args.runs_root, docker_root_dir()], args.min_free_gb, label=run_id)

    # 2. Verify image pin.
    pin = load_image_pin(args.repo_root / cell["engine"]["digest_pin_file"])
    image_full = f'{cell["engine"]["image_repo"]}:{cell["engine"]["image_tag"]}'
    if pin["image_tag"] != image_full:
        die(f"image pin mismatch: cell expects {image_full}, pin file has {pin['image_tag']}")
    verify_image_present(image_full)
    (run_dir / "image_digest.txt").write_text(pin["digest"] + "\n")
    log(f"image: {image_full}  digest: {pin['digest']}")

    # Install the teardown signal handlers BEFORE engine bring-up and child spawn,
    # so a signal during bring-up / the spawn-vs-record window routes through the
    # graceful teardown path instead of a bare kill that would leave an unreapable
    # orphan client loading the NEXT run's endpoint.
    interrupted = False

    def handle_signal(_sig, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    # A dropped SSH session sends SIGHUP; route it through the same graceful path.
    signal.signal(signal.SIGHUP, handle_signal)

    # Pre-run orphan reaper: kill leftover monitor/client children of a prior run
    # (run-id + script-name verified, PID-reuse-safe) BEFORE the engine stack
    # starts, so a stale sudo'd proc monitor is gone before the new one runs.
    # Engine containers are NOT in reaper scope (the bring-up docker rm -f's stale
    # names); the reaper handles host-side processes only.
    for line in reaper.reap_orphans(args.runs_root, current_run_id=run_id):
        log(line)
    # An entry surviving the reap is a live orphan the reaper could not kill;
    # refuse to start over it (a stray client would load the fresh endpoint).
    stuck = reaper.ledger_run_ids(args.runs_root)
    if stuck:
        die(f"pre-run reaper could not kill recorded orphan(s) from prior run(s) {stuck}; "
            f"refusing to start on a host with an unkillable orphan (kill it manually, then retry).",
            rc=8)

    # 3. Select the engine lifecycle (single_container default / dynamo_disagg)
    #    and bring the engine up through it: teardown-stale, start, readiness,
    #    GPU sanity, and VRAM baseline(s) are all lifecycle-specific.
    lifecycle = make_lifecycle(cell, args, run_dir, log_dir, pin, image_full)
    log(f"engine lifecycle: {lifecycle.kind}")
    lifecycle.bring_up()

    # 4. Resolve monitoring identity: a single pidfile + optional daemon
    #    (single_container), or a components-scoped merge (dynamo_disagg -> None).
    pidfile, pid_daemon = lifecycle.resolve_pid_identity()

    # 5. Write the run manifest (started_at). The lifecycle contributes the
    #    engine/container sections and the VRAM-baseline section; the shared
    #    head/tail keys keep their order so single_container manifests stay
    #    byte-identical to pre-refactor runs.
    started_at_unix = time.time()       # wall clock: manifest timestamp only
    started_mono = time.monotonic()     # monotonic: drives the run-duration decision
    manifest = {
        "run_id": run_id,
        "campaign_id": args.campaign_id,
        "cell_id": cell_id,
        "replica": replica,
        "attempt": args.attempt,
        "started_at": utc_iso(),
        "started_at_unix": started_at_unix,
        "host": host_info(lifecycle.primary_gpu()),
        "git_sha": git_sha(args.repo_root),
        "image": {
            "tag": image_full,
            "digest": pin["digest"],
            "source_tag": pin.get("source_tag"),
            "pinned_at": pin.get("pinned_at"),
        },
    }
    manifest.update(lifecycle.manifest_sections())
    manifest.update({
        "monitors": cell["monitors"],
        "proc_prefix": proc_prefix_for_cell(cell),
        "workload": cell["workload"],
        "duration_s": cell["duration_s"],
        "warmup_discard_s": cell["warmup_discard_s"],
    })
    manifest.update(lifecycle.manifest_baseline_sections())
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # 6. Spawn monitors and client.
    duration_s = int(cell["duration_s"])
    monitors_proc = spawn_monitors(
        args.repo_root, run_dir, cell, pidfile, duration_s, log_dir, args.runs_root, run_id
    )
    log(f"monitors orchestrator pid={monitors_proc.pid}")
    # Record the monitors immediately (client_pid=None) to close the crash window
    # between spawn and ledger record; upsert again once the client is up.
    for line in reaper.record_children(args.runs_root, run_dir, run_id,
                                       monitors_proc.pid, None):
        log(line)

    client_config = materialize_client_config(args.repo_root, run_dir, cell, replica)
    manifest["client_config_path"] = str(client_config)

    # Apply the pre-run rate calibration, if provided. The calibrated rate is
    # fixed for the whole run; capacity erosion over the window is a result we
    # want to observe, not absorb by re-calibrating.
    if args.calibration_file is not None:
        try:
            calib = json.loads(args.calibration_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            die(f"could not read calibration file {args.calibration_file}: {e}")
        try:
            rate_cal = resolve_calibrated_rate(calib, args.allow_lower_bound_calibration)
        except CalibrationError as e:
            die(f"{e} (file: {args.calibration_file})", rc=5)
        cfg = yaml.safe_load(client_config.read_text())
        cfg["target_rate_rps"] = float(rate_cal)
        client_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest["calibration"] = {
            "source_file": str(args.calibration_file),
            "ceiling_rps": calib.get("ceiling_rps"),
            "ceiling_offered_rps": calib.get("ceiling_offered_rps"),
            "fraction": calib.get("fraction"),
            "rate_calibrated_rps": float(rate_cal),
            "status": calib.get("status"),
            "allow_lower_bound_override": bool(args.allow_lower_bound_calibration),
        }
        log(f"calibration applied: ceiling={calib.get('ceiling_rps')} rps -> "
            f"target_rate_rps={rate_cal} (fraction {calib.get('fraction')})")

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    client_proc = spawn_client(args.repo_root, run_dir, client_config, duration_s, log_dir)
    log(f"client pid={client_proc.pid}")

    # Upsert with the client pid now that it is spawned.
    for line in reaper.record_children(args.runs_root, run_dir, run_id,
                                       monitors_proc.pid, client_proc.pid):
        log(line)

    # 12. Supervise until duration elapses or any subprocess exits. Signal handlers
    #     were installed before bring-up (above).
    mono_deadline = started_mono + duration_s

    client_forced_kill = False
    client_summary: dict[str, Any] = {}
    health_fail_streak = 0
    last_health_mono = started_mono
    try:
        while not interrupted:
            time.sleep(5)
            if time.monotonic() >= mono_deadline:
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
            if interrupted:
                break

            # Runtime engine health (~every HEALTH_CHECK_EVERY_S). Without this, a
            # dead engine/container at hour 12 would let the run complete "cleanly"
            # with plausible partial data (client just logs errors). We reach here
            # only with the client still alive (an exited client broke above).
            now_mono = time.monotonic()
            if now_mono - last_health_mono >= HEALTH_CHECK_EVERY_S:
                last_health_mono = now_mono
                reason = lifecycle.health_check()
                if reason is None:
                    health_fail_streak = 0
                else:
                    health_fail_streak += 1
                    log(f"WARNING: engine health check failed "
                        f"({health_fail_streak}/{HEALTH_FAIL_CONSECUTIVE}): {reason}")
                    if health_fail_streak >= HEALTH_FAIL_CONSECUTIVE:
                        log(f"FATAL: engine unhealthy for {health_fail_streak} consecutive "
                            f"checks: {reason}; tearing down")
                        interrupted = True
                        break
                # Endpoint-dead detection: a ~5 min window of client rows with ZERO
                # status=ok while the client is alive == the engine is de facto
                # dead. NOT a drop/error-RATE threshold (high drop rates are a
                # legitimate stress signal); only the all-fail window counts.
                dead_rows = client_all_fail_window(run_dir / "client",
                                                   ENDPOINT_DEAD_WINDOW_S, time.time())
                if dead_rows is not None:
                    log(f"FATAL: endpoint dead -- {dead_rows} client rows in the last "
                        f"{ENDPOINT_DEAD_WINDOW_S}s with zero status=ok; tearing down")
                    interrupted = True
                    break
    finally:
        # 13. Graceful teardown.
        if interrupted:
            client_forced_kill = stop_subprocess(client_proc, "client")
        elif client_proc.poll() is None:
            cfg = yaml.safe_load(client_config.read_text())
            request_timeout_s = float(cfg.get("request_timeout_s", 600))
            client_grace_s = max(120.0, request_timeout_s + 90.0)
            log(f"waiting for client to finish and flush (grace={client_grace_s:.0f}s)")
            try:
                client_proc.wait(timeout=client_grace_s)
            except subprocess.TimeoutExpired:
                log("client did not finish after duration; forcing shutdown")
                client_forced_kill = stop_subprocess(client_proc, "client", grace_s=30.0)
                interrupted = True

        client_summary = summarize_client_csvs(run_dir / "client")
        log(
            "client summary: "
            f"total={client_summary['total']} ok={client_summary['ok']} "
            f"statuses={client_summary['status_counts']}"
        )
        if client_summary["total"] == 0:
            log("WARNING: client produced zero request rows")
            interrupted = True
        elif client_summary["ok"] == 0:
            log("WARNING: client produced request rows but zero status=ok rows")
            interrupted = True

        # 14. Teardown. Each step is isolated so one failure (e.g. a docker rm
        #     timeout while the daemon hangs) is RECORDED but the remaining steps
        #     -- crucially deregister and the final manifest write -- still run.
        #     Unattended diagnosis needs the run's final state exactly when
        #     teardown is going wrong.
        teardown_errors: list[dict] = []

        def _teardown_step(name: str, fn) -> None:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - teardown must never abort teardown
                log(f"WARNING: teardown step '{name}' failed: {e!r}")
                teardown_errors.append({"step": name, "error": repr(e)})

        _teardown_step("stop_monitors",
                       lambda: stop_subprocess(monitors_proc, "monitors", grace_s=60.0))
        if pid_daemon is not None:
            _teardown_step("stop_pid_daemon", lambda: stop_subprocess(pid_daemon, "pid_daemon"))
        # Engine teardown: dynamo_disagg tears down the whole stack (serve_down +
        # infra_down + a dyn_* sweep); single_container is one docker rm -f.
        _teardown_step("engine_teardown", lifecycle.teardown)
        disable_abort_cleanup()

        # Always deregister (even if a step above failed): a torn-down run must not
        # linger as a reap candidate for the next launch.
        def _deregister() -> None:
            for line in reaper.deregister_run(args.runs_root, run_id):
                log(line)
        _teardown_step("deregister", _deregister)

        # VRAM quiescence on the cell's GPU(s).
        _teardown_step("vram_quiescence", lifecycle.wait_quiescence)

        # 15. Finalize manifest -- ALWAYS, even if teardown steps failed.
        #     teardown_errors is present only when non-empty (single_container
        #     byte-compat holds for clean runs); lifecycle.finalize_manifest adds
        #     the docker-log path key(s) LAST so the clean-run key order is intact.
        ended_at_unix = time.time()
        manifest["ended_at"] = utc_iso()
        manifest["ended_at_unix"] = ended_at_unix
        manifest["duration_seconds_actual"] = time.monotonic() - started_mono
        manifest["interrupted_early"] = interrupted
        manifest["client_forced_kill"] = client_forced_kill
        manifest["client_summary"] = client_summary
        if teardown_errors:
            manifest["teardown_errors"] = teardown_errors
        lifecycle.finalize_manifest(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        log(f"done. duration={manifest['duration_seconds_actual']:.0f}s "
            f"interrupted={interrupted} teardown_errors={len(teardown_errors)}")

    # Non-zero exit if the run was interrupted OR any teardown step failed.
    sys.exit(0 if (not interrupted and not teardown_errors) else 2)


if __name__ == "__main__":
    main()
