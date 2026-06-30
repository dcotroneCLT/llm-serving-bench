"""Orphan reaper for the monitor + client children of a run.

launch_cell.py and attach_run.py spawn the monitor orchestrator and the load
client in NEW sessions (start_new_session=True), and only clean them up in their
`finally` block. A SIGKILL / OOM / host-crash / tmux-kill of the launcher leaves
those children (and, under them, the sudo'd proc monitor and the per-device GPU /
system monitors) running. An orphan client keeps offering load to the engine and
an orphan monitor keeps writing CSVs and holding the sudo proc read, which
contaminates the NEXT run (a retry, especially).

This module records each run's children to `run_dir/child_pids.json` and to a
small per-runs-root ledger `<runs-root>/.active_children.json`, and reaps any
survivors of a PRIOR run before a new one starts.

Safety (the whole point): a recorded PID is killed ONLY if it is still alive AND
its live cmdline still contains BOTH the recorded run-id and the expected script
name. That makes the kill safe against PID reuse (a recycled PID running an
unrelated program will not match) and scoped to OUR processes. Root-owned
children (the sudo'd proc monitor) are killed via `sudo -n kill` when a direct
signal is refused. Process GROUPS are signalled (killpg) so a child's own
subprocess tree goes with it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except ImportError as e:  # pragma: no cover
    raise SystemExit("psutil not installed. Run: pip install psutil") from e

LEDGER_NAME = ".active_children.json"

# Only ever touch processes whose cmdline names one of these (defence in depth on
# top of the run-id match): our launcher children and their monitor descendants.
OUR_SCRIPTS = (
    "run_monitors.py", "multiproc_monitor.py", "proc_monitor.py",
    "gpu_monitor.py", "system_monitor.py", "run_client.py",
)


def _cmdline(pid: int) -> Optional[str]:
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _is_ours(pid: int, run_id: str) -> Optional[str]:
    """Return the live cmdline iff pid is alive, names one of OUR_SCRIPTS, and
    carries this run_id (PID-reuse-safe). Otherwise None (do NOT kill)."""
    cl = _cmdline(pid)
    if cl is None:
        return None
    if run_id and run_id not in cl:
        return None
    if not any(s in cl for s in OUR_SCRIPTS):
        return None
    return cl


def _kill_pgid(pid: int) -> bool:
    """SIGKILL the process group of pid; fall back to sudo for root-owned ones.
    Returns True if a kill was issued."""
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return False
    try:
        os.killpg(pgid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Root-owned (the sudo'd proc monitor): use the NOPASSWD sudo path.
        subprocess.run(["sudo", "-n", "kill", "-9", f"-{pgid}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True


def _ledger_path(runs_root: Path) -> Path:
    return runs_root / LEDGER_NAME


def _candidate_pids(run_dir: Path, monitors_pid: Optional[int], client_pid: Optional[int]) -> list[int]:
    """All pids that belong to a run: the launcher's direct children plus the
    monitor grandchildren that run_monitors recorded in monitor_manifest.json
    (each spawned in its own session, so killpg of the orchestrator misses them)."""
    pids: list[int] = []
    for p in (monitors_pid, client_pid):
        if p:
            pids.append(int(p))
    mm = run_dir / "monitor_manifest.json"
    if mm.exists():
        try:
            for m in json.loads(mm.read_text()).get("monitors", []):
                if m.get("pid"):
                    pids.append(int(m["pid"]))
        except (OSError, json.JSONDecodeError):
            pass
    # de-dup, preserve order
    seen: set[int] = set()
    return [p for p in pids if not (p in seen or seen.add(p))]


def record_children(runs_root: Path, run_dir: Path, run_id: str,
                    monitors_pid: Optional[int], client_pid: Optional[int]) -> None:
    """Write run_dir/child_pids.json and upsert the run into the runs-root ledger."""
    entry: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "recorded_at_unix": time.time(),
        "monitors_pid": monitors_pid,
        "client_pid": client_pid,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "child_pids.json").write_text(json.dumps(entry, indent=2))
    ledger = _ledger_path(runs_root)
    runs: list[dict] = []
    if ledger.exists():
        try:
            runs = [r for r in json.loads(ledger.read_text()) if r.get("run_id") != run_id]
        except (OSError, json.JSONDecodeError):
            runs = []
    runs.append(entry)
    ledger.write_text(json.dumps(runs, indent=2))


def reap_orphans(runs_root: Path, current_run_id: Optional[str] = None) -> list[str]:
    """Kill survivors of runs recorded in the ledger, then clear it. Returns log
    lines. Safe to call at the start of a run BEFORE its own children are spawned:
    nothing of the current run is in the ledger yet, so every recorded pid is a
    prior-run candidate, gated by the _is_ours run-id + script-name check.

    current_run_id is advisory (kept for call-site clarity / logging); the real
    protection is temporal (reap precedes record_children) plus the cmdline gate.
    """
    ledger = _ledger_path(runs_root)
    if not ledger.exists():
        return ["[reaper] no ledger; nothing to reap"]
    try:
        runs = json.loads(ledger.read_text())
    except (OSError, json.JSONDecodeError):
        return ["[reaper] ledger unreadable; skipping"]

    out: list[str] = []
    for r in runs:
        run_id = r.get("run_id", "")
        run_dir = Path(r.get("run_dir", ""))
        killed = 0
        for pid in _candidate_pids(run_dir, r.get("monitors_pid"), r.get("client_pid")):
            cl = _is_ours(pid, run_id)
            if cl is None:
                continue
            if _kill_pgid(pid):
                killed += 1
                out.append(f"[reaper] killed orphan pid={pid} run={run_id}: {cl[:80]}")
        if killed == 0:
            out.append(f"[reaper] run {run_id}: no live orphans")
    # Every recorded run has been handled; clear the ledger (the current run
    # re-adds itself via record_children right after spawning).
    ledger.write_text("[]")
    return out
