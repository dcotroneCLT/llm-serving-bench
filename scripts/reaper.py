"""Orphan reaper for the monitor + client children of a run.

launch_cell.py and attach_run.py spawn the monitor orchestrator and the load
client in NEW sessions (start_new_session=True), and only clean them up in their
`finally` block. A SIGKILL / OOM / host-crash / tmux-kill of the launcher leaves
those children (and, under them, the sudo'd proc monitor and the per-device GPU /
system monitors) running. An orphan client keeps offering load to the engine and
an orphan monitor keeps writing CSVs and holding the sudo proc read, which
contaminates the NEXT run (a retry, especially).

This module records each run's children to `run_dir/child_pids.json` and to a
small per-runs-root ledger `<runs-root>/.active_children.json`, reaps any
survivors of a PRIOR run before a new one starts, and lets a cleanly-finished run
deregister itself so it never lingers as a reap candidate.

Both launch_cell.py (either lifecycle) and attach_run.py wire it the same way:
reap_orphans() early (before the engine stack starts, so a stale sudo'd monitor
is gone first), record_children() right after spawning the monitors + client, and
deregister_run() in the teardown once those children are stopped.

Ledger integrity: every mutation (record_children's upsert, deregister_run's
removal, and the clear reap_orphans performs) goes through one internal
read-modify-write helper that holds an exclusive flock on
`<runs-root>/.active_children.json.lock` for the whole operation and swaps the
new contents in via an os.replace() of a same-directory temp file, so a
concurrent launcher never loses an entry and no partially-written JSON is ever
visible. A missing or corrupt ledger is treated as empty and REPORTED (a log
line), never fatal. Plain reads need no lock (os.replace keeps them consistent).

Safety (the whole point): a recorded PID is killed ONLY if it is still alive AND
its live cmdline still contains BOTH the recorded run-id and the expected script
name. That makes the kill safe against PID reuse (a recycled PID running an
unrelated program will not match) and scoped to OUR processes. Root-owned
children (the sudo'd proc monitor) are killed via `sudo -n kill` when a direct
signal is refused. Process GROUPS are signalled (killpg) so a child's own
subprocess tree goes with it.

A kill is CONFIRMED (the target must actually be gone) before the ledger entry is
dropped; the sudo fallback checks its return code and is bounded by a timeout. If
a live orphan cannot be killed, its ledger entry is RETAINED and a loud line is
emitted, and the pre-run reap callers (launch_cell, attach_run) refuse to start a
new run on a host that still carries an unkillable recorded orphan.

Concurrent-start protection (the campaign is strictly serial): each entry records
the launcher's pid + create_time, and reap_orphans reaps an entry ONLY if that
launcher is gone (pid dead, or reused with a different create_time). A genuinely
alive launcher is an active run -- its entry is retained and the caller refuses to
start. Entries written by older code (no launcher fields) are treated as
launcher-dead (reapable). As a second line of defense, reap also recovers runs
from run_dir/child_pids.json (written before the ledger upsert) so a launcher that
died between the client spawn and the ledger write still gets its client reaped. A
recovered file confirmed dead (no live orphan) is then removed so completed runs
are not re-scanned/re-logged forever; it is kept when a reap failed (evidence).

Atomic serial-run guard: acquire_run_slot() takes a non-blocking exclusive flock
on <runs_root>/.run_slot.lock, held for the whole run, so two launchers starting
from an empty ledger cannot both proceed (the ledger check alone is not atomic
across reap->bring-up->record). While that slot is held, reap_host_wide() can
safely sweep ANY OUR_SCRIPTS process referencing this runs-root as an orphan --
catching a hard crash between the client spawn and the ledger upsert.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import psutil
except ImportError as e:  # pragma: no cover
    raise SystemExit("psutil not installed. Run: pip install psutil") from e

LEDGER_NAME = ".active_children.json"
LOCK_NAME = LEDGER_NAME + ".lock"
RUN_SLOT_LOCK_NAME = ".run_slot.lock"

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


def _confirm_gone(pid: int, timeout_s: float = 3.0) -> bool:
    """True once pid is no longer a live process (or is a zombie awaiting reap).
    Polls up to timeout_s because SIGKILL delivery is asynchronous. A pid we
    cannot inspect (AccessDenied, e.g. still-running root process) counts as
    ALIVE -- we must not declare a live orphan dead."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True  # dead, just not yet reaped; holds no resources
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            pass  # exists but not inspectable -> treat as alive
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _kill_pgid(pid: int) -> bool:
    """SIGKILL the process group of pid (sudo -n fallback for root-owned ones) and
    CONFIRM the process is actually gone. Returns True ONLY if confirmed
    terminated; False means a still-live orphan we could not kill -- the caller
    must retain the ledger entry rather than silently lose the only record of it.
    """
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        return True  # already gone
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True  # gone between getpgid and kill
    except PermissionError:
        # Root-owned (the sudo'd proc monitor): NOPASSWD sudo path, but CHECK the
        # return code and bound it with a timeout -- a refused/hung sudo must not
        # be mistaken for a successful kill.
        try:
            r = subprocess.run(["sudo", "-n", "kill", "-9", f"-{pgid}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10)
        except (subprocess.SubprocessError, OSError):
            return _confirm_gone(pid)  # sudo unavailable/timed out: gone only if truly dead
        if r.returncode != 0:
            return _confirm_gone(pid)  # sudo refused: gone only if it already died
    # Direct kill or sudo kill issued; verify the process actually terminated.
    return _confirm_gone(pid)


def _ledger_path(runs_root: Path) -> Path:
    return runs_root / LEDGER_NAME


def ledger_run_ids(runs_root) -> list[str]:
    """Plain (lock-free; os.replace keeps reads consistent) list of run_ids
    currently in the ledger. The pre-run reap callers use this to GATE: any entry
    surviving reap_orphans() is a live orphan the reaper could not kill, and the
    next run must refuse to start on that host."""
    runs, _ = _read_ledger(_ledger_path(Path(runs_root)))
    return [r.get("run_id", "") for r in runs]


def _lock_path(runs_root: Path) -> Path:
    return runs_root / LOCK_NAME


def acquire_run_slot(runs_root):
    """Atomic serial-run guard (R3-1). Take a NON-BLOCKING exclusive flock on
    <runs_root>/.run_slot.lock and return the open file object on success, or None
    if another launcher already holds it. The caller MUST keep the returned object
    for the whole run (flock is released when the fd is closed or the process
    dies, so a stale lock file is harmless). This closes the reap->bring-up->record
    gap that R2-2's ledger check alone cannot make atomic: two launchers starting
    from an empty ledger cannot both hold the slot."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    f = open(runs_root / RUN_SLOT_LOCK_NAME, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        f.close()
        return None
    return f


def reap_host_wide(runs_root, current_run_id: Optional[str] = None) -> tuple[list[str], list[int]]:
    """Host-wide fallback sweep -- SAFE ONLY while the caller holds the run-slot
    lock (R3-2). With the exclusive slot held, no other legitimate launcher is
    running on this runs-root, so ANY live process that both (a) matches one of
    OUR_SCRIPTS and (b) references this runs-root in its cmdline, and is not part
    of the current run, is an orphan the ledger may have missed -- e.g. a
    SIGKILL/OOM of a prior launcher between its client spawn and its ledger upsert
    (client_pid=None in the ledger). Matching the LIVE cmdline is PID-reuse-safe;
    the kill is verified (R1-4). Returns (log_lines, unkillable_pids); a non-empty
    unkillable list means the caller must refuse to start."""
    runs_root = Path(runs_root)
    root_str = str(runs_root)
    out: list[str] = []
    unkillable: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(proc.info.get("cmdline") or [])
        except Exception:
            continue
        if not cl or root_str not in cl:
            continue
        if not any(s in cl for s in OUR_SCRIPTS):
            continue
        if current_run_id and current_run_id in cl:
            continue  # belongs to the current run (defensive; none spawned pre-run)
        pid = proc.info["pid"]
        if _kill_pgid(pid):
            out.append(f"[reaper] host-wide killed orphan pid={pid}: {cl[:80]}")
        else:
            unkillable.append(pid)
            out.append(f"[reaper] FATAL: host-wide could NOT kill orphan pid={pid}: {cl[:80]}")
    if not out:
        out.append("[reaper] host-wide sweep: no orphan referencing this runs-root")
    return out, unkillable


def _read_ledger(ledger: Path) -> tuple[list[dict], Optional[str]]:
    """Return (runs, warning). A missing ledger is empty with no warning; a
    corrupt one (bad JSON / not a list) is treated as empty WITH a warning so the
    caller can log it. Never raises: ledger damage must not crash a run."""
    if not ledger.exists():
        return [], None
    try:
        data = json.loads(ledger.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [], f"[reaper] ledger {ledger} unreadable ({e}); treating as empty"
    if not isinstance(data, list):
        return [], f"[reaper] ledger {ledger} is not a JSON list; treating as empty"
    return data, None


def _atomic_write_ledger(ledger: Path, runs: list[dict]) -> None:
    """Write runs to a same-directory temp file, then os.replace() over the
    ledger so no partially-written JSON is ever visible to a concurrent reader."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(ledger.parent), prefix=LEDGER_NAME + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(runs, f, indent=2)
        os.replace(tmp, ledger)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _mutate_ledger(runs_root: Path,
                   transform: Callable[[list[dict], list[str]], list[dict]]) -> list[str]:
    """Locked read-modify-write of the ledger. Holds an exclusive flock for the
    WHOLE operation, so concurrent launchers serialize and no entry is lost.

    `transform(runs, out) -> new_runs` receives the current entries and a log-line
    list it may append to (e.g. reap kill lines); its return value is written back
    atomically. A missing/corrupt ledger is passed as [] and reported in `out`.
    Returns the accumulated log lines.
    """
    ledger = _ledger_path(runs_root)
    lock = _lock_path(runs_root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    with open(lock, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            runs, warning = _read_ledger(ledger)
            if warning:
                out.append(warning)
            new_runs = transform(runs, out)
            _atomic_write_ledger(ledger, new_runs)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return out


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


def _launcher_identity() -> tuple[int, Optional[float]]:
    """(pid, create_time) of THIS launcher process. create_time anchors the pid
    against reuse: a future process reusing the pid will have a different
    create_time, so a stale entry is not mistaken for a live launcher."""
    pid = os.getpid()
    try:
        ct = psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        ct = None
    return pid, ct


def _launcher_alive(entry: dict) -> bool:
    """True iff the launcher recorded in `entry` is genuinely still running (same
    pid AND same create_time). Entries without the launcher fields (written by
    older code) are treated as launcher-dead (reapable), preserving prior
    behavior. An inspectable-but-ambiguous case (AccessDenied) errs on ALIVE so
    we never reap a live run's children."""
    lpid = entry.get("launcher_pid")
    lct = entry.get("launcher_create_time")
    if lpid is None or lct is None:
        return False  # old-format entry -> reapable (prior behavior)
    try:
        pid = int(lpid)
    except (TypeError, ValueError):
        return False
    try:
        ct = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return False  # launcher pid gone
    except psutil.AccessDenied:
        return True   # exists but not inspectable -> assume alive (do not reap)
    return abs(ct - float(lct)) < 1.0  # same pid AND create_time -> live launcher


def record_children(runs_root: Path, run_dir: Path, run_id: str,
                    monitors_pid: Optional[int], client_pid: Optional[int]) -> list[str]:
    """Write run_dir/child_pids.json and upsert the run into the runs-root ledger.
    The upsert is a locked, atomic read-modify-write. Returns log lines (normally
    empty; carries a corrupt-ledger report if one was found)."""
    # Be forgiving on argument types: a str runs_root/run_dir must work. A cleanup
    # tool that raises TypeError on a str path is useless at the moment a crashed
    # run most needs reaping.
    runs_root = Path(runs_root)
    run_dir = Path(run_dir)
    launcher_pid, launcher_ct = _launcher_identity()
    entry: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "recorded_at_unix": time.time(),
        "monitors_pid": monitors_pid,
        "client_pid": client_pid,
        # Concurrent-start protection (R2-2): the reaper only reaps an entry whose
        # launcher is gone; a live launcher means an active run, not an orphan.
        "launcher_pid": launcher_pid,
        "launcher_create_time": launcher_ct,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "child_pids.json").write_text(json.dumps(entry, indent=2))

    def _upsert(runs: list[dict], out: list[str]) -> list[dict]:
        return [r for r in runs if r.get("run_id") != run_id] + [entry]

    return _mutate_ledger(runs_root, _upsert)


def deregister_run(runs_root: Path, run_id: str) -> list[str]:
    """Remove a run's entry from the ledger (locked, atomic). Called on a clean
    teardown so a finished run is never a reap candidate for the next launch.
    Returns log lines (normally empty; carries a corrupt-ledger report if any)."""
    runs_root = Path(runs_root)  # forgiving on a str argument
    def _remove(runs: list[dict], out: list[str]) -> list[dict]:
        return [r for r in runs if r.get("run_id") != run_id]

    return _mutate_ledger(runs_root, _remove)


def launcher_alive_for(runs_root, run_id: str) -> bool:
    """Read-only: True iff the ledger holds an entry for run_id whose launcher
    process is genuinely still alive (pid AND create_time match). A run with no
    ledger entry (crashed before its record, or already deregistered) -> False.

    Used by the campaign's stale-running recovery (hardening item 3) to tell a
    run stranded by a host restart from one whose launcher is still live."""
    runs, _ = _read_ledger(_ledger_path(Path(runs_root)))
    for e in runs:
        if e.get("run_id") == run_id:
            return _launcher_alive(e)
    return False


def recorded_children_alive(runs_root, run_id: str) -> bool:
    """Read-only: True iff any child recorded for run_id (in the ledger AND/OR in
    a run_dir/child_pids.json written before the ledger upsert) is still a live
    process that is ours (run-id + OUR_SCRIPTS match, PID-reuse-safe via
    _is_ours). Kills nothing -- it only observes, for the stale-running recovery
    decision (item 3): a recovery must refuse while any recorded child breathes."""
    runs_root = Path(runs_root)
    entries: list[dict] = []
    runs, _ = _read_ledger(_ledger_path(runs_root))
    entries.extend(e for e in runs if e.get("run_id") == run_id)
    try:
        child_files = sorted(runs_root.glob("*/child_pids.json"))
    except OSError:
        child_files = []
    for cp in child_files:
        try:
            e = json.loads(cp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if e.get("run_id") == run_id:
            entries.append(e)
    for e in entries:
        run_dir = Path(e.get("run_dir", ""))
        for pid in _candidate_pids(run_dir, e.get("monitors_pid"), e.get("client_pid")):
            if _is_ours(pid, run_id) is not None:
                return True
    return False


def reap_orphans(runs_root: Path, current_run_id: Optional[str] = None) -> list[str]:
    """Kill survivors of runs recorded in the ledger, then clear it (locked,
    atomic). Returns log lines. Safe to call at the start of a run BEFORE its own
    children are spawned: nothing of the current run is in the ledger yet, so
    every recorded pid is a prior-run candidate, gated by the _is_ours run-id +
    script-name check.

    current_run_id is advisory (kept for call-site clarity / logging); the real
    protection is temporal (reap precedes record_children) plus the cmdline gate.
    The kill semantics (run-id + OUR_SCRIPTS match, pgid kill, decoy-sparing) are
    unchanged; only the ledger rewrite is now locked/atomic, and an entry whose
    live orphan could NOT be killed is RETAINED (not silently cleared) so the
    caller can refuse to start a new run over an unkillable orphan.
    """
    runs_root = Path(runs_root)  # forgiving on a str argument
    def _reap(runs: list[dict], out: list[str]) -> list[dict]:
        by_id: dict[str, dict] = {}
        for r in runs:
            rid = r.get("run_id", "")
            if rid:
                by_id[rid] = r
        # R2-1(b) second line of defense: recover runs whose ledger upsert never
        # landed (e.g. record_children raised after the client spawned) from the
        # run_dir/child_pids.json each run writes BEFORE the ledger upsert. A
        # finished run's file is harmless -- its pids are dead, so nothing matches.
        try:
            child_files = sorted(runs_root.glob("*/child_pids.json"))
        except OSError:
            child_files = []
        recovered_files: dict[str, Path] = {}  # run_id -> its child_pids.json (recovered)
        for cp in child_files:
            try:
                e = json.loads(cp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rid = e.get("run_id", "")
            if rid and rid not in by_id:
                by_id[rid] = e
                recovered_files[rid] = cp
                out.append(f"[reaper] recovered run {rid} from {cp.name} (not in ledger)")
        if not by_id:
            out.append("[reaper] no recorded runs; nothing to reap")
            return []

        retained: list[dict] = []
        for run_id, r in by_id.items():
            retained_this = False
            # R2-2: never reap a run whose launcher is genuinely alive -- that is
            # an active run (an accidental concurrent start), not an orphan. Retain
            # it so the caller refuses to start over it.
            if _launcher_alive(r):
                out.append(f"[reaper] run {run_id}: launcher pid={r.get('launcher_pid')} is ALIVE; "
                           f"NOT reaping (active run / accidental concurrent start)")
                retained.append(r)
                retained_this = True
            else:
                run_dir = Path(r.get("run_dir", ""))
                killed = 0
                unkillable = 0
                for pid in _candidate_pids(run_dir, r.get("monitors_pid"), r.get("client_pid")):
                    cl = _is_ours(pid, run_id)
                    if cl is None:
                        continue
                    if _kill_pgid(pid):
                        killed += 1
                        out.append(f"[reaper] killed orphan pid={pid} run={run_id}: {cl[:80]}")
                    else:
                        unkillable += 1
                        out.append(f"[reaper] FATAL: could NOT kill live orphan pid={pid} "
                                   f"run={run_id}: {cl[:80]}; ledger entry RETAINED")
                if unkillable:
                    # Keep the entry: a live orphan we could not kill must never be
                    # dropped from the record. The caller treats a retained entry as
                    # fatal for the next run.
                    retained.append(r)
                    retained_this = True
                elif killed == 0:
                    out.append(f"[reaper] run {run_id}: no live orphans")
            # R3-5: a RECOVERED run confirmed dead with no live orphans leaves a
            # stale child_pids.json that would be re-scanned and re-logged on every
            # future launch. Remove it. KEEP it when the entry was retained (active
            # launcher, or an unkillable orphan = evidence).
            if run_id in recovered_files and not retained_this:
                try:
                    recovered_files[run_id].unlink()
                    out.append(f"[reaper] removed stale child_pids.json for completed run {run_id}")
                except OSError:
                    pass
        # Cleaned runs are dropped; active-launcher and unkillable-orphan runs
        # remain (the current run re-adds itself via record_children after spawn).
        return retained

    return _mutate_ledger(runs_root, _reap)
