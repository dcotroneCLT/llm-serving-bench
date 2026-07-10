"""Serial campaign orchestrator for the WoSAR 2026 extension (DoW) campaign.

The extension campaign is ~57 runs of 48h each and is STRICTLY SERIAL by
design: measurement isolation demands one run at a time, and the
dynamo_disagg cells occupy both GPUs anyway. There is no thread pool and no
parallel-slot model. One global ordered queue of (cell, replica) runs is
dispatched, one launch_cell.py subprocess at a time, each waited to full
completion (including teardown and VRAM quiescence inside launch_cell) plus a
configurable inter-run cooldown before the next run starts.

    History note: the earlier n=3 (`campaigns/wosar2026/`) campaign used a
    parallel GPU-slot scheduler (thread-per-slot). That campaign is complete;
    its runtime path is intentionally NOT preserved here. The git history of
    this file preserves the slot model; this revision replaces it with the
    serial scheduler the extension requires. A campaign yaml that tries to
    express parallelism (a `slots:` key, or `mode:` other than "serial") is
    rejected at load time.

Campaign yaml schema (see campaigns/extension/campaign.yaml):

    campaign_id: extension_dow
    mode: serial                 # REQUIRED, must equal "serial"
    replicas_per_cell: 3
    order: round_robin           # round_robin | cell_at_a_time
    inter_run_cooldown_s: 300    # extra wait AFTER launch_cell fully exits
    est_run_overhead_s: 3600     # added to each cell duration_s for the estimate
    min_free_gb: 50              # pre-flight free-space gate on runs_root
    allow_lower_bound_calibration: false   # global default; per-cell override
    retry_policy:
      max_retries: 1             # one automatic re-attempt, then mark failed
    runs_root: /home/dcotrone/wosar/runs
    paths:
      hf_cache_host: /home/dcotrone/wosar/hf_cache
      repo_root: /home/dcotrone/wosar/llm-serving-bench   # optional; else inferred
    state_file: state/campaign_state.json                 # relative to this yaml
    cells:
      - id: val_vllm
        yaml: cells/val_vllm.yaml
        calibration_file: calibration/val_vllm.json   # optional, relative to yaml
        calibration_required: true                    # fail pre-flight if invalid
        allow_lower_bound_calibration: false          # optional per-cell override

Retry policy (requirement 3):
  * launch_cell exit 0            -> completed.
  * launch_cell exit 6 (image-digest / run_dir precondition gate), 7 (free-space
    gate), 8 (orphan gate) or 9 (run-slot lock) -- the set launch_cell exports
    as NONRETRYABLE_EXIT_CODES -> NOT a run failure. These are
    host/precondition fatals: a precondition refused, a filesystem is too
    full/undeterminable, or something else owns the host. Retrying cannot fix
    them, so this is a CAMPAIGN-LEVEL FATAL -- stop the whole campaign loudly
    (do NOT burn a retry) and exit non-zero.
  * any other non-zero exit      -> run failure. Re-attempt once (--attempt
    incremented), then mark failed and move on to the next run.

Resumability (requirement 4):
  campaign_state.json records per-run status (pending|running|completed|failed|
  interrupted|host_conflict|insufficient_space|precondition_failed), written
  atomically (write-tmp + rename). The last three are the persisted labels for
  the non-retryable launch_cell fatals (NONRETRYABLE_EXIT_CODES).

Durable, honest retry accounting (hardening item 2):
  * failed is TERMINAL. --resume skips completed AND failed runs; it re-queues
    only the genuinely-unfinished ones (interrupted / host_conflict /
    insufficient_space / precondition_failed / running). To retry a failed run
    the operator must pass --rerun-failed, which explicitly resets that run's
    attempt budget (attempts -> 0, status -> pending) and logs it.
  * the retry decision counts PERSISTED attempts (RunStatus.attempts, which
    survives every --resume), not a session-local counter. A run can therefore
    never exceed max_retries+1 total launches across any number of resumes.
  * a campaign that drains its queue with ANY failed run exits
    EXIT_COMPLETED_WITH_FAILURES (10), not 0. Exit 0 STRICTLY means every
    scheduled run completed.

Signals (requirement 5):
  SIGTERM/SIGINT/SIGHUP forward SIGTERM to the current launch_cell child (which
  tears down gracefully), persist state, and exit non-zero. Campaign runs live
  inside tmux but must survive its loss -- hence the log file (requirement 7).

Usage:

  python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --dry-run
  python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --start
  python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --resume

Recommended deployment (survives ssh disconnect; the log file survives tmux loss):

  tmux new -d -s ext_campaign \\
      'python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --start'
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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml  # type: ignore

# launch_cell is a sibling module. We reuse its calibration gate so the campaign
# pre-flight accepts EXACTLY what launch_cell will accept at run time -- the
# contract in launch_cell.py is the single source of truth (a pre-flight that
# diverged would defeat requirement 6).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_cell  # noqa: E402


# ---------------------------------------------------------------------------
# Exit codes (campaign.py's own) and the launch_cell contract codes.
# ---------------------------------------------------------------------------

EXIT_OK = 0               # STRICT: every scheduled run completed (no failures)
EXIT_USAGE = 2            # argparse-style usage error
EXIT_PREFLIGHT = 3        # pre-flight failed (calibration/config/free space)
EXIT_INTERRUPTED = 4      # stopped by a signal
EXIT_CAMPAIGN_FATAL = 5   # a child reported a non-retryable fatal (launch_cell.NONRETRYABLE_EXIT_CODES: 6/7/8/9)
# The queue drained but one or more runs ended FAILED (retry budget exhausted).
# Distinct from 0 so an unattended campaign cannot look clean when it is not:
# exit 0 means every scheduled run completed; exit 10 means "finished, but with
# failures -- inspect them". Not in launch_cell.NONRETRYABLE_EXIT_CODES (6/7/8/9)
# and above them by design, so it never collides with a child fatal code.
EXIT_COMPLETED_WITH_FAILURES = 10

# TERMINAL run statuses are never re-queued by pending_specs(): a run either
# completed, or FAILED after exhausting its (persisted) attempt budget. Every
# other status (interrupted / host_conflict / insufficient_space /
# precondition_failed / running) is a genuinely-unfinished run that --resume
# re-queues. failed leaves this set only via the explicit --rerun-failed reset.
TERMINAL_STATUSES = frozenset({"completed", "failed"})

# launch_cell exit codes that are HOST/PRECONDITION fatals, not run failures:
# retrying cannot fix them (something else owns the host, a filesystem is too
# full / undeterminable, or the pinned image is wrong). They stop the campaign
# loudly instead of burning retries (requirement 3, extended to the precondition
# gates on review). These MUST stay in sync with launch_cell's own contract:
# every code launch_cell.NONRETRYABLE_EXIT_CODES lists is enforced fatal here
# (cross-checked by test_launch_cell_nonretryable_codes_are_all_campaign_fatal).
LC_PRECONDITION = 6       # non-retryable precondition gate: image digest mismatch OR run_dir not fresh
LC_FREE_SPACE = 7         # free-space gate (runs-root or docker data-root)
LC_ORPHAN_GATE = 8        # pre-run reaper / host-wide reaper: unkillable orphan
LC_SLOT_LOCKED = 9        # run-slot flock held by another launcher

# rc -> (persisted status, human reason). host_conflict = another launcher/orphan
# owns the host; insufficient_space = a filesystem gate refused the start;
# precondition_failed = a pre-run precondition gate refused (exit 6 is raised by
# BOTH the image-digest-mismatch and the non-fresh-run_dir checks, so the status
# stays honest and the per-attempt child log carries the specific message).
FATAL_STATUS: dict[int, tuple[str, str]] = {
    LC_PRECONDITION: ("precondition_failed", "non-retryable precondition: image digest pin mismatch or run_dir not fresh -- see the run's per-attempt child log for the specific gate message"),
    LC_FREE_SPACE: ("insufficient_space", "free-space gate (runs-root or docker data-root too full / undeterminable)"),
    LC_ORPHAN_GATE: ("host_conflict", "orphan gate: a prior run is still active or has an unkillable orphan"),
    LC_SLOT_LOCKED: ("host_conflict", "run-slot lock held by another launcher"),
}
FATAL_CODES = frozenset(FATAL_STATUS)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    # Prints through sys.stdout, which main() tees into the campaign log file.
    print(f"[campaign] {utc_iso()} {msg}", flush=True)


class Tee:
    """Duplicate everything written to stdout/stderr into a log file too.

    stdout alone dies with the terminal (we learned this the hard way, hence
    requirement 7). tmux may also be lost; the file is the durable record.
    """

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def nearest_existing(path: Path) -> Path:
    """The path itself if it exists, else its nearest existing ancestor."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_gb(path: Path) -> Optional[float]:
    try:
        return shutil.disk_usage(str(nearest_existing(path))).free / (1024 ** 3)
    except OSError:
        return None


def container_running(name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.SubprocessError:
        return False
    return result.returncode == 0 and name in result.stdout.split()


def run_dir_looks_active(run_dir: Path, container_name: Optional[str]) -> bool:
    """A run_dir with an unfinished manifest and a live container is an ACTIVE
    run. During a strictly-serial campaign that means another launcher owns the
    host -- we must not archive over it.

    Single-container cells are active only if the recorded container is still
    running. Multi-container cells (dynamo_disagg) have no single container name
    to key on; if their manifest is unfinished, treat the run_dir as active or
    unknown and refuse to archive it. Renaming an active run_dir would move the
    directory out from under a live launch_cell before its run-slot lock can
    reject us."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("ended_at"):
        return False
    if container_name is None:
        return True
    return (
        manifest.get("container", {}).get("name") == container_name
        and container_running(container_name)
    )


def stale_running_recovery_decision(
    launcher_alive: bool,
    slot_free: bool,
    engine_container_running: bool,
    any_child_alive: bool,
) -> tuple[bool, str]:
    """Reboot/power-loss recovery decision (hardening item 3), pure so the whole
    decision table is unit-testable without docker/GPU/psutil.

    A hard crash or power loss mid-run leaves a run persisted as 'running' with an
    unfinished manifest; for a multi-container (dynamo_disagg) run,
    run_dir_looks_active would then refuse it as host_conflict and strand the
    campaign until a human intervenes. Recover such a run ONLY when EVERY liveness
    signal says it is truly gone:
      - the launcher process is dead (per the reaper ledger's launcher identity),
      - the run-slot lock is free,
      - no engine container of the run's lifecycle is running, and
      - no recorded run child is alive.
    ANY live signal -> refuse (the conservative branch): a genuinely-active run
    (or an ambiguous one) must never be archived out from under a live launcher.
    Returns (recover, reason)."""
    if launcher_alive:
        return False, "launcher process still alive (active run, not stale)"
    if not slot_free:
        return False, "run-slot lock held (another launcher owns the host)"
    if engine_container_running:
        return False, "an engine container of this lifecycle is still running"
    if any_child_alive:
        return False, "a recorded run child is still alive"
    return True, "stale_after_host_restart"


def read_cell_image_digests(schedule: "list[RunSpec]", repo_root: Path) -> dict:
    """{image_tag: pinned_digest} across every cell in the schedule, from each
    cell's engine.digest_pin_file. Best-effort per cell; a cell whose image /
    pin cannot be read is simply omitted (the drift check compares only tags
    present in BOTH baseline and current, so an omission is never a false drift)."""
    out: dict[str, Optional[str]] = {}
    for spec in schedule:
        try:
            cell = yaml.safe_load(Path(spec.cell_yaml).read_text())
            eng = (cell or {}).get("engine", {})
            repo, tag = eng.get("image_repo"), eng.get("image_tag")
            if not (repo and tag):
                continue
            image_tag = f"{repo}:{tag}"
            digest = None
            pin_rel = eng.get("digest_pin_file")
            if pin_rel:
                pin_path = Path(pin_rel)
                if not pin_path.is_absolute():
                    pin_path = repo_root / pin_rel
                digest = str(json.loads(pin_path.read_text()).get("digest") or "") or None
            out[image_tag] = digest
        except (OSError, yaml.YAMLError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue
    return out


def environment_drift(baseline: dict, current: dict) -> Optional[str]:
    """Return a human description of how the CURRENT environment differs from the
    campaign baseline (item 4a), or None if they agree. A field either side could
    not determine (None) is NOT compared -- only a genuine value change counts, so
    a box that briefly cannot read nvidia-smi does not fake a driver drift."""
    diffs: list[str] = []
    for key in ("kernel", "driver_version"):
        b, c = baseline.get(key), current.get(key)
        if b is not None and c is not None and str(b) != str(c):
            diffs.append(f"{key}: baseline={b!r} current={c!r}")
    b_imgs = baseline.get("image_digests") or {}
    c_imgs = current.get("image_digests") or {}
    for tag, bdig in b_imgs.items():
        cdig = c_imgs.get(tag)
        if bdig is not None and cdig is not None and str(bdig) != str(cdig):
            diffs.append(f"image {tag}: baseline={bdig} current={cdig}")
    return "; ".join(diffs) if diffs else None


def expected_container_name(cell_yaml: str, replica: int) -> Optional[str]:
    """The single container name a cell will use, or None for a multi-container
    lifecycle (dynamo_disagg) that declares no container_name_template."""
    cell = yaml.safe_load(Path(cell_yaml).read_text())
    template = cell.get("engine", {}).get("container_name_template")
    if not template:
        return None
    return str(template).replace("{replica}", f"{replica:02d}")


def archive_existing_run_dir(run_dir: Path, attempt: int) -> Optional[Path]:
    """Move a stale run_dir aside so launch_cell's assert_run_dir_fresh passes
    on the next attempt. Returns the archive path, or None if nothing to move."""
    if not run_dir.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = run_dir.with_name(f"{run_dir.name}_stale_attempt{attempt}_{stamp}")
    archive = base
    suffix = 1
    while archive.exists():
        suffix += 1
        archive = base.with_name(f"{base.name}_{suffix}")
    shutil.move(str(run_dir), str(archive))
    return archive


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    cell_id: str
    cell_yaml: str                 # absolute path
    replica: int
    duration_s: int                # from the cell yaml, for the estimate only
    calibration_file: Optional[str] = None      # absolute path or None
    calibration_required: bool = False
    allow_lower_bound_calibration: bool = False

    @property
    def run_key(self) -> str:
        return f"{self.cell_id}_r{self.replica:02d}"


@dataclass
class RunStatus:
    # pending | running | completed | failed | interrupted | host_conflict | insufficient_space
    status: str = "pending"
    attempts: int = 0
    last_started_at: Optional[str] = None
    last_ended_at: Optional[str] = None
    last_rc: Optional[int] = None
    log_path: Optional[str] = None
    # Short machine reason for the LAST status transition, for unattended
    # diagnostics (e.g. "stale_after_host_restart" set by the item-3 recovery).
    # Optional and defaulted so older state files (without it) still load.
    last_reason: Optional[str] = None


@dataclass
class State:
    campaign_id: str
    started_at: str = field(default_factory=utc_iso)
    runs: dict[str, RunStatus] = field(default_factory=dict)
    # Environment baseline captured at --start (item 4a): {kernel, driver_version,
    # image_digests}. Persisted so every subsequent dispatch (across resumes) can
    # fail campaign-fatal if the host drifted. None for pre-hardening state files.
    baseline: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "campaign_id": self.campaign_id,
            "started_at": self.started_at,
            "runs": {k: asdict(v) for k, v in self.runs.items()},
        }
        if self.baseline is not None:
            d["baseline"] = self.baseline
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            campaign_id=d["campaign_id"],
            started_at=d.get("started_at", utc_iso()),
            runs={k: RunStatus(**v) for k, v in d.get("runs", {}).items()},
            baseline=d.get("baseline"),
        )


class CampaignFatal(Exception):
    """A child reported a non-retryable fatal (launch_cell exit 6/7/8/9), or
    the campaign found an active run_dir it must not clobber. The whole campaign
    stops."""

    def __init__(self, run_key: str, rc: int, detail: str) -> None:
        super().__init__(detail)
        self.run_key = run_key
        self.rc = rc
        self.detail = detail


class CampaignInterrupted(Exception):
    """A signal asked the campaign to stop."""


class PreflightError(Exception):
    """Something the operator must fix before the campaign can start."""


# ---------------------------------------------------------------------------
# Campaign yaml -> validated config + ordered schedule
# ---------------------------------------------------------------------------


def load_campaign(campaign_path: Path) -> dict:
    """Load and validate the campaign yaml. Rejects any attempt to express
    parallelism: the extension campaign is strictly serial by design."""
    campaign = yaml.safe_load(campaign_path.read_text())
    if not isinstance(campaign, dict):
        raise PreflightError(f"{campaign_path}: not a mapping")

    mode = campaign.get("mode")
    if mode != "serial":
        raise PreflightError(
            f"{campaign_path}: mode must be 'serial' (got {mode!r}). This "
            "orchestrator only runs strictly-serial campaigns; the parallel "
            "slot model is retired."
        )
    if "slots" in campaign:
        raise PreflightError(
            f"{campaign_path}: 'slots:' is not allowed in a serial campaign. "
            "Parallel GPU slots are retired; use an ordered 'cells:' list."
        )
    for required in ("campaign_id", "cells", "runs_root"):
        if required not in campaign:
            raise PreflightError(f"{campaign_path}: missing required key {required!r}")
    return campaign


def build_schedule(campaign: dict, campaign_path: Path) -> list[RunSpec]:
    """Return the single global ordered queue of RunSpec.

    order == round_robin  : r1 of A, r1 of B, ..., r2 of A, r2 of B, ...
    order == cell_at_a_time: r1..rN of A, then r1..rN of B, ...
    """
    yaml_dir = campaign_path.parent
    replicas = int(campaign.get("replicas_per_cell", 1))
    order = campaign.get("order", "cell_at_a_time")
    if order not in ("round_robin", "cell_at_a_time"):
        raise PreflightError(f"unknown order: {order!r}")
    global_allow_lb = bool(campaign.get("allow_lower_bound_calibration", False))

    # Normalize each cell entry to a dict and load its yaml (for cell_id + duration).
    cells: list[dict] = []
    for raw in campaign["cells"]:
        entry = {"yaml": raw} if isinstance(raw, str) else dict(raw)
        cell_yaml = (yaml_dir / entry["yaml"]).resolve()
        if not cell_yaml.exists():
            raise PreflightError(f"cell yaml not found: {cell_yaml}")
        cell_doc = yaml.safe_load(cell_yaml.read_text())
        cell_id = entry.get("id") or cell_doc.get("cell_id")
        if not cell_id:
            raise PreflightError(f"{cell_yaml}: no cell_id and no 'id' in the campaign entry")
        duration_s = int(entry.get("duration_s_override") or cell_doc.get("duration_s") or 0)

        calib_file = entry.get("calibration_file")
        calib_abs = str((yaml_dir / calib_file).resolve()) if calib_file else None
        cells.append(
            {
                "cell_id": cell_id,
                "cell_yaml": str(cell_yaml),
                "duration_s": duration_s,
                "calibration_file": calib_abs,
                "calibration_required": bool(entry.get("calibration_required", False)),
                "allow_lower_bound_calibration": bool(
                    entry.get("allow_lower_bound_calibration", global_allow_lb)
                ),
            }
        )

    seen = [c["cell_id"] for c in cells]
    dupes = {cid for cid in seen if seen.count(cid) > 1}
    if dupes:
        raise PreflightError(f"duplicate cell ids in campaign: {sorted(dupes)}")

    def make(cell: dict, rep: int) -> RunSpec:
        return RunSpec(
            cell_id=cell["cell_id"],
            cell_yaml=cell["cell_yaml"],
            replica=rep,
            duration_s=cell["duration_s"],
            calibration_file=cell["calibration_file"],
            calibration_required=cell["calibration_required"],
            allow_lower_bound_calibration=cell["allow_lower_bound_calibration"],
        )

    schedule: list[RunSpec] = []
    if order == "round_robin":
        for rep in range(1, replicas + 1):
            for cell in cells:
                schedule.append(make(cell, rep))
    else:  # cell_at_a_time
        for cell in cells:
            for rep in range(1, replicas + 1):
                schedule.append(make(cell, rep))
    return schedule


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def validate_calibration(spec: RunSpec) -> tuple[bool, str]:
    """Return (ok, message) for a spec's calibration file. Uses launch_cell's
    own gate so this matches exactly what the run will accept."""
    if not spec.calibration_file:
        if spec.calibration_required:
            return False, "REQUIRED but no calibration_file given"
        return True, "none (not required)"
    path = Path(spec.calibration_file)
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        calib = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"INVALID JSON: {e}"
    try:
        rate = launch_cell.resolve_calibrated_rate(
            calib, spec.allow_lower_bound_calibration
        )
    except launch_cell.CalibrationError as e:
        return False, f"REJECTED: {e}"
    status = calib.get("status")
    return True, f"ok (status={status}, rate={rate})"


class Campaign:
    """The serial scheduler. Holds config, state, and the signal-forwarding
    handle to the current launch_cell child."""

    def __init__(
        self,
        campaign: dict,
        campaign_path: Path,
        schedule: list[RunSpec],
        state: State,
        state_path: Path,
    ) -> None:
        self.campaign = campaign
        self.campaign_path = campaign_path
        self.schedule = schedule
        self.state = state
        self.state_path = state_path

        yaml_dir = campaign_path.parent
        self.campaign_id = campaign["campaign_id"]
        self.runs_root = Path(campaign["runs_root"])
        paths = campaign.get("paths", {})
        self.hf_cache_host = Path(paths["hf_cache_host"])
        self.repo_root = (
            Path(paths["repo_root"]) if paths.get("repo_root")
            else yaml_dir.parent.parent
        )
        self.max_retries = int(campaign.get("retry_policy", {}).get("max_retries", 1))
        # Staleness gate for calibration files (requirement 1). Checked at
        # PRE-FLIGHT for every queued run AND again at dispatch time (a run at
        # queue position 30 starts weeks after pre-flight).
        self.calibration_max_age_days = float(
            campaign.get("calibration_max_age_days",
                         launch_cell.DEFAULT_CALIBRATION_MAX_AGE_DAYS))
        self.inter_run_cooldown_s = int(campaign.get("inter_run_cooldown_s", 0))
        self.est_run_overhead_s = int(campaign.get("est_run_overhead_s", 0))
        self.min_free_gb = float(campaign.get("min_free_gb", 20.0))
        # SC-2 mid-run disk watchdog floor, passed to launch_cell. Defaults below
        # the pre-run gate so a run that started near it is not killed instantly.
        self.min_free_gb_mid_run = float(campaign.get("min_free_gb_mid_run", 10.0))
        # SC-2 mid-run INODE floor (item 4c), passed to launch_cell.
        self.min_inodes_free_mid_run = int(
            campaign.get("min_inodes_free_mid_run",
                         launch_cell.DEFAULT_MIN_INODES_FREE_MID_RUN))

        # Signal state. current_proc is set only while a child is alive.
        self.current_proc: Optional[subprocess.Popen] = None
        self._interrupted = False
        self._interrupt_signal: Optional[int] = None

        # Injection seams for the unit tests (no docker/GPU/subprocess needed):
        self._popen = subprocess.Popen                 # child spawner
        self._sleep = time.sleep                       # cooldown sleeper
        self._skip_run_dir_prep = False                # bypass docker/fs guard

    # -- state persistence --------------------------------------------------

    def persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a tmp file then rename. On the same filesystem
        # rename(2) is atomic, so a crash mid-write never truncates the state.
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2))
        os.replace(tmp, self.state_path)

    def _status(self, spec: RunSpec) -> RunStatus:
        return self.state.runs.setdefault(spec.run_key, RunStatus())

    # -- operator notification hook (item 5) --------------------------------

    def _alert_path(self) -> Path:
        return self.state_path.parent / "ALERT.json"

    def _next_command(self, event: str) -> str:
        base = (f"python3 scripts/campaign.py --campaign-yaml {self.campaign_path} "
                f"--resume")
        # A failure needs an explicit opt-in to retry (failed is TERMINAL); a
        # fatal / interrupt just resumes once the precondition is resolved.
        if event in ("failed_after_retry", "completed_with_failures"):
            return base + " --rerun-failed"
        return base

    def write_alert(self, event: str, run_key: str, reason: str) -> None:
        """Operator notification hook (item 5). Atomically write state/ALERT.json
        (ts, run_key, event, reason, next command) and, if CAMPAIGN_NOTIFY_CMD is
        set in the environment, invoke it with the alert path as $1. The hook is
        BEST-EFFORT and timeout-bounded: a failing / missing / hung notifier is
        logged and swallowed, NEVER changing the campaign's outcome. Fired on
        campaign-fatal, failed-after-retry, interrupted, and
        completed_with_failures."""
        alert = {
            "ts": utc_iso(),
            "campaign_id": self.campaign_id,
            "event": event,
            "run_key": run_key,
            "reason": reason,
            "next_command": self._next_command(event),
        }
        path = self._alert_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(alert, indent=2))
            os.replace(tmp, path)
            log(f"ALERT written: {path} (event={event} run={run_key or '-'})")
        except OSError as e:
            log(f"WARNING: could not write ALERT.json: {e!r} (continuing)")
            return
        cmd = os.environ.get("CAMPAIGN_NOTIFY_CMD")
        if not cmd:
            return
        try:
            subprocess.run([cmd, str(path)], timeout=30, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"notify hook invoked: {cmd} {path}")
        except Exception as e:  # noqa: BLE001 - notify must never affect the outcome
            log(f"WARNING: CAMPAIGN_NOTIFY_CMD failed (ignored): {e!r}")

    def clear_alert(self) -> None:
        """Remove a stale ALERT.json: on a fresh --start, and on a clean
        completion (the incident, if any, is resolved). Best-effort."""
        try:
            self._alert_path().unlink()
        except OSError:
            pass

    # -- signal handling ----------------------------------------------------

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self.handle_signal)

    def handle_signal(self, signum, _frame) -> None:
        # Runs in the main thread between bytecodes. Forward SIGTERM to the
        # child so it tears down gracefully; mark the in-flight run interrupted
        # and persist so state survives even if we die right after.
        self._interrupted = True
        self._interrupt_signal = signum
        log(f"signal {signum} received: forwarding SIGTERM to launch_cell child and shutting down")
        proc = self.current_proc
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:  # pragma: no cover - defensive
                log(f"failed to signal child: {e}")
        for st in self.state.runs.values():
            if st.status == "running":
                st.status = "interrupted"
        try:
            self.persist_state()
        except Exception as e:  # pragma: no cover - defensive
            log(f"failed to persist state in signal handler: {e}")

    def _sleep_interruptible(self, seconds: float) -> None:
        """Cooldown that returns promptly if a signal arrives. Counted in 1s
        ticks (not wall-clock) so it checks the interrupt flag every second and
        stays unit-testable with an injected sleeper."""
        remaining = float(seconds)
        tick = 1.0
        while remaining > 0 and not self._interrupted:
            dt = min(tick, remaining)
            self._sleep(dt)
            remaining -= dt

    # -- launching one run --------------------------------------------------

    def _build_cmd(self, spec: RunSpec, attempt: int) -> list[str]:
        cmd = [
            sys.executable,
            str(self.repo_root / "scripts" / "launch_cell.py"),
            "--cell-yaml", spec.cell_yaml,
            "--replica", str(spec.replica),
            "--runs-root", str(self.runs_root),
            "--repo-root", str(self.repo_root),
            "--hf-cache-host", str(self.hf_cache_host),
            "--campaign-id", self.campaign_id,
            "--attempt", str(attempt),
            "--min-free-gb", str(self.min_free_gb),
            "--min-free-gb-mid-run", str(self.min_free_gb_mid_run),
            "--min-inodes-free-mid-run", str(self.min_inodes_free_mid_run),
        ]
        if spec.calibration_file:
            cmd += ["--calibration-file", spec.calibration_file,
                    "--calibration-max-age-days", str(self.calibration_max_age_days)]
        if spec.allow_lower_bound_calibration:
            cmd += ["--allow-lower-bound-calibration"]
        return cmd

    def _prepare_run_dir(self, spec: RunSpec, attempt: int) -> Path:
        """Archive any stale run_dir so launch_cell starts fresh. Refuse (fatal)
        if the run_dir belongs to an ACTIVE single-container run owned by another
        launcher. Multi-container cells (dynamo_disagg) declare no single
        container name -- expected_container_name returns None, so an unfinished
        manifest is treated as active/unknown and never archived."""
        run_dir = self.runs_root / f"{self.campaign_id}_{spec.cell_id}_r{spec.replica:02d}"
        if self._skip_run_dir_prep:
            return run_dir
        container_name = expected_container_name(spec.cell_yaml, spec.replica)
        if run_dir_looks_active(run_dir, container_name):
            raise CampaignFatal(
                spec.run_key, LC_ORPHAN_GATE,
                f"run_dir {run_dir} and container {container_name} look ACTIVE; "
                "another launcher owns the host. Stopping the campaign.",
            )
        archived = archive_existing_run_dir(run_dir, attempt)
        if archived is not None:
            log(f"archived pre-existing run_dir for {spec.run_key}: {archived}")
        return run_dir

    def _stream_child_output(self, proc, capture) -> None:
        """Fan the child's merged stdout/stderr, line by line in real time, to
        BOTH the terminal+campaign log (via sys.stdout, which main() tees into
        the campaign log) and the per-attempt capture file.

        Without this the child's lines -- the primary evidence when a run fails
        -- would land only in a file the operator has to go hunting for, and
        never reach the live terminal or the durable campaign log.

        Blocks until the child closes its pipe (EOF), which is what we want on a
        strictly serial campaign. A SIGTERM forwarded by handle_signal() makes
        launch_cell tear down and close the pipe, ending this loop; wait() then
        returns. readline() is auto-retried across the signal (PEP 475), so the
        handler runs without corrupting the stream."""
        out = getattr(proc, "stdout", None)
        if out is None:
            return
        for line in iter(out.readline, ""):
            capture.write(line)
            capture.flush()
            # launch_cell already prefixes its lines with "[launch_cell] ...",
            # so the source stays attributable interleaved with campaign lines.
            sys.stdout.write(line)
            sys.stdout.flush()

    def _launch_cell_rc(self, spec: RunSpec, attempt: int) -> int:
        """Spawn launch_cell for one run and wait for it to fully exit. Returns
        the child exit code. All docker/subprocess/fs side effects live here so
        the retry/resume/signal logic above stays unit-testable."""
        run_dir = self._prepare_run_dir(spec, attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Per-attempt capture file lives beside the campaign log (state/logs),
        # NOT inside run_dir: launch_cell's assert_run_dir_fresh only tolerates
        # a pre-created launch_cell.log, so a per-attempt name or a logs/ subdir
        # there would trip its freshness guard. state/logs also survives the
        # run_dir archival that _prepare_run_dir does on each attempt.
        log_dir = self.state_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt_log = log_dir / f"{run_dir.name}_attempt{attempt}.log"
        self._status(spec).log_path = str(attempt_log)

        cmd = self._build_cmd(spec, attempt)
        log(f"starting {spec.run_key} attempt={attempt}")
        log(f"cmd: {' '.join(cmd)}")
        log(f"child output -> {attempt_log}")

        with open(attempt_log, "a", buffering=1) as capture:
            self.current_proc = self._popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                self._stream_child_output(self.current_proc, capture)
                rc = self.current_proc.wait()
            finally:
                self.current_proc = None
        if rc != 0:
            log(f"{spec.run_key} attempt={attempt} rc={rc}; child log: {attempt_log}")
        return rc

    def _dispatch(self, spec: RunSpec) -> int:
        """One launch_cell invocation: bump the cumulative attempt counter,
        mark running, persist, run, record the result, persist."""
        status = self._status(spec)
        status.attempts += 1
        attempt = status.attempts
        status.status = "running"
        status.last_started_at = utc_iso()
        status.last_ended_at = None
        status.last_rc = None
        self.persist_state()

        rc = self._launch_cell_rc(spec, attempt)

        status.last_ended_at = utc_iso()
        status.last_rc = rc
        self.persist_state()
        return rc

    def _mark_fatal(self, spec: RunSpec, rc: int) -> None:
        """Record the fatal status for a spec before the campaign stops, so the
        state file explains WHY on resume/diagnostics (fixes the earlier
        'stuck in running' after a pre-child fatal)."""
        status, _ = FATAL_STATUS.get(rc, ("host_conflict", "host precondition"))
        self._status(spec).status = status
        self.persist_state()

    def _run_with_retry(self, spec: RunSpec) -> str:
        """Run one spec under the retry policy. Returns 'completed' or 'failed'.
        Raises CampaignFatal on a non-retryable exit (6/7/8/9) or a pre-child
        active-run detection; CampaignInterrupted on signal."""
        # Requirement 1: re-check calibration staleness at DISPATCH, not just at
        # pre-flight. A run at queue position 30 starts weeks after pre-flight;
        # its calibration may have crossed the max-age line since. A stale/
        # mismatched calibration is a non-retryable precondition (retrying cannot
        # un-age it): stop the campaign loudly like the other precondition gates.
        prov_ok, prov_msg = self.check_calibration_provenance(spec, time.time())
        if not prov_ok:
            self._mark_fatal(spec, LC_PRECONDITION)
            raise CampaignFatal(
                spec.run_key, LC_PRECONDITION,
                f"calibration staleness gate at dispatch: {prov_msg}",
            )
        # Item 4a: the host environment must not drift mid-campaign. A driver /
        # kernel / pinned-image change since the --start baseline is a
        # campaign-fatal precondition -- an operator decision, never background
        # noise silently spanning runs. Checked at EVERY dispatch (a run at queue
        # position 30 starts long after the baseline was taken).
        drift = self.check_environment_drift()
        if drift is not None:
            self._mark_fatal(spec, LC_PRECONDITION)
            raise CampaignFatal(
                spec.run_key, LC_PRECONDITION,
                f"environment drift since the campaign baseline: {drift}. A "
                "mid-campaign driver/kernel/image change must be an explicit "
                "operator decision; stopping. Re-establish the baseline "
                "deliberately (--start a fresh campaign) once the change is intended.",
            )
        while True:
            if self._interrupted:
                raise CampaignInterrupted()

            try:
                rc = self._dispatch(spec)
            except CampaignFatal as f:
                # _prepare_run_dir refused BEFORE the child ran (run_dir looks
                # active). _dispatch left status 'running'; correct it so the
                # persisted state is not misleading, then propagate.
                self._mark_fatal(spec, f.rc)
                raise

            # A signal during the run already tore the child down; wait() has
            # returned. Do not treat as a run failure or retry.
            if self._interrupted:
                status = self._status(spec)
                if status.status != "interrupted":
                    status.status = "interrupted"
                    self.persist_state()
                raise CampaignInterrupted()

            if rc == 0:
                self._status(spec).status = "completed"
                self.persist_state()
                log(f"{spec.run_key} COMPLETED")
                return "completed"

            if rc in FATAL_CODES:
                # NOT a run failure: retrying cannot fix host ownership or a full
                # filesystem. Record why and stop the campaign loudly.
                _, human = FATAL_STATUS[rc]
                self._mark_fatal(spec, rc)
                raise CampaignFatal(
                    spec.run_key, rc,
                    f"launch_cell exit {rc} ({human}).",
                )

            # Ordinary failure. The retry decision counts PERSISTED attempts
            # (status.attempts, bumped in _dispatch and surviving every --resume),
            # NOT a session-local counter: a run gets at most max_retries+1 total
            # launches across ANY number of resumes, so an interrupt-then-resume
            # cycle can never silently hand it a fresh attempt budget (item 2).
            status = self._status(spec)
            if status.attempts >= self.max_retries + 1:
                status.status = "failed"
                status.last_reason = f"failed_after_retry rc={rc}"
                self.persist_state()
                log(f"{spec.run_key} FAILED after {status.attempts} attempt(s) rc={rc}")
                self.write_alert("failed_after_retry", spec.run_key,
                                 f"exhausted retry budget ({status.attempts} attempt(s)), last rc={rc}")
                return "failed"
            log(f"{spec.run_key} failed rc={rc}, retrying "
                f"(persisted attempt {status.attempts}/{self.max_retries + 1})")

    # -- pre-flight ---------------------------------------------------------

    def _current_calibration_signature(self, spec: RunSpec) -> dict:
        """The host/image signature of THIS box for `spec`, to compare against
        the calibration's recorded provenance. hostname is always known; the
        image tag+digest come from the cell yaml + its pin file (the same source
        launch_cell uses); GPU name+driver are best-effort (None if no
        nvidia-smi). Any read failure leaves image fields None -> not compared."""
        image_tag = image_digest = None
        try:
            cell = yaml.safe_load(Path(spec.cell_yaml).read_text())
            eng = (cell or {}).get("engine", {})
            repo, tag = eng.get("image_repo"), eng.get("image_tag")
            if repo and tag:
                image_tag = f"{repo}:{tag}"
            pin_rel = eng.get("digest_pin_file")
            if pin_rel:
                pin_path = Path(pin_rel)
                if not pin_path.is_absolute():
                    pin_path = self.repo_root / pin_rel
                pin = json.loads(pin_path.read_text())
                image_digest = (str(pin.get("digest") or "") or None)
        except (OSError, yaml.YAMLError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass
        gpu_name, driver = launch_cell.gpu_name_and_driver()
        return launch_cell.current_calibration_signature(
            socket.gethostname(), gpu_name, driver, image_tag, image_digest)

    def check_calibration_provenance(self, spec: RunSpec, now_unix: float) -> tuple[bool, str]:
        """Return (ok, message) for a spec's calibration staleness / host / image
        signature. Only meaningful when the spec has a calibration_file that
        parses; a spec with no file is (True, 'no calibration'). Uses launch_cell's
        gate so pre-flight, dispatch, and run time agree exactly."""
        if not spec.calibration_file:
            return True, "no calibration"
        path = Path(spec.calibration_file)
        if not path.exists():
            # validate_calibration already reports the missing-file verdict.
            return True, "file absent (reported by calibration check)"
        try:
            calib = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            return False, f"INVALID JSON: {e}"
        current_sig = self._current_calibration_signature(spec)
        try:
            launch_cell.check_calibration_provenance(
                calib, current_sig, self.calibration_max_age_days, now_unix)
        except launch_cell.CalibrationError as e:
            return False, f"STALE/MISMATCH: {e}"
        age = launch_cell.calibration_age_days(calib, now_unix)
        return True, (f"fresh (age={age:.1f}d)" if age is not None else "fresh")

    def pending_specs(self) -> list[RunSpec]:
        """Specs that still need to run: everything whose persisted status is not
        TERMINAL. failed is TERMINAL (hardening item 2) -- it is NOT re-queued on
        --resume, so a run can never silently burn fresh attempts across resumes.
        The operator opts a failed run back in with --rerun-failed, which resets
        its attempt budget explicitly (reset_failed_for_rerun)."""
        out = []
        for spec in self.schedule:
            st = self.state.runs.get(spec.run_key)
            if st and st.status in TERMINAL_STATUSES:
                continue
            out.append(spec)
        return out

    def reset_failed_for_rerun(self, persist: bool = True) -> int:
        """--rerun-failed: turn each FAILED run back into a fresh pending run
        (attempts -> 0, status -> pending) so the queue re-runs it with a full
        attempt budget. failed is otherwise TERMINAL. Returns the count reset and
        logs each one (the reset is an explicit, audited operator decision)."""
        reset = 0
        for key, st in self.state.runs.items():
            if st.status == "failed":
                st.status = "pending"
                st.attempts = 0
                st.last_rc = None
                reset += 1
                log(f"--rerun-failed: reset {key} (was failed) -> pending, attempts=0")
        if not reset:
            log("--rerun-failed: no failed runs to reset")
        elif persist:
            self.persist_state()
        return reset

    # -- environment baseline + drift gate (item 4a) ------------------------

    def capture_environment_baseline(self) -> dict:
        """Snapshot {kernel, driver_version, image_digests} for THIS host now.
        driver comes from nvidia-smi (best-effort None if absent); kernel from
        platform.release(); image digests from every cell's pin file."""
        _, driver = launch_cell.gpu_name_and_driver()
        return {
            "captured_at": utc_iso(),
            "kernel": platform.release(),
            "driver_version": driver,
            "image_digests": read_cell_image_digests(self.schedule, self.repo_root),
        }

    def capture_and_store_baseline(self, persist: bool = True) -> None:
        """Establish the campaign baseline (item 4a): captured once at --start (or
        the first time a pre-hardening state file is resumed) and persisted so
        every subsequent dispatch can gate against it across resumes."""
        self.state.baseline = self.capture_environment_baseline()
        b = self.state.baseline
        log(f"environment baseline: kernel={b['kernel']!r} driver={b['driver_version']!r} "
            f"images={len(b['image_digests'])} pinned")
        if persist:
            self.persist_state()

    def check_environment_drift(self) -> Optional[str]:
        """Compare the current host environment to the campaign baseline. Returns
        a drift description or None. No baseline (pre-hardening state) -> no check
        (and no nvidia-smi probe)."""
        baseline = self.state.baseline
        if not baseline:
            return None
        return environment_drift(baseline, self.capture_environment_baseline())

    # -- reboot / power-loss recovery (item 3) ------------------------------

    def _run_id(self, spec: RunSpec) -> str:
        """The full run_id launch_cell / reaper key on (campaign-scoped), which is
        distinct from spec.run_key (cell-scoped)."""
        return f"{self.campaign_id}_{spec.cell_id}_r{spec.replica:02d}"

    def _run_slot_free(self) -> bool:
        """True iff the run-slot lock is currently free. Probes by acquiring and
        immediately releasing (closing the fd releases the flock)."""
        slot = reaper.acquire_run_slot(self.runs_root)
        if slot is None:
            return False
        try:
            slot.close()
        except OSError:
            pass
        return True

    def _engine_container_running_for(self, spec: RunSpec) -> bool:
        """True iff an engine container for spec's lifecycle is currently running.
        single_container: the expected container name; multi-container
        (dynamo_disagg, no single name): any dyn_* container that is up."""
        name = expected_container_name(spec.cell_yaml, spec.replica)
        if name is not None:
            return container_running(name)
        return any(container_running(n) for n in launch_cell.all_dyn_containers())

    def _stale_running_facts(self, spec: RunSpec, slot_free: bool) -> dict:
        """Gather the four liveness signals for stale_running_recovery_decision."""
        run_id = self._run_id(spec)
        return {
            "launcher_alive": reaper.launcher_alive_for(self.runs_root, run_id),
            "slot_free": slot_free,
            "engine_container_running": self._engine_container_running_for(spec),
            "any_child_alive": reaper.recorded_children_alive(self.runs_root, run_id),
        }

    def recover_stale_running(self) -> None:
        """For each run the state still marks 'running' at resume (only possible
        after a hard crash / power loss -- a signal shutdown rewrites running ->
        interrupted), decide via stale_running_recovery_decision whether it is a
        stranded stale run (safe to recover) or a genuinely-active one (refuse).

        On recover: archive the stale run_dir, mark the run interrupted with reason
        'stale_after_host_restart', and let the normal queue re-run it under its
        PERSISTED attempt budget. On refuse: leave it 'running' so _prepare_run_dir
        still treats it as an active run (host_conflict) -- a human must resolve a
        run that looks alive."""
        running = [
            s for s in self.schedule
            if (st := self.state.runs.get(s.run_key)) is not None and st.status == "running"
        ]
        if not running:
            return
        slot_free = self._run_slot_free()
        changed = False
        for spec in running:
            facts = self._stale_running_facts(spec, slot_free)
            recover, reason = stale_running_recovery_decision(**facts)
            if not recover:
                log(f"stale-running check for {spec.run_key}: NOT recovering ({reason}); "
                    f"leaving 'running' (a human must resolve if it is truly stuck)")
                continue
            run_dir = self.runs_root / self._run_id(spec)
            archived = archive_existing_run_dir(run_dir, self._status(spec).attempts)
            if archived is not None:
                log(f"stale-running recovery: archived {run_dir} -> {archived}")
            st = self._status(spec)
            st.status = "interrupted"
            st.last_reason = "stale_after_host_restart"
            st.last_ended_at = utc_iso()
            changed = True
            log(f"stale-running recovery: {spec.run_key} -> interrupted "
                f"(reason=stale_after_host_restart); re-queued under attempt budget "
                f"(attempts={st.attempts}/{self.max_retries + 1})")
        if changed:
            self.persist_state()

    def preflight(self) -> None:
        """Print the schedule, calibration status, estimate, and free space.
        Raise PreflightError if a REQUIRED calibration is missing/invalid or
        free space is below the gate -- BEFORE any run starts (requirements 6, 8)."""
        log(f"campaign_id: {self.campaign_id}")
        log(f"repo_root:   {self.repo_root}")
        log(f"runs_root:   {self.runs_root}")
        log(f"state_file:  {self.state_path}")

        # Item 5: surface a prior incident's ALERT.json so the operator sees why
        # the last run stopped BEFORE resuming (it is cleared on a clean finish).
        alert_path = self._alert_path()
        if alert_path.exists():
            try:
                a = json.loads(alert_path.read_text())
                log(f"** PRIOR ALERT ({alert_path}): event={a.get('event')} "
                    f"run={a.get('run_key')} ts={a.get('ts')}")
                log(f"**   reason: {a.get('reason')}")
                log(f"**   next:   {a.get('next_command')}")
            except (OSError, json.JSONDecodeError) as e:
                log(f"** PRIOR ALERT present at {alert_path} but unreadable: {e}")
        log(
            f"retry: max_retries={self.max_retries}  "
            f"cooldown={self.inter_run_cooldown_s}s  min_free={self.min_free_gb}GB"
        )

        pending = self.pending_specs()
        completed = len(self.schedule) - len(pending)
        log(f"schedule: {len(self.schedule)} runs total, {completed} already completed, {len(pending)} to run")

        # Full schedule in order, with per-cell calibration status.
        calib_failures: list[str] = []
        now_unix = time.time()
        total_est_s = 0
        for i, spec in enumerate(self.schedule, 1):
            st = self.state.runs.get(spec.run_key)
            state_note = f"  [{st.status}]" if st else ""
            # A present-but-unacceptable calibration_file ALWAYS fails pre-flight:
            # _build_cmd passes it to launch_cell regardless of calibration_required,
            # and launch_cell would reject it at run time (burning the run).
            # calibration_required only governs the file-absent case. So any
            # ok == False (required-and-absent, or present-and-unacceptable) is fatal.
            ok, msg = validate_calibration(spec)
            if not ok:
                calib_failures.append(f"{spec.run_key}: {msg}")
            # Requirement 1: staleness / host / image gate on EVERY queued run,
            # so a month-1 ceiling cannot silently drive a month-3 run. Skipped
            # for a completed run (its rate is already burned into the manifest).
            will_run = not (st and st.status == "completed")
            prov_msg = ""
            if will_run:
                prov_ok, prov_msg = self.check_calibration_provenance(spec, now_unix)
                if not prov_ok:
                    calib_failures.append(f"{spec.run_key}: {prov_msg}")
                total_est_s += spec.duration_s + self.est_run_overhead_s
            log(f"  {i:3d}. {spec.run_key}{state_note}  calib={msg}"
                + (f"  prov={prov_msg}" if prov_msg else ""))

        log(f"estimated remaining wallclock: {format_duration(total_est_s)} "
            f"(sum of duration_s + {self.est_run_overhead_s}s overhead per pending run)")

        fg = free_gb(self.runs_root)
        if fg is None:
            log(f"free space on {self.runs_root}: UNKNOWN (could not stat)")
        else:
            log(f"free space on {nearest_existing(self.runs_root)}: {fg:.1f} GB")

        # Hard gates -- fail before the first run.
        if calib_failures:
            raise PreflightError(
                "required calibration file(s) missing/invalid:\n  "
                + "\n  ".join(calib_failures)
            )
        if fg is not None and fg < self.min_free_gb:
            raise PreflightError(
                f"free space {fg:.1f} GB on {self.runs_root} is below the "
                f"min_free_gb gate ({self.min_free_gb} GB)"
            )

    # -- main loop ----------------------------------------------------------

    def run(self) -> int:
        """Drive the queue serially. Returns the process exit code."""
        # Item 3: before dispatching, recover any run stranded as 'running' by a
        # host restart (archive + mark interrupted) so a crashed multi-container
        # run does not deadlock the queue as a phantom host_conflict.
        self.recover_stale_running()
        pending = self.pending_specs()
        for idx, spec in enumerate(pending):
            if self._interrupted:
                log("interrupted before starting the next run")
                self.persist_state()
                self.write_alert("interrupted", spec.run_key,
                                 "campaign received a stop signal before this run started")
                return EXIT_INTERRUPTED
            try:
                self._run_with_retry(spec)
            except CampaignInterrupted:
                log("campaign interrupted; state persisted")
                self.write_alert("interrupted", spec.run_key,
                                 "campaign received a stop signal during this run")
                return EXIT_INTERRUPTED
            except CampaignFatal as f:
                log(f"CAMPAIGN FATAL on {f.run_key}: {f.detail}")
                log("stopping the campaign. Resolve the precondition (host ownership, "
                    "free space, image-pin mismatch, or a non-fresh run_dir -- see the "
                    "run's per-attempt child log), then --resume.")
                self.write_alert("campaign_fatal", f.run_key, f.detail)
                return EXIT_CAMPAIGN_FATAL

            # Inter-run cooldown before the next run (skip after the last one).
            is_last = idx == len(pending) - 1
            if not is_last and self.inter_run_cooldown_s > 0:
                log(f"inter-run cooldown: {self.inter_run_cooldown_s}s")
                self._sleep_interruptible(self.inter_run_cooldown_s)
                if self._interrupted:
                    log("interrupted during cooldown; state persisted")
                    self.persist_state()
                    self.write_alert("interrupted", spec.run_key,
                                     "campaign received a stop signal during the inter-run cooldown")
                    return EXIT_INTERRUPTED

        log("campaign complete")
        failed = [k for k, s in self.state.runs.items() if s.status == "failed"]
        if failed:
            # Exit 0 STRICTLY means every scheduled run completed. A drained queue
            # that still carries a failed run exits EXIT_COMPLETED_WITH_FAILURES so
            # an unattended campaign cannot look clean when it is not (item 2).
            log(f"NOTE: {len(failed)} run(s) ended FAILED: {sorted(failed)}. "
                f"Inspect them; re-queue with --resume --rerun-failed.")
            self.write_alert("completed_with_failures", ",".join(sorted(failed)),
                             f"{len(failed)} run(s) ended FAILED: {sorted(failed)}")
            return EXIT_COMPLETED_WITH_FAILURES
        # Clean completion: clear any stale ALERT.json from an earlier incident.
        self.clear_alert()
        return EXIT_OK


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def setup_logfile(state_path: Path) -> Path:
    """Tee stdout/stderr into a timestamped log file under the state dir."""
    log_dir = state_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"campaign_{stamp}.log"
    log_f = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    return log_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Serial orchestrator for the extension (DoW) campaign.")
    p.add_argument("--campaign-yaml", type=Path, required=True)
    p.add_argument("--start", action="store_true", help="Start fresh (delete existing state file).")
    p.add_argument("--resume", action="store_true", help="Resume from the state file.")
    p.add_argument("--dry-run", action="store_true", help="Run pre-flight (schedule + gates) and exit.")
    p.add_argument("--rerun-failed", action="store_true",
                   help="Reset FAILED runs (attempts->0, status->pending) so --resume "
                        "re-queues them with a full attempt budget. failed is otherwise "
                        "TERMINAL and skipped on --resume. Each reset is logged.")
    args = p.parse_args(argv)

    if args.start and args.resume:
        print("--start and --resume are mutually exclusive", file=sys.stderr)
        return EXIT_USAGE
    if args.rerun_failed and not args.resume:
        print("--rerun-failed only applies to --resume (failed runs live in an "
              "existing state file)", file=sys.stderr)
        return EXIT_USAGE
    if not (args.start or args.resume or args.dry_run):
        print("must specify --start, --resume, or --dry-run", file=sys.stderr)
        return EXIT_USAGE

    campaign_path = args.campaign_yaml.resolve()
    try:
        campaign = load_campaign(campaign_path)
        schedule = build_schedule(campaign, campaign_path)
    except PreflightError as e:
        print(f"[campaign] PREFLIGHT ERROR: {e}", file=sys.stderr)
        return EXIT_PREFLIGHT

    state_path = (campaign_path.parent / campaign.get("state_file", "state/campaign_state.json")).resolve()

    # --start wipes prior state; --resume loads it; --dry-run neither writes nor deletes.
    if args.start and state_path.exists() and not args.dry_run:
        log(f"--start: deleting existing state at {state_path}")
        state_path.unlink()
    if args.resume and not state_path.exists():
        log(f"--resume: no state file at {state_path}, starting fresh")

    if state_path.exists() and not args.start:
        state = State.from_dict(json.loads(state_path.read_text()))
        # Refuse to resume a state file that belongs to a DIFFERENT campaign:
        # the state_file path can be shared/misconfigured, and silently mixing
        # run provenance across campaigns corrupts the checkpoint. --start is
        # the explicit escape hatch (it wipes the state above).
        if state.campaign_id != campaign["campaign_id"]:
            print(
                f"[campaign] PREFLIGHT ERROR: state at {state_path} belongs to "
                f"campaign {state.campaign_id!r}, not {campaign['campaign_id']!r}. "
                "Point --campaign-yaml at the matching campaign, use a distinct "
                "state_file, or pass --start to wipe and begin fresh.",
                file=sys.stderr,
            )
            return EXIT_PREFLIGHT
    else:
        state = State(campaign_id=campaign["campaign_id"])

    # Tee output to a durable log file (skip for --dry-run to avoid side effects).
    if not args.dry_run:
        log_path = setup_logfile(state_path)
        log(f"campaign log: {log_path}")

    camp = Campaign(campaign, campaign_path, schedule, state, state_path)

    # --rerun-failed: explicitly re-open TERMINAL failed runs before pre-flight so
    # the schedule/estimate reflect them. Persist only for a real resume (not
    # --dry-run, which must never write state).
    if args.rerun_failed:
        camp.reset_failed_for_rerun(persist=not args.dry_run)

    # Item 4a: establish the environment baseline at --start (and, for a
    # pre-hardening state file with none yet, the first real resume). --dry-run
    # never writes state, so it only previews an in-memory baseline.
    if not args.dry_run and camp.state.baseline is None:
        camp.capture_and_store_baseline()

    # Item 5: a fresh --start clears any stale ALERT.json from a prior campaign
    # (a resume keeps it so pre-flight can surface the last incident).
    if args.start and not args.dry_run:
        camp.clear_alert()

    try:
        camp.preflight()
    except PreflightError as e:
        print(f"[campaign] PREFLIGHT ERROR: {e}", file=sys.stderr)
        return EXIT_PREFLIGHT

    if args.dry_run:
        log("--dry-run: pre-flight OK, exiting without running")
        return EXIT_OK

    camp.install_signal_handlers()
    return camp.run()


if __name__ == "__main__":
    sys.exit(main())
