#!/usr/bin/env python3
"""Week-1 calibration orchestrator for the DoW screening campaign.

Turns the 57 MISSING calibrations that dow_campaign.yaml's pre-flight reports
into 57 valid per-cell calibration JSONs, one bring-up per serving system.

Why this exists (and why it is Python, not bash): the DoW cells all bind their
calibration to a specific cell_id + calibration_fraction at dispatch
(launch_cell.check_calibration_binding), and the five DoW factors are all
workload-side, so ONE engine instance per system serves every cell of that
system. This orchestrator reuses launch_cell's PRODUCTION lifecycle to bring the
engine up once per system, then runs the existing calibrate_rate.py sweep once
per cell against that endpoint. There is deliberately NO second bring-up path:
the frontend-identity bug taught us what divergent bring-up paths cost, so we
drive the SAME make_lifecycle / bring_up / teardown launch_cell drives (with
monitors and the run client simply not started), and hold the reaper run-slot
lock for the whole orchestration so a real run cannot start underneath it.

Per-cell, NOT deduped: the 3 center points of a system share one workload shape,
so there are only 51 distinct shapes -- but the binding checks each file's own
cell_id, and every cell names its own calibration_file, so we run one honest
sweep per cell (57) rather than stamping a shared sweep with a foreign cell_id.
The 6 extra center-point sweeps are cheap (calibrate_rate early-stops past the
knee) and keep each file a real independent calibration of its own cell.

Failure policy: a sweep that ends status != ok (no_stable_point / did_not_
saturate) does NOT halt the orchestrator -- unlike a 36h run, a failed
calibration corrupts nothing, and halting 50 good sweeps for one bad shape
wastes the week-1 budget. Such a sweep is recorded, the run continues, and the
process exits non-zero at the end with a summary table (cell, status, suggested
wider/narrower rate grid). A HARD stop happens only for host-level preconditions
(run-slot lock held, image pin / docker / env, or engine bring-up failing after
its retries).

Resumability: a cell whose calibration JSON already exists, is status=ok, and
passes the SAME binding + provenance + max-age gates the campaign applies is
skipped (the check reuses launch_cell's own gate functions, not a reimplement);
if every cell of a system is already valid, that system's bring-up is skipped
entirely. --recalibrate <cell_id|all> forces a re-sweep. The whole thing is
safe to Ctrl-C and re-run.

Verdict: the orchestrator ends by running the campaign's own pre-flight
(campaign.py --dry-run) and printing its head -- success is that dry-run moving
from 57 MISSING to 0.

Usage (run inside tmux; it tees to campaigns/extension/state/logs/):
  tmux new -d -s calib_dow \\
    'python3 scripts/calibrate_dow.py --campaign-yaml campaigns/extension/dow_campaign.yaml'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Reuse the PRODUCTION internals. scripts/ is on sys.path[0] when this file is
# run directly, so these resolve the same way campaign.py's imports do. This is
# the whole point: no second bring-up / provenance / gate path.
import launch_cell
import reaper
import campaign as camp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMPAIGN_YAML = REPO_ROOT / "campaigns" / "extension" / "dow_campaign.yaml"

# Bring engines up (and calibrate) in this order: Dynamo first (the paper plan's
# STEP-1 ordering and the most expensive stack to churn), then the two single-
# container systems. Cells within a system keep the campaign's list order.
SYSTEM_ORDER = ["dynamo_disagg", "triton", "vllm"]

# Per-SYSTEM default ascending offered-rate sweep grids (rps). Each starts low
# enough to anchor a sub-1-rps ceiling (long-prompt / long-output shapes can cap
# below 1 rps) and reaches into the multi-rps range for the light shapes;
# calibrate_rate.py early-stops once it is clearly past the knee, so a wide grid
# costs little. Override per system or per cell with --rate-grids <yaml>, or
# globally with the env var WOSAR_CALIB_RATES (a comma list) -- the LONGTEST-
# style knob. A did_not_saturate / no_stable_point result suggests the next grid
# in the end-of-run summary.
DEFAULT_RATE_GRIDS = {
    # Disaggregated Dynamo pays a network hop per token; it saturates lower than
    # the in-process engines, so the grid tops out sooner.
    "dynamo_disagg": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "triton":        [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
    "vllm":          [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
}
_FALLBACK_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Process exit codes. Non-zero from a non-ok sweep or a still-MISSING dry-run is
# EXIT_SOME_FAILED; the host-precondition codes mirror launch_cell's where they
# overlap (9 = run-slot lock held).
EXIT_OK = 0
EXIT_SOME_FAILED = 1
EXIT_BRINGUP = 2
# Host-precondition codes mirror launch_cell's where they overlap: 7 = free-space
# gate, 8 = residual active run / unkillable orphan, 9 = run-slot lock held.
EXIT_DISK = 7
EXIT_PRECONDITION = 8
EXIT_LOCK_HELD = 9
EXIT_INTERRUPTED = 130


def system_of(cell_id: str) -> str:
    """The serving system a DoW cell_id belongs to.

    dow_dynamo_disagg_cp1 -> dynamo_disagg, dow_triton_p16 -> triton,
    dow_vllm_cp3 -> vllm. Strips the dow_ prefix and the trailing design-point
    token (p<NN> or cp<N>)."""
    base = re.sub(r"^dow_", "", cell_id)
    base = re.sub(r"_c?p\d+$", "", base)
    return base


def group_by_system(specs: list) -> list[tuple[str, list]]:
    """Group RunSpecs by system in SYSTEM_ORDER (unknown systems last, in
    first-seen order), preserving the campaign list order within each system."""
    groups: dict[str, list] = {}
    first_seen: list[str] = []
    for s in specs:
        sysname = system_of(s.cell_id)
        if sysname not in groups:
            groups[sysname] = []
            first_seen.append(sysname)
        groups[sysname].append(s)
    ordered = [sy for sy in SYSTEM_ORDER if sy in groups]
    ordered += [sy for sy in first_seen if sy not in SYSTEM_ORDER]
    return [(sy, groups[sy]) for sy in ordered]


def suggest_next_grid(status: str, grid: list[float]) -> Optional[list[float]]:
    """The grid to try next for a non-ok sweep.

    did_not_saturate: every swept rate stayed stable, so the ceiling is only a
    LOWER bound -- extend the grid upward. no_stable_point: even the lowest rate
    was unstable -- the ceiling is below the grid, so drop an order of magnitude.
    """
    g = sorted(float(x) for x in grid)
    if not g:
        return None
    if status == "did_not_saturate":
        top = g[-1]
        return g + [round(top * 2, 4), round(top * 3, 4)]
    if status == "no_stable_point":
        low = g[0]
        return [round(low / 8, 4), round(low / 4, 4), round(low / 2, 4)]
    return None


def calibration_is_valid(calib_path, cell_id: str, fraction: Optional[float],
                         current_sig: Optional[dict], max_age_days: Optional[float],
                         now: float) -> tuple[bool, str]:
    """Is an existing calibration JSON good enough to SKIP (re-)calibrating this
    cell? Reuses launch_cell's own gate functions so "valid" means exactly what
    the campaign pre-flight (binding + usable rate) AND the run dispatch
    (provenance + max-age) will accept -- no reimplementation.

    current_sig None means the host/image signature could not be built (e.g. the
    image pin is unreadable); the provenance leg is then skipped and the caller
    proceeds to calibrate, where the real pin gate will surface any problem."""
    p = Path(calib_path)
    if not p.exists():
        return False, "missing"
    try:
        calib = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"invalid json: {e}"
    try:
        launch_cell.resolve_calibrated_rate(calib, False)
    except launch_cell.CalibrationError as e:
        return False, f"not usable: {e}"
    try:
        launch_cell.check_calibration_binding(calib, cell_id, fraction)
    except launch_cell.CalibrationError as e:
        return False, f"binding: {e}"
    if current_sig is not None:
        try:
            launch_cell.check_calibration_provenance(calib, current_sig, max_age_days, now)
        except launch_cell.CalibrationError as e:
            return False, f"provenance: {e}"
    return True, f"ok (status={calib.get('status')}, rate={calib.get('rate_calibrated_rps')})"


def publish_calibration(tmp_out: Path, out_path: Path) -> Optional[dict]:
    """Atomically publish a freshly-written calibration, or invalidate a stale one.

    calibrate_rate.py writes to a per-sweep TEMP path; only a parseable temp file
    is os.replace()'d onto the cell's real calibration_file. If the sweep wrote
    nothing (crash, StaleSweepDir, docker failure) or a corrupt file, we remove
    BOTH the partial temp and any prior out_path -- so a stale ok JSON from an
    earlier run can never be mistaken for this (e.g. --recalibrate) sweep's
    success. Returns the parsed result dict on publish, else None. Temp and final
    live in the same directory so the replace is atomic on one filesystem."""
    res = None
    if tmp_out.exists():
        try:
            res = json.loads(tmp_out.read_text())
        except (OSError, json.JSONDecodeError):
            res = None
    if res is not None:
        os.replace(str(tmp_out), str(out_path))
        return res
    for p in (tmp_out, out_path):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    return None


class PreflightAbort(RuntimeError):
    """A host precondition (residual active run / unkillable orphan) refuses the
    orchestration. Carries the process exit code to return."""

    def __init__(self, rc: int, msg: str) -> None:
        super().__init__(msg)
        self.rc = rc


class BringUpFailed(RuntimeError):
    """Engine bring-up for a system exhausted its retries (host precondition)."""


class Engine:
    """A brought-up engine serving one system's sweeps."""

    def __init__(self, lifecycle, system: str, image_full: str, image_digest: str,
                 work_dir: Path, cell: dict) -> None:
        self.lifecycle = lifecycle
        self.system = system
        self.image_full = image_full
        self.image_digest = image_digest
        self.work_dir = work_dir
        self.cell = cell


class Orchestrator:
    """Drives the per-system bring-up + per-cell sweep. The seams prefixed with
    an underscore (_acquire_slot, _bring_up_fn, _sweep_fn, _teardown_fn,
    _status_fn, _dry_run_fn, _now) are injected by the unit tests so run() can be
    exercised with no docker / GPU / subprocess."""

    def __init__(self, *, campaign_yaml: Path, runs_root: Path, repo_root: Path,
                 hf_cache_host: Path, max_age_days: float, window_s: int,
                 cooldown_s: int, bringup_retries: int, recalibrate: set[str],
                 grid_map: dict, env_rates: Optional[list[float]],
                 min_free_gb: float = 20.0, logf=None) -> None:
        self.campaign_yaml = Path(campaign_yaml)
        self.runs_root = Path(runs_root)
        self.repo_root = Path(repo_root)
        self.hf_cache_host = Path(hf_cache_host)
        self.max_age_days = max_age_days
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.bringup_retries = max(1, int(bringup_retries))
        self.recalibrate = recalibrate
        self.grid_map = grid_map or {}
        self.env_rates = env_rates
        self.min_free_gb = float(min_free_gb)
        self._logf = logf

        # The args namespace launch_cell's lifecycle classes read from. Only a
        # handful of fields are touched by bring_up/teardown/wait_quiescence; we
        # supply exactly those (repo_root, runs_root, hf_cache_host,
        # component_pids) plus the two overrides the constructors default-read.
        self.args = argparse.Namespace(
            repo_root=self.repo_root,
            runs_root=self.runs_root,
            hf_cache_host=self.hf_cache_host,
            component_pids=Path(os.environ.get(
                "WOSAR_COMPONENT_PIDS",
                str(Path.home() / "wosar" / "dynamo_component_pids.json"))),
            gpu_device_override=None,
            duration_s_override=None,
        )

        # Runtime state for signal-safe teardown: the slot handle and the engine
        # currently up (bring-up arms launch_cell's abort-cleanup for the engine's
        # containers, so cleanup_after_abort() covers a partial bring-up too).
        self._slot = None
        self._active_engine: Optional[Engine] = None
        self._aborting = False
        self._prev_handlers: dict = {}
        self._install_signals = True

        # Injection seams.
        self._acquire_slot = reaper.acquire_run_slot
        self._preflight_fn = self._real_preflight
        self._bring_up_fn = self._real_bring_up
        self._sweep_fn = self._real_sweep
        self._teardown_fn = self._real_teardown
        self._status_fn = self._real_status
        self._dry_run_fn = self._real_dry_run
        self._abort_cleanup = launch_cell.cleanup_after_abort
        self._invoke_calibrate = self._real_invoke_calibrate
        self._now = time.time
        self._specs = None  # tests may inject a synthetic schedule

    # -- logging ------------------------------------------------------------

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[calibrate_dow] {ts} {msg}"
        print(line, flush=True)
        if self._logf is not None:
            self._logf.write(line + "\n")
            self._logf.flush()

    # -- planning -----------------------------------------------------------

    def specs(self) -> list:
        if self._specs is None:
            campaign = camp.load_campaign(self.campaign_yaml)
            self._specs = camp.build_schedule(campaign, self.campaign_yaml)
        return self._specs

    def plan_groups(self) -> list[tuple[str, list]]:
        return group_by_system(self.specs())

    def forced(self, cell_id: str) -> bool:
        return "all" in self.recalibrate or cell_id in self.recalibrate

    def resolve_grid(self, spec) -> list[float]:
        sysname = system_of(spec.cell_id)
        if self.grid_map:
            if spec.cell_id in self.grid_map:
                return [float(x) for x in self.grid_map[spec.cell_id]]
            if sysname in self.grid_map:
                return [float(x) for x in self.grid_map[sysname]]
        if self.env_rates:
            return list(self.env_rates)
        return list(DEFAULT_RATE_GRIDS.get(sysname, _FALLBACK_GRID))

    # -- cell / image helpers ----------------------------------------------

    def _render_cell(self, cell_yaml) -> dict:
        raw = yaml.safe_load(Path(cell_yaml).read_text())
        return launch_cell.render_in_obj(
            raw, repo_root=str(self.repo_root),
            hf_cache_host=str(self.hf_cache_host), replica="01")

    def _resolve_image(self, cell: dict, verify_digest: bool) -> tuple[str, str, dict]:
        eng = cell["engine"]
        image_full = f'{eng["image_repo"]}:{eng["image_tag"]}'
        pin = launch_cell.load_image_pin(self.repo_root / eng["digest_pin_file"])
        if pin.get("image_tag") != image_full:
            launch_cell.die(
                f"image pin mismatch: cell expects {image_full}, pin file has "
                f"{pin.get('image_tag')!r}", rc=6)
        digest = str(pin.get("digest", "")).strip()
        if not digest:
            launch_cell.die(
                f"image pin file for {image_full} has no 'digest'; run "
                "scripts/utils/pin_images.sh before calibrating.", rc=6)
        if verify_digest:
            launch_cell.verify_image_digest(image_full, digest)
        return image_full, digest, pin

    def _current_sig_for(self, spec) -> Optional[dict]:
        """Host/image signature to check an existing calibration's provenance
        against, mirroring launch_cell's dispatch-time signature. Returns None if
        the image pin cannot be read (die() -> SystemExit), so the skip check
        falls back to binding + usable-rate only."""
        try:
            cell = self._render_cell(spec.cell_yaml)
            image_full, digest, _ = self._resolve_image(cell, verify_digest=False)
        except SystemExit:
            return None
        gpu_name, driver = launch_cell.gpu_name_and_driver()
        return launch_cell.current_calibration_signature(
            socket.gethostname(), gpu_name, driver, image_full, digest)

    def _real_status(self, spec) -> tuple[bool, str]:
        sig = self._current_sig_for(spec)
        return calibration_is_valid(
            spec.calibration_file, spec.cell_id, spec.calibration_fraction,
            sig, self.max_age_days, self._now())

    # -- bring-up / teardown -----------------------------------------------

    def _new_work_dir(self, system: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        d = self.runs_root / "calib_dow" / f"{system}_{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _real_bring_up(self, system: str, rep_spec) -> Engine:
        cell = self._render_cell(rep_spec.cell_yaml)
        image_full, digest, pin = self._resolve_image(cell, verify_digest=True)
        work = self._new_work_dir(system)
        last_rc = None
        for attempt in range(1, self.bringup_retries + 1):
            run_dir = work / f"attempt{attempt}"
            log_dir = run_dir / "logs"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            lifecycle = launch_cell.make_lifecycle(
                cell, self.args, run_dir, log_dir, pin, image_full)
            self.log(f"system {system}: bring-up attempt {attempt}/{self.bringup_retries} "
                     f"(lifecycle={lifecycle.kind}, image={image_full})")
            try:
                lifecycle.bring_up()
                self.log(f"system {system}: engine up")
                return Engine(lifecycle, system, image_full, digest, work, cell)
            except (SystemExit, Exception) as e:  # noqa: BLE001
                # bring_up die()s (SystemExit) on the expected failures; a raw
                # exception is still a failed attempt. Either way, try to tear
                # down whatever partially started before retrying.
                last_rc = getattr(e, "code", repr(e))
                self.log(f"system {system}: bring-up attempt {attempt} FAILED ({last_rc!r})")
                try:
                    lifecycle.teardown()
                except Exception as te:  # noqa: BLE001
                    self.log(f"system {system}: teardown after failed bring-up raised {te!r}")
                try:
                    lifecycle.wait_quiescence()
                except Exception:  # noqa: BLE001
                    pass
        raise BringUpFailed(
            f"{system}: bring-up failed after {self.bringup_retries} attempt(s) (last rc={last_rc})")

    def _real_teardown(self, engine: Engine) -> None:
        self.log(f"system {engine.system}: tearing down engine")
        try:
            engine.lifecycle.teardown()
        except Exception as e:  # noqa: BLE001
            self.log(f"system {engine.system}: teardown raised {e!r}")
        try:
            engine.lifecycle.wait_quiescence()
        except Exception as e:  # noqa: BLE001
            self.log(f"system {engine.system}: wait_quiescence raised {e!r}")

    # -- sweep --------------------------------------------------------------

    def _real_sweep(self, spec, engine: Engine, grid: list[float]) -> dict:
        cell = self._render_cell(spec.cell_yaml)
        overrides = cell["workload"]["client_config_overrides"]
        sweep_root = engine.work_dir / "sweeps" / spec.cell_id
        # Fresh probe dir every time: calibrate_rate refuses to reuse a sweep dir
        # that still holds requests_*.csv (StaleSweepDir), and a re-calibration
        # deliberately discards the prior probe.
        if sweep_root.exists():
            shutil.rmtree(sweep_root)
        sweep_root.mkdir(parents=True, exist_ok=True)

        config = launch_cell.materialize_client_config(
            self.repo_root, sweep_root, cell, 1)
        out_path = Path(spec.calibration_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # calibrate_rate writes to a TEMP path next to the final file; only a
        # parseable temp is atomically published (publish_calibration). This is
        # what stops a stale ok JSON (e.g. under --recalibrate) from surviving a
        # sweep that crashed before rewriting and passing as a false success.
        tmp_out = out_path.with_name(out_path.name + ".tmp")

        fraction = spec.calibration_fraction if spec.calibration_fraction is not None else 0.85
        if spec.calibration_fraction is None:
            self.log(f"{spec.cell_id}: WARNING cell declares no calibration_fraction; "
                     f"using {fraction} (binding will be lenient)")

        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "calibrate_rate.py"),
            "--config", str(config),
            "--base-url", str(overrides["base_url"]),
            "--protocol", str(overrides["protocol"]),
            "--model", str(overrides["model"]),
            "--rates", ",".join(str(r) for r in grid),
            "--window-seconds", str(self.window_s),
            "--cooldown-seconds", str(self.cooldown_s),
            "--concurrency-cap", str(int(overrides.get("concurrency_cap", 64))),
            "--fraction", str(fraction),
            "--cell-id", spec.cell_id,
            "--system", engine.system,
            "--image-tag", engine.image_full,
            "--image-digest", engine.image_digest,
            "--output", str(tmp_out),
            "--sweep-dir", str(sweep_root / "sweep"),
        ]
        rc = self._invoke_calibrate(cmd, tmp_out)
        res = publish_calibration(tmp_out, out_path) or {}
        status = res.get("status") or "no_output"
        result = {
            "status": status,
            "ceiling_rps": res.get("ceiling_rps"),
            "rate_calibrated_rps": res.get("rate_calibrated_rps"),
            "rc": rc,
            "grid": list(grid),
            "skipped": False,
        }
        if status != "ok":
            result["suggested_grid"] = suggest_next_grid(status, grid)
        return result

    @staticmethod
    def _real_invoke_calibrate(cmd: list[str], _tmp_out: Path) -> int:
        return subprocess.run(cmd, capture_output=False).returncode

    # -- verdict ------------------------------------------------------------

    def _real_dry_run(self) -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "campaign.py"),
             "--campaign-yaml", str(self.campaign_yaml), "--dry-run"],
            capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    # -- orchestration ------------------------------------------------------

    def run_plan_only(self) -> int:
        """--dry-run for the orchestrator itself: print the grouped plan and each
        cell's current validity, bring nothing up."""
        groups = self.plan_groups()
        total = sum(len(cells) for _, cells in groups)
        valid = 0
        self.log(f"plan: {total} cells across {len(groups)} systems "
                 f"(order: {', '.join(s for s, _ in groups)})")
        for system, cells in groups:
            self.log(f"  system {system}: {len(cells)} cells")
            for spec in cells:
                ok, reason = self._status_fn(spec)
                if ok:
                    valid += 1
                grid = self.resolve_grid(spec)
                mark = "VALID " if ok else "TODO  "
                self.log(f"    {mark} {spec.cell_id}  frac={spec.calibration_fraction}  "
                         f"grid={grid}  ({reason})")
        self.log(f"plan: {valid}/{total} already valid, {total - valid} to calibrate")
        return EXIT_OK

    def run(self) -> int:
        slot = self._acquire_slot(self.runs_root)
        if slot is None:
            self.log(f"another launcher holds the run-slot lock on {self.runs_root}; "
                     "the host is strictly serial -- refusing to start.")
            return EXIT_LOCK_HELD
        self._slot = slot
        self._install_signal_handlers()
        try:
            # Under the held slot, run the SAME host preconditions launch_cell runs:
            # the free-space gate and the pre-run orphan reap (a stray run_client
            # from a prior crash could load the calibration endpoint and skew the
            # ceiling). A residual active run or unkillable orphan aborts here.
            try:
                self._preflight_fn()
            except PreflightAbort as e:
                self.log(f"FATAL: {e} -- refusing to start.")
                return e.rc
            return self._run_locked()
        finally:
            self._restore_signal_handlers()
            try:
                slot.close()
            except Exception:  # noqa: BLE001
                pass
            self._slot = None

    # -- host preconditions + signal-safe teardown --------------------------

    def _real_preflight(self) -> None:
        # Free-space gate across the runs-root (sweep CSVs) and the docker
        # data-root (engine images + container logs). die()s (rc=7) if below.
        docker_root = launch_cell.docker_root_dir()
        launch_cell.require_free_space(
            [self.runs_root, docker_root], self.min_free_gb, label="calibrate_dow")
        # Pre-run orphan reap, valid ONLY because we hold the run-slot lock. No
        # current_run_id to spare: nothing of ours is running yet at this point.
        for line in reaper.reap_orphans(self.runs_root, current_run_id=None):
            self.log(line)
        stuck = reaper.ledger_run_ids(self.runs_root)
        if stuck:
            raise PreflightAbort(
                EXIT_PRECONDITION,
                f"prior run(s) {stuck} are still active (launcher alive) or have an "
                f"unkillable orphan -- calibrating over a live run would skew ceilings")
        hw_lines, hw_unkillable = reaper.reap_host_wide(self.runs_root, current_run_id=None)
        for line in hw_lines:
            self.log(line)
        if hw_unkillable:
            raise PreflightAbort(
                EXIT_PRECONDITION,
                f"host-wide reaper could not kill orphan process(es) {hw_unkillable} "
                f"referencing {self.runs_root}")

    def _install_signal_handlers(self) -> None:
        """Route SIGTERM/SIGINT/SIGHUP through the SAME abort-cleanup path
        launch_cell uses: tear the active engine down (bring-up armed
        cleanup_after_abort for its containers, so a partial bring-up is covered
        too) and release the lock before exiting, instead of leaving containers
        alive on a kill / dropped SSH session."""
        if not self._install_signals:
            return
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                self._prev_handlers[sig] = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):  # not main thread / unsupported
                pass

    def _restore_signal_handlers(self) -> None:
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers = {}

    def _handle_signal(self, signum, _frame) -> None:
        if self._aborting:
            return
        self._aborting = True
        self.log(f"received signal {signum}: tearing down and releasing the run-slot lock")
        if self._active_engine is not None:
            try:
                self._teardown_fn(self._active_engine)
            except Exception as e:  # noqa: BLE001
                self.log(f"teardown during abort raised {e!r}")
        # Backstop for a partial bring-up (no Engine yet): remove the containers
        # bring-up armed via launch_cell.enable_abort_cleanup.
        try:
            self._abort_cleanup()
        except Exception:  # noqa: BLE001
            pass
        if self._slot is not None:
            try:
                self._slot.close()
            except Exception:  # noqa: BLE001
                pass
        sys.exit(EXIT_INTERRUPTED)

    def _run_locked(self) -> int:
        groups = self.plan_groups()
        total = sum(len(cells) for _, cells in groups)
        results: dict[str, dict] = {}
        durations: list[float] = []
        done = 0
        self.log(f"start: {total} cells across {len(groups)} systems "
                 f"(order: {', '.join(s for s, _ in groups)}); "
                 f"window={self.window_s}s cooldown={self.cooldown_s}s")

        for system, cells in groups:
            todo = []
            for spec in cells:
                if self.forced(spec.cell_id):
                    todo.append(spec)
                    continue
                ok, reason = self._status_fn(spec)
                if ok:
                    done += 1
                    results[spec.cell_id] = {"status": "ok", "skipped": True,
                                             "ceiling_rps": None, "rate_calibrated_rps": None}
                    self.log(f"[{done}/{total}] {spec.cell_id}: already valid ({reason}) -- skip")
                else:
                    todo.append(spec)

            if not todo:
                self.log(f"system {system}: all {len(cells)} calibrations valid -- skipping bring-up")
                continue

            try:
                engine = self._bring_up_fn(system, todo[0])
            except BringUpFailed as e:
                self.log(f"FATAL: {e} -- host precondition, aborting orchestration.")
                return EXIT_BRINGUP
            self._active_engine = engine

            try:
                for spec in todo:
                    done += 1
                    grid = self.resolve_grid(spec)
                    eta = self._eta(durations, total - done)
                    self.log(f"[{done}/{total}] {spec.cell_id} (system={system}) "
                             f"calibrating frac={spec.calibration_fraction} grid={grid} {eta}")
                    t0 = self._now()
                    res = self._sweep_fn(spec, engine, grid)
                    durations.append(self._now() - t0)
                    results[spec.cell_id] = res
                    extra = ""
                    if res.get("status") != "ok" and res.get("suggested_grid"):
                        extra = f"  -> try grid {res['suggested_grid']}"
                    self.log(f"[{done}/{total}] {spec.cell_id}: status={res.get('status')} "
                             f"ceiling={res.get('ceiling_rps')} "
                             f"rate={res.get('rate_calibrated_rps')} "
                             f"({durations[-1] / 60:.1f} min){extra}")
            finally:
                self._teardown_fn(engine)
                self._active_engine = None

        return self._finish(results)

    @staticmethod
    def _eta(durations: list[float], remaining: int) -> str:
        if not durations or remaining <= 0:
            return ""
        mean = sum(durations) / len(durations)
        return f"[ETA ~{mean * remaining / 3600:.1f}h, {remaining} left]"

    def _finish(self, results: dict[str, dict]) -> int:
        non_ok = {cid: r for cid, r in results.items() if r.get("status") != "ok"}
        self.log("=" * 72)
        self.log(f"SUMMARY: {len(results)} cells, "
                 f"{len(results) - len(non_ok)} ok, {len(non_ok)} need attention")
        header = f"{'cell':<30} {'status':<16} {'ceiling':>9} {'rate':>9}  suggested_next_grid"
        self.log(header)
        for spec in self.specs():
            r = results.get(spec.cell_id)
            if r is None:
                continue
            sug = r.get("suggested_grid")
            sug_s = "" if not sug else ",".join(str(x) for x in sug)
            skip = " (skipped)" if r.get("skipped") else ""
            self.log(f"{spec.cell_id:<30} {str(r.get('status')) + skip:<16} "
                     f"{_fmt(r.get('ceiling_rps')):>9} {_fmt(r.get('rate_calibrated_rps')):>9}  {sug_s}")

        self.log("=" * 72)
        self.log("VERDICT: running the campaign pre-flight the runs will use "
                 "(campaign.py --dry-run) ...")
        rc, out = self._dry_run_fn()
        for line in _head(out, 14):
            self.log("  " + line)
        missing = rc != 0
        if missing:
            self.log("VERDICT: dry-run still reports MISSING/invalid calibration(s) "
                     f"(exit={rc}). NOT ready.")
        else:
            self.log("VERDICT: dry-run clean -- 0 MISSING. The screening campaign can start.")

        if non_ok or missing:
            return EXIT_SOME_FAILED
        return EXIT_OK


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _head(text: str, n: int) -> list[str]:
    lines = text.splitlines()
    return lines[:n]


def _load_grid_map(path: Optional[Path]) -> dict:
    if not path:
        return {}
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(doc, dict):
        raise SystemExit(f"--rate-grids {path}: expected a mapping of cell_id/system -> [rates]")
    out = {}
    for k, v in doc.items():
        out[str(k)] = [float(x) for x in v]
    return out


def _env_rates() -> Optional[list[float]]:
    raw = os.environ.get("WOSAR_CALIB_RATES", "").strip()
    if not raw:
        return None
    return [float(x) for x in raw.split(",") if x.strip()]


def build_orchestrator(args) -> Orchestrator:
    """Read the campaign's own host config (runs_root, paths, max-age) so the
    orchestrator targets the same box with the same staleness gate as the run."""
    campaign = camp.load_campaign(args.campaign_yaml)
    yaml_dir = args.campaign_yaml.parent
    runs_root = Path(campaign["runs_root"])
    paths = campaign.get("paths", {})
    hf_cache_host = Path(paths["hf_cache_host"])
    repo_root = (Path(paths["repo_root"]) if paths.get("repo_root")
                 else yaml_dir.parent.parent)
    max_age_days = float(campaign.get("calibration_max_age_days",
                                      launch_cell.DEFAULT_CALIBRATION_MAX_AGE_DAYS))
    # Same pre-run free-space floor the runs use (dow_campaign.yaml: 50 GB).
    min_free_gb = float(campaign.get("min_free_gb", 20.0))

    logf = None
    if not args.no_log_file:
        log_dir = yaml_dir / "state" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logf = (log_dir / f"calibrate_dow_{stamp}.log").open("a", buffering=1)

    recal = set()
    for item in (args.recalibrate or []):
        recal.update(x.strip() for x in item.split(",") if x.strip())

    return Orchestrator(
        campaign_yaml=args.campaign_yaml,
        runs_root=runs_root,
        repo_root=repo_root,
        hf_cache_host=hf_cache_host,
        max_age_days=max_age_days,
        window_s=args.window_seconds,
        cooldown_s=args.cooldown_seconds,
        bringup_retries=args.bringup_retries,
        recalibrate=recal,
        grid_map=_load_grid_map(args.rate_grids),
        env_rates=_env_rates(),
        min_free_gb=min_free_gb,
        logf=logf,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Week-1 calibration orchestrator for the DoW screening campaign.")
    p.add_argument("--campaign-yaml", type=Path, default=DEFAULT_CAMPAIGN_YAML,
                   help="DoW campaign descriptor (default: campaigns/extension/dow_campaign.yaml).")
    p.add_argument("--recalibrate", action="append", default=[],
                   help="Force re-calibration of a cell_id (repeatable, or comma-list), "
                        "or 'all'. Ignores any existing valid JSON for those cells.")
    p.add_argument("--window-seconds", type=int, default=240,
                   help="Per-rate sweep window (passed to calibrate_rate).")
    p.add_argument("--cooldown-seconds", type=int, default=30,
                   help="Idle between sweep rates so the KV cache flushes.")
    p.add_argument("--rate-grids", type=Path, default=None,
                   help="Optional yaml mapping cell_id or system -> [rates] to override "
                        "the built-in per-system default grids (cell_id wins over system). "
                        "The env var WOSAR_CALIB_RATES=r1,r2,... overrides globally.")
    p.add_argument("--bringup-retries", type=int, default=1,
                   help="Engine bring-up attempts per system before a hard stop.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grouped plan and each cell's current validity, then exit "
                        "(no lock, no bring-up).")
    p.add_argument("--no-log-file", action="store_true",
                   help="Do not tee to campaigns/extension/state/logs/ (stdout only).")
    args = p.parse_args()

    orch = build_orchestrator(args)
    if args.dry_run:
        sys.exit(orch.run_plan_only())
    sys.exit(orch.run())


if __name__ == "__main__":
    main()
