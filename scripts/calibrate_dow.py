#!/usr/bin/env python3
"""Week-1 calibration orchestrator for the DoW screening campaign.

Turns the 57 MISSING calibrations that dow_campaign.yaml's pre-flight reports
into 57 valid per-cell calibration JSONs, one FRESH engine bring-up per sweep.

Why this exists (and why it is Python, not bash): the DoW cells all bind their
calibration to a specific cell_id + calibration_fraction at dispatch
(launch_cell.check_calibration_binding), and the five DoW factors are all
workload-side. This orchestrator reuses launch_cell's PRODUCTION lifecycle to
bring the engine up, then runs the existing calibrate_rate.py sweep against that
endpoint. There is deliberately NO second bring-up path: the frontend-identity
bug taught us what divergent bring-up paths cost, so we drive the SAME
make_lifecycle / bring_up / teardown launch_cell drives (with monitors and the
run client simply not started), and hold the reaper run-slot lock for the whole
orchestration so a real run cannot start underneath it.

FRESH STACK PER SWEEP (teardown + bring-up between cells, all systems): the
first cut kept ONE engine per system alive for the whole system's sweeps (the
load-once optimization). A field episode (2026-07-06) showed why that is unsafe:
a single Dynamo stack held alive ~9h developed the NIXL KV-transfer total-stall
pathology mid-way, and the cells scheduled during the sick window recorded dead-
endpoint sweeps -- with no runtime health check (which launch_cell has and this
orchestrator did not) to notice. Fresh-per-sweep is also methodologically
required: every DoW run starts on a freshly brought-up stack, so the ceiling
must be measured on one too. The 2-3 min bring-up cost is negligible against the
40-80 min sweeps.

Per-cell, NOT deduped: the 3 center points of a system share one workload shape,
so there are only 51 distinct shapes -- but the binding checks each file's own
cell_id, and every cell names its own calibration_file, so we run one honest
sweep per cell (57) rather than stamping a shared sweep with a foreign cell_id.
The 6 extra center-point sweeps are cheap (calibrate_rate early-stops past the
knee) and keep each file a real independent calibration of its own cell.

Engine failure vs measurement verdict (the 2026-07-06 finding): a dead/sick
endpoint must NEVER be misfiled as a measurement verdict. A sweep whose LOWEST
grid rate ends with n_ok == 0 (or a catastrophic drop/error storm there), or a
post-sweep engine health check that fails, is classified status "engine_failure"
-- distinct from no_stable_point. On engine_failure we capture evidence (docker
logs tail into the calibration work dir), tear down, bring up a FRESH stack, and
retry that cell once; a second engine_failure records the cell as engine_failure
and moves on. Only a HEALTHY sweep may conclude no_stable_point. Each calibration
JSON also records provenance -- the stack age at sweep start (~0 with fresh-per-
sweep) and the engine_failure retry count -- so any regression of this class is
visible in the data.

Failure policy: a sweep that ends status != ok (no_stable_point / did_not_
saturate / engine_failure) does NOT halt the orchestrator -- unlike a 36h run, a
failed calibration corrupts nothing, and halting 50 good sweeps for one bad shape
wastes the week-1 budget. Such a sweep is recorded, the run continues, and the
process exits non-zero at the end with a summary table (cell, status, suggested
wider/narrower rate grid or engine-failure reason). A HARD stop happens only for
host-level preconditions (run-slot lock held, image pin / docker / env, or engine
bring-up failing after its retries). engine_failure is a non-ok status, so the
campaign pre-flight refuses it -- nothing unsound can leak into a real run.

Resumability: a cell whose calibration JSON already exists, is status=ok, and
passes the SAME binding + provenance + max-age gates the campaign applies is
skipped (the check reuses launch_cell's own gate functions, not a reimplement)
-- and a skipped cell brings no stack up at all. --recalibrate <cell_id|all>
forces a re-sweep. The whole thing is safe to Ctrl-C and re-run.

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
import calibrate_rate
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


def classify_engine_failure(res: dict, health_reason: Optional[str]) -> Optional[str]:
    """Distinguish an ENGINE failure (dead / sick endpoint) from a measurement
    verdict. Returns a short reason string if this sweep is an engine failure,
    else None. This is the 2026-07-06 finding: a Dynamo NIXL total-stall episode
    that sampled several cells as dead-endpoint sweeps had every one misfiled as
    "no_stable_point" -- a measurement verdict -- because the sweep found no
    stable operating point. It found none because NOTHING completed, which is an
    engine failure, not a ceiling below the grid.

    An `ok` sweep is never an engine failure: an ok verdict requires a full
    contiguous stable prefix ending in a graceful knee, so the endpoint
    demonstrably served throughout -- a dead engine breaks the prefix with an
    n_ok==0 step and can never read `ok`. For a NON-ok sweep two signals apply:

      1. The LOWEST offered grid rate ended with n_ok == 0: a dead endpoint.
         A healthy-but-slow server still completes SOME requests at its lowest
         grid rate; only a dead one completes none (field row: 0.25 rps,
         n_offered=159, n_ok=0, drop_rate=0.60, p99=NaN). We key on n_ok==0 --
         NOT on a high drop rate -- deliberately: a high drop/error rate at a
         low rate is a legitimate stress signal (launch_cell's own endpoint-dead
         detector counts only all-fail windows for the same reason), so gating on
         drop_rate would false-reject a genuinely saturated lowest rate that is a
         real no_stable_point. The all-fail (n_ok==0) window is the clean signal.
      2. `health_reason` is set: the post-sweep engine health check (which
         launch_cell runs continuously and this orchestrator now runs too) found
         the stack lost a container or stopped serving during/around the sweep.

    Only a HEALTHY sweep may conclude no_stable_point -- this is what stops a
    sick-episode sample from being recorded as a real unstable ceiling."""
    status = (res.get("status") or "").strip().lower()
    if status == "ok":
        return None
    rows = res.get("sweep_rows") or []
    if rows:
        # The ascending sweep runs the lowest offered rate first; pick it by
        # offered_rate so we do not depend on list order.
        lowest = min(rows, key=lambda r: _row_offered(r))
        n_ok = lowest.get("n_ok")
        if n_ok is not None:
            try:
                if int(n_ok) == 0:
                    return (f"dead endpoint: lowest grid rate "
                            f"{lowest.get('offered_rate')} rps completed 0 of "
                            f"{lowest.get('n_offered')} offered "
                            f"(drop_rate={lowest.get('drop_rate')})")
            except (TypeError, ValueError):
                pass
    if health_reason:
        return f"engine health check failed: {health_reason}"
    return None


def _row_offered(r: dict) -> float:
    try:
        return float(r.get("offered_rate"))
    except (TypeError, ValueError):
        return float("inf")


def calibration_is_valid(calib_path, cell_id: str, fraction: Optional[float],
                         current_sig: Optional[dict], max_age_days: Optional[float],
                         now: float,
                         min_method_version: Optional[int] = None) -> tuple[bool, str]:
    """Is an existing calibration JSON good enough to SKIP (re-)calibrating this
    cell? Reuses launch_cell's own gate functions so "valid" means exactly what
    the campaign pre-flight (binding + usable rate) AND the run dispatch
    (provenance + max-age + method) will accept -- no reimplementation.

    current_sig None means the host/image signature could not be built (e.g. the
    image pin is unreadable); the SIGNATURE leg is then skipped and the caller
    proceeds to calibrate, where the real pin gate will surface any problem. The
    method-version gate does not need the signature, so it is enforced regardless
    -- a v1 file must be re-taken as v2 even on a host whose pin is unreadable."""
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
    # Method gate first, independent of the host/image signature: a superseded
    # (e.g. v1 finite-window) ceiling must be re-taken even when current_sig is
    # None. Pass an empty signature so only the age+method legs run here.
    if min_method_version is not None:
        try:
            launch_cell.check_calibration_provenance(
                calib, {}, None, now, min_method_version=min_method_version)
        except launch_cell.CalibrationError as e:
            return False, f"method: {e}"
    if current_sig is not None:
        try:
            launch_cell.check_calibration_provenance(
                calib, current_sig, max_age_days, now,
                min_method_version=min_method_version)
        except launch_cell.CalibrationError as e:
            return False, f"provenance: {e}"
    return True, f"ok (status={calib.get('status')}, rate={calib.get('rate_calibrated_rps')})"


def publish_calibration(tmp_out: Path, out_path: Path,
                        rc: Optional[int] = None,
                        orchestration: Optional[dict] = None) -> Optional[dict]:
    """Atomically publish a freshly-written calibration, or invalidate a stale one.

    calibrate_rate.py writes to a per-sweep TEMP path; only a parseable temp file
    is os.replace()'d onto the cell's real calibration_file. If the sweep wrote
    nothing (crash, StaleSweepDir, docker failure) or a corrupt file, we remove
    BOTH the partial temp and any prior out_path -- so a stale ok JSON from an
    earlier run can never be mistaken for this (e.g. --recalibrate) sweep's
    success. Returns the parsed result dict on publish, else None. Temp and final
    live in the same directory so the replace is atomic on one filesystem.

    When rc is given, it must be CONSISTENT with the temp file's own verdict:
    calibrate_rate maps status -> exit code (exit_code_for_status), so a parseable
    temp whose rc does not match its status (e.g. an 'ok' JSON left behind by a
    process that was then SIGKILLed, rc=-9) is treated as a crashed sweep and
    purged -- an abnormal exit must never be published as a success.

    `orchestration` (when given) is merged into the published JSON under an
    "orchestration" key BEFORE the atomic replace -- this is where the fresh-per-
    sweep provenance lives (stack age at sweep start, engine_failure retry count),
    recorded on EVERY published calibration regardless of verdict."""
    res = None
    if tmp_out.exists():
        try:
            res = json.loads(tmp_out.read_text())
        except (OSError, json.JSONDecodeError):
            res = None
    if res is not None and rc is not None:
        expected_rc = calibrate_rate.exit_code_for_status(res.get("status", ""))
        if int(rc) != expected_rc:
            res = None  # rc contradicts the written verdict -> not trustworthy
    if res is not None:
        if orchestration is not None:
            res["orchestration"] = dict(orchestration)
            # Re-serialize the merged dict into the temp so the atomic replace
            # publishes the provenance block too (keeps the one-replace guarantee).
            tmp_out.write_text(json.dumps(res, indent=2))
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
                 min_free_gb: float = 20.0, step_min_s: float = 600.0,
                 ceil_mult: float = 20.0, min_method_version: Optional[int] = None,
                 engine_failure_retries: int = 1, logf=None) -> None:
        self.campaign_yaml = Path(campaign_yaml)
        self.runs_root = Path(runs_root)
        self.repo_root = Path(repo_root)
        self.hf_cache_host = Path(hf_cache_host)
        self.max_age_days = max_age_days
        # window_s is the legacy fixed per-rate window; the v2 method sizes each
        # step by wall duration (max(step_min_s, ceil_mult x p99)), so window_s is
        # kept only as the step-floor fallback when step_min_s is unset.
        self.window_s = window_s
        self.step_min_s = float(step_min_s) if step_min_s is not None else float(window_s)
        self.ceil_mult = float(ceil_mult)
        # HARD method floor for the SKIP check: an existing v1 ceiling is treated
        # as invalid so this campaign re-takes it as v2 (mirrors the campaign
        # pre-flight / dispatch gate). None -> no method gate.
        self.min_method_version = (int(min_method_version)
                                   if min_method_version is not None else None)
        self.cooldown_s = cooldown_s
        self.bringup_retries = max(1, int(bringup_retries))
        # Fresh-stack retries granted to a cell that records an engine_failure
        # (dead/sick endpoint). Default 1: retry once on a fresh stack, then a
        # second engine_failure records the cell and moves on.
        self.engine_failure_retries = max(0, int(engine_failure_retries))
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
        self._last_results: dict[str, dict] = {}
        self._aborting = False
        self._prev_handlers: dict = {}
        self._install_signals = True

        # Injection seams.
        self._acquire_slot = reaper.acquire_run_slot
        self._preflight_fn = self._real_preflight
        self._bring_up_fn = self._real_bring_up
        self._sweep_fn = self._real_sweep
        self._teardown_fn = self._real_teardown
        self._health_fn = self._real_health
        self._capture_evidence_fn = self._real_capture_evidence
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
            sig, self.max_age_days, self._now(),
            min_method_version=self.min_method_version)

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

    def _real_sweep(self, spec, engine: Engine, grid: list[float], *,
                    stack_age_s: float = 0.0,
                    engine_failure_retries: int = 0) -> dict:
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
            "--step-min-seconds", str(self.step_min_s),
            "--ceil-mult", str(self.ceil_mult),
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
        # Fresh-per-sweep provenance, recorded on EVERY published calibration:
        # with a fresh stack the age should be ~0, so any non-trivial value in the
        # data flags a regression back to a shared long-lived stack; the retry
        # count makes a dead-endpoint episode visible after the fact.
        orchestration = {
            "stack_age_at_sweep_start_s": round(max(0.0, float(stack_age_s)), 3),
            "engine_failure_retries": int(engine_failure_retries),
            "fresh_stack_per_sweep": True,
        }
        res = publish_calibration(tmp_out, out_path, rc=rc,
                                  orchestration=orchestration) or {}
        status = res.get("status") or "no_output"
        result = {
            "status": status,
            "ceiling_rps": res.get("ceiling_rps"),
            "rate_calibrated_rps": res.get("rate_calibrated_rps"),
            "rc": rc,
            "grid": list(grid),
            "skipped": False,
            "stack_age_s": orchestration["stack_age_at_sweep_start_s"],
            "engine_failure_retries": int(engine_failure_retries),
            # The published sweep rows, for engine-failure classification (a dead
            # endpoint shows n_ok==0 at the lowest grid rate).
            "sweep_rows": res.get("sweep") or [],
        }
        if status != "ok":
            result["suggested_grid"] = suggest_next_grid(status, grid)
        return result

    @staticmethod
    def _real_invoke_calibrate(cmd: list[str], _tmp_out: Path) -> int:
        return subprocess.run(cmd, capture_output=False).returncode

    # -- engine health + failure evidence -----------------------------------

    def _real_health(self, engine: Engine) -> Optional[str]:
        """Post-sweep engine liveness, using launch_cell's OWN per-lifecycle
        health_check (container(s) still running, frontend still listing the
        model, ...). None == healthy, else a short reason. calibrate_dow ran no
        health check before this batch; that is why the 2026-07-06 stall was only
        caught after the fact, in the sweep data."""
        try:
            return engine.lifecycle.health_check()
        except Exception as e:  # noqa: BLE001
            return f"health_check raised {e!r}"

    def _stack_container_names(self, engine: Engine) -> list[str]:
        lc = engine.lifecycle
        if hasattr(lc, "stack_containers"):
            try:
                return list(lc.stack_containers())
            except Exception:  # noqa: BLE001
                return []
        name = getattr(lc, "container_name", None)
        return [name] if name else []

    def _real_capture_evidence(self, engine: Engine, spec, reason: str,
                               ef_retries: int) -> None:
        """Dump each stack container's docker-logs tail into the calibration work
        dir BEFORE teardown removes the containers, so a dead-endpoint episode
        leaves a forensic trail (the finding: calibrate_dow captured nothing, so
        the stall could only be reconstructed from the sweep rows)."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ev_dir = engine.work_dir / "engine_failures" / f"{spec.cell_id}_attempt{ef_retries}_{stamp}"
        try:
            ev_dir.mkdir(parents=True, exist_ok=True)
            (ev_dir / "reason.txt").write_text(reason + "\n")
            for name in self._stack_container_names(engine):
                launch_cell.save_docker_logs(name, ev_dir / f"docker_{name}.log")
            self.log(f"{spec.cell_id}: engine-failure evidence captured -> {ev_dir}")
        except OSError as e:
            self.log(f"{spec.cell_id}: could not capture engine-failure evidence: {e!r}")

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
                 f"(order: {', '.join(s for s, _ in groups)}); FRESH stack per "
                 f"sweep; step_min={self.step_min_s}s cooldown={self.cooldown_s}s")

        for system, cells in groups:
            for spec in cells:
                done += 1
                # Skip a cell whose existing calibration already passes the run's
                # own gates -- and skip its bring-up too (a skipped cell churns no
                # stack). --recalibrate forces a re-sweep.
                if not self.forced(spec.cell_id):
                    ok, reason = self._status_fn(spec)
                    if ok:
                        results[spec.cell_id] = {
                            "status": "ok", "skipped": True,
                            "ceiling_rps": None, "rate_calibrated_rps": None}
                        self.log(f"[{done}/{total}] {spec.cell_id}: already valid "
                                 f"({reason}) -- skip")
                        continue
                try:
                    results[spec.cell_id] = self._calibrate_cell(
                        system, spec, done, total, durations)
                except BringUpFailed as e:
                    self.log(f"FATAL: {e} -- host precondition, aborting orchestration.")
                    self._last_results = results
                    return EXIT_BRINGUP

        self._last_results = results
        return self._finish(results)

    def _calibrate_cell(self, system: str, spec, done: int, total: int,
                        durations: list[float]) -> dict:
        """Calibrate one cell on a FRESH stack. On an engine_failure (dead/sick
        endpoint) capture evidence, tear down, bring up a fresh stack, and retry
        up to self.engine_failure_retries times; a final engine_failure is
        recorded (status engine_failure) and the caller moves on. Only a HEALTHY
        sweep is allowed to conclude no_stable_point. BringUpFailed propagates to
        the caller as a hard stop (a host precondition)."""
        grid = self.resolve_grid(spec)
        ef_retries = 0
        while True:
            # Fresh stack per sweep (methodologically required + the cure for the
            # long-lived-stack stall). BringUpFailed propagates -> hard stop.
            engine = self._bring_up_fn(system, spec)
            self._active_engine = engine
            bringup_done_t = self._now()
            ef_reason: Optional[str] = None
            res: dict = {}
            try:
                fresh = "" if ef_retries == 0 else f" (engine-failure retry {ef_retries})"
                eta = self._eta(durations, total - done)
                self.log(f"[{done}/{total}] {spec.cell_id} (system={system}) "
                         f"calibrating frac={spec.calibration_fraction} grid={grid} "
                         f"on a fresh stack{fresh} {eta}")
                t0 = self._now()
                stack_age_s = max(0.0, t0 - bringup_done_t)
                res = self._sweep_fn(spec, engine, grid,
                                     stack_age_s=stack_age_s,
                                     engine_failure_retries=ef_retries)
                durations.append(self._now() - t0)
                health_reason = self._health_fn(engine)
                ef_reason = classify_engine_failure(res, health_reason)
                # Capture the docker-logs trail WHILE the containers still exist
                # (teardown in the finally removes them).
                if ef_reason is not None:
                    self._capture_evidence_fn(engine, spec, ef_reason, ef_retries)
            finally:
                self._teardown_fn(engine)
                self._active_engine = None

            if ef_reason is None:
                self._log_result(done, total, spec, res, durations[-1] if durations else 0.0)
                return res

            res = self._as_engine_failure(spec, res, ef_reason)
            if ef_retries >= self.engine_failure_retries:
                self.log(f"[{done}/{total}] {spec.cell_id}: ENGINE FAILURE again "
                         f"({ef_reason}) after {ef_retries} retr"
                         f"{'y' if ef_retries == 1 else 'ies'}; recording "
                         f"engine_failure and moving on")
                return res
            ef_retries += 1
            self.log(f"[{done}/{total}] {spec.cell_id}: ENGINE FAILURE ({ef_reason}); "
                     f"tore down, bringing up a FRESH stack to retry (retry {ef_retries})")

    def _as_engine_failure(self, spec, res: dict, reason: str) -> dict:
        """Turn a sweep result into an engine_failure verdict (distinct from the
        measurement verdicts) and stamp the same classification onto the on-disk
        calibration JSON so an audit of the calibration files sees engine_failure,
        not a fabricated no_stable_point. Drops any suggested_grid: a fresh stack,
        not a wider/narrower grid, is the fix."""
        out = dict(res)
        raw_status = out.get("status")
        out["status"] = "engine_failure"
        out["engine_failure_reason"] = reason
        out["raw_status"] = raw_status
        out.pop("suggested_grid", None)
        self._stamp_engine_failure_file(spec, reason, raw_status)
        return out

    def _stamp_engine_failure_file(self, spec, reason: str, raw_status) -> None:
        """Best-effort: rewrite the published calibration JSON's status to
        engine_failure (keeping the measurement verdict under raw_status). The
        campaign pre-flight already refuses any non-ok status, so this is for data
        clarity, not safety -- a future analyst grepping the calibration files for
        `status` must see the truth of the episode."""
        p = Path(spec.calibration_file)
        try:
            calib = json.loads(p.read_text()) if p.exists() else {"cell_id": spec.cell_id}
            calib["status"] = "engine_failure"
            calib["raw_status"] = raw_status
            orch = calib.setdefault("orchestration", {})
            orch["engine_failure_reason"] = reason
            tmp = p.with_name(p.name + ".engine_failure.tmp")
            tmp.write_text(json.dumps(calib, indent=2))
            os.replace(str(tmp), str(p))
        except OSError as e:
            self.log(f"{spec.cell_id}: could not stamp engine_failure onto "
                     f"{spec.calibration_file}: {e!r}")

    def _log_result(self, done: int, total: int, spec, res: dict, dur_s: float) -> None:
        extra = ""
        if res.get("status") != "ok" and res.get("suggested_grid"):
            extra = f"  -> try grid {res['suggested_grid']}"
        self.log(f"[{done}/{total}] {spec.cell_id}: status={res.get('status')} "
                 f"ceiling={res.get('ceiling_rps')} "
                 f"rate={res.get('rate_calibrated_rps')} "
                 f"stack_age={res.get('stack_age_s')}s "
                 f"ef_retries={res.get('engine_failure_retries')} "
                 f"({dur_s / 60:.1f} min){extra}")

    @staticmethod
    def _eta(durations: list[float], remaining: int) -> str:
        if not durations or remaining <= 0:
            return ""
        mean = sum(durations) / len(durations)
        return f"[ETA ~{mean * remaining / 3600:.1f}h, {remaining} left]"

    def _finish(self, results: dict[str, dict]) -> int:
        non_ok = {cid: r for cid, r in results.items() if r.get("status") != "ok"}
        counts: dict[str, int] = {}
        for r in results.values():
            counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
        # engine_failure is broken out from the measurement verdicts (did_not_
        # saturate / no_stable_point): a dead/sick endpoint is a different problem
        # (fix the stack) from a grid that did not bracket the ceiling (fix the
        # grid), and conflating them is exactly what hid the 2026-07-06 stall.
        known = ("ok", "did_not_saturate", "no_stable_point", "engine_failure")
        other = sum(n for s, n in counts.items() if s not in known)
        self.log("=" * 72)
        self.log(f"SUMMARY: {len(results)} cells -- "
                 f"{counts.get('ok', 0)} ok, "
                 f"{counts.get('did_not_saturate', 0)} did_not_saturate, "
                 f"{counts.get('no_stable_point', 0)} no_stable_point, "
                 f"{counts.get('engine_failure', 0)} engine_failure"
                 + (f", {other} other" if other else ""))
        header = f"{'cell':<30} {'status':<16} {'ceiling':>9} {'rate':>9}  note (next grid / failure reason)"
        self.log(header)
        for spec in self.specs():
            r = results.get(spec.cell_id)
            if r is None:
                continue
            if r.get("status") == "engine_failure":
                note = f"ENGINE FAILURE: {r.get('engine_failure_reason', '')}"
            else:
                sug = r.get("suggested_grid")
                note = "" if not sug else ",".join(str(x) for x in sug)
            skip = " (skipped)" if r.get("skipped") else ""
            self.log(f"{spec.cell_id:<30} {str(r.get('status')) + skip:<16} "
                     f"{_fmt(r.get('ceiling_rps')):>9} {_fmt(r.get('rate_calibrated_rps')):>9}  {note}")

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
    _min_mv = campaign.get("calibration_min_method_version")
    min_method_version = int(_min_mv) if _min_mv is not None else None
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
        step_min_s=args.step_min_seconds,
        ceil_mult=args.ceil_mult,
        min_method_version=min_method_version,
        cooldown_s=args.cooldown_seconds,
        bringup_retries=args.bringup_retries,
        engine_failure_retries=args.engine_failure_retries,
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
                   help="DEPRECATED legacy fixed per-rate window; kept only as the "
                        "step-floor fallback. Prefer --step-min-seconds.")
    p.add_argument("--step-min-seconds", type=float, default=600.0,
                   help="Floor for each rate step's WALL duration, passed to "
                        "calibrate_rate (actual = max(step_min, ceil_mult x p99)).")
    p.add_argument("--ceil-mult", type=float, default=20.0,
                   help="Per-step wall duration >= ceil_mult x running p99 estimate.")
    p.add_argument("--cooldown-seconds", type=int, default=30,
                   help="Idle between sweep rates so the KV cache flushes.")
    p.add_argument("--rate-grids", type=Path, default=None,
                   help="Optional yaml mapping cell_id or system -> [rates] to override "
                        "the built-in per-system default grids (cell_id wins over system). "
                        "The env var WOSAR_CALIB_RATES=r1,r2,... overrides globally.")
    p.add_argument("--bringup-retries", type=int, default=1,
                   help="Engine bring-up attempts per sweep before a hard stop.")
    p.add_argument("--engine-failure-retries", type=int, default=1,
                   help="Fresh-stack retries for a cell that records an "
                        "engine_failure (dead/sick endpoint). Default 1: retry "
                        "once on a fresh stack, then record engine_failure and "
                        "move on. 0 disables the retry.")
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
