"""Campaign orchestrator for WoSAR 2026 n=3 replication.

Reads campaigns/<id>/campaign.yaml and dispatches launch_cell.py for
each (cell, replica) pair across parallel GPU slots.

Topology comes from campaign.yaml. For the WoSAR 2026 campaign:

  slot gpu0: cells [e1, a1]  -> 6 runs sequentially on GPU 0
  slot gpu1: cells [e2, a2]  -> 6 runs sequentially on GPU 1
  slot gpu2: cells [e3, e3b] -> 6 runs sequentially on GPU 2

Slots run in parallel; runs within a slot run sequentially. Calendar:
max(per-slot sequential time) = 6 runs * ~26h = ~6.5 days.

Checkpointing:

  campaign_state.json holds the status of every (cell, replica) pair:
    pending | running | completed | failed | skipped
  Updated after each launch_cell.py exits. On orchestrator restart,
  the file is consulted to skip completed runs and resume the rest.

Retry policy:

  Each (cell, replica) gets one automatic retry on failure. After
  two failures, the run is marked 'failed' and the slot moves on
  (campaign.yaml.retry_policy.on_repeated_failure controls this).

Sanity runs:

  After all slot runs complete, sanity_runs from campaign.yaml are
  dispatched on the appropriate slot (single-threaded tail).

Usage:

  python scripts/campaign.py \
      --campaign-yaml campaigns/wosar2026/campaign.yaml \
      --dry-run                  # print schedule and exit

  python scripts/campaign.py \
      --campaign-yaml campaigns/wosar2026/campaign.yaml \
      --start

  python scripts/campaign.py \
      --campaign-yaml campaigns/wosar2026/campaign.yaml \
      --resume                   # pick up from state file

Recommended deployment:

  tmux new -d -s wosar_campaign \\
      'python scripts/campaign.py --campaign-yaml ... --start \\
           > /home/dcotrone/wosar/runs/campaign.log 2>&1'

  Survives ssh disconnect. Reattach: `tmux attach -t wosar_campaign`.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml  # type: ignore


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[campaign] {utc_iso()} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Schedule building
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    cell_id: str
    cell_yaml: str          # relative to repo root
    replica: int
    slot_name: str
    gpu_device: int
    duration_s_override: Optional[int] = None
    gpu_device_override: Optional[int] = None
    sanity: bool = False

    @property
    def run_key(self) -> str:
        suffix = "_sanity" if self.sanity else ""
        return f"{self.cell_id}_r{self.replica:02d}{suffix}"


@dataclass
class RunStatus:
    status: str = "pending"            # pending | running | completed | failed
    attempts: int = 0
    last_started_at: Optional[str] = None
    last_ended_at: Optional[str] = None
    last_rc: Optional[int] = None
    log_path: Optional[str] = None


@dataclass
class State:
    campaign_id: str
    started_at: str = field(default_factory=utc_iso)
    runs: dict[str, RunStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "started_at": self.started_at,
            "runs": {k: asdict(v) for k, v in self.runs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            campaign_id=d["campaign_id"],
            started_at=d.get("started_at", utc_iso()),
            runs={k: RunStatus(**v) for k, v in d.get("runs", {}).items()},
        )


def build_schedule(campaign: dict, campaign_yaml_path: Path) -> dict[str, list[RunSpec]]:
    """Return a dict slot_name -> list of RunSpec, in execution order."""
    cells_by_id: dict[str, str] = {}
    for cell_rel in campaign["cells"]:
        cell_path = (campaign_yaml_path.parent / cell_rel).resolve()
        cell = yaml.safe_load(cell_path.read_text())
        cells_by_id[cell["cell_id"]] = str(cell_path)

    replicas = int(campaign["replicas_per_cell"])
    order = campaign.get("intra_slot_order", "round_robin")
    schedule: dict[str, list[RunSpec]] = {}

    for slot in campaign["slots"]:
        slot_name = slot["name"]
        gpu_device = int(slot["gpu_device"])
        cell_ids = list(slot["cells"])

        if order == "round_robin":
            # r1 of cell A, r1 of B, ..., r2 of A, r2 of B, ...
            slot_runs = []
            for rep in range(1, replicas + 1):
                for cid in cell_ids:
                    slot_runs.append(
                        RunSpec(
                            cell_id=cid,
                            cell_yaml=cells_by_id[cid],
                            replica=rep,
                            slot_name=slot_name,
                            gpu_device=gpu_device,
                        )
                    )
        elif order == "cell_at_a_time":
            slot_runs = []
            for cid in cell_ids:
                for rep in range(1, replicas + 1):
                    slot_runs.append(
                        RunSpec(
                            cell_id=cid,
                            cell_yaml=cells_by_id[cid],
                            replica=rep,
                            slot_name=slot_name,
                            gpu_device=gpu_device,
                        )
                    )
        else:
            raise ValueError(f"unknown intra_slot_order: {order}")

        schedule[slot_name] = slot_runs

    return schedule


def build_sanity_runs(campaign: dict, campaign_yaml_path: Path) -> list[RunSpec]:
    """Return a list of sanity RunSpec, dispatched after the main schedule."""
    cells_by_id: dict[str, str] = {}
    for cell_rel in campaign["cells"]:
        cell_path = (campaign_yaml_path.parent / cell_rel).resolve()
        cell = yaml.safe_load(cell_path.read_text())
        cells_by_id[cell["cell_id"]] = str(cell_path)

    out: list[RunSpec] = []
    for s in campaign.get("sanity_runs", []):
        cid = s["cell"]
        replica_id = s.get("replica_id", "sanity")
        gpu_override = int(s["gpu_device_override"])
        duration_override = int(s["duration_s_override"])
        out.append(
            RunSpec(
                cell_id=cid,
                cell_yaml=cells_by_id[cid],
                replica=99,  # sentinel, real id is via gpu_override + sanity flag
                slot_name=f"sanity_{cid}_gpu{gpu_override}",
                gpu_device=gpu_override,
                duration_s_override=duration_override,
                gpu_device_override=gpu_override,
                sanity=True,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Slot worker
# ---------------------------------------------------------------------------


class SlotWorker(threading.Thread):
    """Drive one GPU slot's run queue sequentially."""

    def __init__(
        self,
        slot_name: str,
        runs: list[RunSpec],
        state: State,
        state_lock: threading.Lock,
        state_path: Path,
        campaign: dict,
        repo_root: Path,
        runs_root: Path,
        hf_cache_host: Path,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"slot-{slot_name}", daemon=False)
        self.slot_name = slot_name
        self.runs = runs
        self.state = state
        self.state_lock = state_lock
        self.state_path = state_path
        self.campaign = campaign
        self.repo_root = repo_root
        self.runs_root = runs_root
        self.hf_cache_host = hf_cache_host
        self.stop_event = stop_event
        self.current_proc: Optional[subprocess.Popen] = None

    def _persist_state(self) -> None:
        with self.state_lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2))

    def _run_one(self, spec: RunSpec) -> int:
        status = self.state.runs.setdefault(spec.run_key, RunStatus())
        status.attempts += 1
        status.status = "running"
        status.last_started_at = utc_iso()
        self._persist_state()

        log_dir = self.runs_root / f"{self.campaign['campaign_id']}_{spec.cell_id}_r{spec.replica:02d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        launch_log = log_dir / "launch_cell.log"
        status.log_path = str(launch_log)

        cmd = [
            sys.executable,
            str(self.repo_root / "scripts" / "launch_cell.py"),
            "--cell-yaml", spec.cell_yaml,
            "--replica", str(spec.replica),
            "--runs-root", str(self.runs_root),
            "--repo-root", str(self.repo_root),
            "--hf-cache-host", str(self.hf_cache_host),
            "--campaign-id", self.campaign["campaign_id"],
        ]
        if spec.gpu_device_override is not None:
            cmd += ["--gpu-device-override", str(spec.gpu_device_override)]
        if spec.duration_s_override is not None:
            cmd += ["--duration-s-override", str(spec.duration_s_override)]

        log(f"[{self.slot_name}] starting {spec.run_key} attempt={status.attempts}")
        log(f"[{self.slot_name}] cmd: {' '.join(cmd)}")

        with launch_log.open("ab", buffering=0) as log_f:
            self.current_proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            try:
                rc = self.current_proc.wait()
            except KeyboardInterrupt:
                rc = -1
            finally:
                self.current_proc = None
        status.last_ended_at = utc_iso()
        status.last_rc = rc
        return rc

    def run(self) -> None:
        max_retries = int(self.campaign.get("retry_policy", {}).get("max_retries", 1))
        on_failure = self.campaign.get("retry_policy", {}).get(
            "on_repeated_failure", "log_and_continue"
        )

        for spec in self.runs:
            if self.stop_event.is_set():
                log(f"[{self.slot_name}] stop_event set, abandoning queue")
                return

            existing = self.state.runs.get(spec.run_key)
            if existing and existing.status == "completed":
                log(f"[{self.slot_name}] skipping {spec.run_key} (already completed)")
                continue

            rc = self._run_one(spec)
            attempts_used = 1
            while rc != 0 and attempts_used <= max_retries:
                if self.stop_event.is_set():
                    return
                log(f"[{self.slot_name}] {spec.run_key} failed rc={rc}, retrying ({attempts_used}/{max_retries})")
                rc = self._run_one(spec)
                attempts_used += 1

            status = self.state.runs[spec.run_key]
            if rc == 0:
                status.status = "completed"
                log(f"[{self.slot_name}] {spec.run_key} COMPLETED")
            else:
                status.status = "failed"
                log(f"[{self.slot_name}] {spec.run_key} FAILED after {attempts_used} attempt(s) rc={rc}")
                if on_failure == "log_and_halt_slot":
                    log(f"[{self.slot_name}] halting slot per retry_policy")
                    self._persist_state()
                    return
            self._persist_state()

    def interrupt(self) -> None:
        if self.current_proc is not None:
            try:
                self.current_proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestrate the n=3 replication campaign.")
    p.add_argument("--campaign-yaml", type=Path, required=True)
    p.add_argument("--start", action="store_true", help="Start fresh (delete existing state file).")
    p.add_argument("--resume", action="store_true", help="Resume from state file.")
    p.add_argument("--dry-run", action="store_true", help="Print schedule and exit.")
    args = p.parse_args()

    if args.start and args.resume:
        print("--start and --resume are mutually exclusive", file=sys.stderr)
        sys.exit(2)
    if not (args.start or args.resume or args.dry_run):
        print("must specify --start, --resume, or --dry-run", file=sys.stderr)
        sys.exit(2)

    campaign_path = args.campaign_yaml.resolve()
    campaign = yaml.safe_load(campaign_path.read_text())
    # Repo root inferred as two levels up from campaigns/<id>/campaign.yaml.
    repo_root = campaign_path.parent.parent.parent
    runs_root = Path(campaign["runs_root"])
    hf_cache_host = Path(campaign["paths"]["hf_cache_host"])

    schedule = build_schedule(campaign, campaign_path)
    sanity = build_sanity_runs(campaign, campaign_path)

    log(f"campaign_id: {campaign['campaign_id']}")
    log(f"repo_root: {repo_root}")
    log(f"runs_root: {runs_root}")
    for slot_name, runs in schedule.items():
        log(f"slot {slot_name}: {len(runs)} runs -> {[r.run_key for r in runs]}")
    if sanity:
        log(f"sanity runs: {[r.run_key + '_' + r.slot_name for r in sanity]}")

    if args.dry_run:
        return

    state_path = (campaign_path.parent / campaign["state_file"]).resolve()
    if args.start and state_path.exists():
        log(f"--start: deleting existing state at {state_path}")
        state_path.unlink()
    if args.resume and not state_path.exists():
        log(f"--resume: no state file at {state_path}, starting fresh")

    if state_path.exists():
        state = State.from_dict(json.loads(state_path.read_text()))
        log(f"loaded state, {sum(1 for s in state.runs.values() if s.status == 'completed')} runs already completed")
    else:
        state = State(campaign_id=campaign["campaign_id"])

    state_lock = threading.Lock()
    stop_event = threading.Event()

    # Build slot workers
    workers: list[SlotWorker] = []
    for slot_name, runs in schedule.items():
        w = SlotWorker(
            slot_name=slot_name,
            runs=runs,
            state=state,
            state_lock=state_lock,
            state_path=state_path,
            campaign=campaign,
            repo_root=repo_root,
            runs_root=runs_root,
            hf_cache_host=hf_cache_host,
            stop_event=stop_event,
        )
        workers.append(w)

    # Signal handling: SIGTERM/SIGINT triggers stop_event and propagates to current launch_cell.
    def handle(_sig, _frame):
        log("signal received, requesting shutdown of all slots")
        stop_event.set()
        for w in workers:
            w.interrupt()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    # Start all slot workers
    for w in workers:
        w.start()
        log(f"started slot worker {w.slot_name}")

    # Wait for all slot workers to finish
    for w in workers:
        w.join()
        log(f"slot {w.slot_name} done")

    if stop_event.is_set():
        log("orchestrator interrupted, sanity runs SKIPPED")
        sys.exit(2)

    # Dispatch sanity runs sequentially (single thread)
    for spec in sanity:
        log(f"sanity: {spec.run_key} on gpu {spec.gpu_device_override}")
        existing = state.runs.get(f"{spec.run_key}_{spec.slot_name}")
        if existing and existing.status == "completed":
            log(f"sanity {spec.run_key}_{spec.slot_name} already completed, skipping")
            continue
        # Reuse slot worker logic for the sanity run on a one-off basis.
        sanity_worker = SlotWorker(
            slot_name=spec.slot_name,
            runs=[spec],
            state=state,
            state_lock=state_lock,
            state_path=state_path,
            campaign=campaign,
            repo_root=repo_root,
            runs_root=runs_root,
            hf_cache_host=hf_cache_host,
            stop_event=stop_event,
        )
        sanity_worker.start()
        sanity_worker.join()

    log("campaign complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
