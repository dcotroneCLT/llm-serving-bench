"""Multi-process per-component memory monitor (for Dynamo and any
multi-process serving system).

A *component* is a GROUP of processes identified by a cmdline regex (with
optional require/exclude). The per-component series is the SUM over the group
of USS/RSS/VMS/PSS. We ALSO write group aggregates (engine, infra) so the
engine total stays comparable to a single-process system.

Design (locked):
  - USS is the authoritative aggregate axis: private, no double-count across
    processes, summable, comparable to the single-process total. RSS/PSS/VMS
    are kept as diagnostics only (shared pages double-count under a naive sum;
    PSS is the right axis for a shared-segment inspection).
  - etcd / NATS are captured as an "infra" group, kept OUT of the engine
    aggregate so engine USS stays comparable across systems.
  - Same-tick sampling: every tick we (1) resolve the FULL PID set for all
    components in ONE process_iter pass under a single timestamp, then
    (2) sample each resolved PID once. A PID that exits between resolve and
    read is counted as not-sampled for that tick WITHOUT crashing the tick.
  - Worker respawn / membership: over a 48h run a worker can crash and respawn
    under a new PID. If a component momentarily has fewer (or zero) PIDs, the
    naive sum would dip and read as a fake leak step. So every row records
    n_pids_matched / n_pids_sampled and a membership_complete flag; the engine
    aggregate sets process_alive = membership_complete so the EXISTING analysis
    process_alive filter drops those ticks instead of treating the dip as data.

Identity, NOT regex (locked after gate 2): each component carries `pgids`, the
host process-group ids recorded by the bring-up (deploy/dynamo/record_component_
pids.py). The component is EXACTLY the processes in those PGIDs; we sum all of
them (which captures vLLM EngineCore subprocess forks a single recorded PID would
miss). The cmdline regex is kept only as a sanity check: a process that matches a
component's regex but lies OUTSIDE its recorded PGIDs is a stale/duplicate worker;
it is NOT summed, and is counted into `n_pids_unexpected` so the validator can
catch it. A spec WITHOUT `pgids` falls back to regex matching (legacy / tests).

Component spec (JSON, --components-file):
  {
    "engine_group": "dynamo",        # name of the engine aggregate -> agg_dynamo_*
    "root_pid": null,                # int to restrict to a container's descendants; null = host-wide scan
    "components": [
      {"label": "dynamo_frontend", "pattern": "dynamo\\.frontend", "group": "engine", "expected_count": 1, "pgids": [763985],
       "note": "ingress + KV-router (router is in-process in the frontend, not a separate PID)"},
      {"label": "dynamo_prefill",  "pattern": "dynamo\\.vllm", "require": "--disaggregation-mode prefill", "group": "engine", "expected_count": 1, "pgids": [751535]},
      {"label": "dynamo_decode",   "pattern": "dynamo\\.vllm", "require": "--disaggregation-mode decode", "group": "engine", "expected_count": 1, "pgids": [751793]},
      {"label": "etcd",            "pattern": "(^|/)etcd($|\\s)", "group": "infra", "expected_count": 1, "pgids": [666100]},
      {"label": "nats",            "pattern": "nats-server", "group": "infra", "expected_count": 1, "pgids": [666239]}
    ]
  }

`expected_count` is the number of recorded PGIDs (logical instances). `pgids`
is injected by attach_run/launch_cell from the bring-up identity file.

Runs under sudo (like proc_monitor) so USS/PSS are readable for processes not
owned by the launching user.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except ImportError as e:
    raise SystemExit("psutil not installed. Run: pip install psutil") from e

from _common import CsvRotatingWriter, ShutdownEvent, Watchdog, WriterConfig
from find_engine_pid import cmdline_matches


COMPONENT_FIELDS = [
    "ts_unix", "label", "group",
    "n_pids_matched", "n_pids_sampled", "n_pgids_alive", "n_pids_unexpected",
    "membership_complete",
    "uss_bytes", "rss_bytes", "vms_bytes", "pss_bytes",
    "pids", "_sample_duration_s", "_wall_clock_unix",
]

AGG_FIELDS = [
    "ts_unix", "group", "process_alive", "membership_complete",
    "n_components_expected", "n_components_complete", "n_pids_sampled",
    "n_pids_unexpected",
    "uss_bytes", "rss_bytes", "vms_bytes", "pss_bytes",
    "_sample_duration_s", "_wall_clock_unix",
]


class Component:
    def __init__(self, spec: dict) -> None:
        self.label = spec["label"]
        self.group = spec.get("group", "engine")
        self.pattern = re.compile(spec["pattern"])
        self.require = re.compile(spec["require"]) if spec.get("require") else None
        self.exclude = re.compile(spec["exclude"]) if spec.get("exclude") else None
        self.expected_count = spec.get("expected_count")  # may be None
        # Recorded process-group identity. When present it is the AUTHORITATIVE
        # scope; the regex is then only a sanity check (n_pids_unexpected).
        self.pgids = set(int(g) for g in spec["pgids"]) if spec.get("pgids") else None
        self.note = spec.get("note", "")

    def matches(self, cmdline: str) -> bool:
        return cmdline_matches(cmdline, self.pattern, self.require, self.exclude)


def _pgid_of(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def resolve_all(
    components: list[Component], root_pid: Optional[int]
) -> tuple[dict[str, list[psutil.Process]], dict[str, int]]:
    """ONE process_iter pass; bucket each live process into its component.

    PGID-scoped components (the locked Dynamo path): a process belongs to the
    component iff its PGID is in the component's recorded `pgids`. A process that
    matches the component's cmdline regex but is OUTSIDE those PGIDs is a stale /
    duplicate worker: it is NOT bucketed (never summed) and is counted into
    `unexpected[label]`.

    Regex-only components (legacy / tests, no `pgids`): bucket by cmdline match,
    as before. The require/exclude split keeps prefill and decode disjoint.
    """
    buckets: dict[str, list[psutil.Process]] = {c.label: [] for c in components}
    unexpected: dict[str, int] = {c.label: 0 for c in components if c.pgids is not None}
    pgid_components = [c for c in components if c.pgids is not None]
    regex_components = [c for c in components if c.pgids is None]

    if root_pid is not None:
        try:
            candidates = psutil.Process(root_pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            candidates = []
    else:
        candidates = list(psutil.process_iter())

    for proc in candidates:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        pgid = _pgid_of(proc.pid)
        for c in pgid_components:
            if pgid is not None and pgid in c.pgids:
                buckets[c.label].append(proc)            # in the recorded group: counted
            elif cmdline and c.matches(cmdline):
                unexpected[c.label] += 1                 # regex match outside the group: stray, NOT counted
        if cmdline:
            for c in regex_components:
                if c.matches(cmdline):
                    buckets[c.label].append(proc)
    return buckets, unexpected


def sample_component(procs: list[psutil.Process]) -> dict[str, Any]:
    """Sum USS/RSS/VMS/PSS over the resolved PIDs, tolerating per-PID exits."""
    uss = rss = vms = pss = 0
    sampled_pids: list[int] = []
    for proc in procs:
        try:
            mem = proc.memory_full_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Exited between resolve and read, or unreadable: do not crash the
            # tick; this PID is simply not counted (membership becomes < matched).
            continue
        uss += getattr(mem, "uss", 0) or 0
        rss += getattr(mem, "rss", 0) or 0
        vms += getattr(mem, "vms", 0) or 0
        pss += getattr(mem, "pss", 0) or 0
        sampled_pids.append(proc.pid)
    return {
        "uss_bytes": uss, "rss_bytes": rss, "vms_bytes": vms, "pss_bytes": pss,
        "sampled_pids": sampled_pids,
    }


def tick(components: list[Component], root_pid: Optional[int], ts: float) -> tuple[list[dict], dict[str, dict]]:
    """One coherent measurement: resolve-all then sample-all under one ts."""
    buckets, unexpected = resolve_all(components, root_pid)
    comp_rows: list[dict] = []
    by_group: dict[str, dict] = {}
    for c in components:
        procs = buckets[c.label]
        n_matched = len(procs)
        s = sample_component(procs)
        n_sampled = len(s["sampled_pids"])
        n_unexpected = unexpected.get(c.label, 0)

        # ---- membership_complete: the data-integrity decision ----
        # Every resolved process must have been sampled (no read failed mid-tick).
        all_sampled = (n_sampled == n_matched) and (n_matched > 0)
        if c.pgids is not None:
            # Identity scope: count how many of the RECORDED pgids are alive this
            # tick, and require EXACT membership (==, never >=). A dead instance
            # (under) or an extra recorded pgid (over) marks the tick incomplete;
            # a stray duplicate (n_unexpected>0) is recorded but, being outside
            # the recorded pgids, was never summed in the first place.
            alive_pgids = {pg for p in procs if (pg := _pgid_of(p.pid)) is not None} & c.pgids
            n_pgids_alive = len(alive_pgids)
            expected = int(c.expected_count) if c.expected_count is not None else len(c.pgids)
            complete = all_sampled and (n_pgids_alive == expected)
        else:
            # Legacy regex path: exact count when expected_count is given.
            n_pgids_alive = None
            complete = all_sampled
            if c.expected_count is not None:
                complete = complete and (n_matched == int(c.expected_count))

        row = {
            "ts_unix": ts, "label": c.label, "group": c.group,
            "n_pids_matched": n_matched, "n_pids_sampled": n_sampled,
            "n_pgids_alive": n_pgids_alive, "n_pids_unexpected": n_unexpected,
            "membership_complete": complete,
            "uss_bytes": s["uss_bytes"] if n_sampled > 0 else None,
            "rss_bytes": s["rss_bytes"] if n_sampled > 0 else None,
            "vms_bytes": s["vms_bytes"] if n_sampled > 0 else None,
            "pss_bytes": s["pss_bytes"] if n_sampled > 0 else None,
            "pids": ",".join(str(p) for p in s["sampled_pids"]),
        }
        comp_rows.append(row)
        g = by_group.setdefault(c.group, {
            "n_components_expected": 0, "n_components_complete": 0, "n_pids_sampled": 0,
            "n_pids_unexpected": 0,
            "uss_bytes": 0, "rss_bytes": 0, "vms_bytes": 0, "pss_bytes": 0,
        })
        g["n_components_expected"] += 1
        g["n_pids_sampled"] += n_sampled
        g["n_pids_unexpected"] += n_unexpected
        if complete:
            g["n_components_complete"] += 1
            g["uss_bytes"] += s["uss_bytes"]
            g["rss_bytes"] += s["rss_bytes"]
            g["vms_bytes"] += s["vms_bytes"]
            g["pss_bytes"] += s["pss_bytes"]
    return comp_rows, by_group


def main() -> None:
    p = argparse.ArgumentParser(description="Per-component multi-process memory monitor.")
    p.add_argument("--components-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--period-seconds", type=float, default=5.0)
    p.add_argument("--rotation-seconds", type=int, default=60)
    p.add_argument("--watchdog-timeout-s", type=float, default=30.0)
    args = p.parse_args()

    spec = json.loads(args.components_file.read_text())
    components = [Component(c) for c in spec["components"]]
    root_pid = spec.get("root_pid")
    engine_group = spec.get("engine_group", "engine")

    # One writer per component + one per group aggregate (agg_<group>).
    comp_writers = {
        c.label: CsvRotatingWriter(WriterConfig(
            output_dir=args.output_dir, base_name=c.label,
            rotation_seconds=args.rotation_seconds, fieldnames=COMPONENT_FIELDS))
        for c in components
    }
    groups = sorted({c.group for c in components})
    agg_writers = {
        g: CsvRotatingWriter(WriterConfig(
            output_dir=args.output_dir,
            base_name=(f"agg_{engine_group}" if g == "engine" else f"agg_{g}"),
            rotation_seconds=args.rotation_seconds, fieldnames=AGG_FIELDS))
        for g in groups
    }

    watchdog = Watchdog(timeout_seconds=args.watchdog_timeout_s, sentinel=None)
    shutdown = ShutdownEvent()

    print(f"[multiproc_monitor] components={[c.label for c in components]} "
          f"root_pid={root_pid} engine_group={engine_group}", flush=True)

    start = time.monotonic()
    n = 0
    try:
        while not shutdown.is_set():
            deadline = start + n * args.period_seconds
            now = time.monotonic()
            if now < deadline:
                if shutdown.wait(deadline - now):
                    break
            ts = time.time()
            t0 = time.monotonic()
            result = watchdog.call(lambda: tick(components, root_pid, ts))
            dur = time.monotonic() - t0
            wall = time.time()
            if result is None:
                # Watchdog timeout: emit incomplete-membership rows so the gap is
                # visible and excluded, never silently collapsed to zero.
                for c in components:
                    comp_writers[c.label].write({
                        "ts_unix": ts, "label": c.label, "group": c.group,
                        "n_pids_matched": None, "n_pids_sampled": 0,
                        "n_pgids_alive": None, "n_pids_unexpected": None,
                        "membership_complete": False, "uss_bytes": None,
                        "rss_bytes": None, "vms_bytes": None, "pss_bytes": None,
                        "pids": "", "_sample_duration_s": dur, "_wall_clock_unix": wall,
                    })
                for g, w in agg_writers.items():
                    w.write({
                        "ts_unix": ts, "group": g, "process_alive": False,
                        "membership_complete": False, "n_components_expected": None,
                        "n_components_complete": 0, "n_pids_sampled": 0,
                        "n_pids_unexpected": None,
                        "uss_bytes": None, "rss_bytes": None, "vms_bytes": None,
                        "pss_bytes": None, "_sample_duration_s": dur, "_wall_clock_unix": wall,
                    })
                n += 1
                continue
            comp_rows, by_group = result
            for row in comp_rows:
                row["_sample_duration_s"] = dur
                row["_wall_clock_unix"] = wall
                comp_writers[row["label"]].write(row)
            for g, agg in by_group.items():
                complete = agg["n_components_complete"] == agg["n_components_expected"]
                agg_writers[g].write({
                    "ts_unix": ts, "group": g,
                    "process_alive": complete,            # so existing analysis drops incomplete ticks
                    "membership_complete": complete,
                    "n_components_expected": agg["n_components_expected"],
                    "n_components_complete": agg["n_components_complete"],
                    "n_pids_sampled": agg["n_pids_sampled"],
                    "n_pids_unexpected": agg["n_pids_unexpected"],
                    "uss_bytes": agg["uss_bytes"], "rss_bytes": agg["rss_bytes"],
                    "vms_bytes": agg["vms_bytes"], "pss_bytes": agg["pss_bytes"],
                    "_sample_duration_s": dur, "_wall_clock_unix": wall,
                })
            n += 1
    finally:
        for w in comp_writers.values():
            w.close()
        for w in agg_writers.values():
            w.close()


if __name__ == "__main__":
    main()
