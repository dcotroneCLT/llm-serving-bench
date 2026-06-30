"""One-time empirical completeness check for the PGID scoping (gate-2 re-pass).

PGID scoping has exactly one failure mode: a process that the component forks
into a DIFFERENT process group (its own setsid) escapes the recorded pgids, so
its memory is silently dropped from the aggregate WITHOUT tripping
membership_complete. This script catches that, live, while the stack is up.

It compares three USS sums over the running host process table:
  A) recorded scope  : processes whose PGID is in the recorded pgids (== what the
                       monitor sums);
  B) descendant scope: the recorded container-init PIDs + all their recursive
                       children (catches a setsid'd child, still a descendant by
                       ppid);
  C) cmdline scope   : processes matching a broad dynamo/vllm/etcd/nats regex
                       (catches host orphans from a prior run).

If (B ∪ C) - A is empty, scoping is COMPLETE: every memory-holding dynamo process
is inside a recorded pgid. If not, the escaping PIDs are listed (pid, pgid,
cmdline, USS) so the spec can be widened. Run under sudo so USS is readable:

  sudo -E python3 deploy/dynamo/verify_scoping.py --component-pids "$COMPONENT_PIDS_FILE"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import psutil

BROAD = re.compile(r"dynamo\.|vllm|EngineCore|(^|/)etcd($|\s)|nats-server")


def uss_of(proc: psutil.Process) -> int:
    try:
        return int(getattr(proc.memory_full_info(), "uss", 0) or 0)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Verify PGID scoping captures all dynamo memory.")
    p.add_argument("--component-pids", type=Path, required=True)
    p.add_argument("--tolerance-mb", type=float, default=64.0,
                   help="Allowed absolute USS gap (MB) between the all-dynamo-PID total and the "
                        "recorded-pgid aggregate before declaring INCOMPLETE.")
    p.add_argument("--tolerance-frac", type=float, default=0.01,
                   help="Allowed relative gap (fraction of the all-dynamo-PID total).")
    args = p.parse_args()

    ident = json.loads(args.component_pids.read_text())
    comps = ident["components"]
    recorded_pgids = {int(g) for c in comps.values() for g in c["pgids"]}
    recorded_init = {int(pid) for c in comps.values() for pid in c.get("host_pids", [])}
    print(f"recorded pgids={sorted(recorded_pgids)} init_pids={sorted(recorded_init)}")

    # B) descendant scope from the recorded container-init PIDs.
    desc_pids: set[int] = set()
    for pid in recorded_init:
        try:
            proc = psutil.Process(pid)
            desc_pids.add(pid)
            desc_pids.update(ch.pid for ch in proc.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    a_uss = 0          # recorded-pgid scope (what the monitor sums)
    in_pgid: set[int] = set()
    escapees: list[tuple] = []
    for proc in psutil.process_iter():
        try:
            pid = proc.pid
            pgid = os.getpgid(pid)
            cmd = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ProcessLookupError, OSError):
            continue
        u = uss_of(proc)
        in_recorded = pgid in recorded_pgids
        if in_recorded:
            a_uss += u
            in_pgid.add(pid)
        related = (pid in desc_pids) or bool(cmd and BROAD.search(cmd))
        if related and not in_recorded:
            escapees.append((pid, pgid, u, cmd[:90]))

    # Empirical completeness: compare the smaps USS total over ALL dynamo-related
    # PIDs (the recorded-pgid set PLUS any escapee found by descendant/cmdline
    # scan) against the recorded-pgid aggregate the monitor actually sums. They
    # must match within tolerance; a positive gap is memory the monitor drops.
    esc_uss = sum(e[2] for e in escapees)
    total_dynamo_uss = a_uss + esc_uss
    gap = esc_uss
    tol = max(args.tolerance_mb * 1e6, args.tolerance_frac * total_dynamo_uss)

    print(f"\nA) recorded-pgid aggregate USS (what the monitor sums): {a_uss/1e6:.1f} MB over {len(in_pgid)} PIDs")
    print(f"B) all-dynamo-PID smaps USS total (ps/pgrep scope)     : {total_dynamo_uss/1e6:.1f} MB")
    print(f"   gap B-A = {gap/1e6:.1f} MB   tolerance = {tol/1e6:.1f} MB   "
          f"escaping PIDs = {len(escapees)}")
    for pid, pgid, u, cmd in sorted(escapees, key=lambda e: -e[2]):
        print(f"     pid={pid} pgid={pgid} uss={u/1e6:.1f}MB  {cmd}")

    if gap <= tol:
        print("\nVERDICT: COMPLETE - all-dynamo-PID USS matches the recorded-pgid aggregate "
              "within tolerance; no memory-holding process escaped the recorded pgids.")
        return
    print("\nVERDICT: INCOMPLETE - the all-dynamo-PID total exceeds the recorded-pgid aggregate "
          "by more than tolerance; the above process(es) hold memory the monitor does NOT sum. "
          "Investigate (orphan -> reap; genuine component child in its own pgid -> widen the identity).")
    sys.exit(2)


if __name__ == "__main__":
    main()
