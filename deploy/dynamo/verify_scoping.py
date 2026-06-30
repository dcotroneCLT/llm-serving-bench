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

    print(f"\nA) recorded-pgid USS sum : {a_uss/1e6:.1f} MB over {len(in_pgid)} PIDs")
    esc_uss = sum(e[2] for e in escapees)
    print(f"   escaping dynamo-related PIDs (descendant or cmdline match, NOT in a recorded pgid): "
          f"{len(escapees)}  USS={esc_uss/1e6:.1f} MB")
    for pid, pgid, u, cmd in sorted(escapees, key=lambda e: -e[2]):
        print(f"     pid={pid} pgid={pgid} uss={u/1e6:.1f}MB  {cmd}")

    if not escapees:
        print("\nVERDICT: COMPLETE - every dynamo-related memory-holding process is inside a recorded pgid.")
        return
    print("\nVERDICT: INCOMPLETE - the above process(es) hold memory the monitor does NOT sum. "
          "Investigate (orphan -> reap; genuine component child in its own pgid -> widen the recorded identity).")
    sys.exit(2)


if __name__ == "__main__":
    main()
