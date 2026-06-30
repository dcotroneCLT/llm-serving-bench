"""Record the runtime process-group identity of the Dynamo components.

The per-component memory monitor must sum EXACTLY the processes that belong to
each component, not whatever the host-wide cmdline regex happens to match. A
stale or duplicate worker that matches the regex would otherwise be summed into
the engine USS aggregate with no error. So the bring-up records, per component,
the host PGID of its container's init process; the monitor then scopes to those
PGIDs (and sums every process in them, which also captures vLLM EngineCore
subprocess forks that a single recorded PID would miss).

Why PGID and not the container PID or the descendant tree: runc starts each
container's init process as a session leader, so its host PGID equals its host
PID and every process the container forks inherits that PGID. Scoping by PGID is
therefore robust to re-parenting (a child whose middle parent dies stays in the
group) in a way that a recursive-children walk is not.

Output (the "identity file"), consumed by attach_run/launch_cell which merge the
pgids into the components.json the monitor reads:

  {
    "engine_group": "dynamo",
    "recorded_at": "<iso>", "recorded_at_unix": <float>, "host": "<hostname>",
    "components": {
      "<label>": {
        "containers": ["dyn_prefill_1", ...],
        "host_pids": [<int>, ...],
        "pgids": [<int>, ...],
        "expected_count": <int>      # number of live instances recorded
      }, ...
    }
  }

Usage (from serve_disaggregated.sh, after the model is served):
  python3 deploy/dynamo/record_component_pids.py --engine-group dynamo \
      --out "$COMPONENT_PIDS_FILE" \
      --component dynamo_frontend dyn_frontend \
      --component dynamo_prefill  dyn_prefill_1 \
      --component dynamo_decode   dyn_decode_1 \
      --component etcd            dyn_etcd \
      --component nats            dyn_nats

Fails LOUDLY (non-zero exit, no file written) if any named container is not
running: a half-recorded identity is worse than none, because the monitor would
silently treat a missing component as "complete".
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def container_host_pid(container: str) -> int:
    """Host PID of the container's init process (0 if not running)."""
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"docker inspect failed for {container!r}: {out.stderr.strip()}")
    try:
        return int(out.stdout.strip())
    except ValueError:
        raise SystemExit(f"unparseable State.Pid for {container!r}: {out.stdout.strip()!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Record component->PGID identity for the monitor.")
    p.add_argument("--engine-group", default="dynamo")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--component", action="append", nargs="+", metavar=("LABEL", "CONTAINER"),
        required=True,
        help="component label followed by one or more container names; repeatable",
    )
    args = p.parse_args()

    components: dict[str, dict] = {}
    for entry in args.component:
        label, containers = entry[0], entry[1:]
        if not containers:
            raise SystemExit(f"component {label!r} has no containers")
        host_pids, pgids = [], []
        for c in containers:
            pid = container_host_pid(c)
            if pid <= 0:
                raise SystemExit(f"container {c!r} (label {label!r}) is not running (State.Pid={pid})")
            host_pids.append(pid)
            pgids.append(os.getpgid(pid))
        components[label] = {
            "containers": list(containers),
            "host_pids": host_pids,
            "pgids": pgids,
            "expected_count": len(containers),
        }

    identity = {
        "engine_group": args.engine_group,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recorded_at_unix": time.time(),
        "host": socket.gethostname(),
        "components": components,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(identity, indent=2))
    print(f"[record_component_pids] wrote {args.out}")
    for label, c in components.items():
        print(f"  {label}: containers={c['containers']} pgids={c['pgids']} expected={c['expected_count']}")


if __name__ == "__main__":
    main()
