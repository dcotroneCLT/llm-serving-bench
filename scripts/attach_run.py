"""Attach monitors + client + manifest to an ALREADY-RUNNING endpoint.

This is the lightweight harness path for the STEP 1 validation runs: you bring
the engine up by hand (e.g. deploy/dynamo/serve_disaggregated.sh, or a vLLM
container), and attach_run.py drives the per-component (or single-process)
monitor, the load client, and writes a run manifest -- WITHOUT managing the
engine lifecycle (no docker run / readyz / teardown). The full unattended
lifecycle (bring-up/readiness/teardown, serial campaign) is the later
launch_cell/campaign.py work; attach_run is intentionally enough for the two
short validation runs that gate that refactor.

It reuses launch_cell's helpers so the monitor/client/manifest/analysis code
paths exercised here are exactly the ones the campaign will use.

Single-process system (e.g. vLLM standalone): pass --engine-pid (the host PID
of the running engine); attach_run writes engine.pid and the single proc
monitor follows it.

Multi-process system (e.g. Dynamo): the cell yaml carries monitors.components;
attach_run materializes components.json and the per-component monitor scans the
host process tree (no --engine-pid needed).

Usage:
  python3 scripts/attach_run.py --cell-yaml campaigns/extension/cells/val_dynamo_disagg.yaml \
      --base-url http://localhost:8400 --runs-root ~/wosar/runs \
      --repo-root ~/wosar/llm-serving-bench --duration-seconds 1500
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore

import launch_cell as lc


def main() -> None:
    p = argparse.ArgumentParser(description="Attach monitors+client to a running endpoint (no engine lifecycle).")
    p.add_argument("--cell-yaml", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--duration-seconds", type=int, required=True)
    p.add_argument("--replica", type=int, default=1)
    p.add_argument("--run-id", type=str, default=None, help="Defaults to ext_attach_<cell>_r<NN>.")
    p.add_argument("--base-url", type=str, default=None, help="Override the cell's client base_url.")
    p.add_argument("--engine-pid", type=int, default=None, help="Single-process systems: host PID of the running engine.")
    p.add_argument("--gpu-indices", type=str, default=None, help="Override GPU device list to sample, e.g. 0,1.")
    p.add_argument("--hf-cache-host", type=Path, default=Path(""), help="Only needed if the cell yaml references {hf_cache_host}.")
    p.add_argument("--min-free-gb", type=float, default=20.0,
                   help="SC-2 pre-run free-space gate (runs-root + docker data-root).")
    args = p.parse_args()

    cell_raw = yaml.safe_load(args.cell_yaml.read_text())
    cell = lc.render_in_obj(
        cell_raw,
        repo_root=str(args.repo_root),
        hf_cache_host=str(args.hf_cache_host),
        replica=f"{args.replica:02d}",
    )
    if args.base_url:
        cell["workload"]["client_config_overrides"]["base_url"] = args.base_url
    if args.gpu_indices:
        cell["engine"]["gpu_devices"] = [int(x) for x in args.gpu_indices.split(",") if x.strip() != ""]

    cell_id = cell["cell_id"]
    run_id = args.run_id or f"ext_attach_{cell_id}_r{args.replica:02d}"
    run_dir = args.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    lc.log(f"attach run_id={run_id} run_dir={run_dir}")

    # SC-2 #1: pre-run free-space gate (runs-root for CSVs, docker data-root for
    # the hand-started containers' fs + logs).
    lc.require_free_space([args.runs_root, lc.docker_root_dir()], args.min_free_gb, label=run_id)

    monitors = cell["monitors"]
    components = monitors.get("components")

    pidfile: Optional[Path] = None
    if not components:
        if args.engine_pid is None:
            lc.die("single-process cell: --engine-pid is required for attach (no components in monitors).")
        pidfile = run_dir / "engine.pid"
        pidfile.write_text(f"{args.engine_pid}\n")
        lc.log(f"single-process: engine.pid={args.engine_pid}")

    gpu_dev0 = lc.gpu_devices_for_cell(cell)[0]
    duration_s = int(args.duration_seconds)
    started_at_unix = time.time()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "cell_id": cell_id,
        "replica": args.replica,
        "attach": True,
        "base_url": cell["workload"]["client_config_overrides"].get("base_url"),
        "started_at": lc.utc_iso(),
        "started_at_unix": started_at_unix,
        "host": lc.host_info(gpu_dev0),
        "git_sha": lc.git_sha(args.repo_root),
        "engine": cell["engine"],
        "monitors": cell["monitors"],
        "proc_prefix": lc.proc_prefix_for_cell(cell),
        "workload": cell["workload"],
        "duration_s": duration_s,
        "warmup_discard_s": cell.get("warmup_discard_s"),
        "engine_pid": args.engine_pid,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    monitors_proc = lc.spawn_monitors(
        args.repo_root, run_dir, cell, pidfile, duration_s, log_dir, args.runs_root, run_id
    )
    lc.log(f"monitors pid={monitors_proc.pid}")

    client_config = lc.materialize_client_config(args.repo_root, run_dir, cell, args.replica)
    manifest["client_config_path"] = str(client_config)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    client_proc = lc.spawn_client(args.repo_root, run_dir, client_config, duration_s, log_dir)
    lc.log(f"client pid={client_proc.pid}")

    interrupted = False

    def handle_signal(_sig, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    deadline = started_at_unix + duration_s
    try:
        while not interrupted:
            time.sleep(5)
            if time.time() >= deadline:
                lc.log("duration elapsed, shutting down")
                break
            for name, proc in [("monitors", monitors_proc), ("client", client_proc)]:
                if proc.poll() is not None:
                    lc.log(f"WARNING: {name} exited early rc={proc.returncode}")
                    interrupted = True
                    break
    finally:
        if client_proc.poll() is None:
            lc.stop_subprocess(client_proc, "client", grace_s=120.0)
        client_summary = lc.summarize_client_csvs(run_dir / "client")
        lc.log(f"client summary: total={client_summary['total']} ok={client_summary['ok']} "
               f"statuses={client_summary['status_counts']}")
        lc.stop_subprocess(monitors_proc, "monitors", grace_s=60.0)
        ended_at_unix = time.time()
        manifest["ended_at"] = lc.utc_iso()
        manifest["ended_at_unix"] = ended_at_unix
        manifest["duration_seconds_actual"] = ended_at_unix - started_at_unix
        manifest["interrupted_early"] = interrupted
        manifest["client_summary"] = client_summary
        # Fold in realized arrival stats if the client wrote them.
        arr = run_dir / "client" / "arrival_stats.json"
        if arr.exists():
            try:
                manifest["arrival_stats"] = json.loads(arr.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        lc.log(f"done. duration={manifest['duration_seconds_actual']:.0f}s interrupted={interrupted}")

    sys.exit(0 if not interrupted else 2)


if __name__ == "__main__":
    main()
