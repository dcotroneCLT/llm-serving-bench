#!/usr/bin/env python3
"""Run the benchmark client at several target rates in sequence.

Useful for unattended rate sweeps: you launch this once, it cycles
through a list of rates, runs the client for each one, and prints a
short summary at the end of each level. The output of each level
goes into a separate subdirectory under --output-root.

Example:

    python sweep_runner.py \\
        --config-base /tmp/sweep_base.yaml \\
        --output-root ~/wosar/runs/pilot_vllm_sweep_v2 \\
        --rates 4,8,16 \\
        --duration-seconds 300

The script invokes run_client.py as a subprocess for each rate, so
all the existing client behavior (open-loop scheduler, drop
accounting, CSV logging) is preserved unchanged. Between rates,
it pauses --cooldown-seconds (default 30) to let the engine drain
in-flight work and stabilize.

Each rate gets its own output directory:
    <output-root>/client_<NN>rps/

A SIGINT (Ctrl-C) interrupts the current level cleanly and skips the
remaining rates. The current level's CSVs are left intact.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-base", type=Path, required=True,
                   help="YAML config to use as a template for each rate.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Parent directory for the per-rate subdirectories.")
    p.add_argument("--rates", type=str, required=True,
                   help="Comma-separated list of target RPS, e.g. 1,2,4,8,16")
    p.add_argument("--duration-seconds", type=int, default=300,
                   help="Duration of each rate level (default 300).")
    p.add_argument("--cooldown-seconds", type=int, default=30,
                   help="Pause between rate levels (default 30).")
    p.add_argument("--client-script", type=Path,
                   default=Path("/home/dcotrone/wosar/llm-serving-bench/client/run_client.py"),
                   help="Path to run_client.py.")
    p.add_argument("--summary-script", type=Path,
                   default=Path("/home/dcotrone/wosar/llm-serving-bench/analysis/sweep_summary.py"),
                   help="Path to sweep_summary.py (optional, used to print results after each level).")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python interpreter to use for the subprocess calls.")
    args = p.parse_args()

    rates = [float(x.strip()) for x in args.rates.split(",") if x.strip()]
    if not rates:
        print("No rates parsed from --rates", file=sys.stderr)
        sys.exit(2)

    args.output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load(args.config_base.read_text())

    overall_start = time.time()
    print(f"[sweep] Will run {len(rates)} rate levels: {rates}")
    print(f"[sweep] Per-level duration: {args.duration_seconds} s, cooldown: {args.cooldown_seconds} s")
    print(f"[sweep] Total estimated time: "
          f"{len(rates) * args.duration_seconds + (len(rates) - 1) * args.cooldown_seconds} s")
    print()

    interrupted = False

    def handle_sigint(_sig, _frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print("\n[sweep] SIGINT received, finishing current level then stopping.")
        else:
            print("\n[sweep] Second SIGINT, exiting now.")
            sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    for i, rate in enumerate(rates):
        if interrupted:
            print(f"[sweep] Skipping remaining rates due to interrupt.")
            break

        # rate in centesimi: 0.10 -> 010rps, 1.0 -> 100rps, 16 -> 1600rps
        level_label = f"{int(round(rate * 100)):03d}rps"
        out_dir = args.output_root / f"client_{level_label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write a per-rate config in the level directory (auditable trail)
        cfg = dict(base_cfg)
        cfg["target_rate_rps"] = rate
        cfg_path = out_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print(f"[sweep] === Level {i + 1}/{len(rates)}: {rate} RPS ===")
        print(f"[sweep] Output: {out_dir}")
        print(f"[sweep] Starting at {time.strftime('%H:%M:%S')}")

        cmd = [
            args.python, str(args.client_script),
            "--config", str(cfg_path),
            "--output-dir", str(out_dir),
            "--duration-seconds", str(args.duration_seconds),
        ]
        t0 = time.time()
        rc = subprocess.call(cmd)
        elapsed = time.time() - t0

        if rc != 0:
            print(f"[sweep] WARNING: client exited with rc={rc} after {elapsed:.0f}s. Continuing.")
        else:
            print(f"[sweep] Level done in {elapsed:.0f}s.")

        # Summary
        if args.summary_script.exists():
            print(f"[sweep] --- Summary {level_label} ---")
            subprocess.call([args.python, str(args.summary_script), str(out_dir)])
            print()

        if i < len(rates) - 1 and not interrupted:
            print(f"[sweep] Cooldown {args.cooldown_seconds}s before next level.")
            time.sleep(args.cooldown_seconds)

    overall_elapsed = time.time() - overall_start
    print(f"[sweep] Sweep finished. Total elapsed: {overall_elapsed:.0f}s "
          f"({overall_elapsed/60:.1f} minutes).")


if __name__ == "__main__":
    main()
