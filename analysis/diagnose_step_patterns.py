#!/usr/bin/env python3
"""Diagnose step-wise RSS growth patterns in aging runs.

For each selected run, produces four hypothesis-driven panels:

  1. RSS vs VMS (heap reservation / resident-set lock-step)
  2. RSS vs smoothed CPU
  3. RSS vs offered request rate
  4. RSS vs voluntary context-switch rate

No-argument mode uses the pilot A1/E2 runs. Production campaign example:

    python3 analysis/diagnose_step_patterns.py \
        --campaign-yaml campaigns/wosar2026/campaign.yaml \
        --runs-root /home/dcotrone/wosar/runs \
        --cells a1,e2 \
        --replicas all
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

_mpl_config_dir = Path(tempfile.gettempdir()) / "llm-serving-bench-matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
_xdg_cache_dir = _mpl_config_dir / "xdg-cache"
_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache_dir))

import matplotlib.pyplot as plt
import numpy as np

from aging_io import (
    FIGURES_DIR,
    default_specs,
    load_client,
    load_proc,
    normalize_memory_frame,
    parse_csv_filter,
)


PROC_COLUMNS = [
    "ts_unix",
    "rss_bytes",
    "vms_bytes",
    "cpu_percent",
    "voluntary_ctx_switches_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-yaml", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None, help="Override campaign runs_root.")
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--cells", default="a1,e2", help="Comma-separated cells, or 'all'. Default: a1,e2.")
    parser.add_argument("--replicas", default=None, help="Comma-separated replica ids, or 'all'.")
    parser.add_argument("--warmup-hours", type=float, default=None)
    parser.add_argument(
        "--smoothing-seconds",
        type=float,
        default=300.0,
        help="Rolling smoothing window for CPU and ctx-switch rates.",
    )
    parser.add_argument(
        "--request-bin-seconds",
        type=float,
        default=300.0,
        help="Request-rate bin size.",
    )
    parser.add_argument("--plot-every-seconds", type=float, default=60.0)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def rolling_window_for_seconds(ts_unix, smoothing_seconds: float) -> int:
    diffs = np.diff(np.asarray(ts_unix, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1
    median_period = float(np.median(diffs))
    return max(1, int(round(smoothing_seconds / median_period)))


def downsample_for_plot(df, seconds: float):
    if seconds <= 0 or df.empty:
        return df
    out = df.copy()
    t0 = float(out["ts_unix"].iloc[0])
    out["_bin"] = ((out["ts_unix"] - t0) // seconds).astype("int64")
    numeric = out.select_dtypes(include="number").columns.tolist()
    numeric = [col for col in numeric if col != "_bin"]
    return out.groupby("_bin", observed=True)[numeric].median().reset_index(drop=True)


def request_rate(client, post_warmup_t0: float, bin_seconds: float):
    if client is None or client.empty or "submitted_at_unix" not in client.columns:
        return None, None
    client = client.dropna(subset=["submitted_at_unix"]).copy()
    client["seconds"] = client["submitted_at_unix"].astype(float) - post_warmup_t0
    client = client[client["seconds"] >= 0]
    if client.empty:
        return None, None
    client["_bin"] = (client["seconds"] // bin_seconds).astype("int64")
    counts = client.groupby("_bin", observed=True).size()
    hours = counts.index.to_numpy(dtype=float) * bin_seconds / 3600.0
    rps = counts.to_numpy(dtype=float) / bin_seconds
    return hours, rps


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def annotate_missing(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9)


def main() -> None:
    args = parse_args()
    cells = parse_csv_filter(args.cells)
    replicas = parse_csv_filter(args.replicas)
    warmup_s = args.warmup_hours * 3600.0 if args.warmup_hours is not None else None

    specs = default_specs(
        campaign_yaml=args.campaign_yaml,
        runs_root=args.runs_root,
        run_dirs=args.run_dir,
        cells=cells,
        replicas=replicas,
        warmup_s=warmup_s,
    )
    if not specs:
        print("No runs selected.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0

    for spec in specs:
        print(f"\n=== Processing {spec.id} ===")
        proc = load_proc(spec.run_dir, spec.proc_prefix, columns=PROC_COLUMNS)
        if proc is None:
            print(f"  no proc data for {spec.id}, skipping")
            continue

        proc, post_warmup_t0 = normalize_memory_frame(
            proc,
            spec.warmup_s,
            memory_cols=("rss_bytes", "vms_bytes"),
        )
        if proc is None or proc.empty or "rss_delta_mb" not in proc.columns:
            print(f"  no RSS data after warmup for {spec.id}, skipping")
            continue

        smooth_window = rolling_window_for_seconds(proc["ts_unix"], args.smoothing_seconds)
        if "cpu_percent" in proc.columns:
            proc["cpu_smoothed"] = proc["cpu_percent"].rolling(window=smooth_window, min_periods=1).mean()
        if "voluntary_ctx_switches_rate" in proc.columns:
            proc["vol_cs_rate_smoothed"] = (
                proc["voluntary_ctx_switches_rate"].rolling(window=smooth_window, min_periods=1).mean()
            )

        client = load_client(spec.run_dir, columns=["submitted_at_unix"])
        rps_hours, rps = request_rate(client, post_warmup_t0, args.request_bin_seconds)
        proc_plot = downsample_for_plot(proc, args.plot_every_seconds)

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Diagnostic: step-wise RSS pattern in {spec.label}", fontsize=12, fontweight="bold")

        ax = axes[0]
        ax2 = ax.twinx()
        ax.plot(proc_plot["hours"], proc_plot["rss_delta_mb"], color="#c0392b", linewidth=1.4)
        if "vms_delta_mb" in proc_plot.columns:
            ax2.plot(proc_plot["hours"], proc_plot["vms_delta_mb"], color="#3498db", linewidth=1.4, alpha=0.7)
        else:
            annotate_missing(ax2, "No VMS samples")
        ax.set_ylabel(r"$\Delta$ RSS (MB)", color="#c0392b")
        ax2.set_ylabel(r"$\Delta$ VMS (MB)", color="#3498db")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#3498db")
        ax.set_title("(1) RSS vs VMS - heap reservation / lock-step check", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax2 = ax.twinx()
        ax.plot(proc_plot["hours"], proc_plot["rss_delta_mb"], color="#c0392b", linewidth=1.4)
        if "cpu_smoothed" in proc_plot.columns:
            ax2.plot(proc_plot["hours"], proc_plot["cpu_smoothed"], color="#16a085", linewidth=1.2, alpha=0.7)
        else:
            annotate_missing(ax2, "No CPU samples")
        ax.set_ylabel(r"$\Delta$ RSS (MB)", color="#c0392b")
        ax2.set_ylabel("CPU % (smoothed)", color="#16a085")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#16a085")
        ax.set_title("(2) RSS vs CPU smoothed - GC / compute-cycle check", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax2 = ax.twinx()
        ax.plot(proc_plot["hours"], proc_plot["rss_delta_mb"], color="#c0392b", linewidth=1.4)
        if rps_hours is not None and rps is not None:
            ax2.plot(rps_hours, rps, color="#8e44ad", linewidth=1.2, alpha=0.7)
        else:
            annotate_missing(ax2, "No client request samples")
        ax.set_ylabel(r"$\Delta$ RSS (MB)", color="#c0392b")
        ax2.set_ylabel("Offered RPS", color="#8e44ad")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#8e44ad")
        ax.set_title("(3) RSS vs offered request rate - workload-burst check", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax2 = ax.twinx()
        ax.plot(proc_plot["hours"], proc_plot["rss_delta_mb"], color="#c0392b", linewidth=1.4)
        if "vol_cs_rate_smoothed" in proc_plot.columns:
            ax2.plot(proc_plot["hours"], proc_plot["vol_cs_rate_smoothed"], color="#e67e22", linewidth=1.2, alpha=0.7)
        else:
            annotate_missing(ax2, "No ctx-switch samples")
        ax.set_ylabel(r"$\Delta$ RSS (MB)", color="#c0392b")
        ax2.set_ylabel("Vol ctx-switches/s", color="#e67e22")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#e67e22")
        ax.set_title("(4) RSS vs voluntary context-switch rate - scheduler / IPC check", fontsize=10)
        ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time since warmup (hours)", fontsize=11)
        x_max = args.duration_hours if args.duration_hours is not None else float(proc["hours"].max())
        if x_max > 0:
            axes[-1].set_xlim(0, x_max)

        plt.tight_layout()

        out_file = args.output_dir / f"diagnostic_step_patterns_{safe_name(spec.id)}.png"
        plt.savefig(out_file, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        plotted += 1
        print(f"  Saved: {out_file}")

    if plotted == 0:
        print("No diagnostic figures were produced.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Diagnostic figures saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
