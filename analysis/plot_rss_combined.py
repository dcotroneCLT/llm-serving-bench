#!/usr/bin/env python3
"""Generate the combined RSS figure for software-aging runs.

Panel (a) plots the 2x2 factorial RSS deltas. Panel (b) overlays RSS
and VMS for one selected lock-step run, E2 by default.

The no-argument mode reproduces the local pilot inputs. For the long
campaign:

    python3 analysis/plot_rss_combined.py \
        --campaign-yaml campaigns/wosar2026/campaign.yaml \
        --runs-root /home/dcotrone/wosar/runs \
        --replicas 1
"""

from __future__ import annotations

import argparse
import os
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

from aging_io import (
    FIGURES_DIR,
    RunSpec,
    default_specs,
    downsample_by_time,
    load_proc,
    max_hours,
    normalize_memory_frame,
    parse_csv_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-yaml", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None, help="Override campaign runs_root.")
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--cells",
        default="e1,a1,e2,a2",
        help="Comma-separated cells for panel (a), or 'all'. Default: e1,a1,e2,a2.",
    )
    parser.add_argument(
        "--replicas",
        default=None,
        help="Comma-separated replica ids to plot, e.g. '1,2,3', or 'all'.",
    )
    parser.add_argument("--warmup-hours", type=float, default=None)
    parser.add_argument("--plot-every-seconds", type=float, default=300.0)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--lockstep-cell", default="e2", help="Cell used for panel (b).")
    parser.add_argument(
        "--lockstep-replica",
        default=None,
        help="Replica used for panel (b). Defaults to the first matching run.",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--output-prefix", default="rss_combined")
    return parser.parse_args()


def choose_lockstep_spec(
    specs: list[RunSpec],
    cell_id: str,
    replica: str | None,
) -> RunSpec | None:
    cell_id = cell_id.lower()
    replica_norm = f"{int(replica):02d}" if replica and replica.isdigit() else replica
    matches = [spec for spec in specs if spec.cell_id == cell_id or spec.id.lower() == cell_id]
    if replica_norm:
        for spec in matches:
            if spec.replica == replica_norm or spec.replica == replica:
                return spec
    return matches[0] if matches else None


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

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.7),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    normalized_frames = []
    plotted = 0

    for spec in specs:
        df = load_proc(spec.run_dir, spec.proc_prefix, columns=["ts_unix", "rss_bytes", "vms_bytes"])
        if df is None:
            print(f"WARNING: no process data for {spec.id} ({spec.run_dir})")
            continue
        df_norm, _ = normalize_memory_frame(df, spec.warmup_s, memory_cols=("rss_bytes", "vms_bytes"))
        if df_norm is None or df_norm.empty:
            print(f"WARNING: {spec.id} has no samples after warmup={spec.warmup_s/3600:.2f}h")
            continue

        df_plot = downsample_by_time(df_norm, args.plot_every_seconds)
        ax_a.plot(
            df_plot["hours"],
            df_plot["rss_delta_mb"],
            label=spec.label,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=1.25 if spec.replica else 1.6,
            alpha=0.78 if spec.replica else 1.0,
        )
        normalized_frames.append(df_norm)
        plotted += 1
        print(
            f"{spec.id}: duration={df_norm['hours'].max():.2f}h, "
            f"delta_rss={df_norm['rss_delta_mb'].iloc[-1]:+.1f} MB"
        )

    if plotted == 0:
        print("No plottable runs found.", file=sys.stderr)
        sys.exit(1)

    x_max = args.duration_hours if args.duration_hours is not None else max_hours(normalized_frames)

    ax_a.set_ylabel(r"$\Delta$ RSS (MB)", fontsize=11)
    if x_max and x_max > 0:
        ax_a.set_xlim(0, x_max)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="upper left", fontsize=8.2, framealpha=0.9)
    ax_a.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax_a.set_title(
        "(a) Process-private memory growth across the factorial",
        fontsize=10.5,
        loc="left",
        pad=8,
    )

    lockstep_spec = choose_lockstep_spec(specs, args.lockstep_cell, args.lockstep_replica)
    if lockstep_spec is None:
        ax_b.text(0.5, 0.5, "No lock-step run selected", ha="center", va="center", transform=ax_b.transAxes)
        print(f"WARNING: no run matched --lockstep-cell={args.lockstep_cell}")
    else:
        e2_df = load_proc(
            lockstep_spec.run_dir,
            lockstep_spec.proc_prefix,
            columns=["ts_unix", "rss_bytes", "vms_bytes"],
        )
        e2_df, _ = normalize_memory_frame(
            e2_df,
            lockstep_spec.warmup_s,
            memory_cols=("rss_bytes", "vms_bytes"),
        )
        if e2_df is None or "vms_delta_mb" not in e2_df.columns:
            ax_b.text(0.5, 0.5, "No VMS data for lock-step panel", ha="center", va="center", transform=ax_b.transAxes)
            print(f"WARNING: no VMS data for {lockstep_spec.id}")
        else:
            e2_plot = downsample_by_time(e2_df, args.plot_every_seconds)
            ax_b.plot(
                e2_plot["hours"],
                e2_plot["rss_delta_mb"],
                color="#c0392b",
                label=r"$\Delta$ RSS (resident)",
                linewidth=1.6,
                linestyle="-",
            )
            ax_b.plot(
                e2_plot["hours"],
                e2_plot["vms_delta_mb"],
                color="#3498db",
                label=r"$\Delta$ VMS (virtual)",
                linewidth=1.6,
                linestyle="--",
                alpha=0.8,
            )

    warmup_label = "per run"
    if args.warmup_hours is not None:
        warmup_label = f"{args.warmup_hours*60:.0f} min"
    ax_b.set_xlabel(f"Time since warmup (hours), warmup = {warmup_label}", fontsize=11)
    ax_b.set_ylabel("Memory delta (MB)", fontsize=11)
    if x_max and x_max > 0:
        ax_b.set_xlim(0, x_max)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_b.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax_b.set_title(
        f"(b) {args.lockstep_cell.upper()} only: RSS and VMS lock-step check",
        fontsize=10.5,
        loc="left",
        pad=8,
    )

    plt.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = args.output_dir / f"{args.output_prefix}.pdf"
    png_out = args.output_dir / f"{args.output_prefix}.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nSaved PDF: {pdf_out}")
    print(f"Saved PNG: {png_out}")


if __name__ == "__main__":
    main()
