#!/usr/bin/env python3
"""Generate RSS time-series plots for the 2x2 aging factorial.

Defaults still reproduce the local 24h pilot figure, but the script can
also read the production campaign descriptor and plot 36h replicated
runs:

    python3 analysis/plot_rss_2x2.py \
        --campaign-yaml campaigns/wosar2026/campaign.yaml \
        --runs-root /home/dcotrone/wosar/runs \
        --replicas all
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
    default_specs,
    downsample_by_time,
    load_proc,
    max_hours,
    normalize_memory_frame,
    parse_csv_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-yaml",
        type=Path,
        default=None,
        help="Campaign descriptor. When omitted, the local 24h pilot logs are used.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Override campaign runs_root.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Explicit run directory. May be passed more than once.",
    )
    parser.add_argument(
        "--cells",
        default="e1,a1,e2,a2",
        help="Comma-separated cells to plot, or 'all'. Default: e1,a1,e2,a2.",
    )
    parser.add_argument(
        "--replicas",
        default=None,
        help="Comma-separated replica ids to plot, e.g. '1,2,3', or 'all'.",
    )
    parser.add_argument(
        "--warmup-hours",
        type=float,
        default=None,
        help="Override per-run warmup discard. Defaults to manifest/cell config or 0.5h for pilots.",
    )
    parser.add_argument(
        "--plot-every-seconds",
        type=float,
        default=300.0,
        help="Median downsample interval for plotting. Use 0 for raw samples.",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="X-axis limit. Defaults to the longest post-warmup run found.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for PDF/PNG outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="rss_2x2_factorial",
        help="Output filename prefix.",
    )
    return parser.parse_args()


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

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    normalized_frames = []
    plotted = 0

    for spec in specs:
        df = load_proc(spec.run_dir, spec.proc_prefix, columns=["ts_unix", "rss_bytes"])
        if df is None:
            print(f"WARNING: no process data for {spec.id} ({spec.run_dir})")
            continue

        df_norm, _ = normalize_memory_frame(df, spec.warmup_s, memory_cols=("rss_bytes",))
        if df_norm is None or df_norm.empty:
            print(f"WARNING: {spec.id} has no samples after warmup={spec.warmup_s/3600:.2f}h")
            continue

        df_plot = downsample_by_time(df_norm, args.plot_every_seconds)
        alpha = 0.78 if spec.replica else 1.0
        linewidth = 1.25 if spec.replica else 1.6
        ax.plot(
            df_plot["hours"],
            df_plot["rss_delta_mb"],
            label=spec.label,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

        normalized_frames.append(df_norm)
        plotted += 1
        print(
            f"{spec.id}: {len(df_norm)} samples post-warmup, "
            f"duration={df_norm['hours'].max():.2f}h, "
            f"delta_rss={df_norm['rss_delta_mb'].iloc[-1]:+.1f} MB"
        )

    if plotted == 0:
        print("No plottable runs found.", file=sys.stderr)
        sys.exit(1)

    x_max = args.duration_hours if args.duration_hours is not None else max_hours(normalized_frames)
    if x_max and x_max > 0:
        ax.set_xlim(0, x_max)

    warmup_label = "per run"
    if args.warmup_hours is not None:
        warmup_label = f"{args.warmup_hours*60:.0f} min"
    ax.set_xlabel(f"Time since warmup (hours), warmup = {warmup_label}", fontsize=11)
    ax.set_ylabel(r"$\Delta$ RSS (MB)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.6, framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

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
