#!/usr/bin/env python3
"""Generate RSS time series plot for the 2x2 factorial."""
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

RUNS_BASE = "/home/dcotrone/wosar/runs"
WARMUP_HOURS = 0.5  # skip the first 30 minutes (engine warmup transient)

RUNS = [
    {
        "id": "E1",
        "dir": f"{RUNS_BASE}/aging_pilot_24h_vllm_v1",
        "proc_prefix": "vllm_standalone",
        "label": "E1: vLLM V1 standalone",
        "color": "#c0392b",
        "linestyle": "-",
    },
    {
        "id": "A1",
        "dir": f"{RUNS_BASE}/aging_pilot_24h_vllm_v0_ablation_v2",
        "proc_prefix": "vllm_v0_standalone",
        "label": "A1: vLLM V0 standalone",
        "color": "#e67e22",
        "linestyle": "--",
    },
    {
        "id": "E2",
        "dir": f"{RUNS_BASE}/aging_pilot_24h_triton_v1",
        "proc_prefix": "triton_vllm",
        "label": "E2: Triton + vLLM V0",
        "color": "#2980b9",
        "linestyle": "-.",
    },
    {
        "id": "A2",
        "dir": f"{RUNS_BASE}/aging_pilot_24h_triton_v1_ablation_v2",
        "proc_prefix": "triton_vllm_v1",
        "label": "A2: Triton + vLLM V1",
        "color": "#27ae60",
        "linestyle": ":",
    },
]

fig, ax = plt.subplots(figsize=(7, 4.2))

for info in RUNS:
    pattern = f"{info['dir']}/{info['proc_prefix']}_*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"WARNING: no files for {info['id']} with pattern {pattern}")
        continue

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if 'process_alive' in d.columns:
                d = d[d['process_alive'] == True]
            if len(d) > 0 and 'rss_bytes' in d.columns and 'ts_unix' in d.columns:
                dfs.append(d[['ts_unix', 'rss_bytes']])
        except Exception as e:
            print(f"  skipping {f}: {e}")

    if not dfs:
        print(f"WARNING: no valid data for {info['id']}")
        continue

    df = pd.concat(dfs, ignore_index=True).sort_values('ts_unix').reset_index(drop=True)

    # Normalize time to hours from start
    t0 = df['ts_unix'].iloc[0]
    df['hours'] = (df['ts_unix'] - t0) / 3600.0

    # Skip warmup window: filter out the first WARMUP_HOURS
    df = df[df['hours'] >= WARMUP_HOURS].reset_index(drop=True)

    if len(df) == 0:
        print(f"WARNING: {info['id']} has no data after warmup window")
        continue

    # Re-zero time and RSS to the start of the post-warmup window
    t0_post = df['ts_unix'].iloc[0]
    df['hours'] = (df['ts_unix'] - t0_post) / 3600.0
    rss0_post = df['rss_bytes'].iloc[0]
    df['rss_delta_mb'] = (df['rss_bytes'] - rss0_post) / (1024 * 1024)

    # Downsample for plotting: one point every 5 minutes
    # (raw sampling is 5s, so 60 raw samples = 5 min)
    df_plot = df.iloc[::60].copy()

    ax.plot(df_plot['hours'], df_plot['rss_delta_mb'],
            label=info['label'],
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=1.6)

    print(f"{info['id']}: {len(df)} samples post-warmup, "
          f"delta_rss(~{24 - WARMUP_HOURS:.1f}h) = {df_plot['rss_delta_mb'].iloc[-1]:.1f} MB")

ax.set_xlabel(f'Time since warmup (hours), warmup window = {WARMUP_HOURS*60:.0f} min',
              fontsize=11)
ax.set_ylabel(r'$\Delta$ RSS (MB)', fontsize=11)
ax.set_xlim(0, 24 - WARMUP_HOURS)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

plt.tight_layout()

out_dir = Path.home() / 'wosar' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
pdf_out = out_dir / 'rss_2x2_factorial.pdf'
png_out = out_dir / 'rss_2x2_factorial.png'

plt.savefig(pdf_out, format='pdf', bbox_inches='tight')
plt.savefig(png_out, format='png', bbox_inches='tight', dpi=150)
print(f"\nSaved PDF: {pdf_out}")
print(f"Saved PNG: {png_out}")