"""
Replicate paper n=1 RSS slope from local CSVs.
Compare to paper Table IV.
Downsamples to ~2000 samples per run to keep Theil-Sen O(n^2) tractable.
"""
# Frozen replication script: BASE is hardcoded to the original sandbox path used
# at submission time; do not refactor to use aging_io.py.

import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import theilslopes

BASE = "/sessions/kind-admiring-ride/mnt/llm-serving-bench/logs"

RUNS = [
    ("E1  vLLM V1 standalone",   "aging_pilot_24h_vllm_v1",                  "vllm_standalone",  "+9.15 MB/h"),
    ("E2  Triton+vLLM V0",       "aging_pilot_24h_triton_v1",                "triton_vllm",      "+2.04 MB/h"),
    ("E3  PyTorch+HF naive",     "aging_pilot_24h_pytorch_naive_v1",         "pytorch_naive",    "+170 KB/h"),
    ("E3b PyTorch+HF low",       "aging_pilot_24h_pytorch_naive_low_rate_v1","pytorch_naive",    "+179 KB/h"),
    ("A1  vLLM V0 standalone",   "aging_pilot_24h_vllm_v0_ablation_v2",      "vllm_v0_standalone","+530 KB/h"),
    ("A2  Triton+vLLM V1",       "aging_pilot_24h_triton_v1_ablation_v2",    "triton_vllm",      "+20 KB/h"),
]


def load_proc(run_dir, label):
    files = sorted(glob.glob(os.path.join(BASE, run_dir, f"{label}_*.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["ts_unix", "process_alive", "rss_bytes"])
            dfs.append(d)
        except Exception:
            pass
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_unix").drop_duplicates("ts_unix").reset_index(drop=True)
    return df


def slope_mb_per_h(df, warmup_s=1800, target_n=2000):
    t0 = df["ts_unix"].min()
    df = df[df["ts_unix"] >= t0 + warmup_s].copy()
    if "process_alive" in df.columns:
        # process_alive may be bool, string "True", or numeric
        df = df[df["process_alive"].astype(str).str.lower().isin(["true", "1"])]
    df = df[df["rss_bytes"].notna()]
    if len(df) < 100:
        return None
    # Downsample evenly to keep Theil-Sen tractable
    if len(df) > target_n:
        idx = np.linspace(0, len(df) - 1, target_n).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    t = (df["ts_unix"].values - df["ts_unix"].values.min()).astype(float)
    y = df["rss_bytes"].values.astype(float)
    sl, intercept, lo, hi = theilslopes(y, t, 0.95)
    return (
        sl * 3600.0 / 1024**2,
        lo * 3600.0 / 1024**2,
        hi * 3600.0 / 1024**2,
        len(df),
        (df["ts_unix"].max() - df["ts_unix"].min()) / 3600.0,
    )


def fmt(sl_mb_h):
    if abs(sl_mb_h) >= 1:
        return f"{sl_mb_h:+.3f} MB/h"
    return f"{sl_mb_h*1024:+.1f} KB/h"


print(f"{'cell':<28} {'paper':<14} {'replicated':<18} {'95% CI MB/h':<28} {'n_used':<8} {'hours':<6}")
print("-" * 110)
for name, run_dir, label, expected in RUNS:
    df = load_proc(run_dir, label)
    if df is None or df.empty:
        print(f"{name:<28} {expected:<14} <no data>")
        continue
    r = slope_mb_per_h(df, warmup_s=1800)
    if r is None:
        print(f"{name:<28} {expected:<14} <insufficient post-warmup data>")
        continue
    sl, lo, hi, n, hours = r
    ci = f"[{lo:+.3f}, {hi:+.3f}]"
    print(f"{name:<28} {expected:<14} {fmt(sl):<18} {ci:<28} {n:<8} {hours:<6.2f}")
