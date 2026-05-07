#!/usr/bin/env python3
"""Aging trend analysis for a single engine run.

Loads gpu, proc, system monitor CSVs and the client requests CSVs from
a run directory, computes per-indicator trend statistics, and prints a
table with: indicator, mean, Sen slope (per hour) and 95% CI, MK p-value
(Hamed-Rao corrected), BH-adjusted q-value, and final decision.

Usage:
    python aging_trends.py <run_dir> [--alpha 0.10] [--block-minutes 5]
                                     [--bootstrap-iter 10000]
                                     [--downsample-seconds 60]

Pipeline:
  1. Load and concat all rotating CSVs from each monitor and the client.
  2. Downsample to one sample per `downsample-seconds` (default 60 s) by
     median-aggregating each indicator within a 1-minute window. Latency
     percentiles are computed from raw client samples per window.
  3. For each numeric indicator: run pymannkendall.hamed_rao_modification_test
     and a Sen's slope point estimate, plus 95% CI from circular block
     bootstrap with block length = `block-minutes` minutes.
  4. Apply Benjamini-Hochberg correction across all indicators with FDR
     target = `alpha` (default 0.10).
  5. Mark a trend as significant when both MK q-value < alpha AND the Sen
     CI excludes zero.

Dependencies: pandas, numpy, pymannkendall.
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pymannkendall as mk
except ImportError:
    print("pymannkendall not installed. Run: pip install pymannkendall", file=sys.stderr)
    sys.exit(2)


# ---------- data loading ----------

def load_csvs(run_dir: Path, prefix: str) -> Optional[pd.DataFrame]:
    """Concatenate rotating CSVs matching <prefix>_NNNNNN.csv under run_dir."""
    files = sorted(glob.glob(str(run_dir / f"{prefix}_*.csv")))
    files = [f for f in files if "requests" not in Path(f).name]
    if not files:
        return None
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if not d.empty:
                parts.append(d)
        except Exception as e:
            print(f"  [warn] skipping {f}: {e}", file=sys.stderr)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df


def load_client(run_dir: Path) -> Optional[pd.DataFrame]:
    """Concatenate all client requests_NNNNNN.csv from the client subdir."""
    client_dir = run_dir / "client"
    if not client_dir.is_dir():
        return None
    files = sorted(glob.glob(str(client_dir / "requests_*.csv")))
    if not files:
        return None
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    return df


# ---------- per-indicator trend computation ----------

def sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median of pairwise slopes (Sen estimator)."""
    n = len(x)
    if n < 2:
        return float("nan")
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def block_bootstrap_ci(
    y: np.ndarray, dt_hours: float, block_size: int, n_iter: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block bootstrap CI for Sen's slope (per hour).

    The series is resampled in contiguous circular blocks of length
    `block_size` to preserve local autocorrelation. For each resample
    we compute Sen's slope vs an evenly-spaced time index, then return
    the alpha/2 and 1-alpha/2 quantiles. Slope is converted to per-hour
    units assuming time index has step `dt_hours` between samples.
    """
    n = len(y)
    if n < 2 * block_size:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=0)
    n_blocks_needed = int(np.ceil(n / block_size))
    # Pre-build the doubled array for circular bootstrapping
    y_circ = np.concatenate([y, y[:block_size]])
    slopes = np.empty(n_iter)
    # Sub-sampling for tractability: with full pairwise Sen this is O(n^2)
    # per iteration. For long series we sub-sample to <=200 points per
    # bootstrap realization, which gives stable quantiles for our use.
    sub_n = min(n, 200)
    for it in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks_needed)
        resampled = np.concatenate([y_circ[s : s + block_size] for s in starts])[:n]
        if sub_n < n:
            idx = np.linspace(0, n - 1, sub_n, dtype=int)
            ys = resampled[idx]
            xs = idx
        else:
            ys = resampled
            xs = np.arange(n)
        slopes[it] = sen_slope(xs.astype(float), ys.astype(float))
    # Convert from "per sample" to "per hour"
    slopes_per_hour = slopes / dt_hours
    return (
        float(np.nanquantile(slopes_per_hour, alpha / 2)),
        float(np.nanquantile(slopes_per_hour, 1 - alpha / 2)),
    )


def trend_one_indicator(
    series: pd.Series, dt_hours: float, block_size: int, n_iter: int
) -> dict:
    """Compute MK Hamed-Rao + Sen point estimate + bootstrap CI for one series."""
    y = series.dropna().to_numpy()
    if len(y) < 10:
        return {
            "n": len(y),
            "mean": float(np.nan),
            "std": float(np.nan),
            "sen_slope_per_hour": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "mk_z": float("nan"),
            "mk_p": float("nan"),
            "mk_trend": "insufficient_data",
        }

    # MK Hamed-Rao
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mk_res = mk.hamed_rao_modification_test(y)
            mk_z = float(mk_res.z)
            mk_p = float(mk_res.p)
            mk_trend = mk_res.trend
        except Exception as e:
            mk_z, mk_p, mk_trend = float("nan"), float("nan"), f"error:{type(e).__name__}"

    # Sen point estimate vs uniform time index
    x = np.arange(len(y), dtype=float)
    sen = sen_slope(x, y) / dt_hours

    # Bootstrap CI
    ci_low, ci_high = block_bootstrap_ci(y, dt_hours, block_size, n_iter)

    return {
        "n": len(y),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "sen_slope_per_hour": sen,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mk_z": mk_z,
        "mk_p": mk_p,
        "mk_trend": mk_trend,
    }


# ---------- downsampling helpers ----------

def downsample_to_minutes(df: pd.DataFrame, ts_col: str, window_seconds: int = 60) -> pd.DataFrame:
    """Aggregate to per-window medians of all numeric columns."""
    df = df.dropna(subset=[ts_col]).copy()
    df["_bin"] = (df[ts_col] // window_seconds).astype(np.int64)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in {ts_col, "_bin"}]
    agg = df.groupby("_bin")[numeric].median().reset_index(drop=True)
    return agg


def downsample_client(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """Per-window client aggregates: latency percentiles, drop rate, tokens/s."""
    df = df.dropna(subset=["submitted_at_unix"]).copy()
    df["_bin"] = (df["submitted_at_unix"] // window_seconds).astype(np.int64)

    rows = []
    for bin_id, group in df.groupby("_bin"):
        n = len(group)
        ok = group[group["status"] == "ok"]
        dropped = group[group["status"] == "dropped"]
        streaming = ok[ok.get("streaming", False) == True]
        rows.append(
            {
                "n_requests": n,
                "drop_rate": (len(dropped) / n) if n > 0 else 0.0,
                "e2e_p50": ok["e2e_latency_s"].quantile(0.5) if len(ok) > 0 else np.nan,
                "e2e_p95": ok["e2e_latency_s"].quantile(0.95) if len(ok) > 0 else np.nan,
                "e2e_p99": ok["e2e_latency_s"].quantile(0.99) if len(ok) > 0 else np.nan,
                "ttft_p50": streaming["ttft_s"].quantile(0.5) if len(streaming) > 0 else np.nan,
                "ttft_p99": streaming["ttft_s"].quantile(0.99) if len(streaming) > 0 else np.nan,
                "tokens_per_sec": (
                    ok["actual_output_tokens"].sum() / window_seconds
                    if "actual_output_tokens" in ok.columns and len(ok) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------- BH FDR ----------

def bh_fdr(pvalues: list[float], alpha: float = 0.10) -> list[float]:
    """Benjamini-Hochberg adjusted q-values (same length as input)."""
    p = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(p)
    n = int(valid.sum())
    q = np.full_like(p, np.nan)
    if n == 0:
        return q.tolist()
    order = np.argsort(p[valid])
    ranked_p = p[valid][order]
    ranks = np.arange(1, n + 1)
    raw_q = ranked_p * n / ranks
    # Enforce monotonicity from the right
    adj = np.minimum.accumulate(raw_q[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    q_valid = np.empty(n)
    q_valid[order] = adj
    q[valid] = q_valid
    return q.tolist()


# ---------- main analysis ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--alpha", type=float, default=0.10, help="FDR target (default 0.10)")
    p.add_argument("--block-minutes", type=int, default=5,
                   help="Block bootstrap block length in minutes (default 5)")
    p.add_argument("--bootstrap-iter", type=int, default=10000,
                   help="Number of bootstrap iterations (default 10000)")
    p.add_argument("--downsample-seconds", type=int, default=60,
                   help="Aggregation window for monitor data (default 60)")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        print(f"Not a directory: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading run from {args.run_dir} ...")

    # GPU monitor
    gpu = load_csvs(args.run_dir, "gpu0")
    if gpu is None:
        print("  [warn] no gpu0 CSVs found", file=sys.stderr)
        gpu_ds = None
    else:
        gpu_ds = downsample_to_minutes(gpu, "ts_unix", args.downsample_seconds)
        print(f"  gpu: {len(gpu)} raw -> {len(gpu_ds)} per-minute samples")

    # Proc monitor (vllm_standalone or similar label)
    # Find the proc CSV prefix automatically
    proc_prefix = None
    for f in args.run_dir.glob("*_000000.csv"):
        name = f.stem.rsplit("_", 1)[0]
        if name not in ("gpu0", "system"):
            proc_prefix = name
            break
    if proc_prefix:
        proc = load_csvs(args.run_dir, proc_prefix)
        if proc is not None:
            # Filter to alive samples only for trend analysis
            if "process_alive" in proc.columns:
                proc = proc[proc["process_alive"] == True]
            proc_ds = downsample_to_minutes(proc, "ts_unix", args.downsample_seconds)
            print(f"  proc ({proc_prefix}): {len(proc)} raw -> {len(proc_ds)} per-minute samples")
        else:
            proc_ds = None
    else:
        print("  [warn] no proc monitor CSV found")
        proc_ds = None

    # System monitor
    system = load_csvs(args.run_dir, "system")
    if system is None:
        print("  [warn] no system CSVs found", file=sys.stderr)
        system_ds = None
    else:
        system_ds = downsample_to_minutes(system, "ts_unix", args.downsample_seconds)
        print(f"  system: {len(system)} raw -> {len(system_ds)} per-minute samples")

    # Client
    client = load_client(args.run_dir)
    if client is None:
        print("  [warn] no client requests CSVs found", file=sys.stderr)
        client_ds = None
    else:
        client_ds = downsample_client(client, args.downsample_seconds)
        print(f"  client: {len(client)} raw -> {len(client_ds)} per-minute samples")

    # Indicator catalog
    catalog = []
    if gpu_ds is not None:
        for col in [
            "vram_used_bytes", "gpu_util_percent", "mem_util_percent",
            "temperature_c", "power_draw_w", "sm_clock_mhz", "mem_clock_mhz",
            "ecc_db_volatile", "ecc_sb_volatile",
        ]:
            if col in gpu_ds.columns:
                catalog.append(("gpu", col, gpu_ds[col]))
    if proc_ds is not None:
        for col in [
            "rss_bytes", "vms_bytes", "uss_bytes", "pss_bytes",
            "num_threads", "num_fds", "cpu_percent",
            "voluntary_ctx_switches", "involuntary_ctx_switches",
            "io_read_bytes", "io_write_bytes", "io_read_count", "io_write_count",
        ]:
            if col in proc_ds.columns:
                catalog.append(("proc", col, proc_ds[col]))
    if system_ds is not None:
        for col in [
            "mem_used_bytes", "swap_used_bytes",
            "load_avg_1m", "cpu_percent_total",
            "fd_allocated",
        ]:
            if col in system_ds.columns:
                catalog.append(("system", col, system_ds[col]))
    if client_ds is not None:
        for col in [
            "drop_rate", "e2e_p50", "e2e_p95", "e2e_p99",
            "ttft_p50", "ttft_p99", "tokens_per_sec",
        ]:
            if col in client_ds.columns:
                catalog.append(("client", col, client_ds[col]))

    if not catalog:
        print("No indicators to analyze. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnalyzing {len(catalog)} indicators ...")
    dt_hours = args.downsample_seconds / 3600.0
    block_size = max(1, args.block_minutes)

    rows = []
    for source, name, series in catalog:
        print(f"  {source}.{name} (n={series.notna().sum()}) ...", end=" ", flush=True)
        stats = trend_one_indicator(series, dt_hours, block_size, args.bootstrap_iter)
        stats["source"] = source
        stats["indicator"] = name
        rows.append(stats)
        print(f"slope={stats['sen_slope_per_hour']:.4g}/h, p={stats['mk_p']:.4f}")

    # FDR correction across all p-values
    pvals = [r["mk_p"] for r in rows]
    qvals = bh_fdr(pvals, alpha=args.alpha)
    for r, q in zip(rows, qvals):
        r["q_value"] = q
        ci_excludes_zero = not (
            np.isnan(r["ci_low"]) or np.isnan(r["ci_high"])
            or (r["ci_low"] <= 0 <= r["ci_high"])
        )
        mk_significant = (not np.isnan(q)) and (q < args.alpha)
        r["significant"] = bool(mk_significant and ci_excludes_zero)

    # Print results table
    print("\n" + "=" * 120)
    print(f"AGING TREND ANALYSIS  (FDR target q={args.alpha:.2f}, "
          f"block={args.block_minutes}min, n_iter={args.bootstrap_iter})")
    print("=" * 120)
    header = (
        f"{'source':<8} {'indicator':<28} {'mean':>14} "
        f"{'slope/h':>14} {'CI_low':>14} {'CI_high':>14} "
        f"{'p_raw':>9} {'q_BH':>9} {'sig':>5}"
    )
    print(header)
    print("-" * 120)
    for r in sorted(rows, key=lambda x: (x["source"], x["indicator"])):
        sig = "YES" if r["significant"] else "no"
        print(
            f"{r['source']:<8} {r['indicator']:<28} "
            f"{r['mean']:>14.4g} {r['sen_slope_per_hour']:>14.4g} "
            f"{r['ci_low']:>14.4g} {r['ci_high']:>14.4g} "
            f"{r['mk_p']:>9.4f} {r['q_value']:>9.4f} {sig:>5}"
        )
    print("=" * 120)

    sig_count = sum(1 for r in rows if r["significant"])
    print(f"\nSignificant trends (MK q<{args.alpha:.2f} AND Sen CI excludes 0): "
          f"{sig_count} / {len(rows)} indicators")
    if sig_count > 0:
        print("\nSignificant indicators:")
        for r in sorted(rows, key=lambda x: x["q_value"] if not np.isnan(x["q_value"]) else 1.0):
            if r["significant"]:
                slope = r["sen_slope_per_hour"]
                unit_per_h = f"{slope:.3g}/h"
                print(f"  {r['source']}.{r['indicator']:<28}  slope={unit_per_h:>14}  "
                      f"CI=[{r['ci_low']:.3g}, {r['ci_high']:.3g}]  q={r['q_value']:.4f}")


if __name__ == "__main__":
    main()
