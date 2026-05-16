#!/usr/bin/env python3
"""Aging trend analysis for a single engine run.

Loads gpu, proc, system monitor CSVs and the client requests CSVs from
a run directory, computes per-indicator trend statistics, and prints a
table with: indicator, mean, Sen slope (per hour) and 95% CI, MK p-value
(Hamed-Rao corrected), BH-adjusted q-value, and final decision.

Usage:
    python3 analysis/aging_trends.py <run_dir> [--alpha 0.10]
                                     [--downsample-seconds 60]

Pipeline:
  1. Load and concat all rotating CSVs from each monitor and the client.
  2. Downsample to one sample per `downsample-seconds` (default 60 s) by
     median-aggregating each indicator within a window. Latency
     percentiles are computed from raw client samples per window.
  3. For each numeric indicator: run pymannkendall.hamed_rao_modification_test
     and a Sen's slope point estimate, plus 95% CI based on the order
     statistics of pairwise slopes (Sen 1968; Hollander & Wolfe 1999),
     with the Mann-Kendall null variance multiplied by the AR(1)
     variance inflation factor (1+rho)/(1-rho). This is the leading
     Hamed-Rao correction term and gives well-calibrated CIs on
     autocorrelated time series.
  4. Apply Benjamini-Hochberg correction across all indicators with FDR
     target = `alpha` (default 0.10).
  5. Mark a trend as significant when both MK q-value < alpha AND the Sen
     CI excludes zero.

Proc monitor catalog:
  Il catalogo include solo metriche istantanee o derivate (rate). I
  counter cumulativi sono esclusi perche' crescono linearmente per
  costruzione.

Why not block bootstrap. The circular block bootstrap rearranges blocks
of the series, which destroys any monotonic trend present in the
data. Its sampling distribution is the null distribution of the slope
under stationarity, which is the wrong target for a CI on the
parameter. The Hollander-Wolfe / Sen exact CI based on the order
statistics of pairwise slopes does not have this problem and admits
a clean autocorrelation correction via the Mann-Kendall variance.

Dependencies: pandas, numpy, scipy, pymannkendall.
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
from scipy import stats

try:
    import pymannkendall as mk
except ImportError:
    print("pymannkendall not installed. Run: pip install pymannkendall", file=sys.stderr)
    sys.exit(2)


# ---------- data loading ----------

def load_csvs(run_dir: Path, prefix: str) -> Optional[pd.DataFrame]:
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
    return pd.concat(parts, ignore_index=True)


def load_client(run_dir: Path) -> Optional[pd.DataFrame]:
    client_dir = run_dir / "client"
    if not client_dir.is_dir():
        return None
    files = sorted(glob.glob(str(client_dir / "requests_*.csv")))
    if not files:
        return None
    parts = [pd.read_csv(f) for f in files]
    return pd.concat(parts, ignore_index=True)


# ---------- core stats ----------

def estimate_lag1_autocorr(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return 0.0
    yc = y - np.mean(y)
    num = float(np.sum(yc[:-1] * yc[1:]))
    den = float(np.sum(yc ** 2))
    if den <= 0:
        return 0.0
    rho = num / den
    return max(0.0, min(0.99, rho))


def ar1_variance_inflation(rho: float) -> float:
    rho = max(0.0, min(0.99, rho))
    return (1.0 + rho) / (1.0 - rho)


def sen_slope_and_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05, ar_correction: float = 1.0
) -> tuple[float, float, float]:
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xi = x[:, None]
    yi = y[:, None]
    dx = x[None, :] - xi
    dy = y[None, :] - yi
    mask = np.triu(np.ones((n, n), dtype=bool), k=1) & (dx != 0)
    slopes = dy[mask] / dx[mask]
    if slopes.size == 0:
        return float("nan"), float("nan"), float("nan")
    slopes_sorted = np.sort(slopes)
    M = len(slopes_sorted)
    median_slope = float(np.median(slopes))

    var_S = n * (n - 1) * (2 * n + 5) / 18.0 * float(ar_correction)
    z = stats.norm.ppf(1 - alpha / 2)
    C_alpha = z * np.sqrt(var_S)

    L = int(np.floor((M - C_alpha) / 2))
    U = int(np.ceil((M + C_alpha) / 2)) + 1
    L = max(0, L)
    U = min(M - 1, U)
    return median_slope, float(slopes_sorted[L]), float(slopes_sorted[U])


def trend_one_indicator(series: pd.Series, dt_hours: float) -> dict:
    y = series.dropna().to_numpy()
    n = len(y)
    if n < 10:
        return {
            "n": n, "mean": float("nan"), "std": float("nan"), "rho": float("nan"),
            "sen_slope_per_hour": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "mk_z": float("nan"), "mk_p": float("nan"), "mk_trend": "insufficient_data",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mk_res = mk.hamed_rao_modification_test(y)
            mk_z = float(mk_res.z); mk_p = float(mk_res.p); mk_trend = mk_res.trend
        except Exception as e:
            mk_z, mk_p, mk_trend = float("nan"), float("nan"), f"error:{type(e).__name__}"

    rho = estimate_lag1_autocorr(y)
    ar_mult = ar1_variance_inflation(rho)
    x = np.arange(n, dtype=float)
    sen_per_sample, ci_lo_per_sample, ci_hi_per_sample = sen_slope_and_ci(
        x, y, alpha=0.05, ar_correction=ar_mult
    )

    return {
        "n": n, "mean": float(np.mean(y)), "std": float(np.std(y)), "rho": rho,
        "sen_slope_per_hour": sen_per_sample / dt_hours,
        "ci_low": ci_lo_per_sample / dt_hours,
        "ci_high": ci_hi_per_sample / dt_hours,
        "mk_z": mk_z, "mk_p": mk_p, "mk_trend": mk_trend,
    }


def downsample_to_minutes(df: pd.DataFrame, ts_col: str, window_seconds: int = 60) -> pd.DataFrame:
    df = df.dropna(subset=[ts_col]).copy()
    df["_bin"] = (df[ts_col] // window_seconds).astype(np.int64)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in {ts_col, "_bin"}]
    return df.groupby("_bin")[numeric].median().reset_index(drop=True)


def downsample_client(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    df = df.dropna(subset=["submitted_at_unix"]).copy()
    df["_bin"] = (df["submitted_at_unix"] // window_seconds).astype(np.int64)
    rows = []
    for bin_id, group in df.groupby("_bin"):
        n = len(group)
        ok = group[group["status"] == "ok"]
        dropped = group[group["status"] == "dropped"]
        streaming = ok[ok.get("streaming", False) == True]
        rows.append({
            "n_requests": n,
            "drop_rate": (len(dropped) / n) if n > 0 else 0.0,
            "e2e_p50": ok["e2e_latency_s"].quantile(0.5) if len(ok) > 0 else np.nan,
            "e2e_p95": ok["e2e_latency_s"].quantile(0.95) if len(ok) > 0 else np.nan,
            "e2e_p99": ok["e2e_latency_s"].quantile(0.99) if len(ok) > 0 else np.nan,
            "ttft_p50": streaming["ttft_s"].quantile(0.5) if len(streaming) > 0 else np.nan,
            "ttft_p99": streaming["ttft_s"].quantile(0.99) if len(streaming) > 0 else np.nan,
            "tokens_per_sec": (
                ok["actual_output_tokens"].sum() / window_seconds
                if "actual_output_tokens" in ok.columns and len(ok) > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def bh_fdr(pvalues: list[float], alpha: float = 0.10) -> list[float]:
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
    adj = np.minimum.accumulate(raw_q[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    q_valid = np.empty(n)
    q_valid[order] = adj
    q[valid] = q_valid
    return q.tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--downsample-seconds", type=int, default=60)
    args = p.parse_args()

    if not args.run_dir.is_dir():
        print(f"Not a directory: {args.run_dir}", file=sys.stderr); sys.exit(1)

    print(f"Loading run from {args.run_dir} ...")

    gpu = load_csvs(args.run_dir, "gpu0")
    if gpu is None:
        print("  [warn] no gpu0 CSVs found", file=sys.stderr); gpu_ds = None
    else:
        gpu_ds = downsample_to_minutes(gpu, "ts_unix", args.downsample_seconds)
        print(f"  gpu: {len(gpu)} raw -> {len(gpu_ds)} per-window samples")

    proc_prefix = None
    for f in args.run_dir.glob("*_000000.csv"):
        name = f.stem.rsplit("_", 1)[0]
        if name not in ("gpu0", "system"):
            proc_prefix = name; break
    if proc_prefix:
        proc = load_csvs(args.run_dir, proc_prefix)
        if proc is not None:
            if "process_alive" in proc.columns:
                proc = proc[proc["process_alive"] == True]
            proc_ds = downsample_to_minutes(proc, "ts_unix", args.downsample_seconds)
            print(f"  proc ({proc_prefix}): {len(proc)} raw -> {len(proc_ds)} per-window samples")
        else:
            proc_ds = None
    else:
        print("  [warn] no proc monitor CSV found"); proc_ds = None

    system = load_csvs(args.run_dir, "system")
    if system is None:
        print("  [warn] no system CSVs found", file=sys.stderr); system_ds = None
    else:
        system_ds = downsample_to_minutes(system, "ts_unix", args.downsample_seconds)
        print(f"  system: {len(system)} raw -> {len(system_ds)} per-window samples")

    client = load_client(args.run_dir)
    if client is None:
        print("  [warn] no client requests CSVs found", file=sys.stderr); client_ds = None
    else:
        client_ds = downsample_client(client, args.downsample_seconds)
        print(f"  client: {len(client)} raw -> {len(client_ds)} per-window samples")

    catalog = []
    if gpu_ds is not None:
        for col in ["vram_used_bytes", "gpu_util_percent", "mem_util_percent",
                    "temperature_c", "power_draw_w", "sm_clock_mhz", "mem_clock_mhz",
                    "ecc_db_volatile", "ecc_sb_volatile"]:
            if col in gpu_ds.columns:
                catalog.append(("gpu", col, gpu_ds[col]))
    if proc_ds is not None:
        for col in ["rss_bytes", "vms_bytes", "uss_bytes", "pss_bytes",
                    "num_threads", "num_fds", "cpu_percent",
                    "voluntary_ctx_switches_rate", "involuntary_ctx_switches_rate",
                    "io_read_bytes_rate", "io_write_bytes_rate",
                    "io_read_count_rate", "io_write_count_rate"]:
            if col in proc_ds.columns:
                catalog.append(("proc", col, proc_ds[col]))
    if system_ds is not None:
        for col in ["mem_used_bytes", "swap_used_bytes",
                    "load_avg_1m", "cpu_percent_total", "fd_allocated"]:
            if col in system_ds.columns:
                catalog.append(("system", col, system_ds[col]))
    if client_ds is not None:
        for col in ["drop_rate", "e2e_p50", "e2e_p95", "e2e_p99",
                    "ttft_p50", "ttft_p99", "tokens_per_sec"]:
            if col in client_ds.columns:
                catalog.append(("client", col, client_ds[col]))

    if not catalog:
        print("No indicators to analyze. Aborting.", file=sys.stderr); sys.exit(1)

    print(f"\nAnalyzing {len(catalog)} indicators ...")
    dt_hours = args.downsample_seconds / 3600.0

    rows = []
    for source, name, series in catalog:
        print(f"  {source}.{name} (n={series.notna().sum()}) ...", end=" ", flush=True)
        s = trend_one_indicator(series, dt_hours)
        s["source"] = source; s["indicator"] = name
        rows.append(s)
        print(f"slope={s['sen_slope_per_hour']:.4g}/h, rho={s['rho']:.2f}, p={s['mk_p']:.4f}")

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

    print("\n" + "=" * 140)
    print(f"AGING TREND ANALYSIS  (FDR target q={args.alpha:.2f}, "
          f"Theil-Sen CI with AR(1) Hamed-Rao correction)")
    print("=" * 140)
    header = (f"{'source':<8} {'indicator':<28} {'mean':>14} {'rho':>5} "
              f"{'slope/h':>14} {'CI_low':>14} {'CI_high':>14} "
              f"{'p_raw':>9} {'q_BH':>9} {'sig':>5}")
    print(header); print("-" * 140)
    for r in sorted(rows, key=lambda x: (x["source"], x["indicator"])):
        sig = "YES" if r["significant"] else "no"
        print(f"{r['source']:<8} {r['indicator']:<28} "
              f"{r['mean']:>14.4g} {r['rho']:>5.2f} {r['sen_slope_per_hour']:>14.4g} "
              f"{r['ci_low']:>14.4g} {r['ci_high']:>14.4g} "
              f"{r['mk_p']:>9.4f} {r['q_value']:>9.4f} {sig:>5}")
    print("=" * 140)

    sig_count = sum(1 for r in rows if r["significant"])
    print(f"\nSignificant trends (MK q<{args.alpha:.2f} AND Sen CI excludes 0): "
          f"{sig_count} / {len(rows)} indicators")
    if sig_count > 0:
        print("\nSignificant indicators:")
        for r in sorted(rows, key=lambda x: x["q_value"] if not np.isnan(x["q_value"]) else 1.0):
            if r["significant"]:
                slope = r["sen_slope_per_hour"]
                print(f"  {r['source']}.{r['indicator']:<28}  slope={slope:>14.4g}/h  "
                      f"CI=[{r['ci_low']:.4g}, {r['ci_high']:.4g}]  q={r['q_value']:.4f}")


if __name__ == "__main__":
    main()
