"""
Quantify the step-wise RSS growth pattern (paper Section IV.E, Figure 2(b)).
Primary metric: RSS-VMS lag-0 correlation (mmap-style allocator signature).
Secondary metric: trimmed excess kurtosis K_trim of ΔRSS post-warmup (winsorized
at 99.9 percentile, bootstrap 95% CI), with raw K kept as a companion.
Operational metric: count of ΔRSS > 1 MB events per hour.
See EXPERIMENT_STATE.md "Open questions #4" for motivation.
CLI/parsing pattern follows replicate_n1.py.
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats.mstats import winsorize

from aging_io import (
    discover_proc_prefix,
    discover_runs,
    infer_cell_id,
    load_manifest,
    load_proc,
    resolve_warmup,
)


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def display_cell_id(cell_id, fallback):
    if not cell_id:
        return fallback
    # "e1" -> "E1", "e3b" -> "E3b", "a1" -> "A1"
    return cell_id[:1].upper() + cell_id[1:]


def filter_run(df, warmup_s):
    t0 = df["ts_unix"].min()
    df = df[df["ts_unix"] >= t0 + warmup_s].copy()
    df = df[df["rss_bytes"].notna()]
    return df.reset_index(drop=True)


def bootstrap_ci(values, n_resamples, rng):
    ks = np.empty(n_resamples)
    n = len(values)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ks[i] = kurtosis(values[idx], fisher=True, bias=False)
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def analyze_run(run_dir, warmup_s, n_bootstrap, seed):
    run_path = Path(run_dir)
    basename = run_path.name
    manifest = load_manifest(run_path)
    label = discover_proc_prefix(run_path, manifest)
    if label is None:
        warn(f"{basename}: no proc_monitor CSV matching known engine labels; skipping")
        return None
    df = load_proc(run_path, label, columns=["rss_bytes", "vms_bytes"])
    if df is None or df.empty:
        warn(f"{basename}: empty or unreadable proc CSVs; skipping")
        return None
    df = df.drop_duplicates("ts_unix").reset_index(drop=True)
    if warmup_s is None:
        warmup_s = resolve_warmup(run_path)
    df = filter_run(df, warmup_s)
    if df.empty:
        warn(f"{basename}: empty after warmup/alive filter; skipping")
        return None

    diff_rss = df["rss_bytes"].astype(float).diff().dropna()
    n = len(diff_rss)
    if n == 0:
        warn(f"{basename}: insufficient samples for diff_rss; skipping")
        return None

    rss_vms_corr = float("nan")
    if "vms_bytes" in df.columns:
        diff_vms = df["vms_bytes"].astype(float).diff().dropna()
        common = diff_rss.index.intersection(diff_vms.index)
        if len(common) >= 2 and diff_rss.loc[common].std() > 0 and diff_vms.loc[common].std() > 0:
            rss_vms_corr = float(np.corrcoef(
                diff_rss.loc[common].values, diff_vms.loc[common].values
            )[0, 1])

    arr = diff_rss.values
    if float(np.var(arr)) == 0.0:
        warn(f"{basename}: ΔRSS has zero variance; K undefined")
        K_raw = float("nan")
        K_raw_ci_lo = K_raw_ci_hi = float("nan")
        K_trim = float("nan")
        K_trim_ci_lo = K_trim_ci_hi = float("nan")
    else:
        K_raw = float(kurtosis(arr, fisher=True, bias=False))
        arr_trim = np.asarray(winsorize(arr, limits=(0, 0.001)))
        K_trim = float(kurtosis(arr_trim, fisher=True, bias=False))
        if n < 100:
            warn(f"{basename}: n={n} < 100, skipping bootstrap CI")
            K_raw_ci_lo = K_raw_ci_hi = float("nan")
            K_trim_ci_lo = K_trim_ci_hi = float("nan")
        else:
            rng = np.random.default_rng(seed)
            K_raw_ci_lo, K_raw_ci_hi = bootstrap_ci(arr, n_bootstrap, rng)
            K_trim_ci_lo, K_trim_ci_hi = bootstrap_ci(arr_trim, n_bootstrap, rng)

    one_mb = 1024**2
    step_count_1mb = int(np.sum(arr > one_mb))
    duration_s = float(df["ts_unix"].max() - df["ts_unix"].min())
    duration_h = duration_s / 3600.0
    steps_per_h_1mb = step_count_1mb / duration_h if duration_h > 0 else float("nan")

    p99 = float(np.percentile(arr, 99))
    top_mask = arr >= p99
    top_vals = arr[top_mask]
    mean_top1_step_mb = float(top_vals.mean()) / 1024**2 if top_vals.size else float("nan")

    ts_values = df["ts_unix"].astype(float).values[1:]
    return {
        "run_id": basename,
        "cell_id": display_cell_id(infer_cell_id(basename, manifest), basename),
        "n_samples": n,
        "rss_vms_corr": rss_vms_corr,
        "K_raw": K_raw,
        "K_raw_ci_lo": K_raw_ci_lo,
        "K_raw_ci_hi": K_raw_ci_hi,
        "K_trim": K_trim,
        "K_trim_ci_lo": K_trim_ci_lo,
        "K_trim_ci_hi": K_trim_ci_hi,
        "steps_per_h_1mb": steps_per_h_1mb,
        "mean_top1_step_mb": mean_top1_step_mb,
        "_diff_rss": arr,
        "_diff_ts": ts_values,
    }


def fmt_num(x, fmt):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return format(x, fmt)


def print_pretty(rows):
    header = (
        f"{'run_id':<44} {'cell':<5} {'n':>6} "
        f"{'corr':>6} "
        f"{'K_raw':>8} {'CI95':<22} "
        f"{'K_trim':>8} {'CI95':<18} "
        f"{'steps>1MB/h':>11} "
        f"{'top1%_step':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ci_raw = (
            f"[{fmt_num(r['K_raw_ci_lo'], '+.0f')}, "
            f"{fmt_num(r['K_raw_ci_hi'], '+.0f')}]"
        )
        ci_trim = (
            f"[{fmt_num(r['K_trim_ci_lo'], '+.1f')}, "
            f"{fmt_num(r['K_trim_ci_hi'], '+.1f')}]"
        )
        print(
            f"{r['run_id']:<44} "
            f"{r['cell_id']:<5} "
            f"{r['n_samples']:>6} "
            f"{fmt_num(r['rss_vms_corr'], '6.2f')} "
            f"{fmt_num(r['K_raw'], '+8.0f')} {ci_raw:<22} "
            f"{fmt_num(r['K_trim'], '+8.1f')} {ci_trim:<18} "
            f"{fmt_num(r['steps_per_h_1mb'], '11.2f')} "
            f"{fmt_num(r['mean_top1_step_mb'], '8.4f')} MB"
        )


def print_csv(rows):
    print(
        "run_id,cell_id,n_samples,rss_vms_corr,"
        "K_raw,K_raw_ci_lo,K_raw_ci_hi,"
        "K_trim,K_trim_ci_lo,K_trim_ci_hi,"
        "steps_per_h_1mb,mean_top1_step_mb"
    )
    for r in rows:
        print(
            f"{r['run_id']},{r['cell_id']},{r['n_samples']},"
            f"{r['rss_vms_corr']:.6f},"
            f"{r['K_raw']:.6f},{r['K_raw_ci_lo']:.6f},{r['K_raw_ci_hi']:.6f},"
            f"{r['K_trim']:.6f},{r['K_trim_ci_lo']:.6f},{r['K_trim_ci_hi']:.6f},"
            f"{r['steps_per_h_1mb']:.6f},{r['mean_top1_step_mb']:.6f}"
        )


def print_top_k(rows, k):
    for r in rows:
        arr = r["_diff_rss"]
        ts = r["_diff_ts"]
        if arr.size == 0:
            continue
        idx = np.argsort(arr)[::-1][:k]
        print(f"\n# top {k} ΔRSS events for {r['run_id']} ({r['cell_id']})")
        for i in idx:
            when = datetime.fromtimestamp(float(ts[i]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"  {when}  +{arr[i] / 1024**2:8.3f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="single run directory")
    g.add_argument("--logs-root", help="parent containing multiple run dirs")
    p.add_argument(
        "--warmup-s",
        type=int,
        default=None,
        help="warmup discard in seconds; if omitted, resolved per-run "
             "(wosar2026_*: campaign cell yaml; aging_pilot_*: 1800s)",
    )
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for K CI")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")
    p.add_argument("--csv", action="store_true", help="machine-readable CSV output")
    p.add_argument("--top-k", type=int, default=0, help="if >0, also print top-N ΔRSS events per run")
    args = p.parse_args()

    if args.run_dir:
        targets = [Path(args.run_dir)]
    else:
        targets = discover_runs(args.logs_root)
        if not targets:
            warn(f"no aging_pilot_* or wosar2026_* subdirs under {args.logs_root}")
            sys.exit(1)

    rows = []
    for rd in targets:
        res = analyze_run(rd, args.warmup_s, args.bootstrap, args.seed)
        if res is not None:
            rows.append(res)

    if not rows:
        warn("no runs produced valid output")
        sys.exit(1)

    if args.csv:
        print_csv(rows)
    else:
        print_pretty(rows)

    if args.top_k > 0:
        print_top_k(rows, args.top_k)


if __name__ == "__main__":
    main()
