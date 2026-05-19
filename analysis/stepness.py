"""
Quantify the step-wise RSS growth pattern (paper Section IV.E, Figure 2(b)).
Headline metric: excess kurtosis K of ΔRSS post-warmup, with bootstrap 95% CI
and RSS-VMS lag-0 correlation as a companion. See EXPERIMENT_STATE.md
"Open questions #4" for motivation. CLI/parsing pattern follows replicate_n1.py.
"""
import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


# Engine labels that proc_monitor writes as <label>_<seq>.csv.
KNOWN_LABELS = [
    "vllm_v0_standalone",
    "vllm_v1_standalone",
    "vllm_standalone",
    "triton_vllm_v0",
    "triton_vllm_v1",
    "triton_vllm",
    "pytorch_naive",
]

# Fixed mapping from pilot run dir basename to paper cell_id.
PILOT_CELL_MAP = {
    "aging_pilot_24h_vllm_v1":                   "E1",
    "aging_pilot_24h_triton_v1":                 "E2",
    "aging_pilot_24h_pytorch_naive_v1":          "E3",
    "aging_pilot_24h_pytorch_naive_low_rate_v1": "E3b",
    "aging_pilot_24h_vllm_v0_ablation_v2":       "A1",
    "aging_pilot_24h_triton_v1_ablation_v2":     "A2",
}


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def detect_label(run_dir):
    """Return the first KNOWN_LABELS entry that has matching CSVs in run_dir."""
    for lab in KNOWN_LABELS:
        if glob.glob(os.path.join(run_dir, f"{lab}_*.csv")):
            return lab
    return None


def cell_id_for(run_basename):
    if run_basename in PILOT_CELL_MAP:
        return PILOT_CELL_MAP[run_basename]
    if run_basename.startswith("wosar2026_"):
        return run_basename[len("wosar2026_"):]
    return run_basename


def load_proc(run_dir, label):
    files = sorted(glob.glob(os.path.join(run_dir, f"{label}_*.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            cols = ["ts_unix", "process_alive", "rss_bytes", "vms_bytes"]
            d = pd.read_csv(f, usecols=lambda c: c in cols)
            dfs.append(d)
        except Exception as e:
            warn(f"failed to read {f}: {e}")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_unix").drop_duplicates("ts_unix").reset_index(drop=True)
    return df


def filter_run(df, warmup_s):
    t0 = df["ts_unix"].min()
    df = df[df["ts_unix"] >= t0 + warmup_s].copy()
    if "process_alive" in df.columns:
        df = df[df["process_alive"].astype(str).str.lower().isin(["true", "1"])]
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
    basename = os.path.basename(os.path.normpath(run_dir))
    label = detect_label(run_dir)
    if label is None:
        warn(f"{basename}: no proc_monitor CSV matching known engine labels; skipping")
        return None
    df = load_proc(run_dir, label)
    if df is None or df.empty:
        warn(f"{basename}: empty or unreadable proc CSVs; skipping")
        return None
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
        K = float("nan")
        ci_lo = ci_hi = float("nan")
    else:
        K = float(kurtosis(arr, fisher=True, bias=False))
        if n < 100:
            warn(f"{basename}: n={n} < 100, skipping bootstrap CI")
            ci_lo = ci_hi = float("nan")
        else:
            rng = np.random.default_rng(seed)
            ci_lo, ci_hi = bootstrap_ci(arr, n_bootstrap, rng)

    p99 = float(np.percentile(arr, 99))
    top_mask = arr >= p99
    top_vals = arr[top_mask]
    mean_top1_step_mb = float(top_vals.mean()) / 1024**2 if top_vals.size else float("nan")

    ts_values = df["ts_unix"].astype(float).values[1:]
    return {
        "run_id": basename,
        "cell_id": cell_id_for(basename),
        "n_samples": n,
        "K": K,
        "K_ci_lo": ci_lo,
        "K_ci_hi": ci_hi,
        "rss_vms_corr": rss_vms_corr,
        "mean_top1_step_mb": mean_top1_step_mb,
        "_diff_rss": arr,
        "_diff_ts": ts_values,
    }


def fmt_num(x, fmt):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return format(x, fmt)


def print_pretty(rows):
    header = f"{'run_id':<44} {'cell':<7} {'n':<7} {'K':>9} {'CI95':<22} {'corr':>6} {'top1%_step':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{fmt_num(r['K_ci_lo'], '+.2f')}, {fmt_num(r['K_ci_hi'], '+.2f')}]"
        print(
            f"{r['run_id']:<44} "
            f"{r['cell_id']:<7} "
            f"{r['n_samples']:<7} "
            f"{fmt_num(r['K'], '+9.2f')} "
            f"{ci:<22} "
            f"{fmt_num(r['rss_vms_corr'], '6.3f')} "
            f"{fmt_num(r['mean_top1_step_mb'], '8.4f')} MB"
        )


def print_csv(rows):
    print("run_id,cell_id,n_samples,K,K_ci_lo,K_ci_hi,rss_vms_corr,mean_top1_step_mb")
    for r in rows:
        print(
            f"{r['run_id']},{r['cell_id']},{r['n_samples']},"
            f"{r['K']:.6f},{r['K_ci_lo']:.6f},{r['K_ci_hi']:.6f},"
            f"{r['rss_vms_corr']:.6f},{r['mean_top1_step_mb']:.6f}"
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


def discover_runs(logs_root):
    out = []
    for entry in sorted(os.listdir(logs_root)):
        if entry.startswith("aging_pilot_") or entry.startswith("wosar2026_"):
            path = os.path.join(logs_root, entry)
            if os.path.isdir(path):
                out.append(path)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="single run directory")
    g.add_argument("--logs-root", help="parent containing multiple run dirs")
    p.add_argument("--warmup-s", type=int, default=1800, help="warmup discard in seconds")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for K CI")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")
    p.add_argument("--csv", action="store_true", help="machine-readable CSV output")
    p.add_argument("--top-k", type=int, default=0, help="if >0, also print top-N ΔRSS events per run")
    args = p.parse_args()

    if args.run_dir:
        targets = [args.run_dir]
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
