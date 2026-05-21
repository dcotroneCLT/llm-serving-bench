"""
Quantify the step-wise memory growth pattern (paper Section IV.E, Figure 2(b)).

Three metrics per run, computed post-warmup:

- ``rss_vms_corr`` (primary): lag-0 cross-correlation of dRSS and dVMS.
  High (> 0.8) is the mmap-style signature (kernel-mapped pages); low
  (< 0.5) means dRSS moves without paired dVMS.
- ``K_trim_dRSS`` (secondary): trimmed excess kurtosis of dRSS after
  winsorize at the 99.9 percentile, with bootstrap 95% CI. The raw
  ``K_raw_dRSS`` is kept as a companion.
- ``K_trim_dVMS`` (secondary, added 2026-05-21): same trimmed-kurtosis
  formula applied to dVMS. Distinguishes VAS-only growth (VMS jumps
  without paged-in RSS) from the other classes. Same fallback rule
  as ``K_trim_dRSS`` for the low-step edge case.

Operational descriptors: ``steps_per_h_1mb`` (dRSS jumps > 1 MB per
hour) and ``mean_top1_step_mb`` (mean of the top 1% dRSS jumps,
expressed in MB).

Five-class taxonomy with a border bucket. See
``classify_stepness`` and ``analysis/README.md``:

- mmap-style step-wise: corr > 0.8 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10
- sbrk-style step-wise: corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS < 5
- VAS-only step-wise: corr < 0.5 AND K_trim_dRSS < 5 AND K_trim_dVMS > 10
- uncorrelated step-wise: corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10
- continuous drift: corr < 0.5 AND K_trim_dRSS < 5 AND K_trim_dVMS < 5
- border: everything else (mixed, NaN, or out-of-bin)

Low-step fallback (operational-driven, NOT math-driven). The rule
keys off the operational descriptor on the same axis, independent of
whether ``K_trim`` was computable numerically. If
``steps_per_h_1mb < 0.01`` on that series (≈ < 1 MB-scale jump per
100 hours), the run has no real step events on that axis and
``K_trim`` is overridden to 0.0 with a stderr warning and a
``*_low_step_operational_drift`` entry in ``notes``. The override
fires even when the raw ``K_trim`` is large and finite: numerically
large kurtosis on a near-flat series is a winsorize artifact, not a
real step-wise signature. Calibration: the threshold is set an order
of magnitude below the lowest mmap-style cell observed in the n=3
campaign (e2_r02 at ≈ 0.09 steps/h) to keep genuine sparse step-wise
runs out of the fallback. If the operational condition does not hold
but ``K_trim`` is still NaN/inf, ``notes`` records
``*_kurtosis_undefined`` so downstream consumers can flag the row.

CLI/parsing pattern follows replicate_n1.py. See EXPERIMENT_STATE.md
"Step-wise mechanism panel" for the paper-side motivation.
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


def _compute_kurtosis_metrics(arr, n_bootstrap, rng, basename, label):
    """Kurtosis + bootstrap CI + low-step fallback for a single diff series.

    Returns ``(K_raw, K_raw_ci, K_trim, K_trim_ci, steps_per_h_1mb,
    mean_top1_step_mb, note)`` where ``note`` is one of ``""``,
    ``"<label>_low_step_fallback"``, or ``"<label>_kurtosis_undefined"``.
    """
    n = arr.size
    one_mb = 1024**2
    duration_marker = None  # caller computes duration; we need only step counts here.
    step_count_1mb = int(np.sum(arr > one_mb)) if n > 0 else 0
    if n > 0:
        p99 = float(np.percentile(arr, 99))
        top_vals = arr[arr >= p99]
        mean_top1_step_mb = float(top_vals.mean()) / one_mb if top_vals.size else float("nan")
    else:
        mean_top1_step_mb = float("nan")

    note = ""
    if n == 0 or float(np.var(arr)) == 0.0:
        warn(f"{basename}: Δ{label} has zero variance; K undefined")
        K_raw = float("nan")
        K_raw_ci_lo = K_raw_ci_hi = float("nan")
        K_trim = float("nan")
        K_trim_ci_lo = K_trim_ci_hi = float("nan")
    else:
        K_raw = float(kurtosis(arr, fisher=True, bias=False))
        arr_trim = np.asarray(winsorize(arr, limits=(0, 0.001)))
        K_trim = float(kurtosis(arr_trim, fisher=True, bias=False))
        if n < 100:
            warn(f"{basename}: n={n} < 100 for Δ{label}, skipping bootstrap CI")
            K_raw_ci_lo = K_raw_ci_hi = float("nan")
            K_trim_ci_lo = K_trim_ci_hi = float("nan")
        else:
            K_raw_ci_lo, K_raw_ci_hi = bootstrap_ci(arr, n_bootstrap, rng)
            K_trim_ci_lo, K_trim_ci_hi = bootstrap_ci(arr_trim, n_bootstrap, rng)

    return {
        "K_raw": K_raw,
        "K_raw_ci_lo": K_raw_ci_lo,
        "K_raw_ci_hi": K_raw_ci_hi,
        "K_trim": K_trim,
        "K_trim_ci_lo": K_trim_ci_lo,
        "K_trim_ci_hi": K_trim_ci_hi,
        "step_count_1mb": step_count_1mb,
        "mean_top1_step_mb": mean_top1_step_mb,
        "note_seed": note,
        "_label": label,
    }


def _apply_low_step_fallback(m, steps_per_h_1mb, basename):
    """Operational-driven low-step override for K_trim.

    The rule keys off ``steps_per_h_1mb`` only, independent of the
    numerical K_trim value. If the series has < 0.01 MB-scale jumps
    per hour, K_trim is forced to 0.0 even when the raw computation
    returned a large finite kurtosis (winsorize on a near-flat series
    inflates K). Mutates the dict in place and returns the note
    string ("" if no fallback was applied,
    ``"<label>_low_step_operational_drift"`` when the override fired,
    or ``"<label>_kurtosis_undefined"`` when K_trim is NaN/inf and
    the fallback did NOT fire).
    """
    label = m["_label"]
    K_trim = m["K_trim"]

    low_step = np.isfinite(steps_per_h_1mb) and steps_per_h_1mb < 0.01
    if low_step:
        warn(
            f"{basename}: Δ{label} no real step events "
            f"(steps/h={steps_per_h_1mb:.4f} < 0.01, "
            f"K_trim_raw={K_trim:+.1f}); "
            f"low-step operational fallback → K_trim=0.0 (drift)"
        )
        m["K_trim"] = 0.0
        m["K_trim_ci_lo"] = float("nan")
        m["K_trim_ci_hi"] = float("nan")
        return f"{label}_low_step_operational_drift"

    if not np.isfinite(K_trim):
        return f"{label}_kurtosis_undefined"
    return ""


def classify_stepness(corr, k_trim_drss, k_trim_dvms):
    """Five-class taxonomy plus a border bucket.

    See EXPERIMENT_STATE.md "Step-wise mechanism panel" and the
    README of analysis/ for the mechanism interpretation of each
    class. Any NaN input falls into ``border``.
    """
    if not (np.isfinite(corr) and np.isfinite(k_trim_drss) and np.isfinite(k_trim_dvms)):
        return "border"
    if corr > 0.8 and k_trim_drss > 10 and k_trim_dvms > 10:
        return "mmap-style step-wise"
    if corr < 0.5 and k_trim_drss > 10 and k_trim_dvms < 5:
        return "sbrk-style step-wise"
    if corr < 0.5 and k_trim_drss < 5 and k_trim_dvms > 10:
        return "VAS-only step-wise"
    if corr < 0.5 and k_trim_drss < 5 and k_trim_dvms < 5:
        return "continuous drift"
    if corr < 0.5 and k_trim_drss > 10 and k_trim_dvms > 10:
        return "uncorrelated step-wise"
    return "border"


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
    if len(diff_rss) == 0:
        warn(f"{basename}: insufficient samples for diff_rss; skipping")
        return None
    arr_rss = diff_rss.values

    has_vms = "vms_bytes" in df.columns
    if has_vms:
        diff_vms = df["vms_bytes"].astype(float).diff().dropna()
        common = diff_rss.index.intersection(diff_vms.index)
        if len(common) >= 2 and diff_rss.loc[common].std() > 0 and diff_vms.loc[common].std() > 0:
            rss_vms_corr = float(np.corrcoef(
                diff_rss.loc[common].values, diff_vms.loc[common].values
            )[0, 1])
        else:
            rss_vms_corr = float("nan")
        arr_vms = diff_vms.values
    else:
        rss_vms_corr = float("nan")
        arr_vms = np.array([], dtype=float)

    duration_s = float(df["ts_unix"].max() - df["ts_unix"].min())
    duration_h = duration_s / 3600.0

    rng = np.random.default_rng(seed)
    m_rss = _compute_kurtosis_metrics(arr_rss, n_bootstrap, rng, basename, "RSS")
    m_vms = _compute_kurtosis_metrics(arr_vms, n_bootstrap, rng, basename, "VMS")

    steps_per_h_1mb = m_rss["step_count_1mb"] / duration_h if duration_h > 0 else float("nan")
    steps_per_h_1mb_dvms = (
        m_vms["step_count_1mb"] / duration_h if duration_h > 0 else float("nan")
    )

    notes = []
    n_rss = _apply_low_step_fallback(m_rss, steps_per_h_1mb, basename)
    if n_rss:
        notes.append(n_rss)
    n_vms = _apply_low_step_fallback(m_vms, steps_per_h_1mb_dvms, basename)
    if n_vms:
        notes.append(n_vms)

    if not has_vms:
        notes.append("VMS_missing")

    cls = classify_stepness(rss_vms_corr, m_rss["K_trim"], m_vms["K_trim"])

    ts_values = df["ts_unix"].astype(float).values[1:]
    return {
        "run_id": basename,
        "cell_id": display_cell_id(infer_cell_id(basename, manifest), basename),
        "n_samples": len(arr_rss),
        "rss_vms_corr": rss_vms_corr,
        # RSS metrics (canonical column names keep paper/CSV compat).
        "K_raw": m_rss["K_raw"],
        "K_raw_ci_lo": m_rss["K_raw_ci_lo"],
        "K_raw_ci_hi": m_rss["K_raw_ci_hi"],
        "K_trim": m_rss["K_trim"],
        "K_trim_ci_lo": m_rss["K_trim_ci_lo"],
        "K_trim_ci_hi": m_rss["K_trim_ci_hi"],
        # Explicit dRSS aliases for any downstream tool that prefers
        # the unambiguous name; values identical to the K_raw/K_trim
        # fields above.
        "K_raw_dRSS": m_rss["K_raw"],
        "K_raw_dRSS_ci_lo": m_rss["K_raw_ci_lo"],
        "K_raw_dRSS_ci_hi": m_rss["K_raw_ci_hi"],
        "K_trim_dRSS": m_rss["K_trim"],
        "K_trim_dRSS_ci_lo": m_rss["K_trim_ci_lo"],
        "K_trim_dRSS_ci_hi": m_rss["K_trim_ci_hi"],
        # dVMS metrics.
        "K_raw_dVMS": m_vms["K_raw"],
        "K_raw_dVMS_ci_lo": m_vms["K_raw_ci_lo"],
        "K_raw_dVMS_ci_hi": m_vms["K_raw_ci_hi"],
        "K_trim_dVMS": m_vms["K_trim"],
        "K_trim_dVMS_ci_lo": m_vms["K_trim_ci_lo"],
        "K_trim_dVMS_ci_hi": m_vms["K_trim_ci_hi"],
        # Operational descriptors.
        "steps_per_h_1mb": steps_per_h_1mb,
        "mean_top1_step_mb": m_rss["mean_top1_step_mb"],
        "steps_per_h_1mb_dVMS": steps_per_h_1mb_dvms,
        "mean_top1_step_mb_dVMS": m_vms["mean_top1_step_mb"],
        # Classification + provenance.
        "class": cls,
        "notes": ";".join(notes) if notes else "",
        # Internal: kept for --top-k path.
        "_diff_rss": arr_rss,
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
        f"{'K_trim_dRSS':>11} {'CI95':<18} "
        f"{'K_trim_dVMS':>11} {'CI95':<18} "
        f"{'steps>1MB/h':>11} "
        f"{'top1%_step':>12} "
        f"{'class':<22}"
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
        ci_trim_vms = (
            f"[{fmt_num(r['K_trim_dVMS_ci_lo'], '+.1f')}, "
            f"{fmt_num(r['K_trim_dVMS_ci_hi'], '+.1f')}]"
        )
        print(
            f"{r['run_id']:<44} "
            f"{r['cell_id']:<5} "
            f"{r['n_samples']:>6} "
            f"{fmt_num(r['rss_vms_corr'], '6.2f')} "
            f"{fmt_num(r['K_raw'], '+8.0f')} {ci_raw:<22} "
            f"{fmt_num(r['K_trim'], '+11.1f')} {ci_trim:<18} "
            f"{fmt_num(r['K_trim_dVMS'], '+11.1f')} {ci_trim_vms:<18} "
            f"{fmt_num(r['steps_per_h_1mb'], '11.2f')} "
            f"{fmt_num(r['mean_top1_step_mb'], '8.4f')} MB "
            f"{r['class']:<22}"
        )


def print_csv(rows):
    # Existing columns kept in their existing positions; new columns
    # appended at the end. K_raw / K_trim are the RSS variants (same
    # values as K_raw_dRSS / K_trim_dRSS).
    print(
        "run_id,cell_id,n_samples,rss_vms_corr,"
        "K_raw,K_raw_ci_lo,K_raw_ci_hi,"
        "K_trim,K_trim_ci_lo,K_trim_ci_hi,"
        "steps_per_h_1mb,mean_top1_step_mb,"
        "K_raw_dVMS,K_raw_dVMS_ci_lo,K_raw_dVMS_ci_hi,"
        "K_trim_dVMS,K_trim_dVMS_ci_lo,K_trim_dVMS_ci_hi,"
        "steps_per_h_1mb_dVMS,mean_top1_step_mb_dVMS,"
        "class,notes"
    )
    for r in rows:
        print(
            f"{r['run_id']},{r['cell_id']},{r['n_samples']},"
            f"{r['rss_vms_corr']:.6f},"
            f"{r['K_raw']:.6f},{r['K_raw_ci_lo']:.6f},{r['K_raw_ci_hi']:.6f},"
            f"{r['K_trim']:.6f},{r['K_trim_ci_lo']:.6f},{r['K_trim_ci_hi']:.6f},"
            f"{r['steps_per_h_1mb']:.6f},{r['mean_top1_step_mb']:.6f},"
            f"{r['K_raw_dVMS']:.6f},{r['K_raw_dVMS_ci_lo']:.6f},{r['K_raw_dVMS_ci_hi']:.6f},"
            f"{r['K_trim_dVMS']:.6f},{r['K_trim_dVMS_ci_lo']:.6f},{r['K_trim_dVMS_ci_hi']:.6f},"
            f"{r['steps_per_h_1mb_dVMS']:.6f},{r['mean_top1_step_mb_dVMS']:.6f},"
            f"{r['class']},{r['notes']}"
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
