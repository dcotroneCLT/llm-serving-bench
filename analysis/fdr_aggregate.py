"""Cross-cell BH-FDR aggregator for the WoSAR 2026 trend pipeline.

Reads one or more --trends-csv FILE paths produced by
analysis/aging_trends.py --csv. Concatenates rows across files (each
row is one (run_id, indicator) hypothesis), applies Benjamini-Hochberg
FDR at q=alpha across the joint family of mk_p_value, and emits a
table that adds q_value and bh_reject columns.

Decision rule (paper):
  A trend is declared significant when ALL THREE hold:
    1. mk_p_value < alpha            (per-indicator MK rejection)
    2. slope CI excludes 0           (Theil-Sen exact order-stat CI
                                      with AR(1) variance inflation)
    3. bh_reject is True             (BH-FDR across the joint family)
  Conditions (1) and (2) are produced by aging_trends.py.
  Condition (3) is the multiplicity correction this script adds.

Uses statsmodels.stats.multitest.multipletests if available; falls
back to a manual BH implementation otherwise (sort, rank, multiply
p_i by N/rank_i, cumulative min).

Usage:
  python3 analysis/aging_trends.py --csv --run-dir <run1> > trends1.csv
  python3 analysis/aging_trends.py --csv --run-dir <run2> > trends2.csv
  python3 analysis/fdr_aggregate.py \\
      --trends-csv trends1.csv --trends-csv trends2.csv [--alpha 0.10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "run_id", "cell_id", "indicator", "n_samples",
    "slope", "slope_ci_lo", "slope_ci_hi",
    "mk_z", "mk_p_value", "lag1_rho",
]


def bh_fdr_manual(pvals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    n = int(valid.sum())
    q = np.full_like(p, np.nan)
    reject = np.zeros(len(p), dtype=bool)
    if n == 0:
        return q, reject
    order = np.argsort(p[valid])
    ranked_p = p[valid][order]
    ranks = np.arange(1, n + 1, dtype=float)
    raw_q = ranked_p * n / ranks
    adj = np.minimum.accumulate(raw_q[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    q_valid = np.empty(n)
    q_valid[order] = adj
    q[valid] = q_valid
    reject[valid] = q_valid < alpha
    return q, reject


def bh_fdr(pvals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        q, reject = bh_fdr_manual(pvals, alpha)
        return q, reject, "manual"
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    reject = np.zeros(len(p), dtype=bool)
    if int(valid.sum()) == 0:
        return q, reject, "statsmodels"
    rej_v, q_v, _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
    q[valid] = q_v
    reject[valid] = rej_v
    return q, reject, "statsmodels"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--trends-csv", type=Path, action="append", required=True,
                   help="CSV produced by aging_trends.py --csv. Pass multiple "
                        "times to concatenate across runs.")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="BH-FDR target q (default 0.10).")
    p.add_argument("--csv", action="store_true",
                   help="machine-readable CSV output to stdout.")
    args = p.parse_args()

    frames = []
    for path in args.trends_csv:
        if not path.is_file():
            print(f"Not a file: {path}", file=sys.stderr); sys.exit(1)
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"{path}: missing required columns: {missing}", file=sys.stderr)
            sys.exit(1)
        frames.append(df)
    if not frames:
        print("no input frames", file=sys.stderr); sys.exit(1)
    df = pd.concat(frames, ignore_index=True)

    q_values, rejected, backend = bh_fdr(
        df["mk_p_value"].to_numpy(dtype=float), args.alpha
    )
    df["q_value"] = q_values
    df["bh_reject"] = rejected

    ci_lo = df["slope_ci_lo"]
    ci_hi = df["slope_ci_hi"]
    ci_excludes_zero = (
        ci_lo.notna() & ci_hi.notna() & ~((ci_lo <= 0) & (ci_hi >= 0))
    )
    mk_reject_raw = df["mk_p_value"] < args.alpha
    df["significant"] = df["bh_reject"] & ci_excludes_zero & mk_reject_raw

    if args.csv:
        df.to_csv(sys.stdout, index=False)
        return

    n_tests = int(df["mk_p_value"].notna().sum())
    n_bh = int(df["bh_reject"].sum())
    n_sig = int(df["significant"].sum())

    print(f"\nCross-cell BH-FDR aggregation  "
          f"(alpha={args.alpha:.2f}, family size N={n_tests}, backend={backend})")
    print("=" * 140)
    header = (f"{'run_id':<40} {'cell':<5} {'indicator':<30} "
              f"{'slope':>12} {'CI_lo':>12} {'CI_hi':>12} "
              f"{'p_raw':>9} {'q_BH':>9} {'sig':>5}")
    print(header); print("-" * 140)
    for _, r in df.sort_values(["run_id", "indicator"]).iterrows():
        sig = "YES" if r["significant"] else "no"
        print(f"{str(r['run_id']):<40} {str(r['cell_id']):<5} {str(r['indicator']):<30} "
              f"{r['slope']:>12.4g} {r['slope_ci_lo']:>12.4g} {r['slope_ci_hi']:>12.4g} "
              f"{r['mk_p_value']:>9.4f} {r['q_value']:>9.4f} {sig:>5}")
    print("=" * 140)
    print(f"\nBH rejected: {n_bh}/{n_tests}  |  "
          f"Significant (MK + CI + BH): {n_sig}/{n_tests}")


if __name__ == "__main__":
    main()
