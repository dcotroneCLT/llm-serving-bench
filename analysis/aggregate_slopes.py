#!/usr/bin/env python3
"""Per-cell slope aggregation across n=3 replicas for the WoSAR 2026 campaign.

Two estimators side-by-side, computed per (cell_id, indicator):

  1. PRIMARY: DerSimonian-Laird random-effects on the k per-replica Theil-Sen
     slope estimates. SE per replica derived from the TS exact 95% CI via
     the Gaussian-equivalent formula (ci_hi - ci_lo) / (2 * 1.96). Reported
     CI is 95% Gaussian on theta_RE.

  2. ROBUSTNESS: sample median of the k per-replica Theil-Sen slopes, with
     the [min, max] of the k reported as the exact non-parametric CI for
     the population median. Coverage as a function of k:
         P(Y_(1) <= median <= Y_(k)) = 1 - 2 * (1/2)^k
     For k=3 this is 1 - 1/4 = 75% (NOT 95%). For k=4 -> 87.5%, k=5 -> 93.75%.
     A percentile bootstrap CI labelled 95% on k=3 would collapse to the
     same [min, max] interval (the bootstrap median can only take one of
     the k observed values) and would therefore mis-state its coverage.
     We report [min, max] honestly with its true coverage.

Per-cell BH-FDR (q = alpha) is applied across the joint family of
(cell_id, indicator) tests. The per-cell p-value comes from Stouffer's
z-score combination of the per-replica MK z statistics in the input
trend CSVs.

Decision rule for "RE_significant" (headline):
    (cell-level BH-FDR rejects the Stouffer-combined p) AND
    (RE 95% CI excludes 0) AND
    (n_replicas, k_used_RE, and k_used_stouffer meet --expected-replicas).

The pooled-median CI-excludes-zero flag is reported separately as
`pooled_ci_excludes_zero` but does NOT gate any significance decision: at
75% coverage for k=3 it is too conservative to serve as a 95% test, and
it is meant only as an informational robustness cross-check. Headline
significance is taken from RE-DL.

Usage:
    # Step 1: per-run trend CSVs (already exists, from aging_trends.py)
    python3 analysis/aging_trends.py --csv --run-dir <run> > /tmp/<run>_trends.csv

    # Step 2: per-cell aggregation across replicas (no raw CSV access needed)
    python3 analysis/aggregate_slopes.py \\
        --trends-csv /tmp/wosar2026_e1_r01_trends.csv \\
        --trends-csv /tmp/wosar2026_e1_r02_trends.csv \\
        --trends-csv /tmp/wosar2026_e1_r03_trends.csv \\
        ... (18 total for full campaign) \\
        [--alpha 0.10] [--expected-replicas 3] [--no-pooled] [--indicator I,J,...] \\
        [--csv] > /tmp/n3_per_cell_slopes.csv

Dependencies: pandas, numpy, scipy. No raw CSV access required: both
estimators operate on the per-replica trend CSVs produced by
aging_trends.py.

CAVEAT 1 (SE-from-CI). The TS exact CI is order-statistics-based, not
Gaussian. Using (hi - lo) / (2 * 1.96) as a Gaussian-equivalent SE is the
standard meta-analysis convention when primary studies do not report SE.
On our data (post-warmup samples uniform across runs, n ~ 2100 per run
post-downsample) the asymmetry is small enough that the approximation
is acceptable. Reported alongside the per-replica raw slopes and CIs so
a reviewer can audit it.

CAVEAT 2 (small k for tau^2). With k=3 replicas, DL tau^2 is noisy. We
report tau^2 and I^2 alongside the RE slope so the heterogeneity is
transparent. CIs may be wide, especially for indicators where one
replica differs by orders of magnitude from the other two.

CAVEAT 3 (pooled-median is not a 95% CI). The [min, max] interval for
the population median has exact coverage 75% at k=3, 87.5% at k=4, etc.
Do NOT compare pooled CI to RE CI as if both were 95%; they have
different formal coverages. The role of the pooled estimator is to
detect when the RE central tendency is being dragged by one outlier
replica.

DESIGN HISTORY. An earlier version of this script computed a "pooled
Theil-Sen on the concatenated post-warmup series" with per-replica
median-centering. That estimator was systematically biased toward zero
because Theil-Sen on a median-centered concatenation gives O(N^2/3)
within-replica pairs (carrying the trend) but O(2 * N^2/3) cross-replica
pairs whose median-centered (y_b[t] - y_a[t]) is dominated by noise
around zero. Empirically pooled slope was 7x-30x below the RE estimate
on e2/e3 USS, and exactly 0 on e1 RSS. We replaced the pooled-series
estimator with the median-of-slopes estimator documented above, which
does not suffer the cross-pair pathology because it never inspects raw
data and operates only on per-replica slope point estimates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Note: this script no longer imports anything from analysis/aging_io.py;
# the median-of-slopes estimator works entirely off the per-replica trend
# CSVs and does not touch the raw monitor CSVs.


TRENDS_REQUIRED_COLUMNS = [
    "run_id", "cell_id", "indicator", "n_samples",
    "slope", "slope_ci_lo", "slope_ci_hi",
    "mk_z", "mk_p_value", "lag1_rho",
]

DEFAULT_ALPHA = 0.10
DEFAULT_EXPECTED_REPLICAS = 3


# ----------------------------- DerSimonian-Laird -----------------------------

def se_from_ci_midrange(ci_lo: float, ci_hi: float,
                        run_id: Optional[str] = None,
                        indicator: Optional[str] = None) -> float:
    """Gaussian-equivalent SE from a 95% CI half-width.

    Returns NaN if either bound is non-finite. Returns NaN with a stderr
    warning if the CI is degenerate (ci_lo == ci_hi, no measurable
    dispersion) or inverted (ci_lo > ci_hi, malformed); the replica is
    then dropped from the DL meta-analysis rather than receiving an
    inflated weight via an SE floor.
    """
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        return np.nan
    if ci_lo > ci_hi:
        print(f"  [warn] inverted CI [{ci_lo}, {ci_hi}] "
              f"(run_id={run_id}, indicator={indicator}); dropping replica",
              file=sys.stderr)
        return np.nan
    if ci_lo == ci_hi:
        print(f"  [warn] degenerate CI [{ci_lo}, {ci_hi}] "
              f"(run_id={run_id}, indicator={indicator}); dropping replica",
              file=sys.stderr)
        return np.nan
    return float((ci_hi - ci_lo) / (2.0 * 1.96))


def dersimonian_laird(slopes: np.ndarray, ses: np.ndarray,
                      ci_los: Optional[np.ndarray] = None,
                      ci_his: Optional[np.ndarray] = None) -> dict:
    """DL random-effects meta-analysis on k independent slope estimates.

    Drops entries where slope or SE is non-finite or SE <= 0. Returns a
    dict with theta_RE, se_RE, ci_lo, ci_hi, tau2, I2, Q, Q_pvalue, k_used.

    k_used = 1 falls back to the single surviving replica's slope. If
    `ci_los` / `ci_his` are provided, the function passes the upstream
    (asymmetric, order-statistics) CI through directly. Otherwise it
    rebuilds a symmetric CI from the Gaussian-equivalent SE. Same
    convention for k_used = 0.
    """
    slopes = np.asarray(slopes, dtype=float)
    ses = np.asarray(ses, dtype=float)
    ci_los_arr = np.asarray(ci_los, dtype=float) if ci_los is not None else None
    ci_his_arr = np.asarray(ci_his, dtype=float) if ci_his is not None else None
    mask = np.isfinite(slopes) & np.isfinite(ses) & (ses > 0)
    slopes_m = slopes[mask]
    ses_m = ses[mask]
    k = int(len(slopes_m))

    if k == 0:
        return dict(theta_RE=np.nan, se_RE=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    tau2=np.nan, I2=np.nan, Q=np.nan, Q_pvalue=np.nan, k_used=0)
    if k == 1:
        s, e = float(slopes_m[0]), float(ses_m[0])
        if ci_los_arr is not None and ci_his_arr is not None:
            los_kept = ci_los_arr[mask]
            his_kept = ci_his_arr[mask]
            ci_lo_val = float(los_kept[0]) if np.isfinite(los_kept[0]) else s - 1.96 * e
            ci_hi_val = float(his_kept[0]) if np.isfinite(his_kept[0]) else s + 1.96 * e
        else:
            ci_lo_val, ci_hi_val = s - 1.96 * e, s + 1.96 * e
        return dict(theta_RE=s, se_RE=e, ci_lo=ci_lo_val, ci_hi=ci_hi_val,
                    tau2=0.0, I2=0.0, Q=0.0, Q_pvalue=1.0, k_used=1)
    slopes = slopes_m
    ses = ses_m

    w = 1.0 / ses ** 2
    sw = float(np.sum(w))
    theta_FE = float(np.sum(w * slopes) / sw)
    Q = float(np.sum(w * (slopes - theta_FE) ** 2))
    df = k - 1
    denom = sw - float(np.sum(w ** 2)) / sw
    tau2 = max(0.0, (Q - df) / denom) if denom > 0 else 0.0
    w_star = 1.0 / (ses ** 2 + tau2)
    sw_star = float(np.sum(w_star))
    theta_RE = float(np.sum(w_star * slopes) / sw_star)
    se_RE = float(1.0 / np.sqrt(sw_star))
    ci_lo = theta_RE - 1.96 * se_RE
    ci_hi = theta_RE + 1.96 * se_RE
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    Q_pvalue = float(stats.chi2.sf(Q, df))
    return dict(theta_RE=theta_RE, se_RE=se_RE, ci_lo=ci_lo, ci_hi=ci_hi,
                tau2=tau2, I2=I2, Q=Q, Q_pvalue=Q_pvalue, k_used=k)


def stouffer_combine(zs: np.ndarray) -> tuple[float, float, int]:
    """Stouffer's z combination on the finite entries of `zs`.

    Returns (z_combined, two_sided_p, k_used). Uses scipy norm.sf for
    numerical stability with large z (where 1 - cdf cancels to 0).
    """
    zs = np.asarray(zs, dtype=float)
    zs = zs[np.isfinite(zs)]
    k = int(len(zs))
    if k == 0:
        return np.nan, np.nan, 0
    z = float(np.sum(zs) / np.sqrt(k))
    p = float(2.0 * stats.norm.sf(abs(z)))
    return z, p, k


# ----------------------------- BH-FDR ----------------------------------------

def bh_fdr(pvals: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    Returns (q_values, reject_mask). NaN p-values are excluded from the
    family count and receive q=NaN, reject=False. Rejection uses
    q <= alpha (BH 1995); equality is treated as a rejection.
    """
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
    reject[valid] = q_valid <= alpha
    return q, reject


# ----------------------------- pooled-median (robustness) --------------------

def pooled_median(slopes: np.ndarray) -> dict:
    """Sample median of k per-replica slope estimates with [min, max] CI.

    The [min, max] interval is the exact non-parametric confidence interval
    for the population median given k i.i.d. observations:

        Pr(Y_(1) <= median <= Y_(k))
            = Pr(1 <= B <= k-1)  with B ~ Binomial(k, 0.5)
            = 1 - 2 * (1/2)^k

    Coverage by k:
        k=2 -> 50.0 %
        k=3 -> 75.0 %
        k=4 -> 87.5 %
        k=5 -> 93.75 %
        k=6 -> 96.875 %

    For k=3 (the campaign default) [min, max] is therefore a 75% CI, NOT a
    95% CI. A percentile bootstrap CI labelled 95% on k=3 would collapse to
    the same [min, max] interval (the bootstrap median can only take one
    of the k observed values) and would mis-state its true coverage; we
    avoid that and report the exact 75% coverage directly.

    Returns: dict with `slope` (median), `ci_lo` (min), `ci_hi` (max),
    `coverage_pct`, and `k_used` (count of finite slopes used).
    Non-finite slopes are dropped from the median and the [min, max].
    """
    slopes = np.asarray(slopes, dtype=float)
    finite = slopes[np.isfinite(slopes)]
    k = int(len(finite))
    if k == 0:
        return dict(slope=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    coverage_pct=np.nan, k_used=0)
    if k == 1:
        # Degenerate: a single replica is its own min, max, and median.
        # Coverage is undefined; we report 0 to flag it.
        s = float(finite[0])
        return dict(slope=s, ci_lo=s, ci_hi=s, coverage_pct=0.0, k_used=1)
    return dict(
        slope=float(np.median(finite)),
        ci_lo=float(np.min(finite)),
        ci_hi=float(np.max(finite)),
        coverage_pct=float(100.0 * (1.0 - 2.0 ** (1 - k))),
        k_used=k,
    )


# ----------------------------- main ------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trends-csv", type=Path, action="append", required=True,
                    help="CSV produced by aging_trends.py --csv. Pass once per run.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help=f"BH-FDR target q (default {DEFAULT_ALPHA}).")
    ap.add_argument("--indicator", type=str, default=None,
                    help="Comma-separated indicator filter (e.g. proc.uss_bytes,"
                         "proc.rss_bytes). Default: all indicators present in inputs.")
    ap.add_argument("--no-pooled", action="store_true",
                    help="Skip the pooled-median robustness estimator (RE-DL only).")
    ap.add_argument("--expected-replicas", type=int, default=DEFAULT_EXPECTED_REPLICAS,
                    help=f"Replicas required for a headline significance decision "
                         f"(default {DEFAULT_EXPECTED_REPLICAS}).")
    ap.add_argument("--csv", action="store_true",
                    help="Machine-readable CSV output to stdout.")
    return ap.parse_args()


def load_trend_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.is_file():
            print(f"  [warn] not a file, skipped: {path}", file=sys.stderr); continue
        df = pd.read_csv(path)
        missing = [c for c in TRENDS_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"  [warn] {path}: missing columns {missing}, skipped", file=sys.stderr)
            continue
        frames.append(df)
    if not frames:
        print("no usable trend CSVs", file=sys.stderr); sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    # drop any duplicate (run_id, indicator) rows defensively
    df = df.drop_duplicates(subset=["run_id", "indicator"]).reset_index(drop=True)
    return df


def aggregate(df_trends: pd.DataFrame, do_pooled: bool) -> pd.DataFrame:
    bad = df_trends["cell_id"].isna() | df_trends["indicator"].isna()
    if bool(bad.any()):
        print(f"  [warn] {int(bad.sum())} trend row(s) with NaN cell_id/indicator dropped",
              file=sys.stderr)
        df_trends = df_trends[~bad].reset_index(drop=True)
    rows = []
    groups = df_trends.groupby(["cell_id", "indicator"])
    for (cell_id, indicator), g in groups:
        g = g.sort_values("run_id").reset_index(drop=True)
        slopes = g["slope"].to_numpy(dtype=float)
        ci_los = g["slope_ci_lo"].to_numpy(dtype=float)
        ci_his = g["slope_ci_hi"].to_numpy(dtype=float)
        mk_zs = g["mk_z"].to_numpy(dtype=float)
        run_ids_list = g["run_id"].tolist()
        ses = np.array([
            se_from_ci_midrange(lo, hi, run_id=rid, indicator=indicator)
            for rid, lo, hi in zip(run_ids_list, ci_los, ci_his)
        ])

        dl = dersimonian_laird(slopes, ses, ci_los=ci_los, ci_his=ci_his)
        z_comb, p_comb, k_stouffer = stouffer_combine(mk_zs)

        slope_strs = ";".join(f"{x:.6g}" for x in slopes)
        ci_strs = ";".join(f"[{lo:.6g},{hi:.6g}]" for lo, hi in zip(ci_los, ci_his))
        run_ids = ";".join(run_ids_list)

        row = {
            "cell_id": cell_id,
            "indicator": indicator,
            "n_replicas": int(len(g)),
            "k_used_RE": int(dl["k_used"]),
            "k_used_stouffer": int(k_stouffer),
            "run_ids": run_ids,
            "slope_per_replica": slope_strs,
            "ci_per_replica": ci_strs,
            "mk_z_per_replica": ";".join(f"{x:.4g}" for x in mk_zs),
            "slope_RE": dl["theta_RE"],
            "se_RE": dl["se_RE"],
            "ci_lo_RE": dl["ci_lo"],
            "ci_hi_RE": dl["ci_hi"],
            "tau2_RE": dl["tau2"],
            "I2_RE": dl["I2"],
            "Q_RE": dl["Q"],
            "Q_pvalue_RE": dl["Q_pvalue"],
            "stouffer_z": z_comb,
            "stouffer_p": p_comb,
        }

        if do_pooled:
            pooled = pooled_median(slopes)
            row.update({
                "slope_pooled": pooled["slope"],
                "ci_lo_pooled": pooled["ci_lo"],
                "ci_hi_pooled": pooled["ci_hi"],
                "pooled_coverage_pct": pooled["coverage_pct"],
                "k_used_pooled": pooled["k_used"],
            })
            # agreement: relative gap between RE central estimate and the
            # pooled median. Useful when the two estimators differ
            # substantially (RE pulled by outlier on per-replica SE).
            if np.isfinite(dl["theta_RE"]) and np.isfinite(pooled["slope"]) and dl["theta_RE"] != 0:
                row["agreement_pct"] = float(
                    100.0 * abs(dl["theta_RE"] - pooled["slope"]) / abs(dl["theta_RE"])
                )
            else:
                row["agreement_pct"] = np.nan
        else:
            row.update(dict(slope_pooled=np.nan, ci_lo_pooled=np.nan, ci_hi_pooled=np.nan,
                            pooled_coverage_pct=np.nan, k_used_pooled=0,
                            agreement_pct=np.nan))

        rows.append(row)
    return pd.DataFrame(rows)


def apply_bh_and_decision(df: pd.DataFrame, alpha: float,
                          expected_replicas: int = DEFAULT_EXPECTED_REPLICAS,
                          high_i2_threshold: float = 75.0) -> pd.DataFrame:
    q, reject = bh_fdr(df["stouffer_p"].to_numpy(dtype=float), alpha)
    df["q_value_cell"] = q
    df["bh_reject_cell"] = reject

    # Headline significance requires the full expected replica set. Rows below
    # this threshold remain visible for audit, but are marked as degraded and
    # excluded from RE_significant.
    df["degraded_replicas"] = (
        (df["n_replicas"] < expected_replicas)
        | (df["k_used_RE"] < expected_replicas)
        | (df["k_used_stouffer"] < expected_replicas)
    )

    re_ci_excludes_zero = (
        df["ci_lo_RE"].notna() & df["ci_hi_RE"].notna()
        & ~((df["ci_lo_RE"] <= 0) & (df["ci_hi_RE"] >= 0))
    )
    df["RE_significant"] = df["bh_reject_cell"] & re_ci_excludes_zero & ~df["degraded_replicas"]

    # Robustness flag, NOT a significance decision: does the pooled-TS 95% CI
    # exclude 0? Deliberately not gated by the Stouffer BH-FDR family, because
    # the pooled estimator runs on the concatenated series, not on the
    # per-replica MK z's that fed Stouffer. Cross-reference with k_used_pooled
    # and replicas_disagree when interpreting.
    df["pooled_ci_excludes_zero"] = (
        df["ci_lo_pooled"].notna() & df["ci_hi_pooled"].notna()
        & ~((df["ci_lo_pooled"] <= 0) & (df["ci_hi_pooled"] >= 0))
    )

    # High heterogeneity warning: when DL I^2 exceeds the threshold, the
    # per-replica slopes disagree enough that the RE meta-estimate may
    # be masking a real cell-level disagreement. Surface this in the
    # output so reviewers can audit borderline cells.
    df["replicas_disagree"] = df["I2_RE"].fillna(0.0) > high_i2_threshold

    return df


def main() -> None:
    args = parse_args()
    if args.expected_replicas <= 0:
        print("--expected-replicas must be positive", file=sys.stderr)
        sys.exit(2)

    df_trends = load_trend_inputs(args.trends_csv)

    if args.indicator:
        wanted = {x.strip() for x in args.indicator.split(",") if x.strip()}
        before = len(df_trends)
        df_trends = df_trends[df_trends["indicator"].isin(wanted)].reset_index(drop=True)
        print(f"  filter --indicator: {len(df_trends)}/{before} rows kept",
              file=sys.stderr)
        if df_trends.empty:
            print("no trend rows left after --indicator filter", file=sys.stderr)
            sys.exit(1)

    do_pooled = not args.no_pooled

    df = aggregate(df_trends, do_pooled)
    df = apply_bh_and_decision(df, args.alpha, args.expected_replicas)

    if args.csv:
        df.to_csv(sys.stdout, index=False)
        return

    n_tests = int(df["stouffer_p"].notna().sum())
    n_bh = int(df["bh_reject_cell"].sum())
    n_re_sig = int(df["RE_significant"].sum())

    n_degraded = int(df["degraded_replicas"].sum())
    n_disagree = int(df["replicas_disagree"].sum())

    print(f"\nPer-cell slope aggregation  (n_replicas typically 3)")
    print(f"BH-FDR family size: {n_tests}, alpha={args.alpha}, "
          f"expected_replicas={args.expected_replicas}")
    print(f"Primary estimator: RE-DL (95% Gaussian CI on theta_RE)")
    print(f"Robustness estimator: median of k per-replica slopes "
          f"with [min, max] exact non-parametric CI "
          f"(coverage = 1 - 2 * 0.5^k; 75% at k=3, 87.5% at k=4)")
    print("=" * 195)
    header = (
        f"{'cell':<5} {'indicator':<32} "
        f"{'n':>2}/{'kRE':<3}/{'kSt':<3} "
        f"{'slope_RE':>14} {'CI_lo_RE':>14} {'CI_hi_RE':>14} "
        f"{'I2_%':>6} {'tau2':>12} {'stouf_p':>10} {'q_BH':>10} {'sig_RE':>7} "
        f"{'slope_med':>14} {'[min,max]':>30} {'cov%':>5}"
    )
    print(header); print("-" * 195)
    for _, r in df.sort_values(["cell_id", "indicator"]).iterrows():
        ci_pooled = (
            f"[{r['ci_lo_pooled']:.4g}, {r['ci_hi_pooled']:.4g}]"
            if np.isfinite(r["ci_lo_pooled"]) else "n/a"
        )
        cov = (f"{r['pooled_coverage_pct']:.1f}"
               if np.isfinite(r.get("pooled_coverage_pct", np.nan)) else "n/a")
        sig = "YES" if r["RE_significant"] else "no"
        flags = ""
        if r["degraded_replicas"]:
            flags += "*"
        if r["replicas_disagree"]:
            flags += "!"
        sig = sig + flags
        print(f"{str(r['cell_id']):<5} {str(r['indicator']):<32} "
              f"{int(r['n_replicas']):>2}/{int(r['k_used_RE']):<3}/{int(r['k_used_stouffer']):<3} "
              f"{r['slope_RE']:>14.4g} {r['ci_lo_RE']:>14.4g} {r['ci_hi_RE']:>14.4g} "
              f"{r['I2_RE']:>6.1f} {r['tau2_RE']:>12.4g} "
              f"{r['stouffer_p']:>10.4g} {r['q_value_cell']:>10.4g} {sig:>7} "
              f"{r['slope_pooled']:>14.4g} {ci_pooled:>30} {cov:>5}")
    print("=" * 195)
    print(f"\nTotals: {n_bh}/{n_tests} cells reject BH at q={args.alpha}; "
          f"{n_re_sig}/{n_tests} RE-significant (BH AND RE CI excludes 0 AND replicas not degraded).")
    if n_degraded:
        print(f"NOTE: {n_degraded}/{n_tests} cells with degraded replicas "
              f"(n_replicas, k_used_RE, or k_used_stouffer below expected_replicas); "
              f"excluded from RE_significant and marked with '*' in the sig_RE column.")
    if n_disagree:
        print(f"NOTE: {n_disagree}/{n_tests} cells with high between-replica heterogeneity "
              f"(I^2 > 75%); marked with '!' in the sig_RE column.")
    if n_disagree:
        print(f"NOTE: {n_disagree}/{n_tests} cells with I2 > 75% (replicas_disagree); "
              f"marked with '!' in the sig_RE column. The RE meta-estimate may be "
              f"masking high between-replica heterogeneity for these cells.")


if __name__ == "__main__":
    main()
