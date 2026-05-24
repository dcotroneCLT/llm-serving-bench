#!/usr/bin/env python3
"""Per-cell slope aggregation across n=3 replicas for the WoSAR 2026 campaign.

Two estimators side-by-side, computed per (cell_id, indicator):

  1. DerSimonian-Laird random-effects on the 3 per-replica Theil-Sen slope
     estimates. SE per replica derived from the TS exact 95% CI via the
     Gaussian-equivalent formula (ci_hi - ci_lo) / (2 * 1.96). PRIMARY
     estimator for the camera-ready tables.

  2. Pooled Theil-Sen on the concatenated post-warmup series of all 3
     replicas, with per-replica time origin reset to 0 and per-replica
     median-centering. ROBUSTNESS CHECK; does not enter the headline
     numbers.

Per-cell BH-FDR (q = alpha) is applied across the joint family of
(cell_id, indicator) tests. The per-cell p-value comes from Stouffer's
z-score combination of the 3 per-replica MK z statistics in the input
trend CSVs.

Decision rule for "RE_significant":
    (cell-level BH-FDR rejects the Stouffer-combined p) AND
    (RE 95% CI excludes 0) AND
    (n_replicas, k_used_RE, and k_used_stouffer meet --expected-replicas).

Usage:
    # Step 1: per-run trend CSVs (already exists, from aging_trends.py)
    python3 analysis/aging_trends.py --csv --run-dir <run> > /tmp/<run>_trends.csv

    # Step 2: per-cell aggregation across replicas
    python3 analysis/aggregate_slopes.py \\
        --trends-csv /tmp/wosar2026_e1_r01_trends.csv \\
        --trends-csv /tmp/wosar2026_e1_r02_trends.csv \\
        --trends-csv /tmp/wosar2026_e1_r03_trends.csv \\
        ... (18 total for full campaign) \\
        --runs-root ~/wosar/runs \\
        [--alpha 0.10] [--expected-replicas 3] [--no-pooled] [--indicator I,J,...] \\
        [--csv] > /tmp/n3_per_cell_slopes.csv

Dependencies: pandas, numpy, scipy. Reuses analysis/aging_io.py.

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

CAVEAT 3 (pooled-TS interpretation). Pooled Theil-Sen on median-centered,
concatenated series treats the 3 replicas as repeated samplings on a common
"time since warmup" axis. The CI does not incorporate between-run variance
into its width (it goes into the residual). RE-DL is the primary estimator.
If RE and pooled disagree, RE goes in the paper and the disagreement becomes
a methodological note.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Make sibling module importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aging_io import (
    discover_proc_prefix,
    load_client,
    load_manifest,
    load_proc,
    resolve_warmup,
)


TRENDS_REQUIRED_COLUMNS = [
    "run_id", "cell_id", "indicator", "n_samples",
    "slope", "slope_ci_lo", "slope_ci_hi",
    "mk_z", "mk_p_value", "lag1_rho",
]

DEFAULT_ALPHA = 0.10
DEFAULT_DOWNSAMPLE_S = 60
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


# ----------------------------- pooled Theil-Sen ------------------------------

def _load_periodic_series(
    files: list[Path], col: str, warmup_s: float, downsample_s: int,
) -> Optional[pd.DataFrame]:
    if not files:
        return None
    try:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    except Exception:
        return None
    if "ts_unix" not in df.columns or col not in df.columns:
        return None
    t0 = float(df["ts_unix"].min())
    df = df[df["ts_unix"] >= t0 + warmup_s].copy()
    if df.empty:
        return None
    df["_bin"] = (df["ts_unix"] // downsample_s).astype(np.int64)
    agg = df.groupby("_bin")[col].median().reset_index()
    if agg.empty:
        return None
    agg["ts_unix"] = agg["_bin"].astype(float) * downsample_s
    agg = agg[["ts_unix", col]].rename(columns={col: "value"})
    agg["ts_unix"] = agg["ts_unix"] - agg["ts_unix"].min()
    return agg.dropna().reset_index(drop=True)


CLIENT_INDICATORS = {
    "drop_rate", "e2e_p50", "e2e_p95", "e2e_p99",
    "ttft_p50", "ttft_p99", "tokens_per_sec",
}
CLIENT_INDICATOR_REQUIRES = {
    "drop_rate": (),  # only needs `status`
    "e2e_p50": ("e2e_latency_s",),
    "e2e_p95": ("e2e_latency_s",),
    "e2e_p99": ("e2e_latency_s",),
    "ttft_p50": ("ttft_s",),
    "ttft_p99": ("ttft_s",),
    "tokens_per_sec": ("actual_output_tokens",),
}


def _load_client_indicator(
    run_dir: Path, sub_indicator: str, warmup_s: float, downsample_s: int,
) -> Optional[pd.DataFrame]:
    if sub_indicator not in CLIENT_INDICATORS:
        print(f"  [warn] unknown client indicator {sub_indicator!r}; supported: "
              f"{sorted(CLIENT_INDICATORS)}", file=sys.stderr)
        return None
    client = load_client(run_dir)
    if client is None or client.empty:
        return None
    if "submitted_at_unix" not in client.columns or "status" not in client.columns:
        print(f"  [warn] client requests CSV in {run_dir} missing submitted_at_unix or status",
              file=sys.stderr)
        return None
    required = CLIENT_INDICATOR_REQUIRES[sub_indicator]
    missing = [c for c in required if c not in client.columns]
    if missing:
        print(f"  [warn] client requests CSV in {run_dir} missing columns "
              f"{missing} for indicator {sub_indicator!r}; skipped", file=sys.stderr)
        return None
    t0 = float(client["submitted_at_unix"].min())
    client = client[client["submitted_at_unix"] >= t0 + warmup_s].copy()
    if client.empty:
        return None
    client["_bin"] = (client["submitted_at_unix"] // downsample_s).astype(np.int64)
    needs_streaming = sub_indicator in {"ttft_p50", "ttft_p99"}
    has_streaming_col = "streaming" in client.columns
    rows = []
    for bin_id, group in client.groupby("_bin"):
        n = len(group)
        ok = group[group["status"] == "ok"]
        if sub_indicator == "drop_rate":
            dropped = group[group["status"] == "dropped"]
            value = (len(dropped) / n) if n > 0 else 0.0
        elif sub_indicator == "e2e_p50":
            value = ok["e2e_latency_s"].quantile(0.5) if len(ok) > 0 else np.nan
        elif sub_indicator == "e2e_p95":
            value = ok["e2e_latency_s"].quantile(0.95) if len(ok) > 0 else np.nan
        elif sub_indicator == "e2e_p99":
            value = ok["e2e_latency_s"].quantile(0.99) if len(ok) > 0 else np.nan
        elif sub_indicator == "tokens_per_sec":
            value = ok["actual_output_tokens"].sum() / downsample_s if len(ok) > 0 else np.nan
        elif needs_streaming:
            if not has_streaming_col:
                value = np.nan
            else:
                streaming = ok[ok["streaming"] == True]  # noqa: E712 (pandas comparison)
                if len(streaming) == 0:
                    value = np.nan
                elif sub_indicator == "ttft_p50":
                    value = streaming["ttft_s"].quantile(0.5)
                else:
                    value = streaming["ttft_s"].quantile(0.99)
        else:
            return None  # unreachable; covered by CLIENT_INDICATORS guard above
        rows.append({"ts_unix": float(bin_id) * downsample_s, "value": value})
    if not rows:
        return None
    out = pd.DataFrame(rows).dropna().reset_index(drop=True)
    if out.empty:
        return None
    out["ts_unix"] = out["ts_unix"] - out["ts_unix"].min()
    return out


def _gpu_files_for_run(run_dir: Path) -> list[Path]:
    """Return GPU monitor CSVs for this run.

    Campaign runs monitor exactly one GPU, but the file prefix is the physical
    index (gpu0_*, gpu1_*, gpu2_*), not always gpu0_*.
    """
    manifest = load_manifest(run_dir)
    gpu_index = None
    engine = manifest.get("engine")
    if isinstance(engine, dict) and engine.get("gpu_device") is not None:
        gpu_index = engine.get("gpu_device")
    else:
        host = manifest.get("host")
        if isinstance(host, dict):
            gpu = host.get("gpu")
            if isinstance(gpu, dict) and gpu.get("index") is not None:
                gpu_index = gpu.get("index")

    if gpu_index is not None:
        try:
            files = sorted(run_dir.glob(f"gpu{int(gpu_index)}_*.csv"))
            if files:
                return files
        except (TypeError, ValueError):
            print(f"  [warn] manifest has non-integer gpu index {gpu_index!r} in {run_dir}; "
                  f"falling back to gpu[0-9]*_*.csv",
                  file=sys.stderr)
    return sorted(run_dir.glob("gpu[0-9]*_*.csv"))


def load_series_for_run(
    run_dir: Path, indicator: str, warmup_s: float, downsample_s: int = DEFAULT_DOWNSAMPLE_S,
) -> Optional[pd.DataFrame]:
    """Load post-warmup, 60s-downsampled series for one (run, indicator).

    Returns a DataFrame with columns [ts_unix, value], ts_unix in seconds
    starting at 0. Returns None on missing data.
    """
    if indicator.startswith("gpu."):
        col = indicator[len("gpu."):]
        files = _gpu_files_for_run(run_dir)
        return _load_periodic_series(files, col, warmup_s, downsample_s)
    if indicator.startswith("system."):
        col = indicator[len("system."):]
        files = sorted(run_dir.glob("system_*.csv"))
        return _load_periodic_series(files, col, warmup_s, downsample_s)
    if indicator.startswith("proc."):
        col = indicator[len("proc."):]
        manifest = load_manifest(run_dir)
        prefix = discover_proc_prefix(run_dir, manifest)
        if prefix is None:
            return None
        # aging_io.load_proc auto-adds ts_unix + process_alive to whatever
        # columns are requested AND applies the process_alive truthy filter
        # internally. Pass [col] explicitly because columns=None would skip
        # the target indicator.
        df = load_proc(run_dir, prefix, columns=[col])
        if df is None or df.empty:
            return None
        if "ts_unix" not in df.columns or col not in df.columns:
            return None
        t0 = float(df["ts_unix"].min())
        df = df[df["ts_unix"] >= t0 + warmup_s].copy()
        if df.empty:
            return None
        df["_bin"] = (df["ts_unix"] // downsample_s).astype(np.int64)
        agg = df.groupby("_bin")[col].median().reset_index()
        if agg.empty:
            return None
        agg["ts_unix"] = agg["_bin"].astype(float) * downsample_s
        agg = agg[["ts_unix", col]].rename(columns={col: "value"})
        agg["ts_unix"] = agg["ts_unix"] - agg["ts_unix"].min()
        return agg.dropna().reset_index(drop=True)
    if indicator.startswith("client."):
        sub = indicator[len("client."):]
        return _load_client_indicator(run_dir, sub, warmup_s, downsample_s)
    return None


def _estimate_lag1_autocorr(y: np.ndarray) -> float:
    """Lag-1 autocorrelation clipped to [0, 0.99]; mirror of aging_trends.py."""
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


def _ar1_variance_inflation(rho: float) -> float:
    rho = max(0.0, min(0.99, rho))
    return (1.0 + rho) / (1.0 - rho)


def _sen_slope_and_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05, ar_correction: float = 1.0,
) -> tuple[float, float, float]:
    """Sen point slope + order-statistics 95% CI with AR(1)-inflated MK variance.

    Same convention as aging_trends.sen_slope_and_ci. Re-implemented locally
    so aggregate_slopes.py does not depend on aging_trends.py (which has a
    hard pymannkendall import at module load).
    """
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
    median_slope = float(np.median(slopes_sorted))

    var_S = n * (n - 1) * (2 * n + 5) / 18.0 * float(ar_correction)
    z = stats.norm.ppf(1 - alpha / 2)
    C_alpha = z * np.sqrt(var_S)

    L = int(np.floor((M - C_alpha) / 2))
    U = int(np.ceil((M + C_alpha) / 2)) + 1
    L = max(0, L)
    U = min(M - 1, U)
    return median_slope, float(slopes_sorted[L]), float(slopes_sorted[U])


def pooled_theil_sen(
    run_dirs: list[Path], indicator: str, warmup_s_per_run: list[float],
    downsample_s: int = DEFAULT_DOWNSAMPLE_S,
) -> dict:
    """Theil-Sen on the concatenation of k post-warmup series.

    Per-replica processing:
      - time origin reset to 0, then OFFSET by (i * T_pad) where T_pad is
        ~2x the longest per-replica duration. Cross-replica pairs then
        span Δx >= T_pad, which avoids the near-collinear-x pathology
        of overlapping replicas (a pair (y_b[t], y_a[t]) with t_a≈t_b
        gives an undefined or huge-magnitude slope estimate).
      - value centered by subtracting the per-replica MEDIAN to remove
        the intercept-difference contribution from cross-replica pairs.

    The CI uses the order-statistics Sen formula with the MK null
    variance multiplied by the AR(1) inflation factor (1+rho)/(1-rho).
    rho is averaged across per-replica within-replica lag-1
    autocorrelations. This makes the pooled CI methodologically
    consistent with the per-replica CIs in aging_trends.py.

    Caveat: this is still a robustness check, not the primary estimator.
    Between-run heterogeneity goes into the residual rather than into
    the CI width. Use RE-DL as the primary.

    slope and CI are reported per HOUR (consistent with aging_trends.py).
    """
    xs, ys, rhos = [], [], []
    durations = []
    for run_dir, warmup_s in zip(run_dirs, warmup_s_per_run):
        s = load_series_for_run(run_dir, indicator, warmup_s, downsample_s)
        if s is None or s.empty:
            continue
        x_r = s["ts_unix"].values.astype(float)
        y_r = s["value"].values.astype(float)
        y_r = y_r - float(np.median(y_r))
        rhos.append(_estimate_lag1_autocorr(y_r))
        durations.append(float(x_r.max() - x_r.min()) if len(x_r) else 0.0)
        xs.append(x_r)
        ys.append(y_r)
    k_used = len(xs)
    if k_used == 0:
        return dict(slope=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_total=0, k_used=0)

    # Offset each replica's x by 2x the longest replica's duration so
    # cross-replica pairs span large, non-degenerate Δx.
    t_pad = 2.0 * max(durations) if durations else 0.0
    xs_off = [x_r + i * t_pad for i, x_r in enumerate(xs)]
    x = np.concatenate(xs_off)
    y = np.concatenate(ys)
    if len(x) < 10:
        return dict(slope=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_total=0, k_used=k_used)
    rho_pooled = float(np.mean(rhos)) if rhos else 0.0
    ar_mult = _ar1_variance_inflation(rho_pooled)
    sl, lo, hi = _sen_slope_and_ci(x, y, alpha=0.05, ar_correction=ar_mult)
    return dict(slope=float(sl) * 3600.0, ci_lo=float(lo) * 3600.0,
                ci_hi=float(hi) * 3600.0, n_total=int(len(x)), k_used=k_used)


# ----------------------------- main ------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trends-csv", type=Path, action="append", required=True,
                    help="CSV produced by aging_trends.py --csv. Pass once per run.")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Root directory containing wosar2026_<cell>_r<NN> subdirs. "
                         "Required if pooled-TS is requested (default). Warmup is "
                         "auto-resolved per-run via aging_io.resolve_warmup, which "
                         "reads the manifest and the cell yaml from the repo's "
                         "campaigns/<campaign_id>/cells/<cell>.yaml convention.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help=f"BH-FDR target q (default {DEFAULT_ALPHA}).")
    ap.add_argument("--indicator", type=str, default=None,
                    help="Comma-separated indicator filter (e.g. proc.uss_bytes,"
                         "proc.rss_bytes). Default: all indicators present in inputs.")
    ap.add_argument("--no-pooled", action="store_true",
                    help="Skip the pooled Theil-Sen estimator (RE-DL only).")
    ap.add_argument("--downsample-seconds", type=int, default=DEFAULT_DOWNSAMPLE_S,
                    help=f"Window for pooled-TS downsampling (default {DEFAULT_DOWNSAMPLE_S}s).")
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


def resolve_run_dirs(
    df: pd.DataFrame, runs_root: Optional[Path]
) -> dict[str, Path]:
    if runs_root is None:
        return {}
    mapping = {}
    for run_id in df["run_id"].unique():
        candidate = runs_root / run_id
        if candidate.is_dir():
            mapping[run_id] = candidate
        else:
            print(f"  [warn] runs-root: no directory {candidate} for run {run_id}",
                  file=sys.stderr)
    return mapping


def aggregate(df_trends: pd.DataFrame, run_dirs: dict[str, Path],
              do_pooled: bool, downsample_s: int) -> pd.DataFrame:
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

        if do_pooled and run_dirs:
            warmup_s_list = []
            dirs_list = []
            for rid in g["run_id"].tolist():
                if rid in run_dirs:
                    rd = run_dirs[rid]
                    dirs_list.append(rd)
                    warmup_s_list.append(float(resolve_warmup(rd)))
            pooled = pooled_theil_sen(dirs_list, indicator, warmup_s_list, downsample_s)
            row.update({
                "slope_pooled": pooled["slope"],
                "ci_lo_pooled": pooled["ci_lo"],
                "ci_hi_pooled": pooled["ci_hi"],
                "n_total_pooled": pooled["n_total"],
                "k_used_pooled": pooled["k_used"],
            })
            # agreement: relative gap between RE and pooled
            if np.isfinite(dl["theta_RE"]) and np.isfinite(pooled["slope"]) and dl["theta_RE"] != 0:
                row["agreement_pct"] = float(
                    100.0 * abs(dl["theta_RE"] - pooled["slope"]) / abs(dl["theta_RE"])
                )
            else:
                row["agreement_pct"] = np.nan
        else:
            row.update(dict(slope_pooled=np.nan, ci_lo_pooled=np.nan, ci_hi_pooled=np.nan,
                            n_total_pooled=0, k_used_pooled=0, agreement_pct=np.nan))

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

    do_pooled = not args.no_pooled and args.runs_root is not None
    run_dirs = resolve_run_dirs(df_trends, args.runs_root) if do_pooled else {}

    df = aggregate(df_trends, run_dirs, do_pooled, args.downsample_seconds)
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
    print("=" * 180)
    header = (
        f"{'cell':<5} {'indicator':<32} "
        f"{'n':>2}/{'kRE':<3}/{'kSt':<3} "
        f"{'slope_RE':>14} {'CI_lo_RE':>14} {'CI_hi_RE':>14} "
        f"{'I2_%':>6} {'tau2':>12} {'stouf_p':>10} {'q_BH':>10} {'sig_RE':>7} "
        f"{'slope_pooled':>14} {'CI_pooled':>30}"
    )
    print(header); print("-" * 180)
    for _, r in df.sort_values(["cell_id", "indicator"]).iterrows():
        ci_pooled = (
            f"[{r['ci_lo_pooled']:.4g}, {r['ci_hi_pooled']:.4g}]"
            if np.isfinite(r["ci_lo_pooled"]) else "n/a"
        )
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
              f"{r['slope_pooled']:>14.4g} {ci_pooled:>30}")
    print("=" * 180)
    print(f"\nTotals: {n_bh}/{n_tests} cells reject BH at q={args.alpha}; "
          f"{n_re_sig}/{n_tests} RE-significant (BH AND RE CI excludes 0).")
    if n_degraded:
        print(f"WARNING: {n_degraded}/{n_tests} cells with degraded replicas "
              f"(n_replicas, k_used_RE, or k_used_stouffer below expected_replicas); "
              f"excluded from RE_significant and marked with '*' in the sig_RE column.")
    if n_disagree:
        print(f"NOTE: {n_disagree}/{n_tests} cells with I2 > 75% (replicas_disagree); "
              f"marked with '!' in the sig_RE column. The RE meta-estimate may be "
              f"masking high between-replica heterogeneity for these cells.")


if __name__ == "__main__":
    main()
