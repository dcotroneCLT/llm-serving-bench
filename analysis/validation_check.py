"""Validation check for a single (cell, replica) run.

Reads the run directory produced by scripts/launch_cell.py and produces
a one-page verdict on whether the framework is sane:

  1. All expected output files present and non-empty.
  2. Monitoring window has the expected duration (within tolerance).
  3. RSS time series (process memory of the engine worker):
       - autocorrelation-aware Mann-Kendall trend test (Hamed-Rao)
       - Theil-Sen slope estimate
     Acceptance: p < 0.01 AND slope > 0 (positive trend).
  4. Client request log has > 0 successful responses.
  5. Container teardown was clean (manifest reports interrupted=False).

Usage:

  python analysis/validation_check.py \\
      --run-dir /tmp/wosar_validation/validation_e1_r99

Outputs a multi-line verdict to stdout and exits with rc=0 if all
checks pass, rc=1 if any soft check fails (informational), rc=2 if
a hard check fails (data integrity, framework bug).

The slope is reported in MB/h (matching the paper convention) but NOT
compared against the n=1 baseline (+9.15 MB/h for E1). The vLLM V1
image digest drifted between the n=1 prototype and this campaign;
the absolute slope value is a property of the new pin, not the
framework.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk


HARD_FAIL = 2
SOFT_FAIL = 1
OK = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def find_proc_csvs(run_dir: Path) -> list[Path]:
    """Return rotated proc_monitor CSVs sorted by index.

    Filename pattern: <label>_NNNNNN.csv (e.g. vllm_v1_standalone_000000.csv).
    Excludes gpu_*, system_*, client/, logs/, and other non-proc files.
    """
    out = []
    for p in run_dir.glob("*.csv"):
        name = p.name
        if name.startswith("gpu") or name.startswith("system") or name.startswith("client"):
            continue
        # Must end in _NNNNNN.csv
        stem = name[:-4]
        if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
            out.append(p)
    return sorted(out)


def find_gpu_csvs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("gpu*_*.csv"))


def find_system_csvs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("system_*.csv"))


def find_client_csvs(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "client").glob("requests_*.csv"))


def load_proc_concat(proc_csvs: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in proc_csvs:
        try:
            dfs.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def rss_slope_mb_per_h(rss_bytes: np.ndarray, ts_unix: np.ndarray, discard_warmup_s: int) -> dict:
    """Compute MK trend test and Theil-Sen slope on RSS time series.

    Returns dict with: n, n_used, warmup_discarded, p_value, slope_mb_per_h.
    Slope in MB / hour (positive = leak).
    """
    if len(rss_bytes) < 10:
        return {"error": f"insufficient samples: {len(rss_bytes)}"}

    # Discard warmup
    t0 = ts_unix.min()
    mask = ts_unix >= (t0 + discard_warmup_s)
    rss_post = rss_bytes[mask]
    ts_post = ts_unix[mask]

    if len(rss_post) < 10:
        return {"error": f"insufficient samples after warmup discard: {len(rss_post)}"}

    # Modified MK with Hamed-Rao correction for autocorrelation.
    # Trend test answers: is there a monotonic trend?
    try:
        mk_result = mk.hamed_rao_modification_test(rss_post)
    except Exception as e:
        return {"error": f"mann-kendall failed: {e}"}

    # Theil-Sen slope: bytes per second (because x = ts_unix in seconds).
    # Convert to MB/h: bytes/s * 3600 / 1e6
    slope_bytes_per_s = mk_result.slope
    slope_mb_per_h = slope_bytes_per_s * 3600.0 / 1e6

    return {
        "n_total": len(rss_bytes),
        "n_used": len(rss_post),
        "warmup_discarded_s": discard_warmup_s,
        "p_value": float(mk_result.p),
        "z_statistic": float(mk_result.z),
        "trend": mk_result.trend,        # 'increasing', 'decreasing', 'no trend'
        "slope_mb_per_h": float(slope_mb_per_h),
        "intercept_bytes": float(mk_result.intercept),
    }


def check(run_dir: Path) -> int:
    log(f"=== Validation check: {run_dir} ===")
    rc = OK

    # 1. Manifest present and parseable
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        log(f"[HARD] manifest.json missing")
        return HARD_FAIL
    manifest = json.loads(manifest_path.read_text())
    log(f"run_id          : {manifest.get('run_id', '?')}")
    log(f"cell_id         : {manifest.get('cell_id', '?')}")
    log(f"replica         : {manifest.get('replica', '?')}")
    log(f"image digest    : {manifest.get('image', {}).get('digest', '?')}")
    log(f"started_at      : {manifest.get('started_at', '?')}")
    log(f"ended_at        : {manifest.get('ended_at', '?')}")
    duration_actual = manifest.get("duration_seconds_actual")
    duration_target = manifest.get("duration_s")
    log(f"duration target : {duration_target}s")
    log(f"duration actual : {duration_actual:.0f}s" if duration_actual else "duration actual : MISSING")
    interrupted = manifest.get("interrupted_early", True)
    log(f"interrupted     : {interrupted}")

    # 2. Duration consistency
    if duration_actual is None:
        log(f"[HARD] manifest missing duration_seconds_actual (run did not finalize)")
        rc = max(rc, HARD_FAIL)
    elif duration_target and abs(duration_actual - duration_target) > duration_target * 0.05:
        log(f"[SOFT] duration deviates by > 5% from target")
        rc = max(rc, SOFT_FAIL)

    # 3. Output files present and non-empty
    proc_csvs = find_proc_csvs(run_dir)
    gpu_csvs = find_gpu_csvs(run_dir)
    sys_csvs = find_system_csvs(run_dir)
    client_csvs = find_client_csvs(run_dir)
    log(f"proc CSVs       : {len(proc_csvs)}")
    log(f"gpu CSVs        : {len(gpu_csvs)}")
    log(f"system CSVs     : {len(sys_csvs)}")
    log(f"client CSVs     : {len(client_csvs)}")

    if not proc_csvs:
        log(f"[HARD] no proc_monitor CSV files")
        rc = max(rc, HARD_FAIL)
    if not gpu_csvs:
        log(f"[HARD] no gpu_monitor CSV files")
        rc = max(rc, HARD_FAIL)
    if not sys_csvs:
        log(f"[HARD] no system_monitor CSV files")
        rc = max(rc, HARD_FAIL)
    if not client_csvs:
        log(f"[HARD] no client requests CSV files")
        rc = max(rc, HARD_FAIL)

    if rc >= HARD_FAIL:
        return rc

    # 4. Process alive throughout
    proc_df = load_proc_concat(proc_csvs)
    if proc_df.empty:
        log(f"[HARD] proc_monitor concat empty")
        return HARD_FAIL

    if "process_alive" in proc_df.columns:
        alive_frac = float(proc_df["process_alive"].astype(bool).mean())
        log(f"proc alive frac : {alive_frac:.3f}")
        if alive_frac < 0.95:
            log(f"[HARD] proc_monitor reports < 95% samples with process_alive=True")
            rc = max(rc, HARD_FAIL)

    # 5. RSS slope test (the main aging signature)
    rss_col = None
    for cand in ["rss_bytes", "rss", "uss_bytes", "uss", "pss_bytes", "pss"]:
        if cand in proc_df.columns:
            rss_col = cand
            break
    if rss_col is None:
        log(f"[HARD] no RSS-like column in proc CSV; columns: {list(proc_df.columns)}")
        return HARD_FAIL

    ts_col = None
    for cand in ["ts_unix", "_wall_clock_unix", "timestamp"]:
        if cand in proc_df.columns:
            ts_col = cand
            break
    if ts_col is None:
        log(f"[HARD] no timestamp column in proc CSV")
        return HARD_FAIL

    # Filter to live samples only
    if "process_alive" in proc_df.columns:
        df = proc_df[proc_df["process_alive"].astype(bool)].copy()
    else:
        df = proc_df.copy()
    df = df.dropna(subset=[rss_col, ts_col])
    df[rss_col] = df[rss_col].astype(float)
    df[ts_col] = df[ts_col].astype(float)
    df = df.sort_values(ts_col)

    # Decide warmup discard. If the run is < 2h, discard 30 min instead of 1h
    # so we have enough samples for MK.
    total_duration = df[ts_col].max() - df[ts_col].min()
    warmup = 3600 if total_duration > 7200 else 1800

    log(f"--- RSS slope test on column '{rss_col}' (timestamp: {ts_col}) ---")
    result = rss_slope_mb_per_h(df[rss_col].values, df[ts_col].values, warmup)
    if "error" in result:
        log(f"[HARD] RSS slope test failed: {result['error']}")
        return HARD_FAIL

    log(f"n samples total : {result['n_total']}")
    log(f"n samples used  : {result['n_used']}  (after {result['warmup_discarded_s']}s warmup discard)")
    log(f"MK trend        : {result['trend']}")
    log(f"MK z statistic  : {result['z_statistic']:.3f}")
    log(f"MK p-value      : {result['p_value']:.4g}")
    log(f"Theil-Sen slope : {result['slope_mb_per_h']:+.3f} MB/h")

    # Acceptance criteria
    if result["p_value"] >= 0.01:
        log(f"[SOFT] MK p-value {result['p_value']:.4g} >= 0.01 (no significant trend at 1% level)")
        rc = max(rc, SOFT_FAIL)
    if result["slope_mb_per_h"] <= 0:
        log(f"[SOFT] slope not strictly positive ({result['slope_mb_per_h']:.3f} MB/h)")
        rc = max(rc, SOFT_FAIL)

    # 6. Client log: check we got real responses
    client_df = pd.concat(
        [pd.read_csv(p) for p in client_csvs if p.stat().st_size > 0],
        ignore_index=True,
    ) if client_csvs else pd.DataFrame()
    if not client_df.empty:
        log(f"--- Client ---")
        log(f"client rows     : {len(client_df)}")
        if "status" in client_df.columns:
            status_counts = client_df["status"].value_counts().to_dict()
            log(f"status breakdown: {status_counts}")
            ok_count = status_counts.get("ok", 0) + status_counts.get("success", 0)
            if ok_count == 0:
                log(f"[HARD] no successful client responses")
                rc = max(rc, HARD_FAIL)

    # 7. Interrupted flag
    if interrupted:
        log(f"[SOFT] run was marked interrupted_early=True (likely SIGINT/SIGTERM)")
        rc = max(rc, SOFT_FAIL)

    # Verdict
    log(f"--- Verdict ---")
    if rc == OK:
        log(f"PASS (rc=0): framework integrity OK, RSS shows positive trend with p<0.01")
    elif rc == SOFT_FAIL:
        log(f"SOFT FAIL (rc=1): framework runs to completion but some acceptance criteria are weak. Inspect manually.")
    else:
        log(f"HARD FAIL (rc=2): framework or data integrity issue. Do not proceed to campaign.")
    return rc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    rc = check(args.run_dir)
    sys.exit(rc)


if __name__ == "__main__":
    main()
