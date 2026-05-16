#!/usr/bin/env python3
"""Aggregate results across all rate levels of a sweep and identify the
saturation point.

Reads every `client_NNrps/` subdirectory under the given parent and
produces a single table with: target rate, effective RPS, drop rate,
E2E latency p50/p95/p99, TTFT p99, output tokens/s.

Then identifies the "knee" of the saturation curve as the highest
target RPS that meets all of:
  - drop rate below --drop-threshold (default 5%)
  - effective RPS within --rps-tolerance of target (default 10%)
  - E2E p99 below --p99-threshold seconds (default 60)

The 85% target rate for aging runs is reported as a recommendation.

Example:
    python3 analysis/sweep_curve.py ~/wosar/runs/pilot_vllm_sweep_v2

Optional flags let you tighten or loosen the knee criteria.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def quantile_or_nan(df: pd.DataFrame, col: str, q: float) -> float:
    if col not in df.columns or df.empty:
        return float("nan")
    return float(df[col].quantile(q))


def truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def target_rps_from_dir(level_dir: Path) -> float:
    """Extract target rate from names like client_04rps or client_2.5rps."""
    m = re.search(r"client_([0-9]+(?:\.[0-9]+)?)rps", level_dir.name)
    return float(m.group(1)) if m else float("nan")


def load_level(level_dir: Path) -> Optional[dict]:
    """Compute aggregate metrics for a single level directory."""
    files = sorted(glob.glob(str(level_dir / "requests_*.csv")))
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
    if df.empty:
        return None
    if "submitted_at_unix" not in df.columns or "status" not in df.columns:
        print(f"  [warn] missing required columns in {level_dir}", file=sys.stderr)
        return None

    ok = df[df["status"].isin(["ok", "success"])]
    dropped = df[df["status"] == "dropped"]
    duration = df["submitted_at_unix"].max() - df["submitted_at_unix"].min()
    if duration <= 0:
        return None

    target_rps = target_rps_from_dir(level_dir)

    if "streaming" in ok.columns:
        streaming = ok[truthy_series(ok["streaming"])]
    else:
        streaming = ok.iloc[0:0]
    out_tokens = (
        ok["actual_output_tokens"].dropna()
        if "actual_output_tokens" in ok.columns
        else pd.Series(dtype=float)
    )

    return {
        "level_dir": level_dir.name,
        "target_rps": target_rps,
        "n_requests": len(df),
        "n_ok": len(ok),
        "n_dropped": len(dropped),
        "drop_pct": 100.0 * len(dropped) / len(df) if len(df) > 0 else 0.0,
        "effective_rps_all": len(df) / duration,
        "effective_rps_ok": len(ok) / duration,
        "e2e_p50": quantile_or_nan(ok, "e2e_latency_s", 0.5),
        "e2e_p95": quantile_or_nan(ok, "e2e_latency_s", 0.95),
        "e2e_p99": quantile_or_nan(ok, "e2e_latency_s", 0.99),
        "ttft_p50": quantile_or_nan(streaming, "ttft_s", 0.5),
        "ttft_p99": quantile_or_nan(streaming, "ttft_s", 0.99),
        "tokens_per_sec": out_tokens.sum() / duration if len(out_tokens) > 0 else float("nan"),
        "duration_s": duration,
    }


def find_knee(rows: list[dict], drop_threshold: float, rps_tolerance: float, p99_threshold: float) -> Optional[dict]:
    """Return the highest-target-RPS row that meets all criteria."""
    candidates = []
    for r in rows:
        if pd.isna(r["target_rps"]) or pd.isna(r["e2e_p99"]):
            continue
        if r["drop_pct"] >= drop_threshold * 100:
            continue
        if r["target_rps"] > 0 and r["effective_rps_all"] / r["target_rps"] < (1 - rps_tolerance):
            continue
        if r["e2e_p99"] > p99_threshold:
            continue
        candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["target_rps"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parent", type=Path,
                   help="Parent directory containing client_NNrps subdirs.")
    p.add_argument("--drop-threshold", type=float, default=0.05,
                   help="Maximum acceptable drop fraction at the knee (default 0.05).")
    p.add_argument("--rps-tolerance", type=float, default=0.10,
                   help="Tolerated relative gap between target and realized RPS (default 0.10).")
    p.add_argument("--p99-threshold", type=float, default=60.0,
                   help="Maximum acceptable E2E p99 latency in seconds (default 60).")
    p.add_argument("--target-fraction", type=float, default=0.85,
                   help="Fraction of saturation RPS to recommend for aging runs (default 0.85).")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Optional path to save the aggregate sweep table as CSV.")
    args = p.parse_args()
    args.parent = args.parent.expanduser()

    if not args.parent.is_dir():
        print(f"Not a directory: {args.parent}", file=sys.stderr)
        sys.exit(1)

    level_dirs = sorted(args.parent.glob("client_*rps"))
    if not level_dirs:
        print(f"No client_*rps subdirectories under {args.parent}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for d in level_dirs:
        row = load_level(d)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No usable data in any level.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (pd.isna(r["target_rps"]), r["target_rps"]))

    # Print table
    cols = [
        ("target_rps", "Target", "{:>6.1f}"),
        ("effective_rps_all", "EffAll", "{:>6.2f}"),
        ("effective_rps_ok", "EffOK", "{:>5.2f}"),
        ("drop_pct", "Drop%", "{:>5.1f}"),
        ("e2e_p50", "p50", "{:>6.2f}"),
        ("e2e_p95", "p95", "{:>6.2f}"),
        ("e2e_p99", "p99", "{:>6.2f}"),
        ("ttft_p99", "TTFTp99", "{:>7.2f}"),
        ("tokens_per_sec", "Tok/s", "{:>6.0f}"),
        ("n_requests", "N", "{:>5d}"),
    ]
    print(f"\nSweep results: {args.parent}")
    print("=" * 120)
    print("  ".join(c[1].rjust(7) for c in cols))
    print("-" * 120)
    for row in rows:
        line = "  ".join(c[2].format(row.get(c[0], 0)).rjust(7) for c in cols)
        print(line)
    print("=" * 120)

    if args.output_csv is not None:
        args.output_csv = args.output_csv.expanduser()
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved sweep table: {args.output_csv}")

    # Identify knee
    knee = find_knee(rows, args.drop_threshold, args.rps_tolerance, args.p99_threshold)
    print()
    print("Knee identification criteria:")
    print(f"  drop rate < {args.drop_threshold*100:.1f}%")
    print(f"  effective_rps_all >= {(1-args.rps_tolerance)*100:.0f}% of target")
    print(f"  E2E p99 < {args.p99_threshold:.1f} s")
    print()
    if knee is None:
        print("  No level satisfies the criteria. Saturation point may be below the lowest")
        print("  tested rate, or the criteria are too strict for this run.")
        sys.exit(0)

    knee_rps = knee["target_rps"]
    knee_effective = knee["effective_rps_all"]
    target = args.target_fraction * knee_effective
    print(f"Saturation knee:")
    print(f"  Last sustainable target: {knee_rps:.1f} RPS")
    print(f"  Effective at that level: {knee_effective:.2f} RPS")
    print(f"  Drop at that level:      {knee['drop_pct']:.1f}%")
    print(f"  E2E p99 at that level:   {knee['e2e_p99']:.2f} s")
    print()
    print(f"Recommended target rate for aging runs ({args.target_fraction*100:.0f}% of effective): "
          f"{target:.2f} RPS")
    print()
    print("Use this value as `target_rate_rps` in the aging run config.")


if __name__ == "__main__":
    main()
