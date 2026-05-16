"""Quick summary of a single rate-sweep run.

Usage:
    python3 analysis/sweep_summary.py <run_dir>

Where <run_dir> is a directory containing requests_*.csv files produced
by the benchmark client. Prints request counts, effective RPS, latency
percentiles, and TTFT percentiles for streaming requests.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd


def summarize(run_dir: Path) -> None:
    files = sorted(glob.glob(str(run_dir / "requests_*.csv")))
    if not files:
        print(f"No requests_*.csv files found in {run_dir}")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ok = df[df["status"] == "ok"]
    dropped = df[df["status"] == "dropped"]
    errors = df[~df["status"].isin(["ok", "dropped"])]

    print(f"Run: {run_dir}")
    print(f"Files: {len(files)}, total requests: {len(df)}")
    print(f"  OK:      {len(ok):>5d}  ({100*len(ok)/len(df):.1f}%)")
    print(f"  Dropped: {len(dropped):>5d}  ({100*len(dropped)/len(df):.1f}%)")
    print(f"  Errors:  {len(errors):>5d}  ({100*len(errors)/len(df):.1f}%)")
    print()

    if len(ok) == 0:
        print("No successful requests, nothing else to report.")
        return

    print("E2E latency (OK only):")
    print(f"  p50: {ok['e2e_latency_s'].quantile(0.5):.2f} s")
    print(f"  p95: {ok['e2e_latency_s'].quantile(0.95):.2f} s")
    print(f"  p99: {ok['e2e_latency_s'].quantile(0.99):.2f} s")
    print(f"  max: {ok['e2e_latency_s'].max():.2f} s")
    print()

    streaming = ok[ok["streaming"] == True]
    if len(streaming) > 0:
        print(f"TTFT (streaming only, n={len(streaming)}):")
        print(f"  p50: {streaming['ttft_s'].quantile(0.5):.3f} s")
        print(f"  p95: {streaming['ttft_s'].quantile(0.95):.3f} s")
        print(f"  p99: {streaming['ttft_s'].quantile(0.99):.3f} s")
        print()

    duration = df["submitted_at_unix"].max() - df["submitted_at_unix"].min()
    rps_effective = len(df) / duration if duration > 0 else 0
    rps_ok = len(ok) / duration if duration > 0 else 0
    print(f"Window: {duration:.1f} s")
    print(f"Effective RPS (all):     {rps_effective:.2f}")
    print(f"Effective RPS (OK only): {rps_ok:.2f}")

    if len(ok) > 0 and "actual_output_tokens" in ok.columns:
        out_tokens = ok["actual_output_tokens"].dropna()
        if len(out_tokens) > 0:
            tokens_per_sec = out_tokens.sum() / duration
            print(f"Output tokens/sec:       {tokens_per_sec:.1f}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1]).expanduser()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        sys.exit(1)
    summarize(run_dir)


if __name__ == "__main__":
    main()
