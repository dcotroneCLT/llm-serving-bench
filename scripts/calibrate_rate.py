#!/usr/bin/env python3
"""Per-cell rate calibration: find the sustainable throughput ceiling.

This calibrates ONE cell (one fixed workload combination: prompt-len,
output-len, prefix-repeat, burstiness) by sweeping ONLY the request rate
against an already-running endpoint, then writes the measured ceiling and
the calibrated steady-state rate (fraction x ceiling) to a JSON file that
launch_cell.py records in the run manifest.

It attaches to a running endpoint and does NOT manage the engine. Because
the five DoW factors are all workload-side, one engine instance per system
serves every cell, so the intended batch usage is "bring the engine up
once, then call this once per cell against the same endpoint" (the load-once
optimization that keeps the calibration budget small).

Conservative ceiling (deliberately NOT the 0.95 saturation edge, so that
85% of ceiling keeps real margin over a 48h run): the highest swept offered
rate for which ALL hold within the measurement window:
  - achieved/offered >= achieved_ratio_min (default 0.98)
  - drop rate <= drop_max (default 0.02)
  - e2e p99 <= p99_bound seconds (a generous absolute bound, per cell)
  - backlog is flat: mean e2e in the second half of the window is not more
    than climb_frac (default 0.20) above the first half (a rising latency
    within a fixed-rate window means the server is accumulating backlog and
    the rate is already unsustainable).
The ceiling is the top of the contiguous stable prefix (ascending sweep);
the first unstable rate is the knee. rate_calibrated = fraction x achieved
throughput at the ceiling.

Calibrate at t0 and FIX the rate for the whole run. Capacity erosion over
48h is a result to observe, not to absorb by re-calibrating mid-run.

Usage (single cell against a running endpoint):
  python3 scripts/calibrate_rate.py \
    --config <materialized client_config.yaml> \
    --base-url http://localhost:8100 --protocol vllm_openai --model Qwen/... \
    --rates 0.5,1,2,4,8 --window-seconds 240 \
    --fraction 0.85 --cell-id e1 --system vllm_standalone \
    --output calibration_e1.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pure ceiling-selection logic (unit-tested without a server)
# ---------------------------------------------------------------------------


def select_ceiling(
    rows: list[dict],
    achieved_ratio_min: float = 0.98,
    drop_max: float = 0.02,
    p99_bound: float = 60.0,
    climb_frac: float = 0.20,
) -> tuple[Optional[dict], str]:
    """Pick the conservative ceiling from ascending-rate sweep rows.

    Each row must have: offered_rate, achieved_rps, achieved_ratio,
    drop_rate, p99_e2e_s, latency_climb_frac. Returns (ceiling_row, status).
    status in {"ok", "no_stable_point", "did_not_saturate"}.
    """
    ordered = sorted(rows, key=lambda r: r["offered_rate"])

    def stable(r: dict) -> bool:
        return (
            r["achieved_ratio"] >= achieved_ratio_min
            and r["drop_rate"] <= drop_max
            and r["p99_e2e_s"] <= p99_bound
            and r["latency_climb_frac"] <= climb_frac
        )

    # Top of the contiguous stable prefix from the lowest rate upward.
    ceiling = None
    for r in ordered:
        if stable(r):
            ceiling = r
        else:
            break
    if ceiling is None:
        return None, "no_stable_point"
    if ceiling is ordered[-1]:
        # Every swept rate stayed stable: the sweep never reached the knee,
        # so this ceiling is a lower bound. Caller should extend the rates.
        return ceiling, "did_not_saturate"
    return ceiling, "ok"


def window_stats_from_csvs(sub: Path, window_s: float) -> dict:
    """Aggregate one rate window's per-request CSVs into summary stats."""
    n_total = n_ok = n_dropped = 0
    e2e_with_ts: list[tuple[float, float]] = []  # (submitted_at_unix, e2e_s)
    for f in sorted(glob.glob(str(sub / "requests_*.csv"))):
        with open(f, newline="") as fp:
            for r in csv.DictReader(fp):
                n_total += 1
                status = (r.get("status") or "").strip().lower()
                if status in ("ok", "success"):
                    n_ok += 1
                    try:
                        e2e_with_ts.append((float(r["submitted_at_unix"]), float(r["e2e_latency_s"])))
                    except (ValueError, KeyError, TypeError):
                        pass
                elif status == "dropped":
                    n_dropped += 1
    achieved = n_ok / window_s if window_s > 0 else 0.0
    e2e_vals = sorted(v for _, v in e2e_with_ts)
    p50 = _quantile(e2e_vals, 0.50)
    p95 = _quantile(e2e_vals, 0.95)
    p99 = _quantile(e2e_vals, 0.99)
    climb = _latency_climb_frac(e2e_with_ts)
    return {
        "n_total": n_total, "n_ok": n_ok, "n_dropped": n_dropped,
        "achieved_rps": achieved,
        "drop_rate": (n_dropped / n_total) if n_total > 0 else 0.0,
        "p50_e2e_s": p50, "p95_e2e_s": p95, "p99_e2e_s": p99,
        "latency_climb_frac": climb,
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[idx]


def _latency_climb_frac(e2e_with_ts: list[tuple[float, float]]) -> float:
    """Mean e2e in the second half of the window vs the first half.

    Backlog proxy: a positive value means latency rose within the
    fixed-rate window, i.e. the server is accumulating work.
    """
    if len(e2e_with_ts) < 8:
        return 0.0
    pts = sorted(e2e_with_ts, key=lambda x: x[0])
    mid_t = (pts[0][0] + pts[-1][0]) / 2.0
    first = [v for t, v in pts if t <= mid_t]
    second = [v for t, v in pts if t > mid_t]
    if not first or not second:
        return 0.0
    m1 = statistics.mean(first)
    m2 = statistics.mean(second)
    if m1 <= 0:
        return 0.0
    return (m2 - m1) / m1


# ---------------------------------------------------------------------------
# Sweep driver (runs the real client against a running endpoint)
# ---------------------------------------------------------------------------


def run_one_rate(
    repo_root: Path, config: Path, base_url: str, protocol: str, model: str,
    rate: float, window_s: int, concurrency_cap: int, sub: Path,
) -> dict:
    sub.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(repo_root / "client" / "run_client.py"),
        "--config", str(config),
        "--output-dir", str(sub),
        "--duration-seconds", str(window_s),
        "--protocol", protocol,
        "--base-url", base_url,
        "--model", model,
        "--target-rate-rps", str(rate),
        "--concurrency-cap", str(concurrency_cap),
    ]
    with (sub / "client.log").open("wb") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    stats = window_stats_from_csvs(sub, float(window_s))
    stats["offered_rate"] = rate
    stats["achieved_ratio"] = (stats["achieved_rps"] / rate) if rate > 0 else 0.0
    return stats


def exit_code_for_status(status: str) -> int:
    """Single source of truth for the calibration verdict -> process exit code.
    0 ok; 3 no_stable_point (no ceiling at all); 4 did_not_saturate (ceiling is
    only a lower bound, so a fraction-of-ceiling rate is not meaningful)."""
    return {"ok": 0, "no_stable_point": 3, "did_not_saturate": 4}.get(status, 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Per-cell rate calibration (conservative ceiling).")
    p.add_argument("--config", type=Path, required=True, help="Materialized client config for the cell (all factors except rate).")
    p.add_argument("--base-url", type=str, required=True)
    p.add_argument("--protocol", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--rates", type=str, required=True, help="Comma-separated ascending offered rates, e.g. 0.5,1,2,4,8")
    p.add_argument("--window-seconds", type=int, default=240)
    p.add_argument("--cooldown-seconds", type=int, default=30)
    p.add_argument("--concurrency-cap", type=int, default=64)
    p.add_argument("--fraction", type=float, default=0.85, help="rate_calibrated = fraction x achieved ceiling (DoW: 0.30 / 0.85).")
    p.add_argument("--achieved-ratio-min", type=float, default=0.98)
    p.add_argument("--drop-max", type=float, default=0.02)
    p.add_argument("--p99-bound", type=float, default=60.0)
    p.add_argument("--latency-climb-frac", type=float, default=0.20)
    p.add_argument("--cell-id", type=str, default="")
    p.add_argument("--system", type=str, default="")
    p.add_argument("--output", type=Path, required=True, help="Calibration JSON path (read by launch_cell --calibration-file).")
    p.add_argument("--sweep-dir", type=Path, default=None, help="Where to keep per-rate client CSVs (default: alongside --output).")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    sweep_dir = args.sweep_dir or args.output.parent / f"calib_sweep_{args.cell_id or 'cell'}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"[calibrate] cell={args.cell_id} system={args.system} base_url={args.base_url}", flush=True)
    print(f"[calibrate] rates={rates} window={args.window_seconds}s", flush=True)

    rows: list[dict] = []
    for rate in rates:
        print(f"[calibrate] rate={rate} rps for {args.window_seconds}s ...", flush=True)
        sub = sweep_dir / f"rate_{rate}"
        stats = run_one_rate(
            repo_root, args.config, args.base_url, args.protocol, args.model,
            rate, args.window_seconds, args.concurrency_cap, sub,
        )
        rows.append(stats)
        print(
            f"[calibrate]   achieved={stats['achieved_rps']:.3f} ratio={stats['achieved_ratio']:.3f} "
            f"drop={stats['drop_rate']:.3f} p99={stats['p99_e2e_s']:.2f}s climb={stats['latency_climb_frac']:+.2f}",
            flush=True,
        )
        # Early stop once clearly past the knee, to save GPU-h.
        if stats["achieved_ratio"] < 0.90 or stats["latency_climb_frac"] > 1.0:
            print("[calibrate]   past the knee, stopping sweep early", flush=True)
            break
        time.sleep(args.cooldown_seconds)

    ceiling, status = select_ceiling(
        rows,
        achieved_ratio_min=args.achieved_ratio_min,
        drop_max=args.drop_max,
        p99_bound=args.p99_bound,
        climb_frac=args.latency_climb_frac,
    )

    out = {
        "cell_id": args.cell_id,
        "system": args.system,
        "base_url": args.base_url,
        "fraction": args.fraction,
        "status": status,
        "criteria": {
            "achieved_ratio_min": args.achieved_ratio_min,
            "drop_max": args.drop_max,
            "p99_bound_s": args.p99_bound,
            "latency_climb_frac": args.latency_climb_frac,
        },
        "sweep": rows,
        "ceiling_rps": None,
        "ceiling_offered_rps": None,
        "rate_calibrated_rps": None,
    }
    if ceiling is not None:
        out["ceiling_rps"] = ceiling["achieved_rps"]
        out["ceiling_offered_rps"] = ceiling["offered_rate"]
        out["rate_calibrated_rps"] = round(ceiling["achieved_rps"] * args.fraction, 4)

    args.output.write_text(json.dumps(out, indent=2))
    print(f"[calibrate] status={status} ceiling={out['ceiling_rps']} "
          f"rate_calibrated={out['rate_calibrated_rps']} -> {args.output}", flush=True)
    # Exit codes carry the calibration verdict so callers (launch_cell / the
    # campaign) can refuse a non-saturated ceiling without re-parsing the file.
    if status == "no_stable_point":
        print("[calibrate] ERROR: no stable operating point found in the sweep.", file=sys.stderr)
    elif status == "did_not_saturate":
        print("[calibrate] ERROR: sweep never saturated; the ceiling is only a LOWER BOUND, "
              "so a 'fraction-of-ceiling' rate is not meaningful. Extend --rates upward and "
              "re-run.", file=sys.stderr)
    code = exit_code_for_status(status)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
