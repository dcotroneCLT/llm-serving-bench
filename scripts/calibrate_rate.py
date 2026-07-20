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

Each rate step runs for a WALL DURATION sized to the workload's own tail
(max(step_min_s, ceil_mult x running p99 estimate)), and the achieved/offered
ratio is measured over an inner sub-window with a warmup discard and a p99
drain grace, attributing requests by SUBMISSION time. This removes the
finite-window bias where requests still in flight at the window edge
mechanically depress completed/offered on an unsaturated server (see
measure_window_stats). The acceptance bar is unchanged.

Conservative ceiling (deliberately NOT the 0.95 saturation edge, so that
85% of ceiling keeps real margin over a 48h run): the highest swept offered
rate for which ALL hold within the measurement sub-window:
  - achieved/offered >= achieved_ratio_min (default 0.98)
  - drop rate <= drop_max (default 0.02)
  - e2e p99 <= p99_bound seconds (a generous absolute bound, per cell)
  - backlog is flat: mean e2e in the second half of the sub-window is not more
    than climb_frac (default 0.20) above the first half (a rising latency
    within a fixed-rate window means the server is accumulating backlog and
    the rate is already unsustainable).
The ceiling is the HIGHEST swept rate that passes every criterion (bracket
selector); a failing rate above it brackets the knee. rate_calibrated =
fraction x achieved throughput at the ceiling. A sub-threshold failure BELOW
the ceiling (e.g. low-rate climb-estimator noise) is recorded as a
low_rate_anomaly and does NOT disqualify the sweep -- so an `ok` verdict no
longer implies a contiguous stable prefix from the lowest rate; see
select_ceiling and, downstream, calibrate_dow.classify_engine_failure.

Calibrate at t0 and FIX the rate for the whole run. Capacity erosion over
48h is a result to observe, not to absorb by re-calibrating mid-run.

Usage (single cell against a running endpoint):
  python3 scripts/calibrate_rate.py \
    --config <materialized client_config.yaml> \
    --base-url http://localhost:8100 --protocol vllm_openai --model Qwen/... \
    --rates 0.5,1,2,4,8 --step-min-seconds 600 --ceil-mult 20 \
    --fraction 0.85 --cell-id e1 --system vllm_standalone \
    --output calibration_e1.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Calibration method version. v1 (the original / unversioned method) sized each
# rate step by REQUEST COUNT / a fixed short window and counted completed-OK over
# the whole wall window, which mechanically depresses achieved/offered on an
# UNSATURATED server whenever e2e latency is a non-trivial fraction of the window
# (requests still in flight at the edge). v2 sizes each step by WALL DURATION
# (>= ceil_mult x p99) and measures over an inner sub-window with an unbiased
# submitted-in-window ratio and a p99 drain grace. Bump this whenever the
# measurement changes so old-method files are identifiable and cross-method
# ceilings are never mixed. See measure_window_stats() for the bias fix.
# Measurement method version: integer part = how the sweep ROWS are produced
# (v2 wall-duration steps + unbiased submitted-in-window ratio; see
# measure_window_stats), fractional part = SELECTOR revision. v2.1 is the bracket
# selector below -- SAME measurement as v2, so a v2.0 file and a v2.1 file carry
# identical rows and differ only in how the verdict was derived (a v2.0 file can
# be re-verdicted from its recorded rows by scripts/reeval_calibration.py without
# re-running the sweep). Bump the integer part only when the measurement changes.
CALIBRATION_METHOD_VERSION = 2.1

# Selector revision recorded alongside the verdict. 1 = the legacy contiguous-
# stable-prefix selector (rejected a sweep the moment the LOWEST rate failed,
# even on low-rate estimator noise); 2 = the bracket selector below.
CALIBRATION_SELECTOR_VERSION = 2

# Below this many completed requests a rate step's latency_climb_frac is computed
# from too few points for its first/second-half means to be meaningful, so the
# climb criterion is treated as INCONCLUSIVE (does not disqualify) and flagged.
# See select_ceiling for why this is a SELECTOR gate (using the recorded n_ok),
# not a change to how climb is measured.
DEFAULT_CLIMB_MIN_SAMPLES = 30


# ---------------------------------------------------------------------------
# Pure ceiling-selection logic (unit-tested without a server)
# ---------------------------------------------------------------------------


def _as_float(v) -> float:
    """Coerce a recorded metric to float; missing/non-numeric -> NaN so the
    caller's math.isfinite() guard fails the row CLOSED. Python's NaN comparisons
    are all False, so a bare `r["p99_e2e_s"] > bound` silently passes a NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(v) -> int:
    """Coerce a recorded integer field (client_rc); unparseable -> a non-zero
    sentinel so a corrupt rc fails closed (treated as a non-zero client exit)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _climb_sample_count(r: dict) -> Optional[int]:
    """Effective number of completed requests behind a row's latency_climb_frac.
    Prefer an explicit climb_samples if a row records one, else fall back to n_ok
    (the count of ok requests in the measurement sub-window, which is what the
    climb estimator is computed over). None when neither is recorded (older unit
    rows) -> the caller treats the climb criterion as fully applicable."""
    for key in ("climb_samples", "n_ok"):
        v = r.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def row_failures(
    r: dict,
    achieved_ratio_min: float,
    drop_max: float,
    p99_bound: float,
    climb_frac: float,
    offered_span_min: float,
    climb_min_samples: int,
) -> tuple[list[str], bool]:
    """Which acceptance criteria a row fails, as short names, plus whether the
    climb criterion was INCONCLUSIVE (too few samples). A row is stable iff the
    returned failure list is empty. The names are stable identifiers recorded on
    the row (failed_criteria) so the orchestrator can tell a saturation failure
    (drops/ratio/p99 -> a real ceiling below the grid) from low-rate climb noise
    -- the two want opposite grid moves.

    Saturation signals: achieved_ratio, drop_rate, p99_e2e_s. Load-generator
    integrity: offered_span, client_rc. Backlog trend: latency_climb -- gated on a
    minimum completed-sample count (see climb_min_samples)."""
    # Required numeric criteria FAIL CLOSED: a NaN/inf/missing value is a real
    # measurement gap, not a pass. Without the math.isfinite() guard a row with
    # e.g. p99_e2e_s=NaN (n_ok>0 but no usable e2e samples) would slip past every
    # threshold (NaN comparisons are always False) and read stable.
    failures: list[str] = []
    ar = _as_float(r.get("achieved_ratio"))
    if not math.isfinite(ar) or ar < achieved_ratio_min:
        failures.append("achieved_ratio")
    dr = _as_float(r.get("drop_rate"))
    if not math.isfinite(dr) or dr > drop_max:
        failures.append("drop_rate")
    p99 = _as_float(r.get("p99_e2e_s"))
    if not math.isfinite(p99) or p99 > p99_bound:
        failures.append("p99_e2e_s")
    span = _as_float(r.get("offered_span_frac", 1.0))
    if not math.isfinite(span) or span < offered_span_min:
        failures.append("offered_span")
    if _as_int(r.get("client_rc", 0)) != 0:
        failures.append("client_rc")
    n = _climb_sample_count(r)
    climb_inconclusive = (n is not None and n < int(climb_min_samples))
    climb = _as_float(r.get("latency_climb_frac"))
    if not climb_inconclusive and (not math.isfinite(climb) or climb > climb_frac):
        failures.append("latency_climb")
    return failures, climb_inconclusive


SATURATION_CRITERIA = frozenset({"achieved_ratio", "drop_rate", "p99_e2e_s"})


def past_knee(stats: dict, climb_min_samples: int) -> bool:
    """Whether a just-completed rate step is CLEARLY past the knee, so the sweep
    can stop early to save GPU-h. Deliberately LOOSER than the acceptance bar --
    this is a budget heuristic, not the ceiling verdict (select_ceiling decides
    that from all the rows afterward). Two independent triggers:

      - achieved_ratio < 0.90 -- UNGATED: a completed/offered ratio is meaningful
        at any sample count, and it independently catches the deeply-backlogged
        case (measure_s<=0 emits ratio 0.0 with climb inf and n_ok 0).
      - latency_climb_frac > 1.0 -- but ONLY when the climb is trustworthy, i.e.
        computed over >= climb_min_samples completions (the SAME gate the ceiling
        selector applies in row_failures). A noisy climb over too few completions
        must not halt the sweep before the real stable point / knee, or we would
        stop at a rate the selector itself would later treat as inconclusive-pass.
    """
    if _as_float(stats.get("achieved_ratio")) < 0.90:
        return True
    n_climb = _climb_sample_count(stats)
    climb_trustworthy = (n_climb is None or n_climb >= int(climb_min_samples))
    climb_val = _as_float(stats.get("latency_climb_frac"))
    return climb_trustworthy and math.isfinite(climb_val) and climb_val > 1.0


def select_ceiling(
    rows: list[dict],
    achieved_ratio_min: float = 0.98,
    drop_max: float = 0.02,
    p99_bound: float = 60.0,
    climb_frac: float = 0.20,
    offered_span_min: float = 0.5,
    climb_min_samples: int = DEFAULT_CLIMB_MIN_SAMPLES,
) -> tuple[Optional[dict], str]:
    """Pick the conservative ceiling from ascending-rate sweep rows (BRACKET
    selector, v2.1). ANNOTATES each row in place with:
      stable (bool), failed_criteria (list), climb_inconclusive (bool),
      and low_rate_anomaly (bool) on a sub-threshold failure BELOW the ceiling.

    Each row must have: offered_rate, achieved_rps, achieved_ratio, drop_rate,
    p99_e2e_s, latency_climb_frac. Returns (ceiling_row, status); status in
    {"ok", "no_stable_point", "did_not_saturate"}.

    The ceiling is the HIGHEST offered rate that passes every criterion. If a
    failing rate sits above it (which is always the case when the highest passing
    rate is not the top of the grid, since every rate above the highest passing
    one has failed) the knee is bracketed -> "ok". If the highest passing rate IS
    the top of the grid -> "did_not_saturate" (the ceiling is only a lower bound;
    extend the grid up). If NO rate passes -> "no_stable_point".

    Why bracket, not the old contiguous-stable-prefix: the prefix logic required
    stability from the LOWEST rate upward, so a single sub-threshold failure at a
    low rate (field: dow_dynamo_disagg_p12 on a fresh, healthy engine -- rate 0.25
    failed ONLY latency_climb by +0.04 over ~130 requests, pure low-rate estimator
    noise, while 0.5 passed everything and 1.0 failed hard with a +1.43 climb)
    made the selector return no_stable_point even though a clean stable point (0.5)
    existed with the knee (1.0) bracketed above it. Such a below-ceiling failure is
    recorded as a low_rate_anomaly and logged, but does NOT disqualify the sweep.
    No individual criterion is weakened.

    Climb estimator robustness (climb_min_samples): the climb criterion is gated
    on a minimum completed-sample count -- below it the row's first/second-half
    latency means are too few to trend and the criterion is inconclusive-pass
    (flagged). This is done as a SELECTOR gate on the recorded n_ok, NOT by
    switching the climb estimate from mean to median: median would change the
    MEASUREMENT (the recorded latency_climb_frac), which (a) would break the v2.1
    'selector-only, measurement unchanged' guarantee and (b) is not recomputable
    by reeval from the recorded rows (the raw per-request timeline is not stored),
    whereas the sample gate uses only recorded scalars and so applies identically
    to fresh sweeps and to offline re-evaluation. The gate deliberately does NOT
    rescue the genuine +1.43 climb at p12's rate 1.0 (hundreds of samples, far
    above the gate) -- that is a real knee and must still fail."""
    ordered = sorted(rows, key=lambda r: r["offered_rate"])
    for r in ordered:
        failures, climb_inconclusive = row_failures(
            r, achieved_ratio_min, drop_max, p99_bound, climb_frac,
            offered_span_min, climb_min_samples)
        r["failed_criteria"] = failures
        r["climb_inconclusive"] = climb_inconclusive
        r["stable"] = not failures
        r.pop("low_rate_anomaly", None)  # recomputed below; clear a stale flag

    passing = [r for r in ordered if r["stable"]]
    if not passing:
        return None, "no_stable_point"

    ceiling = passing[-1]  # highest offered rate that passes every criterion
    for r in ordered:
        if r["offered_rate"] < ceiling["offered_rate"] and not r["stable"]:
            r["low_rate_anomaly"] = True
    if ceiling is ordered[-1]:
        # Every rate up to the top passed: the sweep never bracketed the knee,
        # so this ceiling is only a lower bound. Caller should extend the rates.
        return ceiling, "did_not_saturate"
    return ceiling, "ok"


def size_step_window(p99_est: float, step_min_s: float, ceil_mult: float) -> float:
    """Wall duration for one rate step: max(step_min_s, ceil_mult x p99 estimate).

    Sizing by WALL DURATION (not request count) keeps the edge tail -- the
    warmup fill + drain grace, each ~one p99 -- a bounded small fraction of the
    window: with ceil_mult=20 the two p99 edges are ~10% of the window. The p99
    estimate is the running max over the steps already swept (0 on the first
    step, so it falls back to step_min_s). An ascending sweep has monotone-ish
    growing p99, so the previous step's p99 is a safe pre-run estimate for the
    next; the actual warmup/drain of each step are re-derived post-run from that
    step's own measured p99 (see measure_window_stats)."""
    return max(float(step_min_s), float(ceil_mult) * max(0.0, float(p99_est)))


def burst_cycle_seconds(arrival_mode: str, burst_factor: float,
                        burst_on_seconds: float) -> float:
    """MEAN length of one on/off burst cycle (burst_on + gap), else 0 for
    non-bursty.

    The client's BurstyArrival is an MMPP-2 with EXPONENTIAL on/off sojourns
    (client/benchmark.py), mean rate preserved: it is ON a fraction 1/burst_factor
    of the time, so the mean OFF sojourn is burst_on*(burst_factor-1) and the
    MEAN cycle is burst_on*burst_factor. Because the sojourns are stochastic the
    cycle length is not fixed, so trimming the sub-window to an integer number of
    MEAN cycles does not make the offered accounting exactly phase-independent --
    it removes the systematic bias of ending on a partial burst and lowers the
    phase/seed variance. The dominant robustness comes from the window spanning
    MANY cycles (the wall window is >= ceil_mult x p99 and >= step_min, i.e. tens
    of mean cycles), where the law of large numbers averages the on/off phase
    out. See measure_window_stats."""
    if (arrival_mode or "").strip().lower() != "bursty":
        return 0.0
    on = max(0.0, float(burst_on_seconds))
    bf = max(1.0, float(burst_factor))
    return on * bf


def _empty_window_stats(wall_s: float) -> dict:
    return {
        "n_offered": 0, "n_total": 0, "n_ok": 0, "n_dropped": 0,
        "achieved_rps": 0.0, "achieved_ratio": 0.0, "offered_coverage": 0.0,
        "offered_span_frac": 0.0, "drop_rate": 0.0,
        "p50_e2e_s": float("nan"), "p95_e2e_s": float("nan"),
        "p99_e2e_s": float("nan"), "latency_climb_frac": 0.0,
        "window_seconds": round(float(wall_s), 3), "warmup_s": 0.0,
        "drain_s": 0.0, "measure_seconds": 0.0, "p99_size_s": 0.0,
        "burst_cycle_s": 0.0, "burst_cycles": 0,
    }


def measure_window_stats(
    records: list[dict],
    wall_s: float,
    offered_rate: float,
    *,
    arrival_mode: str = "poisson",
    burst_factor: float = 4.0,
    burst_on_seconds: float = 10.0,
    warmup_mult: float = 1.0,
    drain_mult: float = 1.0,
) -> dict:
    """Unbiased per-rate window stats from per-request records.

    Fixes the finite-window throughput bias: over a wall window of length W at
    offered rate r with e2e latency L, ~r*L requests are still in flight at the
    window edge, so counting completed-OK / W mechanically reports r*(1 - L/W)
    even on an UNSATURATED server -- the shorter W is relative to L, the worse
    the bias (the real-data symptom: a low-rate, zero-drop, flat-latency,
    p99=15s step reporting achieved_ratio=0.83 purely because W~200s).

    We instead measure over an INNER sub-window, attributing each request by its
    SUBMISSION time: discard a warmup at the head (pipeline fill) and reserve a
    drain grace >= p99 at the tail, so every request submitted WITHIN the
    sub-window has had time to complete. Then
        achieved_ratio = completed-OK / submitted   (both counted in-sub-window)
    which converges to 1.0 on an unsaturated server regardless of L, while a
    genuinely saturated server -- which cannot complete what was offered, or
    whose e2e/backlog explodes -- still fails on ratio, drop, p99 or climb.

    For bursty arrivals the sub-window is trimmed to an integer number of MEAN
    burst cycles (burst_on + gap = burst_on_seconds * burst_factor). The client's
    on/off sojourns are exponential, so this does not make the accounting exactly
    phase-independent -- it removes the partial-burst edge bias and, combined with
    a window spanning many cycles, keeps the offered rate close to the mean
    regardless of starting phase (offered_coverage records the residual).

    `records`: dicts with submitted_at_unix (float), status (str), and
    e2e_latency_s (float | None). Returns a stats dict carrying the SAME keys the
    ceiling selector consumes (achieved_ratio, drop_rate, p99_e2e_s,
    latency_climb_frac, achieved_rps) plus the window provenance fields
    (window_seconds, warmup_s, drain_s, measure_seconds, burst_cycles)."""
    recs: list[tuple[float, str, Optional[float]]] = []
    for r in records:
        try:
            sub_t = float(r["submitted_at_unix"])
        except (KeyError, TypeError, ValueError):
            continue
        status = (r.get("status") or "").strip().lower()
        raw = r.get("e2e_latency_s")
        try:
            e2e = float(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            e2e = None
        recs.append((sub_t, status, e2e))

    if not recs:
        return _empty_window_stats(wall_s)

    t0 = min(t for t, _, _ in recs)
    # Size warmup/drain from the p99 of ALL ok e2e in the raw window: a stable
    # estimate that does not depend on the sub-window we are about to carve.
    ok_e2e_all = sorted(e for _, s, e in recs
                        if s in ("ok", "success") and e is not None)
    p99_size = _quantile(ok_e2e_all, 0.99) if ok_e2e_all else 0.0
    if not (p99_size == p99_size):  # NaN guard
        p99_size = 0.0
    warmup_s = max(0.0, float(warmup_mult)) * p99_size
    drain_s = max(0.0, float(drain_mult)) * p99_size

    meas_start = t0 + warmup_s
    meas_end = t0 + float(wall_s) - drain_s

    cycle_s = burst_cycle_seconds(arrival_mode, burst_factor, burst_on_seconds)
    burst_cycles = 0
    if cycle_s > 0 and meas_end > meas_start:
        k = int(math.floor((meas_end - meas_start) / cycle_s))
        if k >= 1:
            burst_cycles = k
            meas_end = meas_start + k * cycle_s
        # else: fewer than one full cycle fits -> keep the raw sub-window and
        # leave burst_cycles=0 so provenance flags the window as undersized.

    measure_s = meas_end - meas_start
    win = {
        "window_seconds": round(float(wall_s), 3),
        "warmup_s": round(warmup_s, 3),
        "drain_s": round(drain_s, 3),
        "measure_seconds": round(max(0.0, measure_s), 3),
        "p99_size_s": round(p99_size, 4),
        "burst_cycle_s": round(cycle_s, 3),
        "burst_cycles": burst_cycles,
    }

    if measure_s <= 0:
        # The tail (warmup+drain, sized from the observed p99) consumed the whole
        # window: e2e is huge relative to W, which only happens when the server is
        # deeply backlogged. Emit an explicitly UNSTABLE row so the ceiling
        # selector breaks the stable prefix here instead of dividing by <= 0.
        out = _empty_window_stats(wall_s)
        out.update(win)
        out["p99_e2e_s"] = p99_size
        out["latency_climb_frac"] = float("inf")
        return out

    sel = [(t, s, e) for (t, s, e) in recs if meas_start <= t < meas_end]
    n_offered = len(sel)
    n_ok = sum(1 for _, s, _ in sel if s in ("ok", "success"))
    n_dropped = sum(1 for _, s, _ in sel if s == "dropped")
    ok_e2e = sorted(e for _, s, e in sel if s in ("ok", "success") and e is not None)
    climb = _latency_climb_frac(
        [(t, e) for (t, s, e) in sel if s in ("ok", "success") and e is not None])
    # Guard the v2 ratio's blind spot: achieved_ratio is now completed/SUBMITTED,
    # so a load generator that stalled or died mid-step could submit a handful of
    # requests that all succeed -> ratio 1.0 on a step that never tested the
    # requested offer (a silently-low ceiling). Two metrics:
    #   offered_coverage = submitted / (offered_rate x measure_s)  [INFORMATIONAL]
    #   offered_span_frac = (last_sub - first_sub) / measure_s      [GATED]
    # offered_coverage (a count ratio) is confounded by the arrival process: the
    # client's bursty MMPP-2 legitimately over/under-shoots the nominal rate by
    # ~+/-20% over a practical window, so a healthy bursty step can read ~0.82 --
    # gating on it would false-reject. offered_span_frac instead asks whether
    # submissions actually spanned the whole sub-window; bursty OFF gaps are
    # INTERIOR so they do not shrink the span, but a mid-window death leaves a
    # large empty tail -> a clean, arrival-mode-agnostic stall signal.
    expected_offered = float(offered_rate) * measure_s if offered_rate > 0 else 0.0
    offered_coverage = (n_offered / expected_offered) if expected_offered > 0 else 0.0
    sub_ts = [t for t, _, _ in sel]
    offered_span_frac = ((max(sub_ts) - min(sub_ts)) / measure_s
                         if len(sub_ts) >= 2 and measure_s > 0 else 0.0)
    out = {
        "n_offered": n_offered, "n_total": n_offered,
        "n_ok": n_ok, "n_dropped": n_dropped,
        "achieved_rps": (n_ok / measure_s) if measure_s > 0 else 0.0,
        "achieved_ratio": (n_ok / n_offered) if n_offered > 0 else 0.0,
        "offered_coverage": offered_coverage,
        "offered_span_frac": offered_span_frac,
        "drop_rate": (n_dropped / n_offered) if n_offered > 0 else 0.0,
        "p50_e2e_s": _quantile(ok_e2e, 0.50),
        "p95_e2e_s": _quantile(ok_e2e, 0.95),
        "p99_e2e_s": _quantile(ok_e2e, 0.99),
        "latency_climb_frac": climb,
    }
    out.update(win)
    return out


def window_stats_from_csvs(
    sub: Path, wall_s: float, offered_rate: float, *,
    arrival_mode: str = "poisson", burst_factor: float = 4.0,
    burst_on_seconds: float = 10.0, warmup_mult: float = 1.0,
    drain_mult: float = 1.0,
) -> dict:
    """Load one rate step's per-request CSVs and reduce them to unbiased window
    stats (see measure_window_stats). Reads EVERY requests_*.csv in `sub`, so the
    caller must guarantee the dir holds only THIS step's CSVs (StaleSweepDir)."""
    records: list[dict] = []
    for f in sorted(glob.glob(str(sub / "requests_*.csv"))):
        with open(f, newline="") as fp:
            records.extend(csv.DictReader(fp))
    return measure_window_stats(
        records, float(wall_s), float(offered_rate),
        arrival_mode=arrival_mode, burst_factor=burst_factor,
        burst_on_seconds=burst_on_seconds, warmup_mult=warmup_mult,
        drain_mult=drain_mult,
    )


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


class StaleSweepDir(RuntimeError):
    """A per-rate sweep subdir already holds requests_*.csv from a prior run."""


def run_one_rate(
    repo_root: Path, config: Path, base_url: str, protocol: str, model: str,
    rate: float, wall_s: float, concurrency_cap: int, sub: Path, *,
    arrival_mode: str = "poisson", burst_factor: float = 4.0,
    burst_on_seconds: float = 10.0, warmup_mult: float = 1.0,
    drain_mult: float = 1.0,
) -> dict:
    # Refuse to reuse a rate subdir that still holds a prior sweep's CSVs:
    # window_stats_from_csvs() reads EVERY requests_*.csv in `sub`, so leftover
    # files would silently inflate n_total/achieved and skew the ceiling and the
    # calibrated rate. Calibration probes are ephemeral, so the fix is to fail
    # loudly rather than mix (mirrors launch_cell's assert_run_dir_fresh).
    stale = sorted(glob.glob(str(sub / "requests_*.csv")))
    if stale:
        raise StaleSweepDir(
            f"refusing to reuse {sub}: it already holds {len(stale)} "
            "requests_*.csv from a prior sweep, which window_stats_from_csvs "
            "would mix into this rate's stats. Remove the stale sweep dir or "
            "pass a fresh --sweep-dir."
        )
    sub.mkdir(parents=True, exist_ok=True)
    # The client submits at `rate` for the ENTIRE wall window; warmup and drain
    # are analysis-time partitions of the collected timeline, not client phases,
    # so the drain grace is naturally satisfied by the client running (and then
    # end-draining) past the measurement sub-window.
    cmd = [
        sys.executable, str(repo_root / "client" / "run_client.py"),
        "--config", str(config),
        "--output-dir", str(sub),
        "--duration-seconds", str(int(math.ceil(float(wall_s)))),
        "--protocol", protocol,
        "--base-url", base_url,
        "--model", model,
        "--target-rate-rps", str(rate),
        "--concurrency-cap", str(concurrency_cap),
    ]
    with (sub / "client.log").open("wb") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    stats = window_stats_from_csvs(
        sub, float(wall_s), rate,
        arrival_mode=arrival_mode, burst_factor=burst_factor,
        burst_on_seconds=burst_on_seconds, warmup_mult=warmup_mult,
        drain_mult=drain_mult,
    )
    stats["offered_rate"] = rate
    # A non-zero client exit (e.g. run_client's writer-fatal abort) means the step
    # did not run cleanly; record it so select_ceiling refuses the step rather
    # than trusting whatever partial CSVs landed. Kept as an int rc for provenance.
    try:
        stats["client_rc"] = int(proc.returncode)
    except (TypeError, ValueError):
        stats["client_rc"] = 0
    # measure_window_stats already computes the unbiased submitted-in-window
    # ratio; keep a fallback for callers/tests that stub window_stats_from_csvs.
    stats.setdefault(
        "achieved_ratio", (stats["achieved_rps"] / rate) if rate > 0 else 0.0)
    return stats


# ---------------------------------------------------------------------------
# Calibration provenance (read by the staleness / host / image gate)
# ---------------------------------------------------------------------------


def _nvidia_gpu_name_driver() -> tuple[Optional[str], Optional[str]]:
    """(gpu_name, driver_version) from nvidia-smi, or (None, None) if it cannot
    be read (no GPU / nvidia-smi absent). Provenance is best-effort: a value we
    cannot determine is recorded as null, never fabricated."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None, None
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    first = r.stdout.strip().splitlines()[0]
    name, _, drv = first.partition(",")
    return (name.strip() or None), (drv.strip() or None)


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def build_provenance(config: Path, image_tag: Optional[str],
                     image_digest: Optional[str], now_unix: float) -> dict:
    """Provenance the staleness / host / image gate reads (launch_cell +
    campaign pre-flight): WHEN the calibration was taken, on WHICH host / GPU /
    driver, against WHICH image, and a hash of the client config swept. A
    calibration whose host/image signature no longer matches the current one, or
    which is older than the campaign's max age, must be refused with the same
    fail-loud semantics as a missing REQUIRED calibration -- a month-1 ceiling
    must never silently drive a month-3 run."""
    gpu_name, driver = _nvidia_gpu_name_driver()
    return {
        "calibrated_at_unix": round(now_unix, 3),
        "calibrated_at_iso": datetime.fromtimestamp(now_unix, timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name,
        "driver_version": driver,
        "image_tag": image_tag,
        "image_digest": image_digest,
        "client_config_hash": _sha256_file(config),
    }


def exit_code_for_status(status: str) -> int:
    """Single source of truth for the calibration verdict -> process exit code.
    0 ok; 3 no_stable_point (no ceiling at all); 4 did_not_saturate (ceiling is
    only a lower bound, so a fraction-of-ceiling rate is not meaningful)."""
    return {"ok": 0, "no_stable_point": 3, "did_not_saturate": 4}.get(status, 0)


def read_arrival_config(config: Path) -> tuple[str, float, float]:
    """(arrival_mode, burst_factor, burst_on_seconds) from the materialized client
    config, mirroring run_client.py's own precedence (arrival_mode supersedes the
    legacy request_distribution). Burst-cycle-aware measurement windows need the
    SAME burst timing the client will actually use. Best-effort: an unreadable
    config falls back to the poisson defaults (non-bursty -> no cycle trimming)."""
    try:
        cfg = yaml.safe_load(Path(config).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return "poisson", 4.0, 10.0
    mode = cfg.get("arrival_mode") or cfg.get("request_distribution") or "poisson"
    try:
        bf = float(cfg.get("burst_factor", 4.0))
    except (TypeError, ValueError):
        bf = 4.0
    try:
        on = float(cfg.get("burst_on_seconds", 10.0))
    except (TypeError, ValueError):
        on = 10.0
    return str(mode).strip().lower(), bf, on


def main() -> None:
    p = argparse.ArgumentParser(description="Per-cell rate calibration (conservative ceiling).")
    p.add_argument("--config", type=Path, required=True, help="Materialized client config for the cell (all factors except rate).")
    p.add_argument("--base-url", type=str, required=True)
    p.add_argument("--protocol", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--rates", type=str, required=True, help="Comma-separated ascending offered rates, e.g. 0.5,1,2,4,8")
    p.add_argument("--step-min-seconds", type=float, default=None,
                   help="Floor for each rate step's WALL duration (default 600). The "
                        "actual per-step duration = max(step_min, ceil_mult x running "
                        "p99 estimate), so a long-latency shape gets a proportionally "
                        "longer window and the edge tail stays a bounded fraction.")
    p.add_argument("--ceil-mult", type=float, default=20.0,
                   help="Per-step wall duration >= ceil_mult x the running p99 estimate "
                        "(default 20 -> the warmup+drain p99 edges are ~10%% of the "
                        "window).")
    p.add_argument("--warmup-mult", type=float, default=1.0,
                   help="In-step warmup discard = warmup_mult x measured p99 (pipeline "
                        "fill; excluded from the measurement sub-window).")
    p.add_argument("--drain-mult", type=float, default=1.0,
                   help="Drain grace = drain_mult x measured p99 (>= p99), reserved at "
                        "the tail so requests submitted in the sub-window can complete.")
    p.add_argument("--window-seconds", type=int, default=None,
                   help="DEPRECATED (v1 fixed-window method). If set and "
                        "--step-min-seconds is not, it is used as the per-step wall "
                        "floor for back-compat; prefer --step-min-seconds.")
    p.add_argument("--cooldown-seconds", type=int, default=30)
    p.add_argument("--concurrency-cap", type=int, default=64)
    p.add_argument("--fraction", type=float, default=0.85, help="rate_calibrated = fraction x achieved ceiling (DoW: 0.30 / 0.85).")
    p.add_argument("--achieved-ratio-min", type=float, default=0.98)
    p.add_argument("--drop-max", type=float, default=0.02)
    p.add_argument("--p99-bound", type=float, default=60.0)
    p.add_argument("--latency-climb-frac", type=float, default=0.20)
    p.add_argument("--climb-min-samples", type=int, default=DEFAULT_CLIMB_MIN_SAMPLES,
                   help="Minimum completed requests for a step's latency_climb_frac "
                        "to be trusted; below it the climb criterion is inconclusive "
                        "(does not disqualify) and flagged. The bracket selector, not "
                        "this gate, is the primary defense against low-rate climb noise.")
    p.add_argument("--offered-span-min", type=float, default=0.5,
                   help="A rate step's submissions must span at least this fraction "
                        "of the measurement sub-window (last-minus-first submit / "
                        "measure_seconds), else it is rejected as a stalled/died "
                        "step. Temporal span, not a count ratio, so it is robust to "
                        "the bursty arrival process's rate variance.")
    p.add_argument("--cell-id", type=str, default="")
    p.add_argument("--system", type=str, default="")
    p.add_argument("--image-tag", type=str, default=None,
                   help="repo:tag of the calibrated engine image, recorded in the "
                        "provenance block so the run-time gate can verify the "
                        "calibration was taken against the SAME image.")
    p.add_argument("--image-digest", type=str, default=None,
                   help="sha256 of the calibrated engine image (from the image pin), "
                        "recorded in provenance for the image-signature gate.")
    p.add_argument("--output", type=Path, required=True, help="Calibration JSON path (read by launch_cell --calibration-file).")
    p.add_argument("--sweep-dir", type=Path, default=None, help="Where to keep per-rate client CSVs (default: alongside --output).")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    sweep_dir = args.sweep_dir or args.output.parent / f"calib_sweep_{args.cell_id or 'cell'}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Step floor: prefer --step-min-seconds; fall back to the deprecated
    # --window-seconds for back-compat; else the 600s default.
    if args.step_min_seconds is not None:
        step_min_s = float(args.step_min_seconds)
    elif args.window_seconds is not None:
        step_min_s = float(args.window_seconds)
    else:
        step_min_s = 600.0

    arrival_mode, burst_factor, burst_on_seconds = read_arrival_config(args.config)

    print(f"[calibrate] cell={args.cell_id} system={args.system} base_url={args.base_url}", flush=True)
    print(f"[calibrate] method=v{CALIBRATION_METHOD_VERSION} rates={rates} "
          f"step_min={step_min_s:.0f}s ceil_mult={args.ceil_mult} "
          f"arrival={arrival_mode}"
          + (f" burst(factor={burst_factor},on={burst_on_seconds}s,"
             f"cycle={burst_cycle_seconds(arrival_mode, burst_factor, burst_on_seconds):.0f}s)"
             if arrival_mode == "bursty" else ""),
          flush=True)

    rows: list[dict] = []
    p99_est = 0.0  # running max p99 across swept steps; sizes the NEXT step.
    for rate in rates:
        wall_s = size_step_window(p99_est, step_min_s, args.ceil_mult)
        print(f"[calibrate] rate={rate} rps for {wall_s:.0f}s "
              f"(p99_est={p99_est:.1f}s) ...", flush=True)
        sub = sweep_dir / f"rate_{rate}"
        stats = run_one_rate(
            repo_root, args.config, args.base_url, args.protocol, args.model,
            rate, wall_s, args.concurrency_cap, sub,
            arrival_mode=arrival_mode, burst_factor=burst_factor,
            burst_on_seconds=burst_on_seconds, warmup_mult=args.warmup_mult,
            drain_mult=args.drain_mult,
        )
        rows.append(stats)
        if stats.get("client_rc", 0) != 0:
            print(f"[calibrate]   WARNING: client exited rc={stats['client_rc']} "
                  "at this rate; step marked incomplete (cannot be a ceiling).",
                  file=sys.stderr, flush=True)
        print(
            f"[calibrate]   achieved={stats['achieved_rps']:.3f} ratio={stats['achieved_ratio']:.3f} "
            f"cover={stats.get('offered_coverage', 1.0):.2f} span={stats.get('offered_span_frac', 1.0):.2f} "
            f"drop={stats['drop_rate']:.3f} p99={stats['p99_e2e_s']:.2f}s climb={stats['latency_climb_frac']:+.2f} "
            f"(measure={stats.get('measure_seconds', 0):.0f}s"
            + (f", {stats.get('burst_cycles', 0)} cycles" if arrival_mode == "bursty" else "")
            + ")",
            flush=True,
        )
        # Grow the running p99 estimate so the NEXT (higher-rate) step gets a
        # window sized to its expected tail. NaN (no completions) does not update.
        p99 = stats.get("p99_e2e_s")
        try:
            if p99 is not None and float(p99) == float(p99):
                p99_est = max(p99_est, float(p99))
        except (TypeError, ValueError):
            pass
        # Early stop once clearly past the knee, to save GPU-h (see past_knee:
        # the climb trigger uses the SAME sample gate as the ceiling selector).
        if past_knee(stats, args.climb_min_samples):
            print("[calibrate]   past the knee, stopping sweep early", flush=True)
            break
        time.sleep(args.cooldown_seconds)

    ceiling, status = select_ceiling(
        rows,
        achieved_ratio_min=args.achieved_ratio_min,
        drop_max=args.drop_max,
        p99_bound=args.p99_bound,
        climb_frac=args.latency_climb_frac,
        offered_span_min=args.offered_span_min,
        climb_min_samples=args.climb_min_samples,
    )
    # Surface the bracket selector's annotations: a low-rate anomaly (a
    # sub-threshold failure BELOW the chosen ceiling, e.g. climb noise at a low
    # rate) is recorded and logged but did not disqualify the sweep; an
    # inconclusive-climb step is flagged too.
    for r in sorted(rows, key=lambda x: x["offered_rate"]):
        if r.get("low_rate_anomaly"):
            print(f"[calibrate]   NOTE low-rate anomaly at {r['offered_rate']} rps: "
                  f"failed {r.get('failed_criteria')} BELOW the ceiling "
                  f"{ceiling['offered_rate'] if ceiling else None} rps "
                  "-- recorded, not disqualifying (knee bracketed above).",
                  flush=True)
        if r.get("climb_inconclusive"):
            print(f"[calibrate]   NOTE climb inconclusive at {r['offered_rate']} rps "
                  f"(< {args.climb_min_samples} completed samples); climb criterion "
                  "skipped for this step.", flush=True)

    out = {
        "cell_id": args.cell_id,
        "system": args.system,
        "base_url": args.base_url,
        "fraction": args.fraction,
        "status": status,
        # Identifies the measurement method so v1 (short-fixed-window, biased on
        # long-latency/bursty shapes) files are distinguishable and never mixed
        # with v2 ceilings. The fractional part is the SELECTOR revision (see
        # CALIBRATION_METHOD_VERSION); launch_cell.check_calibration_provenance
        # warns on a measurement (integer-part) mismatch.
        "calibration_method_version": CALIBRATION_METHOD_VERSION,
        "selector_version": CALIBRATION_SELECTOR_VERSION,
        "criteria": {
            "achieved_ratio_min": args.achieved_ratio_min,
            "drop_max": args.drop_max,
            "p99_bound_s": args.p99_bound,
            "latency_climb_frac": args.latency_climb_frac,
            "offered_span_min": args.offered_span_min,
            "climb_min_samples": args.climb_min_samples,
        },
        # Step-sizing / sub-window provenance: how each rate step's wall window
        # was sized and partitioned (per-row window_seconds/warmup_s/drain_s/
        # measure_seconds/burst_cycles record what was actually used).
        "step_params": {
            "step_min_seconds": step_min_s,
            "ceil_mult": args.ceil_mult,
            "warmup_mult": args.warmup_mult,
            "drain_mult": args.drain_mult,
            "arrival_mode": arrival_mode,
            "burst_factor": burst_factor,
            "burst_on_seconds": burst_on_seconds,
            "burst_cycle_s": burst_cycle_seconds(arrival_mode, burst_factor, burst_on_seconds),
        },
        "sweep": rows,
        "ceiling_rps": None,
        "ceiling_offered_rps": None,
        "rate_calibrated_rps": None,
        # Provenance for the staleness / host / image gate. Recorded on EVERY
        # calibration (even a non-ok one) so a stale/mismatched file is caught
        # regardless of its verdict.
        "provenance": build_provenance(args.config, args.image_tag, args.image_digest, time.time()),
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
    try:
        main()
    except StaleSweepDir as e:
        print(f"[calibrate] ERROR: {e}", file=sys.stderr)
        sys.exit(2)
