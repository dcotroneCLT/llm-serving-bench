#!/usr/bin/env python3
"""Window shifts and DoW effects for the windowed Triton/vLLM rerun (R8).

The rescoped rerun (campaign extension_dow_rescoped) runs in three machine
windows W1..W3 (scripts/generate_rerun_windows.py). Each window holds one center
point (CP) per system; W3 also holds the Dynamo bridge CP cp4. This script:

  1. estimates the WINDOW OFFSETS from the CPs:
       common:      y_cp[s, w] = mu_s + delta_w        (least squares, delta_W1 = 0)
       per system:  delta[s, w] = y_cp[s, w] - y_cp[s, W1]
     and checks their CONSISTENCY across systems (the common correction assumes
     the shift is the same for Triton and vLLM);
  2. reports the Dynamo BRIDGE: cp4 (rerun epoch, W3) against cp1..cp3 (the
     completed Dynamo arm), as an absolute and relative shift and in units of
     the cp1..cp3 spread;
  3. estimates, per system, all five main effects and all ten two-factor
     interactions (effect = mean(y | +1) - mean(y | -1), the 16-run Resolution V
     design), WITHOUT and WITH the CP window correction (y - delta_w(run)), with
     Lenth's pseudo standard error and margin of error (alpha 0.05) for this
     saturated design. By construction of the quarter-block allocation only AC,
     AD and CD can move under the correction; the report prints the largest
     change on the other twelve effects as a check (it must be ~0).

ASSUMPTION (stated, not tested by the design): the window effect is ADDITIVE on
the analysis scale (--scale linear, or multiplicative with --scale log) and the
SAME for every design point of a system in that window (no window x factor
interaction); the CP of the window measures it. With --correction common it is
also assumed equal for Triton and vLLM (checked in step 1).

Response per run (--runs-root): from the canonical engine aggregate
(manifest.proc_prefix, process_alive ticks) after warmup_discard_s, 5-minute
medians, Theil-Sen slope of USS (bytes/h); requests completed ok in the same
window give the served rate. Metrics: growth_kb_per_req (default, = slope /
served rate, 1 KB = 1e3 bytes; the 2026-11-20 first-look script used
1024-byte KB, i.e. values 2.4% lower) or uss_slope_mb_h. --responses CSV (cell_id,
response) bypasses the run dirs; --write-responses saves them.

Usage:
  python3 analysis/window_effects.py --runs-root runs [--metric growth_kb_per_req]
      [--correction common|per_system] [--scale linear|log] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "campaigns" / "extension"
RERUN_MATRIX = EXT / "rerun_design_matrix.csv"
DOW_MATRIX = EXT / "dow_design_matrix.csv"
RERUN_CAMPAIGN = "extension_dow_rescoped"
DYNAMO_CAMPAIGN = "extension_dow_screening"
BRIDGE_CELL = "dow_dynamo_disagg_cp4"
DYNAMO_CPS = ("dow_dynamo_disagg_cp1", "dow_dynamo_disagg_cp2", "dow_dynamo_disagg_cp3")
RERUN_SYSTEMS = ("triton", "vllm")
FACTORS = ("A", "B", "C", "D", "E")
EFFECTS = list(FACTORS) + ["".join(p) for p in itertools.combinations(FACTORS, 2)]
WINDOW_EFFECTS = ("AC", "AD", "CD")
T_975_DF5 = 2.570582       # Student t, 0.975 quantile, 5 df (Lenth: m/3 = 15/3)
BIN_S = 300                # 5-minute medians


# --- Design ----------------------------------------------------------------------


def _read_commented_csv(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def load_design(rerun_matrix: Path = RERUN_MATRIX, dow_matrix: Path = DOW_MATRIX) -> dict[str, dict]:
    """{cell_id: {system, kind, window, codes}}. Triton/vLLM cells and cp4 take
    their window from the rerun matrix; the completed Dynamo cells are window
    'D' (their own epoch)."""
    cells: dict[str, dict] = {}
    for row in _read_commented_csv(dow_matrix):
        if row["system"] != "dynamo_disagg":
            continue
        cells[row["cell_id"]] = {"system": "dynamo_disagg", "kind": row["kind"], "window": "D",
                                 "codes": {f: int(row[f"{f}_code"]) for f in FACTORS}}
    for row in _read_commented_csv(rerun_matrix):
        cells[row["cell_id"]] = {"system": row["system"], "kind": row["kind"],
                                 "window": int(row["window"]),
                                 "codes": {f: int(row[f"{f}_code"]) for f in FACTORS}}
    return cells


def contrast(codes: dict, name: str) -> int:
    value = 1
    for f in name:
        value *= codes[f]
    return value


# --- Responses ---------------------------------------------------------------------


def run_dir_for(cell_id: str, runs_root: Path) -> Path:
    campaign = DYNAMO_CAMPAIGN if (cell_id.startswith("dow_dynamo") and cell_id != BRIDGE_CELL) \
        else RERUN_CAMPAIGN
    return runs_root / f"{campaign}_{cell_id}_r01"


def compute_response(run_dir: Path) -> dict:
    """{uss_slope_mb_h, served_rps, growth_kb_per_req, n_bins} for one run."""
    import pandas as pd
    from scipy.stats import theilslopes
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aging_io import load_proc  # noqa: E402

    manifest = json.loads((run_dir / "manifest.json").read_text())
    start = float(manifest["started_at_unix"]) + float(manifest.get("warmup_discard_s") or 0)
    end = float(manifest["started_at_unix"]) + float(manifest["duration_s"])
    proc = load_proc(run_dir, manifest.get("proc_prefix"), columns=["uss_bytes"])
    if proc is None:
        raise ValueError(f"{run_dir.name}: no engine aggregate samples")
    proc = proc[(proc["ts_unix"] >= start) & (proc["ts_unix"] <= end)].dropna(subset=["uss_bytes"])
    bins = proc.groupby((proc["ts_unix"] // BIN_S).astype("int64"))
    med = bins.agg({"ts_unix": "median", "uss_bytes": "median"})
    if len(med) < 10:
        raise ValueError(f"{run_dir.name}: only {len(med)} post-warmup bins")
    slope_bytes_per_s = theilslopes(med["uss_bytes"].to_numpy(), med["ts_unix"].to_numpy())[0]
    ok = 0
    for path in sorted((run_dir / "client").glob("requests_*.csv")):
        df = pd.read_csv(path, usecols=lambda c: c in ("submitted_at_unix", "status"))
        ok += int(((df["submitted_at_unix"] >= start) & (df["submitted_at_unix"] <= end)
                   & (df["status"] == "ok")).sum())
    served = ok / (end - start)
    if served <= 0:
        raise ValueError(f"{run_dir.name}: no successful requests after warmup")
    return {"uss_slope_mb_h": float(slope_bytes_per_s * 3600 / 1e6), "served_rps": float(served),
            "growth_kb_per_req": float(slope_bytes_per_s / served / 1e3), "n_bins": int(len(med))}


def responses_from_runs(design: dict, runs_root: Path, metric: str,
                        log=print) -> dict[str, float]:
    out = {}
    for cid in sorted(design):
        rd = run_dir_for(cid, runs_root)
        if not (rd / "manifest.json").exists():
            continue
        try:
            out[cid] = compute_response(rd)[metric]
        except (OSError, KeyError, ValueError) as exc:
            log(f"  [warn] {cid}: response not computed ({exc})")
    return out


def read_responses(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        return {r["cell_id"]: float(r["response"]) for r in csv.DictReader(stream)}


def write_responses(path: Path, responses: dict[str, float], metric: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "response", "metric"])
        for cid in sorted(responses):
            w.writerow([cid, repr(float(responses[cid])), metric])


# --- Window offsets ------------------------------------------------------------


def window_offsets(design: dict, y: dict[str, float]) -> dict:
    """Common (least squares, delta_1 = 0) and per-system CP offsets."""
    cps = {(c["system"], c["window"]): y[cid] for cid, c in design.items()
           if c["kind"] == "center" and c["system"] in RERUN_SYSTEMS and cid in y}
    windows = sorted({w for _, w in cps})
    systems = [s for s in RERUN_SYSTEMS if any(k[0] == s for k in cps)]
    out = {"cp": {f"{s}_W{w}": v for (s, w), v in sorted(cps.items())},
           "common": {}, "per_system": {}, "missing_cp": []}
    for s in RERUN_SYSTEMS:
        for w in (1, 2, 3):
            if (s, w) not in cps:
                out["missing_cp"].append(f"{s}_W{w}")
    if not cps or 1 not in windows:
        return out
    # y = mu_s + delta_w, delta_1 = 0.
    other = [w for w in windows if w != 1]
    X, b = [], []
    for (s, w), v in cps.items():
        X.append([1.0 if s == t else 0.0 for t in systems] + [1.0 if w == u else 0.0 for u in other])
        b.append(v)
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(b), rcond=None)
    out["common"] = {1: 0.0, **{w: float(coef[len(systems) + i]) for i, w in enumerate(other)}}
    for s in systems:
        if (s, 1) in cps:
            out["per_system"][s] = {w: cps[(s, w)] - cps[(s, 1)] for w in windows if (s, w) in cps}
    return out


def consistency(offsets: dict, tol_rel: float) -> dict:
    """Do Triton and vLLM agree on each window shift? |delta_triton - delta_vllm|
    against tol_rel x the mean CP level (no replication: a stated tolerance).

    The check needs ALL SIX center points (both systems x three windows): with
    one missing, a window's shift is seen by one system only and agreement is
    unverifiable, so status is "incomplete" and consistent is False (never a
    CONSISTENT verdict over a subset of the windows)."""
    per = offsets.get("per_system", {})
    level = float(np.mean(list(offsets["cp"].values()))) if offsets["cp"] else float("nan")
    missing = list(offsets.get("missing_cp", []))
    diffs = {}
    if all(s in per for s in RERUN_SYSTEMS):
        for w in sorted(set(per["triton"]) & set(per["vllm"])):
            if w != 1:
                diffs[w] = per["triton"][w] - per["vllm"][w]
    tolerance = tol_rel * abs(level)
    if missing:
        status = "incomplete"
    else:
        worst = max(abs(d) for d in diffs.values())
        status = "consistent" if worst <= tolerance else "inconsistent"
    return {"difference_by_window": diffs, "cp_level": level, "tolerance": tolerance,
            "missing_cp": missing, "status": status, "consistent": status == "consistent"}


def bridge(y: dict[str, float]) -> Optional[dict]:
    if BRIDGE_CELL not in y or not all(c in y for c in DYNAMO_CPS):
        return None
    old = np.array([y[c] for c in DYNAMO_CPS])
    shift = y[BRIDGE_CELL] - old.mean()
    sd = float(old.std(ddof=1))
    return {"cp4": y[BRIDGE_CELL], "cp1_cp3_mean": float(old.mean()), "cp1_cp3_sd": sd,
            "cp1_cp3_range": [float(old.min()), float(old.max())], "shift": float(shift),
            "shift_rel": float(shift / old.mean()) if old.mean() else float("nan"),
            "shift_in_sd": float(shift / sd) if sd > 0 else float("nan")}


# --- Effects -------------------------------------------------------------------


def lenth(effects: dict[str, float]) -> tuple[float, float]:
    """Lenth (1989) pseudo standard error and margin of error (alpha 0.05)."""
    abs_e = np.abs(np.array(list(effects.values())))
    s0 = 1.5 * np.median(abs_e)
    pse = 1.5 * np.median(abs_e[abs_e < 2.5 * s0]) if np.any(abs_e < 2.5 * s0) else s0
    return float(pse), float(T_975_DF5 * pse)


def estimate_effects(rows: list[tuple[dict, float]]) -> dict[str, float]:
    """effect = mean(y | contrast +1) - mean(y | contrast -1) over the 16 runs."""
    out = {}
    for name in EFFECTS:
        plus = [v for codes, v in rows if contrast(codes, name) > 0]
        minus = [v for codes, v in rows if contrast(codes, name) < 0]
        out[name] = float(np.mean(plus) - np.mean(minus))
    return out


def system_effects(design: dict, y: dict[str, float], system: str,
                   offset_of) -> dict:
    points = {cid: c for cid, c in design.items() if c["system"] == system and c["kind"] == "design"}
    missing = sorted(cid for cid in points if cid not in y)
    if len(points) != 16 or missing:
        return {"system": system, "complete": False, "missing": missing}
    raw_rows = [(c["codes"], y[cid]) for cid, c in points.items()]
    raw = estimate_effects(raw_rows)
    result = {"system": system, "complete": True, "raw": raw}
    pse, me = lenth(raw)
    result["raw_lenth"] = {"pse": pse, "me": me}
    if offset_of is not None:
        corr_rows = [(c["codes"], y[cid] - offset_of(system, c["window"])) for cid, c in points.items()]
        corrected = estimate_effects(corr_rows)
        pse_c, me_c = lenth(corrected)
        result["corrected"] = corrected
        result["corrected_lenth"] = {"pse": pse_c, "me": me_c}
        result["max_change_outside_window_effects"] = max(
            abs(corrected[e] - raw[e]) for e in EFFECTS if e not in WINDOW_EFFECTS)
    return result


# --- Driver ----------------------------------------------------------------------


def analyze(design: dict, y_in: dict[str, float], correction: str = "common",
            scale: str = "linear", tol_rel: float = 0.10) -> dict:
    if scale == "log":
        bad = [c for c, v in y_in.items() if not v > 0]
        if bad:
            raise ValueError(f"--scale log needs positive responses: {bad}")
        y = {c: math.log(v) for c, v in y_in.items()}
    else:
        y = dict(y_in)
    offsets = window_offsets(design, y)
    report = {"scale": scale, "correction": correction, "offsets": offsets,
              "consistency": consistency(offsets, tol_rel), "bridge": bridge(y), "systems": {}}

    def offset_of(system, window):
        if correction == "per_system":
            return offsets["per_system"].get(system, {}).get(window, float("nan"))
        return offsets["common"].get(window, float("nan"))

    # common: rests on the two systems agreeing, so it needs the six-CP
    # consistency check to PASS. per_system: each system's own three CPs.
    if correction == "per_system":
        can_correct = all(len(offsets["per_system"].get(s, {})) == 3 for s in RERUN_SYSTEMS)
        reason = None if can_correct else (
            "a system lacks a CP in some window: " + ", ".join(offsets["missing_cp"]))
    else:
        status = report["consistency"]["status"]
        can_correct = status == "consistent"
        reason = {
            "incomplete": "missing CPs " + ", ".join(offsets["missing_cp"])
                          + ": the consistency of the two systems' shifts cannot be verified",
            "inconsistent": "Triton and vLLM disagree on the window shift beyond the "
                            "tolerance; use --correction per_system",
        }.get(status)
    report["corrected_available"] = can_correct
    report["correction_unavailable_reason"] = reason
    for system in RERUN_SYSTEMS:
        report["systems"][system] = system_effects(design, y, system,
                                                   offset_of if can_correct else None)
    report["systems"]["dynamo_disagg"] = system_effects(design, y, "dynamo_disagg", None)
    return report


def format_report(r: dict, metric: str) -> str:
    out = [f"== Window shifts and DoW effects (metric={metric}, scale={r['scale']}, "
           f"correction={r['correction']}) =="]
    off = r["offsets"]
    out.append("\n[1] Center points and window offsets (relative to W1):")
    for key, v in off["cp"].items():
        out.append(f"    CP {key:<10} {v:12.4f}")
    if off["missing_cp"]:
        out.append(f"    missing CPs: {', '.join(off['missing_cp'])}")
    if off["common"]:
        out.append("    common offset: " + "  ".join(f"W{w} {d:+.4f}" for w, d in sorted(off["common"].items())))
    for s, d in off["per_system"].items():
        out.append(f"    {s:<7} offset: " + "  ".join(f"W{w} {v:+.4f}" for w, v in sorted(d.items())))
    c = r["consistency"]
    diffs = "  ".join(f"W{w} {d:+.4f}" for w, d in sorted(c["difference_by_window"].items()))
    if c["status"] == "incomplete":
        out.append(f"    consistency (triton - vllm): INCOMPLETE -- needs all six CPs, missing "
                   f"{', '.join(c['missing_cp'])}" + (f" (partial: {diffs})" if diffs else ""))
    else:
        out.append(f"    consistency (triton - vllm): {diffs}  tolerance {c['tolerance']:.4f} -> "
                   + ("CONSISTENT" if c["consistent"]
                      else "INCONSISTENT: prefer --correction per_system"))
    b = r["bridge"]
    out.append("\n[2] Dynamo bridge (cp4 vs cp1..cp3):")
    if b is None:
        out.append("    not available (cp4 or cp1..cp3 response missing)")
    else:
        out.append(f"    cp4 {b['cp4']:.4f} vs cp1..cp3 mean {b['cp1_cp3_mean']:.4f} "
                   f"(sd {b['cp1_cp3_sd']:.4f}, range {b['cp1_cp3_range'][0]:.4f}..{b['cp1_cp3_range'][1]:.4f})")
        out.append(f"    shift {b['shift']:+.4f} ({100 * b['shift_rel']:+.2f}%, {b['shift_in_sd']:+.2f} sd)")
    out.append("\n[3] Effects per system (mean(+1) - mean(-1); Lenth ME, alpha 0.05):")
    if not r["corrected_available"]:
        out.append(f"    window correction NOT available ({r['correction_unavailable_reason']})")
    for system, e in r["systems"].items():
        if not e["complete"]:
            out.append(f"    {system}: design incomplete, missing {e['missing']}")
            continue
        corr = e.get("corrected")
        head = f"    {system}: raw ME {e['raw_lenth']['me']:.4f}"
        if corr:
            head += (f"; corrected ME {e['corrected_lenth']['me']:.4f}; max change outside "
                     f"AC/AD/CD {e['max_change_outside_window_effects']:.2e}")
        out.append(head)
        for name in EFFECTS:
            raw = e["raw"][name]
            line = f"      {name:<3} raw {raw:+10.4f}{' *' if abs(raw) > e['raw_lenth']['me'] else '  '}"
            if corr:
                line += (f"   corrected {corr[name]:+10.4f}"
                         f"{' *' if abs(corr[name]) > e['corrected_lenth']['me'] else '  '}")
                if name in WINDOW_EFFECTS:
                    line += "  (window-exposed)"
            out.append(line)
    out.append("    * = |effect| > Lenth margin of error")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Window offsets and DoW effects for the rerun (R8).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs-root", type=Path, help="Directory holding the run dirs.")
    src.add_argument("--responses", type=Path, help="CSV cell_id,response (bypasses run dirs).")
    p.add_argument("--metric", choices=("growth_kb_per_req", "uss_slope_mb_h"),
                   default="growth_kb_per_req")
    p.add_argument("--correction", choices=("common", "per_system"), default="common")
    p.add_argument("--scale", choices=("linear", "log"), default="linear")
    p.add_argument("--consistency-tol", type=float, default=0.10,
                   help="Relative tolerance (x mean CP level) for the per-system offset check.")
    p.add_argument("--rerun-matrix", type=Path, default=RERUN_MATRIX)
    p.add_argument("--dow-matrix", type=Path, default=DOW_MATRIX)
    p.add_argument("--write-responses", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    design = load_design(args.rerun_matrix, args.dow_matrix)
    if args.responses:
        y = read_responses(args.responses)
    else:
        y = responses_from_runs(design, args.runs_root, args.metric,
                                log=lambda m: print(m, file=sys.stderr))
    if args.write_responses:
        write_responses(args.write_responses, y, args.metric)
    report = analyze(design, y, args.correction, args.scale, args.consistency_tol)
    report["metric"] = args.metric
    report["n_responses"] = len(y)
    print(format_report(report, args.metric))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
