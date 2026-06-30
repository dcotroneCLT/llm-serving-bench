"""STEP 1 validation gate for an extension run directory.

Checks the five things that must hold before the serial-campaign / Dynamo
lifecycle refactor is built on top:

  1. Component PID map: per-component matched PIDs are sane and the prefill vs
     decode groups are DISJOINT (the negative-match split works). Prints the
     map for an eyeball comparison against `ps`.
  2. Engine USS aggregate (agg_<group>) exists and on membership-complete ticks
     equals the sum of its engine components' USS; the infra aggregate
     (agg_infra) is separate; both GPUs (gpu0 + gpu1) were sampled.
  3. Client features took effect: shared-prefix fraction ~ target; realized
     arrival CoV >> 1 for bursty.
  4. Analysis runs end-to-end on the new CSV layout: aging_trends emits a
     proc.uss_bytes row (proc_prefix = agg_<group> for Dynamo).

Single-process runs (val_vllm) run the subset that applies (multi-GPU and
component checks are skipped; the proc series is the single label).

Usage:
  python3 analysis/validate_extension_run.py --run-dir <run_dir> [--repo-root .]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _read_csvs(run_dir: Path, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(run_dir / f"{prefix}_*.csv"))):
        with open(f, newline="") as fp:
            rows.extend(csv.DictReader(fp))
    return rows


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def check_component_pid_map(run_dir: Path, components: list[dict]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    pids_by_label: dict[str, set] = {}
    for c in components:
        label = c["label"]
        rows = _read_csvs(run_dir, label)
        pids: set = set()
        for r in rows:
            for pid in (r.get("pids") or "").split(","):
                if pid.strip():
                    pids.add(pid.strip())
        pids_by_label[label] = pids
        msgs.append(f"    {label} ({c.get('group','?')}): pids={sorted(pids)[:8]}{'...' if len(pids)>8 else ''} n={len(pids)}")
    # prefill vs decode disjoint
    pre = next((c["label"] for c in components if "prefill" in c["label"]), None)
    dec = next((c["label"] for c in components if "decode" in c["label"]), None)
    if pre and dec:
        overlap = pids_by_label.get(pre, set()) & pids_by_label.get(dec, set())
        if overlap:
            ok = False
            msgs.append(f"    FAIL: prefill/decode PID overlap {overlap}")
        else:
            msgs.append("    prefill/decode disjoint: OK")
    front = next((c["label"] for c in components if "frontend" in c["label"]), None)
    if front and not pids_by_label.get(front):
        ok = False
        msgs.append("    FAIL: frontend matched no PID")
    return ok, msgs


def check_aggregate(run_dir: Path, engine_group: str, components: list[dict]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    agg = _read_csvs(run_dir, f"agg_{engine_group}")
    if not agg:
        return False, [f"    FAIL: no agg_{engine_group} CSV"]
    if "uss_bytes" not in agg[0]:
        return False, [f"    FAIL: agg_{engine_group} has no uss_bytes column"]
    complete = [r for r in agg if str(r.get("membership_complete", "")).lower() == "true"]
    msgs.append(f"    agg_{engine_group}: {len(agg)} rows, {len(complete)} membership-complete")
    if not complete:
        ok = False
        msgs.append("    FAIL: no membership-complete engine ticks (workers never all present together)")
    else:
        uss_vals = [_f(r["uss_bytes"]) for r in complete if _f(r["uss_bytes"]) is not None]
        if uss_vals:
            msgs.append(f"    engine USS range: {min(uss_vals)/1e6:.1f}..{max(uss_vals)/1e6:.1f} MB")
    # infra aggregate separate
    infra = _read_csvs(run_dir, "agg_infra")
    msgs.append(f"    agg_infra: {'present' if infra else 'MISSING'} ({len(infra)} rows)")
    if not infra:
        ok = False
    # per-tick consistency: agg uss == sum of engine components' uss on complete ticks
    eng_labels = [c["label"] for c in components if c.get("group", "engine") == "engine"]
    comp_by_ts: dict[str, float] = {}
    for label in eng_labels:
        for r in _read_csvs(run_dir, label):
            if str(r.get("membership_complete", "")).lower() != "true":
                continue
            v = _f(r.get("uss_bytes"))
            if v is not None:
                comp_by_ts[(r["ts_unix"], label)] = v
    mism = 0
    checked = 0
    for r in complete[:200]:
        ts = r["ts_unix"]
        s = sum(comp_by_ts.get((ts, label), 0.0) for label in eng_labels)
        a = _f(r["uss_bytes"]) or 0.0
        if s > 0:
            checked += 1
            if abs(a - s) > max(1.0, 0.001 * a):
                mism += 1
    if checked:
        msgs.append(f"    agg==sum(components) USS on complete ticks: {checked-mism}/{checked} match")
        if mism > 0:
            ok = False
            msgs.append("    FAIL: aggregate USS != sum of components on some ticks")
    return ok, msgs


def check_no_orphans(run_dir: Path, components: list[dict]) -> tuple[bool, list[str]]:
    """Run-level red flag: ANY tick with n_pids_unexpected>0 on a component.

    A stray sits OUTSIDE the recorded pgids, so it is never summed and each
    tick's aggregate is correct (the field is a pure per-tick diagnostic). But a
    process matching a component's cmdline regex that is not in its process group
    almost always means an orphan from a prior run survived (the pre-run reaper
    failed). The data is valid; for a 48h production run it is an operational red
    flag to clear before launch. So we enforce it here, at run level, not per tick.
    """
    msgs: list[str] = []
    ok = True
    for c in components:
        rows = _read_csvs(run_dir, c["label"])
        vals = [int(v) for r in rows if (v := r.get("n_pids_unexpected")) not in (None, "")]
        bad = [v for v in vals if v > 0]
        if bad:
            ok = False
            msgs.append(f"    FAIL: {c['label']} has {len(bad)} tick(s) with a stray regex-match "
                        f"outside its pgids (max {max(bad)}) -> orphan on host, run the reaper")
    if ok:
        msgs.append("    no stray processes outside recorded pgids (n_pids_unexpected==0 everywhere)")
    return ok, msgs


def check_multi_gpu(run_dir: Path, gpu_devices: list[int]) -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    for d in gpu_devices:
        present = bool(glob.glob(str(run_dir / f"gpu{d}_*.csv")))
        msgs.append(f"    gpu{d}: {'present' if present else 'MISSING'}")
        if not present:
            ok = False
    return ok, msgs


def check_client_features(run_dir: Path, target_prefix_frac: float) -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    rows = _read_csvs(run_dir / "client", "requests") if (run_dir / "client").exists() else []
    # also handle requests_*.csv directly under client/
    if not rows:
        for f in sorted(glob.glob(str(run_dir / "client" / "requests_*.csv"))):
            with open(f, newline="") as fp:
                rows.extend(csv.DictReader(fp))
    n = len(rows)
    if n == 0:
        return False, ["    FAIL: no client request rows"]
    applied = sum(1 for r in rows if str(r.get("shared_prefix_applied", "")).lower() == "true")
    frac = applied / n
    msgs.append(f"    shared_prefix_applied: {frac:.2f} of {n} (target ~{target_prefix_frac})")
    if target_prefix_frac > 0 and abs(frac - target_prefix_frac) > 0.15:
        ok = False
        msgs.append("    FAIL: prefix fraction far from target")
    arr_path = run_dir / "client" / "arrival_stats.json"
    if arr_path.exists():
        arr = json.loads(arr_path.read_text())
        cv = arr.get("interarrival_cv")
        msgs.append(f"    arrival_mode={arr.get('arrival_mode')} realized CoV={cv} rate={arr.get('realized_rate_rps')}")
        if arr.get("arrival_mode") == "bursty" and (cv is None or cv < 1.5):
            ok = False
            msgs.append("    FAIL: bursty CoV not >> 1 (burstiness had no effect)")
    else:
        msgs.append("    arrival_stats.json: MISSING")
        ok = False
    return ok, msgs


def check_analysis(run_dir: Path, repo_root: Path) -> tuple[bool, list[str]]:
    cmd = [sys.executable, str(repo_root / "analysis" / "aging_trends.py"),
           "--run-dir", str(run_dir), "--csv"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.SubprocessError as e:
        return False, [f"    FAIL: aging_trends crashed: {e}"]
    if out.returncode != 0:
        return False, [f"    FAIL: aging_trends rc={out.returncode}: {out.stderr.strip()[:300]}"]
    has_uss = any("proc.uss_bytes" in line for line in out.stdout.splitlines())
    n_rows = max(0, len(out.stdout.splitlines()) - 1)
    msg = [f"    aging_trends OK: {n_rows} indicator rows, proc.uss_bytes={'present' if has_uss else 'MISSING'}"]
    return has_uss, msg


def main() -> None:
    p = argparse.ArgumentParser(description="STEP 1 validation gate for an extension run.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--target-prefix-frac", type=float, default=0.8)
    args = p.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    monitors = manifest.get("monitors", {})
    comp_spec = monitors.get("components") if isinstance(monitors, dict) else None
    results: list[tuple[str, bool]] = []

    print(f"== Validation gate: {args.run_dir} ==")
    print(f"proc_prefix = {manifest.get('proc_prefix')}  attach={manifest.get('attach')}")

    if comp_spec:
        components = comp_spec["components"]
        engine_group = comp_spec.get("engine_group", "engine")
        gpu_devices = manifest.get("engine", {}).get("gpu_devices", [0, 1])
        print("\n[1] Component PID map (eyeball vs `ps -eo pid,cmd | grep dynamo`):")
        ok, msgs = check_component_pid_map(args.run_dir, components); print("\n".join(msgs)); results.append(("component_pid_map", ok))
        print("\n[2] Engine USS aggregate + infra + multi-GPU:")
        ok2, m2 = check_aggregate(args.run_dir, engine_group, components); print("\n".join(m2))
        okg, mg = check_multi_gpu(args.run_dir, gpu_devices); print("\n".join(mg))
        results.append(("aggregate", ok2)); results.append(("multi_gpu", okg))
        print("\n[2b] No stray processes outside recorded pgids (orphan red flag):")
        oko, mo = check_no_orphans(args.run_dir, components); print("\n".join(mo)); results.append(("no_orphans", oko))

    print("\n[3] Client features (prefix-repeat + burst):")
    okc, mc = check_client_features(args.run_dir, args.target_prefix_frac); print("\n".join(mc)); results.append(("client_features", okc))

    print("\n[4] Analysis pipeline end-to-end:")
    oka, ma = check_analysis(args.run_dir, args.repo_root); print("\n".join(ma)); results.append(("analysis", oka))

    print("\n== GATE SUMMARY ==")
    allok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        allok = allok and ok
    print("GATE:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
