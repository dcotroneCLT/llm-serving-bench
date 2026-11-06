"""Integrity and analysis gate for extension datasets.

All layouts must finalize their declared window with populated, continuous
process/GPU/system/client data. Whole-container version-2 runs additionally
prove stable, disjoint roles, GPU/model coverage and memory conservation.
Historical Dynamo runs keep their component/PGID and infra checks.

The full gate validates the recorded workload features and requires usable
post-warmup USS statistics, without requiring a significant aging trend.
--scope-only validates complete short whole-container runs without statistics.

Usage:
  python3 analysis/validate_extension_run.py --run-dir <run_dir> [--repo-root .]
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecc_check import ecc_verdict  # noqa: E402
from run_integrity import check_run_integrity, saved_client_config  # noqa: E402


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


def _pids(value) -> set[int]:
    return {int(p) for p in re.split(r"[;,]", value or "") if p.strip()}


def check_container_scope(run_dir: Path, spec: dict) -> tuple[bool, list[str]]:
    """Validate saved scope evidence at every complete tick, not just PID unions.

    No live /proc access is required, so this gate also works on archived runs.
    Incomplete ticks remain explicit; a run with <95% complete ticks fails.
    """
    try:
        if spec.get("scope_version") != 2:
            raise ValueError("legacy whole-container role evidence is unverified; use scope version 2")
        identity = json.loads((run_dir / "scope_identity.json").read_text())
        root_pid = int(spec["root_pid"])
        if (identity.get("scope_version") != 2 or identity["root_pid"] != root_pid or
                identity["root_create_time"] != spec["root_create_time"] or
                not identity.get("pid_namespace") or
                not any(p != "/" for p in identity.get("cgroups", {}).values())):
            return False, ["    FAIL: missing/mismatched container namespace/cgroup/root identity"]
        components = spec["components"]
        if sorted(c["role"] for c in components) != ["auxiliary", "engine", "orchestration"]:
            return False, ["    FAIL: component role partition is incomplete"]
        rows = _read_csvs(run_dir, f"agg_{spec['engine_group']}")
        complete = [r for r in rows if r.get("process_alive", "").lower() == "true"]
        if not rows or len(complete) < 0.95 * len(rows):
            return False, [f"    FAIL: only {len(complete)}/{len(rows)} complete container ticks (minimum 95%)"]
        by_label = {}
        for comp in components:
            indexed = {}
            for row in _read_csvs(run_dir, comp["label"]):
                if row["ts_unix"] in indexed:
                    raise ValueError("duplicate component timestamp")
                indexed[row["ts_unix"]] = row
            by_label[comp["label"]] = indexed
        seen_ts, previous_identities, role_history = set(), set(), {}
        for agg in complete:
            ts = agg["ts_unix"]
            if ts in seen_ts:
                raise ValueError("duplicate aggregate timestamp")
            seen_ts.add(ts)
            if (agg.get("model_scope_complete", "").lower() != "true" or
                    agg.get("membership_complete", "").lower() != "true" or
                    agg.get("n_pids_unexpected") != "0" or agg.get("scope_error")):
                raise ValueError(f"incomplete model/scope evidence at {ts}")
            all_pids, engine_pids, roles = set(), set(), {}
            sums = {f: 0.0 for f in ("uss_bytes", "rss_bytes", "vms_bytes", "pss_bytes")}
            for comp in components:
                row = by_label[comp["label"]][ts]
                pids = _pids(row["pids"])
                if (all_pids & pids or row.get("membership_complete", "").lower() != "true" or
                        len(pids) != int(row["n_pids_sampled"]) or
                        len(pids) != int(row["n_pids_matched"])):
                    raise ValueError(f"overlapping/incomplete component membership at {ts}")
                all_pids |= pids
                roles.update({pid: comp["role"] for pid in pids})
                if comp["role"] == "engine":
                    engine_pids |= pids
                for field in sums:
                    value = float(row[field])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {field} at {ts}")
                    sums[field] += value
            model = _pids(agg.get("model_pids"))
            gpu_pids = _pids(agg.get("gpu_pids"))
            if (root_pid not in all_pids or not model or not model <= engine_pids or
                    not gpu_pids or not model <= gpu_pids or not gpu_pids <= all_pids or
                    all_pids != _pids(agg.get("scope_pids")) or
                    len(all_pids) != int(agg["n_pids_sampled"])):
                raise ValueError(f"PID1 or model-holding process missing from partition at {ts}")
            pairs = [p.split("@", 1) for p in agg["pid_identities"].split(";")]
            identities = dict(pairs)
            if (set(map(int, identities)) != all_pids or
                    len(pairs) != len(all_pids) or
                    any(not math.isfinite(float(ct)) for ct in identities.values()) or
                    float(identities[str(root_pid)]) != spec["root_create_time"]):
                raise ValueError(f"process identity evidence mismatch at {ts}")
            current_identities = {(int(pid), float(ct)) for pid, ct in identities.items()}
            for pid, ctime in current_identities:
                key = (pid, ctime)
                if key in role_history and role_history[key] != roles[pid]:
                    raise ValueError(f"process identity changed semantic role at {ts}")
                role_history[key] = roles[pid]
            if previous_identities - current_identities:
                raise ValueError(f"accepted memory series spans process loss/replacement at {ts}")
            previous_identities = current_identities
            for field, total in sums.items():
                value = float(agg[field])
                if not math.isfinite(value) or abs(value - total) > max(1.0, abs(total) * 1e-9):
                    raise ValueError(f"container {field} != sum of components at {ts}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return False, [f"    FAIL: container scope evidence: {exc}"]
    return True, [f"    whole-container scope: {len(complete)}/{len(rows)} ticks complete; "
                  "PID1 + model covered, disjoint components, memory sums verified"]


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


def check_client_features(run_dir: Path, target_prefix_frac: float,
                          target_arrival_mode: Optional[str] = None) -> tuple[bool, list[str]]:
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
    if any(str(row.get("shared_prefix_applied", "")).lower() not in ("true", "false", "0", "1") for row in rows):
        return False, ["    FAIL: missing/malformed prefix-repeat evidence"]
    applied = sum(1 for r in rows if str(r.get("shared_prefix_applied", "")).lower() in ("true", "1"))
    frac = applied / n
    msgs.append(f"    shared_prefix_applied: {frac:.2f} of {n} (target ~{target_prefix_frac})")
    if (not math.isfinite(target_prefix_frac) or not 0 <= target_prefix_frac <= 1 or
            (target_prefix_frac == 0 and applied != 0) or abs(frac - target_prefix_frac) > 0.15):
        ok = False
        msgs.append("    FAIL: prefix fraction far from target")
    arr_path = run_dir / "client" / "arrival_stats.json"
    if arr_path.exists():
        arr = json.loads(arr_path.read_text())
        cv = arr.get("interarrival_cv")
        try:
            cv_number = float(cv)
        except (ValueError, TypeError):
            cv_number = float("nan")
        msgs.append(f"    arrival_mode={arr.get('arrival_mode')} realized CoV={cv} rate={arr.get('realized_rate_rps')}")
        if target_arrival_mode is not None and arr.get("arrival_mode") != target_arrival_mode:
            ok = False
            msgs.append("    FAIL: realized arrival mode differs from the recorded workload")
        if arr.get("arrival_mode") == "bursty" and (not math.isfinite(cv_number) or cv_number < 1.5):
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
    try:
        indicators = list(csv.DictReader(io.StringIO(out.stdout)))
        uss = next(row for row in indicators if row.get("indicator") == "proc.uss_bytes")
        has_uss = (int(uss["n_samples"]) >= 10 and all(
            math.isfinite(float(uss[field])) for field in
            ("slope", "slope_ci_lo", "slope_ci_hi", "mk_p_value")))
    except (StopIteration, KeyError, TypeError, ValueError):
        has_uss = False
    n_rows = max(0, len(out.stdout.splitlines()) - 1)
    msg = [f"    aging_trends: {n_rows} indicator rows, proc.uss_bytes={'usable' if has_uss else 'missing/insufficient/non-finite'}"]
    return has_uss, msg


def main() -> None:
    p = argparse.ArgumentParser(description="STEP 1 validation gate for an extension run.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--target-prefix-frac", type=float, default=None,
                   help="Override the target saved in client_config.yaml / manifest.")
    p.add_argument("--scope-only", action="store_true",
                   help="Complete short-run integrity + whole-container evidence; no statistical trend test.")
    args = p.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    monitors = manifest.get("monitors", {})
    comp_spec = monitors.get("components") if isinstance(monitors, dict) else None
    results: list[tuple[str, bool]] = []

    print(f"== Validation gate: {args.run_dir} ==")
    print(f"proc_prefix = {manifest.get('proc_prefix')}  attach={manifest.get('attach')}")
    if args.scope_only and (not comp_spec or comp_spec.get("scope_type") != "container_tree"):
        p.error("--scope-only requires a whole-container run")
    print("\n[0] Common run integrity (independent of aging significance):")
    intact, integrity_messages = check_run_integrity(args.run_dir, manifest)
    print("\n".join(integrity_messages))
    if not intact:
        print("GATE: FAIL")
        sys.exit(1)
    results.append(("integrity", True))

    if comp_spec and comp_spec.get("scope_type") == "container_tree":
        print("\n[1-2] Whole-container scope, model coverage and component sums:")
        ok, msgs = check_container_scope(args.run_dir, comp_spec)
        print("\n".join(msgs)); results.append(("container_scope", ok))
        devices = comp_spec.get("gpu_devices", [])
        okg, mg = check_multi_gpu(args.run_dir, devices)
        print("\n".join(mg)); results.append(("gpu", okg and bool(devices)))
    elif comp_spec:
        components = comp_spec["components"]
        engine_group = comp_spec.get("engine_group", "engine")
        gpu_devices = manifest.get("gpu_devices") or manifest.get("engine", {}).get("gpu_devices")
        if not gpu_devices:
            topology = manifest.get("topology") or manifest.get("engine", {}).get("topology") or {}
            gpu_devices = sorted({topology[k] for k in ("prefill_gpu", "decode_gpu") if k in topology})
        print("\n[1] Component PID map (eyeball vs `ps -eo pid,cmd | grep dynamo`):")
        ok, msgs = check_component_pid_map(args.run_dir, components); print("\n".join(msgs)); results.append(("component_pid_map", ok))
        print("\n[2] Engine USS aggregate + infra + multi-GPU:")
        ok2, m2 = check_aggregate(args.run_dir, engine_group, components); print("\n".join(m2))
        okg, mg = check_multi_gpu(args.run_dir, gpu_devices); print("\n".join(mg))
        results.append(("aggregate", ok2)); results.append(("multi_gpu", okg))
        print("\n[2b] No stray processes outside recorded pgids (orphan red flag):")
        oko, mo = check_no_orphans(args.run_dir, components); print("\n".join(mo)); results.append(("no_orphans", oko))

    if args.scope_only:
        passed = all(ok for _, ok in results)
        print("SCOPE GATE:", "PASS" if passed else "FAIL")
        sys.exit(0 if passed else 1)

    print("\n[3] Client features (prefix-repeat + burst):")
    try:
        config = saved_client_config(args.run_dir, manifest)
        target = float(args.target_prefix_frac if args.target_prefix_frac is not None else config["prefix_repeat_fraction"])
        okc, mc = check_client_features(args.run_dir, target, config.get("arrival_mode", "poisson"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        okc, mc = False, [f"    FAIL: cannot resolve recorded prefix target: {exc}"]
    print("\n".join(mc)); results.append(("client_features", okc))

    print("\n[4] Analysis pipeline end-to-end:")
    oka, ma = check_analysis(args.run_dir, args.repo_root); print("\n".join(ma)); results.append(("analysis", oka))

    # [5] GPU ECC drift (item 4b): uncorrected increment -> gate FAIL; corrected
    # increment -> WARN (printed, does not fail the gate).
    print("\n[5] GPU ECC drift:")
    ecc, mecc = ecc_verdict(args.run_dir); print("\n".join(mecc))
    results.append(("ecc", ecc != "fail"))

    print("\n== GATE SUMMARY ==")
    allok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        allok = allok and ok
    print("GATE:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
