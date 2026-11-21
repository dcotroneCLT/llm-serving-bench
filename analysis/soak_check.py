#!/usr/bin/env python3
"""Acceptance check for the W1 whole-container scope soak (R4).

A soak run (cells soak_triton / soak_vllm, 6 h on the p16 shape) must show that
the whole-container v2 scope holds for hours before a window commits to 36 h
runs, where an identity-loss latch would abort the run. PASS requires ALL of:

  0. analysis/validate_extension_run.py --scope-only passes (run integrity,
     container identity, disjoint roles, memory sums) -- skipped with
     --skip-validator, e.g. when it was already run;
  1. the run ACTUALLY lasted at least --min-hours (a short validation run is not
     a soak): the effective duration is the shortest of the recorded wall time
     (duration_seconds_actual), the manifest window (ended_at - started_at) and
     the sampled span of the scope aggregate (last - first tick + one period),
     and the aggregate has no sampling gap longer than MAX_GAP_PERIODS periods.
     The DECLARED duration_s is reported but never accepted as evidence;
  2. ZERO identity-loss / latched events: no tick carries a scope_error of the
     identity family (lost/replaced, PID reused, role change, membership change
     during sampling);
  3. complete scope on ALL ticks (100%, not the 95% of the generic gate):
     process_alive, membership_complete and model_scope_complete true, no
     scope_error, three complete components;
  4. on every tick the model-holding GPU process(es) lie inside the ENGINE
     component's PID set (the engine subtree) and among the GPU compute PIDs;
  5. process-set changes are LISTED from scope_membership.jsonl (identities
     added / removed between audit records, layout-only changes); any removed
     identity fails the check (it would have latched a production run).

Usage:
  python3 analysis/soak_check.py --run-dir <run_dir> [--min-hours 6] [--json out.json]
Exit 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_extension_run import _pids, _read_csvs  # noqa: E402

IDENTITY_LOSS_RE = re.compile(
    r"identity lost|lost/replaced|reused|role change|membership changed during sampling")


MAX_GAP_PERIODS = 3          # a longer hole in the aggregate breaks the 100% claim
DEFAULT_PERIOD_S = 5.0


def _true(value) -> bool:
    return str(value).strip().lower() in ("true", "1")


def membership_changes(run_dir: Path) -> tuple[list[dict], list[str]]:
    """Consecutive diffs of scope_membership.jsonl: [{ts, added, removed,
    layout_only}], plus human-readable lines."""
    path = run_dir / "scope_membership.jsonl"
    if not path.exists():
        raise ValueError("scope_membership.jsonl missing")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("scope_membership.jsonl is empty")
    changes, lines = [], []

    def identities(rec):
        return {f"{p['pid']}@{p['create_time']}": p for p in rec["processes"]}

    first = identities(records[0])
    roles: dict[str, int] = {}
    for p in first.values():
        roles[p.get("role", "?")] = roles.get(p.get("role", "?"), 0) + 1
    lines.append(f"    initial set at {records[0]['ts_unix']}: {len(first)} processes "
                 + ", ".join(f"{r}={n}" for r, n in sorted(roles.items())))
    previous = first
    for rec in records[1:]:
        current = identities(rec)
        added = sorted(set(current) - set(previous))
        removed = sorted(set(previous) - set(current))
        change = {"ts_unix": rec["ts_unix"], "added": added, "removed": removed,
                  "layout_only": not added and not removed}
        changes.append(change)
        if change["layout_only"]:
            lines.append(f"    {rec['ts_unix']}: layout change only (ppid / role / GPU set)")
        for key in added:
            p = current[key]
            lines.append(f"    {rec['ts_unix']}: + {key} role={p.get('role')} cmd={p.get('cmdline', '')[:80]}")
        for key in removed:
            p = previous[key]
            lines.append(f"    {rec['ts_unix']}: - {key} role={p.get('role')} cmd={p.get('cmdline', '')[:80]}")
        previous = current
    return changes, lines


def effective_duration(manifest: dict, rows: list[dict], period: float,
                       min_hours: float) -> tuple[bool, list[str], dict]:
    """Effective (not declared) soak duration; see check 1 in the module doc."""
    need = min_hours * 3600
    declared = float(manifest.get("duration_s") or 0)
    report = {"declared_duration_h": declared / 3600}
    try:
        start = float(manifest["started_at_unix"])
        wall = min(float(manifest["duration_seconds_actual"]),
                   float(manifest["ended_at_unix"]) - start)
    except (KeyError, TypeError, ValueError):
        report["duration_h"] = None
        return False, ["    FAIL: run not finalized (duration_seconds_actual / started_at / "
                       "ended_at missing): effective duration unknown"], report
    stamps = sorted(float(r["ts_unix"]) for r in rows)
    span = stamps[-1] - stamps[0] + period if stamps else 0.0
    max_gap = max((b - a for a, b in zip(stamps, stamps[1:])), default=0.0)
    effective = min(wall, span)
    report.update(duration_h=effective / 3600, wall_duration_h=wall / 3600,
                  sampled_span_h=span / 3600, max_sampling_gap_s=max_gap)
    ok, msgs = True, []
    detail = (f"wall {wall / 3600:.4f} h, sampled span {span / 3600:.4f} h, "
              f"declared {declared / 3600:.2f} h")
    if effective < need:
        ok = False
        msgs.append(f"    FAIL: effective duration {effective / 3600:.4f} h < {min_hours} h soak "
                    f"({detail})")
    else:
        msgs.append(f"    effective duration: {effective / 3600:.4f} h (>= {min_hours} h; {detail})")
    if max_gap > MAX_GAP_PERIODS * period:
        ok = False
        msgs.append(f"    FAIL: sampling gap of {max_gap:.0f} s in the scope aggregate "
                    f"(> {MAX_GAP_PERIODS} x {period:g} s period)")
    return ok, msgs, report


def check_soak(run_dir: Path, min_hours: float = 6.0) -> tuple[bool, list[str], dict]:
    msgs: list[str] = []
    report: dict = {"run_dir": str(run_dir)}
    ok = True
    manifest = json.loads((run_dir / "manifest.json").read_text())
    spec = (manifest.get("monitors") or {}).get("components") or {}
    if spec.get("scope_type") != "container_tree" or spec.get("scope_version") != 2:
        return False, ["    FAIL: not a whole-container v2 run"], report

    period = float(((manifest.get("monitors") or {}).get("proc") or {}).get("period_s")
                   or DEFAULT_PERIOD_S)
    engine_label = next(c["label"] for c in spec["components"] if c["role"] == "engine")
    engine_pids = {r["ts_unix"]: _pids(r.get("pids")) for r in _read_csvs(run_dir, engine_label)}
    rows = _read_csvs(run_dir, f"agg_{spec['engine_group']}")
    identity_events, incomplete, model_outside = [], [], []
    for row in rows:
        ts, error = row["ts_unix"], row.get("scope_error") or ""
        if IDENTITY_LOSS_RE.search(error):
            identity_events.append((ts, error))
        complete = (_true(row.get("process_alive")) and _true(row.get("membership_complete"))
                    and _true(row.get("model_scope_complete")) and not error
                    and str(row.get("n_components_complete")) == "3")
        if not complete:
            incomplete.append(ts)
            continue
        model, gpu = _pids(row.get("model_pids")), _pids(row.get("gpu_pids"))
        if not model or not model <= gpu or not model <= engine_pids.get(ts, set()):
            model_outside.append(ts)
    report.update(ticks=len(rows), identity_loss_events=len(identity_events),
                  incomplete_ticks=len(incomplete), model_outside_engine_ticks=len(model_outside))
    dur_ok, dur_msgs, dur_report = effective_duration(manifest, rows, period, min_hours)
    ok &= dur_ok
    msgs += dur_msgs
    report.update(dur_report)
    if not rows:
        ok = False
        msgs.append("    FAIL: no aggregate rows")
    if identity_events:
        ok = False
        first_ts, first_err = identity_events[0]
        msgs.append(f"    FAIL: {len(identity_events)} identity-loss/latched tick(s); first at "
                    f"{first_ts}: {first_err[:160]}")
    else:
        msgs.append("    identity-loss / latched events: 0")
    if incomplete:
        ok = False
        msgs.append(f"    FAIL: {len(incomplete)}/{len(rows)} tick(s) without complete scope "
                    f"(first at {incomplete[0]})")
    else:
        msgs.append(f"    complete scope on all {len(rows)} ticks")
    if model_outside:
        ok = False
        msgs.append(f"    FAIL: model-holding GPU process outside the engine subtree on "
                    f"{len(model_outside)} tick(s) (first at {model_outside[0]})")
    elif rows:
        msgs.append("    model-holding GPU process inside the engine subtree on every tick")

    try:
        changes, lines = membership_changes(run_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        ok = False
        msgs.append(f"    FAIL: process-set audit: {exc}")
        changes, lines = [], []
    added = sum(len(c["added"]) for c in changes)
    removed = sum(len(c["removed"]) for c in changes)
    report.update(process_set_changes=changes, identities_added=added, identities_removed=removed)
    msgs.append(f"    process-set changes from scope_membership.jsonl: {len(changes)} "
                f"(identities +{added} / -{removed})")
    msgs += lines
    if removed:
        ok = False
        msgs.append("    FAIL: process identities disappeared during the soak (a production run "
                    "would have latched and aborted)")
    report["pass"] = ok
    return ok, msgs, report


def run_scope_only_validator(run_dir: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "validate_extension_run.py"),
         "--run-dir", str(run_dir), "--scope-only"],
        capture_output=True, text=True, timeout=1800)
    return proc.returncode == 0, proc.stdout + proc.stderr


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="W1 whole-container scope soak acceptance (R4).")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--min-hours", type=float, default=6.0)
    p.add_argument("--skip-validator", action="store_true",
                   help="Do not run validate_extension_run.py --scope-only first.")
    p.add_argument("--json", type=Path, default=None, help="Also write the report as JSON.")
    args = p.parse_args(argv)

    print(f"== Soak check: {args.run_dir} ==")
    results = []
    if not args.skip_validator:
        print("\n[0] validate_extension_run.py --scope-only:")
        okv, out = run_scope_only_validator(args.run_dir)
        print("\n".join("    " + line for line in out.strip().splitlines()[-12:]))
        results.append(("scope_only_validator", okv))
    print("\n[1-5] Soak acceptance:")
    try:
        ok, msgs, report = check_soak(args.run_dir, args.min_hours)
    except (OSError, KeyError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        ok, msgs, report = False, [f"    FAIL: {exc!r}"], {"pass": False}
    print("\n".join(msgs))
    results.append(("soak", ok))
    if args.json:
        args.json.write_text(json.dumps({**report, "results": dict(results)}, indent=2) + "\n")
    passed = all(r for _, r in results)
    print("\nSOAK:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
