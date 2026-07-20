#!/usr/bin/env python3
"""Offline re-evaluation of calibration verdicts (selector-only, no re-sweep).

The v2 MEASUREMENT (the sweep rows) is sound; only the ceiling SELECTOR changed
(v2.0 contiguous-stable-prefix -> v2.1 bracket; see calibrate_rate.select_ceiling
for why). This tool recomputes the verdict of an already-recorded calibration
JSON from its OWN recorded sweep rows with the fixed selector, and republishes in
place -- so the fresh sweeps of a running calibration pass can be re-verdicted
WITHOUT re-running any GPU-hours.

It is deliberately conservative about what it will touch:
  - Only measurement v2 files (integer method version EXACTLY 2). A v1 (finite-
    window-biased) file is SKIPPED: re-verdicting a biased measurement is unsound
    -- it must be re-SWEPT, not re-selected. A v>2 file is SKIPPED too: its rows
    are a newer measurement this v2-only tool does not understand, so re-verdicting
    would silently down-stamp it to 2.1 -- upgrade the tool instead.
  - A file with no recorded sweep rows is SKIPPED (nothing to select from).
  - An `engine_failure` file is SKIPPED: that status is a dead/sick-endpoint
    classification the orchestrator makes (calibrate_dow), not a verdict derivable
    from the rows -- re-verdicting it would silently erase the failure. Such a
    cell needs a fresh sweep, not a re-selection.

On a processed file it recomputes status + ceiling_rps/ceiling_offered_rps/
rate_calibrated_rps (rate_calibrated = fraction x achieved-at-ceiling, using the
file's OWN recorded fraction and criteria), stamps calibration_method_version=2.1
and selector_version, and adds a `reevaluation` block (when, from which
status/version, the tool version). The original sweep MEASUREMENTS are preserved
untouched; the selector's per-row annotations (stable/failed_criteria/
climb_inconclusive/low_rate_anomaly) are (re)written onto the rows. Provenance is
untouched. The write is atomic (temp + os.replace in the same directory).

Usage:
  # dry-run: print old -> new for every file, write nothing
  python3 scripts/reeval_calibration.py --calibration 'runs/.../calibration_*.json' --dry-run
  # republish in place
  python3 scripts/reeval_calibration.py --calibration cal_a.json --calibration 'glob/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# scripts/ is sys.path[0] when run directly; reuse the SAME selector the fresh
# sweeps use -- no reimplementation.
import calibrate_rate as cr

REPO_ROOT = Path(__file__).resolve().parent.parent
REEVAL_TOOL_VERSION = 1


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand each --calibration arg as a glob (a literal path with no wildcard
    matches itself). De-duplicated, sorted, so a re-run is deterministic."""
    seen: dict[str, Path] = {}
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches and ("*" not in pat and "?" not in pat and "[" not in pat):
            matches = [pat]  # a literal path that does not exist yet -> report later
        for m in matches:
            seen[str(Path(m))] = Path(m)
    return [seen[k] for k in sorted(seen)]


def _criteria_kwargs(calib: dict, climb_min_samples: Optional[int]) -> dict:
    """Map the file's recorded `criteria` block to select_ceiling kwargs, falling
    back to select_ceiling's own defaults for anything absent. climb_min_samples
    is a v2.1 selector knob absent from v2.0 files -> use the CLI value or the
    default so the gate is applied on re-evaluation."""
    c = calib.get("criteria") or {}
    kw: dict = {}
    if "achieved_ratio_min" in c:
        kw["achieved_ratio_min"] = float(c["achieved_ratio_min"])
    if "drop_max" in c:
        kw["drop_max"] = float(c["drop_max"])
    if "p99_bound_s" in c:
        kw["p99_bound"] = float(c["p99_bound_s"])
    if "latency_climb_frac" in c:
        kw["climb_frac"] = float(c["latency_climb_frac"])
    if "offered_span_min" in c:
        kw["offered_span_min"] = float(c["offered_span_min"])
    if climb_min_samples is not None:
        kw["climb_min_samples"] = int(climb_min_samples)
    elif "climb_min_samples" in c:
        kw["climb_min_samples"] = int(c["climb_min_samples"])
    return kw


def _measurement_version(calib: dict) -> Optional[int]:
    """Integer (measurement) part of calibration_method_version; None if it cannot
    be parsed. Absent field -> treated as v1 (the historical unversioned method)."""
    raw = calib.get("calibration_method_version", 1)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


class Skip(RuntimeError):
    """This file must not be re-evaluated; carries a short human reason."""


def reevaluate(calib: dict, climb_min_samples: Optional[int],
               now_unix: float) -> dict:
    """Recompute the verdict in place from the recorded rows and stamp the
    reevaluation provenance. Raises Skip when the file must not be touched.
    Returns the (mutated) calib dict."""
    if (calib.get("status") or "").strip().lower() == "engine_failure":
        raise Skip("engine_failure (needs a fresh sweep, not a re-verdict)")
    mv = _measurement_version(calib)
    if mv is None:
        raise Skip(f"unparseable method version "
                   f"{calib.get('calibration_method_version')!r}")
    if mv < 2:
        raise Skip(f"measurement version {mv} < 2 "
                   "(must be re-swept, not re-selected)")
    if mv > 2:
        # A future measurement carries rows this v2-only selector does not
        # understand; re-verdicting would silently down-stamp it to 2.1.
        raise Skip(f"measurement version {mv} > 2 (newer than this reeval tool, "
                   "which understands only v2 rows -- upgrade the tool, do not "
                   "down-verdict a future measurement)")
    rows = calib.get("sweep")
    if not rows:
        raise Skip("no recorded sweep rows")

    ceiling, status = cr.select_ceiling(rows, **_criteria_kwargs(calib, climb_min_samples))

    old_status = calib.get("status")
    old_method = calib.get("calibration_method_version")
    old_selector = calib.get("selector_version")
    fraction = calib.get("fraction")
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        raise Skip(f"unusable fraction {fraction!r}")

    calib["status"] = status
    if ceiling is not None:
        calib["ceiling_rps"] = ceiling["achieved_rps"]
        calib["ceiling_offered_rps"] = ceiling["offered_rate"]
        calib["rate_calibrated_rps"] = round(ceiling["achieved_rps"] * fraction, 4)
    else:
        calib["ceiling_rps"] = None
        calib["ceiling_offered_rps"] = None
        calib["rate_calibrated_rps"] = None
    calib["calibration_method_version"] = cr.CALIBRATION_METHOD_VERSION
    calib["selector_version"] = cr.CALIBRATION_SELECTOR_VERSION
    calib["reevaluation"] = {
        "reevaluated_at_unix": round(now_unix, 3),
        "reevaluated_at_iso": datetime.fromtimestamp(now_unix, timezone.utc).isoformat(timespec="seconds"),
        "tool_version": REEVAL_TOOL_VERSION,
        "from_status": old_status,
        "from_method_version": old_method,
        "from_selector_version": old_selector,
        "to_selector_version": cr.CALIBRATION_SELECTOR_VERSION,
    }
    return calib


def publish(path: Path, calib: dict) -> None:
    """Atomic in-place republish (temp + os.replace in the same directory)."""
    tmp = path.with_name(path.name + ".reeval.tmp")
    tmp.write_text(json.dumps(calib, indent=2))
    os.replace(str(tmp), str(path))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-verdict recorded calibration JSONs with the fixed (bracket) "
                    "selector, without re-sweeping.")
    p.add_argument("--calibration", action="append", default=[], required=True,
                   help="Calibration JSON path or glob (repeatable).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print old -> new per file and write nothing.")
    p.add_argument("--climb-min-samples", type=int, default=None,
                   help="Override the climb min-sample gate for re-evaluation "
                        "(default: the file's recorded value, else the selector default).")
    args = p.parse_args()

    paths = expand_paths(args.calibration)
    if not paths:
        print("[reeval] no files matched", file=sys.stderr)
        sys.exit(2)

    now = time.time()
    changed = missing = skipped = errored = 0
    for path in paths:
        if not path.exists():
            print(f"[reeval] MISSING {path}")
            missing += 1
            continue
        try:
            calib = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[reeval] ERROR  {path}: unreadable ({e})")
            errored += 1
            continue
        old_status = calib.get("status")
        try:
            reevaluate(calib, args.climb_min_samples, now)
        except Skip as s:
            print(f"[reeval] SKIP   {path}: {s}")
            skipped += 1
            continue
        except Exception as e:  # noqa: BLE001
            # One malformed file (e.g. a KeyError on an incomplete sweep row) must
            # NOT abort the whole glob: log it, count it, carry on so the other
            # calibrations of a running pass still get re-verdicted. The batch
            # exits non-zero at the end (errored != 0) so the failure is not lost.
            print(f"[reeval] ERROR  {path}: re-evaluation failed "
                  f"({type(e).__name__}: {e})")
            errored += 1
            continue
        new_status = calib.get("status")
        if not args.dry_run:
            try:
                publish(path, calib)
            except OSError as e:
                print(f"[reeval] ERROR  {path}: publish failed ({e})")
                errored += 1
                continue
        verb = "DRY-RUN" if args.dry_run else "WROTE  "
        arrow = f"{old_status} -> {new_status}"
        detail = ""
        if new_status == "ok":
            detail = (f"  ceiling_offered={calib.get('ceiling_offered_rps')} "
                      f"rate_calibrated={calib.get('rate_calibrated_rps')}")
        flip = "  [CHANGED]" if old_status != new_status else ""
        print(f"[reeval] {verb} {path}: {arrow}{detail}{flip}")
        if old_status != new_status:
            changed += 1

    print(f"[reeval] {'(dry-run) ' if args.dry_run else ''}"
          f"{len(paths)} matched: {changed} status-changed, {skipped} skipped, "
          f"{missing} missing, {errored} errored")
    # Non-zero only on hard problems (missing/unreadable inputs), not on a
    # legitimate no-change re-evaluation.
    sys.exit(1 if (missing or errored) else 0)


if __name__ == "__main__":
    main()
