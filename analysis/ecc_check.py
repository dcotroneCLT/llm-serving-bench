"""Post-run GPU ECC drift check (PRE-CAMPAIGN HARDENING item 4b).

Shared by the two run validators. Reads the volatile ECC counter columns that
monitoring/gpu_monitor.py records:

  ecc_db_volatile -- uncorrected (double-bit) volatile ECC errors
  ecc_sb_volatile -- corrected   (single-bit) volatile ECC errors

An UNCORRECTED-ECC increment during the run means device memory was silently
corrupted: the run's numbers cannot be trusted -> FAIL. A CORRECTED-ECC
increment is recoverable but is an early hardware-degradation signal worth
surfacing before a months-long campaign -> WARN.

gpu_monitor records only the VOLATILE counters (not the aggregate lifetime
ones), so this maps the requested "volatile-uncorrected fail / aggregate-
corrected warn" severity onto the columns actually present: uncorrected-volatile
FAIL, corrected-volatile WARN. Columns absent or blank (ECC disabled /
unsupported on the device) -> OK with an explicit skipped note (never a silent
pass that hides "we never looked"). Stdlib only, so both validators can import
it without pulling in pandas/scipy.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Optional

UNCORRECTED_COL = "ecc_db_volatile"   # double-bit, uncorrected
CORRECTED_COL = "ecc_sb_volatile"     # single-bit, corrected


def _int(x) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def scan_ecc(run_dir: Path) -> dict:
    """Aggregate the two ECC columns across all gpu*_*.csv chunks, grouped by
    gpu_index. Returns {gpu_index: {col: {min,max,last,n}}} over numeric samples.
    A counter is per-device, so per-gpu min/max is the right granularity."""
    per_gpu: dict[str, dict] = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "gpu*_*.csv"))):
        try:
            with open(f, newline="") as fp:
                for r in csv.DictReader(fp):
                    gi = (r.get("gpu_index") or "").strip() or "?"
                    for col in (UNCORRECTED_COL, CORRECTED_COL):
                        v = _int(r.get(col))
                        if v is None:
                            continue
                        d = per_gpu.setdefault(gi, {}).setdefault(
                            col, {"min": v, "max": v, "last": v, "n": 0})
                        d["min"] = min(d["min"], v)
                        d["max"] = max(d["max"], v)
                        d["last"] = v
                        d["n"] += 1
        except OSError:
            continue
    return per_gpu


def ecc_verdict(run_dir: Path) -> tuple[str, list[str]]:
    """Return ('ok' | 'warn' | 'fail', messages). An uncorrected-volatile ECC
    increment on ANY device -> 'fail'; a corrected-volatile increment (with no
    uncorrected) -> 'warn'; none / no data -> 'ok'."""
    per_gpu = scan_ecc(run_dir)
    if not per_gpu:
        return "ok", ["    ECC: no numeric ecc_*_volatile samples "
                      "(ECC disabled/unsupported, or no gpu CSV) -> skipped"]
    verdict = "ok"
    msgs: list[str] = []
    for gi in sorted(per_gpu):
        cols = per_gpu[gi]
        unc = cols.get(UNCORRECTED_COL)
        cor = cols.get(CORRECTED_COL)
        unc_inc = (unc["max"] - unc["min"]) if unc else 0
        cor_inc = (cor["max"] - cor["min"]) if cor else 0
        if unc_inc > 0:
            verdict = "fail"
            msgs.append(f"    FAIL gpu{gi}: uncorrected (double-bit) volatile ECC "
                        f"incremented by {unc_inc} during the run -> memory "
                        f"corruption, run INVALID")
        elif cor_inc > 0:
            if verdict != "fail":
                verdict = "warn"
            msgs.append(f"    WARN gpu{gi}: corrected (single-bit) volatile ECC "
                        f"incremented by {cor_inc} during the run -> early "
                        f"hardware-degradation signal")
        else:
            unc_last = unc["last"] if unc else "n/a"
            cor_last = cor["last"] if cor else "n/a"
            msgs.append(f"    gpu{gi}: no ECC increment "
                        f"(uncorrected={unc_last}, corrected={cor_last})")
    return verdict, msgs
