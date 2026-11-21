#!/usr/bin/env python3
"""Generate the Triton/vLLM DoW RERUN as three <= 25-day machine windows.

SINGLE SOURCE OF TRUTH for campaign `extension_dow_rescoped`. Everything below
is GENERATED -- never hand-edit it, regenerate:

    python3 scripts/generate_rerun_windows.py           # write
    python3 scripts/generate_rerun_windows.py --check   # verify up to date (CI)

Outputs (under campaigns/extension/):
    dow_rerun_w1.yaml, dow_rerun_w2.yaml, dow_rerun_w3.yaml   window descriptors
    dow_rerun_w1_soak.yaml                                     W1 scope soak (R4)
    rerun_design_matrix.csv                                    window/position/levels
    cells/rerun/dow_dynamo_disagg_cp4.yaml                     Dynamo bridge CP
    cells/rerun/soak_triton.yaml, cells/rerun/soak_vllm.yaml   soak cells

WHY. The original screening (campaign extension_dow_screening, CLOSED) sampled
Triton and vLLM standalone with an incomplete process scope. Their 38 runs
(16 design points + 3 center points per system, 36 h each, ~57 days serial) are
re-run with the whole-container v2 scope. The host cannot be held for 57 days,
so the rerun is split into three machine windows of at most 25 days, each with
its own state file, its own JUST-IN-TIME calibration directory and a 30-day
calibration age limit.

CELLS. The existing generated container_tree v2 cells (cells/dow/dow_triton_*,
cells/dow/dow_vllm_*) are reused AS IS: same cell ids, so the analysis joins the
design matrix unchanged; the new campaign_id keeps the run dirs
(extension_dow_rescoped_<cell>_r01) distinct from the closed campaign's. One NEW
Dynamo center point, dow_dynamo_disagg_cp4 (48 h, the cp1-cp3 template and
workload), runs in W3 as the bridge between the completed Dynamo arm and the
rerun epoch, with the opt-in per-process split (R5) on.

BLOCKING (the window allocation). For each system the 16 points of the
2^(5-1) design (I=ABCDE, E=ABCD) are cut into four QUARTER-BLOCKS by the signs of
the AC and AD contrasts; their generalized interaction is CD. Within a quarter
block AC and AD are constant, so a quarter block (or any union of them) is
orthogonal to all five main effects and to the seven two-factor interactions
other than AC, AD, CD; only AC, AD, CD (aliased with BDE, BCE, ABE) can be
correlated with window membership. Allocation:
    Triton: W1 = {AC+AD+, AC+AD-} (8 points), W2 = {AC-AD+} (4), W3 = {AC-AD-} (4)
    vLLM:   W1 = {AC+AD-} (4), W2 = {AC+AD+, AC-AD+} (8), W3 = {AC-AD-} (4)
The two systems use different unions on purpose: a window shift that the CP
correction left uncorrected would bias AC (Triton W1 vs rest) and AD (vLLM W2
vs rest) respectively, not the same contrast in both. Each window holds exactly
one center point per system (cp1 in W1, cp2 in W2, cp3 in W3) in the middle of
its sequence; W3 also holds dow_dynamo_disagg_cp4 at its exact middle. The CPs
estimate the window offsets (analysis/window_effects.py).

ORDER inside a window (fixed seed RERUN_SEED): each system's design points are
shuffled; the minority system is spread through the majority as evenly as the
8/4 split allows (centered Bresenham positions); the CP block sits in the
middle. replicas_per_cell 1, so campaign.py runs the cells: list in order.

PROJECTION (asserted here and in tests): runs + calibration (17 h per 14 cells,
the measured v2 pace of ~72 h / 57 cells) + the W1 soak (6 h per system) + a
per-run overhead (bring-up, client drain, teardown, VRAM quiescence, inter-run
cooldown) + window setup (reboot, environment snapshot, dry-runs) must leave at
least max(1.5 days, the longest run of the window) of retry margin inside 25
days. The generator refuses to emit a plan that violates it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import random
import sys
from pathlib import Path

import yaml  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_dow_cells as gdc  # noqa: E402

REPO = gdc.REPO
EXT = gdc.EXT

CAMPAIGN_ID = "extension_dow_rescoped"
# 20261120 = the date the rescoped rerun was designed. Changes the order inside
# each window, never the allocation of points to windows.
RERUN_SEED = 20261120
WINDOWS = (1, 2, 3)
RERUN_SYSTEMS = ("triton", "vllm")

CALIBRATION_DIR = "state/calibration_rerun/w{k}"
CALIBRATION_MAX_AGE_DAYS = 30
CALIBRATION_MIN_METHOD_VERSION = 2

# Quarter blocks keyed by (sign of AC, sign of AD) -> window, per system.
QUARTER_WINDOW = {
    "triton": {(1, 1): 1, (1, -1): 1, (-1, 1): 2, (-1, -1): 3},
    "vllm": {(1, -1): 1, (1, 1): 2, (-1, 1): 2, (-1, -1): 3},
}
WINDOW_INTERACTIONS = ("AC", "AD", "CD")  # the only contrasts windows may touch

BRIDGE_CELL = "dow_dynamo_disagg_cp4"
BRIDGE_WINDOW = 3

# Soak (R4): W1 only, one per system, before calibration.
SOAK_CELLS = {"triton": "soak_triton", "vllm": "soak_vllm"}
SOAK_DURATION_S = 21_600            # 6 h
SOAK_WARMUP_DISCARD_S = 3_600
SOAK_SHAPE_YATES = 16               # p16: A=B=C=D=E=+1, the heaviest shape
# Fixed, uncalibrated rate: ~2/3 of the LOWEST p16 ceiling measured on this host
# (Dynamo p16, 0.150 rps); Triton/vLLM ceilings are higher. Modest on purpose:
# the soak tests process-set stability, not throughput.
SOAK_RATE_RPS = 0.10

# Projection constants (hours).
CALIBRATION_H_PER_CELL = 17.0 / 14.0
RUN_OVERHEAD_H = 0.5
WINDOW_SETUP_H = 2.0
WINDOW_MAX_DAYS = 25.0
MIN_RETRY_MARGIN_DAYS = 1.5

CELLS_SUBDIR = Path("cells") / "rerun"
DESCRIPTOR = "dow_rerun_w{k}.yaml"
SOAK_DESCRIPTOR = "dow_rerun_w1_soak.yaml"
MATRIX_CSV = "rerun_design_matrix.csv"


class PlanError(ValueError):
    """The generated plan violates a design or duration requirement."""


# --- Design allocation --------------------------------------------------------


def interaction(codes: dict, name: str) -> int:
    value = 1
    for factor in name:
        value *= codes[factor]
    return value


def quarter_of(codes: dict) -> tuple[int, int]:
    return interaction(codes, "AC"), interaction(codes, "AD")


def quarter_label(q: tuple[int, int]) -> str:
    return f"AC{'+' if q[0] > 0 else '-'}AD{'+' if q[1] > 0 else '-'}"


def window_points(system: str) -> dict[int, list[dict]]:
    """{window: [design rows]} for one system, in Yates order."""
    out: dict[int, list[dict]] = {k: [] for k in WINDOWS}
    for row in gdc.coded_design():
        out[QUARTER_WINDOW[system][quarter_of(row)]].append(row)
    return out


def _spread(n: int, m: int) -> set[int]:
    """Positions of m minority items among n slots, centered Bresenham."""
    return {int((i + 0.5) * n / m) for i in range(m)} if m else set()


def window_order(seed: int = RERUN_SEED) -> dict[int, list[tuple[str, str]]]:
    """{window: [(system, cell_id), ...]} in execution order."""
    rng = random.Random(seed)
    alloc = {system: window_points(system) for system in RERUN_SYSTEMS}
    out: dict[int, list[tuple[str, str]]] = {}
    for k in WINDOWS:
        seqs = {}
        for system in RERUN_SYSTEMS:
            ids = [gdc.cell_id(system, "p", row["yates"]) for row in alloc[system][k]]
            rng.shuffle(ids)
            seqs[system] = ids
        major, minor = sorted(RERUN_SYSTEMS, key=lambda s: (-len(seqs[s]), RERUN_SYSTEMS.index(s)))
        n = len(seqs[major]) + len(seqs[minor])
        minor_slots = _spread(n, len(seqs[minor]))
        it = {major: iter(seqs[major]), minor: iter(seqs[minor])}
        design = [(minor if i in minor_slots else major) for i in range(n)]
        design = [(s, next(it[s])) for s in design]
        cps = [(system, gdc.cell_id(system, "cp", k)) for system in RERUN_SYSTEMS]
        if rng.random() < 0.5:
            cps.reverse()
        if k == BRIDGE_WINDOW:
            cps = [cps[0], ("dynamo_disagg", BRIDGE_CELL), cps[1]]
        half = n // 2
        out[k] = design[:half] + cps + design[half:]
    return out


# --- Cells --------------------------------------------------------------------


def _codes_of(cid: str) -> dict:
    if cid.endswith(("_cp1", "_cp2", "_cp3", "_cp4")):
        return dict(gdc.CENTER_CODES, yates=None)
    yates = int(cid.rsplit("_p", 1)[1])
    row = gdc.coded_design()[yates - 1]
    return {f: row[f] for f in ("A", "B", "C", "D", "E", "yates")}


def bridge_cell_doc() -> dict:
    template = gdc._load_template("dynamo_disagg")
    codes = dict(gdc.CENTER_CODES, yates=None)
    doc = gdc.build_cell_doc("dynamo_disagg", template, BRIDGE_CELL, "center",
                             codes, gdc.physical(codes), gdc.DUR_48H)
    doc["description"] = (
        "DoW rerun BRIDGE center point for dynamo_disagg (W3): the cp1-cp3 template, "
        "workload and 48 h window, plus the opt-in per-process split. Measures the "
        "shift between the completed Dynamo arm and the rerun epoch. Generated by "
        "scripts/generate_rerun_windows.py.")
    doc["monitors"]["components"]["per_process_split"] = True
    return doc


def soak_cell_doc(system: str) -> dict:
    template = gdc._load_template(system)
    row = gdc.coded_design()[SOAK_SHAPE_YATES - 1]
    phys = gdc.physical(row)
    overrides = gdc._workload_overrides(template["workload"]["client_config_overrides"], phys)
    overrides["target_rate_rps"] = SOAK_RATE_RPS
    cid = SOAK_CELLS[system]
    doc: dict = {
        "cell_id": cid,
        "short_label": cid,
        "description": (
            f"W1 scope soak for {system}: whole-container v2 scope held for 6 h on the "
            f"heaviest DoW shape (p16) at a fixed {SOAK_RATE_RPS} rps, no calibration. "
            "Acceptance: analysis/soak_check.py. Generated by scripts/generate_rerun_windows.py."),
        "engine": copy.deepcopy(template["engine"]),
        "monitors": copy.deepcopy(template["monitors"]),
        "workload": {"client_config_overrides": overrides},
        "duration_s": SOAK_DURATION_S,
        "warmup_discard_s": SOAK_WARMUP_DISCARD_S,
    }
    if "post_run_cooldown_s" in template:
        doc["post_run_cooldown_s"] = template["post_run_cooldown_s"]
    return doc


def _cell_header(lines: list[str]) -> str:
    head = ["# GENERATED by scripts/generate_rerun_windows.py -- DO NOT EDIT.",
            "# Regenerate with `python3 scripts/generate_rerun_windows.py`.", "#"]
    return "\n".join(head + [f"# {line}" if line else "#" for line in lines]) + "\n"


def new_cell_files() -> dict[str, str]:
    """{file name under cells/rerun: yaml text}."""
    out = {}
    out[f"{BRIDGE_CELL}.yaml"] = _cell_header([
        "DoW rerun bridge center point (window W3), system=dynamo_disagg.",
        "Identical template, workload and 48 h window to dow_dynamo_disagg_cp1..cp3",
        "(coded A=B=C=D=E=0), plus monitors.components.per_process_split: true (R5).",
    ]) + "\n" + yaml.safe_dump(bridge_cell_doc(), sort_keys=False, width=100)
    for system in RERUN_SYSTEMS:
        out[f"{SOAK_CELLS[system]}.yaml"] = _cell_header([
            f"W1 scope soak, system={system}: 6 h, workload shape of design point p16",
            f"(all factors +1), fixed target_rate_rps={SOAK_RATE_RPS}, no calibration.",
            "engine + monitors copied verbatim from the validated template "
            f"({gdc.TEMPLATE_CELL[system].name}).",
        ]) + "\n" + yaml.safe_dump(soak_cell_doc(system), sort_keys=False, width=100)
    return out


def cell_doc_path(cid: str) -> Path:
    if cid == BRIDGE_CELL or cid in SOAK_CELLS.values():
        return CELLS_SUBDIR / f"{cid}.yaml"
    return gdc._GENERATED_CELL_SUBDIR / f"{cid}.yaml"


def _duration_s(cid: str) -> int:
    if cid == BRIDGE_CELL:
        return gdc.DUR_48H
    system = "dynamo_disagg" if cid.startswith("dow_dynamo") else cid.split("_")[1]
    return gdc._duration_for(system, "center" if "_cp" in cid else "design")


# --- Projection ---------------------------------------------------------------


def projection(k: int, order: list[tuple[str, str]]) -> dict:
    runs_h = sum(_duration_s(cid) for _, cid in order) / 3600
    n_soak = len(SOAK_CELLS) if k == 1 else 0
    soak_h = n_soak * SOAK_DURATION_S / 3600
    calib_h = len(order) * CALIBRATION_H_PER_CELL
    overhead_h = (len(order) + n_soak) * RUN_OVERHEAD_H
    total_h = WINDOW_SETUP_H + soak_h + calib_h + runs_h + overhead_h
    longest_d = max(_duration_s(cid) for _, cid in order) / 86400
    margin_d = WINDOW_MAX_DAYS - total_h / 24
    # Calibration age when the LAST run of the window is dispatched (calibration
    # ends just before the first run), plus one retry of the longest run.
    last_dispatch_age_d = (runs_h - _duration_s(order[-1][1]) / 3600
                           + (len(order) - 1) * RUN_OVERHEAD_H) / 24
    return {
        "n_runs": len(order), "n_soak_runs": n_soak,
        "setup_h": WINDOW_SETUP_H, "soak_h": soak_h, "calibration_h": round(calib_h, 2),
        "runs_h": runs_h, "overhead_h": overhead_h, "total_h": round(total_h, 2),
        "total_days": round(total_h / 24, 2), "retry_margin_days": round(margin_d, 2),
        "required_margin_days": round(max(MIN_RETRY_MARGIN_DAYS, longest_d), 2),
        "calibration_age_at_last_dispatch_days": round(last_dispatch_age_d, 2),
    }


def check_plan(orders: dict[int, list[tuple[str, str]]]) -> None:
    """Raise PlanError unless every design and duration requirement holds."""
    for system in RERUN_SYSTEMS:
        seen = []
        for k in WINDOWS:
            rows = [_codes_of(cid) for s, cid in orders[k] if s == system and "_p" in cid]
            seen += [r["yates"] for r in rows]
            for f in ("A", "B", "C", "D", "E"):
                if sum(r[f] for r in rows):
                    raise PlanError(f"W{k} {system}: main effect {f} not balanced")
            for a in "ABCDE":
                for b in "ABCDE":
                    name = a + b
                    if a < b and name not in WINDOW_INTERACTIONS and sum(interaction(r, name) for r in rows):
                        raise PlanError(f"W{k} {system}: {name} correlated with the window")
            cps = [cid for s, cid in orders[k] if s == system and "_cp" in cid]
            if cps != [gdc.cell_id(system, "cp", k)]:
                raise PlanError(f"W{k} {system}: expected exactly cp{k}, got {cps}")
        if sorted(seen) != list(range(1, 17)):
            raise PlanError(f"{system}: the 16 design points are not each run once")
    for k in WINDOWS:
        p = projection(k, orders[k])
        if p["total_days"] > WINDOW_MAX_DAYS or p["retry_margin_days"] < p["required_margin_days"]:
            raise PlanError(f"W{k}: projected {p['total_days']} d leaves {p['retry_margin_days']} d "
                            f"of retry margin (< {p['required_margin_days']} d)")
        if p["calibration_age_at_last_dispatch_days"] + p["required_margin_days"] >= CALIBRATION_MAX_AGE_DAYS:
            raise PlanError(f"W{k}: calibration would exceed {CALIBRATION_MAX_AGE_DAYS} d")


# --- Emitters -----------------------------------------------------------------


def _codes_str(codes: dict) -> str:
    return " ".join(f"{f}={codes[f]:+d}" for f in ("A", "B", "C", "D", "E"))


def _host_block() -> list[str]:
    runs_root, paths, min_inodes = gdc._read_host_config_from_validation_campaign()
    lines = [
        "replicas_per_cell: 1",
        "order: cell_at_a_time",
        "inter_run_cooldown_s: 600",
        f"est_run_overhead_s: {int(RUN_OVERHEAD_H * 3600)}",
        "min_free_gb: 50",
        "min_free_gb_mid_run: 25",
        f"min_inodes_free_mid_run: {min_inodes}",
        f"calibration_min_method_version: {CALIBRATION_MIN_METHOD_VERSION}",
        f"calibration_max_age_days: {CALIBRATION_MAX_AGE_DAYS}",
        "retry_policy:",
        "  max_retries: 1",
        "",
        f"runs_root: {runs_root}",
        "paths:",
    ]
    lines += [f"  {key}: {value}" for key, value in paths.items()]
    return lines


def window_descriptor_text(k: int, order: list[tuple[str, str]], seed: int) -> str:
    p = projection(k, order)
    calib_dir = CALIBRATION_DIR.format(k=k)
    alloc = {s: sorted({quarter_label(quarter_of(_codes_of(c))) for sy, c in order
                        if sy == s and "_p" in c}) for s in RERUN_SYSTEMS}
    lines = [
        "# GENERATED by scripts/generate_rerun_windows.py -- DO NOT EDIT.",
        f"# DoW rerun ({CAMPAIGN_ID}), machine window W{k} of {len(WINDOWS)}: Triton and vLLM",
        "# standalone with the whole-container v2 scope. Rationale, blocking and the",
        "# operator sequence: docs/rerun_windows.md and the generator docstring.",
        "#",
        f"# Quarter blocks (signs of AC, AD; E=ABCD): triton {', '.join(alloc['triton'])}; "
        f"vllm {', '.join(alloc['vllm'])}.",
        "# All five main effects balanced per system; only AC, AD, CD can correlate with",
        "# window membership. One center point per system in the middle of the sequence.",
        "#",
        f"# Fixed order seed {seed}: design points shuffled per system, the minority",
        "# system spread evenly through the majority, the center-point block in the middle.",
        "#",
        f"# Projection: setup {p['setup_h']:g} h + soak {p['soak_h']:g} h + calibration "
        f"{p['calibration_h']:g} h ({p['n_runs']} cells x {CALIBRATION_H_PER_CELL:.3f} h)",
        f"#   + runs {p['runs_h']:g} h + overhead {p['overhead_h']:g} h "
        f"({p['n_runs'] + p['n_soak_runs']} runs x {RUN_OVERHEAD_H:g} h)",
        f"#   = {p['total_h']:g} h = {p['total_days']:g} days; retry margin "
        f"{p['retry_margin_days']:g} days in {WINDOW_MAX_DAYS:g} (required "
        f">= {p['required_margin_days']:g}).",
        f"# Calibration age at the last dispatch ~{p['calibration_age_at_last_dispatch_days']:g} d "
        f"(limit {CALIBRATION_MAX_AGE_DAYS} d).",
        "#",
        f"# Execution order ({len(order)} runs):",
    ]
    for i, (system, cid) in enumerate(order, start=1):
        codes = _codes_of(cid)
        tag = ("bridge CP" if cid == BRIDGE_CELL else "center" if codes["yates"] is None
               else quarter_label(quarter_of(codes)))
        lines.append(f"#   {i:2d}. [{system}] {cid}  {_codes_str(codes)}  ({tag})")
    lines += [
        "",
        f"campaign_id: {CAMPAIGN_ID}",
        "mode: serial",
        "description: |",
        f"  DoW rerun window W{k}: {p['n_runs']} strictly serial runs, just-in-time",
        f"  calibration in {calib_dir}, calibrations older than {CALIBRATION_MAX_AGE_DAYS} days refused.",
        "",
        "# Own state file per window; never shared with another window or campaign.",
        f"state_file: state/dow_rerun_w{k}_state.json",
        "",
        "# Generator metadata (ignored by campaign.py; read by calibrate_dow.py and tests).",
        "rerun_window:",
        f"  window: {k}",
        f"  calibration_dir: {calib_dir}",
        f"  seed: {seed}",
    ]
    lines += [f"  projected_{key}: {value}" for key, value in p.items()]
    lines.append("")
    lines += _host_block()
    lines += ["", "cells:"]
    for _, cid in order:
        lines += [f"  - id: {cid}",
                  f"    yaml: {cell_doc_path(cid).as_posix()}",
                  f"    calibration_file: {calib_dir}/{cid}.json",
                  "    calibration_required: true"]
    return "\n".join(lines) + "\n"


def soak_descriptor_text() -> str:
    lines = [
        "# GENERATED by scripts/generate_rerun_windows.py -- DO NOT EDIT.",
        f"# W1 scope soak ({CAMPAIGN_ID}): one 6 h run per system on the p16 shape at a",
        f"# fixed {SOAK_RATE_RPS} rps, BEFORE the W1 calibration. No calibration needed.",
        "# Acceptance per run: validate_extension_run.py --scope-only AND analysis/soak_check.py.",
        "",
        f"campaign_id: {CAMPAIGN_ID}",
        "mode: serial",
        "description: |",
        "  W1 whole-container scope soak (Triton, vLLM standalone): process-set stability",
        "  over hours before committing the window to 36 h runs.",
        "",
        "state_file: state/dow_rerun_w1_soak_state.json",
        "",
    ]
    lines += _host_block()
    lines += ["", "cells:"]
    for system in RERUN_SYSTEMS:
        cid = SOAK_CELLS[system]
        lines += [f"  - id: {cid}", f"    yaml: {cell_doc_path(cid).as_posix()}"]
    return "\n".join(lines) + "\n"


MATRIX_FIELDS = [
    "window", "position", "system", "cell_id", "kind", "yates_index",
    "A_code", "B_code", "C_code", "D_code", "E_code", "AC", "AD", "CD", "quarter_block",
    "rate_fraction", "prompt_len", "max_tokens", "prefix_repeat_fraction",
    "shared_prefix_len", "arrival_mode", "burst_factor", "burst_on_seconds",
    "duration_s", "warmup_discard_s", "calibration_file",
]


def matrix_csv_text(orders: dict[int, list[tuple[str, str]]], seed: int) -> str:
    buf = io.StringIO()
    buf.write("# GENERATED by scripts/generate_rerun_windows.py -- DO NOT EDIT.\n")
    buf.write(f"# campaign_id={CAMPAIGN_ID}; seed={seed}; defining_relation={gdc.DEFINING_RELATION}; "
              "quarter blocks by the signs of AC and AD\n")
    buf.write("# Rows in execution order per window (position 1..n). The W1 soak cells are not "
              "design points and are not listed.\n")
    w = csv.DictWriter(buf, fieldnames=MATRIX_FIELDS, lineterminator="\n")
    w.writeheader()
    for k in WINDOWS:
        for pos, (system, cid) in enumerate(orders[k], start=1):
            codes = _codes_of(cid)
            phys = gdc.physical(codes)
            is_design = codes["yates"] is not None
            w.writerow({
                "window": k, "position": pos, "system": system, "cell_id": cid,
                "kind": "bridge_center" if cid == BRIDGE_CELL else ("design" if is_design else "center"),
                "yates_index": codes["yates"] if is_design else "",
                **{f"{f}_code": codes[f] for f in ("A", "B", "C", "D", "E")},
                **{name: interaction(codes, name) for name in WINDOW_INTERACTIONS},
                "quarter_block": quarter_label(quarter_of(codes)) if is_design else "",
                "rate_fraction": phys["rate_fraction"],
                "prompt_len": phys["prompt_len"], "max_tokens": phys["max_tokens"],
                "prefix_repeat_fraction": phys["prefix_repeat_fraction"],
                "shared_prefix_len": phys["shared_prefix_len"],
                "arrival_mode": phys["arrival_mode"],
                "burst_factor": phys["burst_factor"] if phys["burst_factor"] is not None else "",
                "burst_on_seconds": phys["burst_on_seconds"] if phys["burst_on_seconds"] is not None else "",
                "duration_s": _duration_s(cid),
                "warmup_discard_s": gdc.WARMUP_DISCARD_S,
                "calibration_file": f"{CALIBRATION_DIR.format(k=k)}/{cid}.json",
            })
    return buf.getvalue()


# --- Generate / check ---------------------------------------------------------


def _check_reused_cells(out_base: Path) -> None:
    """The reused Triton/vLLM cells must be the CURRENT generated container_tree
    v2 cells; refuse to build a rerun on stale or legacy-scoped cells."""
    stale = gdc.generate(out_base, check=True)
    if stale["changed"] or stale["removed"]:
        raise PlanError("cells/dow is out of date: run scripts/generate_dow_cells.py first")
    for system in RERUN_SYSTEMS:
        for cid in [gdc.cell_id(system, "p", n) for n in range(1, 17)] + \
                   [gdc.cell_id(system, "cp", k) for k in (1, 2, 3)]:
            doc = yaml.safe_load((out_base / cell_doc_path(cid)).read_text())
            comps = doc["monitors"]["components"]
            if comps.get("scope_type") != "container_tree" or comps.get("scope_version") != 2:
                raise PlanError(f"{cid}: not a container_tree v2 cell")


def generate(out_base: Path = EXT, seed: int = RERUN_SEED, check: bool = False) -> dict:
    _check_reused_cells(out_base)
    orders = window_order(seed)
    check_plan(orders)
    contents: dict[Path, str] = {}
    for k in WINDOWS:
        contents[out_base / DESCRIPTOR.format(k=k)] = window_descriptor_text(k, orders[k], seed)
    contents[out_base / SOAK_DESCRIPTOR] = soak_descriptor_text()
    contents[out_base / MATRIX_CSV] = matrix_csv_text(orders, seed)
    cells_dir = out_base / CELLS_SUBDIR
    for name, text in new_cell_files().items():
        contents[cells_dir / name] = text

    would_change = [str(p) for p, text in contents.items()
                    if not p.exists() or p.read_text() != text]
    existing = set(cells_dir.glob("*.yaml")) if cells_dir.exists() else set()
    stale = sorted(str(p) for p in existing - {p for p in contents if p.parent == cells_dir})
    if check:
        return {"changed": would_change, "removed": stale, "orders": orders}
    changed: list[str] = []
    for path, text in contents.items():
        gdc._write_if_changed(path, text, changed)
    for p in stale:
        Path(p).unlink()
    return {"changed": changed, "removed": stale, "orders": orders}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the DoW rerun window descriptors.")
    ap.add_argument("--out-base", type=Path, default=EXT)
    ap.add_argument("--seed", type=int, default=RERUN_SEED)
    ap.add_argument("--check", action="store_true",
                    help="Do not write; exit 1 if the generated tree is out of date.")
    args = ap.parse_args(argv)
    try:
        result = generate(args.out_base, seed=args.seed, check=args.check)
    except PlanError as e:
        print(f"PLAN ERROR: {e}", file=sys.stderr)
        return 2
    if args.check:
        if result["changed"] or result["removed"]:
            print("Rerun tree is OUT OF DATE. Regenerate with "
                  "`python3 scripts/generate_rerun_windows.py`.")
            for p in result["changed"]:
                print(f"  would write: {p}")
            for p in result["removed"]:
                print(f"  would remove: {p}")
            return 1
        print("Rerun tree is up to date.")
        return 0
    print(f"Rerun windows generated under {args.out_base} (seed={args.seed}).")
    print(f"  wrote/updated {len(result['changed'])} file(s); removed {len(result['removed'])}.")
    for k in WINDOWS:
        p = projection(k, result["orders"][k])
        print(f"\nW{k}: {p['n_runs']} runs, projected {p['total_days']} d, "
              f"retry margin {p['retry_margin_days']} d")
        for i, (system, cid) in enumerate(result["orders"][k], start=1):
            print(f"  {i:2d}. [{system}] {cid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
