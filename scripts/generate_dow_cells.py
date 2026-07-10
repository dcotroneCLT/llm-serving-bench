#!/usr/bin/env python3
"""Generate the DoW screening campaign (57 cells + descriptor + design matrix).

This is the SINGLE SOURCE OF TRUTH for the extension's scientific core. The
design matrix encoded at the top of this file is authoritative; the per-cell
yamls, the campaign descriptor, and dow_design_matrix.csv are GENERATED
artifacts -- never hand-edit them. Regenerate instead:

    python3 scripts/generate_dow_cells.py            # write into campaigns/extension
    python3 scripts/generate_dow_cells.py --check     # verify tree is up to date (CI)

The generator is deterministic and idempotent: the same inputs produce
byte-identical outputs, and running it twice changes nothing on disk.

DESIGN (CONFIRMED 2026-06-29 in paper/PAPER_UPDATE_PLAN.md; window amended
2026-07-10). One screening DoW run replicated identically on three serving
systems (NVIDIA Dynamo disaggregated, Triton+vLLM, vLLM standalone):

  Resolution V 2^(5-1) fractional factorial, defining relation I=ABCDE
  (equivalently E = A*B*C*D in coded +/-1 units): 16 design points + 3 center
  points = 19 runs/system * 3 systems = 57 cells. All main effects and all
  two-factor interactions are estimable, unconfounded.

  Factors and levels (coded -1 / +1, center 0), mapped to the ACTUAL client
  config keys (verified against client/run_client.py + client/config.yaml):
    A rate          -> calibration fraction-of-ceiling 0.30 / 0.85 (center 0.575)
                       realized via each cell's calibration_file, NOT a fixed
                       target_rate_rps (the calibration orchestrator, a separate
                       task, sets target_rate_rps = fraction * measured ceiling).
    B prompt-length -> prompt_len 512 / 6000 tok (center 3256)
    C output-length -> max_tokens 64 / 1024 tok (center 544)
    D prefix-repeat -> prefix_repeat_fraction 0.0 / 0.8 (center 0.4);
                       shared_prefix_len = 512 whenever the fraction > 0.
    E burstiness    -> arrival process. QUALITATIVE/continuous factor:
                       coded -1 = arrival_mode=poisson (the true-Poisson path);
                       coded +1 = arrival_mode=bursty, burst_factor=4.0;
                       center 0 = arrival_mode=bursty, burst_factor=2.5.
                       burst_on_seconds fixed at 5.0 for the bursty levels.
                       WHY not bursty@1.0 for the low level: client/benchmark.py
                       clamps burst_factor>=1.0 and the bursty MMPP-2 path does
                       NOT reproduce the exact Poisson RNG draw even at 1.0, so
                       the low level is the documented arrival_mode=poisson
                       ("DoW levels: poisson / bursty" in client/config.yaml).
                       B and C are set as deterministic point levels
                       ({median=p95=min=max=level}) so a screening estimates the
                       main effect of the LEVEL, not of a within-cell length
                       distribution.

  Windows (2026-07-10 amendment): duration_s=129600 (36h) for ALL cells, EXCEPT
  the 3 Dynamo center points at duration_s=172800 (48h) as the cross-anchor
  against the completed 48h long test. warmup_discard_s=3600 for every cell.

  Per-system engine + monitors blocks are copied VERBATIM from the box-validated
  cells (val_dynamo_disagg / val_triton / val_vllm). Only cell_id, description,
  workload overrides, and durations differ. tests/test_dow_campaign.py enforces
  the engine+monitors byte-identity.

ORDERING (fixed-seed interleaved schedule; the campaign is strictly serial over
~13 weeks, so a naive per-system block would alias host drift with the system
effect). The generator emits the cells: list in an order that:
  (a) interleaves the three systems round-robin (no system concentrated in one
      calendar block; strictly alternating -> never >2 consecutive same-system);
  (b) randomizes the 16 design points WITHIN each system with the fixed seed
      below (the seed changes the ORDER, never the SET of 57);
  (c) spreads each system's 3 center points to the early / middle / late third
      of that system's sequence, so they double as drift sentinels.
With replicas_per_cell: 1, campaign.py's build_schedule executes the cells: list
in exactly its written order, so the interleaving is baked into the list (no
campaign.py change).
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

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "campaigns" / "extension"

# --- Fixed, documented inputs -------------------------------------------------

# Seed for the within-system design-point randomization + interleave. Changing
# it reorders the schedule but never changes the set of 57 cells (enforced by a
# test). 20260710 = the date of the window amendment that locked this campaign.
DESIGN_SEED = 20260710

DEFINING_RELATION = "I=ABCDE"

# System token -> validated template cell. The token appears in every cell_id
# (dow_<system>_pNN / dow_<system>_cpN) and fixes the interleave order.
SYSTEMS = ["dynamo_disagg", "triton", "vllm"]
TEMPLATE_CELL = {
    "dynamo_disagg": EXT / "cells" / "val_dynamo_disagg.yaml",
    "triton": EXT / "cells" / "val_triton.yaml",
    "vllm": EXT / "cells" / "val_vllm.yaml",
}

# Windows (2026-07-10 amendment).
DUR_36H = 129_600
DUR_48H = 172_800
WARMUP_DISCARD_S = 3_600

# Factor -> physical level, keyed by coded value (-1, 0, +1).
RATE_FRACTION = {-1: 0.30, 0: 0.575, 1: 0.85}       # A
PROMPT_LEN = {-1: 512, 0: 3256, 1: 6000}            # B
MAX_TOKENS = {-1: 64, 0: 544, 1: 1024}              # C
PREFIX_FRACTION = {-1: 0.0, 0: 0.4, 1: 0.8}         # D
SHARED_PREFIX_LEN = 512                              # used whenever fraction > 0
# E: coded -> (arrival_mode, burst_factor or None)
BURST = {-1: ("poisson", None), 0: ("bursty", 2.5), 1: ("bursty", 4.0)}
BURST_ON_SECONDS = 5.0

# Workload keys carried over verbatim from each system's template (the
# system-specific transport/identity that must not change between DoW cells).
_CARRY_WORKLOAD_KEYS = [
    "protocol", "base_url", "model", "concurrency_cap",
    "streaming_prob", "request_timeout_s", "seed_template",
]

_GENERATED_CELL_SUBDIR = Path("cells") / "dow"


# --- Design matrix ------------------------------------------------------------


def coded_design() -> list[dict]:
    """The 16 Resolution-V design points in standard Yates order (A fastest),
    with E = A*B*C*D so that I=ABCDE. yates is the 1-based Yates index used in
    the cell id (dow_<system>_pNN)."""
    rows = []
    for k in range(16):
        a = 1 if (k & 1) else -1
        b = 1 if (k & 2) else -1
        c = 1 if (k & 4) else -1
        d = 1 if (k & 8) else -1
        e = a * b * c * d
        rows.append({"yates": k + 1, "A": a, "B": b, "C": c, "D": d, "E": e})
    return rows


CENTER_CODES = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}


# --- Physical realization -----------------------------------------------------


def physical(codes: dict) -> dict:
    """Coded row -> physical factor values (client-config space)."""
    arrival_mode, burst_factor = BURST[codes["E"]]
    frac = PREFIX_FRACTION[codes["D"]]
    return {
        "rate_fraction": RATE_FRACTION[codes["A"]],
        "prompt_len": PROMPT_LEN[codes["B"]],
        "max_tokens": MAX_TOKENS[codes["C"]],
        "prefix_repeat_fraction": frac,
        "shared_prefix_len": SHARED_PREFIX_LEN if frac > 0 else 0,
        "arrival_mode": arrival_mode,
        "burst_factor": burst_factor,
        "burst_on_seconds": BURST_ON_SECONDS if arrival_mode == "bursty" else None,
    }


def _workload_overrides(template_overrides: dict, phys: dict) -> dict:
    """Build the cell's client_config_overrides: the system-specific transport
    keys carried from the template + the DoW factor keys. target_rate_rps is
    intentionally absent -- the rate is set at run time from calibration_file."""
    out: dict = {}
    for key in _CARRY_WORKLOAD_KEYS:
        if key in template_overrides:
            out[key] = copy.deepcopy(template_overrides[key])
    # Deterministic point levels for the length factors.
    out["prompt_len"] = {"median": phys["prompt_len"], "p95": phys["prompt_len"],
                         "min": phys["prompt_len"], "max": phys["prompt_len"]}
    out["max_tokens"] = {"median": phys["max_tokens"], "p95": phys["max_tokens"],
                         "min": phys["max_tokens"], "max": phys["max_tokens"]}
    out["prefix_repeat_fraction"] = phys["prefix_repeat_fraction"]
    out["shared_prefix_len"] = phys["shared_prefix_len"]
    out["arrival_mode"] = phys["arrival_mode"]
    if phys["arrival_mode"] == "bursty":
        out["burst_factor"] = phys["burst_factor"]
        out["burst_on_seconds"] = phys["burst_on_seconds"]
    return out


def cell_id(system: str, kind: str, index: int) -> str:
    """kind 'p' -> dow_<system>_pNN (Yates index); kind 'cp' -> dow_<system>_cpN."""
    if kind == "p":
        return f"dow_{system}_p{index:02d}"
    return f"dow_{system}_cp{index}"


def _duration_for(system: str, kind: str) -> int:
    # Only the 3 Dynamo center points anchor at 48h; everything else is 36h.
    if system == "dynamo_disagg" and kind == "center":
        return DUR_48H
    return DUR_36H


def _coded_str(codes: dict) -> str:
    return " ".join(f"{f}={codes[f]:+d}" for f in ("A", "B", "C", "D", "E"))


def _header_comment(system: str, cid: str, kind: str, codes: dict, phys: dict,
                    duration_s: int) -> str:
    hours = duration_s // 3600
    burst = (f"bursty burst_factor={phys['burst_factor']} "
             f"burst_on_seconds={phys['burst_on_seconds']}"
             if phys["arrival_mode"] == "bursty" else "poisson (true-Poisson arrival)")
    kind_line = (f"design point p{codes['yates']:02d} (standard Yates order index)"
                 if kind == "design"
                 else "center point (1 of 3 identical center replicates; drift sentinel)")
    return (
        "# GENERATED by scripts/generate_dow_cells.py -- DO NOT EDIT.\n"
        "# Source of truth: the design matrix in that script + the validated\n"
        f"# per-system template cell ({TEMPLATE_CELL[system].name}). Regenerate to change.\n"
        "#\n"
        f"# DoW screening cell: system={system}, {kind_line}.\n"
        f"# Resolution V 2^(5-1) fractional factorial, defining relation {DEFINING_RELATION}\n"
        "# (E = A*B*C*D in coded units). All main effects + 2-factor interactions estimable.\n"
        f"# Coded levels: {_coded_str(codes)}\n"
        "# Physical levels:\n"
        f"#   A rate          = {phys['rate_fraction']} fraction-of-ceiling (via calibration_file)\n"
        f"#   B prompt_len    = {phys['prompt_len']} tok (deterministic point level)\n"
        f"#   C max_tokens    = {phys['max_tokens']} tok (deterministic point level)\n"
        f"#   D prefix_repeat = {phys['prefix_repeat_fraction']} (shared_prefix_len={phys['shared_prefix_len']})\n"
        f"#   E burstiness    = {burst}\n"
        f"# Window: {hours}h  duration_s={duration_s}, warmup_discard_s={WARMUP_DISCARD_S}.\n"
    )


def build_cell_doc(system: str, template: dict, cid: str, kind: str,
                   codes: dict, phys: dict, duration_s: int) -> dict:
    """Assemble one cell dict. engine + monitors are copied VERBATIM from the
    template; only cell_id/short_label/description/workload/durations differ.

    calibration_fraction is a top-level METADATA key (factor A): it is read by
    the calibration orchestrator (a separate task) to calibrate this cell's
    ceiling at the right fraction. launch_cell.py ignores it (it consumes only
    engine/monitors/workload/duration_s/warmup_discard_s/cell_id)."""
    kind_word = "design point" if kind == "design" else "center point"
    doc: dict = {
        "cell_id": cid,
        "short_label": cid,
        "description": (
            f"DoW screening {kind_word} for {system}. Coded {_coded_str(codes)} "
            f"({DEFINING_RELATION}). Rate fraction {phys['rate_fraction']} of ceiling "
            f"(via calibration_file). Generated by scripts/generate_dow_cells.py."
        ),
        # Factor A metadata for the calibration orchestrator (NOT consumed by launch_cell).
        "calibration_fraction": phys["rate_fraction"],
        # VERBATIM from the validated template.
        "engine": copy.deepcopy(template["engine"]),
        "monitors": copy.deepcopy(template["monitors"]),
        "workload": {
            "client_config_overrides": _workload_overrides(
                template["workload"]["client_config_overrides"], phys)
        },
        "duration_s": duration_s,
        "warmup_discard_s": WARMUP_DISCARD_S,
    }
    # Carry the (non-engine) VRAM-quiescence cooldown verbatim if the template has one.
    if "post_run_cooldown_s" in template:
        doc["post_run_cooldown_s"] = template["post_run_cooldown_s"]
    return doc


# --- The 19 cells of one system (16 design points + 3 center points) ----------


def system_cells(system: str, template: dict) -> list[dict]:
    """Return [{cell_id, kind, codes, phys, duration_s, doc}] for one system, in
    canonical (Yates then center) order -- NOT schedule order."""
    out = []
    for row in coded_design():
        codes = {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"],
                 "E": row["E"], "yates": row["yates"]}
        cid = cell_id(system, "p", row["yates"])
        phys = physical(codes)
        dur = _duration_for(system, "design")
        out.append({"cell_id": cid, "kind": "design", "codes": codes, "phys": phys,
                    "duration_s": dur,
                    "doc": build_cell_doc(system, template, cid, "design", codes, phys, dur)})
    for k in (1, 2, 3):
        codes = dict(CENTER_CODES, yates=None)
        cid = cell_id(system, "cp", k)
        phys = physical(codes)
        dur = _duration_for(system, "center")
        out.append({"cell_id": cid, "kind": "center", "codes": codes, "phys": phys,
                    "duration_s": dur,
                    "doc": build_cell_doc(system, template, cid, "center", codes, phys, dur)})
    return out


# --- Ordering -----------------------------------------------------------------


def _per_system_sequence(system: str, rng: random.Random) -> list[str]:
    """19 cell_ids for one system: 16 design points shuffled with rng, with the
    3 center points inserted at the early / middle / late third (final indices
    0, 9, 18 of the 19-slot sequence)."""
    dps = [cell_id(system, "p", n) for n in range(1, 17)]
    rng.shuffle(dps)
    cps = [cell_id(system, "cp", k) for k in (1, 2, 3)]
    seq = list(dps)              # 16
    seq.insert(0, cps[0])        # 17: cp1 at 0 (early third)
    seq.insert(9, cps[1])        # 18: cp2 at 9 (middle third)
    seq.append(cps[2])           # 19: cp3 at 18 (late third)
    return seq


def schedule_order(seed: int = DESIGN_SEED) -> list[tuple[str, str]]:
    """The 57-run linear order as (system, cell_id). Systems interleave
    round-robin (strictly alternating: never >2 consecutive same system); design
    points are randomized within each system by `seed`; center points sit at each
    system's early/middle/late positions (and hence early/middle/late globally)."""
    rng = random.Random(seed)
    seqs = {system: _per_system_sequence(system, rng) for system in SYSTEMS}
    order: list[tuple[str, str]] = []
    for rnd in range(19):
        for system in SYSTEMS:
            order.append((system, seqs[system][rnd]))
    return order


# --- Emitters -----------------------------------------------------------------


def _dump_cell_yaml(doc: dict, header: str) -> str:
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)
    return header + "\n" + body


def _read_paths_from_validation_campaign() -> tuple[str, dict]:
    """Reuse the validation campaign's host paths (runs_root + paths) so the DoW
    campaign targets the same box without a second hardcoded copy."""
    val = yaml.safe_load((EXT / "campaign.yaml").read_text())
    runs_root = val["runs_root"]
    paths = val.get("paths", {})
    return runs_root, paths


def _campaign_yaml_text(order: list[tuple[str, str]], seed: int,
                        runs_root: str, paths: dict) -> str:
    lines: list[str] = []
    lines.append("# GENERATED by scripts/generate_dow_cells.py -- DO NOT EDIT.")
    lines.append("# DoW screening campaign (extension). Scientific core of the extension:")
    lines.append("# Resolution V 2^(5-1) fractional factorial (defining relation I=ABCDE),")
    lines.append("# 16 design points + 3 center points per system, replicated identically on")
    lines.append("# 3 serving systems = 57 cells. See scripts/generate_dow_cells.py for the")
    lines.append("# authoritative design matrix and paper/PAPER_UPDATE_PLAN.md (2026-06-12 plan")
    lines.append("# + 2026-07-10 amendment) for the rationale and windows.")
    lines.append("#")
    lines.append(f"# Fixed schedule seed: {seed}. The cells: list below IS the execution order")
    lines.append("# (replicas_per_cell: 1, so campaign.py runs the list in order). The order")
    lines.append("# interleaves the 3 systems round-robin (never >2 consecutive same-system),")
    lines.append("# randomizes design points within each system by the seed, and places each")
    lines.append("# system's 3 center points early/middle/late as drift sentinels.")
    lines.append("#")
    lines.append("# Per-cell calibration: the ceiling depends on the workload SHAPE, so every")
    lines.append("# cell calibrates with its OWN materialized client config and its own")
    lines.append("# fraction. The 3 center points per system share ONE shape (16 design shapes")
    lines.append("# + 1 center shape) x 3 systems = 51 distinct calibrations, though each of the")
    lines.append("# 57 cells names its own calibration_file. The calibration orchestrator is a")
    lines.append("# SEPARATE task; this descriptor only declares the requirement.")
    lines.append("#")
    lines.append(f"# Final 57-run linear order (seed={seed}):")
    for i, (system, cid) in enumerate(order, start=1):
        lines.append(f"#   {i:2d}. [{system}] {cid}")
    lines.append("")
    lines.append("campaign_id: extension_dow_screening")
    lines.append("mode: serial")
    lines.append("description: |")
    lines.append("  WoSAR 2026 extension: workload Design-of-Experiments screening, one strictly")
    lines.append("  serial queue on the launch_cell production path. Resolution V 16+3CP per")
    lines.append("  system x 3 systems = 57 runs (54 x 36h + 3 Dynamo center points x 48h).")
    lines.append("")
    lines.append("# Own state file -- MUST NOT collide with the validation campaign.yaml state.")
    lines.append("state_file: state/dow_campaign_state.json")
    lines.append("")
    lines.append("replicas_per_cell: 1")
    lines.append("# With replicas_per_cell: 1, cell_at_a_time and round_robin are equivalent and")
    lines.append("# both preserve the cells: list order. The interleaving is baked into the list.")
    lines.append("order: cell_at_a_time")
    lines.append("")
    lines.append("inter_run_cooldown_s: 600")
    lines.append("min_free_gb: 50")
    lines.append("min_free_gb_mid_run: 25")
    lines.append("retry_policy:")
    lines.append("  max_retries: 1")
    lines.append("")
    lines.append(f"runs_root: {runs_root}")
    lines.append("paths:")
    for k, v in paths.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("# Every cell: calibration REQUIRED, with a distinct per-cell calibration file")
    lines.append("# (resolved relative to this yaml). calibration_required makes a missing/")
    lines.append("# invalid calibration fail PRE-FLIGHT, not at hour 37.")
    lines.append("cells:")
    for system, cid in order:
        lines.append(f"  - id: {cid}")
        lines.append(f"    yaml: cells/dow/{cid}.yaml")
        lines.append(f"    calibration_file: state/calibration/{cid}.json")
        lines.append("    calibration_required: true")
    return "\n".join(lines) + "\n"


def _design_matrix_rows(order: list[tuple[str, str]]) -> list[dict]:
    """One CSV row per cell, in schedule order (schedule_position 1..57)."""
    # Index cells by (system, cell_id) for physical lookup.
    by_id: dict[str, dict] = {}
    for system in SYSTEMS:
        for c in system_cells(system, _load_template(system)):
            by_id[c["cell_id"]] = {"system": system, **c}
    rows = []
    for pos, (system, cid) in enumerate(order, start=1):
        c = by_id[cid]
        codes, phys = c["codes"], c["phys"]
        rows.append({
            "schedule_position": pos,
            "system": system,
            "cell_id": cid,
            "kind": c["kind"],
            "yates_index": codes["yates"] if codes["yates"] is not None else "",
            "A_code": codes["A"], "B_code": codes["B"], "C_code": codes["C"],
            "D_code": codes["D"], "E_code": codes["E"],
            "rate_fraction": phys["rate_fraction"],
            "prompt_len": phys["prompt_len"],
            "max_tokens": phys["max_tokens"],
            "prefix_repeat_fraction": phys["prefix_repeat_fraction"],
            "shared_prefix_len": phys["shared_prefix_len"],
            "arrival_mode": phys["arrival_mode"],
            "burst_factor": phys["burst_factor"] if phys["burst_factor"] is not None else "",
            "burst_on_seconds": phys["burst_on_seconds"] if phys["burst_on_seconds"] is not None else "",
            "duration_s": c["duration_s"],
            "warmup_discard_s": WARMUP_DISCARD_S,
            "calibration_file": f"state/calibration/{cid}.json",
        })
    return rows


_CSV_FIELDS = [
    "schedule_position", "system", "cell_id", "kind", "yates_index",
    "A_code", "B_code", "C_code", "D_code", "E_code",
    "rate_fraction", "prompt_len", "max_tokens", "prefix_repeat_fraction",
    "shared_prefix_len", "arrival_mode", "burst_factor", "burst_on_seconds",
    "duration_s", "warmup_discard_s", "calibration_file",
]


def _design_matrix_csv_text(order: list[tuple[str, str]], seed: int) -> str:
    buf = io.StringIO()
    buf.write(f"# GENERATED by scripts/generate_dow_cells.py -- DO NOT EDIT.\n")
    buf.write(f"# design_seed={seed}; defining_relation={DEFINING_RELATION}; "
              f"systems={','.join(SYSTEMS)}\n")
    buf.write("# Factor E (burstiness) convention: E=-1 arrival_mode=poisson; "
              "E=+1 bursty burst_factor=4.0; E=0 bursty burst_factor=2.5.\n")
    buf.write("# Rows are in schedule order (schedule_position 1..57 = the linear "
              "execution order).\n")
    w = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for row in _design_matrix_rows(order):
        w.writerow(row)
    return buf.getvalue()


# --- Template loading (cached per process) ------------------------------------

_TEMPLATE_CACHE: dict[str, dict] = {}


def _load_template(system: str) -> dict:
    if system not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[system] = yaml.safe_load(TEMPLATE_CELL[system].read_text())
    return _TEMPLATE_CACHE[system]


# --- Filesystem write (idempotent) --------------------------------------------


def _write_if_changed(path: Path, content: str, changed: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)
    changed.append(str(path))


def generate(out_base: Path = EXT, seed: int = DESIGN_SEED,
             check: bool = False) -> dict:
    """Write (or, with check=True, only diff) the full DoW tree under out_base:
        <out_base>/cells/dow/<cell_id>.yaml     (57)
        <out_base>/dow_campaign.yaml
        <out_base>/dow_design_matrix.csv
    Returns {"changed": [...], "removed": [...], "order": [...]}. Idempotent:
    a second call with the same inputs writes nothing."""
    order = schedule_order(seed)

    # Build all cell docs + their file contents.
    contents: dict[Path, str] = {}
    cells_dir = out_base / _GENERATED_CELL_SUBDIR
    for system in SYSTEMS:
        template = _load_template(system)
        for c in system_cells(system, template):
            header = _header_comment(system, c["cell_id"], c["kind"], c["codes"],
                                     c["phys"], c["duration_s"])
            contents[cells_dir / f"{c['cell_id']}.yaml"] = _dump_cell_yaml(c["doc"], header)

    runs_root, paths = _read_paths_from_validation_campaign()
    contents[out_base / "dow_campaign.yaml"] = _campaign_yaml_text(order, seed, runs_root, paths)
    contents[out_base / "dow_design_matrix.csv"] = _design_matrix_csv_text(order, seed)

    # Determine changes (and stale generated cells to remove).
    changed: list[str] = []
    would_change: list[str] = []
    for path, text in contents.items():
        if not path.exists() or path.read_text() != text:
            would_change.append(str(path))

    wanted_cell_files = {p for p in contents if p.parent == cells_dir}
    existing_cell_files = set(cells_dir.glob("dow_*.yaml")) if cells_dir.exists() else set()
    stale = sorted(str(p) for p in existing_cell_files - wanted_cell_files)

    if check:
        return {"changed": would_change, "removed": stale, "order": order}

    for path, text in contents.items():
        _write_if_changed(path, text, changed)
    removed: list[str] = []
    for p in existing_cell_files - wanted_cell_files:
        p.unlink()
        removed.append(str(p))
    return {"changed": changed, "removed": sorted(removed), "order": order}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the DoW screening campaign.")
    ap.add_argument("--out-base", type=Path, default=EXT,
                    help="Output base dir (default: campaigns/extension).")
    ap.add_argument("--seed", type=int, default=DESIGN_SEED,
                    help=f"Schedule seed (default: {DESIGN_SEED}).")
    ap.add_argument("--check", action="store_true",
                    help="Do not write; exit non-zero if the tree is out of date.")
    args = ap.parse_args(argv)

    result = generate(args.out_base, seed=args.seed, check=args.check)
    order = result["order"]

    if args.check:
        if result["changed"] or result["removed"]:
            print("DoW tree is OUT OF DATE. Regenerate with "
                  "`python3 scripts/generate_dow_cells.py`.")
            for p in result["changed"]:
                print(f"  would write: {p}")
            for p in result["removed"]:
                print(f"  would remove: {p}")
            return 1
        print("DoW tree is up to date.")
        return 0

    print(f"DoW campaign generated under {args.out_base} (seed={args.seed}).")
    print(f"  cells: 57 ({args.out_base / _GENERATED_CELL_SUBDIR})")
    print(f"  descriptor: {args.out_base / 'dow_campaign.yaml'}")
    print(f"  design matrix: {args.out_base / 'dow_design_matrix.csv'}")
    if result["changed"]:
        print(f"  wrote/updated {len(result['changed'])} file(s).")
    else:
        print("  no changes (already up to date).")
    if result["removed"]:
        print(f"  removed {len(result['removed'])} stale cell(s).")
    print(f"\nFinal 57-run linear order (seed={args.seed}):")
    for i, (system, cid) in enumerate(order, start=1):
        print(f"  {i:2d}. [{system}] {cid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
