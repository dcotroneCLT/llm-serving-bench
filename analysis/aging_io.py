"""Shared I/O helpers for long-running software-aging analyses."""

from __future__ import annotations

import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_BASE = REPO_ROOT / "logs"
FIGURES_DIR = LOGS_BASE / "figures"
DEFAULT_WARMUP_S = 1800.0


CELL_META = {
    "e1": {
        "label": "E1: vLLM V1 standalone",
        "color": "#c0392b",
        "linestyle": "-",
    },
    "a1": {
        "label": "A1: vLLM V0 standalone",
        "color": "#e67e22",
        "linestyle": "--",
    },
    "e2": {
        "label": "E2: Triton + vLLM V0",
        "color": "#2980b9",
        "linestyle": "-.",
    },
    "a2": {
        "label": "A2: Triton + vLLM V1",
        "color": "#27ae60",
        "linestyle": ":",
    },
    "e3": {
        "label": "E3: PyTorch naive",
        "color": "#8e44ad",
        "linestyle": "-",
    },
    "e3b": {
        "label": "E3b: PyTorch naive low-rate",
        "color": "#7f8c8d",
        "linestyle": "--",
    },
}
CELL_ORDER = {cell_id: index for index, cell_id in enumerate(["e1", "a1", "e2", "a2", "e3", "e3b"])}


@dataclass(frozen=True)
class RunSpec:
    """Resolved input needed by analysis scripts for one run directory."""

    id: str
    run_dir: Path
    proc_prefix: Optional[str]
    label: str
    color: str
    linestyle: str
    warmup_s: float = DEFAULT_WARMUP_S
    cell_id: Optional[str] = None
    replica: Optional[str] = None


PILOT_RUNS = [
    RunSpec(
        id="E1",
        run_dir=LOGS_BASE / "aging_pilot_24h_vllm_v1",
        proc_prefix="vllm_standalone",
        label=CELL_META["e1"]["label"],
        color=CELL_META["e1"]["color"],
        linestyle=CELL_META["e1"]["linestyle"],
        cell_id="e1",
    ),
    RunSpec(
        id="A1",
        run_dir=LOGS_BASE / "aging_pilot_24h_vllm_v0_ablation_v2",
        proc_prefix="vllm_v0_standalone",
        label=CELL_META["a1"]["label"],
        color=CELL_META["a1"]["color"],
        linestyle=CELL_META["a1"]["linestyle"],
        cell_id="a1",
    ),
    RunSpec(
        id="E2",
        run_dir=LOGS_BASE / "aging_pilot_24h_triton_v1",
        proc_prefix="triton_vllm",
        label=CELL_META["e2"]["label"],
        color=CELL_META["e2"]["color"],
        linestyle=CELL_META["e2"]["linestyle"],
        cell_id="e2",
    ),
    RunSpec(
        id="A2",
        run_dir=LOGS_BASE / "aging_pilot_24h_triton_v1_ablation_v2",
        proc_prefix="triton_vllm_v1",
        label=CELL_META["a2"]["label"],
        color=CELL_META["a2"]["color"],
        linestyle=CELL_META["a2"]["linestyle"],
        cell_id="a2",
    ),
]


def parse_csv_filter(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    items = {x.strip().lower() for x in value.split(",") if x.strip()}
    if not items or "all" in items:
        return None
    return items


def replica_matches(replica: Optional[str], filters: set[str]) -> bool:
    if replica is None:
        return False
    values = {str(replica).lower()}
    if str(replica).isdigit():
        values.add(str(int(replica)))
    return bool(values & filters)


def load_manifest(run_dir: Path) -> dict:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception as exc:
        print(f"  [warn] could not parse {manifest}: {exc}", file=sys.stderr)
        return {}


def infer_cell_id(name: str, manifest: Optional[dict] = None) -> Optional[str]:
    if manifest:
        cell_id = manifest.get("cell_id")
        if cell_id:
            return str(cell_id).lower()
    lowered = name.lower()
    for cell_id in sorted(CELL_META, key=len, reverse=True):
        if re.search(rf"(^|_){re.escape(cell_id)}($|_r\d+|_)", lowered):
            return cell_id
    pilot_map = {
        "aging_pilot_24h_vllm_v1": "e1",
        "aging_pilot_24h_vllm_v0_ablation_v2": "a1",
        "aging_pilot_24h_triton_v1": "e2",
        "aging_pilot_24h_triton_v1_ablation_v2": "a2",
        "aging_pilot_24h_pytorch_naive_v1": "e3",
        "aging_pilot_24h_pytorch_naive_low_rate_v1": "e3b",
    }
    return pilot_map.get(lowered)


def infer_replica(name: str, manifest: Optional[dict] = None) -> Optional[str]:
    if manifest and manifest.get("replica") is not None:
        replica = manifest["replica"]
        return f"{int(replica):02d}" if str(replica).isdigit() else str(replica)
    match = re.search(r"_r(\d+)(?:_|$)", name.lower())
    if match:
        return f"{int(match.group(1)):02d}"
    return None


def proc_prefix_from_manifest(manifest: dict) -> Optional[str]:
    monitors = manifest.get("monitors")
    if isinstance(monitors, dict):
        proc = monitors.get("proc")
        if isinstance(proc, dict) and proc.get("label"):
            return str(proc["label"])

    args = manifest.get("args")
    if isinstance(args, dict) and args.get("label_engine"):
        return str(args["label_engine"])

    if isinstance(monitors, list):
        for monitor in monitors:
            if monitor.get("name") != "proc":
                continue
            cmd = monitor.get("cmd") or []
            for i, token in enumerate(cmd):
                if token == "--label" and i + 1 < len(cmd):
                    return str(cmd[i + 1])
    return None


def discover_proc_prefix(run_dir: Path, manifest: Optional[dict] = None) -> Optional[str]:
    prefix = proc_prefix_from_manifest(manifest or {})
    if prefix:
        return prefix

    counts: dict[str, int] = {}
    for path in run_dir.glob("*.csv"):
        match = re.match(r"(.+)_\d{6}\.csv$", path.name)
        if not match:
            continue
        candidate = match.group(1)
        if candidate.startswith("gpu") or candidate == "system":
            continue
        counts[candidate] = counts.get(candidate, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def warmup_from_manifest(manifest: dict, default_s: float = DEFAULT_WARMUP_S) -> float:
    if manifest.get("warmup_discard_s") is not None:
        return float(manifest["warmup_discard_s"])
    return default_s


def label_for_run(cell_id: Optional[str], replica: Optional[str], run_dir: Path) -> str:
    if cell_id and cell_id in CELL_META:
        base = CELL_META[cell_id]["label"]
    else:
        base = run_dir.name
    if replica:
        return f"{base} r{replica}"
    return base


def style_for_cell(cell_id: Optional[str], index: int = 0) -> tuple[str, str]:
    fallback_colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#7f8c8d", "#d35400"]
    fallback_styles = ["-", "--", "-.", ":"]
    if cell_id and cell_id in CELL_META:
        return CELL_META[cell_id]["color"], CELL_META[cell_id]["linestyle"]
    return fallback_colors[index % len(fallback_colors)], fallback_styles[index % len(fallback_styles)]


def spec_from_run_dir(run_dir: Path, index: int = 0, warmup_s: Optional[float] = None) -> RunSpec:
    run_dir = run_dir.expanduser().resolve()
    manifest = load_manifest(run_dir)
    cell_id = infer_cell_id(run_dir.name, manifest)
    replica = infer_replica(run_dir.name, manifest)
    color, linestyle = style_for_cell(cell_id, index)
    return RunSpec(
        id=(f"{cell_id.upper()} r{replica}" if cell_id and replica else run_dir.name),
        run_dir=run_dir,
        proc_prefix=discover_proc_prefix(run_dir, manifest),
        label=label_for_run(cell_id, replica, run_dir),
        color=color,
        linestyle=linestyle,
        warmup_s=warmup_s if warmup_s is not None else warmup_from_manifest(manifest),
        cell_id=cell_id,
        replica=replica,
    )


def specs_from_run_dirs(run_dirs: Sequence[Path], warmup_s: Optional[float] = None) -> list[RunSpec]:
    return [spec_from_run_dir(path, i, warmup_s=warmup_s) for i, path in enumerate(run_dirs)]


def specs_from_campaign(
    campaign_yaml: Path,
    runs_root: Optional[Path] = None,
    cells: Optional[set[str]] = None,
    replicas: Optional[set[str]] = None,
    warmup_s: Optional[float] = None,
) -> list[RunSpec]:
    campaign_yaml = campaign_yaml.expanduser().resolve()
    campaign = load_yaml_minimal(campaign_yaml, kind="campaign")
    campaign_id = campaign.get("campaign_id", campaign_yaml.parent.name)
    root = (runs_root or Path(campaign["runs_root"])).expanduser()
    cell_paths = [campaign_yaml.parent / rel for rel in campaign["cells"]]
    replicas_per_cell = int(campaign.get("replicas_per_cell", 1))

    specs: list[RunSpec] = []
    for cell_path in cell_paths:
        cell = load_yaml_minimal(cell_path, kind="cell")
        cell_id = str(cell["cell_id"]).lower()
        if cells is not None and cell_id not in cells:
            continue
        proc_prefix = str(cell["monitors"]["proc"]["label"])
        cell_warmup = warmup_s if warmup_s is not None else float(cell.get("warmup_discard_s", DEFAULT_WARMUP_S))
        color, linestyle = style_for_cell(cell_id, len(specs))
        for rep in range(1, replicas_per_cell + 1):
            rep_s = f"{rep:02d}"
            if replicas is not None and rep_s not in replicas and str(rep) not in replicas:
                continue
            run_id = f"{campaign_id}_{cell_id}_r{rep_s}"
            specs.append(
                RunSpec(
                    id=f"{cell_id.upper()} r{rep_s}",
                    run_dir=root / run_id,
                    proc_prefix=proc_prefix,
                    label=label_for_run(cell_id, rep_s, root / run_id),
                    color=color,
                    linestyle=linestyle,
                    warmup_s=cell_warmup,
                    cell_id=cell_id,
                    replica=rep_s,
                )
            )
    return specs


def load_yaml_minimal(path: Path, kind: str) -> dict:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return data
    except ImportError:
        pass
    except Exception as exc:
        print(f"  [warn] PyYAML could not parse {path}: {exc}; using minimal parser", file=sys.stderr)

    if kind == "campaign":
        return parse_campaign_yaml_minimal(path)
    if kind == "cell":
        return parse_cell_yaml_minimal(path)
    raise ValueError(f"unknown YAML kind: {kind}")


def scalar_value(text: str) -> str:
    value = text.split("#", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def parse_campaign_yaml_minimal(path: Path) -> dict:
    data: dict = {"cells": []}
    in_cells = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            in_cells = False
        if stripped == "cells:":
            in_cells = True
            continue
        if in_cells and stripped.startswith("- "):
            data["cells"].append(scalar_value(stripped[2:]))
            continue
        for key in ("campaign_id", "runs_root", "replicas_per_cell"):
            if stripped.startswith(f"{key}:"):
                data[key] = scalar_value(stripped.split(":", 1)[1])
    missing = [key for key in ("campaign_id", "runs_root", "cells") if not data.get(key)]
    if missing:
        raise ValueError(f"minimal campaign parser missing {missing} in {path}")
    return data


def parse_cell_yaml_minimal(path: Path) -> dict:
    data: dict = {"monitors": {"proc": {}}}
    in_monitors = False
    in_proc = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", line):
            in_monitors = False
            in_proc = False
        if stripped.startswith("cell_id:"):
            data["cell_id"] = scalar_value(stripped.split(":", 1)[1])
        elif stripped.startswith("warmup_discard_s:"):
            data["warmup_discard_s"] = float(scalar_value(stripped.split(":", 1)[1]))
        elif stripped == "monitors:":
            in_monitors = True
            in_proc = False
        elif in_monitors and re.match(r"^\s{2}proc:\s*$", line):
            in_proc = True
        elif in_proc and re.match(r"^\s{4}label:", line):
            data["monitors"]["proc"]["label"] = scalar_value(stripped.split(":", 1)[1])
        elif in_proc and re.match(r"^\s{2}[A-Za-z_][A-Za-z0-9_]*:", line):
            in_proc = False
    if "cell_id" not in data or "label" not in data["monitors"]["proc"]:
        raise ValueError(f"minimal cell parser could not find cell_id/proc label in {path}")
    return data


def default_specs(
    campaign_yaml: Optional[Path] = None,
    runs_root: Optional[Path] = None,
    run_dirs: Optional[Sequence[Path]] = None,
    cells: Optional[set[str]] = None,
    replicas: Optional[set[str]] = None,
    warmup_s: Optional[float] = None,
) -> list[RunSpec]:
    if campaign_yaml is not None:
        specs = specs_from_campaign(campaign_yaml, runs_root, cells, replicas, warmup_s)
    elif run_dirs:
        specs = specs_from_run_dirs(run_dirs, warmup_s=warmup_s)
    else:
        specs = [
            RunSpec(
                id=s.id,
                run_dir=s.run_dir,
                proc_prefix=s.proc_prefix,
                label=s.label,
                color=s.color,
                linestyle=s.linestyle,
                warmup_s=warmup_s if warmup_s is not None else s.warmup_s,
                cell_id=s.cell_id,
                replica=s.replica,
            )
            for s in PILOT_RUNS
        ]

    if cells is not None:
        specs = [s for s in specs if s.cell_id in cells or s.id.lower() in cells]
    if replicas is not None:
        specs = [s for s in specs if replica_matches(s.replica, replicas)]
    specs.sort(key=lambda s: (CELL_ORDER.get(s.cell_id or "", 999), s.replica or "", s.id))
    return specs


def truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def load_proc(run_dir: Path, prefix: Optional[str], columns: Optional[Iterable[str]] = None) -> Optional[pd.DataFrame]:
    if prefix is None:
        prefix = discover_proc_prefix(run_dir)
    if prefix is None:
        print(f"  [warn] no proc CSV prefix discovered in {run_dir}", file=sys.stderr)
        return None

    files = sorted(glob.glob(str(run_dir / f"{prefix}_*.csv")))
    if not files:
        print(f"  [warn] no proc CSVs for {run_dir} using prefix {prefix}", file=sys.stderr)
        return None

    wanted = set(columns or [])
    wanted.update({"ts_unix", "process_alive"})
    dfs = []
    for file_name in files:
        try:
            df = pd.read_csv(file_name, usecols=lambda col: col in wanted)
            if "process_alive" in df.columns:
                df = df[truthy_series(df["process_alive"])]
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            print(f"  [warn] skipping {file_name}: {exc}", file=sys.stderr)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True).sort_values("ts_unix").reset_index(drop=True)


def load_client(run_dir: Path, columns: Optional[Iterable[str]] = None) -> Optional[pd.DataFrame]:
    files = sorted(glob.glob(str(run_dir / "client" / "requests_*.csv")))
    if not files:
        return None
    wanted = set(columns or [])
    dfs = []
    for file_name in files:
        try:
            kwargs = {"usecols": (lambda col: col in wanted)} if wanted else {}
            df = pd.read_csv(file_name, **kwargs)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            print(f"  [warn] skipping {file_name}: {exc}", file=sys.stderr)
    if not dfs:
        return None
    sort_col = "submitted_at_unix" if "submitted_at_unix" in dfs[0].columns else None
    out = pd.concat(dfs, ignore_index=True)
    if sort_col:
        out = out.sort_values(sort_col).reset_index(drop=True)
    return out


def normalize_memory_frame(
    df: pd.DataFrame,
    warmup_s: float,
    memory_cols: Sequence[str] = ("rss_bytes", "vms_bytes"),
) -> tuple[Optional[pd.DataFrame], float]:
    """Discard warmup and add hours plus *_delta_mb columns.

    Returns the normalized frame and the Unix timestamp that corresponds
    to hour zero after warmup. If no samples remain after warmup, the
    frame is None and the second return value is still the intended
    post-warmup origin.
    """

    if df is None or df.empty or "ts_unix" not in df.columns:
        return None, float("nan")
    df = df.sort_values("ts_unix").reset_index(drop=True).copy()
    raw_t0 = float(df["ts_unix"].iloc[0])
    post_t0 = raw_t0 + float(warmup_s)
    df["hours"] = (df["ts_unix"] - post_t0) / 3600.0
    df = df[df["hours"] >= 0].reset_index(drop=True)
    if df.empty:
        return None, post_t0
    for col in memory_cols:
        if col in df.columns:
            base = float(df[col].iloc[0])
            out_col = col.replace("_bytes", "_delta_mb")
            df[out_col] = (df[col].astype(float) - base) / (1024 * 1024)
    return df, post_t0


def downsample_by_time(df: pd.DataFrame, seconds: float, ts_col: str = "ts_unix") -> pd.DataFrame:
    if df.empty or seconds <= 0 or ts_col not in df.columns:
        return df
    df = df.copy()
    t0 = float(df[ts_col].iloc[0])
    df["_plot_bin"] = ((df[ts_col] - t0) // seconds).astype("int64")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "_plot_bin"]
    out = df.groupby("_plot_bin", observed=True)[numeric_cols].median().reset_index(drop=True)
    return out


def max_hours(frames: Sequence[pd.DataFrame]) -> Optional[float]:
    values = [float(df["hours"].max()) for df in frames if df is not None and not df.empty and "hours" in df]
    if not values:
        return None
    return max(values)
