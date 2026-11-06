"""Whole-container memory sampling, opt-in for single-container serving stacks.

The root's PID/create-time, PID namespace and cgroup anchor the scope. Namespace
and cgroup enumeration also captures reparented workers and docker-exec children.
Components partition this inventory; they never independently rescan or sum a
PID twice. GPU compute PIDs are mandatory evidence of model coverage. USS is
the summable response; RSS/VMS remain diagnostics (shared mappings can repeat).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
from pathlib import Path

import psutil

MEMORY_FIELDS = ("uss_bytes", "rss_bytes", "vms_bytes", "pss_bytes")
SCOPE_FIELDS = ["model_scope_complete", "membership_changed", "scope_error",
                "pid_identities", "model_pids", "scope_pids"]


def _cgroups(pid: int) -> dict[str, str]:
    result = {}
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        result[controllers] = path
    return result


def _namespace(pid: int) -> str:
    return os.readlink(f"/proc/{pid}/ns/pid")


def _namespace_pid(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            return int(line.split()[-1])
    raise ValueError("kernel does not expose NSpid for container identity")


def _within_cgroups(candidate: dict, anchor: dict) -> bool:
    # A root '/' group is shared with the host and provides no isolation.
    useful = {key: value for key, value in anchor.items() if value != "/"}
    return bool(useful) and all(
        key in candidate and (candidate[key] == value or
                              candidate[key].startswith(value.rstrip("/") + "/"))
        for key, value in useful.items())


def gpu_compute_pids(devices: list[int]) -> set[int]:
    pids = set()
    for device in devices:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
             "-i", str(device)], capture_output=True, text=True, timeout=10, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines or any(not line.isdigit() for line in lines):
            raise ValueError(f"GPU {device}: missing or malformed compute PID evidence")
        pids.update(int(line) for line in lines)
    if not pids:
        raise ValueError("no model-holding GPU process found")
    return pids


class ContainerSampler:
    def __init__(self, spec: dict, output_dir: Path):
        self.spec = spec
        self.root_pid = int(spec["root_pid"])
        self.root_create_time = float(spec["root_create_time"])
        if self.root_pid <= 0 or not math.isfinite(self.root_create_time):
            raise ValueError("invalid container root identity")
        self.components = spec["components"]
        roles = [c["role"] for c in self.components]
        labels = [c["label"] for c in self.components]
        if (sorted(roles) != ["auxiliary", "engine", "orchestration"] or
                len(set(labels)) != 3 or any(c.get("group") != "engine" for c in self.components)):
            raise ValueError("container scope requires three disjoint roles in the engine aggregate")
        self.engine_pattern = re.compile(spec.get("engine_pattern", r"EngineCore"))
        self.orchestration_pattern = re.compile(spec.get(
            "orchestration_pattern", r"triton_python_backend_stub"))
        self.devices = [int(d) for d in spec["gpu_devices"]]
        if not self.devices:
            raise ValueError("model scope requires GPU devices")
        self.namespace = None
        self.cgroups = None
        self.previous: set[tuple[int, float]] = set()
        self.lost_identity_error = ""
        self.output_dir = output_dir
        self.lock = threading.Lock()

    def inventory(self) -> list[dict]:
        root = psutil.Process(self.root_pid)
        if root.create_time() != self.root_create_time:
            raise ValueError("container root PID was reused")
        if _namespace_pid(self.root_pid) != 1:
            raise ValueError("container root must be namespace PID1, not an engine child")
        namespace = _namespace(self.root_pid)
        cgroups = _cgroups(self.root_pid)
        if namespace == _namespace(os.getpid()) or not _within_cgroups(cgroups, cgroups):
            raise ValueError("container scope is not isolated from the host")
        if self.namespace is None:
            identity = self.output_dir / "scope_identity.json"
            identity.write_text(json.dumps({
                "scope_type": "container_tree", "root_pid": self.root_pid,
                "root_create_time": self.root_create_time,
                "pid_namespace": namespace, "cgroups": cgroups,
                "gpu_devices": self.devices}, indent=2))
            identity.chmod(0o644)
            self.namespace, self.cgroups = namespace, cgroups
        if namespace != self.namespace or cgroups != self.cgroups:
            raise ValueError("container namespace/cgroup identity changed")
        found = []
        for proc in psutil.process_iter():
            try:
                proc = psutil.Process(proc.pid)  # Do not reuse process_iter's cached PID identity.
                in_cgroup = _within_cgroups(_cgroups(proc.pid), self.cgroups)
                if not in_cgroup:
                    if _namespace(proc.pid) == self.namespace:
                        raise ValueError(f"PID {proc.pid} shares container namespace outside its cgroup")
                    continue
                # Cgroup is authoritative for descendants that create a nested
                # PID namespace. Intersecting with the root namespace would
                # silently omit those container processes.
                if proc.status() == psutil.STATUS_ZOMBIE:
                    continue
                found.append({"proc": proc, "pid": proc.pid, "ppid": proc.ppid(),
                              "create_time": proc.create_time(), "cmdline": " ".join(proc.cmdline())})
            except (FileNotFoundError, ProcessLookupError, psutil.NoSuchProcess):
                continue  # The post-sample inventory catches membership races.
        if self.root_pid not in {p["pid"] for p in found}:
            raise ValueError("container PID1 missing from scope")
        return found

    def partition(self, inventory: list[dict], model_pids: set[int]) -> dict[int, str]:
        by_pid = {p["pid"]: p for p in inventory}
        engines = {p["pid"] for p in inventory if self.engine_pattern.search(p["cmdline"])} | model_pids
        roles = {}
        for entry in inventory:
            pid = entry["pid"]
            cursor, visited = pid, set()
            while cursor in by_pid and cursor not in visited and cursor not in engines:
                visited.add(cursor)
                cursor = by_pid[cursor]["ppid"]
            if cursor in engines:
                roles[pid] = "engine"
            elif pid == self.root_pid or self.orchestration_pattern.search(entry["cmdline"]):
                roles[pid] = "orchestration"
            else:
                roles[pid] = "auxiliary"
        return roles

    def tick(self, ts: float) -> tuple[list[dict], dict[str, dict]]:
        # Watchdog can abandon a timed-out call. Never let its replacement race
        # the original thread's membership/identity state.
        if not self.lock.acquire(blocking=False):
            raise ValueError("previous container sample still running")
        try:
            return self._tick(ts)
        finally:
            self.lock.release()

    def _tick(self, ts: float) -> tuple[list[dict], dict[str, dict]]:
        error, model_complete, changed = "", False, False
        before, sampled, model_pids, roles = [], {}, set(), {}
        try:
            before = self.inventory()
            model_pids = gpu_compute_pids(self.devices)
            scope_pids = {p["pid"] for p in before}
            if not model_pids <= scope_pids:
                raise ValueError(f"model-holding GPU PIDs outside scope: {sorted(model_pids - scope_pids)}")
            roles = self.partition(before, model_pids)
            for entry in before:
                pid, ctime = entry["pid"], entry["create_time"]
                if psutil.Process(pid).create_time() != ctime:
                    raise ValueError(f"PID {pid} reused before sampling")
                mem = entry["proc"].memory_full_info()
                values = {f: getattr(mem, f.removesuffix("_bytes"), None) for f in MEMORY_FIELDS}
                if any(v is None or not math.isfinite(v) or v < 0 for v in values.values()):
                    raise ValueError(f"PID {pid}: USS/RSS/VMS/PSS unavailable")
                sampled[pid] = values
            after = self.inventory()
            identities = {(p["pid"], p["create_time"]) for p in before}
            post_identities = {(p["pid"], p["create_time"]) for p in after}
            # A disappearing/replaced process can reset memory. Latch the failure
            # instead of accepting a lower total on subsequent stable ticks.
            lost = (self.previous | identities) - post_identities
            if lost:
                self.lost_identity_error = f"sampled process identity lost/replaced: {sorted(lost)}"
            changed = identities != post_identities or bool(self.previous and identities != self.previous)
            self.previous = post_identities
            if changed or not (self.output_dir / "scope_membership.jsonl").exists():
                audit = self.output_dir / "scope_membership.jsonl"
                with audit.open("a") as stream:
                    stream.write(json.dumps({"ts_unix": ts, "model_pids": sorted(model_pids),
                                             "processes": [
                                                 {k: v for k, v in entry.items() if k != "proc"} |
                                                 {"role": roles[entry["pid"]]} for entry in before],
                                             "post_identities": sorted(post_identities)}) + "\n")
                audit.chmod(0o644)
            if self.lost_identity_error:
                raise ValueError(self.lost_identity_error)
            if identities != post_identities:
                raise ValueError("membership changed during sampling")
            model_complete = model_pids <= set(sampled)
        except (OSError, psutil.Error, ValueError, subprocess.SubprocessError) as exc:
            error = str(exc)

        complete = not error and model_complete
        rows = []
        for comp in self.components:
            members = [p for p in before if roles.get(p["pid"]) == comp["role"]]
            pids = {p["pid"] for p in members}
            values = {f: sum(sampled[pid][f] for pid in pids if pid in sampled) for f in MEMORY_FIELDS}
            rows.append({"ts_unix": ts, "label": comp["label"], "group": "engine",
                         "n_pids_matched": len(pids), "n_pids_sampled": len(pids & sampled.keys()),
                         "n_pgids_alive": None, "n_pids_unexpected": 0,
                         "membership_complete": complete,
                         "pids": ";".join(map(str, sorted(pids & sampled.keys()))),
                         **values})
        # Semicolons keep all evidence fields safe for the streaming awk health
        # checker, which deliberately uses comma as its field separator.
        aggregate = {
            "n_components_expected": 3, "n_components_complete": 3 if complete else 0,
            "n_pids_sampled": len(sampled), "n_pids_unexpected": 0,
            "model_scope_complete": model_complete, "membership_changed": changed,
            "scope_error": error.replace(",", ";").replace("\n", " ").replace('"', "'"),
            "pid_identities": ";".join(f"{p['pid']}@{p['create_time']}" for p in before),
            "scope_pids": ";".join(str(p["pid"]) for p in before),
            "model_pids": ";".join(map(str, sorted(model_pids))),
            **{f: sum(s[f] for s in sampled.values()) if complete else None for f in MEMORY_FIELDS},
        }
        return rows, {"engine": aggregate}
