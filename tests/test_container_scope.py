"""Whole-container scope regressions, off-box: no Docker or GPU required."""
import csv
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import psutil
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "monitoring"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "analysis"))
import container_scope as cs
import launch_cell as lc
import validate_extension_run as validator
from _common import CsvRotatingWriter, WriterConfig
from multiproc_monitor import AGG_FIELDS, COMPONENT_FIELDS


def spec():
    result = yaml.safe_load((REPO / "campaigns/extension/cells/val_vllm.yaml").read_text())["monitors"]["components"]
    return dict(result, root_pid=10, root_create_time=1.0, gpu_devices=[0])


def process(pid, parent=10, cmd="python helper", ctime=None, uss=100):
    ctime = float(pid) if ctime is None else ctime
    proc = mock.Mock(pid=pid)
    proc.create_time.return_value = ctime
    proc.ppid.return_value = parent
    proc.cmdline.return_value = cmd.split()
    proc.status.return_value = "running"
    proc.memory_full_info.return_value = types.SimpleNamespace(uss=uss, rss=uss * 2, vms=uss * 3, pss=uss)
    return {"proc": proc, "pid": pid, "ppid": parent, "cmdline": cmd, "create_time": ctime}


def inventory():
    return [process(10, 1, "vllm serve", 1.0, 100),
            process(20, 10, "VLLM::EngineCore", uss=1000),
            process(21, 20, "python engine child", uss=300),
            process(30, 10, "python resource_tracker", uss=20)]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ScopeFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rd = Path(self.tmp.name)
        self.sampler = cs.ContainerSampler(spec(), self.rd)
        self.entries = inventory()
        self.pmap = {p["pid"]: p["proc"] for p in self.entries}

    def sample(self, entries=None, after=None, model=None):
        entries = self.entries if entries is None else entries
        after = entries if after is None else after
        all_entries = entries + after
        pmap = {p["pid"]: p["proc"] for p in all_entries}
        with mock.patch.object(self.sampler, "inventory", side_effect=[entries, after]), \
             mock.patch.object(cs, "gpu_compute_pids", return_value={20} if model is None else model), \
             mock.patch.object(cs.psutil, "Process", side_effect=lambda pid: pmap[pid]):
            rows, groups = self.sampler.tick(100.0)
        return rows, groups["engine"]


class Sampling(ScopeFixture):
    def test_pid1_core_children_and_aux_partition_exactly_once(self):
        rows, agg = self.sample()
        self.assertTrue(agg["model_scope_complete"])
        self.assertEqual(agg["n_components_complete"], 3)
        self.assertEqual(agg["uss_bytes"], 1420)
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["vllm_orchestration"]["pids"], "10")
        self.assertEqual(by_label["vllm_engine"]["pids"], "20;21")
        self.assertEqual(by_label["vllm_auxiliary"]["pids"], "30")
        self.assertEqual(sum(r["uss_bytes"] for r in rows), agg["uss_bytes"])
        for entry in self.entries:
            entry["proc"].memory_full_info.assert_called_once()
        audit = json.loads((self.rd / "scope_membership.jsonl").read_text())
        self.assertEqual(len(audit["processes"]), 4)

    def test_missing_model_process_fails_instead_of_frontend_only_pass(self):
        rows, agg = self.sample(self.entries[:1])
        self.assertFalse(agg["model_scope_complete"])
        self.assertEqual(agg["n_components_complete"], 0)
        self.assertIsNone(agg["uss_bytes"])
        self.assertIn("outside scope", agg["scope_error"])

    def test_external_gpu_process_is_not_silently_ignored(self):
        _, agg = self.sample(model={20, 999})
        self.assertEqual(agg["n_components_complete"], 0)

    def test_same_tick_exit_is_incomplete_and_latched(self):
        self.sample()
        _, broken = self.sample(after=self.entries[:-1])
        self.assertTrue(broken["membership_changed"])
        self.assertEqual(broken["n_components_complete"], 0)
        _, stable = self.sample(self.entries[:-1])
        self.assertEqual(stable["n_components_complete"], 0)
        self.assertIn("lost/replaced", stable["scope_error"])

    def test_engine_respawn_cannot_be_accepted_as_lower_memory(self):
        self.sample()
        new = [p for p in self.entries if p["pid"] not in (20, 21)] + [process(40, cmd="EngineCore", uss=10)]
        _, agg = self.sample(new, model={40})
        self.assertEqual(agg["n_components_complete"], 0)
        self.assertIsNone(agg["uss_bytes"])

    def test_pid_reuse_between_ticks_is_detected(self):
        self.sample()
        new = [p if p["pid"] != 20 else process(20, cmd="EngineCore", ctime=99.0) for p in self.entries]
        _, agg = self.sample(new)
        self.assertIn("lost/replaced", agg["scope_error"])

    def test_birth_during_sample_is_incomplete_until_stable(self):
        new = self.entries + [process(31)]
        _, agg = self.sample(after=new)
        self.assertEqual(agg["n_components_complete"], 0)
        _, agg = self.sample(new)
        self.assertEqual(agg["n_components_complete"], 3)

    def test_unreadable_uss_is_not_summed_as_zero(self):
        self.entries[1]["proc"].memory_full_info.side_effect = psutil.AccessDenied(20)
        _, agg = self.sample()
        self.assertEqual(agg["n_components_complete"], 0)
        self.assertIsNone(agg["uss_bytes"])

    def test_missing_memory_attribute_is_incomplete(self):
        self.entries[1]["proc"].memory_full_info.return_value = types.SimpleNamespace(rss=1, vms=2, pss=1)
        _, agg = self.sample()
        self.assertIn("unavailable", agg["scope_error"])

    def test_empty_optional_component_is_valid_zero(self):
        rows, agg = self.sample(self.entries[:-1])
        self.assertEqual(agg["n_components_complete"], 3)
        self.assertEqual(rows[-1]["uss_bytes"], 0)

    def test_watchdog_reentry_does_not_race_state(self):
        self.sampler.lock.acquire()
        try:
            with self.assertRaisesRegex(ValueError, "still running"):
                self.sampler.tick(1.0)
        finally:
            self.sampler.lock.release()

    def test_inventory_includes_reparented_member_and_root_excludes_host(self):
        reparented = process(40, 1, "helper")
        host = process(999, 1, "host")
        procs = {p["pid"]: p["proc"] for p in self.entries + [reparented, host]}
        with mock.patch.object(cs.psutil, "Process", side_effect=lambda pid: procs[pid]), \
             mock.patch.object(cs.psutil, "process_iter", return_value=list(procs.values())), \
             mock.patch.object(cs, "_namespace", side_effect=lambda pid: "host" if pid in (999, cs.os.getpid()) else "container"), \
             mock.patch.object(cs, "_namespace_pid", return_value=1), \
             mock.patch.object(cs, "_cgroups", side_effect=lambda pid: {"": "/" if pid == 999 else "/docker/test"}):
            found = self.sampler.inventory()
        self.assertEqual({p["pid"] for p in found}, {10, 20, 21, 30, 40})
        self.assertTrue((self.rd / "scope_identity.json").exists())

    def test_nested_pid_namespace_inside_container_cgroup_is_not_omitted(self):
        procs = {p["pid"]: p["proc"] for p in self.entries}
        with mock.patch.object(cs.psutil, "Process", side_effect=lambda pid: procs[pid]), \
             mock.patch.object(cs.psutil, "process_iter", return_value=list(procs.values())), \
             mock.patch.object(cs, "_namespace", side_effect=lambda pid: "host" if pid == cs.os.getpid() else ("nested" if pid == 20 else "container")), \
             mock.patch.object(cs, "_namespace_pid", return_value=1), \
             mock.patch.object(cs, "_cgroups", return_value={"": "/docker/test"}):
            self.assertEqual({p["pid"] for p in self.sampler.inventory()}, {10, 20, 21, 30})

    def test_root_identity_reuse_refused(self):
        with mock.patch.object(cs.psutil, "Process", return_value=process(10)["proc"]):
            with self.assertRaisesRegex(ValueError, "reused"):
                self.sampler.inventory()

    def test_engine_child_cannot_be_used_as_container_root(self):
        with mock.patch.object(cs.psutil, "Process", return_value=self.entries[0]["proc"]), \
             mock.patch.object(cs, "_namespace_pid", return_value=9):
            with self.assertRaisesRegex(ValueError, "namespace PID1"):
                self.sampler.inventory()

    def test_failed_gpu_query_is_an_incomplete_sample(self):
        with mock.patch.object(self.sampler, "inventory", return_value=self.entries), \
             mock.patch.object(cs, "gpu_compute_pids", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
            _, groups = self.sampler.tick(1)
        self.assertFalse(groups["engine"]["model_scope_complete"])
        self.assertEqual(groups["engine"]["n_components_complete"], 0)

    def test_gpu_pid_parser_rejects_unavailable_evidence(self):
        for output in ("", "N/A\n", "20\ninvalid\n"):
            with mock.patch.object(cs.subprocess, "run", return_value=types.SimpleNamespace(stdout=output)):
                with self.assertRaises(ValueError):
                    cs.gpu_compute_pids([0])

    def test_host_cgroup_is_not_container_evidence(self):
        self.assertFalse(cs._within_cgroups({"": "/"}, {"": "/"}))
        self.assertFalse(cs._within_cgroups({"": "/docker/ab"}, {"": "/docker/a"}))
        self.assertTrue(cs._within_cgroups({"": "/docker/a/child"}, {"": "/docker/a"}))


class RecordedEvidence(ScopeFixture):
    def setUp(self):
        super().setUp()
        self.rows, self.agg = self.sample()
        self.agg.update(ts_unix=100.0, process_alive=True, membership_complete=True)
        (self.rd / "scope_identity.json").write_text(json.dumps({
            "root_pid": 10, "root_create_time": 1.0,
            "pid_namespace": "container", "cgroups": {"": "/docker/test"}}))

    def save(self):
        for row in self.rows:
            write_csv(self.rd / f"{row['label']}_000000.csv", [row])
        write_csv(self.rd / "agg_vllm_000000.csv", [self.agg])

    def check(self):
        self.save()
        return validator.check_container_scope(self.rd, spec())[0]

    def test_valid_archived_evidence_passes(self):
        self.assertTrue(self.check())

    def test_overlap_fails(self):
        self.rows[0]["pids"] = "10;20"
        self.rows[0]["n_pids_sampled"] = self.rows[0]["n_pids_matched"] = 2
        self.assertFalse(self.check())

    def test_aggregate_sum_mismatch_fails(self):
        self.agg["uss_bytes"] += 2
        self.assertFalse(self.check())

    def test_model_pid_excluded_fails(self):
        self.agg["model_pids"] = "999"
        self.assertFalse(self.check())

    def test_missing_pid1_fails(self):
        self.rows[0]["pids"] = ""
        self.rows[0]["n_pids_sampled"] = self.rows[0]["n_pids_matched"] = 0
        self.assertFalse(self.check())

    def test_missing_model_flag_fails(self):
        del self.agg["model_scope_complete"]
        self.assertFalse(self.check())

    def test_nan_memory_fails(self):
        self.agg["uss_bytes"] = float("nan")
        self.assertFalse(self.check())

    def test_real_csv_schema_and_analysis_use_total_and_components(self):
        # Exercise actual writer schemas and the real analysis command, including
        # an empty optional component and numeric PID evidence in CSV readers.
        self.save()
        with mock.patch.object(cs, "gpu_compute_pids", return_value={20}):
            self.assertTrue(self.check())
        manifest = {"cell_id": "val_vllm", "proc_prefix": "agg_vllm", "warmup_discard_s": 0,
                    "monitors": {"components": spec()}}
        (self.rd / "manifest.json").write_text(json.dumps(manifest))
        for row in self.rows:
            writer = CsvRotatingWriter(WriterConfig(self.rd, row["label"], 60, COMPONENT_FIELDS))
            for i in range(30):
                writer.write(dict(row, ts_unix=100 + i * 60, uss_bytes=100 + i))
            writer.close()
        writer = CsvRotatingWriter(WriterConfig(self.rd, "agg_vllm", 60, AGG_FIELDS + cs.SCOPE_FIELDS))
        for i in range(30):
            writer.write(dict(self.agg, ts_unix=100 + i * 60, uss_bytes=1000 + i))
        writer.close()
        result = subprocess.run([sys.executable, str(REPO / "analysis/aging_trends.py"),
                                 str(self.rd), "--csv"], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        for indicator in ("proc.uss_bytes", "component.engine.uss_bytes",
                          "component.orchestration.uss_bytes", "component.auxiliary.uss_bytes"):
            self.assertIn(indicator, result.stdout)
        # Standalone run_monitors manifests store a LIST, not a dict. Scope
        # support must not break archived runs with that historical shape.
        manifest["monitors"] = [{"name": "proc"}]
        (self.rd / "manifest.json").write_text(json.dumps(manifest))
        result = subprocess.run([sys.executable, str(REPO / "analysis/aging_trends.py"),
                                 str(self.rd), "--csv"], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("proc.uss_bytes", result.stdout)


class Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rd = Path(self.tmp.name)
        self.cell = yaml.safe_load((REPO / "campaigns/extension/cells/val_vllm.yaml").read_text())
        self.lifecycle = lc.SingleContainerLifecycle(self.cell, types.SimpleNamespace(repo_root=REPO),
                                                      self.rd, self.rd, {}, "image")
        self.lifecycle.container_pid = 10

    def test_scope_identity_materialized_without_single_pid_daemon(self):
        with mock.patch.object(psutil, "Process", return_value=process(10, ctime=1.0)["proc"]):
            self.assertEqual(self.lifecycle.resolve_pid_identity(), (None, None))
        scoped = self.cell["monitors"]["components"]
        self.assertEqual(scoped["root_pid"], 10)
        self.assertEqual(scoped["root_create_time"], 1.0)
        self.assertEqual(lc.proc_prefix_for_cell(self.cell), "agg_vllm")

    def test_runtime_missing_or_incomplete_scope_is_not_healthy(self):
        self.lifecycle.scope_monitoring = True
        with mock.patch.object(lc, "container_running", return_value=True), \
             mock.patch.object(lc, "read_latest_membership_tick", return_value=None), \
             mock.patch.object(lc.time, "monotonic", return_value=self.lifecycle.scope_started_mono + 121):
            self.assertIn("missing/stale", self.lifecycle.health_check())
        row = {"ts_unix": "1", "membership_complete": "True", "process_alive": "True"}
        with mock.patch.object(lc, "container_running", return_value=True), \
             mock.patch.object(lc, "read_latest_membership_tick", return_value=row):
            self.assertIn("model_scope_complete", self.lifecycle.health_check())

    def test_calibration_health_does_not_require_unstarted_monitors(self):
        with mock.patch.object(lc, "container_running", return_value=True), \
             mock.patch.object(lc.time, "monotonic", return_value=self.lifecycle.scope_started_mono + 3600):
            self.assertIsNone(self.lifecycle.health_check())


if __name__ == "__main__":
    unittest.main()
