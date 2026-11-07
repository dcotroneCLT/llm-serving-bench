"""Cross-framework regressions from the 2026-11-06 scope audit."""
import contextlib
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
from test_container_scope import ScopeFixture, process, spec, write_csv
import aging_io
import stepness
import validate_extension_run as validator
from run_integrity import check_run_integrity, saved_client_config


class PartitionCorrections(ScopeFixture):
    def test_gpu_context_on_frontend_does_not_swallow_engine_and_auxiliary(self):
        rows, agg = self.sample(model={10, 20})
        self.assertEqual([r["uss_bytes"] for r in rows], [100, 1300, 20])
        self.assertEqual(agg["model_pids"], "20")
        self.assertEqual(agg["gpu_pids"], "10;20")
        self.assertEqual(agg["n_components_complete"], 3)

    def test_ambiguous_gpu_process_does_not_become_verified_inference(self):
        entries = [dict(p, cmdline="python unknown") if p["pid"] == 20 else p for p in self.entries]
        _, agg = self.sample(entries)
        self.assertEqual(agg["n_components_complete"], 0)
        self.assertIn("ambiguous inference role", agg["scope_error"])

    def test_frontend_gpu_context_alone_is_not_model_evidence(self):
        _, agg = self.sample(model={10})
        self.assertFalse(agg["model_scope_complete"])

    def test_unclassified_gpu_worker_requires_explicit_role_resolution(self):
        _, agg = self.sample(model={20, 30})
        self.assertEqual(agg["n_components_complete"], 0)
        self.assertIn("unclassified GPU", agg["scope_error"])

    def test_reparented_worker_keeps_attribution_and_updates_audit(self):
        before, _ = self.sample()
        entries = [dict(p, ppid=1) if p["pid"] == 21 else p for p in self.entries]
        after, agg = self.sample(entries)
        self.assertEqual([r["uss_bytes"] for r in before], [r["uss_bytes"] for r in after])
        self.assertFalse(agg["membership_changed"])
        audit = [json.loads(line) for line in (self.rd / "scope_membership.jsonl").read_text().splitlines()]
        self.assertEqual(len(audit), 2)
        helper = next(p for p in audit[-1]["processes"] if p["pid"] == 21)
        self.assertEqual((helper["ppid"], helper["role"]), (1, "engine"))

    def test_semantic_role_change_of_existing_identity_fails_loudly(self):
        self.sample()
        entries = [dict(p, cmdline="EngineCore") if p["pid"] == 30 else p for p in self.entries]
        _, agg = self.sample(entries, model={20, 30})
        self.assertEqual(agg["n_components_complete"], 0)
        self.assertIn("ambiguous role change", agg["scope_error"])


class WholeRunFixture(ScopeFixture):
    def setUp(self):
        super().setUp()
        self.rows, self.agg = self.sample()
        self.start, self.duration = 1000, 1200
        self.manifest = {
            "run_id": self.rd.name, "cell_id": "dow_vllm_p01", "proc_prefix": "agg_vllm",
            "started_at_unix": self.start, "ended_at_unix": self.start + self.duration + 120,
            "ended_at": "done", "duration_s": self.duration,
            "duration_seconds_actual": self.duration + 120, "interrupted_early": False,
            "warmup_discard_s": 120,
            "engine": {"gpu_device": 0},
            "monitors": {"components": spec(), "proc": {"period_s": 5},
                         "gpu": {"period_s": 1}, "system": {"period_s": 5}},
            "workload": {"client_config_overrides": {"target_rate_rps": 0.2,
                         "prefix_repeat_fraction": 0.4, "arrival_mode": "poisson"}},
        }
        self.write_manifest()
        (self.rd / "scope_identity.json").write_text(json.dumps({
            "scope_version": 2, "root_pid": 10, "root_create_time": 1.0,
            "pid_namespace": "container", "cgroups": {"": "/docker/test"}}))
        self.aggregates = [dict(self.agg, ts_unix=self.start + t,
                               process_alive=True, membership_complete=True)
                           for t in range(0, self.duration, 5)]
        self.write_aggregate()
        for row in self.rows:
            write_csv(self.rd / f"{row['label']}_000000.csv", [
                dict(row, ts_unix=self.start + t) for t in range(0, self.duration, 5)])
        write_csv(self.rd / "gpu0_000000.csv", [
            {"ts_unix": self.start + t, "gpu_index": 0, "vram_used_bytes": 40_000_000_000}
            for t in range(self.duration)])
        write_csv(self.rd / "system_000000.csv", [
            {"ts_unix": self.start + t, "mem_used_bytes": 5_000_000_000}
            for t in range(0, self.duration, 5)])
        (self.rd / "client").mkdir()
        write_csv(self.rd / "client/requests_000000.csv", [
            {"submitted_at_unix": self.start + t, "status": "ok", "e2e_latency_s": 1,
             "shared_prefix_applied": (t // 5) % 5 < 2}
            for t in range(0, self.duration, 5)])
        (self.rd / "client/arrival_stats.json").write_text(json.dumps({
            "arrival_mode": "poisson", "interarrival_cv": 1}))

    def write_manifest(self):
        (self.rd / "manifest.json").write_text(json.dumps(self.manifest))

    def write_aggregate(self):
        write_csv(self.rd / "agg_vllm_000000.csv", self.aggregates)

    def run_validator(self, *args):
        return subprocess.run([sys.executable, str(REPO / "analysis/validate_extension_run.py"),
                               "--run-dir", str(self.rd), *args],
                              capture_output=True, text=True, timeout=60)


class Integrity(WholeRunFixture):
    def test_saved_workload_target_zero_is_used_by_full_cli(self):
        config = dict(self.manifest["workload"]["client_config_overrides"], prefix_repeat_fraction=0)
        (self.rd / "client_config.yaml").write_text(yaml.safe_dump(config))
        client = self.rd / "client/requests_000000.csv"
        client.write_text(client.read_text().replace(",True", ",False"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_complete_stationary_run_is_eligible_without_significant_aging(self):
        ok, messages = check_run_integrity(self.rd, self.manifest)
        self.assertTrue(ok, messages)
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("GATE: PASS", result.stdout)

    def test_scope_only_still_requires_complete_short_run(self):
        result = self.run_validator("--scope-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_interrupted_run_is_rejected_even_with_good_rows(self):
        self.manifest["interrupted_early"] = True
        self.write_manifest()
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GATE: FAIL", result.stdout)

    def test_interrupted_five_seconds_of_36h_cannot_pass_scope_or_full_gate(self):
        self.manifest.update(duration_s=129600, duration_seconds_actual=5,
                             ended_at_unix=self.start + 5, interrupted_early=True)
        self.write_manifest()
        self.aggregates = self.aggregates[:1]
        self.write_aggregate()
        for args in ((), ("--scope-only",)):
            result = self.run_validator(*args)
            self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_partial_or_header_only_streams_fail(self):
        for prefix in ("gpu0", "system", "agg_vllm"):
            path = self.rd / f"{prefix}_000000.csv"
            original = path.read_text()
            for lines in (1, 2):
                path.write_text("\n".join(original.splitlines()[:lines]) + "\n")
                self.assertFalse(check_run_integrity(self.rd, self.manifest)[0], prefix)
            path.write_text(original)

    def test_missing_finalization_and_bad_status_fail(self):
        for key, value in (("ended_at_unix", None), ("duration_seconds_actual", None),
                           ("interrupted_early", "False"), ("client_forced_kill", True),
                           ("teardown_errors", ["failure"])):
            changed = dict(self.manifest, **{key: value})
            self.assertFalse(check_run_integrity(self.rd, changed)[0], key)

    def test_duplicate_and_large_gap_fail(self):
        self.aggregates[1]["ts_unix"] = self.aggregates[0]["ts_unix"]
        self.write_aggregate()
        self.assertFalse(check_run_integrity(self.rd, self.manifest)[0])
        self.aggregates = self.aggregates[:1] + self.aggregates[20:]
        self.write_aggregate()
        self.assertFalse(check_run_integrity(self.rd, self.manifest)[0])

    def test_unreadable_or_nonfinite_memory_does_not_count_as_coverage(self):
        for row in self.aggregates:
            row["uss_bytes"] = "nan"
        self.write_aggregate()
        self.assertFalse(check_run_integrity(self.rd, self.manifest)[0])

    def _shift_client(self, delay, extra=0):
        """Client started `delay` s after started_at and ran the full duration."""
        write_csv(self.rd / "client/requests_000000.csv", [
            {"submitted_at_unix": self.start + delay + t, "status": "ok", "e2e_latency_s": 1,
             "shared_prefix_applied": (t // 5) % 5 < 2}
            for t in range(0, self.duration + extra, 5)])

    def test_client_spawn_delay_is_tolerated(self):
        for delay in (5, 12, 20):
            self._shift_client(delay)
            ok, messages = check_run_integrity(self.rd, self.manifest)
            self.assertTrue(ok, (delay, messages))

    def test_client_running_far_past_window_or_before_start_fails(self):
        self._shift_client(300)
        self.assertFalse(check_run_integrity(self.rd, self.manifest)[0])
        self._shift_client(-5)
        self.assertFalse(check_run_integrity(self.rd, self.manifest)[0])

    def _extend_monitors(self, until):
        """Monitors keep sampling until `until` seconds after start."""
        self.aggregates = [dict(self.agg, ts_unix=self.start + t, process_alive=True,
                                membership_complete=True) for t in range(0, until, 5)]
        self.write_aggregate()
        for row in self.rows:
            write_csv(self.rd / f"{row['label']}_000000.csv", [
                dict(row, ts_unix=self.start + t) for t in range(0, until, 5)])
        write_csv(self.rd / "gpu0_000000.csv", [
            {"ts_unix": self.start + t, "gpu_index": 0, "vram_used_bytes": 40_000_000_000}
            for t in range(until)])
        write_csv(self.rd / "system_000000.csv", [
            {"ts_unix": self.start + t, "mem_used_bytes": 5_000_000_000}
            for t in range(0, until, 5)])

    def test_monitors_sampling_through_client_drain_pass(self):
        for drain in (300, 690):
            manifest = dict(self.manifest, ended_at_unix=self.start + self.duration + drain + 20,
                            duration_seconds_actual=self.duration + drain + 20)
            self._extend_monitors(self.duration + drain)
            ok, messages = check_run_integrity(self.rd, manifest)
            self.assertTrue(ok, (drain, messages))

    def test_monitor_sample_after_manifest_end_fails(self):
        self._extend_monitors(self.duration + 700)
        manifest = dict(self.manifest, ended_at_unix=self.start + self.duration + 300,
                        duration_seconds_actual=self.duration + 300)
        self.assertFalse(check_run_integrity(self.rd, manifest)[0])

    def test_missing_and_all_failed_client_streams_fail(self):
        path = self.rd / "client/requests_000000.csv"
        path.write_text(path.read_text().replace(",ok,", ",error,"))
        self.assertFalse(check_run_integrity(self.rd, self.manifest)[0])

    def test_insufficient_statistical_rows_are_not_usable_analysis(self):
        header = "indicator,n_samples,slope,slope_ci_lo,slope_ci_hi,mk_p_value\n"
        for row in ("proc.uss_bytes,1,nan,nan,nan,nan\n", "proc.uss_bytes,20,0,0,0,nan\n"):
            with mock.patch.object(validator.subprocess, "run", return_value=mock.Mock(
                    returncode=0, stdout=header + row)):
                self.assertFalse(validator.check_analysis(self.rd, REPO)[0])

    def test_scope_version_one_is_not_silently_approved(self):
        old = dict(spec(), scope_version=1)
        self.assertFalse(validator.check_container_scope(self.rd, old)[0])

    def test_gpu_contexts_are_checked_separately_from_verified_model_pids(self):
        for row in self.aggregates:
            row["gpu_pids"] = "10;20"
            row["model_pids"] = "20"
        self.write_aggregate()
        self.assertTrue(validator.check_container_scope(self.rd, spec())[0])
        self.aggregates[0]["gpu_pids"] = "20;999"
        self.write_aggregate()
        self.assertFalse(validator.check_container_scope(self.rd, spec())[0])

    def test_validator_rejects_role_transfer_even_when_total_is_conserved(self):
        engine_path = self.rd / "vllm_engine_000000.csv"
        aux_path = self.rd / "vllm_auxiliary_000000.csv"
        with engine_path.open() as stream:
            engine = list(csv.DictReader(stream))
        with aux_path.open() as stream:
            aux = list(csv.DictReader(stream))
        engine[1].update(pids="20", n_pids_matched=1, n_pids_sampled=1,
                         uss_bytes=1000, rss_bytes=2000, vms_bytes=3000, pss_bytes=1000)
        aux[1].update(pids="21;30", n_pids_matched=2, n_pids_sampled=2,
                      uss_bytes=320, rss_bytes=640, vms_bytes=960, pss_bytes=320)
        write_csv(engine_path, engine)
        write_csv(aux_path, aux)
        ok, messages = validator.check_container_scope(self.rd, spec())
        self.assertFalse(ok)
        self.assertIn("semantic role", " ".join(messages))

    def test_legacy_dynamo_layout_still_passes_full_gate(self):
        self.manifest.update(proc_prefix="agg_dynamo", gpu_devices=[0, 1],
                             topology={"prefill_gpu": 0, "decode_gpu": 1})
        mapping = {"vllm_orchestration": "dynamo_frontend", "vllm_engine": "dynamo_prefill",
                   "vllm_auxiliary": "dynamo_decode"}
        components = []
        for old, new in mapping.items():
            path = self.rd / f"{old}_000000.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row.update(label=new, pids=row["pids"].replace(";", ","))
            write_csv(self.rd / f"{new}_000000.csv", rows)
            components.append({"label": new, "group": "engine", "pgids": [10]})
        self.manifest["monitors"]["components"] = {"engine_group": "dynamo", "components": components}
        self.write_manifest()
        write_csv(self.rd / "agg_dynamo_000000.csv", self.aggregates)
        write_csv(self.rd / "agg_infra_000000.csv", [{"ts_unix": self.start, "uss_bytes": 50}])
        write_csv(self.rd / "gpu1_000000.csv", [
            {"ts_unix": self.start + t, "gpu_index": 1, "vram_used_bytes": 40_000_000_000}
            for t in range(self.duration)])
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class MembershipAnalysis(WholeRunFixture):
    def test_missing_membership_evidence_cannot_fall_back_to_naive_diffs(self):
        for row in self.aggregates:
            del row["pid_identities"]
        self.write_aggregate()
        self.assertIsNone(stepness.analyze_run(self.rd, warmup_s=0, n_bootstrap=1, seed=1))

    def test_batch_cli_includes_completed_extension_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "extension_dow_screening_dow_vllm_p01_r01"
            shutil.copytree(self.rd, run)
            manifest = dict(self.manifest, run_id=run.name)
            (run / "manifest.json").write_text(json.dumps(manifest))
            result = subprocess.run([sys.executable, str(REPO / "analysis/stepness.py"),
                                     "--logs-root", str(root), "--bootstrap", "1", "--csv"],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(run.name, result.stdout)

    def test_birth_is_not_reported_as_allocator_step(self):
        extra = process(40, 10, "python helper", uss=5_000_000)
        rows2, agg2 = self.sample(self.entries + [extra])
        for i in range(120, len(self.aggregates)):
            self.aggregates[i].update(agg2, membership_changed=(i == 120))
        self.write_aggregate()
        result = stepness.analyze_run(self.rd, warmup_s=0, n_bootstrap=10, seed=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["steps_per_h_1mb"], 0)
        self.assertEqual(result["class"], "continuous drift")

    def test_identity_order_does_not_create_boundaries(self):
        ids = self.aggregates[0]["pid_identities"]
        for i, row in enumerate(self.aggregates):
            row["pid_identities"] = ids if i % 2 else ";".join(reversed(ids.split(";")))
        self.write_aggregate()
        result = stepness.analyze_run(self.rd, warmup_s=0, n_bootstrap=1, seed=1)
        self.assertEqual(result["n_samples"], len(self.aggregates) - 1)


class WorkloadFeatures(WholeRunFixture):
    def test_saved_config_overrides_manifest_without_default_point_eight(self):
        config = dict(self.manifest["workload"]["client_config_overrides"], prefix_repeat_fraction=0)
        (self.rd / "client_config.yaml").write_text(yaml.safe_dump(config))
        self.assertEqual(saved_client_config(self.rd, self.manifest)["prefix_repeat_fraction"], 0)

    def test_zero_center_and_high_targets(self):
        path = self.rd / "client/requests_000000.csv"
        for target in (0, 0.4, 0.8):
            write_csv(path, [{"shared_prefix_applied": i < 100 * target} for i in range(100)])
            self.assertTrue(validator.check_client_features(self.rd, target, "poisson")[0])

    def test_zero_target_rejects_any_applied_prefix(self):
        self.assertFalse(validator.check_client_features(self.rd, 0)[0])
        write_csv(self.rd / "client/requests_000000.csv", [{"shared_prefix_applied": "1"}])
        self.assertFalse(validator.check_client_features(self.rd, 0)[0])

    def test_malformed_feature_evidence_and_wrong_mode_fail(self):
        self.assertFalse(validator.check_client_features(self.rd, 0.4, "bursty")[0])
        write_csv(self.rd / "client/requests_000000.csv", [{"status": "ok"}])
        self.assertFalse(validator.check_client_features(self.rd, 0)[0])


class Discovery(unittest.TestCase):
    def test_only_completed_production_runs_join_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, changes in (("extension_dow_screening_dow_vllm_p01_r01", {}),
                                  ("extension_new_dow_triton_p01_r01", {"interrupted_early": True}),
                                  ("extension_new_dow_dynamo_p01_r01", {"ended_at_unix": None}),
                                  ("scopecheck_val_vllm_r01", {"cell_id": "val_vllm"}),
                                  ("copy_of_completed", {"run_id": "different"})):
                (root / name).mkdir()
                (root / name / "manifest.json").write_text(json.dumps(dict({
                    "run_id": name, "cell_id": "dow_vllm_p01", "ended_at_unix": 200,
                    "interrupted_early": False}, **changes)))
            self.assertEqual([p.name for p in aging_io.discover_runs(root)],
                             ["extension_dow_screening_dow_vllm_p01_r01"])


class AttachSupervision(unittest.TestCase):
    def run_attach(self, row_provider, duration=10000):
        import attach_run as attach
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = yaml.safe_load((REPO / "campaigns/extension/cells/val_vllm.yaml").read_text())
            cell_path = root / "cell.yaml"
            cell_path.write_text(yaml.safe_dump(cell))
            config = root / "client_config.yaml"
            config.write_text("target_rate_rps: 1\n")
            fake = mock.Mock(pid=1234)
            fake.poll.return_value = None
            ticks = iter(range(0, 100000, 31))
            patches = [
                mock.patch.object(sys, "argv", ["attach", "--cell-yaml", str(cell_path),
                                  "--runs-root", str(root / "runs"), "--repo-root", str(REPO),
                                  "--duration-seconds", str(duration), "--engine-pid", "10"]),
                mock.patch.object(attach.lc, "require_free_space"),
                mock.patch.object(attach.lc, "docker_root_dir", return_value=root),
                mock.patch.object(attach.lc, "prepare_container_scope"),
                mock.patch.object(attach.lc, "gpu_sanity_check"),
                mock.patch.object(attach.lc, "host_info", return_value={}),
                mock.patch.object(attach.lc, "git_sha", return_value="test"),
                mock.patch.object(attach.lc, "spawn_monitors", return_value=fake),
                mock.patch.object(attach.lc, "spawn_client", return_value=fake),
                mock.patch.object(attach.lc, "materialize_client_config", return_value=config),
                mock.patch.object(attach.lc, "stop_subprocess"),
                mock.patch.object(attach.lc, "summarize_client_csvs", return_value={
                    "total": 1, "ok": 1, "status_counts": {"ok": 1}}),
                mock.patch.object(attach.lc, "log"),
                mock.patch.object(attach.lc, "read_latest_membership_tick", side_effect=row_provider),
                mock.patch.object(attach.reaper, "acquire_run_slot", return_value=object()),
                mock.patch.object(attach.reaper, "reap_orphans", return_value=[]),
                mock.patch.object(attach.reaper, "ledger_run_ids", return_value=[]),
                mock.patch.object(attach.reaper, "reap_host_wide", return_value=([], [])),
                mock.patch.object(attach.reaper, "record_children", return_value=[]),
                mock.patch.object(attach.reaper, "deregister_run", return_value=[]),
                mock.patch.object(attach.time, "sleep"),
                mock.patch.object(attach.time, "monotonic", side_effect=lambda: next(ticks)),
                mock.patch.object(attach.signal, "signal"),
            ]
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with self.assertRaises(SystemExit) as exit:
                    attach.main()
            manifests = list((root / "runs").glob("*/manifest.json"))
            return exit.exception.code, json.loads(manifests[0].read_text())

    def test_alive_but_incomplete_monitor_aborts_attach(self):
        def bad(*args):
            return {"ts_unix": "1", "membership_complete": "False"}
        rc, manifest = self.run_attach(bad)
        self.assertEqual(rc, 2)
        self.assertTrue(manifest["interrupted_early"])
        self.assertTrue(manifest["interruption_reason"].startswith("health:"))

    def test_alive_but_stale_monitor_aborts_attach(self):
        rc, manifest = self.run_attach(lambda *args: None)
        self.assertEqual(rc, 2)
        self.assertIn("missing/stale", manifest["interruption_reason"])

    def test_no_final_evidence_fails_even_before_warmup_expires(self):
        rc, manifest = self.run_attach(lambda *args: None, duration=1)
        self.assertEqual(rc, 2)
        self.assertIn("no final evidence", manifest["interruption_reason"])

    def test_fresh_complete_monitor_keeps_attach_successful(self):
        stamps = iter(range(1, 10000))
        def good(*args):
            return {"ts_unix": str(next(stamps)), "process_alive": "True",
                    "membership_complete": "True", "model_scope_complete": "True", "n_pids_unexpected": "0"}
        rc, manifest = self.run_attach(good, duration=400)
        self.assertEqual(rc, 0)
        self.assertFalse(manifest["interrupted_early"])


if __name__ == "__main__":
    unittest.main()
