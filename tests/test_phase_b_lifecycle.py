"""PHASE B point 2: typed Dynamo lifecycle in scripts/launch_cell.py.

Synthetic unittests that run off-box (no docker / GPU / network), same style as
tests/test_phase_a_*.py. They cover: lifecycle selection from the cell yaml,
fail-hard on a bring-up script's non-zero exit, identity-merge failure on a
component/topology mismatch, abort cleanup covering the WHOLE stack, and the
single_container manifest-section regression (shape/order unchanged).

Run: python3 -m unittest tests.test_phase_b_lifecycle
"""
import importlib.util
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent


def _load_launch_cell():
    spec = importlib.util.spec_from_file_location("launch_cell", REPO / "scripts" / "launch_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = _load_launch_cell()


class _Result:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def _single_cell():
    return {
        "cell_id": "e1",
        "engine": {
            "image_repo": "vllm/vllm-openai",
            "image_tag": "wosar2026_e1",
            "digest_pin_file": "engines/vllm_standalone/image_pin_e1.json",
            "container_name_template": "wosar2026_e1_r01",
            "gpu_device": 0,
            "shm_size": "8g",
            "readyz": {"url": "http://localhost:8100/health", "timeout_s": 600},
            "pid_strategy": {"type": "container_pid1"},
        },
        "monitors": {"proc": {"label": "vllm_v1_standalone", "period_s": 5}},
        "workload": {"client_config_overrides": {"model": "Qwen/Qwen2.5-7B-Instruct"}},
        "duration_s": 100,
        "warmup_discard_s": 10,
        "post_run_cooldown_s": 5,
    }


def _dynamo_cell(n_prefill=1, n_decode=1, prefill_gpu=0, decode_gpu=1):
    return {
        "cell_id": "val_dynamo_disagg",
        "engine": {
            "image_repo": "nvcr.io/nvidia/ai-dynamo/vllm-runtime",
            "image_tag": "1.2.0-cuda13",
            "digest_pin_file": "engines/dynamo_vllm/image_pin.json",
            "lifecycle": "dynamo_disagg",
            "topology": {
                "n_prefill": n_prefill, "n_decode": n_decode,
                "prefill_gpu": prefill_gpu, "decode_gpu": decode_gpu,
            },
            "readyz": {"url": "http://localhost:8400/health", "timeout_s": 900},
        },
        "monitors": {
            "proc": {"label": "dynamo_engine", "period_s": 5},
            "components": {
                "engine_group": "dynamo",
                "components": [
                    {"label": "dynamo_frontend", "pattern": "dynamo\\.frontend", "group": "engine"},
                    {"label": "dynamo_prefill", "pattern": "dynamo\\.vllm", "group": "engine", "expected_count": 1},
                    {"label": "dynamo_decode", "pattern": "dynamo\\.vllm", "group": "engine", "expected_count": 1},
                    {"label": "etcd", "pattern": "etcd", "group": "infra"},
                    {"label": "nats", "pattern": "nats-server", "group": "infra"},
                ],
            },
        },
        "workload": {"client_config_overrides": {
            "model": "Qwen/Qwen2.5-7B-Instruct", "base_url": "http://localhost:8400"}},
        "duration_s": 100,
        "warmup_discard_s": 10,
        "post_run_cooldown_s": 5,
    }


def _args(tmp):
    return types.SimpleNamespace(
        repo_root=REPO,
        component_pids=Path(tmp) / "dynamo_component_pids.json",
    )


class LifecycleSelection(unittest.TestCase):
    def test_default_is_single_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_single_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            self.assertIsInstance(lf, lc.SingleContainerLifecycle)
            self.assertEqual(lf.kind, "single_container")

    def test_dynamo_disagg_selected_and_topology_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(n_prefill=2, n_decode=1, prefill_gpu=0, decode_gpu=1),
                                   _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            self.assertIsInstance(lf, lc.DynamoDisaggLifecycle)
            self.assertEqual(lf.n_prefill, 2)
            self.assertEqual(lf.n_decode, 1)
            self.assertEqual(lf.gpus, [0, 1])
            self.assertEqual(lf.stack_containers(),
                             ["dyn_frontend", "dyn_prefill_1", "dyn_prefill_2", "dyn_decode_1", "dyn_etcd", "dyn_nats"])

    def test_unknown_lifecycle_dies(self):
        cell = _single_cell()
        cell["engine"]["lifecycle"] = "bogus"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                lc.make_lifecycle(cell, _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")


class DynamoFailHard(unittest.TestCase):
    def test_run_script_nonzero_exit_dies(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            with mock.patch.object(lc.subprocess, "run", return_value=_Result(returncode=1)):
                with self.assertRaises(SystemExit) as cm:
                    lf._run_script(REPO / "deploy" / "dynamo" / "infra_up.sh", {}, "infra_up")
            self.assertEqual(cm.exception.code, 2)
            # The invocation is still recorded for provenance.
            self.assertEqual(len(lf.serve_invocations), 1)

    def test_run_script_records_topology_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            env = lf._script_env()
            for k in ("N_PREFILL", "N_DECODE", "PREFILL_GPU", "DECODE_GPU", "MODEL", "WOSAR_COMPONENT_PIDS"):
                self.assertIn(k, env)
            self.assertEqual(env["PREFILL_GPU"], "0")
            self.assertEqual(env["DECODE_GPU"], "1")


class IdentityMerge(unittest.TestCase):
    def _write_identity(self, path, prefill_expected=1, decode_expected=1):
        path.write_text(json.dumps({
            "engine_group": "dynamo",
            "components": {
                "dynamo_frontend": {"containers": ["dyn_frontend"], "pgids": [10], "expected_count": 1},
                "dynamo_prefill": {"containers": ["dyn_prefill_1"], "pgids": [20], "expected_count": prefill_expected},
                "dynamo_decode": {"containers": ["dyn_decode_1"], "pgids": [30], "expected_count": decode_expected},
                "etcd": {"containers": ["dyn_etcd"], "pgids": [40], "expected_count": 1},
                "nats": {"containers": ["dyn_nats"], "pgids": [50], "expected_count": 1},
            },
        }))

    def test_merge_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            idp = Path(tmp) / "id.json"
            self._write_identity(idp)
            cell = _dynamo_cell()
            lc.merge_component_identity(cell, idp)
            comps = {c["label"]: c for c in cell["monitors"]["components"]["components"]}
            self.assertEqual(comps["dynamo_prefill"]["pgids"], [20])
            self.assertEqual(comps["dynamo_prefill"]["expected_count"], 1)

    def test_merge_expected_count_mismatch_dies(self):
        with tempfile.TemporaryDirectory() as tmp:
            idp = Path(tmp) / "id.json"
            # Cell declares expected_count=1 for prefill; identity says 2.
            self._write_identity(idp, prefill_expected=2)
            cell = _dynamo_cell()
            with self.assertRaises(SystemExit):
                lc.merge_component_identity(cell, idp)

    def test_merge_missing_component_dies(self):
        with tempfile.TemporaryDirectory() as tmp:
            idp = Path(tmp) / "id.json"
            idp.write_text(json.dumps({"engine_group": "dynamo", "components": {
                "dynamo_frontend": {"containers": ["dyn_frontend"], "pgids": [10], "expected_count": 1}}}))
            cell = _dynamo_cell()
            with self.assertRaises(SystemExit):
                lc.merge_component_identity(cell, idp)


class AbortCleanupCoversStack(unittest.TestCase):
    def test_cleanup_removes_every_container(self):
        removed = []

        def fake_run(argv, *a, **kw):
            if argv[:3] == ["docker", "rm", "-f"]:
                removed.extend(argv[3:])
            return _Result(returncode=0)

        with tempfile.TemporaryDirectory() as tmp:
            stack = ["dyn_frontend", "dyn_prefill_1", "dyn_decode_1", "dyn_etcd", "dyn_nats"]
            with mock.patch.object(lc, "save_docker_logs", lambda *a, **kw: None), \
                 mock.patch.object(lc.subprocess, "run", side_effect=fake_run):
                lc.enable_abort_cleanup(stack, Path(tmp))
                lc.cleanup_after_abort()
            lc.disable_abort_cleanup()
        self.assertEqual(sorted(removed), sorted(stack))

    def test_single_container_abort_unchanged(self):
        removed = []

        def fake_run(argv, *a, **kw):
            if argv[:3] == ["docker", "rm", "-f"]:
                removed.extend(argv[3:])
            return _Result(returncode=0)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(lc, "save_docker_logs", lambda *a, **kw: None), \
                 mock.patch.object(lc.subprocess, "run", side_effect=fake_run):
                lc.enable_abort_cleanup("wosar2026_e1_r01", Path(tmp))
                lc.cleanup_after_abort()
            lc.disable_abort_cleanup()
        self.assertEqual(removed, ["wosar2026_e1_r01"])


class ManifestShape(unittest.TestCase):
    def test_single_container_sections_order_and_no_lifecycle_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_single_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            lf.container_pid = 12345
            lf.docker_cmd = ["docker", "run", "-d"]
            lf.baseline_mib = 777
            sections = lf.manifest_sections()
            # Byte-identity guard: single_container keeps exactly container+engine,
            # in that order, with NO extra keys (e.g. no "lifecycle").
            self.assertEqual(list(sections.keys()), ["container", "engine"])
            self.assertNotIn("lifecycle", sections)
            self.assertEqual(sections["container"],
                             {"name": "wosar2026_e1_r01", "host_pid": 12345, "docker_run_cmd": ["docker", "run", "-d"]})
            self.assertEqual(lf.manifest_baseline_sections(), {"vram_baseline_mib_pre_run": 777})

    def test_single_container_finalize_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_single_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            m = {}
            lf.finalize_manifest(m)
            self.assertEqual(list(m.keys()), ["docker_log_path"])

    def test_dynamo_sections_have_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            sections = lf.manifest_sections()
            for k in ("lifecycle", "engine", "topology", "gpu_devices", "containers",
                      "components", "serve_invocations"):
                self.assertIn(k, sections)
            self.assertEqual(sections["lifecycle"], "dynamo_disagg")
            self.assertEqual(sections["gpu_devices"], [0, 1])
            self.assertEqual(lf.manifest_baseline_sections(),
                             {"vram_baselines_mib_pre_run": {}})


class DynamoTeardown(unittest.TestCase):
    def test_teardown_saves_logs_runs_downs_and_sweeps(self):
        saved, argvs = [], []

        def fake_run(argv, *a, **kw):
            argvs.append(argv)
            if argv[:3] == ["docker", "ps", "-a"]:
                return _Result(returncode=0, stdout="dyn_leftover other_container\n")
            return _Result(returncode=0, stdout="")

        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "img:tag")
            with mock.patch.object(lc, "save_docker_logs", lambda name, path, **kw: saved.append(name)), \
                 mock.patch.object(lc.subprocess, "run", side_effect=fake_run):
                lf.teardown()
            # Logs saved for every stack container BEFORE teardown.
            self.assertEqual(sorted(saved), sorted(lf.stack_containers()))
            # serve_down.sh + infra_down.sh invoked.
            scripts = [a[1] for a in argvs if a[0] == "bash"]
            self.assertTrue(any("serve_down.sh" in s for s in scripts))
            self.assertTrue(any("infra_down.sh" in s for s in scripts))
            # The leftover dyn_* container was force-removed (sweep), not other_container.
            rm = [a for a in argvs if a[:3] == ["docker", "rm", "-f"]]
            self.assertTrue(rm, "expected a force-remove sweep of dyn_* stragglers")
            self.assertIn("dyn_leftover", rm[-1])
            self.assertNotIn("other_container", rm[-1])


if __name__ == "__main__":
    unittest.main()
