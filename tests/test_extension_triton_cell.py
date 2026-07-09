"""Cell-contract for the extension Triton+vLLM validation cell
(campaigns/extension/cells/val_triton.yaml) -- the third and last extension
system through the harness.

Off-box (no docker / GPU / real launch_cell subprocess), same style as
tests/test_phase_b_lifecycle.py. Covers:

  * the cell YAML parses and carries the required single_container keys;
  * make_lifecycle() resolves it to SingleContainerLifecycle (the validated,
    byte-compatible path) -- it declares no engine.lifecycle, so this asserts
    the tested default, exactly like val_vllm;
  * the SC-1 pin: digest_pin_file exists and its image_tag equals the cell's
    image_repo:image_tag (what launch_cell's image-pin gate compares);
  * the Triton engine block: readiness on /v2/health/ready (NOT /v1/models),
    the 8600-2 host ports (distinct from the other extension cells), the
    triton_child pid strategy, gpu_device 0, the model_repository mount;
  * the client block mirrors val_vllm's workload-factor structure (protocol
    triton_vllm, both new client features exercised, short validation window);
  * proc_prefix + container name are distinct from the other extension cells.

Run: python3 -m unittest tests.test_extension_triton_cell
"""
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CELLS = REPO / "campaigns" / "extension" / "cells"
TRITON_CELL = CELLS / "val_triton.yaml"


def _load_launch_cell():
    spec = importlib.util.spec_from_file_location(
        "launch_cell", REPO / "scripts" / "launch_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = _load_launch_cell()


def _load_cell(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


class CellParses(unittest.TestCase):
    def test_yaml_loads(self):
        cell = _load_cell(TRITON_CELL)
        self.assertEqual(cell["cell_id"], "val_triton")
        # Required single_container keys the historical path reads.
        eng = cell["engine"]
        for k in ("image_repo", "image_tag", "digest_pin_file",
                  "container_name_template", "command", "shm_size",
                  "gpu_device", "port_mapping", "volumes", "readyz",
                  "pid_strategy"):
            self.assertIn(k, eng, f"engine.{k} missing")
        for k in ("duration_s", "warmup_discard_s", "post_run_cooldown_s"):
            self.assertIn(k, cell)


class LifecycleIsSingleContainer(unittest.TestCase):
    def test_resolves_to_single_container(self):
        # No engine.lifecycle key -> the validated default. Assert the resolved
        # class, not a string, so this tests the real selection path.
        cell = _load_cell(TRITON_CELL)
        self.assertNotIn("lifecycle", cell["engine"])
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                repo_root=REPO,
                component_pids=Path(tmp) / "pids.json",
            )
            lf = lc.make_lifecycle(cell, args, Path(tmp), Path(tmp), {}, "img:tag")
        self.assertIsInstance(lf, lc.SingleContainerLifecycle)
        self.assertEqual(lf.kind, "single_container")


class ImagePinSC1(unittest.TestCase):
    def test_pin_file_exists_and_matches_image(self):
        cell = _load_cell(TRITON_CELL)
        eng = cell["engine"]
        pin_path = REPO / eng["digest_pin_file"]
        self.assertTrue(pin_path.exists(), f"{pin_path} missing")
        pin = json.loads(pin_path.read_text())
        image_full = f'{eng["image_repo"]}:{eng["image_tag"]}'
        # This is exactly the equality launch_cell's image-pin gate enforces.
        self.assertEqual(pin["image_tag"], image_full)

    def test_pinned_to_the_locked_extension_image(self):
        # SC-1: the whole extension holds vLLM 0.20.1; the Triton arm is the
        # 26.05 image documented in docs/extension_pin_constraint.md.
        cell = _load_cell(TRITON_CELL)
        eng = cell["engine"]
        self.assertEqual(
            f'{eng["image_repo"]}:{eng["image_tag"]}',
            "nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3",
        )
        pin = json.loads((REPO / eng["digest_pin_file"]).read_text())
        self.assertEqual(pin.get("vllm_version"), "0.20.1")


class TritonEngineBlock(unittest.TestCase):
    def setUp(self):
        self.cell = _load_cell(TRITON_CELL)
        self.eng = self.cell["engine"]

    def test_readyz_is_triton_ready_not_openai(self):
        url = self.eng["readyz"]["url"]
        self.assertTrue(url.endswith("/v2/health/ready"), url)
        self.assertNotIn("/v1/models", url)
        self.assertIn(":8600/", url)   # readiness on the HTTP port

    def test_port_layout_http_grpc_metrics(self):
        # HTTP 8000 / gRPC 8001 / metrics 8002 -> distinct host ports 8600-2.
        self.assertEqual(
            self.eng["port_mapping"],
            ["8600:8000", "8601:8001", "8602:8002"],
        )

    def test_pid_strategy_is_triton_child(self):
        ps = self.eng["pid_strategy"]
        self.assertEqual(ps["type"], "triton_child")
        # The daemon regex must name the memory-holding python child.
        self.assertIn("EngineCore", ps["process_pattern"])

    def test_gpu_device_zero(self):
        self.assertEqual(self.eng["gpu_device"], 0)

    def test_command_invokes_tritonserver_with_model_repo(self):
        cmd = self.eng["command"]
        self.assertEqual(cmd[0], "tritonserver")
        self.assertIn("--model-repository=/models", cmd)

    def test_model_repository_is_mounted_and_materialized(self):
        # The mount target the tritonserver command reads.
        self.assertTrue(
            any(v.endswith("/engines/triton_vllm/model_repository:/models")
                for v in self.eng["volumes"]),
            self.eng["volumes"],
        )
        # And the repo it mounts is actually materialized for Qwen at ctx 8192.
        model_json = json.loads(
            (REPO / "engines" / "triton_vllm" / "model_repository"
             / "qwen" / "1" / "model.json").read_text())
        self.assertEqual(model_json["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(model_json["max_model_len"], 8192)

    def test_no_vllm_use_v1_env(self):
        # 26.05/vLLM 0.20.1 is V1-only; the baseline's VLLM_USE_V1 toggle would
        # be meaningless here. Guard against it being copied in by mistake.
        self.assertNotIn("VLLM_USE_V1", self.eng.get("env", {}))


class ClientMirrorsValVllm(unittest.TestCase):
    def setUp(self):
        self.triton = _load_cell(TRITON_CELL)
        self.vllm = _load_cell(CELLS / "val_vllm.yaml")

    def test_protocol_and_model(self):
        ov = self.triton["workload"]["client_config_overrides"]
        self.assertEqual(ov["protocol"], "triton_vllm")
        self.assertEqual(ov["model"], "qwen")   # model_repository folder name
        self.assertEqual(ov["base_url"], "http://localhost:8600")

    def test_same_validation_window_as_val_vllm(self):
        for k in ("duration_s", "warmup_discard_s", "post_run_cooldown_s"):
            self.assertEqual(self.triton[k], self.vllm[k])
        self.assertEqual(self.triton["duration_s"], 1200)

    def test_both_new_client_features_exercised(self):
        ov = self.triton["workload"]["client_config_overrides"]
        # Same workload-factor structure as val_vllm: prefix reuse + bursty arrival.
        self.assertGreater(ov["prefix_repeat_fraction"], 0.0)
        self.assertGreater(ov["shared_prefix_len"], 0)
        self.assertEqual(ov["arrival_mode"], "bursty")
        for k in ("prefix_repeat_fraction", "shared_prefix_len", "arrival_mode",
                  "burst_factor", "burst_on_seconds"):
            self.assertIn(k, ov)

    def test_fixed_low_rate(self):
        ov = self.triton["workload"]["client_config_overrides"]
        self.assertEqual(ov["target_rate_rps"], 1.0)


class DistinctFromOtherExtensionCells(unittest.TestCase):
    def test_proc_prefix_and_container_and_ports_are_unique(self):
        cells = {p.stem: _load_cell(p) for p in CELLS.glob("val_*.yaml")}
        self.assertIn("val_triton", cells)

        # proc_prefix (analysis series) must be distinct across single-process cells.
        prefixes = {}
        names = {}
        for cid, cell in cells.items():
            prefixes[cid] = lc.proc_prefix_for_cell(cell)
            eng = cell.get("engine", {})
            if "container_name_template" in eng:
                names[cid] = eng["container_name_template"]
        self.assertEqual(len(set(prefixes.values())), len(prefixes),
                         f"duplicate proc_prefix: {prefixes}")
        self.assertEqual(len(set(names.values())), len(names),
                         f"duplicate container name: {names}")
        self.assertEqual(prefixes["val_triton"], "triton_vllm_020")
        self.assertEqual(names["val_triton"], "val_triton_r{replica}")

    def test_host_ports_do_not_collide_with_other_cells(self):
        triton_hosts = {pm.split(":")[0]
                        for pm in _load_cell(TRITON_CELL)["engine"]["port_mapping"]}
        # val_vllm 8500, val_dynamo_disagg 8400 -- none may overlap 8600-2.
        for p in TRITON_CELL.parent.glob("val_*.yaml"):
            if p.name == "val_triton.yaml":
                continue
            other = _load_cell(p).get("engine", {})
            other_hosts = {pm.split(":")[0]
                           for pm in other.get("port_mapping", [])}
            # val_dynamo readyz uses 8400 but declares no port_mapping; check the url too.
            self.assertEqual(triton_hosts & other_hosts, set(),
                             f"host-port collision with {p.name}")
        self.assertEqual(triton_hosts, {"8600", "8601", "8602"})


if __name__ == "__main__":
    unittest.main()
