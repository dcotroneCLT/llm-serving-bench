"""R3: per-window calibration with scripts/calibrate_dow.py.

A window descriptor is calibrated on its own: only its cells, one fresh stack
per sweep, method v2.1 unchanged, every JSON written inside the window's
calibration_dir. Descriptors that must not be calibrated (CLOSED campaign, soak
cells without calibration, a calibration path outside the window) are refused
before anything starts. --dry-run shows the projected duration.

Run: python3 -m unittest tests.test_rerun_calibration
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "campaigns" / "extension"
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_dow as cdow  # noqa: E402
import calibrate_rate  # noqa: E402
import campaign as camp  # noqa: E402

REASON = "superseded: test"


class _Lock:
    def close(self):
        pass


def window_copy(tmp: Path, k: int) -> Path:
    """A private copy of the real window descriptor (cells symlinked), so tests
    can add state files without touching the repo's state/ dir."""
    shutil.copy(EXT / f"dow_rerun_w{k}.yaml", tmp / f"dow_rerun_w{k}.yaml")
    os.symlink(EXT / "cells", tmp / "cells")
    return tmp / f"dow_rerun_w{k}.yaml"


def orchestrator(yaml_path: Path, logf=None) -> cdow.Orchestrator:
    o = cdow.Orchestrator(
        campaign_yaml=yaml_path, runs_root=Path("/tmp"), repo_root=REPO,
        hf_cache_host=Path("/tmp"), max_age_days=30, window_s=1, cooldown_s=0,
        bringup_retries=1, recalibrate=set(), grid_map={}, env_rates=None,
        min_method_version=2, logf=logf or io.StringIO())
    o._now = lambda: 0.0
    o._install_signals = False
    o._preflight_fn = lambda: None
    return o


class DescriptorGuard(unittest.TestCase):
    def test_real_window_descriptors_are_calibratable(self):
        for k in (1, 2, 3):
            self.assertIsNone(cdow.check_descriptor(EXT / f"dow_rerun_w{k}.yaml"), k)

    def test_soak_descriptor_refused(self):
        rc, msg = cdow.check_descriptor(EXT / "dow_rerun_w1_soak.yaml")
        self.assertEqual(rc, cdow.EXIT_DESCRIPTOR)
        self.assertIn("soak_triton", msg)

    def test_closed_campaign_refused(self):
        with tempfile.TemporaryDirectory() as t:
            y = window_copy(Path(t), 1)
            state = Path(t) / "state" / "dow_rerun_w1_state.json"
            state.parent.mkdir()
            state.write_text(json.dumps(camp.State(campaign_id="extension_dow_rescoped").to_dict()))
            camp.close_campaign_state(state, "extension_dow_rescoped", REASON)
            rc, msg = cdow.check_descriptor(y)
            self.assertEqual(rc, cdow.EXIT_CAMPAIGN_CLOSED)
            self.assertEqual(rc, camp.EXIT_CAMPAIGN_CLOSED)
            self.assertIn("CLOSED", msg)

    def test_calibration_file_outside_window_dir_refused(self):
        with tempfile.TemporaryDirectory() as t:
            y = window_copy(Path(t), 2)
            text = y.read_text()
            # One W2 cell writing into W1's directory.
            text = text.replace("calibration_rerun/w2/dow_vllm_p12.json",
                                "calibration_rerun/w1/dow_vllm_p12.json", 1)
            y.write_text(text)
            rc, msg = cdow.check_descriptor(y)
            self.assertEqual(rc, cdow.EXIT_DESCRIPTOR)
            self.assertIn("dow_vllm_p12", msg)

    def test_main_refuses_before_any_side_effect(self):
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "calibrate_dow.py"),
             "--campaign-yaml", str(EXT / "dow_rerun_w1_soak.yaml"), "--dry-run"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, cdow.EXIT_DESCRIPTOR)
        self.assertIn("REFUSED", proc.stderr)


class WindowCalibrationRun(unittest.TestCase):
    def test_only_window_cells_fresh_stack_each_into_window_dir(self):
        y = EXT / "dow_rerun_w1.yaml"
        o = orchestrator(y)
        calls = {"bringup": 0, "teardown": 0, "swept": []}

        def bring_up(system, spec):
            calls["bringup"] += 1
            return cdow.Engine(None, system, "img", "sha256:x", Path("/tmp"), {})

        def sweep(spec, engine, grid, *, stack_age_s=0.0, engine_failure_retries=0):
            calls["swept"].append((spec.cell_id, spec.calibration_file))
            return {"status": "ok", "ceiling_rps": 1.0, "rate_calibrated_rps": 0.3,
                    "rc": 0, "grid": grid, "skipped": False, "stack_age_s": stack_age_s,
                    "engine_failure_retries": 0, "sweep_rows": []}

        o._acquire_slot = lambda rr: _Lock()
        o._status_fn = lambda spec: (False, "missing")
        o._bring_up_fn = bring_up
        o._sweep_fn = sweep
        o._teardown_fn = lambda engine: calls.__setitem__("teardown", calls["teardown"] + 1)
        o._health_fn = lambda engine: None
        o._dry_run_fn = lambda: (0, "")
        self.assertEqual(o.run(), cdow.EXIT_OK)

        w1_cells = [s.cell_id for s in camp.build_schedule(camp.load_campaign(y), y)]
        self.assertEqual(sorted(c for c, _ in calls["swept"]), sorted(w1_cells))
        self.assertEqual(calls["bringup"], 14)   # fresh stack per sweep
        self.assertEqual(calls["teardown"], 14)
        calib_dir = (EXT / "state" / "calibration_rerun" / "w1").resolve()
        for cid, path in calls["swept"]:
            self.assertEqual(Path(path).resolve(), calib_dir / f"{cid}.json")

    def test_dry_run_shows_projected_duration(self):
        for k, n in ((1, 14), (3, 11)):
            log = io.StringIO()
            o = orchestrator(EXT / f"dow_rerun_w{k}.yaml", logf=log)
            o._status_fn = lambda spec: (False, "missing")
            self.assertEqual(o.run_plan_only(), cdow.EXIT_OK)
            out = log.getvalue()
            hours = n * 17 / 14
            self.assertIn(f"projected duration: {hours:.1f} h for {n} cells", out)
            self.assertIn(f"calibration_rerun/w{k}/", out)
        self.assertAlmostEqual(cdow.DEFAULT_EST_SWEEP_MIN * 14 / 60, 17.0)

    def test_dry_run_counts_only_cells_still_to_do(self):
        log = io.StringIO()
        o = orchestrator(EXT / "dow_rerun_w2.yaml", logf=log)
        o._status_fn = lambda spec: (spec.cell_id.startswith("dow_vllm"), "x")
        o.run_plan_only()
        self.assertIn("projected duration: 6.1 h for 5 cells", log.getvalue())
        o.recalibrate = {"all"}
        log.truncate(0)
        o.run_plan_only()
        self.assertIn("for 14 cells", log.getvalue())

    def test_method_unchanged(self):
        self.assertEqual(calibrate_rate.CALIBRATION_METHOD_VERSION, 2.1)
        for k in (1, 2, 3):
            self.assertEqual(camp.load_campaign(EXT / f"dow_rerun_w{k}.yaml")
                             ["calibration_min_method_version"], 2)


if __name__ == "__main__":
    unittest.main()
