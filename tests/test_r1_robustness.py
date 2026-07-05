"""Batch R1 robustness fixes (reliability-review triage), synthetic/off-box.

Covers the pure-function and lifecycle-method pieces of:
  R1-1  runtime health checks (container liveness, /v1/models, endpoint-dead
        window that is NOT a drop-rate threshold);
  R1-3  pre-bring-up sweep of ALL dyn_* containers + fail-hard on a surviving
        dynamo engine process;
  R1-4  confirmed kill + retain-unkillable ledger entry in the reaper.

The launch_cell wiring for R1-2/R1-4-gate/R1-5 lives in tests/test_phase_b_reaper.py.
Run: python3 -m unittest tests.test_r1_robustness
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import reaper  # noqa: E402
import launch_cell as lc  # noqa: E402


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _args(tmp):
    return types.SimpleNamespace(repo_root=REPO, component_pids=Path(tmp) / "pids.json")


def _single_cell():
    return {
        "cell_id": "e1",
        "engine": {"container_name_template": "c_r01", "gpu_device": 0},
        "monitors": {"proc": {"label": "eng"}},
        "workload": {"client_config_overrides": {"model": "M"}},
        "post_run_cooldown_s": 0,
    }


def _dynamo_cell():
    return {
        "cell_id": "d",
        "engine": {"lifecycle": "dynamo_disagg",
                   "topology": {"n_prefill": 1, "n_decode": 1, "prefill_gpu": 0, "decode_gpu": 1},
                   "readyz": {"timeout_s": 900}},
        "monitors": {"proc": {"label": "eng"}, "components": {"engine_group": "dynamo", "components": []}},
        "workload": {"client_config_overrides": {"model": "M", "base_url": "http://localhost:8400"}},
        "post_run_cooldown_s": 0,
    }


# --- R1-1: health checks ------------------------------------------------------
class ContainerRunning(unittest.TestCase):
    def test_running_true(self):
        with mock.patch.object(lc.subprocess, "run", return_value=_Result(0, "true\n")):
            self.assertTrue(lc.container_running("c"))

    def test_stopped_false(self):
        with mock.patch.object(lc.subprocess, "run", return_value=_Result(0, "false\n")):
            self.assertFalse(lc.container_running("c"))

    def test_absent_is_not_running(self):
        with mock.patch.object(lc.subprocess, "run",
                               return_value=_Result(1, "", "Error: No such object: c")):
            self.assertFalse(lc.container_running("c"))

    def test_docker_hiccup_tolerated_as_running(self):
        # Ambiguous docker error (daemon blip) must NOT false-trigger a health kill.
        with mock.patch.object(lc.subprocess, "run",
                               return_value=_Result(1, "", "Cannot connect to the Docker daemon")):
            self.assertTrue(lc.container_running("c"))
        with mock.patch.object(lc.subprocess, "run", side_effect=lc.subprocess.SubprocessError()):
            self.assertTrue(lc.container_running("c"))


class LifecycleHealthCheck(unittest.TestCase):
    def test_single_container_healthy_and_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_single_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")
            with mock.patch.object(lc, "container_running", return_value=True):
                self.assertIsNone(lf.health_check())
            with mock.patch.object(lc, "container_running", return_value=False):
                self.assertIn("not running", lf.health_check())

    def test_dynamo_containers_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")
            with mock.patch.object(lc, "container_running", return_value=False):
                self.assertIn("not running", lf.health_check())

    def test_dynamo_models_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")
            with mock.patch.object(lc, "container_running", return_value=True), \
                 mock.patch.object(lf, "_models_listed_quick", return_value=False):
                self.assertIn("/v1/models", lf.health_check())

    def test_dynamo_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")
            with mock.patch.object(lc, "container_running", return_value=True), \
                 mock.patch.object(lf, "_models_listed_quick", return_value=True):
                self.assertIsNone(lf.health_check())


class EndpointDeadWindow(unittest.TestCase):
    def _write(self, d, rows):
        # rows: list of (finished_at_unix, status)
        lines = ["submitted_at_unix,finished_at_unix,status"]
        for ts, st in rows:
            lines.append(f"{ts},{ts},{st}")
        (d / "requests_000.csv").write_text("\n".join(lines) + "\n")

    def test_all_fail_in_window_is_death(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            now = 10_000.0
            self._write(d, [(now - i, "timeout") for i in range(8)])  # 8 rows, all non-ok
            res = lc.client_all_fail_window(d, window_s=300, now_unix=now)
            self.assertEqual(res, 8)

    def test_any_ok_is_not_death_even_with_high_drop_rate(self):
        # THE stress-signal guard: 85% dropped but some ok -> NOT death.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            now = 10_000.0
            rows = [(now - i, "dropped") for i in range(17)] + [(now - 1, "ok"), (now - 2, "ok"), (now - 3, "ok")]
            self._write(d, rows)
            self.assertIsNone(lc.client_all_fail_window(d, window_s=300, now_unix=now))

    def test_too_few_rows_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            now = 10_000.0
            self._write(d, [(now - 1, "timeout"), (now - 2, "error")])  # < min_rows
            self.assertIsNone(lc.client_all_fail_window(d, window_s=300, now_unix=now))

    def test_old_rows_outside_window_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            now = 10_000.0
            # All failures but OUTSIDE the window -> nothing counted -> not death.
            self._write(d, [(now - 1000 - i, "timeout") for i in range(8)])
            self.assertIsNone(lc.client_all_fail_window(d, window_s=300, now_unix=now))


# --- R1-3: stale-container sweep + host-process gate --------------------------
class DynSweepAndGate(unittest.TestCase):
    def test_all_dyn_containers_filters_prefix(self):
        out = "dyn_frontend\ndyn_worker\nother_container\ndyn_prefill_2\n"
        with mock.patch.object(lc.subprocess, "run", return_value=_Result(0, out)):
            self.assertEqual(sorted(lc.all_dyn_containers()),
                             ["dyn_frontend", "dyn_prefill_2", "dyn_worker"])

    def test_bring_up_sweeps_all_dyn_and_fails_on_surviving_engine_proc(self):
        removed = []

        def fake_run(argv, *a, **k):
            if argv[:3] == ["docker", "rm", "-f"]:
                removed.extend(argv[3:])
            return _Result(0)

        # A stray engine process that never clears -> bring_up must die (rc 3),
        # BEFORE baselines/infra_up (so no real docker/infra work happens).
        step = {"t": 0.0}

        def fast_monotonic():
            step["t"] += 100.0
            return step["t"]

        with tempfile.TemporaryDirectory() as tmp:
            lf = lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")
            with mock.patch.object(lc, "all_dyn_containers", return_value=["dyn_worker", "dyn_prefill_2"]), \
                 mock.patch.object(lc, "dynamo_engine_procs_on_host", return_value=[12345]), \
                 mock.patch.object(lc.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(lc.time, "sleep", lambda *_: None), \
                 mock.patch.object(lc.time, "monotonic", fast_monotonic):
                with self.assertRaises(SystemExit) as cm:
                    lf.bring_up()
            self.assertEqual(cm.exception.code, 3)
            # The stale containers (any name/index) were swept before the gate.
            self.assertIn("dyn_worker", removed)
            self.assertIn("dyn_prefill_2", removed)


# --- R1-4: confirmed kill + retain-unkillable ---------------------------------
class ConfirmGone(unittest.TestCase):
    def test_nonexistent_pid_is_gone(self):
        self.assertTrue(reaper._confirm_gone(2_000_000_000, timeout_s=0.2))

    def test_own_live_pid_not_gone(self):
        import os
        self.assertFalse(reaper._confirm_gone(os.getpid(), timeout_s=0.2))

    def test_kill_pgid_nonexistent_is_confirmed_gone(self):
        self.assertTrue(reaper._kill_pgid(2_000_000_000))


class RetainUnkillable(unittest.TestCase):
    # These isolate the R1-4 kill/retain path, so the launcher is treated as DEAD
    # (record_children stamps the live test process as launcher; R2-2 would
    # otherwise correctly refuse to reap an "active" run -- covered separately).
    def _seed(self, runs_root):
        reaper.record_children(runs_root, runs_root / "r", "r", 4242, 4243)

    def test_unkillable_orphan_retains_entry_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            self._seed(runs_root)
            with mock.patch.object(reaper, "_launcher_alive", return_value=False), \
                 mock.patch.object(reaper, "_is_ours", return_value="python run_client.py --run r"), \
                 mock.patch.object(reaper, "_kill_pgid", return_value=False):
                lines = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertTrue(any("could NOT kill" in ln and "RETAINED" in ln for ln in lines), lines)
            # The entry is RETAINED (not silently lost) so the caller can gate.
            self.assertEqual(reaper.ledger_run_ids(runs_root), ["r"])

    def test_killable_orphan_clears_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            self._seed(runs_root)
            with mock.patch.object(reaper, "_launcher_alive", return_value=False), \
                 mock.patch.object(reaper, "_is_ours", return_value="python run_client.py --run r"), \
                 mock.patch.object(reaper, "_kill_pgid", return_value=True):
                lines = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertTrue(any("killed orphan" in ln for ln in lines), lines)
            self.assertEqual(reaper.ledger_run_ids(runs_root), [])  # cleared


if __name__ == "__main__":
    unittest.main()
