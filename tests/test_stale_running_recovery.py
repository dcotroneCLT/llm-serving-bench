"""PRE-CAMPAIGN HARDENING item 3: reboot / power-loss recovery for a run left
persisted as 'running' by a hard crash.

A signal shutdown rewrites running -> interrupted, so a 'running' status at
resume can only come from a crash / power loss. For a multi-container
(dynamo_disagg) run the old run_dir_looks_active would then refuse it as
host_conflict and strand the whole campaign. The recovery archives such a run
and re-queues it -- but ONLY when every liveness signal says it is truly gone;
anything alive keeps the conservative refuse.

Synthetic, no docker/GPU. Run: python3 -m unittest tests.test_stale_running_recovery
"""
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import campaign as camp  # noqa: E402
import reaper  # noqa: E402


# --------------------------------------------------------------------------
# Pure decision table
# --------------------------------------------------------------------------


class DecisionTable(unittest.TestCase):
    def _decide(self, **over):
        facts = {
            "launcher_alive": False,
            "slot_free": True,
            "engine_container_running": False,
            "any_child_alive": False,
        }
        facts.update(over)
        return camp.stale_running_recovery_decision(**facts)

    def test_all_dead_recovers(self):
        recover, reason = self._decide()
        self.assertTrue(recover)
        self.assertEqual(reason, "stale_after_host_restart")

    def test_launcher_alive_refuses(self):
        recover, reason = self._decide(launcher_alive=True)
        self.assertFalse(recover)
        self.assertIn("launcher", reason)

    def test_slot_held_refuses(self):
        recover, reason = self._decide(slot_free=False)
        self.assertFalse(recover)
        self.assertIn("run-slot", reason)

    def test_engine_container_running_refuses(self):
        recover, reason = self._decide(engine_container_running=True)
        self.assertFalse(recover)
        self.assertIn("container", reason)

    def test_child_alive_refuses(self):
        recover, reason = self._decide(any_child_alive=True)
        self.assertFalse(recover)
        self.assertIn("child", reason)

    def test_any_single_live_signal_refuses(self):
        # Exhaustive: each signal alone flips the verdict to refuse.
        for k in ("launcher_alive", "engine_container_running", "any_child_alive"):
            self.assertFalse(self._decide(**{k: True})[0], k)
        self.assertFalse(self._decide(slot_free=False)[0])


# --------------------------------------------------------------------------
# Campaign.recover_stale_running integration (facts injected)
# --------------------------------------------------------------------------


def _campaign(tmp, spec, state):
    cfg = {
        "campaign_id": "testc",
        "mode": "serial",
        "runs_root": str(Path(tmp) / "runs"),
        "paths": {"hf_cache_host": str(Path(tmp) / "hf"), "repo_root": str(Path(tmp) / "repo")},
        "retry_policy": {"max_retries": 1},
        "inter_run_cooldown_s": 0,
        "min_free_gb": 0.0,
    }
    c = camp.Campaign(cfg, Path(tmp) / "campaign.yaml", [spec], state,
                      Path(tmp) / "state" / "s.json")
    c._skip_run_dir_prep = True
    return c


def _running_state(run_key, attempts=1):
    st = camp.State(campaign_id="testc")
    st.runs[run_key] = camp.RunStatus(status="running", attempts=attempts)
    return st


class RecoverStaleRunning(unittest.TestCase):
    def _spec(self):
        return camp.RunSpec(cell_id="a", cell_yaml="/nonexistent/a.yaml",
                            replica=1, duration_s=100)

    def test_recovers_when_all_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec()
            c = _campaign(tmp, spec, _running_state("a_r01", attempts=1))
            # A crashed run left a run_dir behind (unfinished manifest).
            run_dir = c.runs_root / "testc_a_r01"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text('{"ended_at": null}')
            c._run_slot_free = lambda: True
            c._stale_running_facts = lambda s, slot_free: {
                "launcher_alive": False, "slot_free": True,
                "engine_container_running": False, "any_child_alive": False}

            c.recover_stale_running()

            st = c.state.runs["a_r01"]
            self.assertEqual(st.status, "interrupted")
            self.assertEqual(st.last_reason, "stale_after_host_restart")
            self.assertEqual(st.attempts, 1)  # budget preserved, not reset
            self.assertFalse(run_dir.exists())  # archived aside
            archives = list(c.runs_root.glob("testc_a_r01_stale_*"))
            self.assertEqual(len(archives), 1)
            # Re-queued (interrupted is not terminal).
            self.assertIn("a_r01", [s.run_key for s in c.pending_specs()])
            # Persisted to disk.
            on_disk = json.loads(c.state_path.read_text())
            self.assertEqual(on_disk["runs"]["a_r01"]["status"], "interrupted")

    def test_refuses_when_something_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec()
            c = _campaign(tmp, spec, _running_state("a_r01"))
            run_dir = c.runs_root / "testc_a_r01"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text('{"ended_at": null}')
            c._run_slot_free = lambda: True
            c._stale_running_facts = lambda s, slot_free: {
                "launcher_alive": True, "slot_free": True,
                "engine_container_running": False, "any_child_alive": False}

            c.recover_stale_running()

            # Left 'running'; run_dir NOT archived (conservative branch).
            self.assertEqual(c.state.runs["a_r01"].status, "running")
            self.assertTrue(run_dir.exists())
            self.assertEqual(list(c.runs_root.glob("testc_a_r01_stale_*")), [])

    def test_no_running_runs_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec()
            st = camp.State(campaign_id="testc")
            st.runs["a_r01"] = camp.RunStatus(status="completed", attempts=1)
            c = _campaign(tmp, spec, st)
            # Would raise if it tried to probe the (unpatched) slot/docker.
            called = {"slot": False}
            c._run_slot_free = lambda: called.__setitem__("slot", True) or True
            c.recover_stale_running()
            self.assertFalse(called["slot"])  # never probed: nothing 'running'


# --------------------------------------------------------------------------
# reaper read-only liveness helpers
# --------------------------------------------------------------------------


class ReaperLivenessHelpers(unittest.TestCase):
    def _write_ledger(self, runs_root, entries):
        runs_root.mkdir(parents=True, exist_ok=True)
        (runs_root / reaper.LEDGER_NAME).write_text(json.dumps(entries))

    def test_launcher_alive_true_for_self(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr = Path(tmp) / "runs"
            import psutil
            me = os.getpid()
            ct = psutil.Process(me).create_time()
            self._write_ledger(rr, [{"run_id": "testc_a_r01", "run_dir": str(rr / "x"),
                                     "launcher_pid": me, "launcher_create_time": ct}])
            self.assertTrue(reaper.launcher_alive_for(rr, "testc_a_r01"))

    def test_launcher_alive_false_for_dead_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr = Path(tmp) / "runs"
            # A pid that is essentially certain not to exist, with a bogus ctime.
            self._write_ledger(rr, [{"run_id": "testc_a_r01", "run_dir": str(rr / "x"),
                                     "launcher_pid": 2 ** 31 - 1,
                                     "launcher_create_time": 1.0}])
            self.assertFalse(reaper.launcher_alive_for(rr, "testc_a_r01"))

    def test_launcher_alive_false_when_no_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr = Path(tmp) / "runs"
            self._write_ledger(rr, [])
            self.assertFalse(reaper.launcher_alive_for(rr, "testc_a_r01"))

    def test_recorded_children_alive_false_when_pids_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr = Path(tmp) / "runs"
            self._write_ledger(rr, [{"run_id": "testc_a_r01", "run_dir": str(rr / "x"),
                                     "monitors_pid": 2 ** 31 - 1, "client_pid": None}])
            self.assertFalse(reaper.recorded_children_alive(rr, "testc_a_r01"))

    def test_recorded_children_alive_true_when_ours_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr = Path(tmp) / "runs"
            me = os.getpid()
            self._write_ledger(rr, [{"run_id": "testc_a_r01", "run_dir": str(rr / "x"),
                                     "monitors_pid": me, "client_pid": None}])
            # _is_ours is the PID-reuse-safe gate; stub it to treat our pid as ours.
            orig = reaper._is_ours
            reaper._is_ours = lambda pid, run_id: "cmd" if pid == me else None
            try:
                self.assertTrue(reaper.recorded_children_alive(rr, "testc_a_r01"))
            finally:
                reaper._is_ours = orig


if __name__ == "__main__":
    unittest.main()
