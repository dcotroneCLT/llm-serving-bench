"""Review batch R2 reliability fixes, synthetic/off-box.

R2-1(b) reap recovers a run from child_pids.json when the ledger upsert was lost.
R2-2   launcher-liveness gate: reap only when the recorded launcher is gone.
R2-3   runtime membership / n_pids_unexpected / PGID-leader identity health check.
R2-4   client duration math uses monotonic anchors (NTP-immune); CSV unchanged.

The launch_cell wiring for R2-1(a) lives in tests/test_phase_b_reaper.py.
Run: python3 -m unittest tests.test_r2_robustness
"""
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "client"))

import reaper  # noqa: E402
import launch_cell as lc  # noqa: E402
from _types import RequestResult, CSV_FIELDNAMES  # noqa: E402
from benchmark import fill_derived_latencies  # noqa: E402


def _args(tmp):
    return types.SimpleNamespace(repo_root=REPO, component_pids=Path(tmp) / "pids.json")


def _dynamo_cell():
    return {
        "cell_id": "d",
        "engine": {"lifecycle": "dynamo_disagg",
                   "topology": {"n_prefill": 1, "n_decode": 1, "prefill_gpu": 0, "decode_gpu": 1},
                   "readyz": {"timeout_s": 900}},
        "monitors": {"proc": {"label": "eng"},
                     "components": {"engine_group": "dynamo", "components": []}},
        "workload": {"client_config_overrides": {"model": "M", "base_url": "http://localhost:8400"}},
        "post_run_cooldown_s": 0,
    }


# --- R2-1(b): recover from child_pids.json ------------------------------------
class ChildPidsRecovery(unittest.TestCase):
    def test_reap_recovers_run_not_in_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            run_dir = runs_root / "lost_run"
            run_dir.mkdir(parents=True)
            # child_pids.json exists (written before the lost ledger upsert), with
            # a DEAD launcher so it is reapable.
            (run_dir / "child_pids.json").write_text(json.dumps({
                "run_id": "lost_run", "run_dir": str(run_dir),
                "monitors_pid": 5551, "client_pid": 5552,
                "launcher_pid": 999999999, "launcher_create_time": 1.0,
            }))
            # Ledger is empty. The reaper must recover the run and reap its client.
            killed = []
            with mock.patch.object(reaper, "_is_ours", return_value="python run_client.py --run lost_run"), \
                 mock.patch.object(reaper, "_kill_pgid", side_effect=lambda pid: killed.append(pid) or True):
                lines = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertTrue(any("recovered run lost_run" in ln for ln in lines), lines)
            self.assertTrue(any("killed orphan" in ln for ln in lines), lines)
            self.assertTrue(killed)  # the recovered run's pids were reaped


# --- R2-2: launcher-liveness gate ---------------------------------------------
class LauncherLiveness(unittest.TestCase):
    def test_live_launcher_same_createtime_is_alive(self):
        me = os.getpid()
        ct = __import__("psutil").Process(me).create_time()
        self.assertTrue(reaper._launcher_alive({"launcher_pid": me, "launcher_create_time": ct}))

    def test_dead_launcher_pid_is_not_alive(self):
        self.assertFalse(reaper._launcher_alive(
            {"launcher_pid": 999999999, "launcher_create_time": 1.0}))

    def test_reused_pid_different_createtime_is_not_alive(self):
        me = os.getpid()
        real_ct = __import__("psutil").Process(me).create_time()
        # Same pid, but a create_time that does not match -> PID reuse, not ours.
        self.assertFalse(reaper._launcher_alive(
            {"launcher_pid": me, "launcher_create_time": real_ct - 10_000.0}))

    def test_missing_fields_treated_as_dead(self):
        self.assertFalse(reaper._launcher_alive({"run_id": "x"}))  # old-format entry

    def test_reap_refuses_to_kill_when_launcher_alive(self):
        # record_children stamps THIS live test process as the launcher, so reap
        # must NOT reap and must RETAIN the entry (caller then refuses to start).
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            reaper.record_children(runs_root, runs_root / "active", "active", 1, 2)
            killed = []
            with mock.patch.object(reaper, "_is_ours", return_value="python run_client.py --run active"), \
                 mock.patch.object(reaper, "_kill_pgid", side_effect=lambda pid: killed.append(pid) or True):
                lines = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertFalse(killed, "must not reap an active launcher's children")
            self.assertTrue(any("is ALIVE" in ln for ln in lines), lines)
            self.assertEqual(reaper.ledger_run_ids(runs_root), ["active"])  # retained -> caller refuses


# --- R2-3: runtime membership / leader health ---------------------------------
class MembershipTick(unittest.TestCase):
    def _write_agg(self, run_dir, rows):
        # rows: list of (membership_complete, n_pids_unexpected)
        lines = ["ts_unix,group,membership_complete,n_pids_unexpected"]
        for i, (mc, npu) in enumerate(rows):
            lines.append(f"{i},dynamo,{mc},{npu}")
        (run_dir / "agg_dynamo_000.csv").write_text("\n".join(lines) + "\n")

    def test_latest_complete_tick_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_agg(d, [("True", 0), ("True", 0)])
            self.assertEqual(lc.read_latest_membership_tick(d, "agg_dynamo"), (True, 0))

    def test_incomplete_and_unexpected_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_agg(d, [("True", 0), ("False", 0)])
            self.assertEqual(lc.read_latest_membership_tick(d, "agg_dynamo"), (False, 0))
            self._write_agg(d, [("True", 3)])
            self.assertEqual(lc.read_latest_membership_tick(d, "agg_dynamo"), (True, 3))

    def test_no_file_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(lc.read_latest_membership_tick(Path(tmp), "agg_dynamo"))

    def test_partial_last_row_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # A partial last line during rotation (membership_complete missing).
            (d / "agg_dynamo_000.csv").write_text(
                "ts_unix,group,membership_complete,n_pids_unexpected\n1,dynamo,True,0\n2,dynamo,")
            self.assertIsNone(lc.read_latest_membership_tick(d, "agg_dynamo"))


class DynamoHealthMembership(unittest.TestCase):
    def _lf(self, tmp):
        return lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")

    def test_membership_incomplete_is_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc, "container_running", return_value=True), \
                 mock.patch.object(lf, "_models_listed_quick", return_value=True), \
                 mock.patch.object(lc, "read_latest_membership_tick", return_value=(False, 0)):
                self.assertIn("membership incomplete", lf.health_check())

    def test_n_pids_unexpected_is_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc, "container_running", return_value=True), \
                 mock.patch.object(lf, "_models_listed_quick", return_value=True), \
                 mock.patch.object(lc, "read_latest_membership_tick", return_value=(True, 2)):
                self.assertIn("n_pids_unexpected", lf.health_check())

    def test_healthy_when_tick_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc, "container_running", return_value=True), \
                 mock.patch.object(lf, "_models_listed_quick", return_value=True), \
                 mock.patch.object(lc, "read_latest_membership_tick", return_value=(True, 0)):
                self.assertIsNone(lf.health_check())

    def test_leader_gone_is_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            lf._leader_ctimes = {999999999: 1.0}  # a pgid that does not exist
            self.assertIn("gone", lf._leader_violation())

    def test_leader_alive_same_createtime_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            me = os.getpid()
            lf._leader_ctimes = {me: __import__("psutil").Process(me).create_time()}
            self.assertIsNone(lf._leader_violation())


# --- R2-4: client latency monotonicity ----------------------------------------
class ClientLatencyMonotonic(unittest.TestCase):
    def test_ntp_backward_step_does_not_corrupt_durations(self):
        # Wall-clock jumps BACKWARDS mid-request (NTP step): first_token/finished
        # are "before" started in wall time. Monotonic anchors are sane, so the
        # durations must come from monotonic deltas (positive), not clamped-to-0.
        r = RequestResult(
            req_id=1, submitted_at_unix=1000.0, started_at_unix=1000.0,
            first_token_at_unix=990.0, finished_at_unix=980.0, status="ok",
            actual_output_tokens=10,
        )
        r.submitted_at_mono = 99.5
        r.started_at_mono = 100.0
        r.first_token_at_mono = 100.5
        r.finished_at_mono = 102.0
        fill_derived_latencies(r)
        self.assertAlmostEqual(r.queue_time_s, 0.5)
        self.assertAlmostEqual(r.ttft_s, 0.5)      # 100.5 - 100.0, NOT max(0, 990-1000)=0
        self.assertAlmostEqual(r.e2e_latency_s, 2.0)
        self.assertAlmostEqual(r.inter_token_latency_mean_s, 1.5 / 9)

    def test_falls_back_to_wallclock_without_mono_anchors(self):
        # Error/partial records without monotonic anchors still get wall-clock deltas.
        r = RequestResult(
            req_id=2, submitted_at_unix=1000.0, started_at_unix=1001.0,
            first_token_at_unix=1002.0, finished_at_unix=1004.0, status="ok",
            actual_output_tokens=5,
        )
        fill_derived_latencies(r)
        self.assertAlmostEqual(r.ttft_s, 1.0)
        self.assertAlmostEqual(r.e2e_latency_s, 3.0)

    def test_stamp_sets_both_clocks(self):
        r = RequestResult(req_id=3, submitted_at_unix=time.time(), started_at_unix=None,
                          first_token_at_unix=None, finished_at_unix=None, status="ok")
        r.stamp("started")
        self.assertIsNotNone(r.started_at_unix)
        self.assertIsNotNone(r.started_at_mono)

    def test_csv_row_excludes_monotonic_anchors(self):
        r = RequestResult(req_id=4, submitted_at_unix=1.0, started_at_unix=None,
                          first_token_at_unix=None, finished_at_unix=None, status="ok")
        r.stamp("finished")
        row = r.to_csv_row()
        self.assertEqual(set(row), set(CSV_FIELDNAMES))  # CSV schema unchanged
        for k in ("submitted_at_mono", "started_at_mono", "first_token_at_mono", "finished_at_mono"):
            self.assertNotIn(k, row)


if __name__ == "__main__":
    unittest.main()
