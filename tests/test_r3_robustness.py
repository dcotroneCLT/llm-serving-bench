"""Review batch R3 reliability fixes, synthetic/off-box.

R3-1 atomic serial-run guard: run-slot flock (incl. a two-process contention test).
R3-2 host-wide fallback sweep gated on holding the slot (PID-reuse-safe + verified kill).
R3-3 membership tick freshness (wedged monitor) + strict clean-tick definition.
R3-4 client drain: cancelled tasks flush before the writer closes; no write-after-close.

Run: python3 -m unittest tests.test_r3_robustness
"""
import asyncio
import multiprocessing as mp
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
import benchmark  # noqa: E402


# --- R3-1: run-slot lock ------------------------------------------------------
def _slot_worker(runs_root: str, q, hold_s: float) -> None:
    import sys as _sys
    _sys.path.insert(0, str(SCRIPTS))
    import reaper as _r
    import time as _t
    f = _r.acquire_run_slot(runs_root)
    got = f is not None
    q.put(got)
    if got:
        _t.sleep(hold_s)  # hold so the other worker's attempt overlaps


class RunSlotLock(unittest.TestCase):
    def test_second_acquire_in_process_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = reaper.acquire_run_slot(tmp)
            self.assertIsNotNone(f1)
            self.assertIsNone(reaper.acquire_run_slot(tmp))  # already held
            f1.close()  # release
            f3 = reaper.acquire_run_slot(tmp)
            self.assertIsNotNone(f3)  # free again
            f3.close()

    def test_two_processes_contend_exactly_one_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            ps = [ctx.Process(target=_slot_worker, args=(tmp, q, 1.0)) for _ in range(2)]
            for p in ps:
                p.start()
            for p in ps:
                p.join(30)
            results = [q.get(timeout=5), q.get(timeout=5)]
            self.assertEqual(sum(1 for r in results if r), 1, f"exactly one must win: {results}")


# --- R3-2: host-wide fallback sweep -------------------------------------------
class _FakeProc:
    def __init__(self, pid, cmdline):
        self.info = {"pid": pid, "cmdline": cmdline}


class HostWideSweep(unittest.TestCase):
    def _run(self, kill_ok=True):
        procs = [
            _FakeProc(101, ["python", "run_client.py", "--output-dir", "/runs/x/run1/client"]),   # orphan
            _FakeProc(102, ["python", "run_monitors.py", "--runs-root", "/runs/x", "--run-id", "cur"]),  # current run
            _FakeProc(103, ["python", "run_client.py", "--output-dir", "/other/client"]),          # other runs-root
            _FakeProc(104, ["bash", "unrelated"]),                                                 # not OUR_SCRIPTS
        ]
        killed = []
        with mock.patch.object(reaper.psutil, "process_iter", return_value=procs), \
             mock.patch.object(reaper, "_kill_pgid",
                               side_effect=lambda pid: (killed.append(pid), kill_ok)[1]):
            lines, unkillable = reaper.reap_host_wide("/runs/x", current_run_id="cur")
        return killed, unkillable, lines

    def test_kills_only_matching_orphans(self):
        killed, unkillable, _ = self._run(kill_ok=True)
        self.assertEqual(killed, [101])          # 102 (current), 103 (other root), 104 (not ours) skipped
        self.assertEqual(unkillable, [])

    def test_unkillable_is_reported_for_gate(self):
        killed, unkillable, lines = self._run(kill_ok=False)
        self.assertEqual(killed, [101])
        self.assertEqual(unkillable, [101])      # caller must refuse to start
        self.assertTrue(any("could NOT kill" in ln for ln in lines), lines)


# --- R3-5: stale child_pids.json cleanup --------------------------------------
class StaleChildPidsCleanup(unittest.TestCase):
    def _seed_recovered(self, runs_root, run_id="lost"):
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        cp = run_dir / "child_pids.json"
        cp.write_text('{"run_id": "%s", "run_dir": "%s", "monitors_pid": 4242, '
                      '"client_pid": 4243, "launcher_pid": 999999999, '
                      '"launcher_create_time": 1.0}' % (run_id, run_dir))
        return cp

    def test_recovered_dead_file_cleaned_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            cp = self._seed_recovered(runs_root)
            # No live orphans (pids not ours) -> clean dead -> file removed.
            with mock.patch.object(reaper, "_is_ours", return_value=None):
                lines = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertTrue(any("removed stale child_pids.json" in ln for ln in lines), lines)
            self.assertFalse(cp.exists(), "stale child_pids.json should be removed")
            # A second reap does not re-recover / re-log it (file is gone).
            lines2 = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertFalse(any("recovered run lost" in ln for ln in lines2), lines2)

    def test_unkillable_recovered_file_retained_as_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            cp = self._seed_recovered(runs_root)
            # A live orphan we cannot kill -> keep the file (evidence) + retain ledger.
            with mock.patch.object(reaper, "_is_ours", return_value="python run_client.py --run lost"), \
                 mock.patch.object(reaper, "_kill_pgid", return_value=False):
                lines = reaper.reap_orphans(runs_root, current_run_id="new")
            self.assertTrue(any("could NOT kill" in ln for ln in lines), lines)
            self.assertTrue(cp.exists(), "unkillable-case child_pids.json must be kept")
            self.assertEqual(reaper.ledger_run_ids(runs_root), ["lost"])


# --- R3-3: membership freshness + strictness ----------------------------------
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


class MembershipStrictness(unittest.TestCase):
    def _lf(self, tmp):
        return lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")

    def _check(self, row):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc, "read_latest_membership_tick", return_value=row):
                return lf._membership_violation()

    def test_clean_tick_passes(self):
        self.assertIsNone(self._check(
            {"ts_unix": "1", "membership_complete": "True", "n_pids_unexpected": "0"}))

    def test_not_affirmatively_complete_is_violation(self):
        self.assertIn("not affirmatively complete", self._check(
            {"ts_unix": "1", "membership_complete": "False", "n_pids_unexpected": "0"}))

    def test_missing_n_pids_unexpected_is_violation(self):
        # Old behavior coerced missing -> 0 (clean). Now it is a violation.
        self.assertIn("missing", self._check(
            {"ts_unix": "1", "membership_complete": "True"}))

    def test_unparseable_n_pids_unexpected_is_violation(self):
        self.assertIn("unparseable", self._check(
            {"ts_unix": "1", "membership_complete": "True", "n_pids_unexpected": "oops"}))

    def test_positive_n_pids_unexpected_is_violation(self):
        self.assertIn("n_pids_unexpected=3", self._check(
            {"ts_unix": "1", "membership_complete": "True", "n_pids_unexpected": "3"}))


class MembershipFreshness(unittest.TestCase):
    def _lf(self, tmp):
        return lc.make_lifecycle(_dynamo_cell(), _args(tmp), Path(tmp), Path(tmp), {}, "i:t")

    def test_stale_tick_is_violation(self):
        # Same tick (ts_unix unchanged) while the monotonic clock advances past the
        # staleness bound => wedged monitor.
        clock = {"t": 0.0}
        row = {"ts_unix": "100", "membership_complete": "True", "n_pids_unexpected": "0"}
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc.time, "monotonic", lambda: clock["t"]), \
                 mock.patch.object(lc, "read_latest_membership_tick", return_value=row):
                self.assertIsNone(lf._membership_violation())        # t=0, fresh & clean
                clock["t"] = lc.MEMBERSHIP_STALE_S + 10.0
                self.assertIn("no new tick", lf._membership_violation())  # ts never advanced

    def test_advancing_tick_stays_fresh(self):
        clock = {"t": 0.0}
        rows = [
            {"ts_unix": "100", "membership_complete": "True", "n_pids_unexpected": "0"},
            {"ts_unix": "200", "membership_complete": "True", "n_pids_unexpected": "0"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc.time, "monotonic", lambda: clock["t"]), \
                 mock.patch.object(lc, "read_latest_membership_tick", side_effect=rows):
                self.assertIsNone(lf._membership_violation())        # t=0, ts=100
                clock["t"] = lc.MEMBERSHIP_STALE_S + 10.0
                self.assertIsNone(lf._membership_violation())        # ts advanced to 200 -> fresh

    def test_no_tick_yet_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf = self._lf(tmp)
            with mock.patch.object(lc, "read_latest_membership_tick", return_value=None):
                self.assertIsNone(lf._membership_violation())  # monitor warming up


# --- R3-4: client drain flush -------------------------------------------------
class ClientDrain(unittest.TestCase):
    def test_drain_awaits_cancelled_tasks_before_proceeding(self):
        flushed = []

        async def _run():
            async def worker():
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    flushed.append("final-row")  # the except-block flush in _dispatch_one
                    raise
            tasks = {asyncio.create_task(worker())}
            await benchmark.drain_and_grace(tasks, drain_s=0.05, cancel_grace_s=1.0)

        asyncio.run(_run())
        # The cancelled task's final row was flushed BEFORE drain_and_grace returned
        # (i.e. before the caller closes the writer).
        self.assertEqual(flushed, ["final-row"])

    def test_writer_refuses_write_after_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = benchmark.CsvRotatingWriter(Path(tmp), "requests", 60, ["req_id"])
            w.write({"req_id": 1})
            w.close()
            with self.assertRaises(RuntimeError):
                w.write({"req_id": 2})   # no silent reopen -> fail loud
            w.close()                    # idempotent, no error

    def test_writer_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = benchmark.CsvRotatingWriter(Path(tmp), "requests", 60, ["req_id"])
            w.write({"req_id": 1})
            w.close()
            w.close()
            self.assertTrue(w._closed)


if __name__ == "__main__":
    unittest.main()
