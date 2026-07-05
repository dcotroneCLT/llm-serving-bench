"""PHASE B point 4: orphan reaper wired into launch_cell + atomic/locked ledger.

Synthetic unittests, off-box (no docker / GPU), in the style of
tests/test_phase_a_*.py. They cover the locked ledger (concurrent upserts lose no
entries, deregister is surgical, a corrupt ledger is treated as empty and logged)
and the launch_cell wiring (reap before bring-up, record after spawn, deregister
on both the clean and the abort teardown), plus an attach_run regression.

Run: python3 -m unittest tests.test_phase_b_reaper
"""
import json
import multiprocessing as mp
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


# --- multiprocessing worker (must be top-level so spawn can pickle it) --------
def _upsert_worker(runs_root: str, runs_root_dir: str, run_id: str) -> None:
    import sys as _sys
    _sys.path.insert(0, str(SCRIPTS))
    import reaper as _r
    from pathlib import Path as _P
    _r.record_children(_P(runs_root), _P(runs_root_dir) / run_id, run_id, 1234, 5678)


class LockedLedgerConcurrency(unittest.TestCase):
    def test_concurrent_upserts_lose_no_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            # Seed a baseline entry the concurrent writers must preserve.
            reaper.record_children(runs_root, runs_root / "base", "base", 1, 2)

            n = 16
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=_upsert_worker,
                                 args=(str(runs_root), str(runs_root), f"run_{i:02d}"))
                     for i in range(n)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(30)
                self.assertEqual(p.exitcode, 0, "worker crashed")

            runs = json.loads((runs_root / reaper.LEDGER_NAME).read_text())
            ids = sorted(r["run_id"] for r in runs)
            expected = sorted(["base"] + [f"run_{i:02d}" for i in range(n)])
            # If the lock failed, interleaved read-modify-write would drop entries.
            self.assertEqual(ids, expected)


class Deregister(unittest.TestCase):
    def test_deregister_removes_only_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            for rid in ("a", "b", "c"):
                reaper.record_children(runs_root, runs_root / rid, rid, 10, 20)
            reaper.deregister_run(runs_root, "b")
            runs = json.loads((runs_root / reaper.LEDGER_NAME).read_text())
            self.assertEqual(sorted(r["run_id"] for r in runs), ["a", "c"])

    def test_deregister_missing_run_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            reaper.record_children(runs_root, runs_root / "a", "a", 10, 20)
            reaper.deregister_run(runs_root, "does_not_exist")
            runs = json.loads((runs_root / reaper.LEDGER_NAME).read_text())
            self.assertEqual([r["run_id"] for r in runs], ["a"])


class CorruptLedger(unittest.TestCase):
    def test_corrupt_ledger_treated_as_empty_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            (runs_root / reaper.LEDGER_NAME).write_text("{ this is not valid json ][")
            # Upsert must not crash; it treats the ledger as empty, reports it,
            # and proceeds -- the new entry ends up as the sole valid content.
            lines = reaper.record_children(runs_root, runs_root / "a", "a", 1, 2)
            self.assertTrue(any("treating as empty" in ln for ln in lines),
                            f"expected a corrupt-ledger report, got {lines}")
            runs = json.loads((runs_root / reaper.LEDGER_NAME).read_text())
            self.assertEqual([r["run_id"] for r in runs], ["a"])

    def test_reap_on_corrupt_ledger_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            (runs_root / reaper.LEDGER_NAME).write_text("not json")
            lines = reaper.reap_orphans(runs_root, current_run_id="x")
            self.assertTrue(any("treating as empty" in ln for ln in lines), lines)
            # Ledger is left valid (empty list) afterwards.
            self.assertEqual(json.loads((runs_root / reaper.LEDGER_NAME).read_text()), [])


# --- launch_cell wiring -------------------------------------------------------
class _FakeProc:
    def __init__(self, pid, poll_val=None):
        self.pid = pid
        self._poll = poll_val
        self.returncode = poll_val

    def poll(self):
        return self._poll

    def wait(self, timeout=None):
        return 0


def _single_cell_yaml(tmp: Path) -> Path:
    cell = {
        "cell_id": "e1",
        "engine": {
            "image_repo": "img", "image_tag": "tag",
            "digest_pin_file": "engines/x/pin.json",
            "container_name_template": "c_r{replica}",
            "gpu_device": 0, "shm_size": "8g",
            "readyz": {"url": "http://x/health", "timeout_s": 60},
            "pid_strategy": {"type": "container_pid1"},
        },
        "monitors": {
            "proc": {"label": "eng", "period_s": 5},
            "gpu": {"period_s": 1}, "system": {"period_s": 5}, "rotation_s": 60,
        },
        "workload": {"client_config_overrides": {
            "protocol": "vllm_openai", "base_url": "http://x", "model": "M",
            "target_rate_rps": 1.0}},
        "duration_s": 1, "warmup_discard_s": 0, "post_run_cooldown_s": 0,
    }
    import yaml
    p = tmp / "cell.yaml"
    p.write_text(yaml.safe_dump(cell))
    return p


class LaunchCellWiring(unittest.TestCase):
    def _run_main(self, tmp: Path, client_poll, mono_step, duration_s):
        events: list[str] = []

        # Mock lifecycle: only the seams main() calls.
        lf = mock.MagicMock()
        lf.kind = "single_container"
        lf.bring_up.side_effect = lambda: events.append("bringup")
        lf.resolve_pid_identity.return_value = (None, None)
        lf.primary_gpu.return_value = 0
        lf.manifest_sections.return_value = {}
        lf.manifest_baseline_sections.return_value = {}
        lf.teardown.side_effect = lambda: events.append("teardown")
        lf.finalize_manifest.side_effect = lambda m: None

        # Client config file the clean-teardown path reads.
        cc = tmp / "client_config.yaml"
        cc.write_text("request_timeout_s: 1\n")

        reaper_mock = mock.MagicMock()
        reaper_mock.reap_orphans.side_effect = lambda *a, **k: (events.append("reap"), [])[1]
        reaper_mock.record_children.side_effect = lambda *a, **k: (events.append("record"), [])[1]
        reaper_mock.deregister_run.side_effect = lambda *a, **k: (events.append("deregister"), [])[1]

        args = types.SimpleNamespace(
            cell_yaml=_single_cell_yaml(tmp), replica=1, runs_root=tmp / "runs",
            repo_root=REPO, hf_cache_host=Path(""), campaign_id="test", attempt=1,
            component_pids=tmp / "pids.json", gpu_device_override=None,
            duration_s_override=duration_s, min_free_gb=20.0,
            calibration_file=None, allow_lower_bound_calibration=False,
        )

        # A monotonic counter that steps by mono_step each call.
        state = {"t": 0.0}

        def fake_monotonic():
            state["t"] += mono_step
            return state["t"]

        with mock.patch.object(lc.argparse.ArgumentParser, "parse_args", return_value=args), \
             mock.patch.multiple(
                 lc,
                 require_free_space=mock.DEFAULT, docker_root_dir=mock.DEFAULT,
                 load_image_pin=mock.DEFAULT, verify_image_present=mock.DEFAULT,
                 make_lifecycle=mock.DEFAULT, host_info=mock.DEFAULT, git_sha=mock.DEFAULT,
                 spawn_monitors=mock.DEFAULT, materialize_client_config=mock.DEFAULT,
                 spawn_client=mock.DEFAULT, summarize_client_csvs=mock.DEFAULT,
                 reaper=reaper_mock,
             ) as m, \
             mock.patch.object(lc.time, "sleep", lambda *_: None), \
             mock.patch.object(lc.time, "monotonic", fake_monotonic), \
             mock.patch.object(lc.time, "time", lambda: 1000.0):
            m["require_free_space"].return_value = None
            m["docker_root_dir"].return_value = tmp
            m["load_image_pin"].return_value = {"image_tag": "img:tag", "digest": "sha256:d",
                                                "source_tag": "s", "pinned_at": "2026"}
            m["verify_image_present"].return_value = None
            m["make_lifecycle"].return_value = lf
            m["host_info"].return_value = {}
            m["git_sha"].return_value = None
            m["spawn_monitors"].side_effect = lambda *a, **k: (events.append("spawn_monitors"), _FakeProc(111))[1]
            m["spawn_client"].side_effect = lambda *a, **k: (events.append("spawn_client"), _FakeProc(222, client_poll))[1]
            m["materialize_client_config"].return_value = cc
            m["summarize_client_csvs"].return_value = {"total": 1, "ok": 1, "status_counts": {}}
            with self.assertRaises(SystemExit):
                lc.main()
        return events

    def test_clean_teardown_wiring_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            # mono_step large so the very first loop check trips the deadline
            # (clean "duration elapsed" path); client stays alive (poll None).
            events = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1)
        # reap BEFORE bring-up.
        self.assertLess(events.index("reap"), events.index("bringup"))
        # record AFTER spawning both monitors and client.
        self.assertGreater(events.index("record"), events.index("spawn_monitors"))
        self.assertGreater(events.index("record"), events.index("spawn_client"))
        # deregister on the (clean) teardown, after teardown ran.
        self.assertIn("deregister", events)
        self.assertGreater(events.index("deregister"), events.index("teardown"))

    def test_abort_path_still_deregisters(self):
        with tempfile.TemporaryDirectory() as tmp:
            # mono_step tiny + huge duration so the deadline is NOT reached; the
            # client "exited early" (poll=1) drives the interrupted/abort path.
            events = self._run_main(Path(tmp), client_poll=1, mono_step=1.0, duration_s=100000)
        self.assertLess(events.index("reap"), events.index("bringup"))
        self.assertIn("record", events)
        self.assertIn("deregister", events)


class AttachRunRegression(unittest.TestCase):
    def test_attach_run_calls_deregister_at_clean_end(self):
        src = (SCRIPTS / "attach_run.py").read_text()
        # The only reaper additions to attach_run: it already reaped + recorded;
        # PHASE B #4 adds exactly the deregister at the clean end.
        self.assertIn("reaper.deregister_run(args.runs_root, run_id)", src)
        self.assertIn("reaper.reap_orphans", src)
        self.assertIn("reaper.record_children", src)


if __name__ == "__main__":
    unittest.main()
