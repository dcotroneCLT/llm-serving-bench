"""PHASE B point 4: orphan reaper wired into launch_cell + atomic/locked ledger.

Synthetic unittests, off-box (no docker / GPU), in the style of
tests/test_phase_a_*.py. They cover the locked ledger (concurrent upserts lose no
entries, deregister is surgical, a corrupt ledger is treated as empty and logged)
and the launch_cell wiring (reap before bring-up, record after spawn, deregister
on both the clean and the abort teardown), plus an attach_run regression.

Run: python3 -m unittest tests.test_phase_b_reaper
"""
import errno
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
def _last_index(seq, value):
    return max(i for i, v in enumerate(seq) if v == value)


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
    def _run_main(self, tmp: Path, client_poll, mono_step, duration_s,
                  ledger_stuck=None, teardown_raises=False, record_raises=None,
                  disk_free_gb=None, disk_check_every=None,
                  health_reason=None, health_every=None, append_error=None,
                  finalize_error=None, heartbeat_every=None, free_inodes_val=None):
        events: list[str] = []
        logs: list[str] = []
        run_dir = tmp / "runs" / "test_e1_r01"
        real_append = lc.append_disk_usage_row  # capture before patching

        # Mock lifecycle: only the seams main() calls.
        lf = mock.MagicMock()
        lf.kind = "single_container"
        lf.bring_up.side_effect = lambda: events.append("bringup")
        lf.resolve_pid_identity.return_value = (None, None)
        lf.primary_gpu.return_value = 0
        lf.manifest_sections.return_value = {}
        lf.manifest_baseline_sections.return_value = {}
        # health_reason=None -> engine stays healthy; a string -> unhealthy every check.
        lf.health_check.return_value = health_reason
        if teardown_raises:
            def _boom():
                events.append("teardown")
                raise RuntimeError("docker rm timed out")
            lf.teardown.side_effect = _boom
        else:
            lf.teardown.side_effect = lambda: events.append("teardown")
        # finalize_manifest sits in the wrapped finalization tail immediately
        # before the manifest write; raising here injects a finalization-time
        # failure (e.g. ENOSPC on the final write) for the exit-code tests.
        def _finalize(_m):
            if finalize_error is not None:
                raise finalize_error
        lf.finalize_manifest.side_effect = _finalize

        # Client config file the clean-teardown path reads.
        cc = tmp / "client_config.yaml"
        cc.write_text("request_timeout_s: 1\n")

        reaper_mock = mock.MagicMock()
        reaper_mock.acquire_run_slot.return_value = object()  # holds the run-slot lock
        reaper_mock.reap_host_wide.return_value = ([], [])    # (lines, unkillable)
        reaper_mock.reap_orphans.side_effect = lambda *a, **k: (events.append("reap"), [])[1]
        reaper_mock.ledger_run_ids.return_value = ledger_stuck or []

        def _record(runs_root, run_dir_, run_id_, monitors_pid, client_pid):
            events.append("record")
            # record_raises: "monitors" -> raise on the client_pid=None call;
            # "client" -> raise on the upsert (client_pid set).
            if record_raises == "monitors" and client_pid is None:
                raise OSError("ledger write failed (monitors)")
            if record_raises == "client" and client_pid is not None:
                raise OSError("ledger write failed (client upsert)")
            return []
        reaper_mock.record_children.side_effect = _record
        reaper_mock.deregister_run.side_effect = lambda *a, **k: (events.append("deregister"), [])[1]

        stops: list[str] = []

        args = types.SimpleNamespace(
            cell_yaml=_single_cell_yaml(tmp), replica=1, runs_root=tmp / "runs",
            repo_root=REPO, hf_cache_host=Path(""), campaign_id="test", attempt=1,
            component_pids=tmp / "pids.json", gpu_device_override=None,
            duration_s_override=duration_s, min_free_gb=20.0,
            min_free_gb_mid_run=10.0,
            # Hardening item 4c: the mid-run inode floor. Default it comfortably
            # below the real free-inode count of the tempdir so the existing
            # healthy-path tests never trip it; the inode-floor test injects a
            # low free_inodes() value to force a breach.
            min_inodes_free_mid_run=lc.DEFAULT_MIN_INODES_FREE_MID_RUN,
            # Hardening item 1: calibration staleness gate max-age. Present so the
            # calibration path (when a --calibration-file is supplied) can read it;
            # these wiring tests pass calibration_file=None so it is not exercised.
            calibration_file=None, calibration_max_age_days=14.0,
            allow_lower_bound_calibration=False,
        )

        # A monotonic counter that steps by mono_step each call.
        state = {"t": 0.0}

        def fake_monotonic():
            state["t"] += mono_step
            return state["t"]

        exit_code = {"code": None}
        with mock.patch.object(lc.argparse.ArgumentParser, "parse_args", return_value=args), \
             mock.patch.multiple(
                 lc,
                 require_free_space=mock.DEFAULT, docker_root_dir=mock.DEFAULT,
                 load_image_pin=mock.DEFAULT, verify_image_digest=mock.DEFAULT,
                 make_lifecycle=mock.DEFAULT, host_info=mock.DEFAULT, git_sha=mock.DEFAULT,
                 spawn_monitors=mock.DEFAULT, materialize_client_config=mock.DEFAULT,
                 spawn_client=mock.DEFAULT, summarize_client_csvs=mock.DEFAULT,
                 stop_subprocess=mock.DEFAULT, reaper=reaper_mock, free_gb=mock.DEFAULT,
                 free_inodes=mock.DEFAULT, append_disk_usage_row=mock.DEFAULT,
             ) as m, \
             mock.patch.object(lc, "DISK_CHECK_EVERY_S",
                               disk_check_every if disk_check_every is not None else lc.DISK_CHECK_EVERY_S), \
             mock.patch.object(lc, "HEALTH_CHECK_EVERY_S",
                               health_every if health_every is not None else lc.HEALTH_CHECK_EVERY_S), \
             mock.patch.object(lc, "HEARTBEAT_EVERY_S",
                               heartbeat_every if heartbeat_every is not None else lc.HEARTBEAT_EVERY_S), \
             mock.patch.object(lc, "log", lambda msg: logs.append(msg)), \
             mock.patch.object(lc.time, "sleep", lambda *_: None), \
             mock.patch.object(lc.time, "monotonic", fake_monotonic), \
             mock.patch.object(lc.time, "time", lambda: 1000.0):
            m["require_free_space"].return_value = None
            m["docker_root_dir"].return_value = tmp
            # Free space seen by the mid-run watchdog + the disk_usage.csv row.
            # Default is comfortably above the floor so existing tests are healthy.
            m["free_gb"].return_value = disk_free_gb if disk_free_gb is not None else 500.0
            # Free inodes seen by the mid-run inode floor (item 4c). Default is far
            # above the floor so the byte-space and healthy-path tests are unaffected;
            # the inode-floor test injects a value below it to force a breach.
            m["free_inodes"].return_value = (
                free_inodes_val if free_inodes_val is not None else 10_000_000)
            # Trend-CSV append: raise the injected error, else use the real writer.
            if append_error is not None:
                m["append_disk_usage_row"].side_effect = append_error
            else:
                m["append_disk_usage_row"].side_effect = real_append
            m["load_image_pin"].return_value = {"image_tag": "img:tag", "digest": "sha256:d",
                                                "source_tag": "s", "pinned_at": "2026"}
            m["verify_image_digest"].return_value = None
            m["make_lifecycle"].return_value = lf
            m["host_info"].return_value = {}
            m["git_sha"].return_value = None
            m["spawn_monitors"].side_effect = lambda *a, **k: (events.append("spawn_monitors"), _FakeProc(111))[1]
            m["spawn_client"].side_effect = lambda *a, **k: (events.append("spawn_client"), _FakeProc(222, client_poll))[1]
            m["materialize_client_config"].return_value = cc
            m["summarize_client_csvs"].return_value = {"total": 1, "ok": 1, "status_counts": {}}
            m["stop_subprocess"].side_effect = lambda proc, name, **k: (stops.append(name), False)[1]
            try:
                lc.main()
            except SystemExit as e:
                exit_code["code"] = e.code
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return {"events": events, "exit_code": exit_code["code"], "run_dir": run_dir,
                "stops": stops, "manifest": manifest, "logs": logs}

    def test_clean_teardown_wiring_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            # mono_step large so the very first loop check trips the deadline
            # (clean "duration elapsed" path); client stays alive (poll None).
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1)
        events = r["events"]
        # reap BEFORE bring-up.
        self.assertLess(events.index("reap"), events.index("bringup"))
        # A record right after the monitors spawn (client_pid=None) closes the
        # crash window; the LAST record (the upsert) is after the client spawn.
        self.assertGreater(events.index("record"), events.index("spawn_monitors"))
        self.assertGreater(_last_index(events, "record"), events.index("spawn_client"))
        # deregister on the (clean) teardown, after teardown ran.
        self.assertIn("deregister", events)
        self.assertGreater(events.index("deregister"), events.index("teardown"))
        self.assertEqual(r["exit_code"], 0)

    def test_abort_path_still_deregisters(self):
        with tempfile.TemporaryDirectory() as tmp:
            # mono_step tiny + huge duration so the deadline is NOT reached; the
            # client "exited early" (poll=1) drives the interrupted/abort path.
            r = self._run_main(Path(tmp), client_poll=1, mono_step=1.0, duration_s=100000)
        events = r["events"]
        self.assertLess(events.index("reap"), events.index("bringup"))
        self.assertIn("record", events)
        self.assertIn("deregister", events)
        self.assertEqual(r["exit_code"], 2)
        # early_exit interruption class is recorded in the manifest.
        self.assertTrue(r["manifest"]["interrupted_early"])
        self.assertTrue(r["manifest"]["interruption_reason"].startswith("early_exit"))

    def test_mid_run_disk_floor_breach_exits_7_and_tears_down(self):
        # SC-2 mid-run watchdog: free space below the floor -> graceful teardown,
        # exit 7 (campaign-fatal), interruption_reason recorded, trend row written.
        # File assertions stay INSIDE the tempdir block (it is deleted on exit).
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=1.0, disk_check_every=0)
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(r["manifest"]["interrupted_early"])
            self.assertTrue(r["manifest"]["interruption_reason"].startswith("disk_floor"))
            # Graceful teardown still ran (deregister after teardown).
            self.assertIn("teardown", r["events"])
            self.assertIn("deregister", r["events"])
            # Trend row(s) written.
            csv_path = r["run_dir"] / "disk_usage.csv"
            self.assertTrue(csv_path.exists())
            rows = csv_path.read_text().splitlines()
            self.assertEqual(rows[0], "ts_unix,runs_root_free_gb,docker_root_free_gb,run_dir_size_mb")
            self.assertGreaterEqual(len(rows), 2)  # header + at least one measurement

    def test_mid_run_inode_floor_breach_exits_7_and_tears_down(self):
        # Item 4c end-to-end: byte-space is comfortably OK, but free inodes drop
        # below the floor -> the SAME campaign-fatal exit 7 / graceful teardown as
        # a byte breach, with an inode_floor interruption_reason. This closes the
        # exhaustion mode a byte-free-space floor cannot see (~8k small files/run).
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=500.0,
                               free_inodes_val=1000, disk_check_every=0)
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(r["manifest"]["interrupted_early"])
            self.assertTrue(r["manifest"]["interruption_reason"].startswith("inode_floor"))
            # Graceful teardown still ran (deregister after teardown).
            self.assertIn("teardown", r["events"])
            self.assertIn("deregister", r["events"])

    def test_healthy_disk_completes_and_logs_trend(self):
        # Healthy path unchanged: the disk check fires (csv written) but does not
        # abort; the run ends on the duration deadline with exit 0.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=3, disk_free_gb=500.0, disk_check_every=0)
            self.assertEqual(r["exit_code"], 0)
            self.assertNotIn("interruption_reason", r["manifest"])  # clean run byte-compat
            csv_path = r["run_dir"] / "disk_usage.csv"
            self.assertTrue(csv_path.exists())
            data_rows = csv_path.read_text().splitlines()[1:]
            self.assertTrue(data_rows)
            # runs_root_free_gb column reflects the measured value.
            self.assertEqual(data_rows[0].split(",")[1], "500.0")

    def test_health_failure_reason_recorded(self):
        # A different interruption class: N consecutive engine-health failures.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, health_reason="engine down",
                               health_every=0)
        self.assertEqual(r["exit_code"], 2)
        self.assertTrue(r["manifest"]["interrupted_early"])
        self.assertTrue(r["manifest"]["interruption_reason"].startswith("health:"))

    def test_disk_csv_append_failure_does_not_mask_a_breach(self):
        # FIX 1(a): the trend-CSV append is a side channel. Even if it raises,
        # a genuine floor breach must still exit 7 with the breach reason.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=1.0, disk_check_every=0,
                               append_error=OSError("boom"))
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(r["manifest"]["interrupted_early"])
            self.assertTrue(r["manifest"]["interruption_reason"].startswith("disk_floor"))
            self.assertIn("deregister", r["events"])

    def test_disk_csv_append_enospc_is_treated_as_breach(self):
        # FIX 1(b): an ENOSPC write failure IS breach evidence (runs-root full),
        # even when the free_gb probe read nominally OK -> exit 7.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=500.0, disk_check_every=0,
                               append_error=OSError(errno.ENOSPC, "No space left on device"))
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(r["manifest"]["interrupted_early"])
            self.assertIn("ENOSPC", r["manifest"]["interruption_reason"])

    def test_disk_csv_append_edquot_is_treated_as_breach(self):
        # FIX 1 (follow-up): a quota-capped filesystem fails EDQUOT while free_gb
        # still reports global space OK. EDQUOT is storage-fatal -> exit 7, and
        # the reason names the errno.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=500.0, disk_check_every=0,
                               append_error=OSError(errno.EDQUOT, "Disk quota exceeded"))
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(r["manifest"]["interrupted_early"])
            self.assertIn("EDQUOT", r["manifest"]["interruption_reason"])

    def test_disk_csv_transient_append_error_warns_once_and_continues(self):
        # FIX 1(c): a transient write error (EACCES) with the floor OK does not
        # abort the run and is logged ONCE (no per-check spam over a 48h run).
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=20, disk_free_gb=500.0, disk_check_every=0,
                               append_error=OSError(errno.EACCES, "Permission denied"))
            self.assertEqual(r["exit_code"], 0)  # ran to the duration deadline
            self.assertNotIn("interruption_reason", r["manifest"])
            warnings = [m for m in r["logs"] if "could not write disk_usage.csv" in m]
            self.assertEqual(len(warnings), 1)  # warn-once, not per-check

    def test_finalization_failure_does_not_downgrade_decided_exit_7(self):
        # FIX (this pass) (a): storage-fatal on the trend CSV decides exit 7,
        # then a storage-fatal on the final manifest write must NOT replace it
        # with an uncaught OSError -- the decided 7 survives.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=500.0, disk_check_every=0,
                               append_error=OSError(errno.ENOSPC, "No space"),
                               finalize_error=OSError(errno.ENOSPC, "No space"))
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(any("manifest finalization failed" in m for m in r["logs"]))

    def test_finalization_storage_fatal_alone_sets_exit_7(self):
        # (b): a storage-fatal on the final manifest write with NO prior breach
        # is itself a disk-floor condition -> exit 7 (not a masked error/0).
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0,
                               duration_s=1,
                               finalize_error=OSError(errno.ENOSPC, "No space"))
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(any("manifest finalization failed" in m for m in r["logs"]))

    def test_finalization_nonstorage_error_preserves_override(self):
        # (c): a non-storage exception during finalization with an override
        # already decided (disk breach -> 7) preserves it and logs the error.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=100000, disk_free_gb=1.0, disk_check_every=0,
                               finalize_error=RuntimeError("boom"))
            self.assertEqual(r["exit_code"], 7)
            self.assertTrue(any("manifest finalization failed" in m for m in r["logs"]))

    def test_heartbeat_emitted_with_expected_fields(self):
        # Heartbeat fires on its cadence and carries the progress fields; it
        # reaches this captured log the same way it reaches the campaign
        # per-attempt log (the child stdout stream), no second transport.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=5, heartbeat_every=0)
            self.assertEqual(r["exit_code"], 0)
            beats = [m for m in r["logs"] if m.startswith("progress:")]
            self.assertTrue(beats)
            line = beats[0]
            # Structural fields (duration_s here is 5 SECONDS, so the hour
            # display is 0.0h -- the numbers are exercised in test_heartbeat.py).
            for token in ("elapsed ", "h / ", "client total=", "ok=",
                          "dropped=", "runs-root free ", "GB, health "):
                self.assertIn(token, line)

    def test_heartbeat_at_most_once_on_short_run(self):
        # A short run whose whole elapsed span never reaches the cadence prints
        # no heartbeat (<= one) -- e.g. the 25-min validation cell vs the 30-min
        # cadence. Here the cadence is longer than any elapsed the loop reaches.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1.0,
                               duration_s=5, heartbeat_every=10_000_000)
            beats = [m for m in r["logs"] if m.startswith("progress:")]
            self.assertLessEqual(len(beats), 1)

    def test_reap_gate_fatal_refuses_to_start(self):
        # R1-4: an unkillable recorded orphan (ledger entry survives reap) is
        # fatal -- the run must refuse to start, BEFORE bring-up.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1,
                               ledger_stuck=["stuck_run"])
        self.assertEqual(r["exit_code"], 8)
        self.assertIn("reap", r["events"])
        self.assertNotIn("bringup", r["events"])  # died before bring-up

    def test_record_failure_after_client_stops_children_and_exits(self):
        # R2-1(a): record_children raising after the client spawn must stop the
        # client AND monitors before dying, so no live orphan is left unrecorded.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1,
                               record_raises="client")
        self.assertEqual(r["exit_code"], 8)
        self.assertIn("spawn_client", r["events"])
        # Both children were stopped on the failure path.
        self.assertIn("client", r["stops"])
        self.assertIn("monitors", r["stops"])
        # Died before the supervise/teardown finally: no teardown/deregister.
        self.assertNotIn("teardown", r["events"])
        self.assertNotIn("deregister", r["events"])

    def test_record_failure_after_monitors_stops_monitors_and_exits(self):
        # R2-1(a): record_children raising right after the monitors spawn (before
        # the client) must stop the monitors and die before spawning the client.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1,
                               record_raises="monitors")
        self.assertEqual(r["exit_code"], 8)
        self.assertNotIn("spawn_client", r["events"])  # died before client spawn
        self.assertIn("monitors", r["stops"])

    def test_interruption_reason_recorded_on_abort_absent_on_clean(self):
        # Diagnosability: interrupted runs record WHY in manifest.interruption_reason;
        # clean runs omit the key (byte-compat).
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1)
            m = json.loads((r["run_dir"] / "manifest.json").read_text())
            self.assertFalse(m["interrupted_early"])
            self.assertNotIn("interruption_reason", m)  # clean -> key absent
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=1, mono_step=1.0, duration_s=100000)
            m = json.loads((r["run_dir"] / "manifest.json").read_text())
            self.assertTrue(m["interrupted_early"])
            self.assertEqual(m.get("interruption_reason"), "early_exit: client rc=1")

    def test_teardown_error_recorded_and_still_finalizes(self):
        # R1-5: a teardown step raising must be recorded, the remaining steps
        # (deregister, manifest finalize) must still run, and exit is non-zero.
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_main(Path(tmp), client_poll=None, mono_step=1000.0, duration_s=1,
                               teardown_raises=True)
            events = r["events"]
            # deregister STILL ran despite the engine_teardown exception.
            self.assertIn("deregister", events)
            self.assertGreater(events.index("deregister"), events.index("teardown"))
            # Manifest written with the teardown_errors key, and exit non-zero.
            manifest = json.loads((r["run_dir"] / "manifest.json").read_text())
            self.assertIn("teardown_errors", manifest)
            self.assertTrue(any(te["step"] == "engine_teardown" for te in manifest["teardown_errors"]))
            # A teardown failure is explained too (clean run had no earlier reason).
            self.assertEqual(manifest.get("interruption_reason"), "teardown_failed: engine_teardown")
            self.assertEqual(r["exit_code"], 2)


class StrPathForgiving(unittest.TestCase):
    """A str runs_root/run_dir must work: today a str runs_root raised
    'unsupported operand type(s) for /: str and str' exactly when the reaper was
    needed to clean up a crashed run. A cleanup tool must be forgiving on types."""

    def test_record_children_accepts_str_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            reaper.record_children(tmp, str(Path(tmp) / "a"), "a", 1, 2)  # str args
            runs = json.loads((Path(tmp) / reaper.LEDGER_NAME).read_text())
            self.assertEqual([r["run_id"] for r in runs], ["a"])
            self.assertTrue((Path(tmp) / "a" / "child_pids.json").exists())

    def test_deregister_run_accepts_str_runs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            reaper.record_children(Path(tmp), Path(tmp) / "a", "a", 1, 2)
            reaper.deregister_run(tmp, "a")  # str runs_root
            self.assertEqual(json.loads((Path(tmp) / reaper.LEDGER_NAME).read_text()), [])

    def test_reap_orphans_accepts_str_runs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Must not raise TypeError on a str argument, even with an empty ledger.
            lines = reaper.reap_orphans(tmp, current_run_id="x")  # str runs_root
            self.assertIsInstance(lines, list)


class SighupTeardown(unittest.TestCase):
    """A dropped SSH session (SIGHUP) must trigger the graceful teardown path,
    not a bare kill that skips cleanup."""

    def test_launch_cell_registers_sighup_with_handle_signal(self):
        registered = {}

        def fake_signal(sig, handler):
            registered[sig] = handler

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(lc.signal, "signal", side_effect=fake_signal):
                LaunchCellWiring()._run_main(Path(tmp), client_poll=None,
                                             mono_step=1000.0, duration_s=1)
        import signal as _signal
        self.assertIn(_signal.SIGHUP, registered, "SIGHUP handler was not registered")
        # SIGHUP shares the SAME handler as SIGTERM/SIGINT (the graceful path).
        self.assertIs(registered[_signal.SIGHUP], registered[_signal.SIGTERM])
        self.assertIs(registered[_signal.SIGHUP], registered[_signal.SIGINT])


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
