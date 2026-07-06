"""PHASE B item 3: serial campaign orchestrator (scripts/campaign.py).

Synthetic unittests, off-box (no docker / GPU / real launch_cell subprocess),
in the style of tests/test_phase_*.py. They cover:

  * scheduler ordering (round_robin / cell_at_a_time) and the serial-only guard
    (mode must be "serial"; a `slots:` key is rejected);
  * retry semantics: one automatic re-attempt then FAILED; launch_cell exit
    7/8/9 are CAMPAIGN-FATAL, not run failures (no retry burned);
  * resume from a partial state file (completed skipped, others re-queued);
  * signal forwarding to a mock child + state persisted + non-zero exit;
  * pre-flight calibration failure raised BEFORE any run starts.

Run: python3 -m unittest tests.test_phase_b_campaign
"""
import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import campaign as camp  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_spec(cell_id, replica=1, duration_s=100, calibration_file=None,
              calibration_required=False, allow_lb=False):
    return camp.RunSpec(
        cell_id=cell_id,
        cell_yaml=f"/nonexistent/{cell_id}.yaml",
        replica=replica,
        duration_s=duration_s,
        calibration_file=calibration_file,
        calibration_required=calibration_required,
        allow_lower_bound_calibration=allow_lb,
    )


def make_campaign(tmp, schedule, state=None, **overrides):
    """Build a Campaign wired to a temp state file, with the docker/fs run-dir
    guard bypassed so the retry/resume/signal logic can be driven directly."""
    tmp = Path(tmp)
    cfg = {
        "campaign_id": "testc",
        "mode": "serial",
        "runs_root": str(tmp / "runs"),
        "paths": {"hf_cache_host": str(tmp / "hf"), "repo_root": str(tmp / "repo")},
        "retry_policy": {"max_retries": 1},
        "inter_run_cooldown_s": 0,
        "min_free_gb": 0.0,
    }
    cfg.update(overrides)
    state = state or camp.State(campaign_id="testc")
    state_path = tmp / "state" / "campaign_state.json"
    c = camp.Campaign(cfg, tmp / "campaign.yaml", schedule, state, state_path)
    c._skip_run_dir_prep = True
    return c


class ScriptedLauncher:
    """Replaces Campaign._launch_cell_rc with a scripted rc sequence, recording
    the (run_key, attempt) of every dispatch."""

    def __init__(self, rc_by_key):
        # rc_by_key: dict run_key -> list of rc to return on successive attempts
        self.rc_by_key = {k: list(v) for k, v in rc_by_key.items()}
        self.calls = []  # (run_key, attempt)

    def __call__(self, spec, attempt):
        self.calls.append((spec.run_key, attempt))
        seq = self.rc_by_key.get(spec.run_key, [0])
        return seq.pop(0) if seq else 0


# --------------------------------------------------------------------------
# Scheduler ordering + serial-only guard
# --------------------------------------------------------------------------


class SchedulerOrdering(unittest.TestCase):
    def _write_campaign(self, tmp, order, extra=""):
        tmp = Path(tmp)
        (tmp / "cells").mkdir(parents=True, exist_ok=True)
        for cid in ("a", "b"):
            (tmp / "cells" / f"{cid}.yaml").write_text(f"cell_id: {cid}\nduration_s: 100\n")
        y = tmp / "campaign.yaml"
        y.write_text(
            "campaign_id: testc\n"
            "mode: serial\n"
            "runs_root: /tmp/runs\n"
            "replicas_per_cell: 2\n"
            f"order: {order}\n"
            "paths:\n  hf_cache_host: /tmp/hf\n"
            "cells:\n"
            "  - id: a\n    yaml: cells/a.yaml\n"
            "  - id: b\n    yaml: cells/b.yaml\n"
            + extra
        )
        return y

    def test_round_robin_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = self._write_campaign(tmp, "round_robin")
            sched = camp.build_schedule(camp.load_campaign(y), y)
            self.assertEqual(
                [s.run_key for s in sched],
                ["a_r01", "b_r01", "a_r02", "b_r02"],
            )

    def test_cell_at_a_time_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = self._write_campaign(tmp, "cell_at_a_time")
            sched = camp.build_schedule(camp.load_campaign(y), y)
            self.assertEqual(
                [s.run_key for s in sched],
                ["a_r01", "a_r02", "b_r01", "b_r02"],
            )

    def test_non_serial_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "c.yaml"
            y.write_text("campaign_id: c\nmode: parallel\nruns_root: /x\ncells: []\n")
            with self.assertRaises(camp.PreflightError):
                camp.load_campaign(y)

    def test_slots_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "c.yaml"
            y.write_text(
                "campaign_id: c\nmode: serial\nruns_root: /x\ncells: []\n"
                "slots:\n  - name: gpu0\n"
            )
            with self.assertRaises(camp.PreflightError):
                camp.load_campaign(y)

    def test_duplicate_cell_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = self._write_campaign(
                tmp, "round_robin",
                extra="  - id: a\n    yaml: cells/a.yaml\n",
            )
            with self.assertRaises(camp.PreflightError):
                camp.build_schedule(camp.load_campaign(y), y)


# --------------------------------------------------------------------------
# Retry semantics
# --------------------------------------------------------------------------


class RetrySemantics(unittest.TestCase):
    def test_success_first_try(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            c._launch_cell_rc = ScriptedLauncher({"a_r01": [0]})
            self.assertEqual(c._run_with_retry(spec), "completed")
            st = c.state.runs["a_r01"]
            self.assertEqual((st.status, st.attempts, st.last_rc), ("completed", 1, 0))

    def test_retry_then_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            launcher = ScriptedLauncher({"a_r01": [1, 0]})
            c._launch_cell_rc = launcher
            self.assertEqual(c._run_with_retry(spec), "completed")
            st = c.state.runs["a_r01"]
            self.assertEqual((st.status, st.attempts), ("completed", 2))
            # --attempt incremented across the re-attempt.
            self.assertEqual(launcher.calls, [("a_r01", 1), ("a_r01", 2)])

    def test_failed_after_retry_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            c._launch_cell_rc = ScriptedLauncher({"a_r01": [1, 2]})
            self.assertEqual(c._run_with_retry(spec), "failed")
            st = c.state.runs["a_r01"]
            self.assertEqual((st.status, st.attempts), ("failed", 2))

    def test_exit_9_is_campaign_fatal_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            launcher = ScriptedLauncher({"a_r01": [camp.LC_SLOT_LOCKED, 0]})
            c._launch_cell_rc = launcher
            with self.assertRaises(camp.CampaignFatal) as ctx:
                c._run_with_retry(spec)
            self.assertEqual(ctx.exception.rc, camp.LC_SLOT_LOCKED)
            # Exactly one dispatch: no retry was burned.
            self.assertEqual(len(launcher.calls), 1)
            self.assertEqual(c.state.runs["a_r01"].status, "host_conflict")

    def test_exit_8_is_campaign_fatal_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            launcher = ScriptedLauncher({"a_r01": [camp.LC_ORPHAN_GATE, 0]})
            c._launch_cell_rc = launcher
            with self.assertRaises(camp.CampaignFatal):
                c._run_with_retry(spec)
            self.assertEqual(len(launcher.calls), 1)
            self.assertEqual(c.state.runs["a_r01"].status, "host_conflict")

    def test_run_loop_fatal_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [make_spec("a"), make_spec("b")]
            c = make_campaign(tmp, specs)
            c._launch_cell_rc = ScriptedLauncher({"a_r01": [camp.LC_SLOT_LOCKED]})
            self.assertEqual(c.run(), camp.EXIT_CAMPAIGN_FATAL)
            # b never started -- campaign stopped on the fatal.
            self.assertNotIn("b_r01", c.state.runs)


# --------------------------------------------------------------------------
# Ordering + cooldown through the run loop
# --------------------------------------------------------------------------


class RunLoopOrderingCooldown(unittest.TestCase):
    def test_serial_order_and_cooldown_between_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [make_spec("a"), make_spec("b"), make_spec("c")]
            c = make_campaign(tmp, specs, inter_run_cooldown_s=30)
            launcher = ScriptedLauncher({})  # all succeed
            c._launch_cell_rc = launcher
            sleeps = []
            c._sleep = lambda s: sleeps.append(s)
            self.assertEqual(c.run(), camp.EXIT_OK)
            self.assertEqual([k for k, _ in launcher.calls], ["a_r01", "b_r01", "c_r01"])
            # Cooldown applied after the 1st and 2nd run, not after the last.
            self.assertEqual(len(sleeps), 2 * 30)  # two 30s cooldowns, 1s ticks
            self.assertTrue(all(s == 1.0 for s in sleeps))


# --------------------------------------------------------------------------
# Resume from a partial state file
# --------------------------------------------------------------------------


class ResumePartialState(unittest.TestCase):
    def test_pending_specs_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [make_spec("a"), make_spec("b"), make_spec("c")]
            state = camp.State(campaign_id="testc")
            state.runs["a_r01"] = camp.RunStatus(status="completed", attempts=1)
            state.runs["b_r01"] = camp.RunStatus(status="failed", attempts=2)
            state.runs["c_r01"] = camp.RunStatus(status="interrupted", attempts=1)
            c = make_campaign(tmp, specs, state=state)
            self.assertEqual([s.run_key for s in c.pending_specs()], ["b_r01", "c_r01"])

    def test_resume_reruns_failed_and_interrupted_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [make_spec("a"), make_spec("b"), make_spec("c")]
            # Persist a partial state to disk, then reload it (real resume path).
            state = camp.State(campaign_id="testc")
            state.runs["a_r01"] = camp.RunStatus(status="completed", attempts=1)
            state.runs["b_r01"] = camp.RunStatus(status="failed", attempts=2)
            sp = Path(tmp) / "state" / "campaign_state.json"
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps(state.to_dict()))

            reloaded = camp.State.from_dict(json.loads(sp.read_text()))
            c = make_campaign(tmp, specs, state=reloaded)
            launcher = ScriptedLauncher({})
            c._launch_cell_rc = launcher
            self.assertEqual(c.run(), camp.EXIT_OK)
            dispatched = [k for k, _ in launcher.calls]
            self.assertNotIn("a_r01", dispatched)   # already completed
            self.assertEqual(dispatched, ["b_r01", "c_r01"])
            # b_r01 attempts continue from where it left off (cumulative provenance).
            self.assertEqual(launcher.calls[0], ("b_r01", 3))


# --------------------------------------------------------------------------
# Signal forwarding
# --------------------------------------------------------------------------


class FakeProc:
    def __init__(self):
        self.signals = []

    def send_signal(self, sig):
        self.signals.append(sig)


class SignalForwarding(unittest.TestCase):
    def test_handler_forwards_sigterm_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            c.state.runs["a_r01"] = camp.RunStatus(status="running", attempts=1)
            fake = FakeProc()
            c.current_proc = fake

            c.handle_signal(signal.SIGTERM, None)

            self.assertTrue(c._interrupted)
            self.assertEqual(fake.signals, [signal.SIGTERM])
            # In-flight run marked interrupted and state persisted to disk.
            self.assertEqual(c.state.runs["a_r01"].status, "interrupted")
            on_disk = json.loads(c.state_path.read_text())
            self.assertEqual(on_disk["runs"]["a_r01"]["status"], "interrupted")

    def test_handler_survives_no_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = make_campaign(tmp, [make_spec("a")])
            c.current_proc = None
            c.handle_signal(signal.SIGINT, None)  # must not raise
            self.assertTrue(c._interrupted)

    def test_interrupt_during_run_raises_and_marks_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])

            def racing_launch(s, attempt):
                # Simulate a SIGTERM landing during the run: the child was told
                # to tear down and exited 2.
                c._interrupted = True
                return 2

            c._launch_cell_rc = racing_launch
            with self.assertRaises(camp.CampaignInterrupted):
                c._run_with_retry(spec)
            self.assertEqual(c.state.runs["a_r01"].status, "interrupted")

    def test_run_loop_returns_interrupted_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [make_spec("a"), make_spec("b")]
            c = make_campaign(tmp, specs)

            def racing_launch(s, attempt):
                c._interrupted = True
                return 2

            c._launch_cell_rc = racing_launch
            self.assertEqual(c.run(), camp.EXIT_INTERRUPTED)
            # b never started.
            self.assertNotIn("b_r01", c.state.runs)


# --------------------------------------------------------------------------
# Pre-flight calibration
# --------------------------------------------------------------------------


class PreflightCalibration(unittest.TestCase):
    def _calib(self, tmp, name, obj):
        p = Path(tmp) / name
        p.write_text(json.dumps(obj))
        return str(p)

    def test_required_missing_fails_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a", calibration_file=str(Path(tmp) / "nope.json"),
                             calibration_required=True)
            c = make_campaign(tmp, [spec])
            with self.assertRaises(camp.PreflightError):
                c.preflight()

    def test_required_but_no_file_fails_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a", calibration_file=None, calibration_required=True)
            c = make_campaign(tmp, [spec])
            with self.assertRaises(camp.PreflightError):
                c.preflight()

    def test_invalid_json_fails_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json")
            spec = make_spec("a", calibration_file=str(bad), calibration_required=True)
            c = make_campaign(tmp, [spec])
            with self.assertRaises(camp.PreflightError):
                c.preflight()

    def test_non_ok_status_fails_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._calib(tmp, "c.json",
                            {"status": "did_not_saturate", "rate_calibrated_rps": 5.0})
            spec = make_spec("a", calibration_file=f, calibration_required=True)
            c = make_campaign(tmp, [spec])
            with self.assertRaises(camp.PreflightError):
                c.preflight()

    def test_lower_bound_override_accepts_non_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._calib(tmp, "c.json",
                            {"status": "did_not_saturate", "rate_calibrated_rps": 5.0})
            spec = make_spec("a", calibration_file=f, calibration_required=True,
                             allow_lb=True)
            c = make_campaign(tmp, [spec])
            c.preflight()  # must not raise
            ok, _ = camp.validate_calibration(spec)
            self.assertTrue(ok)

    def test_valid_ok_calibration_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._calib(tmp, "c.json",
                            {"status": "ok", "rate_calibrated_rps": 12.5})
            spec = make_spec("a", calibration_file=f, calibration_required=True)
            c = make_campaign(tmp, [spec])
            c.preflight()  # must not raise
            ok, msg = camp.validate_calibration(spec)
            self.assertTrue(ok)
            self.assertIn("12.5", msg)

    def test_not_required_missing_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a", calibration_file=None, calibration_required=False)
            c = make_campaign(tmp, [spec])
            c.preflight()  # must not raise


# --------------------------------------------------------------------------
# Real _launch_cell_rc plumbing (injected Popen, no real subprocess)
# --------------------------------------------------------------------------


class FakePopenFactory:
    def __init__(self, rc, out_lines=None):
        self.rc = rc
        self.cmd = None
        self.waited = False
        # Emulate a child whose merged stdout/stderr is a PIPE. None means "no
        # pipe" (the plumbing path where _stream_child_output is a no-op).
        self.stdout = io.StringIO("".join(out_lines)) if out_lines else None

    def __call__(self, cmd, stdout=None, stderr=None, **kwargs):
        self.cmd = cmd
        return self

    def wait(self):
        self.waited = True
        return self.rc


class LaunchCellPlumbing(unittest.TestCase):
    def test_launch_builds_cmd_and_clears_current_proc(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a", calibration_file="/x/c.json", allow_lb=True)
            c = make_campaign(tmp, [spec])
            fake = FakePopenFactory(rc=0)
            c._popen = fake
            rc = c._launch_cell_rc(spec, attempt=2)
            self.assertEqual(rc, 0)
            self.assertTrue(fake.waited)
            self.assertIsNone(c.current_proc)  # cleared after wait
            self.assertIn("--attempt", fake.cmd)
            self.assertIn("2", fake.cmd)
            self.assertIn("--calibration-file", fake.cmd)
            self.assertIn("--allow-lower-bound-calibration", fake.cmd)
            # Per-attempt capture file created under state/logs (NOT run_dir,
            # which must stay fresh for launch_cell's freshness guard), and
            # recorded on the run status for diagnostics.
            attempt_log = c.state_path.parent / "logs" / "testc_a_r01_attempt2.log"
            self.assertTrue(attempt_log.exists())
            self.assertEqual(c.state.runs["a_r01"].log_path, str(attempt_log))

    def test_child_output_streamed_and_captured_on_failure(self):
        # Regression for the child-output capture gap: a failing child's stdout
        # must reach BOTH the live terminal (sys.stdout) and the per-attempt
        # capture file, and the failure must be traceable to that file.
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            child_lines = [
                "[launch_cell] starting engine\n",
                "[launch_cell] FATAL container crashed\n",
            ]
            c._popen = FakePopenFactory(rc=5, out_lines=child_lines)

            terminal = io.StringIO()
            real_stdout = sys.stdout
            sys.stdout = terminal
            try:
                rc = c._launch_cell_rc(spec, attempt=1)
            finally:
                sys.stdout = real_stdout

            self.assertEqual(rc, 5)
            # Child lines were streamed to the live terminal in real time.
            streamed = terminal.getvalue()
            for line in child_lines:
                self.assertIn(line, streamed)
            # ...and durably captured to the per-attempt file.
            attempt_log = c.state_path.parent / "logs" / "testc_a_r01_attempt1.log"
            captured = attempt_log.read_text()
            for line in child_lines:
                self.assertIn(line, captured)
            # The failure is traceable to the capture file (campaign log ref).
            self.assertIn(str(attempt_log), streamed)


# --------------------------------------------------------------------------
# Review fixes (autonomous review, 2026-07-06)
# --------------------------------------------------------------------------


class ReviewFixes(unittest.TestCase):
    # Finding 1: prep must be lifecycle-aware. A dynamo_disagg cell declares no
    # engine.container_name_template; the old code KeyError'd before reaching
    # launch_cell, blocking the real seed campaign.
    def test_dynamo_cell_has_no_container_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp) / "dyn.yaml"
            cell.write_text(
                "cell_id: val_dynamo_disagg\n"
                "engine:\n  lifecycle: dynamo_disagg\n  topology: {n_prefill: 1}\n"
                "duration_s: 100\n"
            )
            self.assertIsNone(camp.expected_container_name(str(cell), 1))
            # A path with no manifest is not active.
            self.assertFalse(camp.run_dir_looks_active(Path(tmp), None))

    def test_prepare_run_dir_dynamo_archives_completed_stale_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cell = tmp / "dyn.yaml"
            cell.write_text(
                "cell_id: val_dynamo_disagg\n"
                "engine:\n  lifecycle: dynamo_disagg\n"
                "duration_s: 100\n"
            )
            spec = camp.RunSpec(
                cell_id="val_dynamo_disagg", cell_yaml=str(cell), replica=1,
                duration_s=100,
            )
            c = make_campaign(tmp, [spec])
            c._skip_run_dir_prep = False  # exercise the real prep path
            # A completed stale run_dir from a prior attempt must be archived,
            # not KeyError.
            run_dir = c.runs_root / "testc_val_dynamo_disagg_r01"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text('{"ended_at": "2026-07-06T00:00:00+00:00"}')
            out = c._prepare_run_dir(spec, attempt=2)
            self.assertEqual(out, run_dir)
            self.assertFalse(run_dir.exists())  # moved aside
            archives = list(c.runs_root.glob("testc_val_dynamo_disagg_r01_stale_*"))
            self.assertEqual(len(archives), 1)

    def test_prepare_run_dir_dynamo_unfinished_manifest_is_fatal_not_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cell = tmp / "dyn.yaml"
            cell.write_text(
                "cell_id: val_dynamo_disagg\n"
                "engine:\n  lifecycle: dynamo_disagg\n"
                "duration_s: 100\n"
            )
            spec = camp.RunSpec(
                cell_id="val_dynamo_disagg", cell_yaml=str(cell), replica=1,
                duration_s=100,
            )
            c = make_campaign(tmp, [spec])
            c._skip_run_dir_prep = False
            run_dir = c.runs_root / "testc_val_dynamo_disagg_r01"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text('{"ended_at": null}')

            with self.assertRaises(camp.CampaignFatal):
                c._prepare_run_dir(spec, attempt=2)
            self.assertTrue(run_dir.exists())  # must not move an active/unknown run_dir
            self.assertEqual(list(c.runs_root.glob("testc_val_dynamo_disagg_r01_stale_*")), [])

    def test_single_container_cell_keeps_container_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp) / "sc.yaml"
            cell.write_text(
                "cell_id: val_vllm\n"
                "engine:\n  container_name_template: val_vllm_r{replica}\n"
                "duration_s: 100\n"
            )
            self.assertEqual(camp.expected_container_name(str(cell), 3), "val_vllm_r03")

    # Finding 2: a present-but-unacceptable calibration_file must fail pre-flight
    # even when calibration_required is false (launch_cell would reject it).
    def test_present_invalid_calibration_not_required_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"status": "did_not_saturate",
                                       "rate_calibrated_rps": 5.0}))
            spec = make_spec("a", calibration_file=str(bad),
                             calibration_required=False)
            c = make_campaign(tmp, [spec])
            with self.assertRaises(camp.PreflightError):
                c.preflight()

    # Finding 3: launch_cell exit 7 (free-space/precondition) is campaign-fatal,
    # not a burned retry.
    def test_exit_7_is_campaign_fatal_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])
            launcher = ScriptedLauncher({"a_r01": [camp.LC_FREE_SPACE, 0]})
            c._launch_cell_rc = launcher
            with self.assertRaises(camp.CampaignFatal) as ctx:
                c._run_with_retry(spec)
            self.assertEqual(ctx.exception.rc, camp.LC_FREE_SPACE)
            self.assertEqual(len(launcher.calls), 1)   # no retry burned
            self.assertEqual(c.state.runs["a_r01"].status, "insufficient_space")

    def test_run_loop_exit_7_returns_fatal_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = make_campaign(tmp, [make_spec("a"), make_spec("b")])
            c._launch_cell_rc = ScriptedLauncher({"a_r01": [camp.LC_FREE_SPACE]})
            self.assertEqual(c.run(), camp.EXIT_CAMPAIGN_FATAL)
            self.assertNotIn("b_r01", c.state.runs)

    # Finding 4: a fatal raised BEFORE the child ran (run_dir looks active) must
    # not leave the run persisted as 'running'.
    def test_prechild_fatal_marks_host_conflict_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec("a")
            c = make_campaign(tmp, [spec])

            def prep_fatal(s, attempt):
                # Simulate _prepare_run_dir refusing (active run_dir) before rc.
                raise camp.CampaignFatal(s.run_key, camp.LC_ORPHAN_GATE, "looks active")

            c._launch_cell_rc = prep_fatal
            with self.assertRaises(camp.CampaignFatal):
                c._run_with_retry(spec)
            st = c.state.runs["a_r01"]
            self.assertEqual(st.status, "host_conflict")
            self.assertNotEqual(st.status, "running")
            # And it is persisted to disk that way.
            on_disk = json.loads(c.state_path.read_text())
            self.assertEqual(on_disk["runs"]["a_r01"]["status"], "host_conflict")


if __name__ == "__main__":
    unittest.main()
