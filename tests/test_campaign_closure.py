"""R1: closing the original DoW screening campaign (scripts/campaign.py --close).

A superseded campaign must be impossible to start or resume by accident, its
evidence must survive byte-identical, and no operator-facing text (ALERT.json
next command, pre-flight banner, longtest_status.sh) may suggest resuming it.

Run: python3 -m unittest tests.test_campaign_closure
"""
import contextlib
import hashlib
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
sys.path.insert(0, str(REPO / "scripts"))

import campaign as camp  # noqa: E402

REASON = ("superseded: incomplete Triton/vLLM process scope, rerun under "
          "extension_dow_rescoped")


def _write_campaign(tmp: Path, campaign_id="testc") -> Path:
    (tmp / "cells").mkdir(parents=True, exist_ok=True)
    (tmp / "cells" / "a.yaml").write_text("cell_id: a\nduration_s: 100\n")
    y = tmp / "campaign.yaml"
    y.write_text(
        f"campaign_id: {campaign_id}\n"
        "mode: serial\n"
        f"runs_root: {tmp / 'runs'}\n"
        "replicas_per_cell: 1\n"
        "min_free_gb: 0.0\n"
        "state_file: state/campaign_state.json\n"
        "paths:\n"
        f"  hf_cache_host: {tmp / 'hf'}\n"
        f"  repo_root: {tmp / 'repo'}\n"
        "cells:\n"
        "  - id: a\n    yaml: cells/a.yaml\n")
    return y


def _write_state(path: Path, campaign_id="testc", status="interrupted") -> bytes:
    """A realistic pre-close state (44/57-like: completed + interrupted, baseline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    st = camp.State(campaign_id=campaign_id, started_at="2026-08-30T23:17:00+00:00")
    st.runs["a_r01"] = camp.RunStatus(status=status, attempts=1, last_rc=2)
    st.runs["z_r01"] = camp.RunStatus(status="completed", attempts=1, last_rc=0)
    st.baseline = {"kernel": "7.0.0-30", "driver_version": "580.159.03", "image_digests": {}}
    raw = json.dumps(st.to_dict(), indent=2).encode()
    path.write_bytes(raw)
    return raw


def _quiet_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = camp.main(argv)
    return rc, out.getvalue() + err.getvalue()


class _Stop(Exception):
    """Ends a --start in the test right after its closure gate."""


class ConcurrentClose(unittest.TestCase):
    """--close against a concurrent --start/--resume (reviewer repro: close
    writes CLOSED while --start sits between its closure check and its delete)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.y = _write_campaign(self.tmp)
        self.state = self.tmp / "state" / "campaign_state.json"
        self.raw = _write_state(self.state)

    def tearDown(self):
        self._tmp.cleanup()

    def test_close_inside_the_start_window_is_refused(self):
        attempts = []
        real_closure = camp.state_closure

        def closure_then_concurrent_close(path):
            verdict = real_closure(path)
            # The window between --start's closure check and its state delete.
            try:
                camp.close_campaign_state(path, "testc", REASON)
                attempts.append("closed")
            except camp.StateLocked as e:
                attempts.append(str(e))
            raise _Stop()

        from unittest import mock
        with mock.patch.object(camp, "state_closure", side_effect=closure_then_concurrent_close):
            with self.assertRaises(_Stop):
                _quiet_main(["--campaign-yaml", str(self.y), "--start"])
        self.assertEqual(len(attempts), 1)
        self.assertIn("held by another campaign.py process", attempts[0])
        self.assertIn(f"pid={os.getpid()}", attempts[0])
        self.assertEqual(self.state.read_bytes(), self.raw)          # never CLOSED
        self.assertFalse((self.state.parent / "closed").exists())
        # main() released the lock on the way out: a later close succeeds.
        rc, _ = _quiet_main(["--campaign-yaml", str(self.y), "--close", "--reason", REASON])
        self.assertEqual(rc, camp.EXIT_OK)

    def test_start_and_resume_refused_while_lock_is_held(self):
        fd = camp.acquire_state_lock(self.state)      # a close (or orchestrator) in progress
        try:
            for flag in ("--start", "--resume"):
                rc, out = _quiet_main(["--campaign-yaml", str(self.y), flag])
                self.assertEqual(rc, camp.EXIT_PREFLIGHT, flag)
                self.assertIn("REFUSED: state lock", out)
                self.assertEqual(self.state.read_bytes(), self.raw, flag)   # not deleted
        finally:
            camp.release_state_lock(fd)

    def test_close_from_another_process_refused_while_campaign_runs(self):
        fd = camp.acquire_state_lock(self.state)      # the live orchestrator
        try:
            res = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "campaign.py"), "--campaign-yaml",
                 str(self.y), "--close", "--reason", REASON],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(res.returncode, camp.EXIT_PREFLIGHT, res.stderr)
            self.assertIn("CLOSE REFUSED: state lock", res.stderr)
            self.assertEqual(self.state.read_bytes(), self.raw)
        finally:
            camp.release_state_lock(fd)
        res = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "campaign.py"), "--campaign-yaml",
             str(self.y), "--close", "--reason", REASON],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, camp.EXIT_OK, res.stderr)
        self.assertEqual(json.loads(self.state.read_text())["state"], "CLOSED")

    def test_closed_state_refused_under_lock_and_dry_run_needs_no_lock(self):
        camp.close_campaign_state(self.state, "testc", REASON)
        rc, _ = _quiet_main(["--campaign-yaml", str(self.y), "--start"])
        self.assertEqual(rc, camp.EXIT_CAMPAIGN_CLOSED)
        fd = camp.acquire_state_lock(self.state)
        try:
            rc, out = _quiet_main(["--campaign-yaml", str(self.y), "--dry-run"])
            self.assertEqual(rc, camp.EXIT_CAMPAIGN_CLOSED, out)
        finally:
            camp.release_state_lock(fd)


class CloseOperation(unittest.TestCase):
    def test_close_marks_state_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            y = _write_campaign(tmp)
            state = tmp / "state" / "campaign_state.json"
            raw = _write_state(state)
            alert = state.parent / "ALERT.json"
            alert.write_text(json.dumps({"campaign_id": "testc", "event": "interrupted",
                                         "next_command": f"campaign.py --campaign-yaml {y} --resume"}))
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_OK)

            d = json.loads(state.read_text())
            self.assertEqual(d["state"], "CLOSED")
            self.assertEqual(d["closed_reason"], REASON)
            self.assertTrue(d["closed_at"])
            # Every pre-existing key is untouched.
            before = json.loads(raw)
            for key in ("campaign_id", "started_at", "runs", "baseline"):
                self.assertEqual(d[key], before[key], key)

            evidence = state.parent / d["closed_evidence_dir"]
            self.assertEqual((evidence / state.name).read_bytes(), raw)
            record = json.loads((evidence / "CLOSED.json").read_text())
            self.assertEqual(record["state_sha256_before_close"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record["reason"], REASON)
            # The campaign's own ALERT (which suggests --resume) left the active path.
            self.assertFalse(alert.exists())
            self.assertIn("--resume", (evidence / "ALERT.json").read_text())

    def test_close_leaves_another_campaigns_alert_in_place(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            y = _write_campaign(tmp)
            state = tmp / "state" / "campaign_state.json"
            _write_state(state)
            alert = state.parent / "ALERT.json"
            alert.write_text(json.dumps({"campaign_id": "other", "event": "interrupted"}))
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_OK)
            self.assertTrue(alert.exists())

    def test_close_refusals(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            y = _write_campaign(tmp)
            state = tmp / "state" / "campaign_state.json"
            # No state file: nothing to close.
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            # Missing / empty reason.
            _write_state(state)
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close"])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", "  "])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            # A run still marked running.
            raw = _write_state(state, status="running")
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            self.assertEqual(state.read_bytes(), raw)
            # State of another campaign.
            raw = _write_state(state, campaign_id="OTHER")
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            self.assertEqual(state.read_bytes(), raw)
            # --close cannot be combined with a run mode; --reason needs --close.
            for extra in (["--start"], ["--resume"], ["--dry-run"]):
                rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close",
                                     "--reason", REASON] + extra)
                self.assertEqual(rc, camp.EXIT_USAGE, extra)
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--dry-run", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_USAGE)

    def test_close_is_idempotent_for_same_reason_and_refuses_another(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            y = _write_campaign(tmp)
            state = tmp / "state" / "campaign_state.json"
            _write_state(state)
            self.assertEqual(_quiet_main(["--campaign-yaml", str(y), "--close",
                                          "--reason", REASON])[0], camp.EXIT_OK)
            closed = state.read_bytes()
            self.assertEqual(_quiet_main(["--campaign-yaml", str(y), "--close",
                                          "--reason", REASON])[0], camp.EXIT_OK)
            self.assertEqual(state.read_bytes(), closed)
            self.assertEqual(len(list((state.parent / "closed").iterdir())), 1)
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--close", "--reason", "other"])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            self.assertEqual(state.read_bytes(), closed)


class ClosedCampaignRefusesToRun(unittest.TestCase):
    def _closed(self, tmp: Path):
        y = _write_campaign(tmp)
        state = tmp / "state" / "campaign_state.json"
        _write_state(state)
        camp.close_campaign_state(state, "testc", REASON)
        return y, state

    def test_start_resume_dry_run_refused_with_distinct_exit_code(self):
        self.assertNotIn(camp.EXIT_CAMPAIGN_CLOSED,
                         {camp.EXIT_OK, camp.EXIT_USAGE, camp.EXIT_PREFLIGHT,
                          camp.EXIT_INTERRUPTED, camp.EXIT_CAMPAIGN_FATAL,
                          camp.EXIT_COMPLETED_WITH_FAILURES}
                         | set(camp.FATAL_CODES))
        for mode in (["--start"], ["--resume"], ["--resume", "--rerun-failed"],
                     ["--dry-run"], ["--start", "--dry-run"]):
            with tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                y, state = self._closed(tmp)
                before = state.read_bytes()
                rc, out = _quiet_main(["--campaign-yaml", str(y)] + mode)
                self.assertEqual(rc, camp.EXIT_CAMPAIGN_CLOSED, mode)
                self.assertIn("CLOSED", out)
                self.assertIn(REASON, out)
                self.assertNotIn("--resume", out.replace(" ".join(mode), ""))
                # Nothing was wiped, rewritten or logged.
                self.assertEqual(state.read_bytes(), before, mode)
                self.assertFalse((state.parent / "logs").exists(), mode)
                self.assertFalse((tmp / "runs").exists(), mode)

    def test_corrupt_state_is_not_treated_as_open(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            y = _write_campaign(tmp)
            state = tmp / "state" / "campaign_state.json"
            state.parent.mkdir(parents=True)
            state.write_text("{not json")
            rc, _ = _quiet_main(["--campaign-yaml", str(y), "--start"])
            self.assertEqual(rc, camp.EXIT_PREFLIGHT)
            self.assertEqual(state.read_text(), "{not json")

    def test_next_command_never_suggests_resume_for_closed_state(self):
        st = camp.State(campaign_id="testc", state=camp.STATE_CLOSED, closed_reason=REASON)
        with tempfile.TemporaryDirectory() as t:
            c = camp.Campaign({"campaign_id": "testc", "mode": "serial",
                               "runs_root": str(Path(t) / "runs"),
                               "paths": {"hf_cache_host": t}},
                              Path(t) / "c.yaml", [], st, Path(t) / "state" / "s.json")
            for event in ("campaign_fatal", "failed_after_retry", "interrupted",
                          "completed_with_failures"):
                text = c._next_command(event)
                self.assertNotIn("--resume", text, event)
                self.assertIn("CLOSED", text)

    def test_closed_state_round_trips_through_state_dataclass(self):
        st = camp.State(campaign_id="x", state=camp.STATE_CLOSED, closed_reason=REASON)
        again = camp.State.from_dict(st.to_dict())
        self.assertTrue(again.closed)
        self.assertEqual(again.closed_reason, REASON)
        # An open state keeps its historical layout (no new keys).
        self.assertNotIn("state", camp.State(campaign_id="x").to_dict())


class RealDescriptorClosure(unittest.TestCase):
    """The actual 57-run descriptor: once its state is CLOSED it cannot run."""

    def test_dow_campaign_closed_state_refuses_start(self):
        real = REPO / "campaigns" / "extension" / "dow_campaign.yaml"
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            shutil.copy(real, tmp / "dow_campaign.yaml")
            os.symlink(REPO / "campaigns" / "extension" / "cells", tmp / "cells")
            state = tmp / "state" / "dow_campaign_state.json"
            _write_state(state, campaign_id="extension_dow_screening")
            rc, _ = _quiet_main(["--campaign-yaml", str(tmp / "dow_campaign.yaml"),
                                 "--close", "--reason", REASON])
            self.assertEqual(rc, camp.EXIT_OK)
            before = state.read_bytes()
            for mode in ("--start", "--resume", "--dry-run"):
                rc, _ = _quiet_main(["--campaign-yaml", str(tmp / "dow_campaign.yaml"), mode])
                self.assertEqual(rc, camp.EXIT_CAMPAIGN_CLOSED, mode)
            self.assertEqual(state.read_bytes(), before)


class AlertTextNeverSuggestsResumingAnotherCampaign(unittest.TestCase):
    def test_preflight_hides_foreign_alert_next_command(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            y = _write_campaign(tmp, campaign_id="newc")
            alert = tmp / "state" / "ALERT.json"
            alert.parent.mkdir(parents=True)
            alert.write_text(json.dumps({
                "campaign_id": "oldc", "event": "interrupted", "run_key": "x_r01",
                "next_command": "python3 scripts/campaign.py --campaign-yaml old.yaml --resume"}))
            rc, out = _quiet_main(["--campaign-yaml", str(y), "--dry-run"])
            self.assertEqual(rc, camp.EXIT_OK)
            self.assertIn("PRIOR ALERT", out)
            self.assertNotIn("old.yaml --resume", out)

    def test_clear_alert_archives_foreign_alert_and_deletes_own(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            c = camp.Campaign({"campaign_id": "newc", "mode": "serial",
                               "runs_root": str(tmp / "runs"),
                               "paths": {"hf_cache_host": t}},
                              tmp / "c.yaml", [], camp.State(campaign_id="newc"),
                              tmp / "state" / "s.json")
            alert = c._alert_path()
            alert.parent.mkdir(parents=True)
            alert.write_text(json.dumps({"campaign_id": "oldc"}))
            with contextlib.redirect_stdout(io.StringIO()):
                c.clear_alert()
            self.assertFalse(alert.exists())
            archived = list((alert.parent / "alerts_archive").glob("ALERT_oldc_*.json"))
            self.assertEqual(len(archived), 1)
            alert.write_text(json.dumps({"campaign_id": "newc"}))
            c.clear_alert()
            self.assertFalse(alert.exists())
            self.assertEqual(len(list((alert.parent / "alerts_archive").iterdir())), 1)

    def test_alert_records_its_state_file(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            c = camp.Campaign({"campaign_id": "newc", "mode": "serial",
                               "runs_root": str(tmp / "runs"),
                               "paths": {"hf_cache_host": t}},
                              tmp / "c.yaml", [], camp.State(campaign_id="newc"),
                              tmp / "state" / "s.json")
            with contextlib.redirect_stdout(io.StringIO()):
                c.write_alert("interrupted", "a_r01", "stop")
            self.assertEqual(json.loads(c._alert_path().read_text())["state_file"],
                             str(tmp / "state" / "s.json"))


@unittest.skipUnless(shutil.which("bash"), "bash required")
class LongtestStatusShowsClosure(unittest.TestCase):
    def test_status_shows_closed_and_no_resume_hint(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "scripts").mkdir()
            shutil.copy(REPO / "scripts" / "longtest_status.sh", tmp / "scripts")
            ext = tmp / "campaigns" / "extension"
            (ext / "state").mkdir(parents=True)
            (ext / "dow_campaign.yaml").write_text(
                "campaign_id: extension_dow_screening\n"
                "state_file: state/dow_campaign_state.json\n"
                f"runs_root: {tmp / 'runs'}\n")
            state = ext / "state" / "dow_campaign_state.json"
            _write_state(state, campaign_id="extension_dow_screening")
            # Simulate an alert that was NOT moved (e.g. written before the close).
            (ext / "state" / "ALERT.json").write_text(json.dumps({
                "campaign_id": "extension_dow_screening", "event": "interrupted",
                "run_key": "a_r01", "ts": "x", "reason": "stop",
                "next_command": "python3 scripts/campaign.py --campaign-yaml d.yaml --resume"}))
            d = json.loads(state.read_text())
            d.update(state="CLOSED", closed_reason=REASON, closed_at="2026-11-20",
                     closed_evidence_dir="closed/x")
            state.write_text(json.dumps(d))
            out = subprocess.run(["bash", str(tmp / "scripts" / "longtest_status.sh")],
                                 capture_output=True, text=True, timeout=60).stdout
            self.assertIn("CLOSED", out)
            self.assertIn(REASON, out)
            self.assertNotIn("--resume", out)


if __name__ == "__main__":
    unittest.main()
