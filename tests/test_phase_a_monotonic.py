"""PHASE A point 9: run durations/timeouts use the MONOTONIC clock (immune to
wall-clock jumps); Unix time is kept only for manifest timestamps.
Run: python3 -m unittest tests.test_phase_a_monotonic
"""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = [REPO / "scripts" / "attach_run.py",
         REPO / "scripts" / "launch_cell.py",
         REPO / "monitoring" / "run_monitors.py"]


class MonotonicDurations(unittest.TestCase):
    def setUp(self):
        self.txt = {f.name: f.read_text() for f in FILES}

    def test_monotonic_start_recorded(self):
        for name, txt in self.txt.items():
            self.assertIn("started_mono = time.monotonic()", txt,
                          f"{name} must capture a monotonic start")

    def test_stop_decision_is_monotonic(self):
        for name, txt in self.txt.items():
            self.assertIn("time.monotonic() >= mono_deadline", txt,
                          f"{name} run-stop must compare against a monotonic deadline")

    def test_no_wallclock_stop_decision(self):
        # The old wall-clock stop checks must be gone.
        for name, txt in self.txt.items():
            self.assertNotIn("if time.time() >= deadline", txt,
                             f"{name} still stops on a wall-clock deadline")
            self.assertNotIn("deadline = started_at_unix +", txt,
                             f"{name} still derives the stop deadline from wall clock")

    def test_unix_time_kept_for_manifest_timestamps(self):
        for name, txt in self.txt.items():
            self.assertIn("started_at_unix = time.time()", txt,
                          f"{name} must keep a Unix start timestamp for the manifest")
            self.assertIn("duration_seconds_actual", txt)
            self.assertIn("time.monotonic() - started_mono", txt,
                          f"{name} duration_seconds_actual must be measured monotonically")


if __name__ == "__main__":
    unittest.main()
