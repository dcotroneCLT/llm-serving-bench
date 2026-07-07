"""Unattended progress heartbeat line (launch_cell.heartbeat_line).

Covers the content fields and the monotonic-elapsed math. Cadence gating (fires
on schedule; a short run prints at most one) is covered end-to-end in
tests/test_phase_b_reaper.py via the supervision-loop harness.

Synthetic, no docker/GPU. Run: python3 -m unittest tests.test_heartbeat
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import launch_cell as lc  # noqa: E402


class HeartbeatLine(unittest.TestCase):
    def test_fields_and_math(self):
        # 12.5h into a 48h run -> 26%. Counts come straight from the summary dict
        # (the summarize_client_csvs shape).
        line = lc.heartbeat_line(
            elapsed_s=12.5 * 3600,
            duration_s=48.0 * 3600,
            client_summary={"total": 1234, "ok": 1200, "dropped": 30},
            runs_root_free_gb=1234.5,
            health_note="OK",
        )
        self.assertEqual(
            line,
            "progress: elapsed 12.5h / 48.0h (26%), "
            "client total=1234 ok=1200 dropped=30, "
            "runs-root free 1234.5 GB, health OK",
        )

    def test_health_warning_passed_through(self):
        line = lc.heartbeat_line(3600, 7200, {"total": 1, "ok": 0, "dropped": 1},
                                 100.0, "dynamo_decode container not running")
        self.assertIn("health dynamo_decode container not running", line)
        self.assertIn("(50%)", line)

    def test_free_none_reads_unknown(self):
        line = lc.heartbeat_line(0, 3600, {}, None, "OK")
        self.assertIn("runs-root free unknown GB", line)
        # Missing count keys default to 0 (summary before any request rows).
        self.assertIn("client total=0 ok=0 dropped=0", line)

    def test_zero_duration_guard(self):
        # duration 0 must not divide-by-zero; pct is 0.
        line = lc.heartbeat_line(10, 0, {"total": 0, "ok": 0, "dropped": 0}, 5.0, "OK")
        self.assertIn("(0%)", line)


if __name__ == "__main__":
    unittest.main()
