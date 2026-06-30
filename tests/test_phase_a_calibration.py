"""PHASE A point 7: calibration must be enforced (not just warned).
Run: python3 -m unittest tests.test_phase_a_calibration
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_rate as cr  # noqa: E402
import launch_cell as lc  # noqa: E402


def _row(offered, ratio, drop=0.0, p99=1.0, climb=0.0):
    return {"offered_rate": offered, "achieved_rps": offered * ratio,
            "achieved_ratio": ratio, "drop_rate": drop, "p99_e2e_s": p99,
            "latency_climb_frac": climb}


class SelectCeiling(unittest.TestCase):
    def test_ok_when_sweep_passes_the_knee(self):
        rows = [_row(1, 1.0), _row(2, 1.0), _row(4, 0.5, drop=0.3, climb=2.0)]
        ceiling, status = cr.select_ceiling(rows)
        self.assertEqual(status, "ok")
        self.assertEqual(ceiling["offered_rate"], 2)

    def test_did_not_saturate_when_all_stable(self):
        rows = [_row(1, 1.0), _row(2, 1.0), _row(4, 1.0)]
        _, status = cr.select_ceiling(rows)
        self.assertEqual(status, "did_not_saturate")

    def test_no_stable_point_when_lowest_unstable(self):
        rows = [_row(1, 0.4, drop=0.5, climb=3.0), _row(2, 0.3, drop=0.6)]
        ceiling, status = cr.select_ceiling(rows)
        self.assertIsNone(ceiling)
        self.assertEqual(status, "no_stable_point")


class ExitCodes(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(cr.exit_code_for_status("ok"), 0)
        self.assertEqual(cr.exit_code_for_status("no_stable_point"), 3)
        self.assertEqual(cr.exit_code_for_status("did_not_saturate"), 4)
        self.assertEqual(cr.exit_code_for_status("anything_else"), 0)


class ResolveCalibratedRate(unittest.TestCase):
    def test_ok_returns_rate(self):
        self.assertEqual(
            lc.resolve_calibrated_rate({"status": "ok", "rate_calibrated_rps": 3.5}, False), 3.5)

    def test_non_ok_refused_by_default(self):
        with self.assertRaises(lc.CalibrationError):
            lc.resolve_calibrated_rate({"status": "did_not_saturate", "rate_calibrated_rps": 8.0}, False)

    def test_non_ok_allowed_with_override(self):
        self.assertEqual(
            lc.resolve_calibrated_rate({"status": "did_not_saturate", "rate_calibrated_rps": 8.0}, True), 8.0)

    def test_missing_rate_refused_even_with_override(self):
        with self.assertRaises(lc.CalibrationError):
            lc.resolve_calibrated_rate({"status": "ok", "rate_calibrated_rps": None}, True)


if __name__ == "__main__":
    unittest.main()
