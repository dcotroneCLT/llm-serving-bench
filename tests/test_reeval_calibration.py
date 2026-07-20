"""Unit tests for scripts/reeval_calibration.py -- offline re-verdict of recorded
calibration JSONs with the fixed (bracket) selector, no re-sweep.

Run: python3 -m unittest tests.test_reeval_calibration
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import reeval_calibration as rc  # noqa: E402
import calibrate_rate as cr  # noqa: E402

NOW = 1_760_000_000.0


def _row(offered, ratio, drop=0.0, p99=1.0, climb=0.0, n_ok=1000):
    return {"offered_rate": offered, "achieved_rps": round(offered * ratio, 4),
            "achieved_ratio": ratio, "drop_rate": drop, "p99_e2e_s": p99,
            "latency_climb_frac": climb, "offered_span_frac": 0.99,
            "client_rc": 0, "n_ok": n_ok}


def _calib(status="no_stable_point", method=2, fraction=0.85, rows=None):
    return {
        "cell_id": "dow_dynamo_disagg_p12", "system": "dynamo_disagg",
        "fraction": fraction, "status": status,
        "calibration_method_version": method,
        "criteria": {"achieved_ratio_min": 0.98, "drop_max": 0.02,
                     "p99_bound_s": 60.0, "latency_climb_frac": 0.20,
                     "offered_span_min": 0.5},
        "sweep": rows if rows is not None else [],
        "ceiling_rps": None, "ceiling_offered_rps": None, "rate_calibrated_rps": None,
        "provenance": {"calibrated_at_unix": 1_750_000_000.0, "hostname": "h"},
    }


# The field p12 sweep: a low-rate climb-noise failure below a clean stable point.
P12_ROWS = [
    _row(0.25, 0.999, p99=14.0, climb=0.24, n_ok=130),
    _row(0.5, 0.999, p99=22.0, climb=0.06, n_ok=270),
    _row(1.0, 0.94, drop=0.01, p99=54.0, climb=1.43, n_ok=520),
]


class Reevaluate(unittest.TestCase):
    def test_p12_flips_to_ok_at_the_bracketed_ceiling(self):
        calib = _calib(rows=[dict(r) for r in P12_ROWS])
        out = rc.reevaluate(calib, climb_min_samples=None, now_unix=NOW)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["ceiling_offered_rps"], 0.5)
        # rate_calibrated = fraction x achieved-at-ceiling.
        self.assertAlmostEqual(out["rate_calibrated_rps"],
                               round(out["ceiling_rps"] * 0.85, 4))
        self.assertEqual(out["calibration_method_version"], cr.CALIBRATION_METHOD_VERSION)
        self.assertEqual(out["selector_version"], cr.CALIBRATION_SELECTOR_VERSION)
        self.assertEqual(out["reevaluation"]["from_status"], "no_stable_point")
        self.assertEqual(out["reevaluation"]["from_method_version"], 2)
        # Original measurements preserved; selector annotations added.
        self.assertEqual(out["sweep"][0]["achieved_ratio"], 0.999)
        self.assertTrue(out["sweep"][0]["low_rate_anomaly"])

    def test_all_fail_stays_no_stable_point(self):
        rows = [_row(0.25, 0.4, drop=0.5, climb=3.0), _row(0.5, 0.3, drop=0.6)]
        calib = _calib(rows=rows)
        out = rc.reevaluate(calib, climb_min_samples=None, now_unix=NOW)
        self.assertEqual(out["status"], "no_stable_point")
        self.assertIsNone(out["rate_calibrated_rps"])

    def test_top_of_grid_becomes_did_not_saturate(self):
        rows = [_row(1, 1.0), _row(2, 1.0), _row(4, 1.0)]
        out = rc.reevaluate(_calib(rows=rows), climb_min_samples=None, now_unix=NOW)
        self.assertEqual(out["status"], "did_not_saturate")

    def test_engine_failure_is_skipped(self):
        with self.assertRaises(rc.Skip):
            rc.reevaluate(_calib(status="engine_failure", rows=[dict(r) for r in P12_ROWS]),
                          climb_min_samples=None, now_unix=NOW)

    def test_v1_measurement_is_skipped(self):
        with self.assertRaises(rc.Skip):
            rc.reevaluate(_calib(method=1, rows=[dict(r) for r in P12_ROWS]),
                          climb_min_samples=None, now_unix=NOW)

    def test_no_rows_is_skipped(self):
        with self.assertRaises(rc.Skip):
            rc.reevaluate(_calib(rows=[]), climb_min_samples=None, now_unix=NOW)


class PublishAndDriver(unittest.TestCase):
    def test_dry_run_writes_nothing_apply_republishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cal_p12.json"
            p.write_text(json.dumps(_calib(rows=[dict(r) for r in P12_ROWS])))
            before = p.read_text()

            # dry-run: file unchanged
            sys.argv = ["reeval", "--calibration", str(p), "--dry-run"]
            with self.assertRaises(SystemExit) as ctx:
                rc.main()
            self.assertEqual(ctx.exception.code, 0)
            self.assertEqual(p.read_text(), before)

            # apply: republished with the new verdict
            sys.argv = ["reeval", "--calibration", str(p)]
            with self.assertRaises(SystemExit) as ctx:
                rc.main()
            self.assertEqual(ctx.exception.code, 0)
            after = json.loads(p.read_text())
            self.assertEqual(after["status"], "ok")
            self.assertEqual(after["ceiling_offered_rps"], 0.5)
            self.assertEqual(after["calibration_method_version"],
                             cr.CALIBRATION_METHOD_VERSION)
            self.assertFalse((p.with_name(p.name + ".reeval.tmp")).exists())

    def test_missing_file_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["reeval", "--calibration", str(Path(tmp) / "nope.json")]
            with self.assertRaises(SystemExit) as ctx:
                rc.main()
            self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
