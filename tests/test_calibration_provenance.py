"""PRE-CAMPAIGN HARDENING item 1: calibration provenance + staleness/host/image gate.

A month-1 ceiling must never silently drive a month-3 run. calibrate_rate.py now
records a provenance block (when/where/against-which-image), and both launch_cell
and campaign pre-flight REJECT a calibration that is too old or whose host/image
signature no longer matches -- at pre-flight for every queued run AND again at
dispatch time.

Synthetic, no docker/GPU. Run: python3 -m unittest tests.test_calibration_provenance
"""
import json
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_rate as cr  # noqa: E402
import campaign as camp  # noqa: E402
import launch_cell as lc  # noqa: E402


def _calib(now=None, **prov_overrides):
    """An 'ok' calibration with a provenance block taken 'now' on this host."""
    prov = {
        "calibrated_at_unix": now if now is not None else time.time(),
        "calibrated_at_iso": "now",
        "hostname": socket.gethostname(),
        "gpu_name": None,
        "driver_version": None,
        "image_tag": None,
        "image_digest": None,
        "client_config_hash": None,
    }
    prov.update(prov_overrides)
    return {"status": "ok", "rate_calibrated_rps": 3.0, "provenance": prov}


# --------------------------------------------------------------------------
# calibrate_rate.py provenance emission
# --------------------------------------------------------------------------


class BuildProvenance(unittest.TestCase):
    def test_fields_and_config_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "client_config.yaml"
            cfg.write_text("target_rate_rps: 2.0\n")
            prov = cr.build_provenance(cfg, "repo/img:tag", "sha256:abc", 1_700_000_000.0)
        self.assertEqual(prov["calibrated_at_unix"], 1_700_000_000.0)
        self.assertEqual(prov["hostname"], socket.gethostname())
        self.assertEqual(prov["image_tag"], "repo/img:tag")
        self.assertEqual(prov["image_digest"], "sha256:abc")
        # A real sha256 hex digest of the config bytes.
        self.assertRegex(prov["client_config_hash"], r"^[0-9a-f]{64}$")
        self.assertIn("calibrated_at_iso", prov)

    def test_missing_config_hash_is_none_not_crash(self):
        prov = cr.build_provenance(Path("/nonexistent/x.yaml"), None, None, 1.0)
        self.assertIsNone(prov["client_config_hash"])
        self.assertIsNone(prov["image_tag"])


# --------------------------------------------------------------------------
# launch_cell gate (pure function)
# --------------------------------------------------------------------------


class CheckCalibrationProvenance(unittest.TestCase):
    def _sig(self, **over):
        s = lc.current_calibration_signature(
            socket.gethostname(), None, None, None, None)
        s.update(over)
        return s

    def test_fresh_matching_passes(self):
        now = time.time()
        lc.check_calibration_provenance(_calib(now=now), self._sig(), 14, now)

    def test_too_old_refused(self):
        now = time.time()
        old = _calib(now=now - 30 * 86400)
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_provenance(old, self._sig(), 14, now)
        self.assertIn("days old", str(ctx.exception))

    def test_missing_provenance_refused(self):
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_provenance(
                {"status": "ok", "rate_calibrated_rps": 3.0}, self._sig(), 14, time.time())

    def test_hostname_mismatch_refused(self):
        now = time.time()
        calib = _calib(now=now, hostname="some-other-box")
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_provenance(calib, self._sig(), 14, now)
        self.assertIn("hostname", str(ctx.exception))

    def test_image_digest_mismatch_refused(self):
        now = time.time()
        calib = _calib(now=now, image_digest="sha256:OLD")
        cur = self._sig(image_digest="sha256:NEW")
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_provenance(calib, cur, 14, now)
        self.assertIn("image_digest", str(ctx.exception))

    def test_field_absent_on_current_side_is_skipped(self):
        # calibration recorded a gpu_name, but this box cannot read one (None):
        # the field is not compared, so a match on everything else passes.
        now = time.time()
        calib = _calib(now=now, gpu_name="NVIDIA L40S")
        lc.check_calibration_provenance(calib, self._sig(), 14, now)  # no raise

    def test_driver_mismatch_when_both_present_refused(self):
        now = time.time()
        calib = _calib(now=now, driver_version="535.10")
        cur = self._sig(driver_version="545.20")
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_provenance(calib, cur, 14, now)

    def test_none_max_age_only_signature(self):
        # max_age None -> age never trips; only the signature is gated.
        now = time.time()
        lc.check_calibration_provenance(_calib(now=now - 999 * 86400), self._sig(), None, now)


class CalibrationAgeDays(unittest.TestCase):
    def test_age(self):
        now = 1_700_000_000.0
        self.assertAlmostEqual(
            lc.calibration_age_days(_calib(now=now - 3 * 86400), now), 3.0, places=3)

    def test_no_timestamp(self):
        self.assertIsNone(lc.calibration_age_days({"provenance": {}}, 1.0))


# --------------------------------------------------------------------------
# campaign wiring: pre-flight (all queued) + dispatch re-check
# --------------------------------------------------------------------------


def _make_campaign(tmp, spec, **overrides):
    cfg = {
        "campaign_id": "testc",
        "mode": "serial",
        "runs_root": str(Path(tmp) / "runs"),
        "paths": {"hf_cache_host": str(Path(tmp) / "hf"), "repo_root": str(Path(tmp) / "repo")},
        "retry_policy": {"max_retries": 1},
        "inter_run_cooldown_s": 0,
        "min_free_gb": 0.0,
    }
    cfg.update(overrides)
    c = camp.Campaign(cfg, Path(tmp) / "campaign.yaml", [spec],
                      camp.State(campaign_id="testc"), Path(tmp) / "state" / "s.json")
    c._skip_run_dir_prep = True
    return c


def _spec(tmp, calib_obj):
    p = Path(tmp) / "calib.json"
    p.write_text(json.dumps(calib_obj))
    return camp.RunSpec(cell_id="a", cell_yaml=str(Path(tmp) / "nope.yaml"),
                        replica=1, duration_s=100, calibration_file=str(p),
                        calibration_required=True)


class CampaignProvenanceGate(unittest.TestCase):
    def test_preflight_passes_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = _make_campaign(tmp, _spec(tmp, _calib()))
            c.preflight()  # must not raise

    def test_preflight_rejects_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = _calib(now=time.time() - 30 * 86400)
            c = _make_campaign(tmp, _spec(tmp, stale))
            with self.assertRaises(camp.PreflightError) as ctx:
                c.preflight()
            self.assertIn("STALE", str(ctx.exception))

    def test_configurable_max_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 3-day-old calibration with a 1-day max age -> stale.
            three_day = _calib(now=time.time() - 3 * 86400)
            c = _make_campaign(tmp, _spec(tmp, three_day), calibration_max_age_days=1)
            with self.assertRaises(camp.PreflightError):
                c.preflight()

    def test_dispatch_recheck_is_campaign_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = _calib(now=time.time() - 30 * 86400)
            spec = _spec(tmp, stale)
            c = _make_campaign(tmp, spec)
            # Never reaches the launcher: the dispatch gate fires first.
            c._launch_cell_rc = lambda s, a: 0
            with self.assertRaises(camp.CampaignFatal) as ctx:
                c._run_with_retry(spec)
            self.assertEqual(ctx.exception.rc, camp.LC_PRECONDITION)
            self.assertEqual(c.state.runs["a_r01"].status, "precondition_failed")

    def test_build_cmd_passes_max_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp, _calib())
            c = _make_campaign(tmp, spec, calibration_max_age_days=9)
            cmd = c._build_cmd(spec, attempt=1)
            self.assertIn("--calibration-max-age-days", cmd)
            self.assertIn("9.0", cmd)


if __name__ == "__main__":
    unittest.main()
