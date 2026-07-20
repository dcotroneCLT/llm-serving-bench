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

    def test_non_numeric_timestamp_refused(self):
        # A present-but-non-numeric calibrated_at_unix ("bad") used to slip the
        # staleness gate: calibration_age_days returned None, so the age check was
        # skipped. Unverifiable provenance must raise, not pass.
        now = time.time()
        calib = _calib(now=now)
        calib["provenance"]["calibrated_at_unix"] = "bad"
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_provenance(calib, self._sig(), 14, now)
        self.assertIn("not numeric", str(ctx.exception))

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

    def test_method_version_mismatch_warns_but_does_not_refuse(self):
        # An old-method (v1 / unversioned) file is still accepted -- host/image/age
        # gate it -- but check_calibration_provenance WARNS when asked to, so a
        # finite-window-biased ceiling is not silently reused after the v2 fix.
        import io
        from contextlib import redirect_stderr
        now = time.time()
        calib = _calib(now=now)  # no calibration_method_version -> treated as v1
        buf = io.StringIO()
        with redirect_stderr(buf):
            lc.check_calibration_provenance(
                calib, self._sig(), 14, now,
                warn_method_version=lc.EXPECTED_CALIBRATION_METHOD_VERSION)
        self.assertIn("calibration_method_version", buf.getvalue())

    def test_method_version_match_is_silent(self):
        import io
        from contextlib import redirect_stderr
        now = time.time()
        calib = _calib(now=now)
        calib["calibration_method_version"] = lc.EXPECTED_CALIBRATION_METHOD_VERSION
        buf = io.StringIO()
        with redirect_stderr(buf):
            lc.check_calibration_provenance(
                calib, self._sig(), 14, now,
                warn_method_version=lc.EXPECTED_CALIBRATION_METHOD_VERSION)
        self.assertNotIn("WARNING", buf.getvalue())

    def test_no_warn_param_is_backward_compatible(self):
        # Existing callers pass no warn_method_version -> no version check at all.
        import io
        from contextlib import redirect_stderr
        now = time.time()
        buf = io.StringIO()
        with redirect_stderr(buf):
            lc.check_calibration_provenance(_calib(now=now), self._sig(), 14, now)
        self.assertNotIn("WARNING", buf.getvalue())

    def test_min_method_version_hard_rejects_old(self):
        # A v1 (unversioned) file is REFUSED when a minimum method version is
        # required, even if it is fresh and host/image-matching.
        now = time.time()
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_provenance(
                _calib(now=now), self._sig(), 14, now, min_method_version=2)
        self.assertIn("method_version", str(ctx.exception))

    def test_min_method_version_accepts_current(self):
        now = time.time()
        calib = _calib(now=now)
        calib["calibration_method_version"] = 2
        lc.check_calibration_provenance(
            calib, self._sig(), 14, now, min_method_version=2)  # no raise

    def test_min_method_version_rejects_unparseable(self):
        now = time.time()
        calib = _calib(now=now)
        calib["calibration_method_version"] = "bad"
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_provenance(
                calib, self._sig(), 14, now, min_method_version=2)

    def test_selector_revision_2_1_accepted_by_min_2(self):
        # A v2.1 (bracket-selector) file has the SAME measurement as v2, so a
        # min_method_version=2 gate must accept it (2.1 >= 2), not reject it.
        now = time.time()
        calib = _calib(now=now)
        calib["calibration_method_version"] = 2.1
        lc.check_calibration_provenance(
            calib, self._sig(), 14, now, min_method_version=2)  # no raise

    def test_selector_revision_2_1_does_not_warn_against_expected_2(self):
        # The measurement (integer) part matches expected v2, so the soft warn
        # must stay silent for a selector-only revision.
        import io
        from contextlib import redirect_stderr
        now = time.time()
        calib = _calib(now=now)
        calib["calibration_method_version"] = 2.1
        buf = io.StringIO()
        with redirect_stderr(buf):
            lc.check_calibration_provenance(
                calib, self._sig(), 14, now,
                warn_method_version=lc.EXPECTED_CALIBRATION_METHOD_VERSION)
        self.assertNotIn("WARNING", buf.getvalue())


class CheckCalibrationBinding(unittest.TestCase):
    """A calibration must be for THIS cell at THIS fraction, else Factor A (rate)
    of the DoW is silently wrong."""

    def _calib(self, cell_id="dow_vllm_p01", fraction=0.30):
        c = {"status": "ok", "rate_calibrated_rps": 3.0}
        if cell_id is not None:
            c["cell_id"] = cell_id
        if fraction is not None:
            c["fraction"] = fraction
        return c

    def test_matching_cell_and_fraction_passes(self):
        lc.check_calibration_binding(self._calib(), "dow_vllm_p01", 0.30)

    def test_wrong_cell_refused(self):
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_binding(self._calib(cell_id="dow_vllm_p02"),
                                         "dow_vllm_p01", 0.30)
        self.assertIn("cell", str(ctx.exception))

    def test_wrong_fraction_refused(self):
        # The 0.85 file used on a 0.30 cell -- the exact review scenario.
        with self.assertRaises(lc.CalibrationError) as ctx:
            lc.check_calibration_binding(self._calib(fraction=0.85),
                                         "dow_vllm_p01", 0.30)
        self.assertIn("fraction", str(ctx.exception))

    def test_missing_cell_id_when_fraction_expected_refused(self):
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_binding(self._calib(cell_id=None),
                                         "dow_vllm_p01", 0.30)

    def test_missing_fraction_when_expected_refused(self):
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_binding(self._calib(fraction=None),
                                         "dow_vllm_p01", 0.30)

    def test_non_numeric_fraction_refused(self):
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_binding(self._calib(fraction="bad"),
                                         "dow_vllm_p01", 0.30)

    def test_no_expected_fraction_is_lenient_but_catches_wrong_cell(self):
        # A non-DoW cell (no declared fraction): an unlabeled calibration is
        # tolerated, but a file that names a DIFFERENT cell is still refused.
        lc.check_calibration_binding({"status": "ok"}, "val_vllm", None)  # no raise
        lc.check_calibration_binding(self._calib(cell_id="val_vllm", fraction=None),
                                     "val_vllm", None)  # matches -> no raise
        with self.assertRaises(lc.CalibrationError):
            lc.check_calibration_binding(self._calib(cell_id="other", fraction=None),
                                         "val_vllm", None)


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


def _spec(tmp, calib_obj, calibration_fraction=None):
    p = Path(tmp) / "calib.json"
    p.write_text(json.dumps(calib_obj))
    return camp.RunSpec(cell_id="a", cell_yaml=str(Path(tmp) / "nope.yaml"),
                        replica=1, duration_s=100, calibration_file=str(p),
                        calibration_required=True,
                        calibration_fraction=calibration_fraction)


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

    def test_preflight_rejects_wrong_fraction_calibration(self):
        # A cell declaring fraction 0.30 must reject a file calibrated at 0.85
        # (the review's Factor-A-invalidation scenario) BEFORE run 1.
        with tempfile.TemporaryDirectory() as tmp:
            calib = _calib()
            calib["cell_id"] = "a"
            calib["fraction"] = 0.85
            spec = _spec(tmp, calib, calibration_fraction=0.30)
            ok, msg = camp.validate_calibration(spec)
            self.assertFalse(ok)
            self.assertIn("fraction", msg)

    def test_preflight_rejects_wrong_cell_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = _calib()
            calib["cell_id"] = "some_other_cell"
            calib["fraction"] = 0.30
            spec = _spec(tmp, calib, calibration_fraction=0.30)
            ok, msg = camp.validate_calibration(spec)
            self.assertFalse(ok)
            self.assertIn("cell", msg)

    def test_preflight_accepts_matching_cell_and_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            calib = _calib()
            calib["cell_id"] = "a"
            calib["fraction"] = 0.30
            spec = _spec(tmp, calib, calibration_fraction=0.30)
            ok, _ = camp.validate_calibration(spec)
            self.assertTrue(ok)

    def test_build_cmd_passes_max_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp, _calib())
            c = _make_campaign(tmp, spec, calibration_max_age_days=9)
            cmd = c._build_cmd(spec, attempt=1)
            self.assertIn("--calibration-max-age-days", cmd)
            self.assertIn("9.0", cmd)


if __name__ == "__main__":
    unittest.main()
