"""PRE-CAMPAIGN HARDENING item 4: scoped drift gates.

 (a) campaign environment baseline (kernel / driver / image digests) captured at
     --start and enforced campaign-fatal at every dispatch;
 (b) post-run GPU ECC drift check (analysis/ecc_check.py) shared by the run
     validators: uncorrected-volatile increment FAIL, corrected-volatile WARN.

(4c, the mid-run inode floor, is covered in tests/test_disk_watchdog.py.)

Synthetic, no docker/GPU. Run: python3 -m unittest tests.test_drift_gates
"""
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "analysis"))

import campaign as camp  # noqa: E402
import ecc_check  # noqa: E402


# --------------------------------------------------------------------------
# 4a: environment baseline + drift
# --------------------------------------------------------------------------


class EnvironmentDrift(unittest.TestCase):
    def test_no_drift(self):
        b = {"kernel": "5.15", "driver_version": "535.10",
             "image_digests": {"repo:tag": "sha256:aaa"}}
        self.assertIsNone(camp.environment_drift(b, dict(b)))

    def test_driver_change_is_drift(self):
        b = {"kernel": "5.15", "driver_version": "535.10", "image_digests": {}}
        c = {"kernel": "5.15", "driver_version": "545.20", "image_digests": {}}
        d = camp.environment_drift(b, c)
        self.assertIn("driver_version", d)

    def test_kernel_change_is_drift(self):
        b = {"kernel": "5.15", "driver_version": None, "image_digests": {}}
        c = {"kernel": "6.2", "driver_version": None, "image_digests": {}}
        self.assertIn("kernel", camp.environment_drift(b, c))

    def test_image_digest_change_is_drift(self):
        b = {"kernel": "5.15", "driver_version": "535", "image_digests": {"r:t": "sha256:OLD"}}
        c = {"kernel": "5.15", "driver_version": "535", "image_digests": {"r:t": "sha256:NEW"}}
        self.assertIn("image r:t", camp.environment_drift(b, c))

    def test_unknown_field_not_compared(self):
        # A field None on either side is not compared (a transient nvidia-smi
        # miss must not fake a driver drift).
        b = {"kernel": "5.15", "driver_version": "535.10", "image_digests": {}}
        c = {"kernel": "5.15", "driver_version": None, "image_digests": {}}
        self.assertIsNone(camp.environment_drift(b, c))

    def test_new_image_tag_not_a_drift(self):
        # A tag only present on the current side (never in baseline) is not drift.
        b = {"kernel": "5.15", "driver_version": "535", "image_digests": {"a:1": "sha256:x"}}
        c = {"kernel": "5.15", "driver_version": "535",
             "image_digests": {"a:1": "sha256:x", "b:2": "sha256:y"}}
        self.assertIsNone(camp.environment_drift(b, c))


def _campaign(tmp, spec):
    cfg = {
        "campaign_id": "testc", "mode": "serial",
        "runs_root": str(Path(tmp) / "runs"),
        "paths": {"hf_cache_host": str(Path(tmp) / "hf"), "repo_root": str(Path(tmp) / "repo")},
        "retry_policy": {"max_retries": 1}, "inter_run_cooldown_s": 0, "min_free_gb": 0.0,
    }
    c = camp.Campaign(cfg, Path(tmp) / "campaign.yaml", [spec],
                      camp.State(campaign_id="testc"), Path(tmp) / "state" / "s.json")
    c._skip_run_dir_prep = True
    return c


class CampaignBaselineGate(unittest.TestCase):
    def _spec(self):
        return camp.RunSpec(cell_id="a", cell_yaml="/nonexistent/a.yaml",
                            replica=1, duration_s=100)

    def test_no_baseline_no_check(self):
        # A pre-hardening state (baseline None) never probes the environment.
        with tempfile.TemporaryDirectory() as tmp:
            c = _campaign(tmp, self._spec())
            probed = {"n": 0}
            c.capture_environment_baseline = lambda: probed.__setitem__("n", probed["n"] + 1) or {}
            self.assertIsNone(c.check_environment_drift())
            self.assertEqual(probed["n"], 0)

    def test_dispatch_drift_is_campaign_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec()
            c = _campaign(tmp, spec)
            c.state.baseline = {"kernel": "5.15", "driver_version": "535.10",
                                "image_digests": {}}
            # Current host drifted (driver upgraded).
            c.capture_environment_baseline = lambda: {
                "kernel": "5.15", "driver_version": "545.20", "image_digests": {}}
            c._launch_cell_rc = lambda s, a: 0  # never reached
            with self.assertRaises(camp.CampaignFatal) as ctx:
                c._run_with_retry(spec)
            self.assertEqual(ctx.exception.rc, camp.LC_PRECONDITION)
            self.assertEqual(c.state.runs["a_r01"].status, "precondition_failed")

    def test_no_drift_dispatches_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec()
            c = _campaign(tmp, spec)
            base = {"kernel": "5.15", "driver_version": "535.10", "image_digests": {}}
            c.state.baseline = base
            c.capture_environment_baseline = lambda: dict(base)
            c._launch_cell_rc = lambda s, a: 0
            self.assertEqual(c._run_with_retry(spec), "completed")

    def test_read_cell_image_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pin = tmp / "pin.json"
            pin.write_text('{"image_tag": "repo/img:tag", "digest": "sha256:abc"}')
            cell = tmp / "cell.yaml"
            cell.write_text(
                "cell_id: a\n"
                "engine:\n"
                "  image_repo: repo/img\n"
                "  image_tag: tag\n"
                f"  digest_pin_file: {pin}\n"
            )
            spec = camp.RunSpec(cell_id="a", cell_yaml=str(cell), replica=1, duration_s=1)
            out = camp.read_cell_image_digests([spec], tmp)
            self.assertEqual(out, {"repo/img:tag": "sha256:abc"})


# --------------------------------------------------------------------------
# 4b: GPU ECC drift check
# --------------------------------------------------------------------------


def _gpu_csv(run_dir, rows, gpu_index=0):
    run_dir.mkdir(parents=True, exist_ok=True)
    header = "ts_unix,gpu_index,vram_used_bytes,ecc_db_volatile,ecc_sb_volatile\n"
    body = "".join(
        f"{ts},{gpu_index},1000,{db},{sb}\n" for ts, db, sb in rows)
    (run_dir / f"gpu{gpu_index}_000000.csv").write_text(header + body)


class EccVerdict(unittest.TestCase):
    def test_no_increment_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            _gpu_csv(rd, [(1, 0, 0), (2, 0, 0), (3, 0, 0)])
            verdict, msgs = ecc_check.ecc_verdict(rd)
            self.assertEqual(verdict, "ok")

    def test_uncorrected_increment_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            _gpu_csv(rd, [(1, 0, 0), (2, 0, 0), (3, 2, 0)])  # ecc_db 0 -> 2
            verdict, msgs = ecc_check.ecc_verdict(rd)
            self.assertEqual(verdict, "fail")
            self.assertTrue(any("uncorrected" in m for m in msgs))

    def test_corrected_increment_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            _gpu_csv(rd, [(1, 0, 0), (2, 0, 3), (3, 0, 5)])  # ecc_sb 0 -> 5
            verdict, msgs = ecc_check.ecc_verdict(rd)
            self.assertEqual(verdict, "warn")
            self.assertTrue(any("corrected" in m for m in msgs))

    def test_uncorrected_dominates_corrected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            _gpu_csv(rd, [(1, 0, 0), (2, 1, 4)])  # both incremented
            verdict, _ = ecc_check.ecc_verdict(rd)
            self.assertEqual(verdict, "fail")

    def test_blank_ecc_columns_skipped_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            (rd / "gpu0_000000.csv").write_text(
                "ts_unix,gpu_index,vram_used_bytes,ecc_db_volatile,ecc_sb_volatile\n"
                "1,0,1000,,\n2,0,1000,,\n")
            verdict, msgs = ecc_check.ecc_verdict(rd)
            self.assertEqual(verdict, "ok")
            self.assertTrue(any("skipped" in m for m in msgs))

    def test_no_gpu_csv_ok_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdict, msgs = ecc_check.ecc_verdict(Path(tmp))
            self.assertEqual(verdict, "ok")

    def test_per_gpu_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            _gpu_csv(rd, [(1, 0, 0), (2, 0, 0)], gpu_index=0)   # clean
            _gpu_csv(rd, [(1, 0, 0), (2, 3, 0)], gpu_index=1)   # gpu1 uncorrected
            verdict, msgs = ecc_check.ecc_verdict(rd)
            self.assertEqual(verdict, "fail")
            self.assertTrue(any("gpu1" in m and "uncorrected" in m for m in msgs))


if __name__ == "__main__":
    unittest.main()
