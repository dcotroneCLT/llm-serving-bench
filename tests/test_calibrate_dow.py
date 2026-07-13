"""Unit tests for the DoW calibration orchestrator (scripts/calibrate_dow.py).

Synthetic and hermetic: the docker / GPU / subprocess boundaries are the
orchestrator's injection seams (_acquire_slot, _bring_up_fn, _sweep_fn,
_teardown_fn, _status_fn, _dry_run_fn), so these tests exercise the real control
flow -- system grouping, skip-valid / force-recalibrate, the failure policy
(one bad sweep must not halt the rest), and lock refusal -- with no host.

The skip-valid check is verified against the REAL launch_cell gate functions
(resolve_calibrated_rate / check_calibration_binding / check_calibration_
provenance), not a reimplementation, so "valid" here means exactly what a run
will accept.

Run: python3 -m unittest tests.test_calibrate_dow
"""
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_dow as cdow  # noqa: E402
import campaign as camp  # noqa: E402


def _spec(cell_id, fraction=0.3, calib="/does/not/exist.json"):
    return camp.RunSpec(
        cell_id=cell_id, cell_yaml="", replica=1, duration_s=1,
        calibration_file=calib, calibration_required=True,
        calibration_fraction=fraction)


class _Lock:
    def close(self):  # slot handle stand-in
        pass


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


class SystemOf(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(cdow.system_of("dow_dynamo_disagg_cp1"), "dynamo_disagg")
        self.assertEqual(cdow.system_of("dow_dynamo_disagg_p16"), "dynamo_disagg")
        self.assertEqual(cdow.system_of("dow_triton_p02"), "triton")
        self.assertEqual(cdow.system_of("dow_triton_cp3"), "triton")
        self.assertEqual(cdow.system_of("dow_vllm_p01"), "vllm")
        self.assertEqual(cdow.system_of("dow_vllm_cp2"), "vllm")


class Grouping(unittest.TestCase):
    def test_system_grouped_order_preserves_within_order(self):
        # A scrambled, interleaved campaign order (as dow_campaign.yaml really is).
        specs = [
            _spec("dow_vllm_cp1"), _spec("dow_dynamo_disagg_cp1"),
            _spec("dow_triton_cp1"), _spec("dow_vllm_p11"),
            _spec("dow_dynamo_disagg_p12"), _spec("dow_triton_p02"),
        ]
        groups = cdow.group_by_system(specs)
        self.assertEqual([g[0] for g in groups], ["dynamo_disagg", "triton", "vllm"])
        # Within each system, the campaign list order is preserved.
        self.assertEqual([s.cell_id for s in groups[0][1]],
                         ["dow_dynamo_disagg_cp1", "dow_dynamo_disagg_p12"])
        self.assertEqual([s.cell_id for s in groups[2][1]],
                         ["dow_vllm_cp1", "dow_vllm_p11"])


class SuggestNextGrid(unittest.TestCase):
    def test_did_not_saturate_extends_up(self):
        g = cdow.suggest_next_grid("did_not_saturate", [0.5, 1, 2, 4, 8])
        self.assertEqual(g[-2:], [16.0, 24.0])
        self.assertTrue(all(g[i] <= g[i + 1] for i in range(len(g) - 1)))

    def test_no_stable_point_drops_down(self):
        g = cdow.suggest_next_grid("no_stable_point", [0.25, 0.5, 1])
        self.assertTrue(g[-1] < 0.25)  # below the previous floor
        self.assertTrue(all(g[i] <= g[i + 1] for i in range(len(g) - 1)))

    def test_ok_has_no_suggestion(self):
        self.assertIsNone(cdow.suggest_next_grid("ok", [1, 2, 4]))


# --------------------------------------------------------------------------
# Skip-valid / force-recalibrate against the REAL launch_cell gates
# --------------------------------------------------------------------------


class CalibrationIsValid(unittest.TestCase):
    def _write(self, tmp, obj):
        p = Path(tmp) / "c.json"
        p.write_text(json.dumps(obj))
        return p

    def _ok_calib(self, now, cell_id="dow_vllm_p01", fraction=0.3):
        return {
            "cell_id": cell_id, "fraction": fraction, "status": "ok",
            "rate_calibrated_rps": 2.5,
            "provenance": {"calibrated_at_unix": now},
        }

    def test_valid_when_ok_bound_and_fresh(self):
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, self._ok_calib(now))
            ok, reason = cdow.calibration_is_valid(
                p, "dow_vllm_p01", 0.3, current_sig=None, max_age_days=14, now=now)
            self.assertTrue(ok, reason)

    def test_missing_file(self):
        ok, reason = cdow.calibration_is_valid(
            "/nope.json", "dow_vllm_p01", 0.3, None, 14, 1.0)
        self.assertFalse(ok)
        self.assertIn("missing", reason)

    def test_wrong_fraction_rejected(self):
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, self._ok_calib(now, fraction=0.85))
            ok, reason = cdow.calibration_is_valid(
                p, "dow_vllm_p01", 0.3, None, 14, now)
            self.assertFalse(ok)
            self.assertIn("binding", reason)

    def test_stale_rejected_via_provenance(self):
        now = 1_700_000_000.0
        old = now - 100 * 86400
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, self._ok_calib(old))
            sig = {"hostname": None, "gpu_name": None, "driver_version": None,
                   "image_tag": None, "image_digest": None}
            ok, reason = cdow.calibration_is_valid(
                p, "dow_vllm_p01", 0.3, sig, 14, now)
            self.assertFalse(ok)
            self.assertIn("provenance", reason)

    def test_sig_none_skips_provenance_leg(self):
        # A very old file is accepted when we cannot build a signature AND no age
        # can be checked... but age IS in the file, so with sig=None the age gate
        # is skipped only because check_calibration_provenance is not called.
        now = 1_700_000_000.0
        old = now - 100 * 86400
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, self._ok_calib(old))
            ok, _ = cdow.calibration_is_valid(
                p, "dow_vllm_p01", 0.3, current_sig=None, max_age_days=14, now=now)
            self.assertTrue(ok)  # binding + usable rate pass; provenance not run

    def test_non_ok_status_rejected(self):
        now = 1_700_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            calib = self._ok_calib(now)
            calib["status"] = "did_not_saturate"
            p = self._write(tmp, calib)
            ok, reason = cdow.calibration_is_valid(p, "dow_vllm_p01", 0.3, None, 14, now)
            self.assertFalse(ok)
            self.assertIn("not usable", reason)


# --------------------------------------------------------------------------
# resolve_grid precedence
# --------------------------------------------------------------------------


class ResolveGrid(unittest.TestCase):
    def _orch(self, grid_map=None, env_rates=None):
        return cdow.Orchestrator(
            campaign_yaml=Path("x.yaml"), runs_root=Path("/tmp"), repo_root=REPO,
            hf_cache_host=Path("/tmp"), max_age_days=14, window_s=1, cooldown_s=0,
            bringup_retries=1, recalibrate=set(), grid_map=grid_map or {},
            env_rates=env_rates)

    def test_builtin_per_system_default(self):
        o = self._orch()
        self.assertEqual(o.resolve_grid(_spec("dow_dynamo_disagg_cp1")),
                         cdow.DEFAULT_RATE_GRIDS["dynamo_disagg"])
        self.assertEqual(o.resolve_grid(_spec("dow_vllm_p01")),
                         cdow.DEFAULT_RATE_GRIDS["vllm"])

    def test_cell_id_beats_system_beats_env(self):
        o = self._orch(grid_map={"vllm": [1, 2], "dow_vllm_p01": [9, 9]},
                       env_rates=[7])
        self.assertEqual(o.resolve_grid(_spec("dow_vllm_p01")), [9.0, 9.0])   # cell
        self.assertEqual(o.resolve_grid(_spec("dow_vllm_p02")), [1.0, 2.0])   # system
        self.assertEqual(o.resolve_grid(_spec("dow_triton_p01")), [7.0])      # env


# --------------------------------------------------------------------------
# Orchestration control flow (failure policy, skip, lock)
# --------------------------------------------------------------------------


class OrchestrationFlow(unittest.TestCase):
    def _orch(self, specs):
        o = cdow.Orchestrator(
            campaign_yaml=Path("x.yaml"), runs_root=Path("/tmp"), repo_root=REPO,
            hf_cache_host=Path("/tmp"), max_age_days=14, window_s=1, cooldown_s=0,
            bringup_retries=1, recalibrate=set(), grid_map={}, env_rates=None,
            logf=io.StringIO())
        o._specs = specs
        o._now = lambda: 0.0
        return o

    def _instrument(self, o, statuses, sweep_result):
        calls = {"bringup": [], "teardown": [], "sweep": []}
        o._acquire_slot = lambda rr: _Lock()
        o._status_fn = lambda spec: statuses(spec)
        o._bring_up_fn = lambda system, rep: (
            calls["bringup"].append(system)
            or types.SimpleNamespace(system=system))
        o._sweep_fn = lambda spec, eng, grid: (
            calls["sweep"].append(spec.cell_id) or sweep_result(spec, grid))
        o._teardown_fn = lambda eng: calls["teardown"].append(eng.system)
        o._dry_run_fn = lambda: (0, "0 MISSING")
        return calls

    def test_one_bad_sweep_does_not_halt_and_exit_is_nonzero(self):
        specs = [
            _spec("dow_dynamo_disagg_cp1"), _spec("dow_dynamo_disagg_p12"),
            _spec("dow_triton_cp1"), _spec("dow_triton_p02"),
            _spec("dow_vllm_cp1"), _spec("dow_vllm_p11"),
        ]
        o = self._orch(specs)
        bad = "dow_triton_p02"

        def sweep_result(spec, grid):
            if spec.cell_id == bad:
                return {"status": "did_not_saturate", "ceiling_rps": 8.0,
                        "rate_calibrated_rps": None, "rc": 4, "grid": list(grid),
                        "skipped": False,
                        "suggested_grid": cdow.suggest_next_grid("did_not_saturate", grid)}
            return {"status": "ok", "ceiling_rps": 3.0, "rate_calibrated_rps": 2.55,
                    "rc": 0, "grid": list(grid), "skipped": False}

        calls = self._instrument(o, lambda s: (False, "missing"), sweep_result)
        rc = o.run()

        # Every cell was swept (the bad one did NOT halt the rest).
        self.assertEqual(len(calls["sweep"]), 6)
        self.assertIn(bad, calls["sweep"])
        # One bring-up + teardown per system.
        self.assertEqual(calls["bringup"], ["dynamo_disagg", "triton", "vllm"])
        self.assertEqual(calls["teardown"], ["dynamo_disagg", "triton", "vllm"])
        # Non-zero exit and the summary carries the bad cell + a wider grid.
        self.assertEqual(rc, cdow.EXIT_SOME_FAILED)
        log = o._logf.getvalue()
        self.assertIn(bad, log)
        self.assertIn("did_not_saturate", log)

    def test_all_ok_exits_zero(self):
        specs = [_spec("dow_vllm_cp1"), _spec("dow_vllm_p11")]
        o = self._orch(specs)
        calls = self._instrument(
            o, lambda s: (False, "missing"),
            lambda spec, grid: {"status": "ok", "ceiling_rps": 3.0,
                                "rate_calibrated_rps": 2.55, "rc": 0,
                                "grid": list(grid), "skipped": False})
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_OK)
        self.assertEqual(len(calls["sweep"]), 2)
        self.assertEqual(calls["bringup"], ["vllm"])

    def test_all_valid_skips_bringup_entirely(self):
        specs = [_spec("dow_vllm_cp1"), _spec("dow_vllm_p11")]
        o = self._orch(specs)
        calls = self._instrument(
            o, lambda s: (True, "ok (cached)"),
            lambda spec, grid: self.fail("sweep must not run when all valid"))
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_OK)
        self.assertEqual(calls["bringup"], [])   # no engine churn on a clean resume
        self.assertEqual(calls["sweep"], [])

    def test_recalibrate_all_forces_sweep_over_valid(self):
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)
        o.recalibrate = {"all"}
        calls = self._instrument(
            o, lambda s: (True, "ok (cached)"),   # would skip, but forced
            lambda spec, grid: {"status": "ok", "ceiling_rps": 3.0,
                                "rate_calibrated_rps": 2.55, "rc": 0,
                                "grid": list(grid), "skipped": False})
        o.run()
        self.assertEqual(calls["sweep"], ["dow_vllm_cp1"])

    def test_lock_held_refuses_without_bringup(self):
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)
        calls = self._instrument(o, lambda s: (False, "missing"),
                                 lambda spec, grid: self.fail("must not sweep"))
        o._acquire_slot = lambda rr: None   # another launcher holds the slot
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_LOCK_HELD)
        self.assertEqual(calls["bringup"], [])

    def test_bringup_failure_is_a_hard_stop(self):
        specs = [_spec("dow_dynamo_disagg_cp1"), _spec("dow_vllm_cp1")]
        o = self._orch(specs)
        calls = self._instrument(o, lambda s: (False, "missing"),
                                 lambda spec, grid: {"status": "ok"})

        def boom(system, rep):
            raise cdow.BringUpFailed(f"{system}: exhausted")
        o._bring_up_fn = boom
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_BRINGUP)
        self.assertEqual(calls["sweep"], [])   # aborted before any sweep


if __name__ == "__main__":
    unittest.main()
