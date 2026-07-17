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


class ClassifyEngineFailure(unittest.TestCase):
    """The 2026-07-06 finding in one function: a dead endpoint (n_ok==0 at the
    lowest grid rate) must classify as engine_failure, never no_stable_point; a
    HEALTHY-but-unstable sweep must still be no_stable_point."""

    def _row(self, offered, n_offered, n_ok, drop=0.0):
        return {"offered_rate": offered, "n_offered": n_offered, "n_ok": n_ok,
                "drop_rate": drop}

    def test_dead_endpoint_lowest_rate_is_engine_failure(self):
        # The exact field row: 0.25 rps, 159 offered, 0 ok, 0.60 dropped, p99 NaN.
        res = {"status": "no_stable_point",
               "sweep_rows": [self._row(0.25, 159, 0, drop=0.60)]}
        reason = cdow.classify_engine_failure(res, health_reason=None)
        self.assertIsNotNone(reason)
        self.assertIn("dead endpoint", reason)

    def test_lowest_rate_picked_by_offered_not_list_order(self):
        # A higher rate listed first must not mask the dead lowest rate.
        res = {"status": "no_stable_point",
               "sweep_rows": [self._row(1.0, 50, 5), self._row(0.25, 159, 0)]}
        self.assertIsNotNone(cdow.classify_engine_failure(res, None))

    def test_healthy_unstable_is_not_engine_failure(self):
        # Lowest rate completed SOME requests but the sweep found no stable
        # prefix -> a real no_stable_point, NOT an engine failure.
        res = {"status": "no_stable_point",
               "sweep_rows": [self._row(0.25, 100, 90, drop=0.10),
                              self._row(0.5, 100, 60, drop=0.40)]}
        self.assertIsNone(cdow.classify_engine_failure(res, health_reason=None))

    def test_ok_sweep_is_never_engine_failure(self):
        # An ok verdict implies a full stable prefix -> the engine served; even a
        # (spurious) post-sweep health blip must not override it.
        res = {"status": "ok", "sweep_rows": [self._row(0.25, 100, 100)]}
        self.assertIsNone(cdow.classify_engine_failure(res, health_reason="blip"))

    def test_health_check_failure_on_non_ok_is_engine_failure(self):
        res = {"status": "no_stable_point",
               "sweep_rows": [self._row(0.25, 100, 90)]}  # rows look alive...
        reason = cdow.classify_engine_failure(res, health_reason="stack container down")
        self.assertIsNotNone(reason)                       # ...but the stack died
        self.assertIn("health check", reason)

    def test_no_rows_no_health_is_not_engine_failure(self):
        self.assertIsNone(cdow.classify_engine_failure(
            {"status": "no_output", "sweep_rows": []}, health_reason=None))


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
        # Hermetic: no real signal handlers, no real host preconditions.
        o._install_signals = False
        o._preflight_fn = lambda: None
        return o

    def _instrument(self, o, statuses, sweep_result, health=None):
        # health(spec, ef_retries) -> reason|None; default: always healthy.
        calls = {"bringup": [], "teardown": [], "sweep": [], "evidence": []}
        o._acquire_slot = lambda rr: _Lock()
        o._status_fn = lambda spec: statuses(spec)
        o._bring_up_fn = lambda system, rep: (
            calls["bringup"].append(system)
            or types.SimpleNamespace(system=system, lifecycle=None,
                                     work_dir=Path("/tmp")))
        o._sweep_fn = lambda spec, eng, grid, **kw: (
            calls["sweep"].append((spec.cell_id, kw.get("engine_failure_retries", 0)))
            or sweep_result(spec, grid, kw.get("engine_failure_retries", 0)))
        o._teardown_fn = lambda eng: calls["teardown"].append(eng.system)
        o._health_fn = (lambda eng: None) if health is None else (
            lambda eng: health(eng, len(calls["sweep"]) - 1))
        o._capture_evidence_fn = lambda eng, spec, reason, ef: calls["evidence"].append(
            (spec.cell_id, ef))
        # Do not touch the real filesystem when stamping engine_failure onto the
        # (non-existent) calibration file.
        o._stamp_engine_failure_file = lambda spec, reason, raw: None
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

        def sweep_result(spec, grid, ef_retries):
            if spec.cell_id == bad:
                return {"status": "did_not_saturate", "ceiling_rps": 8.0,
                        "rate_calibrated_rps": None, "rc": 4, "grid": list(grid),
                        "skipped": False, "sweep_rows": [],
                        "suggested_grid": cdow.suggest_next_grid("did_not_saturate", grid)}
            return {"status": "ok", "ceiling_rps": 3.0, "rate_calibrated_rps": 2.55,
                    "rc": 0, "grid": list(grid), "skipped": False, "sweep_rows": []}

        calls = self._instrument(o, lambda s: (False, "missing"), sweep_result)
        rc = o.run()

        # Every cell was swept (the bad one did NOT halt the rest).
        swept = [cid for cid, _ in calls["sweep"]]
        self.assertEqual(len(swept), 6)
        self.assertIn(bad, swept)
        # FRESH stack per sweep: one bring-up + teardown PER CELL, in system order.
        self.assertEqual(calls["bringup"],
                         ["dynamo_disagg", "dynamo_disagg", "triton", "triton",
                          "vllm", "vllm"])
        self.assertEqual(calls["teardown"], calls["bringup"])
        # A did_not_saturate (healthy, no dead-endpoint rows) is NOT an engine
        # failure -- no evidence captured, no retry.
        self.assertEqual(calls["evidence"], [])
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
            lambda spec, grid, ef=0: {"status": "ok", "ceiling_rps": 3.0,
                                      "rate_calibrated_rps": 2.55, "rc": 0,
                                      "grid": list(grid), "skipped": False,
                                      "sweep_rows": []})
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_OK)
        self.assertEqual(len(calls["sweep"]), 2)
        # FRESH stack per sweep: a bring-up (+teardown) per cell, not per system.
        self.assertEqual(calls["bringup"], ["vllm", "vllm"])
        self.assertEqual(calls["teardown"], ["vllm", "vllm"])

    def test_all_valid_skips_bringup_entirely(self):
        specs = [_spec("dow_vllm_cp1"), _spec("dow_vllm_p11")]
        o = self._orch(specs)
        calls = self._instrument(
            o, lambda s: (True, "ok (cached)"),
            lambda spec, grid, ef=0: self.fail("sweep must not run when all valid"))
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
            lambda spec, grid, ef=0: {"status": "ok", "ceiling_rps": 3.0,
                                      "rate_calibrated_rps": 2.55, "rc": 0,
                                      "grid": list(grid), "skipped": False,
                                      "sweep_rows": []})
        o.run()
        self.assertEqual([cid for cid, _ in calls["sweep"]], ["dow_vllm_cp1"])

    def test_lock_held_refuses_without_bringup(self):
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)
        calls = self._instrument(o, lambda s: (False, "missing"),
                                 lambda spec, grid, ef=0: self.fail("must not sweep"))
        o._acquire_slot = lambda rr: None   # another launcher holds the slot
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_LOCK_HELD)
        self.assertEqual(calls["bringup"], [])

    def test_bringup_failure_is_a_hard_stop(self):
        specs = [_spec("dow_dynamo_disagg_cp1"), _spec("dow_vllm_cp1")]
        o = self._orch(specs)
        calls = self._instrument(o, lambda s: (False, "missing"),
                                 lambda spec, grid, ef=0: {"status": "ok"})

        def boom(system, rep):
            raise cdow.BringUpFailed(f"{system}: exhausted")
        o._bring_up_fn = boom
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_BRINGUP)
        self.assertEqual(calls["sweep"], [])   # aborted before any sweep

    def test_preflight_abort_refuses_before_bringup(self):
        # A residual active run / unkillable orphan must refuse the whole thing
        # BEFORE any engine is brought up (finding: reap must run under the lock).
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)
        calls = self._instrument(o, lambda s: (False, "missing"),
                                 lambda spec, grid, ef=0: self.fail("must not sweep"))

        def refuse():
            raise cdow.PreflightAbort(cdow.EXIT_PRECONDITION, "prior run still active")
        o._preflight_fn = refuse
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_PRECONDITION)
        self.assertEqual(calls["bringup"], [])

    def _dead_row(self):
        # The field signature: lowest rate, many offered, zero completed.
        return [{"offered_rate": 0.25, "n_offered": 159, "n_ok": 0,
                 "drop_rate": 0.60}]

    def test_dead_endpoint_retries_on_fresh_stack_then_records_engine_failure(self):
        # A cell whose stack is dead on BOTH the first sweep and the retry:
        # classified engine_failure (NEVER no_stable_point), evidence captured on
        # each attempt, torn down + freshly brought up between them, moves on.
        specs = [_spec("dow_dynamo_disagg_cp1")]
        o = self._orch(specs)

        def sweep_result(spec, grid, ef_retries):
            return {"status": "no_stable_point", "ceiling_rps": None,
                    "rate_calibrated_rps": None, "rc": 3, "grid": list(grid),
                    "skipped": False, "sweep_rows": self._dead_row(),
                    "engine_failure_retries": ef_retries,
                    "suggested_grid": cdow.suggest_next_grid("no_stable_point", grid)}

        calls = self._instrument(o, lambda s: (False, "missing"), sweep_result)
        rc = o.run()

        self.assertEqual(rc, cdow.EXIT_SOME_FAILED)
        # 1 initial + 1 fresh-stack retry (default engine_failure_retries=1).
        self.assertEqual([cid for cid, _ in calls["sweep"]],
                         ["dow_dynamo_disagg_cp1", "dow_dynamo_disagg_cp1"])
        # The retry ran on a FRESH stack: a bring-up + teardown per attempt.
        self.assertEqual(calls["bringup"], ["dynamo_disagg", "dynamo_disagg"])
        self.assertEqual(calls["teardown"], ["dynamo_disagg", "dynamo_disagg"])
        # Evidence captured on BOTH attempts, tagged with the retry index.
        self.assertEqual(calls["evidence"],
                         [("dow_dynamo_disagg_cp1", 0), ("dow_dynamo_disagg_cp1", 1)])
        # The retry sweep saw the incremented retry counter (provenance).
        self.assertEqual([ef for _, ef in calls["sweep"]], [0, 1])
        # Recorded as engine_failure, NEVER no_stable_point; no misleading grid.
        r = o._last_results["dow_dynamo_disagg_cp1"]
        self.assertEqual(r["status"], "engine_failure")
        self.assertEqual(r["raw_status"], "no_stable_point")
        self.assertNotIn("suggested_grid", r)
        log = o._logf.getvalue()
        self.assertIn("ENGINE FAILURE", log)

    def test_dead_endpoint_then_healthy_retry_succeeds(self):
        # First sweep dead, retry lands on a healthy fresh stack -> final status
        # ok, exactly one retry, evidence captured once (for the failed attempt).
        specs = [_spec("dow_dynamo_disagg_cp1")]
        o = self._orch(specs)

        def sweep_result(spec, grid, ef_retries):
            if ef_retries == 0:
                return {"status": "no_stable_point", "ceiling_rps": None,
                        "rate_calibrated_rps": None, "rc": 3, "grid": list(grid),
                        "skipped": False, "sweep_rows": self._dead_row()}
            return {"status": "ok", "ceiling_rps": 3.0, "rate_calibrated_rps": 2.55,
                    "rc": 0, "grid": list(grid), "skipped": False,
                    "sweep_rows": [{"offered_rate": 0.25, "n_offered": 100, "n_ok": 100}]}

        calls = self._instrument(o, lambda s: (False, "missing"), sweep_result)
        rc = o.run()

        self.assertEqual(rc, cdow.EXIT_OK)
        self.assertEqual(calls["bringup"], ["dynamo_disagg", "dynamo_disagg"])
        self.assertEqual(calls["evidence"], [("dow_dynamo_disagg_cp1", 0)])
        self.assertEqual(o._last_results["dow_dynamo_disagg_cp1"]["status"], "ok")

    def test_healthy_unstable_stays_no_stable_point_no_retry(self):
        # A HEALTHY sweep that found no stable point: recorded no_stable_point,
        # brought up ONCE (no engine-failure retry), no evidence captured.
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)

        def sweep_result(spec, grid, ef_retries):
            return {"status": "no_stable_point", "ceiling_rps": None,
                    "rate_calibrated_rps": None, "rc": 3, "grid": list(grid),
                    "skipped": False,
                    "sweep_rows": [{"offered_rate": 0.25, "n_offered": 100,
                                    "n_ok": 88, "drop_rate": 0.12}],
                    "suggested_grid": cdow.suggest_next_grid("no_stable_point", grid)}

        calls = self._instrument(o, lambda s: (False, "missing"), sweep_result)
        rc = o.run()

        self.assertEqual(rc, cdow.EXIT_SOME_FAILED)
        self.assertEqual([cid for cid, _ in calls["sweep"]], ["dow_vllm_cp1"])
        self.assertEqual(calls["bringup"], ["vllm"])          # no retry
        self.assertEqual(calls["evidence"], [])
        self.assertEqual(o._last_results["dow_vllm_cp1"]["status"], "no_stable_point")

    def test_health_failure_reclassifies_as_engine_failure(self):
        # Rows look alive, but the post-sweep health check reports the stack down:
        # engine_failure, retried on a fresh stack.
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)

        def sweep_result(spec, grid, ef_retries):
            return {"status": "no_stable_point", "ceiling_rps": None,
                    "rate_calibrated_rps": None, "rc": 3, "grid": list(grid),
                    "skipped": False,
                    "sweep_rows": [{"offered_rate": 0.25, "n_offered": 100,
                                    "n_ok": 80}]}

        calls = self._instrument(o, lambda s: (False, "missing"), sweep_result,
                                 health=lambda eng, i: "stack container(s) not running")
        rc = o.run()
        self.assertEqual(rc, cdow.EXIT_SOME_FAILED)
        self.assertEqual(calls["bringup"], ["vllm", "vllm"])   # retried fresh
        self.assertEqual(o._last_results["dow_vllm_cp1"]["status"], "engine_failure")

    def test_signal_tears_down_active_engine_and_releases_lock(self):
        # A kill mid-run must teardown the live engine, run the abort-cleanup
        # backstop, close the slot, and exit -- not leave containers alive.
        specs = [_spec("dow_vllm_cp1")]
        o = self._orch(specs)
        torn, cleaned, closed = [], [], []
        o._teardown_fn = lambda eng: torn.append(eng.system)
        o._abort_cleanup = lambda: cleaned.append(True)
        o._slot = types.SimpleNamespace(close=lambda: closed.append(True))
        o._active_engine = types.SimpleNamespace(system="vllm")
        with self.assertRaises(SystemExit) as ctx:
            o._handle_signal(15, None)
        self.assertEqual(ctx.exception.code, cdow.EXIT_INTERRUPTED)
        self.assertEqual(torn, ["vllm"])
        self.assertEqual(cleaned, [True])   # backstop ran (covers partial bring-up)
        self.assertEqual(closed, [True])    # lock released


class PublishCalibration(unittest.TestCase):
    """Atomic publish / stale invalidation -- a crashed sweep must never leave a
    prior ok JSON masquerading as this sweep's success."""

    def test_fresh_result_is_published_over_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 1.0}))
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 9.9}))
            res = cdow.publish_calibration(tmp_out, out)
            self.assertEqual(res["rate_calibrated_rps"], 9.9)
            self.assertFalse(tmp_out.exists())                       # consumed
            self.assertEqual(json.loads(out.read_text())["rate_calibrated_rps"], 9.9)

    def test_no_new_output_invalidates_stale_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 1.0}))
            tmp_out = out.with_name(out.name + ".tmp")   # not written (crash)
            res = cdow.publish_calibration(tmp_out, out)
            self.assertIsNone(res)
            self.assertFalse(out.exists())   # stale ok removed -> reads as MISSING

    def test_corrupt_temp_is_discarded_and_stale_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            out.write_text(json.dumps({"status": "ok"}))
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text("{ not json")
            res = cdow.publish_calibration(tmp_out, out)
            self.assertIsNone(res)
            self.assertFalse(tmp_out.exists())
            self.assertFalse(out.exists())

    def test_rc_inconsistent_with_ok_status_is_purged(self):
        # An 'ok' JSON left behind by a process that then died (rc=-9 SIGKILL)
        # must NOT be published: the abnormal exit contradicts the ok verdict.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 1.0}))
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 9.9}))
            res = cdow.publish_calibration(tmp_out, out, rc=-9)
            self.assertIsNone(res)
            self.assertFalse(tmp_out.exists())
            self.assertFalse(out.exists())   # stale removed too -> reads as MISSING

    def test_rc_consistent_ok_is_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 2.5}))
            res = cdow.publish_calibration(tmp_out, out, rc=0)  # ok -> exit 0
            self.assertEqual(res["rate_calibrated_rps"], 2.5)

    def test_rc_consistent_non_ok_verdict_is_published(self):
        # did_not_saturate legitimately exits 4; rc matching the status is fine and
        # the (non-ok) result is still published for the campaign to act on.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text(json.dumps({"status": "did_not_saturate"}))
            res = cdow.publish_calibration(tmp_out, out, rc=4)
            self.assertEqual(res["status"], "did_not_saturate")

    def test_orchestration_provenance_block_is_recorded(self):
        # Fresh-per-sweep provenance (stack age at sweep start + engine-failure
        # retry count) is merged into the published JSON, on disk and in-memory.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal.json"
            tmp_out = out.with_name(out.name + ".tmp")
            tmp_out.write_text(json.dumps({"status": "ok", "rate_calibrated_rps": 2.5}))
            orch = {"stack_age_at_sweep_start_s": 0.0, "engine_failure_retries": 1,
                    "fresh_stack_per_sweep": True}
            res = cdow.publish_calibration(tmp_out, out, rc=0, orchestration=orch)
            self.assertEqual(res["orchestration"], orch)
            on_disk = json.loads(out.read_text())
            self.assertEqual(on_disk["orchestration"]["stack_age_at_sweep_start_s"], 0.0)
            self.assertEqual(on_disk["orchestration"]["engine_failure_retries"], 1)
            self.assertFalse(tmp_out.exists())


if __name__ == "__main__":
    unittest.main()
