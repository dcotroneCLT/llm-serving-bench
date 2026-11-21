"""R4: W1 scope soak cells and their acceptance check (analysis/soak_check.py).

Run artefacts are produced by the REAL ContainerSampler (inventory/GPU probes
mocked as in tests/test_container_scope.py), so the CSV rows and the
scope_membership.jsonl audit have exactly the production shape.

Run: python3 -m unittest tests.test_soak
"""
import contextlib
import copy
import io
import json
import sys
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "campaigns" / "extension"
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "analysis"))

from test_container_scope import ScopeFixture, process, spec, write_csv  # noqa: E402
from test_scope_regressions import WholeRunFixture  # noqa: E402
import generate_dow_cells as gdc  # noqa: E402
import campaign as camp  # noqa: E402
import soak_check  # noqa: E402


class SoakCells(unittest.TestCase):
    def test_soak_cells_are_templates_on_the_p16_shape(self):
        p16 = {s: yaml.safe_load((EXT / "cells" / "dow" / f"dow_{s}_p16.yaml").read_text())
               for s in ("triton", "vllm")}
        for system in ("triton", "vllm"):
            soak = yaml.safe_load((EXT / "cells" / "rerun" / f"soak_{system}.yaml").read_text())
            template = gdc._load_template(system)
            self.assertEqual(soak["cell_id"], f"soak_{system}")
            self.assertFalse(soak["cell_id"].startswith("dow_"))  # never joins the DoW batch
            self.assertEqual(soak["engine"], template["engine"])
            self.assertEqual(soak["monitors"], template["monitors"])
            comps = soak["monitors"]["components"]
            self.assertEqual((comps["scope_type"], comps["scope_version"]), ("container_tree", 2))
            workload = copy.deepcopy(soak["workload"]["client_config_overrides"])
            self.assertEqual(workload.pop("target_rate_rps"), 0.10)
            self.assertEqual(workload, p16[system]["workload"]["client_config_overrides"])
            self.assertEqual(soak["duration_s"], 6 * 3600)
            self.assertNotIn("calibration_fraction", soak)

    def test_soak_descriptor_needs_no_calibration(self):
        path = EXT / "dow_rerun_w1_soak.yaml"
        d = camp.load_campaign(path)
        schedule = camp.build_schedule(d, path)
        self.assertEqual([s.cell_id for s in schedule], ["soak_triton", "soak_vllm"])
        for s in schedule:
            self.assertIsNone(s.calibration_file)
            self.assertFalse(s.calibration_required)
            self.assertEqual(camp.validate_calibration(s), (True, "none (not required)"))
        self.assertEqual(d["campaign_id"], "extension_dow_rescoped")
        self.assertEqual(d["state_file"], "state/dow_rerun_w1_soak_state.json")


class SamplerRun(ScopeFixture):
    """A synthetic soak written from consecutive real sampler ticks."""

    def run_ticks(self, inventories, model=None, duration_s=21_600, actual_s=None):
        """Ticks spread evenly over duration_s (one sampling period apart), and a
        finalized manifest whose recorded wall time is actual_s (default: the
        declared duration plus a 60 s drain)."""
        start = 1000.0
        period = duration_s / len(inventories)
        actual_s = duration_s + 60 if actual_s is None else actual_s
        agg_rows, comp_rows = [], {}
        for i, entries in enumerate(inventories):
            ts = start + period * i
            pmap = {p["pid"]: p["proc"] for p in entries}
            from unittest import mock
            import container_scope as cs
            with mock.patch.object(self.sampler, "inventory", side_effect=[entries, entries]), \
                 mock.patch.object(cs, "gpu_compute_pids", return_value=model or {20}), \
                 mock.patch.object(cs.psutil, "Process", side_effect=lambda pid: pmap[pid]):
                rows, groups = self.sampler.tick(ts)
            agg = dict(groups["engine"], ts_unix=ts,
                       process_alive=groups["engine"]["n_components_complete"] == 3,
                       membership_complete=groups["engine"]["n_components_complete"] == 3)
            agg_rows.append(agg)
            for row in rows:
                comp_rows.setdefault(row["label"], []).append(row)
        write_csv(self.rd / "agg_vllm_000000.csv", agg_rows)
        for label, rows in comp_rows.items():
            write_csv(self.rd / f"{label}_000000.csv", rows)
        self.manifest = {
            "duration_s": duration_s, "started_at_unix": start,
            "ended_at_unix": start + actual_s, "duration_seconds_actual": actual_s,
            "monitors": {"components": spec(), "proc": {"period_s": period}}}
        self.write_manifest()

    def write_manifest(self):
        (self.rd / "manifest.json").write_text(json.dumps(self.manifest))

    def check(self, **kw):
        return soak_check.check_soak(self.rd, **kw)

    def test_stable_soak_passes(self):
        self.run_ticks([self.entries] * 20)
        ok, msgs, report = self.check()
        self.assertTrue(ok, msgs)
        self.assertEqual(report["identity_loss_events"], 0)
        self.assertEqual(report["incomplete_ticks"], 0)
        self.assertEqual(report["identities_removed"], 0)
        self.assertTrue(any("initial set" in m and "engine=2" in m for m in msgs), msgs)

    def test_new_helper_is_listed_but_accepted(self):
        grown = self.entries + [process(31, 10, "python late_helper", uss=5)]
        self.run_ticks([self.entries] * 5 + [grown] * 5)
        ok, msgs, report = self.check()
        self.assertTrue(ok, msgs)
        self.assertEqual(report["identities_added"], 1)
        self.assertTrue(any("+ 31@31.0" in m and "late_helper" in m for m in msgs), msgs)

    def test_lost_process_latches_and_fails(self):
        self.run_ticks([self.entries] * 5 + [self.entries[:-1]] * 5)
        ok, msgs, report = self.check()
        self.assertFalse(ok)
        self.assertGreater(report["identity_loss_events"], 0)
        self.assertGreater(report["incomplete_ticks"], 0)
        self.assertEqual(report["identities_removed"], 1)
        self.assertTrue(any("- 30@30.0" in m for m in msgs), msgs)

    def test_model_outside_engine_subtree_fails(self):
        self.run_ticks([self.entries] * 5)
        rows = soak_check._read_csvs(self.rd, "vllm_engine")
        for row in rows:
            row["pids"] = "21"            # the model PID 20 no longer in the engine set
        write_csv(self.rd / "vllm_engine_000000.csv", rows)
        ok, msgs, report = self.check()
        self.assertFalse(ok)
        self.assertEqual(report["model_outside_engine_ticks"], 5)

    def test_short_run_is_not_a_soak(self):
        self.run_ticks([self.entries] * 5, duration_s=1200)
        ok, msgs, _ = self.check()
        self.assertFalse(ok)
        self.assertTrue(ok is False and any("< 6.0 h" in m for m in msgs))
        ok, _, _ = self.check(min_hours=0.3)
        self.assertTrue(ok)

    def test_declared_duration_is_not_evidence(self):
        # Declared 6 h, but the run actually lasted 5 h 59 min 50 s.
        self.run_ticks([self.entries] * 20, actual_s=6 * 3600 - 10)
        ok, msgs, report = self.check()
        self.assertFalse(ok)
        self.assertAlmostEqual(report["duration_h"], (6 * 3600 - 10) / 3600)
        self.assertEqual(report["declared_duration_h"], 6.0)
        self.assertTrue(any("FAIL: effective duration 5.9972 h < 6.0 h" in m for m in msgs), msgs)

    def test_sampled_span_shorter_than_wall_time_fails(self):
        # Wall time is fine, but the scope aggregate stops after ~3 h.
        self.run_ticks([self.entries] * 20)
        rows = soak_check._read_csvs(self.rd, "agg_vllm")
        write_csv(self.rd / "agg_vllm_000000.csv", rows[:10])
        ok, msgs, report = self.check()
        self.assertFalse(ok)
        self.assertAlmostEqual(report["sampled_span_h"], 3.0)
        self.assertTrue(any("FAIL: effective duration 3.0000 h" in m for m in msgs), msgs)

    def test_sampling_gap_fails(self):
        self.run_ticks([self.entries] * 20)
        rows = soak_check._read_csvs(self.rd, "agg_vllm")
        write_csv(self.rd / "agg_vllm_000000.csv", rows[:8] + rows[12:])
        ok, msgs, _ = self.check()
        self.assertFalse(ok)
        self.assertTrue(any("sampling gap" in m for m in msgs), msgs)

    def test_unfinalized_run_fails(self):
        self.run_ticks([self.entries] * 20)
        del self.manifest["duration_seconds_actual"]
        self.write_manifest()
        ok, msgs, report = self.check()
        self.assertFalse(ok)
        self.assertIsNone(report["duration_h"])
        self.assertTrue(any("effective duration unknown" in m for m in msgs), msgs)

    def test_missing_audit_fails(self):
        self.run_ticks([self.entries] * 3)
        (self.rd / "scope_membership.jsonl").unlink()
        ok, msgs, _ = self.check()
        self.assertFalse(ok)

    def test_non_container_run_rejected(self):
        self.run_ticks([self.entries] * 3)
        (self.rd / "manifest.json").write_text(json.dumps({
            "duration_s": 21600, "monitors": {"components": {"engine_group": "dynamo"}}}))
        ok, _, _ = self.check()
        self.assertFalse(ok)


class EndToEnd(WholeRunFixture):
    """soak_check main(): scope-only validator + soak acceptance on a complete run."""

    def _main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = soak_check.main(["--run-dir", str(self.rd), *args])
        return rc, out.getvalue()

    def test_complete_run_passes_with_validator(self):
        json_path = self.rd / "soak_report.json"
        rc, out = self._main("--min-hours", "0.3", "--json", str(json_path))
        self.assertEqual(rc, 0, out)
        self.assertIn("SCOPE GATE: PASS", out)
        report = json.loads(json_path.read_text())
        self.assertTrue(report["pass"])
        self.assertEqual(report["results"], {"scope_only_validator": True, "soak": True})

    def test_short_actual_run_fails_even_when_the_validator_passes(self):
        # Reviewer repro: the declared window is met to the generic 95% gate, the
        # effective duration is 10 s short of the soak minimum.
        self.manifest.update(duration_seconds_actual=self.duration - 10,
                             ended_at_unix=self.start + self.duration - 10)
        self.write_manifest()
        rc, out = self._main("--min-hours", str(self.duration / 3600))
        self.assertIn("SCOPE GATE: PASS", out)
        self.assertIn("FAIL: effective duration", out)
        self.assertEqual(rc, 1)
        self.assertIn("SOAK: FAIL", out)

    def test_validator_failure_fails_the_soak(self):
        self.manifest["interrupted_early"] = True
        self.write_manifest()
        rc, out = self._main("--min-hours", "0.3")
        self.assertEqual(rc, 1)
        self.assertIn("SOAK: FAIL", out)
        rc, _ = self._main("--min-hours", "0.3", "--skip-validator")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
