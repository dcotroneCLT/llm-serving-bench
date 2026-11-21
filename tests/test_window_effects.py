"""R8: window offsets from the center points and DoW effects with/without the
CP window correction (analysis/window_effects.py).

Synthetic responses are generated on the REAL generated design
(rerun_design_matrix.csv + dow_design_matrix.csv) from a known model, so every
estimate has an exact expected value.

Run: python3 -m unittest tests.test_window_effects
"""
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "tests"))

import window_effects as we  # noqa: E402

DESIGN = we.load_design()
TRUE = {  # true effects (mean(+) - mean(-)) per system
    "triton": {"A": 2.0, "B": 9.0, "C": 1.0, "AC": 3.0, "AD": -2.0, "CD": 1.5, "BE": 0.7},
    "vllm": {"A": -1.0, "B": 6.0, "E": 1.5, "AC": -2.5, "AD": 4.0, "CD": 0.5, "AB": 0.9},
    "dynamo_disagg": {"B": 8.8, "C": 1.2, "E": 1.4},
}
MU = {"triton": 10.0, "vllm": 20.0, "dynamo_disagg": 9.0}
SHIFT = {1: 0.0, 2: 1.7, 3: -2.3, "D": 0.0}


def synth(shift=SHIFT, per_system_shift=None, bridge_shift=0.4):
    y = {}
    for cid, c in DESIGN.items():
        s = c["system"]
        v = MU[s] + sum(e / 2 * we.contrast(c["codes"], name) for name, e in TRUE[s].items()
                        if c["kind"] == "design")
        w = c["window"]
        v += (per_system_shift or {}).get((s, w), shift[w])
        y[cid] = v
    y.update({"dow_dynamo_disagg_cp1": 9.0, "dow_dynamo_disagg_cp2": 9.2,
              "dow_dynamo_disagg_cp3": 8.8})
    y[we.BRIDGE_CELL] = 9.0 + bridge_shift
    return y


class Design(unittest.TestCase):
    def test_design_covers_all_systems_and_windows(self):
        for s in ("triton", "vllm", "dynamo_disagg"):
            self.assertEqual(sum(1 for c in DESIGN.values()
                                 if c["system"] == s and c["kind"] == "design"), 16, s)
        self.assertEqual(DESIGN[we.BRIDGE_CELL]["window"], 3)
        self.assertEqual(DESIGN["dow_dynamo_disagg_cp1"]["window"], "D")
        self.assertEqual({DESIGN[f"dow_{s}_cp{k}"]["window"] for s in ("triton", "vllm")
                          for k in (1, 2, 3)}, {1, 2, 3})


class Offsets(unittest.TestCase):
    def test_common_and_per_system_offsets_recovered(self):
        r = we.analyze(DESIGN, synth())
        off = r["offsets"]
        for w in (1, 2, 3):
            self.assertAlmostEqual(off["common"][w], SHIFT[w])
            for s in ("triton", "vllm"):
                self.assertAlmostEqual(off["per_system"][s][w], SHIFT[w])
        self.assertTrue(r["consistency"]["consistent"])

    def test_inconsistent_systems_are_flagged(self):
        per = {("triton", 2): 1.7, ("vllm", 2): -1.7, ("triton", 3): 0.0, ("vllm", 3): 0.0}
        r = we.analyze(DESIGN, synth(per_system_shift=per))
        self.assertFalse(r["consistency"]["consistent"])
        self.assertAlmostEqual(r["consistency"]["difference_by_window"][2], 3.4)

    def test_one_missing_cp_is_incomplete_not_consistent(self):
        # Reviewer repro: without dow_vllm_cp3 only W2 is comparable; the check
        # must not print CONSISTENT and must not enable the common correction.
        y = synth()
        del y["dow_vllm_cp3"]
        r = we.analyze(DESIGN, y)
        c = r["consistency"]
        self.assertEqual(c["status"], "incomplete")
        self.assertFalse(c["consistent"])
        self.assertEqual(c["missing_cp"], ["vllm_W3"])
        self.assertFalse(r["corrected_available"])
        self.assertNotIn("corrected", r["systems"]["triton"])
        text = we.format_report(r, "m")
        self.assertNotIn("CONSISTENT", text.replace("INCONSISTENT", ""))
        self.assertIn("INCOMPLETE -- needs all six CPs, missing vllm_W3", text)
        self.assertIn("cannot be verified", text)
        # per_system cannot correct vLLM either (no W3 CP).
        self.assertFalse(we.analyze(DESIGN, y, correction="per_system")["corrected_available"])

    def test_inconsistent_shift_disables_common_but_not_per_system(self):
        per = {("triton", 2): 1.7, ("vllm", 2): -1.7, ("triton", 3): 0.0, ("vllm", 3): 0.0}
        y = synth(per_system_shift=per)
        r = we.analyze(DESIGN, y)
        self.assertEqual(r["consistency"]["status"], "inconsistent")
        self.assertFalse(r["corrected_available"])
        self.assertIn("per_system", r["correction_unavailable_reason"])
        self.assertTrue(we.analyze(DESIGN, y, correction="per_system")["corrected_available"])

    def test_missing_center_point_disables_correction_loudly(self):
        y = synth()
        del y["dow_triton_cp2"], y["dow_vllm_cp2"]
        r = we.analyze(DESIGN, y)
        self.assertFalse(r["corrected_available"])
        self.assertIn("triton_W2", r["offsets"]["missing_cp"])
        self.assertNotIn("corrected", r["systems"]["triton"])
        self.assertIn("NOT available", we.format_report(r, "m"))


class Effects(unittest.TestCase):
    def test_correction_recovers_window_exposed_interactions(self):
        r = we.analyze(DESIGN, synth())
        for s in ("triton", "vllm"):
            e = r["systems"][s]
            self.assertTrue(e["complete"])
            for name in we.EFFECTS:
                self.assertAlmostEqual(e["corrected"][name], TRUE[s].get(name, 0.0), places=9,
                                       msg=(s, name))
            # Uncorrected, the window shift leaks only into AC / AD / CD.
            leaked = {n for n in we.EFFECTS if abs(e["raw"][n] - TRUE[s].get(n, 0.0)) > 1e-9}
            self.assertTrue(leaked)
            self.assertLessEqual(leaked, set(we.WINDOW_EFFECTS), (s, leaked))
            self.assertLess(e["max_change_outside_window_effects"], 1e-9)

    def test_per_system_correction(self):
        per = {("triton", 2): 1.0, ("triton", 3): -1.0, ("vllm", 2): 3.0, ("vllm", 3): 0.5,
               ("triton", 1): 0.0, ("vllm", 1): 0.0}
        r = we.analyze(DESIGN, synth(per_system_shift=per), correction="per_system")
        for s in ("triton", "vllm"):
            for name in we.WINDOW_EFFECTS:
                self.assertAlmostEqual(r["systems"][s]["corrected"][name], TRUE[s].get(name, 0.0))

    def test_dynamo_effects_without_windows(self):
        e = we.analyze(DESIGN, synth())["systems"]["dynamo_disagg"]
        self.assertNotIn("corrected", e)
        for name in we.EFFECTS:
            self.assertAlmostEqual(e["raw"][name], TRUE["dynamo_disagg"].get(name, 0.0))

    def test_log_scale_multiplicative_shift(self):
        y = {c: math.exp(v / 10) for c, v in synth().items()}
        r = we.analyze(DESIGN, y, scale="log")
        self.assertAlmostEqual(r["offsets"]["common"][2], SHIFT[2] / 10)
        with self.assertRaises(ValueError):
            we.analyze(DESIGN, dict(y, dow_vllm_p01=-1.0), scale="log")

    def test_lenth_flags_only_large_effects(self):
        effects = {n: 0.1 * (i % 3 - 1) for i, n in enumerate(we.EFFECTS)}
        effects["B"] = 9.0
        pse, me = we.lenth(effects)
        self.assertGreater(9.0, me)
        self.assertTrue(all(abs(v) < me for n, v in effects.items() if n != "B"))

    def test_incomplete_design_reported(self):
        y = synth()
        del y["dow_vllm_p07"]
        e = we.analyze(DESIGN, y)["systems"]["vllm"]
        self.assertFalse(e["complete"])
        self.assertEqual(e["missing"], ["dow_vllm_p07"])


class StatedLimit(unittest.TestCase):
    """docs/rerun_windows.md: the blocking protects contrasts from window offsets
    that are ADDITIVE on the analysis scale only. A window x A interaction (a
    shift whose size depends on the level of A) leaks into the A main effect,
    and the CP correction cannot remove it: the CPs sit at A = 0."""

    def test_window_by_factor_interaction_biases_a_even_after_correction(self):
        g = {1: 0.0, 2: 1.0, 3: -0.5}
        y = synth()
        for cid, c in DESIGN.items():
            if c["system"] == "triton" and c["kind"] == "design":
                y[cid] += g[c["window"]] * we.contrast(c["codes"], "A")
        pts = [c for c in DESIGN.values() if c["system"] == "triton" and c["kind"] == "design"]
        expected_bias = 2 * sum(g[c["window"]] for c in pts) / len(pts)
        self.assertGreater(abs(expected_bias), 0.1)
        e = we.analyze(DESIGN, y)["systems"]["triton"]
        self.assertAlmostEqual(e["raw"]["A"] - TRUE["triton"]["A"], expected_bias)
        self.assertAlmostEqual(e["corrected"]["A"] - TRUE["triton"]["A"], expected_bias)


class Bridge(unittest.TestCase):
    def test_bridge_shift(self):
        b = we.analyze(DESIGN, synth(bridge_shift=0.4))["bridge"]
        self.assertAlmostEqual(b["cp1_cp3_mean"], 9.0)
        self.assertAlmostEqual(b["shift"], 0.4)
        self.assertAlmostEqual(b["cp1_cp3_sd"], 0.2)
        self.assertAlmostEqual(b["shift_in_sd"], 2.0)

    def test_bridge_absent_without_cp4(self):
        y = synth()
        del y[we.BRIDGE_CELL]
        self.assertIsNone(we.analyze(DESIGN, y)["bridge"])


class ResponsesAndCli(unittest.TestCase):
    def test_responses_csv_round_trip_and_cli(self):
        y = synth()
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "resp.csv"
            we.write_responses(path, y, "growth_kb_per_req")
            self.assertEqual(we.read_responses(path), y)
            out = io.StringIO()
            with redirect_stdout(out):
                rc = we.main(["--responses", str(path), "--json", str(Path(t) / "r.json")])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn("common offset: W1 +0.0000  W2 +1.7000  W3 -2.3000", text)
            self.assertIn("CONSISTENT", text)
            self.assertIn("(window-exposed)", text)
            report = json.loads((Path(t) / "r.json").read_text())
            self.assertEqual(report["n_responses"], len(y))

    def test_run_dir_naming(self):
        root = Path("/r")
        self.assertEqual(we.run_dir_for("dow_vllm_p01", root).name,
                         "extension_dow_rescoped_dow_vllm_p01_r01")
        self.assertEqual(we.run_dir_for("dow_dynamo_disagg_p01", root).name,
                         "extension_dow_screening_dow_dynamo_disagg_p01_r01")
        self.assertEqual(we.run_dir_for(we.BRIDGE_CELL, root).name,
                         "extension_dow_rescoped_dow_dynamo_disagg_cp4_r01")


class ComputeResponse(unittest.TestCase):
    """Exact growth per request from a synthetic run: USS +3.6 MB/h, 0.5 rps."""

    def test_growth_per_request(self):
        from test_container_scope import write_csv
        with tempfile.TemporaryDirectory() as t:
            rd = Path(t)
            start, dur, warm = 1_000_000.0, 6 * 3600, 3600
            (rd / "manifest.json").write_text(json.dumps({
                "started_at_unix": start, "duration_s": dur, "warmup_discard_s": warm,
                "proc_prefix": "agg_x"}))
            write_csv(rd / "agg_x_000000.csv", [
                {"ts_unix": start + t, "process_alive": True, "uss_bytes": 5e9 + 1000.0 * t}
                for t in range(0, dur, 5)])
            (rd / "client").mkdir()
            write_csv(rd / "client" / "requests_000000.csv", [
                {"submitted_at_unix": start + t, "status": "ok" if t % 4 else "error"}
                for t in range(0, dur, 1)][: dur])
            res = we.compute_response(rd)
            self.assertAlmostEqual(res["uss_slope_mb_h"], 3.6, places=6)
            self.assertAlmostEqual(res["served_rps"], 0.75, places=3)
            self.assertAlmostEqual(res["growth_kb_per_req"], 1000.0 / 0.75 / 1e3, places=3)


if __name__ == "__main__":
    unittest.main()
