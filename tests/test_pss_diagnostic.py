"""R6: PSS total trend next to USS for every aggregate, warning on PSS - USS growth.

The diagnostic must (a) cover the canonical aggregate and every monitored
component, (b) flag only significant AND material growth of the PSS - USS gap,
(c) never fail a gate, and (d) leave the BH-FDR family and the --csv rows of
aging_trends unchanged.

Run: python3 -m unittest tests.test_pss_diagnostic
"""
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "analysis"))

from test_container_scope import write_csv  # noqa: E402
from test_scope_regressions import WholeRunFixture  # noqa: E402
import aging_trends  # noqa: E402
import validate_extension_run as validator  # noqa: E402


def frame(gap_slope_bytes_per_h, uss_slope=0.0, n=600, noise=2e5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60.0                       # 60-s bins, hours
    uss = 5e9 + uss_slope * t + rng.normal(0, noise, n)
    gap = 3e8 + gap_slope_bytes_per_h * t + rng.normal(0, noise, n)
    df = pd.DataFrame({"uss_bytes": uss, "pss_bytes": uss + gap, "_t_hours": t})
    return aging_trends.add_pss_gap(df)


class Diagnostic(unittest.TestCase):
    def test_growing_gap_is_flagged(self):
        d = aging_trends.pss_uss_diagnostic("proc", frame(5e6))
        self.assertTrue(d["flag_shared_growth"])
        self.assertAlmostEqual(d["pss_minus_uss_slope_bytes_per_h"] / 1e6, 5.0, delta=0.2)
        self.assertIn("WARNING", aging_trends.format_pss_diagnostic(d))

    def test_flat_gap_with_uss_growth_is_not_flagged(self):
        d = aging_trends.pss_uss_diagnostic("proc", frame(0.0, uss_slope=4e8))
        self.assertFalse(d["flag_shared_growth"])
        self.assertAlmostEqual(d["uss_slope_bytes_per_h"] / 1e6, 400, delta=2)
        self.assertAlmostEqual(d["pss_slope_bytes_per_h"] / 1e6, 400, delta=2)

    def test_significant_but_immaterial_growth_is_not_flagged(self):
        d = aging_trends.pss_uss_diagnostic("proc", frame(2e5, noise=1e3))
        self.assertLess(d["pss_minus_uss_mk_p"], 0.01)
        self.assertGreater(d["pss_minus_uss_ci_low"], 0)
        self.assertFalse(d["flag_shared_growth"])

    def test_missing_pss_column_is_skipped(self):
        df = pd.DataFrame({"uss_bytes": [1.0] * 20, "_t_hours": np.arange(20) / 60})
        self.assertIsNone(aging_trends.pss_uss_diagnostic("proc", aging_trends.add_pss_gap(df)))


class SharedGrowthRun(WholeRunFixture):
    """A complete whole-container run whose engine PSS grows 1 GB/h at flat USS."""

    def setUp(self):
        super().setUp()
        for row in self.aggregates:
            dt_h = (row["ts_unix"] - self.start) / 3600
            row["pss_bytes"] = row["pss_bytes"] + int(1e9 * dt_h)
        self.write_aggregate()
        path = self.rd / "vllm_engine_000000.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            dt_h = (float(row["ts_unix"]) - self.start) / 3600
            row["pss_bytes"] = int(float(row["pss_bytes"]) + 1e9 * dt_h)
        write_csv(path, rows)

    def aging(self, *extra):
        return subprocess.run([sys.executable, str(REPO / "analysis/aging_trends.py"),
                               "--run-dir", str(self.rd), "--csv", *extra],
                              capture_output=True, text=True, timeout=300)

    def test_every_aggregate_reported_and_engine_flagged(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "pss.json"
            res = self.aging("--pss-json", str(out))
            self.assertEqual(res.returncode, 0, res.stderr)
            diags = {d["aggregate"]: d for d in json.loads(out.read_text())}
        self.assertEqual(set(diags), {"proc", "component.orchestration", "component.engine",
                                      "component.auxiliary"})
        self.assertTrue(diags["proc"]["flag_shared_growth"])
        self.assertTrue(diags["component.engine"]["flag_shared_growth"])
        self.assertFalse(diags["component.auxiliary"]["flag_shared_growth"])
        self.assertIn("[pss-vs-uss]", res.stderr)

    def test_csv_rows_and_fdr_family_unchanged(self):
        plain = self.aging()
        with tempfile.TemporaryDirectory() as t:
            with_json = self.aging("--pss-json", str(Path(t) / "p.json"))
        self.assertEqual(plain.stdout, with_json.stdout)
        indicators = [r["indicator"] for r in csv.DictReader(io.StringIO(plain.stdout))]
        self.assertFalse(any("pss_minus_uss" in i for i in indicators), indicators)
        self.assertIn("proc.pss_bytes", indicators)
        self.assertIn("component.engine.pss_bytes", indicators)

    def test_validator_warns_but_does_not_fail(self):
        ok, msgs = validator.check_analysis(self.rd, REPO)
        self.assertTrue(ok, msgs)
        text = "\n".join(msgs)
        self.assertIn("PSS vs USS per aggregate", text)
        self.assertIn("WARN: proc: PSS - USS grows significantly", text)
        self.assertNotIn("FAIL", text)

    def test_human_table_shows_the_section(self):
        res = subprocess.run([sys.executable, str(REPO / "analysis/aging_trends.py"),
                              "--run-dir", str(self.rd)], capture_output=True, text=True, timeout=300)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("PSS vs USS per aggregate", res.stdout)
        self.assertIn("shared-memory growth invisible to USS", res.stdout)


if __name__ == "__main__":
    unittest.main()
