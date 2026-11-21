"""R7: burstiness check = realized vs theoretical interarrival CoV.

The fixed CoV >= 1.5 rule is replaced by the theory of the client's own arrival
processes (analysis/arrival_theory.py) for the RECORDED rate, burst_factor and
burst_on_seconds, with a tolerance band. Required: PASS the real Dynamo p11
case (0.046 rps realized, CoV 1.46) and FAIL a generator that emits Poisson
when bursty was requested.

The analytic theory is cross-checked against seeded simulations of the ACTUAL
client classes (client/benchmark.py BurstyArrival / PoissonArrival).

Run: python3 -m unittest tests.test_arrival_theory
"""
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "client"))
sys.path.insert(0, str(REPO))

import arrival_theory as at  # noqa: E402
import validate_extension_run as ver  # noqa: E402
from client.benchmark import BurstyArrival, ConstantArrival, PoissonArrival  # noqa: E402

# client/arrival_stats.json of extension_dow_screening_dow_dynamo_disagg_p11_r01
# (off-box validation 2026-11-07; failed the old fixed threshold).
P11 = {
    "arrival_mode": "bursty", "target_rate_rps": 0.0477, "realized_count": 6018,
    "realized_duration_s": 129603.25789174018, "realized_rate_rps": 0.04643401792435602,
    "interarrival_mean_s": 21.539514357942526, "interarrival_cv": 1.4604721193724923,
    "burst_factor": 4.0, "burst_on_seconds": 5.0,
}


def simulate(process, n):
    gaps = [process.next_interval() for _ in range(n)]
    mean = sum(gaps) / n
    var = sum((g - mean) ** 2 for g in gaps) / n
    return math.sqrt(var) / mean, n / sum(gaps)


def stats_from(process, n, mode, rate, bf=4.0, on=5.0):
    cv, _ = simulate(process, n)
    out = {"arrival_mode": mode, "target_rate_rps": rate, "realized_count": n + 1,
           "interarrival_cv": cv}
    if mode == "bursty":
        out.update(burst_factor=bf, burst_on_seconds=on)
    return out


class Theory(unittest.TestCase):
    def test_closed_form_scv(self):
        # IPP: SCV = 1 + 2*lam*alpha/(alpha+beta)^2.
        for rate, bf, on in ((0.0477, 4.0, 5.0), (0.2055, 2.5, 5.0), (2.9, 4.0, 5.0), (1.0, 2.0, 10.0)):
            lam, alpha, beta = rate * bf, 1 / on, 1 / (on * (bf - 1))
            expected = math.sqrt(1 + 2 * lam * alpha / (alpha + beta) ** 2)
            self.assertAlmostEqual(at.theoretical_cv("bursty", rate, bf, on)[0], expected, places=10)
        self.assertAlmostEqual(at.theoretical_cv("poisson", 3.0)[0], 1.0, places=12)
        self.assertAlmostEqual(at.theoretical_cv("bursty", 3.0, 1.0)[0], 1.0, places=12)

    def test_mean_rate_is_preserved(self):
        for rate in (0.05, 0.5, 5.0):
            self.assertAlmostEqual(at.interarrival_moments("bursty", rate, 4.0, 5.0)[0], 1 / rate)

    def test_poisson_se_is_one_over_sqrt_n(self):
        _, se = at.theoretical_cv("poisson", 1.0, n_gaps=10_000)
        self.assertAlmostEqual(se, 0.01, places=6)

    def test_matches_seeded_simulation_of_the_client_generator(self):
        for rate, bf in ((0.0477, 4.0), (0.2055, 2.5), (0.5, 4.0)):
            n = 40_000
            cv_sim, _ = simulate(BurstyArrival(rate, random.Random(7), bf, 5.0), n)
            cv_th, se = at.theoretical_cv("bursty", rate, bf, 5.0, n)
            self.assertLess(abs(cv_sim - cv_th), 4 * se, (rate, bf, cv_sim, cv_th, se))


class Check(unittest.TestCase):
    def test_real_dynamo_p11_passes(self):
        verdict, msg = at.check_realized_cv(P11)
        self.assertEqual(verdict, "pass", msg)
        self.assertIn("theoretical 1.44", msg)

    def test_poisson_generator_for_bursty_request_fails(self):
        # Same rate and length as p11, but the generator ignored the burst mode.
        n = P11["realized_count"] - 1
        stats = stats_from(PoissonArrival(P11["target_rate_rps"], random.Random(11)), n,
                           "bursty", P11["target_rate_rps"])
        verdict, msg = at.check_realized_cv(stats)
        self.assertEqual(verdict, "fail", msg)

    def test_bursty_generator_for_poisson_request_fails(self):
        n = 20_000
        stats = stats_from(BurstyArrival(0.2, random.Random(3), 4.0, 5.0), n, "poisson", 0.2)
        self.assertEqual(at.check_realized_cv(stats)[0], "fail")

    def test_correct_generators_pass_across_rates(self):
        for seed, rate in enumerate((0.03, 0.2, 2.9)):
            n = int(rate * 36 * 3600) if rate < 1 else 50_000
            ok_b = stats_from(BurstyArrival(rate, random.Random(seed), 4.0, 5.0), n, "bursty", rate)
            ok_p = stats_from(PoissonArrival(rate, random.Random(seed)), n, "poisson", rate)
            self.assertNotEqual(at.check_realized_cv(ok_b)[0], "fail", rate)
            self.assertEqual(at.check_realized_cv(ok_p)[0], "pass", rate)

    def test_indistinguishable_rate_is_low_power_not_fail(self):
        # 0.003 rps for 36 h: ~390 arrivals, a burst holds ~0.06 requests.
        n = 388
        stats = stats_from(BurstyArrival(0.003, random.Random(5), 4.0, 5.0), n, "bursty", 0.003)
        verdict, msg = at.check_realized_cv(stats)
        self.assertEqual(verdict, "low_power", msg)

    def test_incomplete_evidence_fails(self):
        for drop in ("realized_count", "interarrival_cv", "target_rate_rps"):
            stats = {k: v for k, v in P11.items() if k != drop}
            self.assertEqual(at.check_realized_cv(stats)[0], "fail", drop)
        self.assertEqual(at.check_realized_cv(dict(P11, interarrival_cv="nan"))[0], "fail")
        # The recorded config supplies a missing rate.
        stats = {k: v for k, v in P11.items() if k != "target_rate_rps"}
        self.assertEqual(at.check_realized_cv(stats, {"target_rate_rps": 0.0477})[0], "pass")


def dispatch_cv(process, n, jitter_s, seed=0):
    """CoV of gaps between DISPATCH timestamps: the client's scheduled times plus
    event-loop jitter (what client/benchmark.py actually records)."""
    rng = random.Random(seed)
    t, stamps = 0.0, []
    for _ in range(n + 1):
        stamps.append(t + abs(rng.gauss(0, jitter_s)))
        t += process.next_interval()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    mean = sum(gaps) / n
    return math.sqrt(sum((g - mean) ** 2 for g in gaps) / n) / mean


class ConstantMode(unittest.TestCase):
    def test_theory_is_zero(self):
        self.assertEqual(at.theoretical_cv("constant", 0.2, n_gaps=1000), (0.0, 0.0))
        self.assertEqual(at.interarrival_moments("constant", 0.5)[:2], [2.0, 4.0])

    def test_reviewer_case_cov_0002_passes(self):
        stats = {"arrival_mode": "constant", "target_rate_rps": 0.2, "realized_count": 25_921,
                 "interarrival_cv": 0.002}
        verdict, msg = at.check_realized_cv(stats)
        self.assertEqual(verdict, "pass", msg)
        self.assertIn("mode=constant", msg)

    def test_client_constant_generator_with_dispatch_jitter_passes(self):
        for rate, jitter in ((0.05, 0.002), (0.2, 0.01), (2.9, 0.01)):
            n = 20_000
            cv = dispatch_cv(ConstantArrival(rate), n, jitter)
            stats = {"arrival_mode": "constant", "target_rate_rps": rate,
                     "realized_count": n + 1, "interarrival_cv": cv}
            self.assertEqual(at.check_realized_cv(stats)[0], "pass", (rate, cv))

    def test_random_generators_labelled_constant_fail(self):
        for proc in (PoissonArrival(0.2, random.Random(1)), BurstyArrival(0.2, random.Random(2), 4.0, 5.0)):
            stats = stats_from(proc, 10_000, "constant", 0.2)
            verdict, msg = at.check_realized_cv(stats)
            self.assertEqual(verdict, "fail", msg)

    def test_constant_generator_labelled_poisson_fails(self):
        cv = dispatch_cv(ConstantArrival(0.2), 10_000, 0.01)
        stats = {"arrival_mode": "poisson", "target_rate_rps": 0.2, "realized_count": 10_001,
                 "interarrival_cv": cv}
        self.assertEqual(at.check_realized_cv(stats)[0], "fail")

    def test_unknown_mode_is_checked_as_the_poisson_the_client_ran(self):
        self.assertEqual(at.normalize_mode("Poisson"), "poisson")
        self.assertEqual(at.normalize_mode("uniform"), "poisson")
        self.assertEqual(at.normalize_mode(None), "poisson")
        self.assertEqual(at.normalize_mode("CONSTANT"), "constant")


class ValidatorIntegration(unittest.TestCase):
    def _run_dir(self, tmp, arrival):
        rd = Path(tmp)
        (rd / "client").mkdir()
        rows = ["shared_prefix_applied"] + ["True" if i % 5 < 4 else "False" for i in range(100)]
        (rd / "client" / "requests_000000.csv").write_text("\n".join(rows) + "\n")
        (rd / "client" / "arrival_stats.json").write_text(json.dumps(arrival))
        return rd

    def test_p11_passes_client_features(self):
        with tempfile.TemporaryDirectory() as t:
            ok, msgs = ver.check_client_features(self._run_dir(t, P11), 0.8, "bursty")
            self.assertTrue(ok, msgs)
            self.assertFalse(any("FAIL" in m for m in msgs), msgs)

    def test_constant_run_passes_client_features(self):
        stats = {"arrival_mode": "constant", "target_rate_rps": 0.2, "realized_count": 25_921,
                 "interarrival_cv": 0.002}
        with tempfile.TemporaryDirectory() as t:
            ok, msgs = ver.check_client_features(self._run_dir(t, stats), 0.8, "constant")
            self.assertTrue(ok, msgs)

    def test_fixed_threshold_is_gone(self):
        # CoV 1.46 < 1.5 used to fail; a CoV of 2.0 at p11's rate (way above
        # theory) must now fail instead.
        with tempfile.TemporaryDirectory() as t:
            ok, msgs = ver.check_client_features(
                self._run_dir(t, dict(P11, interarrival_cv=2.0)), 0.8, "bursty")
            self.assertFalse(ok)
            self.assertTrue(any("theoretical" in m for m in msgs), msgs)


@unittest.skipUnless((REPO / "runs").is_dir() and
                     list((REPO / "runs").glob("extension_dow_screening_*/client/arrival_stats.json")),
                     "local copies of the Dynamo runs not present")
class RealDynamoRuns(unittest.TestCase):
    def test_all_completed_dynamo_runs_pass(self):
        for path in sorted((REPO / "runs").glob("extension_dow_screening_*/client/arrival_stats.json")):
            verdict, msg = at.check_realized_cv(json.loads(path.read_text()))
            self.assertEqual(verdict, "pass", f"{path.parent.parent.name}: {msg}")


if __name__ == "__main__":
    unittest.main()
