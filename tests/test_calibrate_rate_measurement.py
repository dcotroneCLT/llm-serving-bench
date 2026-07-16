"""Measurement-bias fix for calibrate_rate.py's per-rate sweep.

The v1 method sized each rate step by a fixed short window and counted
completed-OK over the WHOLE window, so an unsaturated server whose e2e latency
was a non-trivial fraction of the window mechanically reported
achieved/offered < 1.0 -- requests still in flight at the window edge were never
counted. The real symptom: dow_dynamo_disagg_cp1 failed at the LOWEST grid rate
with achieved_ratio=0.83 (50/50 ok, zero drops, flat latency, p99=15s) purely
because the ~240s window was short relative to e2e; dow_vllm shapes were worse.

These tests drive a synthetic sweep simulator (a K-server FIFO with configurable
service time and arrival pattern -- no docker/GPU/network) to prove:
  (a) the old finite-window bias is reproduced, then eliminated -- an
      unsaturated server with e2e comparable to the OLD window now yields
      achieved_ratio ~ 1.0 under the v2 sub-window measurement;
  (b) a truly saturated server still fails the (unchanged) acceptance criteria;
  (c) bursty offered-rate accounting is phase-independent when the measurement
      sub-window is trimmed to whole burst cycles.

Run: python3 -m unittest tests.test_calibrate_rate_measurement
"""
import heapq
import statistics
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_rate as cr  # noqa: E402

T0 = 1_000_000.0  # arbitrary wall-clock origin for the synthetic timeline


# ---------------------------------------------------------------------------
# Synthetic sweep simulator
# ---------------------------------------------------------------------------


def _poisson_free_arrivals(rate, wall_s):
    """Evenly-spaced (constant-interval) arrivals over [T0, T0+wall_s)."""
    if rate <= 0:
        return []
    gap = 1.0 / rate
    n = int(rate * wall_s)
    return [T0 + i * gap for i in range(n)]


def _bursty_arrivals(rate, wall_s, burst_factor, burst_on_seconds, phase=0.0):
    """Deterministic MMPP-2-style on/off arrivals with the mean rate preserved.

    ON for burst_on_seconds at on_rate = rate*burst_factor, then OFF for
    burst_on_seconds*(burst_factor-1); one cycle therefore offers
    rate*cycle arrivals and the long-run mean is `rate`. `phase` slides the cycle
    boundaries relative to T0 so a test can probe different starting phases."""
    bf = max(1.0, burst_factor)
    on = burst_on_seconds
    off = on * (bf - 1.0)
    cycle = on + off
    on_rate = rate * bf
    on_gap = 1.0 / on_rate if on_rate > 0 else cycle
    arrivals = []
    # Start a cycle boundary `phase` seconds before T0 and march forward until
    # we have covered the whole window.
    cycle_start = T0 - phase
    while cycle_start < T0 + wall_s:
        t = cycle_start
        while t < cycle_start + on:
            if T0 <= t < T0 + wall_s:
                arrivals.append(t)
            t += on_gap
        cycle_start += cycle
    return sorted(arrivals)


def simulate_records(rate, wall_s, service_s, concurrency, *,
                     arrival="poisson", burst_factor=4.0, burst_on_seconds=10.0,
                     phase=0.0, collection_grace_s=0.0, offer_wall_s=None,
                     explicit_arrivals=None):
    """One rate step's per-request records from a K-server FIFO queue.

    Each arrival takes the earliest-free server; completion = max(arrival,
    free) + service. A request that completes by the collection cutoff (the
    client stops recording at T0+wall_s + collection_grace_s) is 'ok' with its
    measured e2e; one still in flight at the cutoff is 'error' (the client
    cancels it -- exactly how in-flight requests get recorded in the real
    client), with e2e=None. Returns dicts shaped like the client's CSV rows.

    offer_wall_s < wall_s models a STALLED load generator that stopped submitting
    early (only offers over [T0, T0+offer_wall_s)); explicit_arrivals overrides
    the generator entirely (e.g. the real client BurstyArrival)."""
    offer_wall_s = wall_s if offer_wall_s is None else offer_wall_s
    if explicit_arrivals is not None:
        arrivals = [a for a in explicit_arrivals if T0 <= a < T0 + offer_wall_s]
    elif arrival == "bursty":
        arrivals = [a for a in _bursty_arrivals(rate, wall_s, burst_factor,
                                                burst_on_seconds, phase)
                    if a < T0 + offer_wall_s]
    else:
        arrivals = [a for a in _poisson_free_arrivals(rate, wall_s)
                    if a < T0 + offer_wall_s]

    cutoff = T0 + wall_s + collection_grace_s
    free = [0.0] * max(1, int(concurrency))
    heapq.heapify(free)
    records = []
    for a in arrivals:
        earliest = heapq.heappop(free)
        start = max(a, earliest)
        finish = start + service_s
        heapq.heappush(free, finish)
        if finish <= cutoff:
            records.append({"submitted_at_unix": a, "status": "ok",
                            "e2e_latency_s": finish - a})
        else:
            records.append({"submitted_at_unix": a, "status": "error",
                            "e2e_latency_s": ""})
    return records


def old_method_ratio(records, wall_s, rate):
    """The v1 achieved/offered: completed-OK / wall_s / offered_rate. Counts only
    requests that finished within the wall window (finish <= T0+wall_s), which is
    what left the in-flight tail uncounted and depressed the ratio."""
    n_ok = sum(1 for r in records
               if r["status"] == "ok"
               and r["submitted_at_unix"] + float(r["e2e_latency_s"]) <= T0 + wall_s)
    achieved = n_ok / wall_s if wall_s > 0 else 0.0
    return achieved / rate if rate > 0 else 0.0


# ---------------------------------------------------------------------------
# (a) old bias reproduced, then eliminated
# ---------------------------------------------------------------------------


class BiasReproducedAndFixed(unittest.TestCase):
    # Unsaturated: 0.25 rps, 64 servers, service 40s -- capacity is 64/40=1.6 rps,
    # far above 0.25, so NOTHING queues. e2e (~40s) is a large fraction of the
    # 240s v1 window: the exact regime that produced achieved_ratio=0.83.
    RATE = 0.25
    OLD_WALL = 240.0
    SERVICE = 40.0
    CONC = 64

    def _records(self, wall_s):
        # collection_grace lets the v2 sub-window's requests drain; the v1 ratio
        # ignores the grace (counts only finish <= wall).
        return simulate_records(self.RATE, wall_s, self.SERVICE, self.CONC,
                                collection_grace_s=self.SERVICE)

    def test_old_method_is_biased_below_the_bar(self):
        recs = self._records(self.OLD_WALL)
        old = old_method_ratio(recs, self.OLD_WALL, self.RATE)
        # ~1 - service/wall = ~1 - 40/240: reproduces the field symptom (a low-0.8s
        # ratio) and would FAIL the achieved_ratio_min=0.98 bar on an UNSATURATED
        # server. The exact value (~0.85) is the discrete count 51/60.
        self.assertGreater(old, 0.80)
        self.assertLess(old, 0.90)
        self.assertLess(old, 0.98)

    def test_v2_ratio_converges_to_one_same_window(self):
        # Same short window, same unsaturated server: the v2 sub-window ratio must
        # recover ~1.0 despite e2e being comparable to the window.
        recs = self._records(self.OLD_WALL)
        stats = cr.measure_window_stats(recs, self.OLD_WALL, self.RATE)
        self.assertGreaterEqual(stats["achieved_ratio"], 0.98)
        self.assertAlmostEqual(stats["achieved_ratio"], 1.0, places=6)
        self.assertEqual(stats["drop_rate"], 0.0)
        # And it is accepted by the UNCHANGED ceiling criteria.
        self.assertTrue(_stable(stats))

    def test_v2_ratio_stable_across_window_lengths(self):
        # The whole point: the ratio no longer depends on the window/e2e ratio.
        for wall in (200.0, 240.0, 600.0, 1200.0):
            stats = cr.measure_window_stats(self._records(wall), wall, self.RATE)
            self.assertAlmostEqual(stats["achieved_ratio"], 1.0, places=6,
                                   msg=f"wall={wall}")

    def test_recommended_window_makes_edge_a_small_fraction(self):
        # v2 default sizing: wall = max(600, 20*p99). With p99~40s the ceil_mult
        # leg wins (20*40=800 > 600), and the warmup+drain edge (2*p99=80s) is
        # exactly ~10% of the window -- a bounded small fraction, by construction.
        wall = cr.size_step_window(self.SERVICE, step_min_s=600.0, ceil_mult=20.0)
        self.assertEqual(wall, 800.0)  # ceil_mult * p99 = 20 * 40
        stats = cr.measure_window_stats(self._records(wall), wall, self.RATE)
        edge = stats["warmup_s"] + stats["drain_s"]
        self.assertLessEqual(edge / wall, 0.10 + 1e-9)
        self.assertAlmostEqual(stats["achieved_ratio"], 1.0, places=6)


# ---------------------------------------------------------------------------
# (b) a truly saturated server still fails the criteria
# ---------------------------------------------------------------------------


class SaturatedStillFails(unittest.TestCase):
    def test_saturated_rate_is_not_stable(self):
        # 1 server, 1s service -> capacity 1 rps. Offer 4 rps: the queue grows
        # without bound, e2e climbs, and the tail never drains -> the v2 stats
        # must still FAIL (ratio, p99 or climb), not be rescued by the sub-window.
        recs = simulate_records(4.0, 600.0, 1.0, concurrency=1,
                                collection_grace_s=0.0)
        stats = cr.measure_window_stats(recs, 600.0, 4.0)
        self.assertFalse(_stable(stats),
                         f"saturated server wrongly accepted: {stats}")

    def test_saturation_knee_detected_in_a_sweep(self):
        # A sweep low->high: the low rate is stable, the saturated rate is the
        # knee. select_ceiling must stop at the last stable rate.
        rows = []
        for rate in (0.5, 1.0, 4.0):
            # capacity ~1 rps (1 server, 1s service): 0.5 & 1.0 ok, 4.0 saturates.
            recs = simulate_records(rate, 600.0, 1.0, concurrency=1,
                                    collection_grace_s=1.0)
            s = cr.measure_window_stats(recs, 600.0, rate)
            s["offered_rate"] = rate
            rows.append(s)
        ceiling, status = cr.select_ceiling(rows)
        self.assertEqual(status, "ok")
        self.assertLessEqual(ceiling["offered_rate"], 1.0)

    def test_dropped_requests_fail_the_bar(self):
        # Saturation can also surface as drops: a batch of dropped requests in
        # the sub-window must push drop_rate over the bar / ratio under it.
        recs = simulate_records(1.0, 600.0, 1.0, concurrency=8,
                                collection_grace_s=1.0)
        # Mark 10% of the measure-window requests as dropped.
        for i, r in enumerate(recs):
            if i % 10 == 0:
                r["status"] = "dropped"
                r["e2e_latency_s"] = ""
        stats = cr.measure_window_stats(recs, 600.0, 1.0)
        self.assertGreater(stats["drop_rate"], 0.02)
        self.assertFalse(_stable(stats))


# ---------------------------------------------------------------------------
# (c) bursty phase-independence
# ---------------------------------------------------------------------------


class BurstyCycleTrimMechanics(unittest.TestCase):
    """Trim MECHANICS on a DETERMINISTIC on/off model (fixed cycle length): with
    exact cycles, integer-cycle alignment removes the partial-burst edge and is
    exactly phase-independent. The real client sojourns are exponential, so this
    is a lower-bound sanity check of the trimming logic, not a claim about the
    live process -- that is covered by BurstyStochasticFidelity."""
    RATE = 1.0
    WALL = 600.0
    SERVICE = 2.0
    CONC = 64
    BF = 4.0
    ON = 10.0  # cycle = ON*BF = 40s

    def _phase_records(self, phase):
        return simulate_records(
            self.RATE, self.WALL, self.SERVICE, self.CONC, arrival="bursty",
            burst_factor=self.BF, burst_on_seconds=self.ON, phase=phase,
            collection_grace_s=self.SERVICE)

    def test_cycle_length(self):
        self.assertEqual(cr.burst_cycle_seconds("bursty", self.BF, self.ON), 40.0)
        self.assertEqual(cr.burst_cycle_seconds("poisson", self.BF, self.ON), 0.0)

    def test_measure_window_is_whole_cycles(self):
        stats = cr.measure_window_stats(
            self._phase_records(0.0), self.WALL, self.RATE,
            arrival_mode="bursty", burst_factor=self.BF, burst_on_seconds=self.ON)
        cycle = cr.burst_cycle_seconds("bursty", self.BF, self.ON)
        self.assertGreaterEqual(stats["burst_cycles"], 1)
        # measure_seconds is an exact integer number of cycles.
        self.assertAlmostEqual(stats["measure_seconds"] % cycle, 0.0, places=6)
        self.assertAlmostEqual(stats["measure_seconds"],
                               stats["burst_cycles"] * cycle, places=6)

    def test_burst_aware_accounting_is_phase_independent(self):
        cycle = cr.burst_cycle_seconds("bursty", self.BF, self.ON)
        phases = [i * cycle / 8.0 for i in range(8)]  # sweep phase across a cycle
        aware, naive = [], []
        for ph in phases:
            recs = self._phase_records(ph)
            a = cr.measure_window_stats(
                recs, self.WALL, self.RATE, arrival_mode="bursty",
                burst_factor=self.BF, burst_on_seconds=self.ON)
            # "naive": same sub-window carving but NO cycle trimming (treat as
            # poisson), so the window can end mid-cycle and catch a partial burst.
            n = cr.measure_window_stats(recs, self.WALL, self.RATE,
                                        arrival_mode="poisson")
            aware.append(a["achieved_rps"])
            naive.append(n["achieved_rps"])
        # Both should be unsaturated (ratio ~ 1), but the cycle-aligned achieved
        # rate is far more phase-stable than the un-trimmed one.
        self.assertLess(statistics.pstdev(aware), statistics.pstdev(naive))
        # And the burst-aware achieved rate tracks the true mean rate tightly.
        self.assertAlmostEqual(statistics.mean(aware), self.RATE, delta=0.05)

    def test_bursty_unsaturated_ratio_is_one_every_phase(self):
        cycle = cr.burst_cycle_seconds("bursty", self.BF, self.ON)
        for i in range(8):
            recs = self._phase_records(i * cycle / 8.0)
            stats = cr.measure_window_stats(
                recs, self.WALL, self.RATE, arrival_mode="bursty",
                burst_factor=self.BF, burst_on_seconds=self.ON)
            self.assertAlmostEqual(stats["achieved_ratio"], 1.0, places=6,
                                   msg=f"phase index {i}")


# ---------------------------------------------------------------------------
# window provenance recorded on every row
# ---------------------------------------------------------------------------


class StalledStepGuard(unittest.TestCase):
    """v2's completed/submitted ratio is blind to a load generator that stalled:
    the few requests it managed to submit can all succeed -> ratio 1.0 on a step
    that never tested the requested offer. offered_span_frac (a TEMPORAL span,
    robust to the bursty rate variance that confounds a count ratio) catches it."""

    def test_healthy_step_spans_the_window(self):
        recs = simulate_records(1.0, 600.0, 2.0, 64, collection_grace_s=2.0)
        stats = cr.measure_window_stats(recs, 600.0, 1.0)
        self.assertGreaterEqual(stats["offered_span_frac"], 0.98)
        self.assertTrue(_stable_full(stats))

    def test_stalled_client_low_span(self):
        # Client stops submitting at ~1/3 of the window: every offered request
        # succeeds (ratio ~ 1.0) but submissions only span ~1/3 of the sub-window.
        recs = simulate_records(1.0, 600.0, 2.0, 64, collection_grace_s=2.0,
                                offer_wall_s=200.0)
        stats = cr.measure_window_stats(recs, 600.0, 1.0)
        self.assertAlmostEqual(stats["achieved_ratio"], 1.0, places=6)
        self.assertLess(stats["offered_span_frac"], 0.5)

    def test_stalled_step_cannot_be_a_ceiling(self):
        # A stalled step must be rejected by select_ceiling even though its ratio,
        # drop, p99 and climb all look healthy -> ceiling falls to the last GOOD
        # rate instead of the stalled (silently-low) one.
        good = cr.measure_window_stats(
            simulate_records(0.5, 600.0, 2.0, 64, collection_grace_s=2.0),
            600.0, 0.5)
        good["offered_rate"] = 0.5
        stalled = cr.measure_window_stats(
            simulate_records(1.0, 600.0, 2.0, 64, collection_grace_s=2.0,
                             offer_wall_s=150.0),
            600.0, 1.0)
        stalled["offered_rate"] = 1.0
        ceiling, status = cr.select_ceiling([good, stalled])
        self.assertEqual(ceiling["offered_rate"], 0.5)

    def test_bursty_healthy_step_not_flagged_as_stall(self):
        # The confounder that killed a count-ratio gate: a healthy bursty step
        # whose realized rate ran ~18% low still SPANS the window -> not a stall.
        import random
        sys.path.insert(0, str(REPO / "client"))
        from benchmark import BurstyArrival
        ba = BurstyArrival(1.0, random.Random(0), burst_factor=4.0, on_seconds=10.0)
        t, arr = T0, []
        while t < T0 + 1200.0:
            t += ba.next_interval()
            if t < T0 + 1200.0:
                arr.append(t)
        recs = simulate_records(1.0, 1200.0, 2.0, 64, collection_grace_s=2.0,
                                explicit_arrivals=arr)
        stats = cr.measure_window_stats(recs, 1200.0, 1.0, arrival_mode="bursty",
                                        burst_factor=4.0, burst_on_seconds=10.0)
        self.assertLess(stats["offered_coverage"], 0.85)   # count ratio WOULD flag
        self.assertGreaterEqual(stats["offered_span_frac"], 0.9)  # span does not
        self.assertTrue(_stable_full(stats))

    def test_nonzero_client_rc_cannot_be_a_ceiling(self):
        # A step whose client exited non-zero is not trustworthy regardless of its
        # stats: select_ceiling must refuse it.
        row = cr.measure_window_stats(
            simulate_records(1.0, 600.0, 2.0, 64, collection_grace_s=2.0),
            600.0, 1.0)
        row["offered_rate"] = 1.0
        self.assertTrue(_stable_full(row))       # stats alone look fine
        row["client_rc"] = 1
        self.assertFalse(_stable_full(row))      # ... but rc!=0 rejects it


class BurstyStochasticFidelity(unittest.TestCase):
    """Validate the burst handling against the REAL client arrival process
    (exponential on/off sojourns), NOT the deterministic model.

    Empirical fact (measured from client.benchmark.BurstyArrival): with
    exponential sojourns the realized offered rate over a practical window is both
    variable (pstdev ~0.18 over ~30 mean cycles) and biased ~8% high, converging
    to nominal only near ~20000s. So mean-cycle alignment does NOT make the
    OFFERED throughput phase-independent, and achieved_rps for a bursty cell
    inherits that process variance. What IS robust is the count-based VERDICT:
    achieved_ratio = completed/submitted stays ~1.0 on an unsaturated server for
    every seed regardless of the offered-rate wobble -- which is what select_
    ceiling actually keys on. These tests assert that robust property, not a false
    'achieved_rps == nominal rate'."""

    def _real_bursty_arrivals(self, rate, wall_s, bf, on, seed):
        import random
        sys.path.insert(0, str(REPO / "client"))
        from benchmark import BurstyArrival
        ba = BurstyArrival(rate, random.Random(seed), burst_factor=bf, on_seconds=on)
        t, arrivals = T0, []
        while t < T0 + wall_s:
            t += ba.next_interval()
            if t < T0 + wall_s:
                arrivals.append(t)
        return arrivals

    def test_verdict_is_robust_across_seeds(self):
        # The unsaturated-server VERDICT (achieved_ratio ~ 1.0) is stable across
        # seeds even though the offered rate wobbles seed-to-seed.
        rate, wall, bf, on = 1.0, 1200.0, 4.0, 10.0  # cycle ~40s -> ~30 cycles
        for seed in range(10):
            arr = self._real_bursty_arrivals(rate, wall, bf, on, seed)
            recs = simulate_records(rate, wall, 2.0, 64, collection_grace_s=2.0,
                                    explicit_arrivals=arr)
            stats = cr.measure_window_stats(
                recs, wall, rate, arrival_mode="bursty",
                burst_factor=bf, burst_on_seconds=on)
            self.assertGreaterEqual(stats["achieved_ratio"], 0.98, msg=f"seed {seed}")
            # The count-ratio coverage wobbles with the process (can dip below
            # 0.85), but the temporal span stays ~1.0 -- a healthy step is never
            # mistaken for a stall regardless of the seed's rate draw.
            self.assertGreaterEqual(stats["offered_span_frac"], 0.9, msg=f"seed {seed}")

    def test_measure_window_is_whole_mean_cycles(self):
        arr = self._real_bursty_arrivals(1.0, 1200.0, 4.0, 10.0, seed=0)
        recs = simulate_records(1.0, 1200.0, 2.0, 64, collection_grace_s=2.0,
                                explicit_arrivals=arr)
        stats = cr.measure_window_stats(recs, 1200.0, 1.0, arrival_mode="bursty",
                                        burst_factor=4.0, burst_on_seconds=10.0)
        cycle = cr.burst_cycle_seconds("bursty", 4.0, 10.0)
        self.assertGreaterEqual(stats["burst_cycles"], 1)
        self.assertAlmostEqual(stats["measure_seconds"],
                               stats["burst_cycles"] * cycle, places=6)


class WindowProvenance(unittest.TestCase):
    def test_row_carries_window_params(self):
        recs = simulate_records(0.5, 600.0, 10.0, 64, collection_grace_s=10.0)
        stats = cr.measure_window_stats(recs, 600.0, 0.5)
        for k in ("window_seconds", "warmup_s", "drain_s", "measure_seconds",
                  "burst_cycles", "p99_size_s"):
            self.assertIn(k, stats)
        self.assertEqual(stats["window_seconds"], 600.0)
        self.assertEqual(stats["burst_cycles"], 0)  # poisson -> no trimming

    def test_empty_window_is_unstable_not_a_crash(self):
        stats = cr.measure_window_stats([], 600.0, 1.0)
        self.assertEqual(stats["achieved_ratio"], 0.0)
        self.assertFalse(_stable(stats))


def _stable(stats):
    """The core latency/ratio acceptance bar (the four original criteria)."""
    return (stats["achieved_ratio"] >= 0.98
            and stats["drop_rate"] <= 0.02
            and stats["p99_e2e_s"] <= 60.0
            and stats["latency_climb_frac"] <= 0.20)


def _stable_full(stats):
    """The full bar select_ceiling now applies (core + offered-span + rc)."""
    return (_stable(stats)
            and stats.get("offered_span_frac", 1.0) >= 0.5
            and int(stats.get("client_rc", 0)) == 0)


if __name__ == "__main__":
    unittest.main()
