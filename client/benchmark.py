"""Open-loop benchmark engine with concurrency cap.

Scheduling model. Requests are generated at a fixed target rate via a
Poisson arrival process (configurable to deterministic). Each generated
request is dispatched immediately if the in-flight count is below the
concurrency cap; otherwise it is dropped and recorded as status="dropped"
with a row in the CSV. Drop rate over time is itself an aging
indicator: if the engine slows down, more arrivals find the cap full.

Output. Per-request rows are written to CSV files under
output-dir/, base name `requests`, with the same rotating writer used
by the monitoring agents. This gives rotation every 60 s by default
and bounded data loss on crash.

Restartability. State (last req_id, last log file index) is persisted
to output-dir/state.json every 30 s and on shutdown. On restart the
client reads it and continues numbering req_ids monotonically.

Workload shape. Each request draws:
  - a target prompt length from a log-normal-ish distribution
  - a max_tokens from another distribution
  - streaming or not, by Bernoulli with configured probability
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import httpx

from _types import CSV_FIELDNAMES, RequestResult
from prompt_sampler import PromptSampler
from protocols import ProtocolAdapter


# ---------------------------------------------------------------------------
# Arrival processes
#
# Every process exposes next_interval() -> seconds until the next arrival.
# poisson reproduces the historical exponential interarrival exactly (same
# RNG draw on the shared engine RNG), so a poisson run is unchanged.
# ---------------------------------------------------------------------------


class PoissonArrival:
    def __init__(self, rate: float, rng: random.Random) -> None:
        self.rate = rate
        self.rng = rng

    def next_interval(self) -> float:
        return self.rng.expovariate(self.rate) if self.rate > 0 else float("inf")


class ConstantArrival:
    def __init__(self, rate: float) -> None:
        self.gap = 1.0 / rate if rate > 0 else float("inf")

    def next_interval(self) -> float:
        return self.gap


class BurstyArrival:
    """MMPP-2 on/off arrival process with the mean rate preserved.

    Arrivals occur only during ON periods, at on_rate = rate * burst_factor;
    the process spends a fraction 1/burst_factor of time ON, so the long-run
    mean rate equals `rate` and only the variance (burstiness) changes. This
    keeps the DoW "rate" factor comparable across Poisson and bursty levels.
    burst_factor is the peak/mean ratio (CoV knob); on_seconds is the mean ON
    sojourn (burst timescale). next_interval() returns the gap to the next
    arrival, transparently spanning idle OFF periods.
    """

    def __init__(self, rate: float, rng: random.Random, burst_factor: float = 4.0, on_seconds: float = 10.0) -> None:
        self.rng = rng
        self.mean_rate = rate
        self.burst_factor = max(1.0, float(burst_factor))
        self.on_rate = rate * self.burst_factor if rate > 0 else 0.0
        on_frac = 1.0 / self.burst_factor
        self.mean_on = max(1e-6, float(on_seconds))
        self.mean_off = self.mean_on * (1.0 - on_frac) / on_frac if on_frac < 1.0 else 0.0
        self.in_on = True
        self.t_left = self._draw_on()

    def _draw_on(self) -> float:
        return self.rng.expovariate(1.0 / self.mean_on)

    def _draw_off(self) -> float:
        return self.rng.expovariate(1.0 / self.mean_off) if self.mean_off > 0 else 0.0

    def next_interval(self) -> float:
        if self.mean_rate <= 0 or self.on_rate <= 0:
            return float("inf")
        elapsed = 0.0
        while True:
            if self.in_on:
                gap = self.rng.expovariate(self.on_rate)
                if gap <= self.t_left:
                    self.t_left -= gap
                    return elapsed + gap
                elapsed += self.t_left
                self.in_on = False
                self.t_left = self._draw_off()
            else:
                elapsed += self.t_left
                self.in_on = True
                self.t_left = self._draw_on()


def make_arrival_process(
    arrival_mode: str, rate: float, rng: random.Random,
    burst_factor: float = 4.0, burst_on_seconds: float = 10.0,
):
    mode = (arrival_mode or "poisson").lower()
    if mode == "constant":
        return ConstantArrival(rate)
    if mode == "bursty":
        return BurstyArrival(rate, rng, burst_factor=burst_factor, on_seconds=burst_on_seconds)
    # default / "poisson"
    return PoissonArrival(rate, rng)


class CsvRotatingWriter:
    """Same idea as the monitoring writer, kept local to avoid cross-package imports."""

    def __init__(self, output_dir: Path, base_name: str, rotation_seconds: int, fieldnames: list[str]) -> None:
        self.output_dir = output_dir
        self.base_name = base_name
        self.rotation_seconds = rotation_seconds
        self.fieldnames = fieldnames
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Resume past any existing files so a restart into the same output dir
        # (the documented req_id recovery path) does not overwrite the prior
        # run's requests_000000.csv in "w" mode.
        existing = sorted(self.output_dir.glob(f"{base_name}_*.csv"))
        last_seq = existing[-1].stem.rsplit("_", 1)[-1] if existing else ""
        self._seq = int(last_seq) + 1 if last_seq.isdigit() else 0
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._opened_at = 0.0

    def _open_new(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
        path = self.output_dir / f"{self.base_name}_{self._seq:06d}.csv"
        self._file = path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._opened_at = time.monotonic()
        self._seq += 1

    def write(self, row: dict[str, Any]) -> None:
        now = time.monotonic()
        if self._file is None or (now - self._opened_at) >= self.rotation_seconds:
            self._open_new()
        assert self._writer is not None
        self._writer.writerow(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None


def fill_derived_latencies(r: RequestResult) -> None:
    if r.started_at_unix is not None:
        r.queue_time_s = max(0.0, r.started_at_unix - r.submitted_at_unix)
    if r.first_token_at_unix is not None and r.started_at_unix is not None:
        r.ttft_s = max(0.0, r.first_token_at_unix - r.started_at_unix)
    if r.finished_at_unix is not None and r.started_at_unix is not None:
        r.e2e_latency_s = max(0.0, r.finished_at_unix - r.started_at_unix)
    if (
        r.first_token_at_unix is not None
        and r.finished_at_unix is not None
        and r.actual_output_tokens
        and r.actual_output_tokens > 1
    ):
        gen_time = max(0.0, r.finished_at_unix - r.first_token_at_unix)
        r.inter_token_latency_mean_s = gen_time / max(1, r.actual_output_tokens - 1)


def sample_prompt_length(rng: random.Random, median: int, p95: int, lo: int, hi: int) -> int:
    """Log-normal sample with given median and ~p95, clipped to [lo, hi]."""
    # mu = ln(median); sigma chosen so that exp(mu + 1.645*sigma) = p95
    if p95 <= median:
        p95 = median + 1
    mu = math.log(median)
    sigma = (math.log(p95) - mu) / 1.645
    val = int(round(math.exp(rng.gauss(mu, sigma))))
    return max(lo, min(hi, val))


def sample_max_tokens(rng: random.Random, median: int, p95: int, lo: int, hi: int) -> int:
    return sample_prompt_length(rng, median, p95, lo, hi)


class BenchmarkEngine:
    def __init__(
        self,
        adapter: ProtocolAdapter,
        sampler: PromptSampler,
        output_dir: Path,
        target_rate_rps: float,
        concurrency_cap: int,
        prompt_len: dict[str, int],
        max_tokens: dict[str, int],
        streaming_prob: float,
        request_distribution: str = "poisson",
        rotation_seconds: int = 60,
        seed: int = 0,
        prefix_repeat_fraction: float = 0.0,
        shared_prefix_len: int = 0,
        arrival_mode: Optional[str] = None,
        burst_factor: float = 4.0,
        burst_on_seconds: float = 10.0,
    ) -> None:
        self.adapter = adapter
        self.sampler = sampler
        self.output_dir = output_dir
        self.target_rate_rps = target_rate_rps
        self.concurrency_cap = concurrency_cap
        self.prompt_len = prompt_len
        self.max_tokens = max_tokens
        self.streaming_prob = streaming_prob
        self.rotation_seconds = rotation_seconds
        self.rng = random.Random(seed)

        # arrival_mode supersedes the legacy request_distribution; the old
        # field is still read for back-compat (poisson | constant). poisson
        # uses the shared engine RNG with the same expovariate(rate) draw as
        # before, so a poisson run is unchanged.
        self.arrival_mode = (arrival_mode or request_distribution or "poisson").lower()
        self.request_distribution = self.arrival_mode  # kept for any external reader
        self.burst_factor = float(burst_factor)
        self.burst_on_seconds = float(burst_on_seconds)
        self.arrival = make_arrival_process(
            self.arrival_mode, target_rate_rps, self.rng,
            burst_factor=self.burst_factor, burst_on_seconds=self.burst_on_seconds,
        )

        # Prefix-repeat injection. The shared prefix is built once, RNG-free,
        # so it is identical across requests (KV-cache reuse) and so that
        # enabling the feature does NOT shift the per-request sampling stream.
        # fraction 0 (or len 0) disables it: no extra RNG draw is taken, so a
        # disabled run is byte-identical to the prior behavior.
        self.prefix_repeat_fraction = float(prefix_repeat_fraction)
        self.shared_prefix_len = int(shared_prefix_len)
        if self.prefix_repeat_fraction > 0.0 and self.shared_prefix_len > 0:
            self._shared_prefix, self._shared_prefix_tokens = sampler.fixed_prefix(self.shared_prefix_len)
        else:
            self._shared_prefix, self._shared_prefix_tokens = "", 0

        self.writer = CsvRotatingWriter(
            output_dir=output_dir,
            base_name="requests",
            rotation_seconds=rotation_seconds,
            fieldnames=CSV_FIELDNAMES,
        )
        self.state_path = output_dir / "state.json"
        self.req_id_next = 0
        self.in_flight = 0
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._restore_state()

    def _restore_state(self) -> None:
        if self.state_path.exists():
            try:
                s = json.loads(self.state_path.read_text())
                self.req_id_next = int(s.get("req_id_next", 0))
                # writer seq does not need to be restored: a new run uses fresh files
            except (json.JSONDecodeError, ValueError):
                pass

    def _persist_state(self) -> None:
        try:
            self.state_path.write_text(json.dumps({"req_id_next": self.req_id_next}))
        except OSError:
            pass

    def request_stop(self) -> None:
        self._stop.set()

    async def _dispatch_one(self, http: httpx.AsyncClient, req_id: int, submitted_at_unix: float) -> None:
        target_in_tok = sample_prompt_length(
            self.rng, self.prompt_len["median"], self.prompt_len["p95"],
            self.prompt_len["min"], self.prompt_len["max"],
        )
        prompt, approx_in = self.sampler.sample(target_in_tok)
        target_out = sample_max_tokens(
            self.rng, self.max_tokens["median"], self.max_tokens["p95"],
            self.max_tokens["min"], self.max_tokens["max"],
        )
        stream = self.rng.random() < self.streaming_prob
        # Prefix-repeat draw LAST, and only when enabled, so prompt_len /
        # max_tokens / streaming see the same RNG draws as a prefix-disabled
        # run for the same seed (only the prefix decision is appended).
        shared_prefix_applied = False
        if self._shared_prefix:
            shared_prefix_applied = self.rng.random() < self.prefix_repeat_fraction
            if shared_prefix_applied:
                prompt = self._shared_prefix + "\n\n" + prompt
                approx_in += self._shared_prefix_tokens
        try:
            result = await self.adapter.request(
                http=http,
                req_id=req_id,
                submitted_at_unix=submitted_at_unix,
                prompt=prompt,
                max_tokens=target_out,
                stream=stream,
            )
            result.requested_input_tokens = approx_in
            result.shared_prefix_applied = shared_prefix_applied
            fill_derived_latencies(result)
            self.writer.write(result.to_csv_row())
        except asyncio.CancelledError:
            result = RequestResult(
                req_id=req_id,
                submitted_at_unix=submitted_at_unix,
                started_at_unix=None,
                first_token_at_unix=None,
                finished_at_unix=time.time(),
                status="error",
                error_message="client task cancelled",
                requested_input_tokens=approx_in,
                requested_max_output_tokens=target_out,
                streaming=stream,
                shared_prefix_applied=shared_prefix_applied,
            )
            fill_derived_latencies(result)
            self.writer.write(result.to_csv_row())
            raise
        except Exception as e:
            result = RequestResult(
                req_id=req_id,
                submitted_at_unix=submitted_at_unix,
                started_at_unix=None,
                first_token_at_unix=None,
                finished_at_unix=time.time(),
                status="error",
                error_message=f"{type(e).__name__}: {str(e)[:500]}",
                requested_input_tokens=approx_in,
                requested_max_output_tokens=target_out,
                streaming=stream,
                shared_prefix_applied=shared_prefix_applied,
            )
            fill_derived_latencies(result)
            self.writer.write(result.to_csv_row())
        finally:
            self.in_flight -= 1

    def _drop(self, req_id: int, submitted_at_unix: float) -> None:
        result = RequestResult(
            req_id=req_id,
            submitted_at_unix=submitted_at_unix,
            started_at_unix=None,
            first_token_at_unix=None,
            finished_at_unix=time.time(),
            status="dropped",
            error_message="concurrency cap reached",
            streaming=False,
        )
        self.writer.write(result.to_csv_row())

    async def run(self, duration_seconds: float) -> None:
        rate = self.target_rate_rps
        run_until = time.monotonic() + duration_seconds
        next_arrival = time.monotonic()
        last_state_persist = time.monotonic()

        # Realized arrival statistics (streaming accumulators, no per-arrival
        # storage). Measured on the actual dispatch instants so the recorded
        # CoV reflects what the server saw, including bursty idle gaps.
        arr_n = 0
        arr_first = arr_last = arr_prev = None
        arr_gap_sum = 0.0
        arr_gap_sumsq = 0.0

        timeout = httpx.Timeout(self.adapter.timeout_s)
        limits = httpx.Limits(max_connections=self.concurrency_cap * 2, max_keepalive_connections=self.concurrency_cap)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as http:
            while not self._stop.is_set() and time.monotonic() < run_until:
                now = time.monotonic()
                if now < next_arrival:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=next_arrival - now)
                        break
                    except asyncio.TimeoutError:
                        pass

                req_id = self.req_id_next
                self.req_id_next += 1
                submitted_at_unix = time.time()

                # Record the realized interarrival (on the monotonic clock).
                arr_now = time.monotonic()
                arr_n += 1
                if arr_first is None:
                    arr_first = arr_now
                else:
                    gap = arr_now - arr_prev
                    arr_gap_sum += gap
                    arr_gap_sumsq += gap * gap
                arr_prev = arr_last = arr_now

                if self.in_flight >= self.concurrency_cap:
                    self._drop(req_id, submitted_at_unix)
                else:
                    self.in_flight += 1
                    task = asyncio.create_task(self._dispatch_one(http, req_id, submitted_at_unix))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

                # Schedule next arrival.
                #
                # Accumulate onto the previous *scheduled* time, not onto a
                # freshly sampled monotonic clock. Rebasing on time.monotonic()
                # here would fold every per-iteration cost (RNG, task creation,
                # the inline CSV write on the drop path) into the inter-arrival
                # gap, pushing the realized rate below target by an amount that
                # grows as the server ages and the loop gets busier. That would
                # couple offered load to the aging signal under measurement.
                # With the accumulator, if the loop falls behind, next_arrival
                # lands in the past and the `if now < next_arrival` guard above
                # dispatches back-to-back to recover the long-run rate, which is
                # the correct open-loop behavior. The arrival process owns the
                # interarrival law (poisson / constant / bursty); poisson draws
                # the same expovariate(rate) on the shared RNG as before.
                inter = self.arrival.next_interval()
                next_arrival += inter

                # Periodic state persistence
                if (time.monotonic() - last_state_persist) >= 30.0:
                    self._persist_state()
                    last_state_persist = time.monotonic()

            # Drain in-flight tasks with a generous grace period
            if self._tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=60.0)
                except asyncio.TimeoutError:
                    for t in self._tasks:
                        if not t.done():
                            t.cancel()
            self._persist_state()
            self.writer.close()

            # Record the realized arrival statistics so the manifest can show
            # what the server actually saw (esp. the bursty CoV vs poisson).
            realized_dur = (arr_last - arr_first) if (arr_first is not None and arr_last is not None) else 0.0
            mean_gap = (arr_gap_sum / (arr_n - 1)) if arr_n > 1 else float("nan")
            cv = float("nan")
            if arr_n > 2 and mean_gap and mean_gap == mean_gap and mean_gap > 0:
                var = max(0.0, arr_gap_sumsq / (arr_n - 1) - mean_gap * mean_gap)
                cv = (var ** 0.5) / mean_gap
            stats = {
                "arrival_mode": self.arrival_mode,
                "target_rate_rps": self.target_rate_rps,
                "realized_count": arr_n,
                "realized_duration_s": realized_dur,
                "realized_rate_rps": (arr_n / realized_dur) if realized_dur > 0 else 0.0,
                "interarrival_mean_s": mean_gap,
                "interarrival_cv": cv,
            }
            if self.arrival_mode == "bursty":
                stats["burst_factor"] = self.burst_factor
                stats["burst_on_seconds"] = self.burst_on_seconds
            try:
                (self.output_dir / "arrival_stats.json").write_text(json.dumps(stats, indent=2))
            except OSError:
                pass
