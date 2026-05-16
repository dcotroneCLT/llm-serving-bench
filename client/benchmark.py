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


class CsvRotatingWriter:
    """Same idea as the monitoring writer, kept local to avoid cross-package imports."""

    def __init__(self, output_dir: Path, base_name: str, rotation_seconds: int, fieldnames: list[str]) -> None:
        self.output_dir = output_dir
        self.base_name = base_name
        self.rotation_seconds = rotation_seconds
        self.fieldnames = fieldnames
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
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
    ) -> None:
        self.adapter = adapter
        self.sampler = sampler
        self.output_dir = output_dir
        self.target_rate_rps = target_rate_rps
        self.concurrency_cap = concurrency_cap
        self.prompt_len = prompt_len
        self.max_tokens = max_tokens
        self.streaming_prob = streaming_prob
        self.request_distribution = request_distribution
        self.rotation_seconds = rotation_seconds
        self.rng = random.Random(seed)

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

                if self.in_flight >= self.concurrency_cap:
                    self._drop(req_id, submitted_at_unix)
                else:
                    self.in_flight += 1
                    task = asyncio.create_task(self._dispatch_one(http, req_id, submitted_at_unix))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

                # Schedule next arrival
                if self.request_distribution == "poisson":
                    inter = self.rng.expovariate(rate) if rate > 0 else float("inf")
                else:
                    inter = 1.0 / rate if rate > 0 else float("inf")
                next_arrival = time.monotonic() + inter

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
