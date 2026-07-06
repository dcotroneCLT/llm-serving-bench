"""Client writer-error handling (F4): a CSV writer I/O failure is FATAL and
distinct from a per-request adapter error. It must not be silently turned into
an 'error' row (or swallowed by gather); it must stop the run and surface a
non-zero exit + an explicit marker file.

Off-box, no server. Run: python3 -m unittest tests.test_client_writer_fatal
"""
import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))

import benchmark as bm  # noqa: E402


class StubAdapter:
    timeout_s = 5.0

    def __init__(self, fail=False):
        self.fail = fail

    async def request(self, http, req_id, submitted_at_unix, prompt, max_tokens, stream):
        if self.fail:
            raise RuntimeError("adapter boom")
        return bm.RequestResult(
            req_id=req_id,
            submitted_at_unix=submitted_at_unix,
            started_at_unix=submitted_at_unix,
            first_token_at_unix=submitted_at_unix,
            finished_at_unix=submitted_at_unix,
            status="ok",
            streaming=stream,
        )


class StubSampler:
    def sample(self, target_tokens, tolerance_frac=0.15):
        return ("hello world", target_tokens)


class FakeWriter:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail
        self.attempts = 0
        self.closed = False

    def write(self, row):
        self.attempts += 1
        if self.fail:
            raise OSError("No space left on device")
        self.rows.append(row)

    def close(self):
        self.closed = True


def make_engine(tmp, adapter=None, writer=None):
    eng = bm.BenchmarkEngine(
        adapter=adapter or StubAdapter(),
        sampler=StubSampler(),
        output_dir=Path(tmp),
        target_rate_rps=50.0,
        concurrency_cap=8,
        prompt_len={"median": 10, "p95": 20, "min": 1, "max": 100},
        max_tokens={"median": 10, "p95": 20, "min": 1, "max": 100},
        streaming_prob=0.0,
    )
    eng.writer = writer if writer is not None else FakeWriter()
    return eng


class WriteRowHelper(unittest.TestCase):
    def test_writer_failure_is_fatal_stops_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = FakeWriter(fail=True)
            eng = make_engine(tmp, writer=w)
            eng._write_row({"req_id": 0})
            self.assertIsNotNone(eng._writer_error)
            self.assertTrue(eng._stop.is_set())
            self.assertEqual(w.attempts, 1)
            # Second call must NOT thrash the broken writer.
            eng._write_row({"req_id": 1})
            self.assertEqual(w.attempts, 1)

    def test_writer_success_records_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = FakeWriter(fail=False)
            eng = make_engine(tmp, writer=w)
            eng._write_row({"req_id": 7})
            self.assertIsNone(eng._writer_error)
            self.assertEqual(w.rows, [{"req_id": 7}])


class DispatchClassification(unittest.TestCase):
    def test_adapter_error_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = FakeWriter(fail=False)
            eng = make_engine(tmp, adapter=StubAdapter(fail=True), writer=w)
            eng.in_flight = 1
            asyncio.run(eng._dispatch_one(None, 0, time.time(), time.monotonic()))
            # Adapter failure -> one 'error' row, run NOT marked fatal.
            self.assertIsNone(eng._writer_error)
            self.assertEqual(len(w.rows), 1)
            self.assertEqual(w.rows[0]["status"], "error")

    def test_writer_failure_on_success_path_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = FakeWriter(fail=True)
            eng = make_engine(tmp, adapter=StubAdapter(fail=False), writer=w)
            eng.in_flight = 1
            asyncio.run(eng._dispatch_one(None, 0, time.time(), time.monotonic()))
            self.assertIsNotNone(eng._writer_error)
            self.assertTrue(eng._stop.is_set())


class RunEscalation(unittest.TestCase):
    def test_run_raises_and_writes_marker_on_writer_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = FakeWriter(fail=True)
            eng = make_engine(tmp, writer=w)
            with self.assertRaises(bm.WriterFatalError):
                asyncio.run(eng.run(duration_seconds=0.3))
            marker = Path(tmp) / "client_fatal.json"
            self.assertTrue(marker.exists())
            data = json.loads(marker.read_text())
            self.assertEqual(data["fatal"], "writer_error")


class ResumeReqId(unittest.TestCase):
    """F5: on resume, req_id_next must clear BOTH the last state checkpoint and
    the highest req_id already on disk. state.json is only checkpointed every
    ~30s, so CSVs can hold newer req_ids after a crash; resuming from the stale
    checkpoint alone would duplicate req_ids across files."""

    def _write_csv(self, path, req_ids):
        with open(path, "w", newline="") as f:
            f.write("req_id,status\n")
            for r in req_ids:
                f.write(f"{r},ok\n")

    def test_fresh_run_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(make_engine(tmp).req_id_next, 0)

    def test_state_used_when_ahead_of_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "state.json").write_text(json.dumps({"req_id_next": 100}))
            self._write_csv(Path(tmp) / "requests_000000.csv", [0, 1, 2])
            self.assertEqual(make_engine(tmp).req_id_next, 100)

    def test_csv_used_when_ahead_of_stale_state(self):
        # Crash after CSVs advanced past the last 30s checkpoint.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "state.json").write_text(json.dumps({"req_id_next": 50}))
            self._write_csv(Path(tmp) / "requests_000000.csv", [48, 49, 50, 51, 73])
            self.assertEqual(make_engine(tmp).req_id_next, 74)  # max(73)+1

    def test_no_state_falls_back_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_csv(Path(tmp) / "requests_000000.csv", [0, 1, 2, 3])
            self.assertEqual(make_engine(tmp).req_id_next, 4)


if __name__ == "__main__":
    unittest.main()
