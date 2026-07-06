"""pytorch_hf streaming TTFT (F8): first_token must be stamped on the first
non-empty TEXT frame, never on a usage/empty/malformed frame that arrives
first (which would corrupt the time-to-first-token measurement).

Contract (engines/pytorch_naive/server/streaming.py): every SSE frame is JSON --
text chunks {"text": "..."}, then a {"prompt_tokens","completion_tokens"} usage
frame, then [DONE].

Off-box, no server (httpx.stream is faked). Run:
    python3 -m unittest tests.test_pytorch_hf_ttft
"""
import asyncio
import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))

from protocols.pytorch_hf import PyTorchHFAdapter  # noqa: E402


class FakeStreamCM:
    def __init__(self, status, lines):
        self.status_code = status
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeHttp:
    """Stands in for httpx.AsyncClient.stream(...)."""

    def __init__(self, status, lines):
        self._status = status
        self._lines = lines

    def stream(self, method, url, json=None, timeout=None):
        return FakeStreamCM(self._status, self._lines)


def _run(lines):
    adapter = PyTorchHFAdapter(base_url="http://x/generate", model="m", timeout_s=5.0)
    http = FakeHttp(200, lines)
    return asyncio.run(adapter.request(
        http, req_id=0, submitted_at_unix=time.time(),
        prompt="hi", max_tokens=8, stream=True))


class StreamingTTFT(unittest.TestCase):
    def test_ttft_stamped_on_first_text_frame(self):
        res = _run([
            'data: {"text": "Hello"}',
            'data: {"text": " world"}',
            'data: {"prompt_tokens": 5, "completion_tokens": 2}',
            'data: [DONE]',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNotNone(res.first_token_at_unix)
        self.assertEqual(res.actual_input_tokens, 5)
        self.assertEqual(res.actual_output_tokens, 2)

    def test_usage_only_stream_does_not_stamp_ttft(self):
        # Degenerate completion: usage frame arrives with no text frame.
        res = _run([
            'data: {"prompt_tokens": 5, "completion_tokens": 0}',
            'data: [DONE]',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNone(res.first_token_at_unix)  # no text -> no TTFT
        self.assertEqual(res.actual_output_tokens, 0)

    def test_empty_text_frame_does_not_stamp(self):
        res = _run([
            'data: {"text": ""}',
            'data: {"prompt_tokens": 3, "completion_tokens": 0}',
            'data: [DONE]',
        ])
        self.assertIsNone(res.first_token_at_unix)

    def test_empty_and_malformed_frames_skipped_until_text(self):
        res = _run([
            'data: ',                 # empty payload
            'data: not-json',         # malformed
            'data: {"text": ""}',     # empty text
            'data: {"text": "hi"}',   # first REAL token
            'data: {"prompt_tokens": 4, "completion_tokens": 1}',
            'data: [DONE]',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNotNone(res.first_token_at_unix)
        self.assertEqual(res.actual_output_tokens, 1)

    def test_ttft_is_not_after_finish(self):
        res = _run([
            'data: {"text": "a"}',
            'data: {"prompt_tokens": 1, "completion_tokens": 1}',
            'data: [DONE]',
        ])
        # Sanity: first token stamped at or before finish.
        self.assertIsNotNone(res.first_token_at_unix)
        self.assertIsNotNone(res.finished_at_unix)
        self.assertLessEqual(res.first_token_at_unix, res.finished_at_unix)


if __name__ == "__main__":
    unittest.main()
