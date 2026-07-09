"""Triton+vLLM protocol adapter (client/protocols/triton_vllm.py).

The extension Triton cell drives 70% streaming requests (val_triton.yaml
streaming_prob: 0.7), so the adapter's frame parsing IS on the validation path.
This exercises it directly, off-box (httpx faked), same style as
tests/test_pytorch_hf_ttft.py. Covers:

  * V1 SSE framing (`data: {...}` + `[DONE]`) -- the 0.20.1 default;
  * V0 NDJSON framing (raw `{...}` lines) -- the compat branch the adapter keeps;
  * TTFT stamped on the first NON-EMPTY text frame, not on empty/malformed ones;
  * streaming and non-streaming HTTP non-200 -> status "error" (not a false "ok");
  * non-streaming happy path (output_chars from text_output).

Run: python3 -m unittest tests.test_triton_vllm_adapter
"""
import asyncio
import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))

from protocols.triton_vllm import TritonVLLMAdapter  # noqa: E402


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


class FakePostResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class FakeHttp:
    """Stands in for httpx.AsyncClient: .stream(...) for streaming requests and
    .post(...) for non-streaming ones."""

    def __init__(self, status, lines=None, post_body=None):
        self._status = status
        self._lines = lines or []
        self._post_body = post_body if post_body is not None else {}

    def stream(self, method, url, json=None, timeout=None):
        return FakeStreamCM(self._status, self._lines)

    async def post(self, url, json=None, timeout=None):
        return FakePostResp(self._status, self._post_body)


def _adapter():
    return TritonVLLMAdapter(base_url="http://localhost:8600", model="qwen",
                             timeout_s=5.0)


def _stream(status, lines):
    return asyncio.run(_adapter().request(
        FakeHttp(status, lines=lines), req_id=0, submitted_at_unix=time.time(),
        prompt="hi", max_tokens=8, stream=True))


def _post(status, body):
    return asyncio.run(_adapter().request(
        FakeHttp(status, post_body=body), req_id=0, submitted_at_unix=time.time(),
        prompt="hi", max_tokens=8, stream=False))


class StreamingV1SSE(unittest.TestCase):
    def test_sse_frames_ok_and_ttft_stamped(self):
        res = _stream(200, [
            'data: {"text_output": "Hel"}',
            'data: {"text_output": "lo world"}',
            'data: [DONE]',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNotNone(res.first_token_at_unix)
        self.assertEqual(res.extras["output_chars"], len("Hello world"))

    def test_ttft_only_on_first_nonempty_text(self):
        res = _stream(200, [
            'data: ',                        # empty payload
            'data: {"text_output": ""}',     # empty text
            'data: not-json',                # malformed
            'data: {"text_output": "hi"}',   # first REAL token
            'data: [DONE]',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNotNone(res.first_token_at_unix)
        self.assertLessEqual(res.first_token_at_unix, res.finished_at_unix)
        self.assertEqual(res.extras["output_chars"], 2)

    def test_no_text_frames_no_ttft(self):
        res = _stream(200, [
            'data: {"text_output": ""}',
            'data: [DONE]',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNone(res.first_token_at_unix)
        self.assertEqual(res.extras["output_chars"], 0)


class StreamingV0NDJSON(unittest.TestCase):
    def test_raw_json_lines_still_parse(self):
        # V0 emitted newline-delimited JSON with no "data:" prefix; the adapter
        # keeps that branch. Prove it does not regress.
        res = _stream(200, [
            '{"text_output": "abc"}',
            '{"text_output": "de"}',
        ])
        self.assertEqual(res.status, "ok")
        self.assertIsNotNone(res.first_token_at_unix)
        self.assertEqual(res.extras["output_chars"], 5)


class StreamingErrors(unittest.TestCase):
    def test_non_200_stream_is_error_not_ok(self):
        res = _stream(500, ['data: {"text_output": "should be ignored"}'])
        self.assertEqual(res.status, "error")
        self.assertEqual(res.http_status, 500)
        self.assertIn("500", res.error_message)
        self.assertIsNotNone(res.finished_at_unix)


class NonStreaming(unittest.TestCase):
    def test_post_ok(self):
        res = _post(200, {"text_output": "hello"})
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.extras["output_chars"], len("hello"))

    def test_post_non_200_is_error(self):
        res = _post(500, {"error": "boom"})
        self.assertEqual(res.status, "error")
        self.assertEqual(res.http_status, 500)
        self.assertIn("500", res.error_message)


if __name__ == "__main__":
    unittest.main()
