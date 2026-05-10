"""PyTorch + HuggingFace adapter for the minimal FastAPI loop we ship.

Protocol adapter for the pytorch_naive engine (engines/pytorch_naive/).
Named pytorch_hf for historical reasons; refers to the HuggingFace
transformers API used by the naive baseline.

The PyTorch+HF naive engine in this project exposes:

  POST /generate
    request body:  {"prompt": str, "max_tokens": int, "stream": bool}
    non-streaming response:
      {"text": str, "prompt_tokens": int, "completion_tokens": int}
    streaming response: SSE with "data: {token text}" lines, terminated by
                       "data: [DONE]" and a final usage frame "data: {...}".

This adapter intentionally uses the same data shape as a stripped-down
OpenAI completions API to keep the three adapters comparable. The HF
engine implementation lives under engines/pytorch_naive/.
"""

from __future__ import annotations

import json
import time

import httpx

from _types import RequestResult
from . import ProtocolAdapter


class PyTorchHFAdapter(ProtocolAdapter):
    name = "pytorch_hf"

    async def request(
        self,
        http: httpx.AsyncClient,
        req_id: int,
        submitted_at_unix: float,
        prompt: str,
        max_tokens: int,
        stream: bool,
    ) -> RequestResult:
        url = f"{self.base_url}/generate"
        payload = {"prompt": prompt, "max_tokens": max_tokens, "stream": stream}
        result = RequestResult(
            req_id=req_id,
            submitted_at_unix=submitted_at_unix,
            started_at_unix=None,
            first_token_at_unix=None,
            finished_at_unix=None,
            status="error",
            requested_max_output_tokens=max_tokens,
            streaming=stream,
        )
        try:
            if stream:
                async with http.stream("POST", url, json=payload, timeout=self.timeout_s) as r:
                    result.started_at_unix = time.time()
                    result.http_status = r.status_code
                    if r.status_code != 200:
                        result.error_message = f"http {r.status_code}"
                        result.finished_at_unix = time.time()
                        return result
                    last_usage: dict = {}
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            continue
                        if result.first_token_at_unix is None:
                            result.first_token_at_unix = time.time()
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if "prompt_tokens" in obj or "completion_tokens" in obj:
                            last_usage = obj
                    result.finished_at_unix = time.time()
                    result.actual_input_tokens = last_usage.get("prompt_tokens")
                    result.actual_output_tokens = last_usage.get("completion_tokens")
                    result.status = "ok"
            else:
                result.started_at_unix = time.time()
                r = await http.post(url, json=payload, timeout=self.timeout_s)
                result.finished_at_unix = time.time()
                result.http_status = r.status_code
                if r.status_code != 200:
                    result.error_message = f"http {r.status_code}: {r.text[:200]}"
                    return result
                body = r.json()
                result.actual_input_tokens = body.get("prompt_tokens")
                result.actual_output_tokens = body.get("completion_tokens")
                result.status = "ok"
        except httpx.TimeoutException:
            result.status = "timeout"
            result.error_message = "client timeout"
            result.finished_at_unix = time.time()
        except httpx.HTTPError as e:
            result.status = "error"
            result.error_message = f"http error: {e}"
            result.finished_at_unix = time.time()
        return result
