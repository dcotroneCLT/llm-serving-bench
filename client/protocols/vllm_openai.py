"""vLLM adapter: hits the OpenAI-compatible /v1/completions endpoint.

vLLM stand-alone exposes both a native /generate and an OpenAI-compatible
/v1/completions. We pick the OpenAI route because Triton+vLLM exposes
the same shape via Triton's HTTP API for vLLM backend, which makes the
adapter for C2 and C3 nearly identical and the comparison cleaner.

For streaming, vLLM emits SSE (text/event-stream). We parse line-by-line
and capture the first non-empty data line as first_token_at_unix.
"""

from __future__ import annotations

import json
import time

import httpx

from _types import RequestResult
from . import ProtocolAdapter


class VLLMOpenAIAdapter(ProtocolAdapter):
    name = "vllm_openai"

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/completions"
        return self.endpoint_url("v1", "completions")

    @property
    def diagnostic_url(self) -> str:
        return self.completions_url

    async def request(
        self,
        http: httpx.AsyncClient,
        req_id: int,
        submitted_at_unix: float,
        prompt: str,
        max_tokens: int,
        stream: bool,
    ) -> RequestResult:
        url = self.completions_url
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": stream,
        }
        result = RequestResult(
            req_id=req_id,
            submitted_at_unix=submitted_at_unix,
            started_at_unix=None,
            first_token_at_unix=None,
            finished_at_unix=None,
            status="error",
            requested_input_tokens=None,
            requested_max_output_tokens=max_tokens,
            streaming=stream,
        )
        try:
            if stream:
                payload["stream_options"] = {"include_usage": True}
                async with http.stream("POST", url, json=payload, timeout=self.timeout_s) as r:
                    result.started_at_unix = time.time()
                    result.http_status = r.status_code
                    if r.status_code != 200:
                        result.error_message = f"http {r.status_code}"
                        result.finished_at_unix = time.time()
                        return result
                    output_tokens = 0
                    actual_input_tokens = None
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if result.first_token_at_unix is None:
                            result.first_token_at_unix = time.time()
                        # Count only chunks that carry text to avoid usage-only frames.
                        choices = chunk.get("choices") or []
                        for c in choices:
                            txt = c.get("text") or ""
                            if txt:
                                # Rough output token accounting: defer to usage if engine reports it.
                                pass
                        usage = chunk.get("usage")
                        if usage:
                            actual_input_tokens = usage.get("prompt_tokens", actual_input_tokens)
                            output_tokens = usage.get("completion_tokens", output_tokens)
                    result.finished_at_unix = time.time()
                    result.actual_input_tokens = actual_input_tokens
                    result.actual_output_tokens = output_tokens
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
                usage = body.get("usage") or {}
                result.actual_input_tokens = usage.get("prompt_tokens")
                result.actual_output_tokens = usage.get("completion_tokens")
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
