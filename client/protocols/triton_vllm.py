"""Triton Inference Server adapter (vLLM backend).

Triton with the vLLM backend (NGC containers from 23.10 onward) exposes
two relevant routes:

  POST /v2/models/<model>/generate
       non-streaming. Body:
         {"text_input": str, "parameters": {"max_tokens": int, ...}}
       Response:
         {"text_output": str, ...}

  POST /v2/models/<model>/generate_stream
       streaming as application/x-ndjson, one JSON object per line, each
       carrying a "text_output" partial.

The model name is the model_repository folder name we configure in
engines/triton_vllm/. Triton does not natively report token counts in
the response body for the vLLM backend; we approximate output_tokens
client-side using the tokenizer of the prompt sampler. This is
imprecise but consistent across all three adapters when activated, and
we document the limitation in the paper's threats to validity.
"""

from __future__ import annotations

import json
import time

import httpx

from _types import RequestResult
from . import ProtocolAdapter


class TritonVLLMAdapter(ProtocolAdapter):
    name = "triton_vllm"

    async def request(
        self,
        http: httpx.AsyncClient,
        req_id: int,
        submitted_at_unix: float,
        prompt: str,
        max_tokens: int,
        stream: bool,
    ) -> RequestResult:
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
        body_payload = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
                "stream": stream,
            },
        }
        try:
            if stream:
                url = f"{self.base_url}/v2/models/{self.model}/generate_stream"
                async with http.stream("POST", url, json=body_payload, timeout=self.timeout_s) as r:
                    result.started_at_unix = time.time()
                    result.http_status = r.status_code
                    if r.status_code != 200:
                        result.error_message = f"http {r.status_code}"
                        result.finished_at_unix = time.time()
                        return result
                    output_chars = 0
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        # Strip SSE "data: " prefix if present (Triton+vLLM V1 uses SSE,
                        # Triton+vLLM V0 used NDJSON). Skip non-data SSE lines.
                        if line.startswith("data:"):
                            line = line[5:].strip()
                            if not line or line == "[DONE]":
                                continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if result.first_token_at_unix is None:
                            result.first_token_at_unix = time.time()
                        text = obj.get("text_output") or ""
                        output_chars += len(text)
                    result.finished_at_unix = time.time()
                    # Char-based estimate, refined later by the orchestrator using the tokenizer.
                    result.extras["output_chars"] = output_chars
                    result.status = "ok"
            else:
                url = f"{self.base_url}/v2/models/{self.model}/generate"
                result.started_at_unix = time.time()
                r = await http.post(url, json=body_payload, timeout=self.timeout_s)
                result.finished_at_unix = time.time()
                result.http_status = r.status_code
                if r.status_code != 200:
                    result.error_message = f"http {r.status_code}: {r.text[:200]}"
                    return result
                body = r.json()
                text = body.get("text_output") or ""
                result.extras["output_chars"] = len(text)
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
