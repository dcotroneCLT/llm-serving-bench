"""Protocol adapter base class.

Every adapter implements:

  async def request(client, prompt, max_tokens, stream) -> RequestResult

The adapter is responsible for filling timing fields (started_at,
first_token_at, finished_at) and the actual_input/output token counts
when reported by the engine. The orchestrator fills derived fields
(latencies) post-hoc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from _types import RequestResult


class ProtocolAdapter(ABC):
    name: str = "base"

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def endpoint_url(self, *parts: str) -> str:
        return "/".join([self.base_url, *[p.strip("/") for p in parts if p.strip("/")]])

    @property
    def diagnostic_url(self) -> str:
        return self.base_url

    @abstractmethod
    async def request(
        self,
        http: httpx.AsyncClient,
        req_id: int,
        submitted_at_unix: float,
        prompt: str,
        max_tokens: int,
        stream: bool,
    ) -> RequestResult:
        ...
