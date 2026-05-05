"""Shared types for the client.

A RequestResult is what every protocol adapter returns. It is what the
logger writes to CSV. Fields are intentionally engine-agnostic; if a
field cannot be measured for a given engine, it is left as None.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RequestResult:
    # Identifiers and timing
    req_id: int
    submitted_at_unix: float
    started_at_unix: Optional[float]      # when the request actually leaves the client
    first_token_at_unix: Optional[float]  # streaming only; None for non-streaming
    finished_at_unix: Optional[float]
    # Outcome
    status: str                           # "ok" | "error" | "dropped" | "timeout"
    http_status: Optional[int] = None
    error_message: Optional[str] = None
    # Workload shape (what we asked for)
    requested_input_tokens: Optional[int] = None
    requested_max_output_tokens: Optional[int] = None
    streaming: bool = False
    # Workload outcome (what actually happened)
    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    # Derived latencies in seconds (filled by the logger to keep adapters thin)
    queue_time_s: Optional[float] = None
    ttft_s: Optional[float] = None
    e2e_latency_s: Optional[float] = None
    inter_token_latency_mean_s: Optional[float] = None
    # Engine-specific extras for debugging; not always present
    extras: dict[str, Any] = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, Any]:
        d = asdict(self)
        # Flatten extras into JSON string so the CSV stays a flat table.
        import json

        d["extras"] = json.dumps(d["extras"], default=str) if d["extras"] else ""
        return d


CSV_FIELDNAMES = [
    "req_id",
    "submitted_at_unix",
    "started_at_unix",
    "first_token_at_unix",
    "finished_at_unix",
    "status",
    "http_status",
    "error_message",
    "requested_input_tokens",
    "requested_max_output_tokens",
    "streaming",
    "actual_input_tokens",
    "actual_output_tokens",
    "queue_time_s",
    "ttft_s",
    "e2e_latency_s",
    "inter_token_latency_mean_s",
    "extras",
]
