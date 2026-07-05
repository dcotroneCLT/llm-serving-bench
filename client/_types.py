"""Shared types for the client.

A RequestResult is what every protocol adapter returns. It is what the
logger writes to CSV. Fields are intentionally engine-agnostic; if a
field cannot be measured for a given engine, it is left as None.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# Internal monotonic-anchor fields, used ONLY for duration math (they are dropped
# from the CSV row). Wall-clock *_at_unix timestamps remain the CSV provenance.
_MONO_FIELDS = (
    "submitted_at_mono", "started_at_mono", "first_token_at_mono", "finished_at_mono",
)


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
    shared_prefix_applied: bool = False   # prefix-repeat injection (KV-cache reuse stress)
    # Workload outcome (what actually happened)
    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    # Derived latencies in seconds (filled by the logger to keep adapters thin)
    queue_time_s: Optional[float] = None
    ttft_s: Optional[float] = None
    e2e_latency_s: Optional[float] = None
    inter_token_latency_mean_s: Optional[float] = None
    # Monotonic anchors for the SAME events (time.monotonic()). Durations are
    # computed from these so an NTP step during a request cannot corrupt them; the
    # wall-clock *_at_unix fields stay in the CSV for provenance. NOT written to
    # the CSV (dropped in to_csv_row; absent from CSV_FIELDNAMES).
    submitted_at_mono: Optional[float] = None
    started_at_mono: Optional[float] = None
    first_token_at_mono: Optional[float] = None
    finished_at_mono: Optional[float] = None
    # Engine-specific extras for debugging; not always present
    extras: dict[str, Any] = field(default_factory=dict)

    def stamp(self, event: str) -> None:
        """Record BOTH timestamps for an event ('started'|'first_token'|'finished'):
        the wall-clock (CSV provenance) and the monotonic anchor (duration math)."""
        setattr(self, f"{event}_at_unix", time.time())
        setattr(self, f"{event}_at_mono", time.monotonic())

    def to_csv_row(self) -> dict[str, Any]:
        d = asdict(self)
        # Internal monotonic anchors are not part of the CSV schema.
        for k in _MONO_FIELDS:
            d.pop(k, None)
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
    "shared_prefix_applied",
    "actual_input_tokens",
    "actual_output_tokens",
    "queue_time_s",
    "ttft_s",
    "e2e_latency_s",
    "inter_token_latency_mean_s",
    "extras",
]
