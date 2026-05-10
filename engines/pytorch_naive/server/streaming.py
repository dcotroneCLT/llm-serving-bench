"""SSE wrapper for the streaming response.

Frame ordering matches what client/protocols/pytorch_hf.py expects:

    data: {"text": "..."}     # one or more text chunks
    data: {"prompt_tokens": N, "completion_tokens": M}   # final usage frame
    data: [DONE]              # sentinel

The text-then-usage-then-[DONE] order matters. The adapter sets
first_token_at_unix on the first non-empty data line; if the usage frame
came before any text, ttft would be wrong.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi.responses import StreamingResponse


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable buffering on common reverse proxies. We don't expect one in
    # this lab setup, but the header is harmless and protects against the
    # case where the client is upgraded to go through nginx in the future.
    "X-Accel-Buffering": "no",
}


def sse_response(frames: AsyncIterator[dict]) -> StreamingResponse:
    async def _gen():
        async for frame in frames:
            yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
