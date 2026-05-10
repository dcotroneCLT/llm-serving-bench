"""End-to-end contract test for the pytorch_naive engine.

Hits a running server (default http://localhost:8002) with one
non-streaming and one streaming /generate call, and checks that the
response shape matches what client/protocols/pytorch_hf.py expects.
Validation only — no quality assertions on the generated text.

Usage on the server, after launch.sh has brought the engine up:

    python tests/test_generate.py
    python tests/test_generate.py --base-url http://localhost:8002

Exit code is 0 on success, 1 on any contract violation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import urllib.request
import urllib.error


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def http_post_json(url: str, body: dict, timeout: float = 120.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def http_post_stream(url: str, body: dict, timeout: float = 120.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def test_readyz(base_url: str) -> None:
    try:
        with urllib.request.urlopen(f"{base_url}/readyz", timeout=5) as r:
            if r.status != 200:
                fail(f"/readyz returned {r.status}")
    except urllib.error.URLError as e:
        fail(f"/readyz unreachable: {e}")
    ok("/readyz responds 200")


def test_nonstream(base_url: str) -> None:
    body = {"prompt": "Hello, my name is", "max_tokens": 16, "stream": False}
    t0 = time.time()
    status, payload = http_post_json(f"{base_url}/generate", body)
    dt = time.time() - t0
    if status != 200:
        fail(f"non-stream POST returned HTTP {status}")
    for key in ("text", "prompt_tokens", "completion_tokens"):
        if key not in payload:
            fail(f"non-stream payload missing '{key}': got keys {list(payload)}")
    if not isinstance(payload["text"], str):
        fail(f"non-stream 'text' is not a string: {type(payload['text']).__name__}")
    if not isinstance(payload["prompt_tokens"], int) or payload["prompt_tokens"] <= 0:
        fail(f"non-stream prompt_tokens invalid: {payload['prompt_tokens']!r}")
    if not isinstance(payload["completion_tokens"], int) or payload["completion_tokens"] <= 0:
        fail(f"non-stream completion_tokens invalid: {payload['completion_tokens']!r}")
    ok(
        f"non-stream: prompt_tokens={payload['prompt_tokens']} "
        f"completion_tokens={payload['completion_tokens']} "
        f"text_len={len(payload['text'])} dt={dt:.2f}s"
    )


def test_stream(base_url: str) -> None:
    body = {"prompt": "The quick brown fox", "max_tokens": 16, "stream": True}
    t0 = time.time()
    resp = http_post_stream(f"{base_url}/generate", body)
    if resp.status != 200:
        fail(f"stream POST returned HTTP {resp.status}")

    text_frames = 0
    usage: Optional[dict] = None
    saw_done = False
    first_frame_at: Optional[float] = None

    for raw in resp:
        line = raw.decode("utf-8").rstrip("\n").rstrip("\r")
        if not line:
            continue
        if not line.startswith("data:"):
            fail(f"stream emitted non-SSE line: {line!r}")
        data = line[len("data:"):].strip()
        if first_frame_at is None:
            first_frame_at = time.time()
        if data == "[DONE]":
            saw_done = True
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            fail(f"stream emitted invalid JSON frame: {data!r}")
        if "text" in obj:
            text_frames += 1
            if not isinstance(obj["text"], str):
                fail(f"stream text frame has non-string text: {obj!r}")
        if "prompt_tokens" in obj or "completion_tokens" in obj:
            usage = obj

    if not saw_done:
        fail("stream did not emit [DONE]")
    if usage is None:
        fail("stream did not emit usage frame with prompt_tokens/completion_tokens")
    for key in ("prompt_tokens", "completion_tokens"):
        if key not in usage or not isinstance(usage[key], int) or usage[key] <= 0:
            fail(f"stream usage frame invalid for {key!r}: {usage!r}")
    if text_frames == 0:
        fail("stream emitted no text frames")
    if first_frame_at is None:
        fail("stream emitted no frames at all")

    dt = time.time() - t0
    ttf = first_frame_at - t0
    ok(
        f"stream: text_frames={text_frames} "
        f"prompt_tokens={usage['prompt_tokens']} "
        f"completion_tokens={usage['completion_tokens']} "
        f"ttf={ttf:.2f}s dt={dt:.2f}s [DONE]={saw_done}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8002")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    print(f"contract test against {base}")
    test_readyz(base)
    test_nonstream(base)
    test_stream(base)
    print("ALL OK")


if __name__ == "__main__":
    main()
