"""Naive HuggingFace Transformers serving engine.

Design contract for E3 (the unoptimized baseline in the WoSAR 2026 study):

  * NO PagedAttention — KV-cache is whatever transformers' generate() builds
    per call, allocated and released through PyTorch's caching allocator.
    This is the behavior we want to study; do not override the cache type.

  * NO continuous batching — exactly one request occupies the GPU at a time.
    Concurrency is serialized by `_lock`. Other requests queue inside the
    FastAPI/uvicorn event loop. Server-side queue_time is observable via
    the client's submitted_at vs started_at timestamps.

  * NO static padded batching — single-sequence forward only. Adding a
    batcher would change the engine's identity for the paper.

  * NO empty_cache() between requests — let the caching allocator behave
    as it does in stock HF code. Fragmentation over time is itself a
    candidate aging signature.

Truncation policy (matches paper protocol):

  MAX_BUDGET = MAX_MODEL_LEN - max_new_tokens
  prompt left-truncated if needed (we keep the tail of the prompt, not
  the head, so the model always sees the most recent context).
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import Config


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_STREAM_SENTINEL = object()


class Engine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._ready = False
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Blocking model load. Call once at startup, off the event loop."""
        if self.cfg.dtype not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype {self.cfg.dtype!r}")
        dtype = _DTYPE_MAP[self.cfg.dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.cfg.device)
        self.model.eval()
        self._ready = True

    def _truncate(self, prompt: str, max_new_tokens: int) -> tuple[torch.Tensor, int]:
        budget = max(1, self.cfg.max_model_len - max_new_tokens)
        ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids[0]
        if ids.size(0) > budget:
            ids = ids[-budget:]
        ids = ids.unsqueeze(0).to(self.cfg.device)
        return ids, int(ids.size(1))

    def _gen_kwargs(self, max_new_tokens: int, streamer: Optional[TextIteratorStreamer] = None) -> dict:
        kw = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if streamer is not None:
            kw["streamer"] = streamer
        return kw

    async def generate_nonstream(
        self, prompt: str, max_new_tokens: int
    ) -> tuple[str, int, int]:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_blocking, prompt, max_new_tokens
            )

    def _generate_blocking(self, prompt: str, max_new_tokens: int) -> tuple[str, int, int]:
        input_ids, prompt_tokens = self._truncate(prompt, max_new_tokens)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self._gen_kwargs(max_new_tokens),
            )
        gen_ids = output[0, input_ids.size(1):]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, prompt_tokens, int(gen_ids.size(0))

    async def generate_stream(
        self, prompt: str, max_new_tokens: int
    ) -> AsyncIterator[dict]:
        """Yields dict frames in this order:

          {"text": <chunk>}            zero or more
          {"prompt_tokens": int, "completion_tokens": int}   exactly one (final)

        The SSE wrapper appends "[DONE]" after the iterator is exhausted.
        Token chunks come from TextIteratorStreamer; completion_tokens is
        computed from the actual output IDs (not by re-tokenizing chunks)
        so the count is exact.
        """
        async with self._lock:
            input_ids, prompt_tokens = self._truncate(prompt, max_new_tokens)
            attention_mask = torch.ones_like(input_ids)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            holder: dict = {}

            def _run() -> None:
                try:
                    with torch.inference_mode():
                        holder["ids"] = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **self._gen_kwargs(max_new_tokens, streamer=streamer),
                        )
                except BaseException as exc:  # noqa: BLE001
                    holder["error"] = exc
                    # Ensure the consumer doesn't block forever on a crashed thread.
                    streamer.end()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            loop = asyncio.get_running_loop()

            def _next_chunk():
                return next(streamer, _STREAM_SENTINEL)

            try:
                while True:
                    chunk = await loop.run_in_executor(None, _next_chunk)
                    if chunk is _STREAM_SENTINEL:
                        break
                    if chunk:
                        yield {"text": chunk}
                # Generation completed naturally; collect the IDs.
                await loop.run_in_executor(None, thread.join)
                if "error" in holder:
                    raise holder["error"]
                gen_ids = holder["ids"][0, input_ids.size(1):]
                yield {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(gen_ids.size(0)),
                }
            finally:
                # Always wait for the GPU thread before releasing the lock,
                # otherwise the next request would race on the device. HF
                # generate() cannot be interrupted, so client disconnects
                # still pay the full generation time on the server side.
                if thread.is_alive():
                    await loop.run_in_executor(None, thread.join)
