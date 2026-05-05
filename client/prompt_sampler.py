"""Prompt sampler.

Loads a corpus of arXiv abstracts/intros from a JSONL file and samples
prompts that approximate a target token length. The corpus is built
once via build_corpus.py and committed to the repo for reproducibility.

Why not random/synthetic text. Truly random tokens trigger pathological
cache and tokenizer paths in real models; the workload should look like
real user traffic to be representative of long-run behavior in
production.

Token-length matching strategy. We pre-tokenize the entire corpus once
at startup using a fast tokenizer (tiktoken's cl100k_base by default,
which is a reasonable proxy for most modern LLM tokenizers and very
fast). At sample time, we draw a target length from the configured
distribution, then pick a corpus item whose token length is within a
tolerance of the target; if none exists, we concatenate items until we
hit the target, then truncate.

The same sampler instance is used across all engines in a run, so the
realized prompt lengths are matched up to the small concatenation
randomness.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CorpusItem:
    text: str
    token_count: int


class PromptSampler:
    def __init__(self, corpus_path: Path, seed: int = 0, tokenizer_name: str = "cl100k_base") -> None:
        self.rng = random.Random(seed)
        self.tokenizer_name = tokenizer_name
        self._encode = self._make_encoder(tokenizer_name)
        self.corpus: list[CorpusItem] = self._load_and_tokenize(corpus_path)
        if not self.corpus:
            raise RuntimeError(f"Empty corpus at {corpus_path}")
        # Sort once by token count for fast nearest-length search.
        self.corpus.sort(key=lambda c: c.token_count)
        self._token_counts = [c.token_count for c in self.corpus]

    @staticmethod
    def _make_encoder(name: str):
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding(name)
            return lambda text: enc.encode(text)
        except ImportError:
            # Fallback: rough estimate of 1 token per 4 characters. Acceptable
            # because we only use token counts to shape the workload; the
            # real engines do their own tokenization.
            return lambda text: list(range(max(1, len(text) // 4)))

    def _load_and_tokenize(self, corpus_path: Path) -> list[CorpusItem]:
        items: list[CorpusItem] = []
        with corpus_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("abstract") or ""
                if not text:
                    continue
                tokens = self._encode(text)
                items.append(CorpusItem(text=text, token_count=len(tokens)))
        return items

    def _nearest_idx(self, target: int) -> int:
        # Bisect on sorted token_counts.
        import bisect

        i = bisect.bisect_left(self._token_counts, target)
        if i == 0:
            return 0
        if i == len(self._token_counts):
            return len(self._token_counts) - 1
        before = self._token_counts[i - 1]
        after = self._token_counts[i]
        return i if (after - target) < (target - before) else i - 1

    def sample(self, target_tokens: int, tolerance_frac: float = 0.15) -> tuple[str, int]:
        """Return (prompt_text, approx_token_count)."""
        idx = self._nearest_idx(target_tokens)
        item = self.corpus[idx]
        tol = max(50, int(tolerance_frac * target_tokens))
        if abs(item.token_count - target_tokens) <= tol:
            return item.text, item.token_count

        # Need to assemble. Concatenate random items until we exceed target,
        # then truncate by character ratio (engines re-tokenize anyway).
        pieces: list[str] = []
        token_total = 0
        # Pick smaller items first to give us granularity.
        small_pool = [c for c in self.corpus if c.token_count <= max(target_tokens // 4, 200)]
        if not small_pool:
            small_pool = self.corpus
        while token_total < target_tokens:
            c = self.rng.choice(small_pool)
            pieces.append(c.text)
            token_total += c.token_count
        text = "\n\n".join(pieces)
        # Truncate by characters proportional to the overshoot.
        if token_total > target_tokens:
            ratio = target_tokens / token_total
            text = text[: max(1, int(len(text) * ratio))]
            token_total = target_tokens
        return text, token_total
