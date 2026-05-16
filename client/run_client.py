"""Entry point for the benchmark client.

Usage:

    python run_client.py --config config.yaml \\
        --output-dir ../runs/<run_id>/client \\
        --duration-seconds 86400

The config file is YAML and is the canonical source of truth for the
workload parameters. The CLI args are limited to the few values that
must vary between runs (output directory, duration, optional override
of the protocol target).
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import yaml  # type: ignore

from benchmark import BenchmarkEngine
from prompt_sampler import PromptSampler
from protocols.pytorch_hf import PyTorchHFAdapter
from protocols.triton_vllm import TritonVLLMAdapter
from protocols.vllm_openai import VLLMOpenAIAdapter


PROTOCOL_BUILDERS = {
    "vllm_openai": VLLMOpenAIAdapter,
    "pytorch_hf": PyTorchHFAdapter,
    "triton_vllm": TritonVLLMAdapter,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--duration-seconds", type=float, required=True)
    p.add_argument("--protocol", type=str, default=None, help="Override config.protocol.")
    p.add_argument("--base-url", type=str, default=None, help="Override config.base_url.")
    p.add_argument("--model", type=str, default=None, help="Override config.model.")
    p.add_argument("--target-rate-rps", type=float, default=None, help="Override config.target_rate_rps.")
    p.add_argument("--concurrency-cap", type=int, default=None, help="Override config.concurrency_cap.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    protocol = args.protocol or cfg["protocol"]
    base_url = args.base_url or cfg["base_url"]
    model = args.model or cfg["model"]
    target_rate_rps = args.target_rate_rps if args.target_rate_rps is not None else cfg["target_rate_rps"]
    concurrency_cap = args.concurrency_cap if args.concurrency_cap is not None else cfg["concurrency_cap"]

    if protocol not in PROTOCOL_BUILDERS:
        print(f"unknown protocol {protocol}; available: {list(PROTOCOL_BUILDERS)}", file=sys.stderr)
        sys.exit(2)

    adapter = PROTOCOL_BUILDERS[protocol](
        base_url=base_url,
        model=model,
        timeout_s=float(cfg.get("request_timeout_s", 600)),
    )
    print(
        f"[run_client] protocol={protocol} base_url={base_url} "
        f"endpoint={adapter.diagnostic_url} model={model} "
        f"target_rate_rps={target_rate_rps}",
        flush=True,
    )

    corpus_path = Path(cfg["corpus_path"])
    if not corpus_path.is_absolute():
        corpus_path = args.config.parent / corpus_path
    sampler = PromptSampler(
        corpus_path=corpus_path,
        seed=int(cfg.get("seed", 0)),
        tokenizer_name=cfg.get("tokenizer", "cl100k_base"),
    )

    engine = BenchmarkEngine(
        adapter=adapter,
        sampler=sampler,
        output_dir=args.output_dir,
        target_rate_rps=float(target_rate_rps),
        concurrency_cap=int(concurrency_cap),
        prompt_len=cfg["prompt_len"],
        max_tokens=cfg["max_tokens"],
        streaming_prob=float(cfg.get("streaming_prob", 0.7)),
        request_distribution=cfg.get("request_distribution", "poisson"),
        rotation_seconds=int(cfg.get("rotation_seconds", 60)),
        seed=int(cfg.get("seed", 0)),
    )

    loop = asyncio.new_event_loop()

    def handle(_sig, _frame):
        engine.request_stop()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    try:
        loop.run_until_complete(engine.run(duration_seconds=args.duration_seconds))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
