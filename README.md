# llm-serving-bench

Experimental benchmarking framework for comparing LLM serving engines under
sustained, long-duration GPU workloads. The project investigates the
behavior of modern serving stacks when run continuously, with a focus on
resource-management characteristics that are not visible in short-form
throughput or latency benchmarks.

This is a research codebase. It is being developed for an academic
publication and is currently in early alpha. APIs, configurations, and the
overall structure are expected to change.

## Scope

Three serving configurations are compared on a fixed model and a fixed,
controlled workload:

1. PyTorch + HuggingFace Transformers, served via a minimal FastAPI loop.
2. vLLM stand-alone (PagedAttention, continuous batching).
3. NVIDIA Triton Inference Server with vLLM backend.

All three are run on dedicated NVIDIA GPUs. Measurements are collected at
the system, process, and GPU level, plus engine-native metrics where
available. Statistical analysis follows established methodology for
long-running system studies (non-parametric trend tests, slope estimation
with confidence intervals).

## Repository layout

```
docs/         protocol, decision records, notes
client/       async benchmarking client (separate machine recommended)
monitoring/   metric collection agents (system / process / GPU)
engines/      configurations and Dockerfiles per serving engine
analysis/     statistical analysis scripts and notebooks
paper/        manuscript sources (LaTeX)
runs/         experiment outputs (gitignored)
```

## Status

Phase 1 (setup and pilot tooling) in progress.

## Reproducibility

Once the experimental campaign is complete, the repository will include
pinned engine versions, Docker images, the full monitoring stack, raw data
links, and the analysis pipeline used to produce the figures and tables in
the paper.

## License

To be added.
