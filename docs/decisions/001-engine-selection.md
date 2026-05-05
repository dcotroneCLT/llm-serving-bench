# ADR 001: Choice of serving engines

Status: accepted
Date: 2026-05-05

## Context

The study compares three LLM serving configurations to characterize how
memory-management design choices in modern serving stacks affect long-run
behavior under sustained GPU stress. Several candidate configurations
exist (PyTorch + HF naive, vLLM stand-alone, Triton with vLLM backend,
Triton with TensorRT-LLM backend, TGI, SGLang, others). A choice of three
configurations is made to keep the experimental matrix tractable within
the available time budget (two weeks of dedicated server access, four
NVIDIA L40 GPUs).

## Decision

Three configurations selected:

C1. PyTorch + HuggingFace Transformers, served via a minimal FastAPI loop.
C2. vLLM stand-alone, default configuration.
C3. NVIDIA Triton Inference Server with vLLM backend.

## Rationale

This triple isolates two independent axes of variation cleanly.

The first axis, between C1 and C2, captures the effect of modern
LLM-specialized memory management (PagedAttention, continuous batching) on
long-run behavior, with all other layers (Python runtime, CUDA, container)
held as similar as possible.

The second axis, between C2 and C3, captures the effect of the Triton
serving wrapper (request scheduling, dynamic batching, HTTP/gRPC frontend)
at fixed underlying engine. Differences between C2 and C3 can be
attributed to the wrapper rather than to the inference engine.

Together, the three configurations support two independent claims that
the paper can make and defend:
- the design of the memory manager affects the long-run behavior;
- the serving wrapper affects the long-run behavior, independently of the
  underlying engine.

## Alternatives considered and rejected

- Triton + TensorRT-LLM as third engine. Rejected because TensorRT-LLM
  requires building per-model engines from source, adding significant
  setup time and reducing the cleanness of the comparison with C2 (the
  inference engine itself would change, not just the wrapper).
- TGI or SGLang as additional engines. Rejected to keep the matrix to
  three points; can be added in a future extended version.
- Two engines instead of three. Rejected because losing either axis
  (memory manager or wrapper) materially weakens the contribution.

## Consequences

- The setup of C3 relies on the official NGC pre-built container with the
  vLLM backend, avoiding custom builds.
- The vLLM core in C2 and C3 must be the same version, pinned and
  declared in the paper, otherwise the C2 vs C3 comparison is confounded.
- Future work can add a fourth configuration (e.g., Triton +
  TensorRT-LLM) without invalidating the present results.
