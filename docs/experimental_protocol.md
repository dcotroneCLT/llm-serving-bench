# Experimental Protocol: Aging in LLM Serving Engines on GPU

WoSAR 2026 submission. Working document, v0.1.

## 1. Research questions

RQ1. Do modern LLM serving engines exhibit statistically significant aging signatures under sustained, accelerated stress on GPU?

RQ2. Do the signatures differ qualitatively across engines with different memory-management architectures, and on which indicators?

RQ3. Which classical SAR effect categories remain relevant in this setting, and which novel categories emerge that call for indicators not currently part of the SAR vocabulary?

## 2. System under test

Three configurations, each run on a dedicated NVIDIA L40 (46 GB usable VRAM).

C1. PyTorch + HuggingFace Transformers, served via a minimal FastAPI/uvicorn loop. No continuous batching, naive caching allocator, single-request processing with a small request queue. Represents the unoptimized baseline.

C2. vLLM stand-alone, default configuration. PagedAttention, continuous batching, vLLM's own memory manager. Represents the state-of-the-art LLM-specialized engine.

C3. NVIDIA Triton Inference Server with vLLM backend. Same vLLM core as C2, wrapped by Triton's request scheduling, dynamic batching layer, and HTTP/gRPC frontend. Isolates the effect of the serving wrapper at fixed engine.

Hardware. Server with 4 x NVIDIA L40 (46 GB VRAM each), 256 GB RAM, Ubuntu. C1, C2, C3 run in parallel on GPU 0, 1, 2 respectively. GPU 3 reserved for pilots, sensitivity, and recovery from failed runs.

Software. CUDA driver and runtime versions to be pinned and reported. vLLM, Triton, PyTorch, Transformers versions pinned via Docker images for reproducibility. Same Python and CUDA versions across the three configurations to the extent possible; deviations to be documented.

## 3. Models

Primary: Llama-3-8B-Instruct, BF16. Supported as a first-class citizen by all three engines, well-known baseline in serving literature.

Sensitivity: Qwen2.5-7B-Instruct, BF16. Same parameter scale, different family and tokenizer, to test generalization of findings.

Rationale for staying at 7-8B. With 46 GB per GPU, an 8B model in BF16 occupies roughly 16 GB for weights, leaving ~30 GB for KV-cache and activations. This headroom is the resource we want to stress; loading larger models would saturate VRAM with weights and mask the memory-management phenomena we aim to study.

## 4. Workload

Accelerated stress workload, constant in time. Workload shift is intentionally excluded from this paper (already covered by Moura, Nascimento, Machida, Andrade, arXiv 2511.03103).

Request rate. Calibrated per-engine in pilot at 85% of measured saturation throughput. Each engine is stressed relative to its own ceiling, not in absolute terms; this is a deliberate methodological choice and will be declared explicitly in the paper.

Prompt length distribution. Skewed toward long prompts to stress KV-cache: median 1500 tokens, P95 around 3500, capped at the model context window minus 512 to leave room for generation. Sampled from a curated mix of long-form sources (arXiv abstracts, news articles, code snippets) to avoid degenerate repetition.

Output length distribution. Variable, sampled from a truncated log-normal with median 200 tokens and P95 around 800, to exercise both short- and long-generation paths.

Streaming/non-streaming mix. 70/30 streaming to non-streaming, to exercise both response paths.

Client. Asynchronous Python client (httpx + asyncio), running on a separate machine on the same LAN to avoid contaminating server-side measurements with client-side aging. State persisted to disk every minute so it can be restarted without losing progress.

## 5. Run plan

Phase 1 (days 1-3). Setup, model verification, monitoring scripts, client implementation.

Phase 2 (day 3). Pilot runs, 2-3 hours per engine, parallel on three GPUs. Outputs: saturation throughput per engine, validated monitoring stack, request-rate calibration.

Phase 3 (days 4-7). Primary campaign. Llama-3-8B on C1, C2, C3 in parallel. Three replicates of 24 hours each. Total: 9 run-days in ~4 calendar days.

Phase 4 (days 8-11). Sensitivity campaign. Qwen2.5-7B, same structure, three replicates of 24 hours. Same total.

Phase 5 (days 12-14). Buffer for re-runs, full statistical analysis, paper drafting kickoff.

## 6. Measurement instrumentation

System level (sampling 5 s). Free physical memory, swap usage, file descriptor count, thread count, CPU utilization. Tool: collectl + custom /proc readers.

Process level (sampling 5 s). RSS, VSS of the serving process. Python-level heap and live tensor counts where engine cooperates. Tool: psutil keyed on engine PID.

GPU level (sampling 1 s). Per GPU: VRAM allocated, VRAM reserved, fragmentation proxy = (reserved - allocated) / reserved, GPU utilization, SM occupancy, memory bandwidth utilization, temperature. Tool: pynvml + torch.cuda.memory_stats() for the PyTorch-based engines, augmented with engine-native metrics where available (vLLM Prometheus endpoint exposes KV-cache block usage, preemption count, num_running, num_waiting; Triton exposes queue depth, batcher stats).

Application level (per request). Time-to-first-token, inter-token-latency, end-to-end latency, input and output token counts, error code if any. Logged by the client; throughput aggregated post-hoc.

Storage. All metrics streamed to local Parquet files with 1-minute rotation, plus a backup mirror on a second disk. Sufficient given the modest data volume (well under 100 MB per run-day).

## 7. Statistical analysis

Per indicator, per engine, per run.

Preprocessing. Discard the first 60 minutes (warm-up: JIT, KV-cache fill, first GC cycles). Downsample by minute-level mean for trend tests; keep raw for tail analysis.

Trend tests. Modified Mann-Kendall with Hamed-Rao correction for autocorrelation (system-monitoring time series are strongly autocorrelated; vanilla MK over-rejects). Significance threshold p < 0.01 to be conservative.

Effect size. Sen's slope with 95% confidence interval via bootstrap (10000 resamples).

Cross-engine comparison. Two engines are considered to differ on a given indicator iff the 95% CIs of their Sen's slopes do not overlap. Same approach as Andrade et al. (LLM aging on CPU, JSS 2025); ensures direct methodological comparability.

Replicate aggregation. Each engine x indicator yields 3 slope estimates. Report median slope and the union/intersection of the three CIs; flag indicators where replicates disagree as candidates for additional discussion.

## 8. Risks and mitigations

R1. Setup complexity for Triton+vLLM. Mitigation: use NGC pre-built containers (nvcr.io/nvidia/tritonserver:<ver>-vllm-python-py3), no custom builds.

R2. Workload-rate calibration error. If 85% of saturation is too aggressive and an engine collapses mid-run, the run is wasted. Mitigation: pilot enforces sustained 2-hour stability at the chosen rate before phase 3 begins; if any engine fails, lower target to 75% globally for fairness.

R3. Single-point monitoring failures. nvidia-smi can hang; pynvml can deadlock under heavy contention. Mitigation: monitoring agent runs in a separate process with a watchdog and a 30-second sample timeout that emits a sentinel value rather than blocking.

R4. Disk saturation. 24-hour runs at 1 Hz GPU sampling produce manageable volumes, but logs from the engines themselves can balloon. Mitigation: log rotation enforced via journald, with size caps; pre-flight disk check before each run.

R5. Client-side aging contaminating measurements. Mitigation: client on separate LAN-attached machine; client process restarted at every replicate boundary; client-side metrics logged and inspected for drift as a sanity check.

## 9. Deliverable structure (forward-look at the paper)

Section 1, Introduction. Frame: SAR has matured on classical systems; LLM serving on GPU is a new frontier with its own memory-management primitives that have not yet been studied through an aging lens.

Section 2, Background and related work. Concise SAR primer pointing to the canon; recent LLM-aging work (Andrade et al. JSS 2025; Santos, Andrade, Natella arXiv 2510.24188; Moura et al. arXiv 2511.03103) and gap statement on the GPU side.

Section 3, Experimental design. From this protocol.

Section 4, Results. Per-RQ structure. RQ1 establishes presence of aging in all three configurations; RQ2 contrasts signatures; RQ3 reads off the indicators that traditional SAR vocabulary does not cover.

Section 5, Discussion: open challenges for SAR research in LLM serving systems. The core message of the paper. Three to four open problems framed as a research agenda for the community.

Section 6, Threats to validity. Single hardware platform, two models, fixed workload, version-pinned engines.

Section 7, Conclusion.

Target length: 7 pages, full paper track.
