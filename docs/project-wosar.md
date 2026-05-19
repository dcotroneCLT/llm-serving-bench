# WoSAR 2026 project status

Living document tracking the state of the WoSAR 2026 paper on software aging in
LLM serving engines. Updated as the campaign progresses. Designed as a hand-off
artifact so a new chat session (or a co-author) can pick up the thread in five
minutes.

## Paper

- **Venue**: WoSAR 2026 (ISSRE workshop on Software Aging and Rejuvenation)
- **Deadline**: 30 June 2026 (extended)
- **Authors**: Domenico Cotroneo (UniNa Federico II / UNC Charlotte), Bojan Cukic (UNCC Dean of CCI)
- **Title (working)**: aging in LLM serving engines on GPU
- **Target length**: 7 pages, full paper track

## Research questions

- **RQ1**: do modern LLM serving engines exhibit statistically significant aging signatures under sustained, accelerated stress on GPU?
- **RQ2**: do the signatures differ qualitatively across engines with different memory-management architectures?
- **RQ3**: which classical SAR effect categories remain relevant in this setting, and which novel categories emerge?

## Hardware

- Host: cci-csgpu11 (UNC Charlotte, dedicated 2-week window)
- GPUs: 4 x NVIDIA L40S (46 GB VRAM each)
- CPU: Intel Xeon Gold 6526Y, 32 physical / 64 logical cores
- RAM: 256 GB
- OS: Ubuntu 24.04.4 LTS, kernel 6.17.0-23-generic
- NVIDIA driver: 580.159.03, CUDA 13.0 (pinned via apt-mark hold)
- Docker: 29.4.2 (data root on /home, 6.9 TB)

## Cells

Six production cells, n=3 replicas each = 18 long runs of 36h + 1 sanity (6h).

| cell | engine | image (wosar2026 pin) | gpu | host port | rate (rps) |
|---|---|---|---|---|---|
| e1 | vLLM V1 standalone | vllm/vllm-openai:wosar2026_e1 (latest) | 0 | 8100 | 2.545 |
| a1 | vLLM V0 standalone | vllm/vllm-openai:wosar2026_a1 (v0.7.3) | 0 | 8100 | 0.796 |
| e2 | Triton + vLLM V0 | tritonserver:wosar2026_e2_a2 | 1 | 8200-2 | 2.172 |
| a2 | Triton + vLLM V1 | tritonserver:wosar2026_e2_a2 (+ VLLM_USE_V1=1) | 1 | 8200-2 | 1.753 |
| e3 | PyTorch HF naive sat | pytorch_naive:wosar2026 | 2 | 8300 | 0.174 |
| e3b | PyTorch HF naive low | pytorch_naive:wosar2026 | 2 | 8300 | 0.050 |

Model: Qwen/Qwen2.5-7B-Instruct, BF16, max_model_len=8192.

Workload (client/config.yaml): prompt log-normal median 1500 tok p95 3500, output median 200 p95 800, streaming 70%, Poisson arrivals.

## Topology

Three GPU slots in parallel:

- gpu0 hosts e1, a1 (sequential within slot)
- gpu1 hosts e2, a2 (sequential within slot)
- gpu2 hosts e3, e3b (sequential within slot)

Round-robin within slot (r01 of each cell first, then r02, then r03).

Plus 1 sanity run: e2 on gpu0 (override) for 6h, to argue GPU index does not materially shift slopes in V (Threats to Validity).

## Calendar

- Campaign launched: 16 May 2026 11:12 UTC
- Expected r01 end: ~17 May 23:00 UTC
- Expected r02 end: ~19 May
- Expected r03 end: ~21 May
- Expected sanity end: ~21 May
- **Expected total end: ~21-22 May**
- Deadline: 30 June 2026
- **Buffer: ~35 days** for analysis pipeline + paper rewrite

## Framework

- `campaigns/wosar2026/campaign.yaml`: top-level orchestration (slots, replicas, retry policy, sanity_runs)
- `campaigns/wosar2026/cells/<cell>.yaml`: single source of truth per cell (image, gpu_device, port, engine command, monitors, workload, duration_s=129600)
- `scripts/campaign.py`: orchestrator, threads one slot worker per GPU, dispatches launch_cell.py, manages campaign_state.json checkpoint
- `scripts/launch_cell.py`: per-run launcher (docker run, readyz, GPU sanity gate, PID resolution, monitors, client, VRAM quiescence)
- `monitoring/run_monitors.py`: spawns gpu/proc/system monitors; proc wrapped in sudo (sudoers NOPASSWD entry at /etc/sudoers.d/wosar_proc_monitor)
- `monitoring/find_engine_pid.py`: dynamic PID resolver for Triton's vLLM Python child (triton_child pid_strategy)
- `client/run_client.py`: async Poisson client
- `scripts/smoke_test_run.sh`: 5-min GO/NO-GO gate before each cell
- `scripts/campaign_health.sh`: periodic read-only health monitor (run every 6-12h during campaign)
- `analysis/validation_check.py`: post-run verdict (Mann-Kendall + Theil-Sen, p<0.01 + slope>0)

## Critical fixes applied (chronological)

| Date | File | Fix |
|---|---|---|
| 15 May | scripts/launch_cell.py | container_name_template replica formatting via Python (not YAML format spec) |
| 15 May | scripts/launch_cell.py | corpus_path resolved to absolute in materialized client config |
| 15 May | cells/*.yaml | per-slot host port allocation (8100/8200/8300) to avoid cross-slot collision |
| 15 May | monitoring/run_monitors.py | invoke proc_monitor via sudo -n -- with NOPASSWD entry (ptrace_scope=1 + /home nosuid blocked file caps) |
| 15 May | monitoring/_common.py | CsvRotatingWriter chmod 0644 + chown SUDO_USER on each new file |
| 15 May | scripts/smoke_test_run.sh | engine version detection anchored on "Initializing a V[01] LLM engine" |
| 16 May | client/protocols/vllm_openai.py | idempotent URL construction (handle base_url ending in /v1) |
| 16 May | cells/e1.yaml, cells/a1.yaml | base_url without trailing /v1 |
| 16 May | scripts/campaign.py | assert slot.gpu_device == cell.engine.gpu_device at build_schedule |
| 16 May | scripts/launch_cell.py | save docker logs to run_dir/logs/container.log before docker rm -f |
| 16 May | scripts/launch_cell.py | parse engine version from logs, store in manifest.engine_version_detected |
| 16 May | scripts/launch_cell.py | nvidia-smi failure retried once, then die() |
| 16 May | scripts/launch_cell.py | --attempt N CLI flag for retry tracking |
| 16 May | scripts/campaign_health.sh | yaml_path helper, manifest_kind detection, robust fallback chain |

## Monitoring routine

Every 6-12 hours, from any shell on cci-csgpu11:

```bash
cd ~/wosar/llm-serving-bench
bash scripts/campaign_health.sh
```

Expected: 38+ PASS, 0-3 WARN, 0 FAIL. Exit code 0 (OK), 1 (WARN), or 2 (FAIL).

Interpret FAILs:
- `container.alive` FAIL: container died unexpectedly. Check `docker logs $name` and run_dir/logs/container.log.
- `proc.alive < 95%`: PID tracking or sudo permission issue.
- `gpu.vram` FAIL: engine not actually serving on the expected GPU.
- `manifest.interrupted` mid-campaign: launch_cell.py was SIGINT'd; should not happen in steady state.
- `client.ok zero` or `client.dropped > 5%`: workload or engine pathology.

## Cell-specific notes

- **e1**: vLLM V1 latest pinned to digest sha256:a230095847e9... at 2026-05-15. Drifted from the n=1 baseline digest sha256:9eff9734...; the campaign reports the new digest in the paper.
- **a1**: vLLM v0.7.3 (last with V0 default).
- **e2**: Triton 25.09 with vLLM 0.10.x backend. The "Engine in background thread is experimental on VLLM_USE_V1=1. Falling back to V0 Engine" log line appears at startup even without VLLM_USE_V1 set; the authoritative line is "Initializing a V[01] LLM engine".
- **a2**: VLLM_USE_V1=1 env var enables V1 in Triton (confirmed empirically: rss_p50 ~1.8 GB vs e2's ~0.8 GB, distinct engine profile). Rate recalibrated from n=1's 0.897 rps to 1.753 rps (85% of measured V1 saturation in Triton).
- **e3**: PyTorch + HF naive baseline, no PagedAttention, no continuous batching. VRAM ~17 GB (model weights only).
- **e3b**: same engine as e3, workload rate 0.050 rps (30% of e3). Ablation tests whether aging slope depends on input rate for the naive engine.

## What to do at end of campaign

1. Run `python analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_<cell>_r<NN>` for each of the 19 runs. Each emits PASS / SOFT FAIL / HARD FAIL verdict on the RSS slope.
2. Aggregate r01/r02/r03 slopes per cell (median + union/intersection of 95% CIs).
3. Apply Benjamini-Hochberg FDR control at q=0.10 across cells.
4. Re-write paper sections IV.C (results per RQ), IV.E (ablations), V (Threats to Validity).
5. Update docs/experimental_protocol.md to reflect the actual campaign (Qwen2.5-7B, n=3, 36h windows, 3-slot parallel).
6. Submit by 30 June 2026.

## Pending technical debt (post-campaign)

- `run_monitors.py` no longer overwrites `manifest.json`; legacy archived runs from before this fix have monitor-shape manifest. Health check has fallback logic to handle both shapes.
- Other client adapters (triton_vllm.py, pytorch_hf.py) could benefit from strict base_url validation similar to vllm_openai.py. Defense-in-depth, not urgent (current yamls are correct).
- VRAM saturation differs slightly between Triton (98%) and vLLM standalone (90-92%) due to backend default `gpu_memory_utilization`. Cite in paper as out-of-the-box production deployment characteristic.

## Archive of aborted attempts

- `~/wosar/runs_n1_baseline/`: original n=1 runs from May 6-15, kept for reference but not used in the paper (n=3 clean campaign supersedes).
- `~/wosar/runs_failed_attempts/`: first attempts of A1/A2 ablations (May 12-13) before find_engine_pid.py fix; data corrupted (process_alive=False from stale PID tracking).
- `~/wosar/runs_aborted_20260516_052308/`: the May 16 first attempt of the n=3 campaign, aborted at 7h after detecting the vllm_openai URL doubling bug (e1 client 100% HTTP 404). 3 runs archived; the relaunched campaign at 11:12 UTC is the canonical one.
