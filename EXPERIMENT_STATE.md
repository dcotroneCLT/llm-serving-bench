# EXPERIMENT_STATE.md

Living hand-off document for the WoSAR 2026 n=3 campaign. Updated by hand
whenever something material changes. Designed so a new chat session (or
a co-author) can pick up the thread in under five minutes.

Last updated: 2026-05-19 (afternoon ET).

---

## How to use this document (for a new chat session)

If you are an LLM assistant picking this up two days from now, read this
section first.

**Working language and tone.** The principal investigator (Domenico)
prefers to chat in Italian. Replies in Italian. The committed
artifacts (this doc, code, comments, paper) stay in English. Style:
peer-level engagement, no em-dashes, no excessive hedging. Push back
on inconsistencies rather than agreeing reflexively.

**Workflow.** Domenico applies code changes himself via Claude Code
inside VS Code. When you need to modify code, describe the problem
and the proposed change in prose; do not edit code in the chat unless
explicitly asked. Verification commands you can give freely; he runs
them on the server and pastes back the output. When he has applied
a patch via Claude Code, do not re-review it line-by-line unless he
asks: just confirm the operational behavior is what was intended.

**Server vs laptop.** Domenico runs the campaign on cci-csgpu11
(UNCC, dedicated host) and pushes/pulls code via git from his Mac.
You do not have direct SSH access to the server. Operationally, you
give him shell commands and he pastes output back. The repo is mirrored
on his laptop; in this environment you can read files locally
(including the n=1 pilot CSVs in `logs/aging_pilot_24h_*/`), run local
analysis, and write to the repo. Git commits are usually done by
Domenico from his Mac because the lock can be held by Claude Code.

**Where to start when he opens this file.**
1. Read the Background section below for the paper context.
2. Read the Status snapshot for where we are in the campaign.
3. Read "Results r01 vs paper Table IV" and "What rules out which
   hypothesis" for the open puzzle.
4. Suggest he runs `bash scripts/campaign_health.sh` on the server
   first thing, to get a fresh state. Then suggest
   `python3 analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_<cell>_<rXX>`
   on any newly-completed runs since this doc was written.
5. The "Open TODOs" and "Open questions" sections at the bottom are
   the action items.

**Companion documents.**
- `docs/WOSAR_2026.pdf`: the submitted (n=1) paper. Read pages 4-6
  (Section IV, Results) for the headline findings and Table IV.
- `docs/project-wosar.md`: longer-form project doc, older snapshot.
  This file (EXPERIMENT_STATE.md) supersedes it as the operational
  hand-off; `project-wosar.md` is kept for historical reference and
  detailed framework documentation (e.g. the catalog of critical
  fixes during framework development).
- `replicate_n1.py` (repo root): script that re-derives paper Table IV
  numbers from the local n=1 CSVs. Useful if you doubt the analytical
  pipeline. Already validated; do not need to re-run unless you have
  reason to suspect drift.

**When in doubt, ask before acting.** Especially before suggesting to
stop the campaign or restart anything. The host window is fixed and
every wasted run is a wasted day.

---

## Background: what the n=1 paper found

The submitted version of the paper (single-run 24h per cell, no replicas)
established three findings that the n=3 campaign was designed to replicate
and strengthen.

1. **Software aging exists in modern LLM serving on the GPU.** All three
   primary deployments (vLLM standalone V1, Triton + vLLM V0, naive
   PyTorch + HuggingFace) showed statistically significant monotonically
   increasing process-private memory (RSS, USS, PSS coincident) over the
   24-hour window. Trend was MK-significant after FDR correction; slope
   was non-zero with 95% Theil-Sen CI excluding zero.
2. **The aging surface lies in the framework orchestration layers, not
   in the inference path.** Counter-intuitive ordering: the naive PyTorch
   baseline (no paged-attention, no continuous batching, no scheduler)
   leaked the least (+170 KB/h). The production-grade engines leaked
   1-2 orders of magnitude more (Triton+V0 +2.04 MB/h, vLLM V1 standalone
   +9.15 MB/h). Since the inference compute path is identical across
   the three (same Qwen2.5-7B weights, same attention math), the aging
   must live in the orchestration layers around it.
3. **The leak rate is a property of the full deployment, not of any
   single component.** A 2x2 factorial across engine generation (V0
   vs V1) and hosting layer (standalone vs Triton wrapper) showed leak
   rates spanning nearly three orders of magnitude across the four
   cells, with engine and hosting interacting (V0 leaks less than V1
   standalone, but V0 leaks more than V1 in Triton).
4. **Step-wise lock-step growth of resident and virtual memory in the
   V0 engine (qualitative finding, Figure 2b of the paper).** This is
   the finding the authors care most about. In E2 (Triton + vLLM V0),
   memory does not grow as a smooth quasi-linear drift: it remains flat
   for hours, then jumps abruptly by several MB at discrete step events.
   At each step, RSS (resident, user-space) and VMS (virtual, includes
   pages mapped via mmap from the kernel) increase TOGETHER, perfectly
   correlated. This signature is consistent with the engine
   periodically requesting NEW memory blocks from the kernel via mmap
   (or sbrk-extended heap) and never releasing them. The paper text
   leaves the two mechanism hypotheses (mmap vs sbrk) unresolved.
   The same pattern shows up qualitatively in three of the four
   factorial cells but is cleanest in E2. Diagnostic analysis on
   secondary indicators (process CPU, request rate, voluntary
   ctx-switches, Python GC activity) shows no correlation with the
   step events: i.e. no obvious cause among the usual suspects. If
   confirmed and mechanism-localized, this would imply UNBOUNDED
   resident memory growth on multi-day timescales (each step adds
   bytes that never return). The classical SAR toolkit cannot
   diagnose this without complementary heap-profiling instrumentation;
   it is the most actionable finding for the vLLM community.

   **Refinement obtained during n=3 preparation (NEW vs paper):**
   the (rss_vms_corr, K_trim) metric panel introduced for n=3
   analysis separates two distinct step-wise mechanisms that the
   paper had conflated. E2 is mmap-style (RSS and VMS lock-step,
   corr ~0.93). A1 is sbrk-style (RSS steps without paired VMS
   steps, corr ~0.20), i.e. heap-arena extension rather than
   kernel-mapped blocks. See Open Question #4 below for the
   classification rule.

Other key observations:
- **Client-side aging is essentially undetectable over 24h** in all
  three primary engines. End-to-end latency, TTFT, throughput, and
  drop rate are stationary. The aging only surfaces when looking
  inside the engine process.
- **One GPU-side aging signature**, on A1 (vLLM V0 standalone): VRAM
  grew at +124 MB/h, accumulating ~3 GB over 24h. None of the other
  five runs showed any VRAM trend.

Paper Table IV (the headline RSS slope table that the n=3 campaign
aims to replicate):

| ID  | Deployment         | RSS slope    | 95% CI               |
|-----|--------------------|--------------|----------------------|
| E1  | vLLM standalone V1 | +9.15 MB/h   | [+9.03, +9.24] MB/h  |
| E2  | Triton + vLLM V0   | +2.04 MB/h   | [+0.82, +3.12] MB/h  |
| E3  | PyTorch + HF naive | +170 KB/h    | [+85, +389] KB/h     |
| E3b | PyTorch + HF low   | +179 KB/h    | overlapping with E3  |
| A1  | vLLM V0 standalone | +530 KB/h    | (Table V of paper)   |
| A2  | Triton + vLLM V1   | +20 KB/h     | (Table V of paper)   |

The n=3 campaign was set up to replicate these numbers with n=3 runs
per cell and a longer 36h window, to estimate run-to-run variance
(the primary threat-to-validity called out in the submitted paper).
The campaign is currently in progress. Section "Results r01 vs paper"
below documents how the first round actually compares.

Paper Threats-to-Validity that the n=3 campaign was designed to address:
- **Single-run design** (the priority): no run-to-run variance bound.
  n=3 fixes this.
- **24h window may miss late-onset effects**: n=3 uses 36h, partial
  improvement.
- **Confounded factorial** (vLLM version drift across cells): unchanged
  by n=3, persists as a residual confound.
- **Single hardware, single model**: unchanged by n=3.

The campaign was NOT designed to address: parallel-vs-sequential
topology. The paper ran one cell at a time. The n=3 campaign runs three
cells in parallel on three GPUs to fit inside the 2-week host window.
This design choice is now under scrutiny (see "Open questions").

---

## TL;DR

The n=3 campaign is on day 3.5 of 9.5. Round 1 (r01 of all 6 cells) is
complete. The RSS slope numbers from r01 are 3 to 260 times smaller than
the n=1 paper Table IV. Image drift has been ruled out as the dominant
cause because a1 (vLLM V0 v0.7.3, immutable 15-month-old binary, identical
args) also drops by 8x. The leading hypothesis is a topology effect:
the paper ran one cell at a time on the host; the n=3 campaign runs three
cells in parallel on three GPUs. E3b (sub-saturated) replicates the paper
exactly while E3 (saturated) does not, which is consistent with a CPU-host
contention mechanism. Plan: let the r02 batch finish (~30h), then stop
the campaign and run a1 in isolation for 24h. If isolated a1 returns
to ~+530 KB/h, topology effect is confirmed and the paper narrative needs
revision.

---

## Paper

- Venue: WoSAR 2026 (ISSRE workshop on Software Aging and Rejuvenation)
- Deadline: 30 June 2026
- Authors: Domenico Cotroneo (UniNa Federico II / UNCC), Bojan Cukic (UNCC)
- Working title: "The Aging Surface of LLM Serving Engines: An Empirical Study"
- Submitted (n=1) version: 6 cells, 24h each, single-GPU sequential.
  Available at docs/WOSAR_2026.pdf (uploaded version).
- Target (n=3) version: 6 cells, 3 replicas each, 36h each, 3-GPU parallel.
  Currently in execution.

---

## Hardware / OS / Driver

- Host: cci-csgpu11 (UNC Charlotte, dedicated 2-week window)
- GPUs: 4 x NVIDIA L40S (46 GB VRAM each), only 3 used (gpu 0/1/2)
- CPU: Intel Xeon Gold 6526Y, 32 physical / 64 logical
- RAM: 256 GB
- OS: Ubuntu 24.04.4 LTS, kernel 6.17.0-23-generic
- NVIDIA driver: 580.159.03, CUDA 13.0 (pinned via apt-mark hold)
- Docker: 29.4.2, data root on /home (6.9 TB available)
- Identical across pilot (n=1) and current campaign (n=3).

---

## Campaign topology (n=3)

Six production cells, n=3 replicas each = 18 long runs of 36h + 1 sanity
(6h). Round-robin within slot: r01 of every cell first, then r02, then r03.

```
gpu0 slot: e1, a1   (sequential within slot)
gpu1 slot: e2, a2   (sequential within slot)
gpu2 slot: e3, e3b  (sequential within slot)
```

| cell | engine            | image tag                          | gpu | host port | rate rps |
|------|-------------------|------------------------------------|-----|-----------|----------|
| e1   | vLLM V1 standalone| wosar2026_e1 (vllm:latest @ 5/15)  | 0   | 8100      | 2.545    |
| a1   | vLLM V0 standalone| wosar2026_a1 (vllm:v0.7.3 fixed)   | 0   | 8100      | 0.796    |
| e2   | Triton + vLLM V0  | wosar2026_e2_a2 (triton 25.09)     | 1   | 8200-2    | 2.172    |
| a2   | Triton + vLLM V1  | wosar2026_e2_a2 + VLLM_USE_V1=1    | 1   | 8200-2    | 1.753    |
| e3   | PyTorch+HF naive  | pytorch_naive:wosar2026 (local)    | 2   | 8300      | 0.174    |
| e3b  | PyTorch+HF low    | pytorch_naive:wosar2026 (local)    | 2   | 8300      | 0.050    |

Model in all cells: Qwen/Qwen2.5-7B-Instruct, BF16, max_model_len 8192,
gpu_memory_utilization 0.9.

---

## Image pinning (digests, as of 2026-05-15T20:02:14Z)

- e1: `sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9`
  (vllm/vllm-openai:latest as of 15 May; drifted from n=1 pilot digest
  `sha256:9eff9734...` which was the latest on 7 May, now pruned)
- a1: `sha256:4f4037303e8c7b69439db1077bb849a0823517c0f785b894dc8e96d58ef3a0c2`
  (vllm/vllm-openai:v0.7.3, immutable semver tag, 15 months old, identical
  to what the paper used)
- e2/a2: `sha256:1fb3d156d4959b83cb7a9bd172f9b86135f97cafcc1b5899292e042536d90141`
  (nvcr.io/nvidia/tritonserver:25.09-vllm-python-py3, 7 months old)
- e3/e3b: `sha256:452c860860870ee50f19575264c12b647a550ac7f0fbaafbfc6d0e33249c7985`
  (local build from `engines/pytorch_naive/Dockerfile`, deterministic
  if the Dockerfile and pinned deps are unchanged in git)

Storage: pin files at `engines/<engine>/image_pin*.json`. The `:latest`
pre-pin image (pre-15 May) for vllm has been pruned and is no longer
recoverable locally.

---

## Workload pinning

Client config (`client/config.yaml`):
- protocol: vllm_openai | triton_vllm | pytorch_hf (per cell)
- target_rate_rps: per cell (see table above)
- concurrency_cap: 64
- request_distribution: poisson (open-loop)
- prompt_len: log-normal, median 1500, p95 3500, min 256, max 7500 (tokens)
- max_tokens: log-normal, median 200, p95 800, min 32, max 1500
- streaming_prob: 0.7
- corpus: `client/prompts/arxiv_corpus.jsonl`,
  md5 `d2962afb0ff05d7df3131856873b41fd` (under git, deterministic)
- seed_template: `{replica}` (r01 uses seed=1, r02 seed=2, r03 seed=3)

The corpus and config are identical across pilot (n=1) and current
campaign (n=3). Verified by md5.

---

## Status snapshot (2026-05-19 afternoon ET)

```
Campaign launched: 2026-05-16T11:12 UTC
state.summary    : completed=6, running=3, failed=0
```

Completed (all r01):
- e1_r01: 36h, image sha256:a23009..., RSS slope +0.035 MB/h
- e2_r01: 36h, image sha256:1fb3d1..., RSS slope +0.109 MB/h
- e3_r01: 36h, image sha256:452c86..., RSS slope +0.058 MB/h
- a1_r01: 36h, image sha256:4f4037..., RSS slope +0.064 MB/h
- a2_r01: 36h, image sha256:1fb3d1..., RSS slope +0.217 MB/h
- e3b_r01: 36h, image sha256:452c86..., RSS slope +0.201 MB/h

Running (started ~13:00 UTC 18 May, ~22000s elapsed of 129600s as of last check):
- e1_r02 on gpu0
- e2_r02 on gpu1
- e3_r02 on gpu2

Expected r02 batch end: ~21 May 11:00 UTC.

Pending (not yet started):
- a1_r02, a2_r02, e3b_r02 (start after the r02 batch above finishes)
- All r03 batch
- Sanity run (e2 on gpu0, 6h)

---

## Results r01 vs paper Table IV

| cell | paper | n=3 r01 | factor | image identical to paper? |
|---|---|---|---|---|
| E1 V1 standalone | +9.15 MB/h | +0.035 MB/h | 260x smaller | NO (drift) |
| E2 Triton+V0     | +2.04 MB/h | +0.109 MB/h | 18x smaller  | yes |
| E3 PyT+HF sat    | +170 KB/h  | +58 KB/h    | 3x smaller   | yes |
| E3b PyT+HF low   | +179 KB/h  | +201 KB/h   | **~1.1x (REPLICATES)** | yes |
| A1 V0 standalone | +530 KB/h  | +64 KB/h    | **8.3x smaller** | **YES** (identical immutable binary) |
| A2 Triton+V1     | +20 KB/h   | +217 KB/h   | 10x larger | yes (but rate differs, see note) |

Notes:
- A2 had its target rate recalibrated from 0.90 (paper) to 1.753 (n=3)
  after smoke tests confirmed V1 is actually initialized in Triton with
  VLLM_USE_V1=1. So A2 is not directly comparable to paper A2.
- Pipeline correctness validated: `replicate_n1.py` (committed at repo root)
  reads the local n=1 CSVs and reproduces the paper numbers within 5-20%.
  So the n=3 numbers above are correct, not a pipeline artefact.

---

## What rules out which hypothesis

| hypothesis | status | evidence |
|---|---|---|
| Analytical pipeline broken | RULED OUT | replicate_n1.py reproduces paper Tab IV from n=1 CSVs (E1 +8.72 vs +9.15, A1 +505 vs +530 KB/h, etc.) |
| Corpus changed | RULED OUT | single md5, under git, deterministic |
| Args differ from paper | RULED OUT for a1/a2 | bash history of pilot vs cell yaml shows identical docker run args |
| Image drift (vllm latest) | TRUE for e1 only | a1 binary is immutable v0.7.3 yet drops 8x; cannot be image drift |
| Topology effect | LEADING HYPOTHESIS | e3b (sub-saturated, 50 req/h) replicates paper; e3 (saturated, 624 req/h, same engine) does not. Paper ran one cell at a time; n=3 runs three cells in parallel. Plausible CPU-host contention mechanism. |
| Workload regime (saturated vs sub-saturated) | PARTIAL EXPLANATION | e3 vs e3b makes this clear: leak rate drops in saturated cells under parallel topology, not in sub-saturated ones |
| Theil-Sen on 36h vs paper's 24h | RULED OUT | replicate_n1.py confirms warmup-discard and run-duration choices do not move the number more than a few percent |

---

## Decisions taken (chronological)

- 2026-05-15: Pin all images at `wosar2026_*` tags, with digests recorded
  in `engines/*/image_pin*.json`. vLLM `latest` digest captured on this
  day (sha256:a23009...). Previous digest (sha256:9eff9734..., on the
  host since the pilot of 7 May) was later pruned by docker system prune.
- 2026-05-16 11:12 UTC: Campaign launched. First attempt aborted at
  ~7h due to URL doubling bug in vllm_openai client adapter. Three runs
  archived to `~/wosar/runs_aborted_20260516_052308/`.
- 2026-05-16 (re-launch): Same day, after fixing the URL bug, campaign
  re-launched. This is the canonical run.
- 2026-05-17 23:13 UTC: All three r01 of slot batch 1 (e1, e2, e3) end.
- 2026-05-17 23:14 UTC: Slot batch 2 (a1, a2, e3b) starts.
- 2026-05-18 04:48 ET: Disk pressure on /var/lib (free dropped to 9 GB).
  Resolved by `docker system prune -f` which reclaimed 22.88 GB (mostly
  the pre-pin vllm/vllm-openai:latest dangling image). Logged in
  `campaigns/wosar2026/state/mitigations.log`.
- 2026-05-19 11:14 UTC: All three r01 of slot batch 2 (a1, a2, e3b) end.
  Round 1 complete. Slot batch 1 of r02 starts (e1, e2, e3 r02).
- 2026-05-19 afternoon ET: Health-check script extended with
  early-warning thresholds on /var/lib (WARN 20 GB, FAIL 10 GB),
  embedded docker system df snapshot on disk WARN/FAIL, and a manual
  mitigations log in `campaigns/wosar2026/state/mitigations.log`.
- 2026-05-19 afternoon ET: Decision to let the r02 batch complete
  (~30h) before any intervention. Stop campaign only after r02 batch
  finishes, then run a1 in isolation as a topology-effect test.

---

## Open TODOs

In order of priority for the next session:

1. **Wait for r02 batch end (~21 May 11:00 UTC).** Then run
   `validation_check.py` on e1_r02, e2_r02, e3_r02. If r02 slopes are
   consistent with r01 (within ~30%), the within-cell pattern is stable
   and we can act. If r02 slopes diverge wildly from r01, we have a
   second problem to investigate.
2. **Decision point after r02 batch.** Two choices:
   - **A: stop campaign, run a1 isolated.** Lose a1_r02 / a2_r02 /
     e3b_r02 (we already have r01 of each), gain a clean diagnostic.
   - **B: let the campaign continue through r03.** Then have full
     n=3 in the parallel-topology setting. Useful only if we decide to
     reframe the paper around the parallel setup; loses the chance to
     replicate the original paper.
3. **a1 isolated test.** When we decide to run it: stop campaign,
   launch only a1 on gpu0, target duration 24h (matching paper),
   identical args to current `cells/a1.yaml`. Expected: ~+530 KB/h if
   topology is the confound; ~+64 KB/h if topology is not the cause
   (in which case we need a deeper investigation).
4. **Paper narrative decision.** Conditional on a1 isolated result:
   - If +530 KB/h, narrative becomes "we replicate the pilot in a
     controlled single-cell setting; parallel-topology deployment in
     real production reduces apparent aging rates by 5-10x, which is
     itself an operationally relevant observation."
   - If +64 KB/h, narrative becomes "the pilot numbers in Table IV did
     not survive replication. Possible cause: something in the launch
     framework not captured by the bash-history args (env vars, ulimits,
     CPU pinning). Investigation continues."
5. **Backup hypothesis to test if a1 isolated does not replicate.**
   diff the bash-history docker run env from `launch_cell.py` invocation:
   ulimits, --cpus, --memory-reservation, namespaces, security-opt,
   user/uid. Any difference could be the cause.
6. **Stepness analysis (corr / K_trim / steps_per_hour).**
   `analysis/stepness.py` has been written and partially validated
   on pilot n=1 (2026-05-19). Metric panel: corr (primary, mechanism
   signature), K_trim (intensity), steps_per_hour (operational).
   See Open Question #4 for definition and classification rule.
   Remaining work:
   - Complete the K_trim CI table on n=1 (A2, E3, E3b values still
     TBD as of 2026-05-19).
   - Run on n=3 r01 (server). Compare per-cell mechanism-class
     assignment n=1 vs n=3.
   - Re-run on r02, r03 once available, compute between-run CI for
     each metric (3 replicates per cell).
   - The headline paper outcome is whether mechanism class
     (mmap / sbrk / drift) is preserved across n=1 and n=3 despite
     the slope collapse.
7. **Long-tail TODO (post-campaign).**
   - Run validation_check.py on a1_r02 / a2_r02 / e3b_r02 if/when they
     complete.
   - Update `docs/experimental_protocol.md` to reflect the actual
     executed protocol (n=3, 36h, parallel topology, Qwen2.5-7B).
   - Fix run_monitors.py manifest collision (currently a workaround).
   - Restore Docker data-root on /home per ADR-002.

---

## Files and paths

On the laptop (this repo):
- `EXPERIMENT_STATE.md` (this file)
- `replicate_n1.py` (script that reads `logs/aging_pilot_24h_*/` CSVs and
  reproduces paper Table IV numbers via Theil-Sen, used to validate the
  pipeline)
- `docs/project-wosar.md` (longer-form project doc)
- `docs/WOSAR_2026.pdf` (paper draft as submitted)
- `campaigns/wosar2026/{campaign.yaml, cells/*.yaml}` (campaign config)
- `scripts/{campaign.py, launch_cell.py, smoke_test_run.sh,
  campaign_health.sh}` (campaign machinery)
- `monitoring/{gpu_monitor.py, proc_monitor.py, system_monitor.py,
  run_monitors.py, find_engine_pid.py, _common.py}` (monitoring)
- `client/{run_client.py, config.yaml, prompts/arxiv_corpus.jsonl,
  protocols/*.py}` (workload)
- `analysis/validation_check.py` (post-run verdict per run)
- `engines/{vllm_standalone, triton_vllm, pytorch_naive}/` (engine
  definitions, Dockerfiles, model_repository for Triton)

On the server (cci-csgpu11):
- `~/wosar/llm-serving-bench/` (this repo, checked out)
- `~/wosar/runs/wosar2026_<cell>_r<NN>/` (current campaign runs)
- `~/wosar/runs_n1_baseline/aging_pilot_24h_*/` (paper pilot runs,
  May 6-15; CSVs available, but pilot launch scripts not archived: args
  must be recovered from bash history)
- `~/wosar/runs_aborted_20260516_052308/` (failed first attempt)
- `~/wosar/hf_cache/` (HuggingFace cache, mounted into all containers)

---

## Standard commands (server, in ~/wosar/llm-serving-bench)

```bash
# Periodic health check, run every 6-12h during the campaign
bash scripts/campaign_health.sh 2>&1 | tee /tmp/health.log
echo "exit code: ${PIPESTATUS[0]}"
# Exit 0=OK, 1=WARN (campaign OK, inspect when convenient), 2=FAIL (intervention needed)

# Per-run post-completion verdict
python3 analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_<cell>_r<NN>

# All r01 verdicts in batch
for cell in e1 e2 e3 a1 a2 e3b; do
  echo "=== ${cell}_r01 ==="
  python3 analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_${cell}_r01
done

# Log a manual mitigation (e.g. after running docker prune by hand)
echo "$(date -Iseconds) | <category> | <free-text note>" \
  >> campaigns/wosar2026/state/mitigations.log
# categories: disk_prune, container_restart, engine_relaunch,
#             gpu_intervention, workload_param_change, host_intervention
```

---

## Pipeline analytical details (for paper)

- Trend detection: Mann-Kendall with Hamed-Rao correction for
  autocorrelation. Significance at p < 0.01.
- Slope estimation: Theil-Sen with exact 95% CI based on order
  statistics of pairwise slopes, variance inflated by lag-1 AR(1)
  factor (1+rho)/(1-rho).
- Multi-test correction: Benjamini-Hochberg FDR at q = 0.10 across all
  indicators and cells (approx 200 tests).
- Decision rule: a trend is declared significant only when both the
  Mann-Kendall test rejects the null and the Theil-Sen 95% CI excludes
  zero. (analysis/validation_check.py currently checks Mann-Kendall +
  positive slope only; the full pipeline including CI and BH-FDR is in
  the analysis notebook used to generate Table IV.)
- Magnitude criterion: an open question is whether to add an
  operationally meaningful threshold (e.g. slope > 1 MB/h) on top of
  the statistical significance, to avoid declaring trends significant
  on slopes of practical irrelevance. Discussed but not implemented.

---

## Open questions for the next session

1. After r02 batch: do the r02 slopes match r01? If yes, topology
   confound hypothesis stays in play. If no, deeper problem.
2. After a1 isolated: does it return to ~+530 KB/h? Decides paper
   narrative.
3. If topology is confirmed as the confound, do we have time to run a
   sequential n=3 (6 cells x 3 replicas x 24h = 18 days)? No. So the
   paper has to be framed around the realistic deployment (parallel
   topology) or around a single-cell controlled replication of the
   pilot. Decide before 30 June.
4. **Does the step-wise RSS/VMS lock-step pattern persist in the n=3
   data?** This is the qualitative finding the authors care most about,
   independent of slope magnitudes. Even if the leak rates have dropped
   by an order of magnitude in n=3, the question is whether the
   signature is still present:
   - On E2 (Triton + vLLM V0): the cleanest case in the paper.
   - On A1 (vLLM V0 standalone): also showed step-wise pattern.
   - On A2 (Triton + V1) and E1 (V1 standalone): NOT step-wise in the
     paper (E1 was quasi-linear).

   What to check (post-completion, on each r01 and onwards):
   - Plot RSS(t) and VMS(t) on the same axis with second-by-second
     resolution. Look for flat-then-jump pattern with both curves
     stepping at the same instants.
   - Compute first-difference dRSS/dt and dVMS/dt. The step events
     should appear as paired spikes.
   - Cross-correlation of dRSS and dVMS at lag 0 should be near 1
     during the step events.
   - If the slope magnitude has collapsed but the qualitative pattern
     is preserved, the finding survives and the paper can keep it as
     the centerpiece of Section IV.E. If the pattern has disappeared
     (smooth growth instead of step-wise), the finding does not
     replicate and the section needs rewriting.

   Implementation: add an analysis script that loads the proc CSVs of
   a run, extracts rss_bytes and vms_bytes, and reports (a) the number
   of detected step events above a threshold, (b) the mean step
   amplitude, (c) the dRSS-dVMS lag-0 correlation, (d) optionally a
   plot. Run on all completed r01 immediately. Do not wait for r02 or
   r03 to start this analysis.

   **Metrics policy for the paper (calibrated on pilot n=1, 2026-05-19).**
   A single scalar (K alone) proved fragile: raw kurtosis is dominated
   by single extreme outliers (E1 pilot showed K=16412 with bootstrap
   CI [2, 16893] driven by one anomalous sample). We adopt a
   three-metric panel per cell.

   1. **Primary — rss_vms_corr**: lag-0 cross-correlation of dRSS and
      dVMS post-warmup. Identifies the MECHANISM:
      - corr > 0.8 → lock-step → mmap-style allocation (RSS and VMS
        grow together: kernel-mapped blocks, never released)
      - corr < 0.5 → not lock-step → either continuous drift or
        sbrk-style heap-internal accumulation
   2. **Secondary — K_trim**: excess kurtosis of ΔRSS after
      winsorization at the 99.9 percentile. Quantifies the INTENSITY
      of the step-wise pattern, robust to single outliers:
      - K_trim > 10 → tail-heavy → punctuated dynamics
      - K_trim < 5 → gaussian-like → continuous drift
   3. **Operational — steps_per_hour**: count of ΔRSS > 1 MB per
      hour of runtime. Reader-friendly descriptor: "this cell shows
      N step events of at least 1 MB per hour."

   Raw K is reported alongside K_trim for transparency but does not
   enter the classification rule.

   **Three-category classification rule.** Apply jointly:

   | pattern              | corr   | K_trim | mechanism interpretation                                                                  |
   |----------------------|--------|--------|-------------------------------------------------------------------------------------------|
   | mmap-style step-wise | > 0.8  | > 10   | RSS and VMS step together at discrete events → kernel-mapped blocks never returned        |
   | sbrk-style step-wise | < 0.5  | > 10   | RSS steps without paired VMS steps → glibc heap-arena extension, no kernel mmap involved  |
   | continuous drift     | < 0.5  | < 5    | smooth small-grain accumulation, no big steps                                             |
   | (border)             | mixed  | mixed  | needs n=3 confirmation                                                                    |

   **Why this is a new finding versus the n=1 paper.** The paper text
   (Section IV.E) mentions both mmap and sbrk-extended heap as
   alternative hypotheses but does NOT distinguish them. The
   (corr, K_trim) pair separates them for the first time. In
   particular, A1 in the paper was lumped with the other
   "three-of-four step-wise cells"; the metric panel reveals A1
   belongs to a DIFFERENT mechanism class than E2 (sbrk-style vs
   mmap-style). This is a paper-worthy refinement obtainable without
   any new instrumentation, just from the existing proc CSVs.

   **Baseline K_n1 values (pilot CSVs, 2026-05-19).**

   | cell | corr  | K_trim | classification             |
   |------|-------|--------|----------------------------|
   | E1   | 0.32  | 0.0    | continuous drift           |
   | E2   | 0.93  | +356   | mmap-style step-wise       |
   | A1   | 0.20  | +402   | sbrk-style step-wise       |
   | A2   | 0.58  | TBD    | border (intermediate)      |
   | E3   | 0.40  | TBD    | border                     |
   | E3b  | 0.45  | TBD    | border                     |

   K_trim values for A2, E3, E3b to be filled in after re-running the
   updated stepness.py on n=1.

   **Replication plan against n=3.**
   1. Re-run stepness.py on all pilot n=1 CSVs to complete the
      baseline table above.
   2. Run on n=3 r01 (server). Compare per-cell:
      corr_n3 vs corr_n1, K_trim_n3 vs K_trim_n1.
   3. The paper claim survives if MECHANISM CLASS is preserved
      between n=1 and n=3 even when slope magnitudes have collapsed:
      - E2: corr_n3 stays > 0.8 → mmap-style intact
      - A1: corr_n3 stays < 0.5 AND K_trim_n3 > 10 → sbrk-style intact
      - E1: corr_n3 stays low AND K_trim_n3 low → drift class confirmed
   4. If class assignments hold across n=1 and n=3 despite 3-260x
      slope drop, this is the strongest paper claim available:
      "the aging MECHANISM is invariant under the operational regime;
      only the RATE is workload-modulated."

   Why this metric set was chosen: (a) all three quantities are
   one-line definitions, no bibliography needed beyond a Fisher
   footnote for K_trim; (b) robust to single outliers (Winsor on
   K, correlation is bounded in [-1,1]); (c) the metric panel
   discriminates mechanism, not just magnitude; (d) trivially
   comparable across cells, replicas, n=1 and n=3.

   **Analysis sequencing (important).** Compute K on the n=1 pilot
   CSVs FIRST, before looking at any n=3 numbers. The n=1 K values
   are the baseline against which the n=3 K values are interpreted.
   Sequence:
   1. Run the kurtosis script on the local n=1 CSVs in
      `logs/aging_pilot_24h_*/` (E1, E2, E3, E3b, A1, A2). Record
      K_n1 per cell. Expectation from paper Figure 2(b):
      K_n1(E2) and K_n1(A1) are large (step-wise dominant cells),
      K_n1(E1) is small (quasi-linear), K_n1(E3, E3b, A2) somewhere
      in between but with documented step-wise visual signature on
      three of the four factorial cells.
   2. Run the same script on the n=3 r01 CSVs on the server
      (`~/wosar/runs/wosar2026_<cell>_r01/`). Record K_n3 per cell.
   3. Compare per cell. The headline result for the paper is
      whether K_n3 / K_n1 stays close to 1 across cells, despite
      the slope ratios collapsing by 3-260x. Specifically:
      - If K_n3(E2) is in the same order of magnitude as K_n1(E2),
        the step-wise mechanism is intact and the paper's central
        qualitative claim survives.
      - If K_n3(E2) collapses to single digits, both the magnitude
        AND the mechanism have changed under the parallel topology.
        This is a worse outcome but still a clean finding: it would
        say the mmap-style allocation behavior is itself
        topology-dependent.
   4. Once r02 and r03 finish, repeat the analysis to estimate
      within-cell variance of K (3 replicates per cell). Bootstrap
      CI from a single run is a within-run uncertainty; replicate
      CI is the between-run uncertainty, the more important one.

   Implementation note: a small script `analysis/stepness.py`
   should be added to the repo. Inputs: `--run-dir` for a single
   run, or `--logs-root` to scan all `aging_pilot_*` directories
   in `logs/`. Outputs: per-run row with cell_id, K, bootstrap CI,
   number of post-warmup samples, RSS-VMS lag-0 correlation. As
   of 2026-05-19 this script does not yet exist and is the next
   concrete deliverable.
