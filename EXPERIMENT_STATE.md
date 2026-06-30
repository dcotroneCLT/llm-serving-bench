# EXPERIMENT_STATE.md

Living hand-off document for the WoSAR 2026 n=3 campaign. Updated by hand
whenever something material changes. Designed so a new chat session (or
a co-author) can pick up the thread in under five minutes.

Last updated: 2026-06-07 (ET).

---

## How to use this document (for a new chat session)

If you are an LLM assistant picking this up, read this section first.

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
1. Read "Paper framing" section below for the decision that drives
   everything else in this document.
2. Read the Status snapshot for where we are in the campaign.
3. Read "Open TODOs" and "Open questions" at the bottom for the
   action items.
4. Suggest he runs `bash scripts/campaign_health.sh` on the server
   first thing, to get a fresh state. Then suggest
   `python3 analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_<cell>_<rXX>`
   on any newly-completed runs since this doc was written.

**Companion documents.**
- `docs/WOSAR_2026.pdf`: the submitted (n=1) preprint. Background
  reading for the study design, NOT a reference against which the
  n=3 campaign is validated. See "Paper framing" below.
- `docs/project-wosar.md`: longer-form project doc, older snapshot.
  This file (EXPERIMENT_STATE.md) supersedes it as the operational
  hand-off; `project-wosar.md` is kept for historical reference and
  detailed framework documentation (e.g. the catalog of critical
  fixes during framework development).

**When in doubt, ask before acting.** Especially before suggesting to
stop the campaign or restart anything. The host window is fixed and
every wasted run is a wasted day.

---

## Standing constraints (extension campaign) — READ BEFORE RUNNING ANYTHING

### SC-1. Cross-system vLLM pin (hard)
All three systems (Dynamo / Triton+vLLM / standalone) run the IDENTICAL vLLM,
pinned by image digest, from the NATIVE three-way intersection; verified on the
box by `pip show vllm` (remote notes are NOT authoritative). Full statement +
version map + current pin (0.20.1) in **`docs/extension_pin_constraint.md`**.
Never bump one system's release alone.

### SC-2. Disk-space management (hard, enforced in code — not by discipline)
The preprint campaign hit disk exhaustion (Docker images/layers + container
json-file logs on the 126G `/var/lib` data-root, NOT the trace CSVs). The
extension is higher-risk (3 large new images, per-component monitoring multiplies
Dynamo CSVs, 2 GPUs sampled). Enforce in code:

1. **Pre-run free-space GATE** — refuse to start a run if free space on the
   runs-root AND the docker data-root is below a configurable threshold.
   STATUS: implemented in `launch_cell.py` + `attach_run.py` (helper
   `require_free_space`).
2. **Mid-run disk WATCHDOG** — if free space drops below a floor during a run,
   trigger a GRACEFUL teardown (finalize manifest, stop cleanly) and mark the
   run for reschedule; never let a run die uncontrolled. STATUS: to land with
   the lifecycle/orchestration refactor (post-gate), in the supervise loops.
3. **Docker container log rotation** on every `docker run`:
   `--log-opt max-size=50m --log-opt max-file=3`. STATUS: implemented in
   `deploy/dynamo/*.sh`, `launch_cell.build_docker_run_cmd`, README.
4. **Gzip rotated CSV segments** on rotation (client per-request CSV is the
   largest single stream). STATUS: to land with the lifecycle, together with
   making the analysis readers (`aging_io`, `aging_trends`, gate checker,
   calibrate) transparently read `.csv` and `.csv.gz`. Deferred deliberately so
   the CSV read path is not changed in the same step the gate validates it.
5. **Images pre-pulled ONCE, reused across all runs** (never a per-run pull;
   launch_cell/attach verify the local digest, they do not pull). **Docker
   data-root MUST be on /home (6.9T), not /var/lib (126G)** — ADR-002's move was
   lost; restore it before the campaign (ops action on the box: set
   `data-root` in `/etc/docker/daemon.json` to `/home/dcotrone/docker-data`,
   restart docker, re-pull the 3 images). Cleanup discipline: `docker system df`,
   `docker image prune`, remove superseded pins.
6. **Footprint estimate (computed 2026-06-29).** Per 48h run, new layout:
   - gpu 1 Hz x 2 GPUs ~= 100 MB; system 5 s ~= 7 MB; proc/per-component:
     single-process ~7 MB, Dynamo (5 components + 2 aggregates @ 5 s) ~50 MB;
     client per-request (the largest) ~70-300 MB depending on rate.
   - => ~0.15-0.25 GB/run single-process, ~0.26-0.46 GB/run Dynamo (uncompressed);
     gzip ~5-10x smaller.
   - x ~57 runs + calibrations + 1-2 7-day runs => ~20-35 GB uncompressed
     (~3-7 GB gzipped) of TRACE data -> trivial on /home (6.3T).
   - **Conclusion: the trace CSVs are NOT the disk risk.** The risk is Docker
     images (~75 GB for 3) + 48h container json-file logs, both on the 126G
     data-root. Hence the priority order: data-root on /home (#5) + log rotation
     (#3) >> CSV gzip (#4).

---

## Paper framing (the key decision)

The camera-ready WoSAR 2026 paper is a **standalone study on n=3 data**.
It is not framed as a replication of the n=1 preprint. The narrative
follows the same skeleton as the preprint (same research questions,
same factorial cells, same family of indicators), but every numerical
claim, every figure, and every classification rule in the camera-ready
is computed on the n=3 campaign data and stands on its own.

What the n=3 paper inherits from the n=1 preprint, as design and
narrative scaffolding:
1. Aging exists in modern LLM serving on the GPU. Same RQ1.
2. Aging is localized in the framework orchestration layers, not the
   inference compute path. Same RQ2.
3. The leak rate is a property of the full deployment, not a single
   component. Same 2x2 factorial across engine generation (V0/V1)
   and hosting layer (standalone/Triton).
4. Step-wise lock-step growth of RSS and VMS in V0-based engines is
   the most actionable qualitative finding.

What is **new in the n=3 camera-ready** vs the preprint:
- **n=3 replication.** Three replicas per cell give a between-run
  variance bound, addressing the primary threat-to-validity of the
  preprint.
- **36h window per run.** Up from 24h; partial improvement against
  the late-onset-effects TTV.
- **BH-FDR control** at q=0.10 across the joint family of
  (run_id, indicator) tests. The preprint did per-indicator MK only.
- **Stepness mechanism panel** (`rss_vms_corr`, `K_trim_dRSS`,
  `K_trim_dVMS`, `steps_per_h_1mb`, `mean_top1_step_mb`) with the
  five-class taxonomy: mmap-style / sbrk-style / VAS-only /
  uncorrelated / continuous drift, plus border for unusable VMS or
  out-of-bin rows. The preprint conflated mmap and sbrk under
  "step-wise"; the RSS/VMS split is the paper-worthy mechanism
  refinement.
- **Realistic parallel multi-tenant topology.** Three GPUs in
  parallel rather than the preprint's sequential single-GPU. Declared
  as a feature of the design (Section III) and discussed in Threats
  to Validity (Section V) as the operational regime the paper reports
  on.
- **e3 vs e3b rate-sensitivity ablation** at proper saturation (e3
  drops ~15-16% at 0.174 rps, e3b drops <2% at 0.050 rps; see
  Section IV.D below).

The preprint Table IV numbers do not appear in the camera-ready. They
are referenced as the prior state of the art that motivated the n=3
study design, and that is the only relationship between the two.

---

## Background: prior state of the art (n=1 preprint)

Context only. Kept short. The preprint informed the design of the
n=3 campaign (cell selection, target rates, monitoring stack) but
does not feed into any numerical claim of the camera-ready.

The preprint (single-run 24h per cell, single-GPU sequential)
established three findings on the same hardware and model:

1. **Software aging exists in modern LLM serving on the GPU.** All
   three primary deployments (vLLM standalone V1, Triton + vLLM V0,
   naive PyTorch + HF) showed monotonically increasing process-private
   memory over 24h; MK-significant under FDR correction.
2. **The aging surface lies in the framework orchestration layers,
   not in the inference path.** The naive PyTorch baseline leaked the
   least; production-grade engines leaked 1-2 orders of magnitude
   more. Since the inference compute path is identical across the
   three (same Qwen2.5-7B weights, same attention math), aging must
   live in orchestration.
3. **The leak rate is a property of the full deployment.** A 2x2
   factorial across engine generation (V0 vs V1) and hosting layer
   (standalone vs Triton wrapper) showed leak rates spanning nearly
   three orders of magnitude across the four cells, with engine and
   hosting interacting.
4. **Step-wise lock-step growth of RSS and VMS in V0-based engines
   (qualitative, Figure 2b of the preprint).** Memory stays flat for
   hours, then jumps abruptly by several MB at discrete step events,
   with RSS and VMS stepping together. Diagnostic analysis on
   secondary indicators (CPU, request rate, voluntary ctx-switches,
   Python GC) showed no correlation with the step events. Hypothesis:
   periodic mmap-style allocation of new blocks from the kernel that
   are never released. The preprint left the mmap-vs-sbrk mechanism
   alternative unresolved; the n=3 paper resolves it with the metric
   panel.

Two secondary observations from the preprint:
- Client-side aging (latency, TTFT, throughput, drop rate) was
  essentially undetectable over 24h on all three primary engines.
- One GPU-side aging signature on A1 (vLLM V0 standalone): VRAM
  grew at +124 MB/h, accumulating ~3 GB over 24h. No other run
  showed any VRAM trend.

Preprint Threats-to-Validity that the n=3 campaign addresses:
- **Single-run design** (the priority): no run-to-run variance bound.
  n=3 fixes this.
- **24h window may miss late-onset effects**: n=3 uses 36h, partial
  improvement.
- **Confounded factorial** (vLLM version drift across cells):
  persists as a residual confound, declared explicitly in TTV.
- **Single hardware, single model**: persists.

The n=3 campaign was **not** designed to address parallel-vs-sequential
topology. The preprint ran one cell at a time; the n=3 campaign runs
three cells in parallel on three GPUs to fit inside the 2-week host
window. This design choice is declared as a feature of the n=3 paper:
it reports on a realistic multi-tenant deployment.

---

## TL;DR (operational, 2026-06-06)

**CAMPAGNA COMPLETATA.** 18 production run (6 cells × 3 replicas
× 36h ciascuno) + 1 sanity run (e2_r99: 6h, e2 on gpu0 invece di
gpu1, per Section V threats-to-validity sulla GPU-index sensitivity).
State summary: `completed=19, running=0, failed=0`. 0 FAIL, 10 soft
WARN (3 `proc.sample_error` storici di r01 batch 2 + 3 `client.dropped`
su e3 r01/r02/r03 a 14.3-15.7% = regime stazionario confermato su
n=3). Campagna terminata intorno al 2026-05-25.

**Dati sincronizzati in locale sul Mac** (2026-06-06): tutte le 19
run directory in `~/Documents/Github/llm-serving-bench/runs/`
(2.2 GB totali, gitignored). Trasferimento via tar.gz + scp
(611 MB compressi) dopo che rsync incrementale aveva avuto
problemi di rete stallita. Pipeline locale validata bit-exact
vs server: `aging_trends.py` su `wosar2026_e1_r01` ridà USS slope
6957.82 byte/h con CI [1495.5, 22341.8] byte/h, identico al numero
documentato il 23 maggio.

**Pulizia eseguita** (2026-06-06):
- Server: tar di trasferimento cancellato (611 MB), `runs_aborted_20260516_052308`
  cancellato (54 MB), `runs_failed_attempts` cancellato (97 MB).
  Conservati `~/wosar/runs/wosar2026_*`, `~/wosar/runs_n1_baseline/`,
  immagini Docker `wosar2026_*`, `~/wosar/hf_cache/`.
- Mac: tar di trasferimento cancellato (611 MB). `runs/` directory
  conservata (gitignored, 2.2 GB).

**Stato analisi paper-grade n=3.**
- aging_trends.py girato in locale su tutti i 18 production
  run, output in `/tmp/wosar_n3/*_trends.csv`. **Anomalia da
  indagare nella prossima sessione**: 405 righe totali vs ~520
  attese (alcuni run mancano indicator, probabilmente per
  detection di GPU index o client missing). I file CSV per-run
  comunque tutti generati. Da rilanciare con debug per capire
  cosa manca.
- fdr_aggregate.py e aggregate_slopes.py NON ancora lanciati su
  n=3 (sono i comandi che producono la **tabella per-cell
  paper-grade source of truth** per Section IV.C). Pipeline pronta
  in repo, basta lanciarla.
- stepness.py ancora da lanciare sui 6 run di r03 per consolidare
  la classificazione mechanism a 5 classi su n=3 (su n=2 era già:
  E2 mmap-style step-wise × 2, altre 5 celle continuous drift × 2).

**Stato scrittura paper.** Preprint (`docs/WOSAR_2026.pdf`, n=1
24h, 8 pagine) ricaricato e riletto integralmente il 2026-06-06.
**LaTeX sources del preprint vivono su Overleaf** (canonical source,
non nel repo Git). La cartella `paper/` nel repo è vuota a parte
`.gitkeep` e `PAPER_UPDATE_PLAN.md` (documento di riferimento per
la rescrittura, vedi sotto).

**Workflow per la scrittura del camera-ready** (deciso 2026-06-06):
- Domenico scrive personalmente il paper su Overleaf interagendo
  con l'assistente. NON deleghiamo il drafting full-text
  all'assistente.
- Si lavora **sezione per sezione**: Domenico discute una sezione
  con l'assistente, scrive il LaTeX su Overleaf, pasta il testo
  in chat, l'assistente commenta/suggerisce/propone alternative,
  Domenico itera sul Overleaf.
- `paper/PAPER_UPDATE_PLAN.md` resta come **reference document**
  per sapere cosa cambia in ogni sezione (preprint → camera-ready),
  con draft di testo proposti come spunto. NON è da incollare
  meccanicamente nel LaTeX; è guida e checklist.
- I numeri paper-grade (Tabelle III, IV, V e dati Figure 2) vanno
  prodotti dalla pipeline `analysis/aggregate_slopes.py` +
  `analysis/stepness.py` sui 18 run n=3 in locale, una volta sola,
  e poi citati nel paper.

Five findings are now locked in on n>=2 data (will appear in the
camera-ready regardless of r03 outcome):

- **e3 drop rate at saturated load.** PyTorch + HF naive at 0.174 rps
  drops ~15.6-15.7% of offered requests, confirmed on n=2 (r03 in
  progress at 13%, may rise to ~15% by end of window). e3b at 0.050
  rps drops <2%. Capacity ceiling is a property of the naive baseline
  at saturated load. Section IV.D rate-sensitivity ablation.
- **Mechanism class headline: E2 is the only step-wise cell on n=3
  campaign; all other 5 cells are continuous drift.** Under the n=3
  setting, mmap-style step-wise allocation (RSS+VMS lock-step, top 1%
  ~1.7-1.8 MB) emerges exclusively in the Triton + vLLM V0 deployment
  (E2), class and magnitude both reproducible across n=2 (n=3 pending
  r03). All other deployments exhibit continuous drift with NO
  MB-scale step events. Stronger finding than "five mechanism classes
  observed": canonical example + null findings on 5 contrast cells.
- **USS adopted as canonical leak indicator for the camera-ready.**
  USS is paper-grade significant on 12/12 n=2 runs; RSS is significant
  on 11/12 (a1_r02 fails on CI lower bound = 0.0, an AR(1) inflation
  artifact when rho_RSS = 0.99). Point estimates of USS and RSS agree
  within <10% on all 12 runs. USS is also semantically cleaner
  (process-private resident memory; RSS includes shared mappings that
  are a confound). Section IV.B (instrumentation) declares USS as
  primary; RSS reported alongside as secondary.
- **VAS-only smooth drift on PyTorch+HF naive, r02-only.** On both E3
  (0.174 rps) and E3b (0.050 rps), the second replica shows
  paper-grade significant VMS growth at 52 MB/h and 127 MB/h, with
  RSS at only 31 KB/h and 109 KB/h on the same runs (ratio
  ~1000-1700x). Empirically verified on e3b_r02 (`vms_bytes` 38.5 →
  46.0 GB over 36h, num_fds 53→52, num_threads 200→199, rss_bytes
  +20 MB): rules out FD-leak and thread-stack growth, leaves only
  anonymous mmap reservation by the PyTorch CUDA caching allocator
  host-side as the compatible mechanism. K_trim_dVMS <= 2.2 → smooth
  drift, not step-wise. Phenomenon absent on r01 of both cells; r03
  will determine whether r02 was the outlier or r01 was.
- **Pipeline hardened and validated.** Paper pipeline is end-to-end
  in-repo: `aging_trends.py` + `fdr_aggregate.py` for slope+CI+FDR,
  `stepness.py` for the (corr, K_trim_dRSS, K_trim_dVMS, steps>1MB/h,
  top1%_step) panel with five-class taxonomy + priority short-circuits
  (low-step → drift, VMS missing/unusable → border). Five hardening
  fixes landed 2026-05-20/21: math fallback → operational fallback,
  classification short-circuit on both-axes-fallback, top1%_step
  zero-heavy bug, VMS missing/unusable handling, PID-aware diff
  segmentation plus aligned top-k timestamps and sparse-bootstrap
  handling.

Open headline questions for the paper (within-n=3, not vs preprint):

1. Per-cell **between-replica variance** of the USS slope. Full n=2
   CI-aware picture (12 runs): 4/6 cells CI-compatible (a1, a2, e1,
   e3b), 2/6 CI-disjoint (e2, e3). Both disjoint cells are in slot
   batch 1, both have r02 > r01 (gap 4.2 KB/h and 1.9 KB/h
   respectively on USS). r03 will settle whether the gap is real or
   r02 was the outlier.
2. **Triton-wrapper between-run variance hypothesis.** USS r02/r01
   ratios suggest the Triton wrapper amplifies between-run
   variability regardless of the underlying engine: standalone cells
   (e1, a1) ratios 0.26 and 1.02 (tight); Triton cells (e2, a2) ratios
   8.2 and 0.14 (highly variable); PyTorch+HF cells (e3, e3b) ratios
   2.8 and 2.7 (intermediate). r03 will determine whether this is a
   structural property or noise on n=2.
3. Per-cell **step-wise classification** stability across r03 — E2
   mmap-style is reproducible on n=2, expected to hold on r03.
   The other 5 cells should consolidate continuous drift if the
   n=2 picture stays.
4. e3 drop rate **mechanism**: time-clustered vs uniform, correlated
   with GPU 2 VRAM/util spikes or not, behavior stable across
   r01/r02/r03.

---

## Paper

- Venue: WoSAR 2026 (ISSRE workshop on Software Aging and Rejuvenation)
- Deadline: 30 June 2026
- Authors: Domenico Cotroneo (UNC Charlotte), Bojan Cukic (UNCC)
- Working title: "The Aging Surface of LLM Serving Engines: An Empirical Study"
- **Preprint (n=1) version**: `docs/WOSAR_2026.pdf`. Used as background
  scaffolding for the camera-ready, not as a reference to replicate.
- **Camera-ready (final) version**: standalone paper on n=3 campaign
  data, on the same narrative skeleton as the preprint (RQs, factorial,
  step-wise mechanism), with all numbers, figures, and classifications
  computed on n=3.

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
  to what the preprint used)
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

## Status snapshot (2026-05-23 afternoon ET)

```
Campaign launched: 2026-05-16T11:12 UTC
state.summary    : completed=12, running=3, failed=0
```

Completed: all 12 r01+r02 runs across the 6 cells. All paper-grade
significant on USS (12/12), 11/12 on RSS (a1_r02 fails on CI lower
bound = 0 artifact, see "Pipeline note" below). All PASS on
validation_check.

**Paper-grade USS slopes** (aging_trends + fdr_aggregate, decision
rule = MK p<0.01 AND Theil-Sen CI excludes 0 AND bh_reject=True).
USS is adopted as the canonical leak indicator for the camera-ready;
see "Pipeline analytical details" below for the rationale.

| cell | r01 slope  | r01 CI [lo, hi]         | r02 slope   | r02 CI [lo, hi]           | CI relation             |
|------|------------|---------------------------|--------------|-----------------------------|-------------------------|
| a1   | 12.9 KB/h  | [9.5, 16.0] KB/h         | 13.2 KB/h    | [12.9, 13.4] KB/h          | overlap (r02 CI inside r01) |
| a2   | 42.5 KB/h  | [15.6, 61.9] KB/h        | 5.9 KB/h     | [1.4, 87.4] KB/h           | overlap (r02 CI contains r01) |
| e1   | 7.0 KB/h   | [1.5, 22.3] KB/h         | 1.8 KB/h     | [0.7, 23.8] KB/h           | overlap massive          |
| e2   | 19.1 KB/h  | [10.9, 29.0] KB/h        | 157.2 KB/h   | [33.2, 228.1] KB/h         | DISJOINT (+4.2 KB/h gap) |
| e3   | 11.0 KB/h  | [5.0, 19.5] KB/h         | 31.1 KB/h    | [21.4, 87.9] KB/h          | DISJOINT (+1.9 KB/h gap, barely) |
| e3b  | 38.0 KB/h  | [15.7, 132.2] KB/h       | 103.3 KB/h   | [56.2, 206.1] KB/h         | overlap (intersection [56.2, 132.2]) |

**Reading n=2.** 4/6 cells CI-compatible across replicas (a1, a2,
e1, e3b). 2/6 CI-disjoint (e2 with 4.2 KB/h gap, robust;
e3 with 1.9 KB/h gap, borderline). a1 is the most reproducible cell
of the campaign (r02/r01 ratio 1.02, CI of r02 inside CI of r01).
Both disjoint cells are in slot batch 1; both have r02 > r01 in the
disjoint direction. r03 will settle whether the gap is structural
or noise on n=2.

**Paper-grade RSS slopes** (for cross-reference; point estimates
agree with USS within <10% on all 12 runs):

| cell | r01 slope  | r01 CI [lo, hi]         | r02 slope   | r02 CI [lo, hi]           | RSS verdict             |
|------|------------|---------------------------|--------------|-----------------------------|-------------------------|
| a1   | 12.9 KB/h  | [9.5, 16.0] KB/h         | 13.2 KB/h    | [0.0, 18.0] KB/h           | r02 fails: CI floor=0 (rho_RSS=0.99 inflation; USS recovers it) |
| a2   | 43.3 KB/h  | [17.8, 62.0] KB/h        | 3.2 KB/h     | [1.3, 98.0] KB/h           | overlap |
| e1   | 7.0 KB/h   | [1.5, 22.3] KB/h         | 1.8 KB/h     | [0.7, 23.8] KB/h           | overlap |
| e2   | 21.7 KB/h  | [12.9, 31.6] KB/h        | 161.4 KB/h   | [40.3, 231.4] KB/h         | DISJOINT (+8.7 KB/h gap) |
| e3   | 11.0 KB/h  | [4.9, 20.1] KB/h         | 31.1 KB/h    | [21.4, 88.0] KB/h          | DISJOINT (+1.3 KB/h gap) |
| e3b  | 38.0 KB/h  | [15.6, 135.6] KB/h       | 108.9 KB/h   | [58.5, 210.8] KB/h         | overlap |

**Paper-grade VMS slopes** (decision rule applied; only 2/12 pass):

| run     | slope     | CI                | mk_p     | sig | mechanism note |
|---------|-----------|---------------------|----------|-----|----------------|
| e3_r02  | +52.2 MB/h  | [20.9 KB, 77.3 MB]/h | 7.7e-08 | True | VAS-only smooth drift |
| e3b_r02 | +127 MB/h   | [5.3 KB, 179 MB]/h   | 9.1e-05 | True | VAS-only smooth drift |
| other 10 runs | 0 or near-zero | — | — | False | no significant VMS growth |

Two "near misses" (`bh_reject=True` but CI lower bound = 0 from
AR(1) inflation, so fail the strict decision rule): a2_r01 at +37.5
KB/h and e2_r02 at +37.8 KB/h. Magnitudes 3 orders below the
PyTorch+HF VAS finding, not worth promoting.

**VMS-only growth empirical verification (e3b_r02)**. Direct
inspection of the proc CSV at run start vs end (raw values, no
fitting):

```
VMS start: 38.52 GB     RSS start: 1.62 GB
VMS end:   46.04 GB     RSS end:   1.64 GB
VMS delta: +7.52 GB     RSS delta: +20 MB
num_fds:    53 → 52     num_threads: 200 → 199
```

VMS grew by 7.52 GB while RSS grew by 20 MB (ratio ~375x), with
num_fds and num_threads both flat-to-slightly-decreasing. This
empirically rules out file-mapping accumulation (would show in
num_fds) and thread-stack growth (would show in num_threads), and
narrows the mechanism to anonymous mmap reservation by the PyTorch
CUDA caching allocator host-side metadata layer. K_trim_dVMS=2.2 →
smooth, not step-wise.

**Pipeline note 1 (calibration gotcha vs validation_check).**
validation_check.py and aging_trends.py give the SAME point estimate
on r02 (smooth-ish runs) but DIFFER by ~5x on r01 (step-heavy runs).
validation_check operates on raw 5s samples; aging_trends.py
downsamples to 60s windows before Theil-Sen. On step-heavy series
the two methods are not equivalent. For the paper, only
aging_trends + fdr_aggregate is paper-grade. validation_check
remains a per-run sanity gate (PASS/FAIL on trend direction), not
a slope source.

**Pipeline note 2 (RSS CI floor on a1_r02).** On a1_r02 alone,
lag1_rho_RSS = 0.99 while lag1_rho_USS = 0.005. The AR(1) variance
inflation factor (1+rho)/(1-rho) is ~199x on RSS, ~1.01x on USS.
The RSS Theil-Sen CI gets blown to [0.0, 17956] KB/h (lower bound at
floor=0, fails "CI excludes 0"), while the USS CI is a tight
[12.9, 13.4] KB/h. The slope point estimates agree (RSS 13.16 KB/h,
USS 13.17 KB/h). The anomaly is isolated to this single run; in the
other 11 runs RSS and USS rho values are quasi-identical. The case
for USS as canonical leak indicator is partly built on this
robustness.

**Validation_check slopes (sanity-gate values only, not paper-grade):**
- e1_r01: +0.035 MB/h, drop trivial
- e2_r01: +0.109 MB/h, drop trivial
- e3_r01: +0.058 MB/h, drop 15.7%
- a1_r01: +0.064 MB/h, drop trivial
- a2_r01: +0.217 MB/h, drop trivial
- e3b_r01: +0.201 MB/h, drop trivial
- e1_r02: +0.002 MB/h, drop trivial
- e2_r02: +0.161 MB/h, drop trivial
- e3_r02: +0.031 MB/h, drop 15.6%

(r02 batch 2 sanity values not re-tabulated here; aging_trends +
fdr_aggregate is the source of truth from now on.)

Running (r03 batch 1, started ~22 May UTC):
- e1_r03 on gpu0 (~21h elapsed, ~15h to go)
- e2_r03 on gpu1 (~21h elapsed, ~15h to go)
- e3_r03 on gpu2 (~21h elapsed, ~15h to go)

Expected r03 batch 1 end: ~2026-05-24 mattina ET.

Pending:
- r03 batch 2 (a1, a2, e3b r03), to launch right after batch 1
  completes. Each 36h → expected end ~2026-05-26.
- Sanity run (e2 on gpu0, 6h), schedulable after all production runs
  complete.

Notable n=3 findings already locked (n>=2):
- e3 drop rate at saturated load: 15.7% (r01) and 15.6% (r02), with
  r03 in-progress at 13.0% (still has ~15h to potentially rise).
  The PyTorch naive baseline at 0.174 rps hits a capacity ceiling;
  e3b at 0.050 rps drops <2% on the same engine and same GPU.

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
- 2026-05-19 evening ET: **Paper framing decision.** The camera-ready
  is a standalone paper on n=3 data, on the same narrative skeleton as
  the preprint but with all numbers, figures, and classifications
  computed on n=3. The preprint is background, not reference.
- 2026-05-19 evening ET: **Analysis pipeline hardened (5 fixes).**
  `aging_trends.py` now uses `aging_io.resolve_warmup` for per-run
  warmup discard (3600s campaign / 1800s pilot, auto-resolved from
  cell yaml) and emits machine-readable `--csv` output.
  `validation_check.py` and `aging_trends.py` now parse
  `process_alive` via `aging_io.truthy_series` (was `astype(bool)`,
  false-PASS for "False" string rows). `validation_check.py`
  Theil-Sen slope now computed on real `ts_unix` axis instead of
  sample indices (was ~5x inflated). New `fdr_aggregate.py`
  applies BH-FDR at q=0.10 across the joint family of trends.
  Decision rule for significance: MK Hamed-Rao p<0.01 AND Theil-Sen CI
  excludes zero AND bh_reject=True.
- 2026-05-20 (this update): r02 batch 1 (e1, e2, e3) completed.
  Batch 2 (a1, a2, e3b r02) currently running. e3 drop rate confirmed
  on n=2 (r01 15.7%, r02 15.6%). Document cleaned to fully reflect
  standalone-n=3 framing: the n=3-vs-preprint sanity check, the
  "what rules out which hypothesis" diagnostic, and the planned
  a1-isolated topology test are no longer part of the active plan
  and have been moved to the Archive section at the bottom for
  traceability.
- 2026-05-20 afternoon ET: **First scan from validation_check showed
  apparent r02/r01 ratios of 0.06 (e1, 17x smaller), 1.48 (e2),
  0.53 (e3).** Interpreted at first as large between-replica
  variance; see follow-up below for the CI-aware reading.
- 2026-05-20 evening ET: **Paper-grade pipeline (aging_trends +
  fdr_aggregate) recomputed on r01 and r02 of e1/e2/e3.** Findings:
  (1) The "17x" on e1 was a validation_check artifact. Paper-grade
  CIs for e1_r01 [1.5, 22.3] KB/h and e1_r02 [0.74, 23.8] KB/h
  overlap completely — the two replicas are reproducible within CI.
  (2) e2 shows a real between-run effect: r01 [12.9, 31.6] KB/h
  and r02 [40.3, 231.4] KB/h are disjoint by 8.7 KB/h. r02 slope is
  ~7x higher than r01.
  (3) e3 r01 [4.9, 20.1] KB/h vs r02 [21.4, 88.0] KB/h: borderline
  disjoint (gap 1.3 KB/h). r02 ~3x higher than r01.
  (4) **e3_r02 also shows paper-grade significant proc.vms_bytes
  growth at +52.2 MB/h** (CI [20.9 KB, 77.3 MB]/h, MK p=7.7e-08).
  In r01 of e3, VMS was NOT significant (slope=0, p=0.097). This
  is a "VMS-only growth" pattern, distinct from mmap-lock-step
  (RSS+VMS together) and from sbrk-RSS-only (RSS without VMS), and
  is not captured by the current three-class stepness panel.
- 2026-05-20 evening ET: **Pipeline calibration note.**
  validation_check and aging_trends produce the SAME slope on r02
  (smooth) but DIFFER by ~5x on r01 (step-heavy). Reason: raw 5s
  samples vs 60s-window downsampling. The two pipelines are not
  equivalent on step-event-dominated series. **Decision: only
  aging_trends + fdr_aggregate is paper-grade.** validation_check
  is reduced to a per-run sanity gate (PASS/FAIL on trend direction),
  not a slope source for the camera-ready.
- 2026-05-20 evening ET: **Stepness panel on r02 confirms two of the
  four expected classes and reveals the fourth.**
  - E2 r02: corr=0.83, K_trim=648.6 [513.5, 846.1] → mmap-style
    step-wise confirmed on n=3, matches preprint expectation.
  - E1 r02: corr=0.31, K_trim=NaN (script edge case to fix),
    top1%_step=0.0001 MB → continuous drift on RSS confirmed, matches
    preprint quasi-linear expectation.
  - E3 r02: corr=0.24, K_trim_dRSS=1.1 → "continuous drift on RSS"
    by the three-class rule, but the cell ALSO shows paper-grade
    significant VMS growth at +52 MB/h. Fourth class needed:
    VAS-only growth.
- 2026-05-20 evening ET: **stepness.py patched (committed) — first
  pass, four-class taxonomy.** Fix K_trim=NaN edge case via
  operational fallback (when top1%_step < 1 MB AND steps>1MB/h < 0.1,
  set K_trim=0.0 with stderr warning). Add `K_trim_dVMS` metric on
  proc.vms_bytes deltas, mirroring K_trim_dRSS. New `class` column in
  CSV output. Four classes: mmap-style / sbrk-style / VAS-only /
  continuous drift / (border).
- 2026-05-21 afternoon ET: **Full 12-run stepness panel after Fix P1
  (commit `cb7ad4a`). Major paper-side reframing.** Five-class
  taxonomy + priority short-circuits applied to all 9 completed + 3
  in-progress runs of the n=3 campaign. Result: **only E2 is
  step-wise; all other 5 cells (E1, E3, A1, A2, E3b) are continuous
  drift.** The Fix P1 (top1%_step zero-heavy bug + VMS-missing
  handling + PID-aware diff segmentation per audit follow-up) made
  the top 1% magnitude correct for the paper table — E2 pilot 0.005
  → 2.57 MB (525x), E2 campaign 0.001 → 1.8 MB. All class
  assignments unchanged before/after Fix P1 (the fix is descriptive,
  not classifying). The headline paper claim becomes: "under n=3
  parallel multi-tenant deployment, mmap-style step-wise allocation
  emerges exclusively in Triton + vLLM V0 (E2), with canonical
  magnitude ~1.7-1.8 MB top 1% step and 0.09-0.23 MB-scale events
  per hour, reproducible across n=2 replicas. All other deployments
  exhibit continuous drift with no MB-scale step events." Pilot
  retrospective shows "uncorrelated step-wise" (A1 pilot) and
  "VAS-only step-wise" (E1 pilot) instantiated in vitro, but no
  cell in the n=3 setting falls in those classes — declared in the
  paper as supporting evidence that the classes are observable in
  principle while drift is the dominant non-mmap pattern under
  realistic deployment.
- 2026-05-21 morning ET: **Two incremental fixes to stepness.py
  (committed up to `1c84e9e`).**
  - Fix n.1: low-step fallback made unconditional on operational
    metrics (`steps/h_1mb < 0.01`), not subordinated to K_trim=NaN.
    Discovered after running the four-class patch on e1_r01 of the
    campaign and getting K_trim_dRSS=928 from kurtosis on micro-noise
    (top1%_step=100 byte, no real step events). The original fallback
    required NaN to trigger, missing this case.
  - Fix n.2: `classify_stepness` short-circuits to "continuous drift"
    when both axes are in low-step fallback, before evaluating
    corr-based rules. Discovered after fix n.1: e1_r01 had both
    K_trim=0 from fallback but corr=0.64 was in the grey zone 0.5-0.8,
    so the cell fell into "border". Mechanism justification: corr in
    that zone on a no-step run is correlation of sampling micro-noise,
    not of allocation mechanism. The branch "border" is for grey-zone
    runs with real step events, not for runs with no events at all.
  - Pilot sanity check after fix n.2: A2 pilot moved from border to
    continuous drift (was border for corr=0.58 with K_raw spuriously
    high; now correctly drift since both axes scatter the fallback).
    All five other pilot cells unchanged. The reclassification of A2
    pilot is consistent with the mechanism (no step events anywhere
    → drift), accepted as the correct behavior.
- 2026-05-21 morning ET: r02 batch 2 (a1, a2, e3b r02) launched on
  gpu0/1/2 right after r02 batch 1 completion. Each scheduled for 36h.
- 2026-05-22 ET: r02 batch 2 (a1, a2, e3b r02) completed. All 12
  r01+r02 runs now in `~/wosar/runs/`. Health check 0 FAIL.
- 2026-05-22 ET: r03 batch 1 (e1, e2, e3 r03) launched on gpu0/1/2.
  Each scheduled for 36h, ETA fine batch ~2026-05-24 mattina ET.
- 2026-05-23 ET: **Paper-grade analysis on the full n=2 (12 runs).**
  Ran aging_trends.py + fdr_aggregate.py over all r01+r02 of the 6
  cells. Headlines:
  - **USS = 12/12 paper-grade significant**. RSS = 11/12 (a1_r02
    fails on CI floor=0).
  - 4/6 cells CI-compatible across replicas on USS (a1, a2, e1,
    e3b). 2/6 CI-disjoint (e2 with 4.2 KB/h gap, e3 with 1.9 KB/h
    gap). Both disjoint cells in slot batch 1, both r02 > r01.
  - VMS = 2/12 paper-grade significant: e3_r02 at +52 MB/h,
    e3b_r02 at +127 MB/h. r02-only phenomenon on both PyTorch+HF
    cells.
- 2026-05-23 ET: **USS adopted as canonical leak indicator for the
  camera-ready.** Rationale: (a) 12/12 significant vs 11/12 RSS,
  (b) robust to AR(1) edge cases (a1_r02 RSS hits CI floor=0 from
  rho_RSS=0.99 inflation; USS at rho=0.005 has tight [12.9, 13.4]
  KB/h CI on the same data), (c) semantically cleaner (private
  resident pages; RSS includes shared mappings that are a
  confound), (d) point estimates agree with RSS within <10% so
  no narrative claim changes. RSS is reported alongside as
  secondary indicator. Preprint used RSS in Table IV but that
  table is in the archive section already (standalone-n=3
  framing).
- 2026-05-23 ET: **VMS-only growth on PyTorch+HF mechanistically
  verified.** Direct inspection of e3b_r02 proc CSV: VMS 38.52 →
  46.04 GB (+7.52 GB) over 36h, RSS +20 MB only, num_fds 53→52,
  num_threads 200→199. Rules out file-mapping accumulation and
  thread-stack growth. Mechanism: anonymous mmap reservation by
  PyTorch CUDA caching allocator host-side metadata. K_trim_dVMS
  = 2.2 → smooth drift, not step-wise. Both e3 (+52 MB/h) and
  e3b (+127 MB/h) show the pattern on r02; r01 absent on both.
  Plausible cause: r02 seed=2 produces a different request
  pattern that engages the allocator path more heavily, or host
  state accumulation through the campaign. r03 will discriminate.
  e3b > e3 on VMS slope despite e3b at 3.5x lower load is
  counter-intuitive on per-request leak terms; consistent with
  time-driven rather than load-driven allocator dynamics.
- 2026-05-23 ET: **New hypothesis: Triton wrapper amplifies
  between-run variance independent of underlying engine.**
  Observation on n=2 USS ratios r02/r01:
  - Standalone (e1, a1): 0.26 and 1.02 (tight)
  - Triton wrapper (e2, a2): 8.2 and 0.14 (highly variable)
  - PyTorch+HF naive (e3, e3b): 2.8 and 2.7 (intermediate)
  Standalone cells show the smallest between-run variance,
  Triton-wrapped cells show the largest. Hypothesis: Triton
  scheduler internal state (dynamic batching queues, model
  instance allocation) is non-deterministic across replicas and
  propagates into the engine process footprint. To be checked
  on n=3 with r03.
- ~2026-05-25 ET: **Campaign completed.** All 18 production +
  1 sanity. r03 batch 1 (e1, e2, e3) finished ~24 May, r03 batch
  2 (a1, a2, e3b) finished ~26 May, sanity e2_r99 (6h e2 on gpu0)
  scheduled and completed after.
- 2026-06-06 ET: **Local sync + cleanup.** All 19 run directories
  rsynced (via tar.gz + scp due to rsync incremental stalling on
  one of the runs) to `~/Documents/Github/llm-serving-bench/runs/`
  on the Mac (2.2 GB, gitignored). Pipeline `aging_trends.py`
  validated bit-exact in local against server numbers. Cleanup
  on server: cancellati `runs_aborted_20260516_052308` (54 MB),
  `runs_failed_attempts` (97 MB), tar di trasferimento (611 MB).
- 2026-06-06 ET: **Paper writing kickoff.** Preprint riletto
  integralmente (`docs/WOSAR_2026.pdf`, 8 pagine). Creato
  `paper/PAPER_UPDATE_PLAN.md` con piano di rescrittura
  sezione-per-sezione per la nuova chat di Opus 4.8.
  **Open question per Domenico**: dove sono le LaTeX sources
  originali del preprint? La cartella `paper/` nel repo è vuota
  (solo `.gitkeep`), il preprint è solo in PDF. Per applicare
  i cambiamenti serve recuperare le sources (Overleaf? altra
  repo? file locale?) o ricostruire il LaTeX da zero dal PDF.
- 2026-06-07 ET: **Paper-grade n=3 pipeline RUN in locale (TODO #1/#2
  RESOLVED).** Output committati in `paper/n3_analysis/`: `per_cell.csv`,
  `per_cell.txt`, `fdr_per_run.csv`, `stepness_all.csv`, e il riepilogo
  `N3_RESULTS.md`. Pipeline = aging_trends (per-run) -> fdr_aggregate
  (per-run BH-FDR) -> aggregate_slopes (per-cell DL-RE + median, q=0.10,
  expected-replicas=3) -> stepness (5-class). Tutti i 18 production run
  danno il catalogo completo a 34 indicatori = 612 righe totali.
- 2026-06-07 ET: **BUG in `aging_trends.py` trovato e diagnosticato
  (causa dell'anomalia 405-vs-520).** Due difetti nella discovery locale
  dei monitor, che NON usa `aging_io.discover_proc_prefix`:
  (1) prefisso GPU hardcoded a `gpu0` (riga ~285) -> indicatori GPU persi
      su tutte le celle gpu1/gpu2 (e2, a2, e3, e3b).
  (2) il loop del `proc_prefix` (righe ~293-297) esclude solo
      `("gpu0","system")` e itera un glob NON ordinato -> sulle celle
      gpu1/gpu2 puo' scegliere `gpu1`/`gpu2` come monitor di processo,
      calcolando proc.* (RSS/USS/VMS, cioe' il leak indicator canonico)
      sul file sbagliato. Esito dipendente dall'ordine del filesystem,
      quindi NON-DETERMINISTICO tra macchine (nel sandbox a2 si rompeva
      ed e2 no). Spiega sia le 405 righe sia perche' alcune celle uscivano
      e altre no sul server.
  Workaround usato per produrre i numeri n=3: copia patchata
  `paper/n3_analysis/aging_trends_FIXED_reference.py` (gpu-prefix discovery
  + esclusione `gpuN`/`system` via glob ordinato). Riproduce ESATTAMENTE i
  valori gia' documentati (a2_r01 USS 42.5, e2_r01 19.1, e1_r01 7.0 KB/h),
  quindi i numeri n=3 sono affidabili.
  **FIX UFFICIALE DA APPLICARE IN REPO (via Claude Code):** sostituire la
  discovery locale in `aging_trends.py` con `aging_io.discover_proc_prefix`
  (gia' corretta: legge il manifest, esclude gpu*/system) + una discovery
  analoga del prefisso GPU. Chiude il long-tail TODO #9.
- 2026-06-07 ET: **Numeri n=3 chiave (USS DL-RE, KB/h).** E1 4.73 (CI
  [-1.66, 11.11], NON significativo), A1 13.16 (I^2=0, la piu' riproducibile),
  E2 55.75 (I^2=89%, repliche in forte disaccordo), A2 22.74 (I^2=60%),
  E3 19.95 (I^2=54%), E3b 61.65 (I^2=0). 5/6 celle RE-significative su USS
  (E1 l'eccezione). Stepness: E2 unica con step events (r01/r02 mmap-style,
  r03 border; sanity e2_r99 border), tutte le altre 5 continuous drift 3/3.
  VMS VAS-only solo su r02 di e3 (+52 MB/h) e e3b (+127 MB/h), non per-cella.
  Drop rate e3 15.7/15.6/14.3%, e3b 0/0/0%.
- 2026-06-07 ET: **Ribaltamenti rispetto al draft n=1** (vedi
  `paper/PAPER_UPDATE_PLAN.md` per le conseguenze editoriali). Il gap
  draft->n=3 e' STRUTTURALE (pipeline + version drift), non varianza tra
  run: anche la replica piu' alta resta 13-1300x sotto i numeri del draft
  sugli engine ottimizzati (E1 ~1300x). Due claim del draft cadono:
  "naive baseline e' il piu' pulito" (falsa: E1 e' il piu' pulito) e
  "leak rates span ~3 ordini di grandezza" (su n=3 e' ~1 ordine, tutto in
  KB/h, decine di MB/mese). Il draft non era mai stato sottomesso; le sue
  cifre eclatanti erano artefatti, come sospettato.
- 2026-06-07 ET: **DECISIONE EDITORIALE: due paper (vedi
  PAPER_UPDATE_PLAN.md).** Workshop WoSAR 2026 = single-run (replica r02),
  36h, framing multi-tenant, stesso scheletro del draft con risultati
  sostituiti e reframing del testo. Journal = estensione n=3 +
  meta-analisi (DL-RE + Stouffer + per-cell FDR) + riproducibilita'/
  eterogeneita'. Intro del journal gia' redatta e CONGELATA in questa
  sessione (nuovi bib key: dersimonian1986, stouffer1949, opz. whitlock2005).
- 2026-06-07 ET (sessione 2): **PRIMA BOZZA WORKSHOP COMPLETA.** Tutte le
  sezioni redatte su r02 (Abstract, I, II, III, IV.A-E, V, VI; Table I-V;
  Figura 2 = `paper/n3_analysis/figures/uss_factorial.pdf`, pannello
  singolo USS). Dettagli completi di decisioni, numeri e finding in
  `paper/PAPER_UPDATE_PLAN.md` sezione "2026-06-07 (session 2)". Da
  riprendere: pass di coerenza titoli, scelta abstract, discussione
  "c'e' aging davvero?", fix pipeline in repo (TODO #9), e la **proposal
  NVIDIA** (angolo reliability->security sullo step-wise di E2: vedi seed
  in PAPER_UPDATE_PLAN). Estensione journal: n=3 + meta-analisi.
- 2026-06-10 ET: **Paper finalizzato e su arXiv (arXiv:2606.11916, cs.SE/cs.AI, 10 giu).** Titolo finale
  "Characterizing Software Aging in GPU-Based LLM Serving Systems" (7 pagine
  IEEE, dentro il limite WoSAR). Hardware UNCC = L40S (4 GPU). WoSAR 2026:
  submission 20 luglio, notifica 10 agosto, camera-ready 17 agosto;
  review NON double-blind -> arXiv col proprio nome consentito. Licenza
  arXiv: non-exclusive (compatibile col copyright transfer IEEE). Fix
  minori residui sul paper: citazione rotta `[?]` nell'intro, grammatica
  "we give the evidence", coerenza maiuscole titoli di sezione.
- 2026-06-12 ET: **PROSSIMA CAMPAGNA RIDISEGNATA come DoW di workload**
  (supera il "DoE 2x2" del 2026-06-10). Esperimenti PRIMA in locale su L40S
  (de-risking); ~2 mesi di server disponibili. Pipeline: **fix di
  `aging_trends.py` (TODO #9) APPLICATO** (ora usa
  `aging_io.discover_proc_prefix` + discovery dinamica del prefisso GPU).
  Disegno: un unico **screening DoW** girato identico su **3 sistemi**
  (Dynamo disaggregato 2 GPU, Triton+vLLM, vLLM standalone), 5 fattori
  (rate, prompt-len, output-len, prefix-repeat, burstiness) in **Res V
  16-run + 3 center point, finestra 48h**, rate = frazione-del-ceiling con
  calibrazione per run; risposte: slope/ora + slope/richiesta (time vs
  load) + stepness + per-componente (router/prefill/decode/KV-transfer).
  Modello **Qwen** in locale; **Nemotron + asse hardware A100 = arm grant**.
  **Piano completo + razionale + budget in `paper/PAPER_UPDATE_PLAN.md`**,
  sezione "2026-06-12: EXPERIMENTAL PLAN — DoW".
- 2026-06-29 ET: **Livelli fattori e STEP 1 CONFERMATI.** Livelli mild/stressful
  (Qwen, ctx 8192): rate 30%/85% ceiling; prompt ~512/~6000 tok; output
  ~64/~1024 tok; prefix-repeat 0%/80%; burstiness Poisson/bursty; center point
  = valori mediani. **PROSSIMO PASSO = STEP 1**: bring-up Dynamo (aggregato +
  disaggregato) + decisione di monitoring (mappare i PID dei componenti:
  router/prefill/decode/KV-transfer) + due feature client (prefix-repeat
  injection, burst arrival) + validazione harness (2 run brevi) + tooling di
  calibrazione rate per-run. **Nessun run DoW parte prima di STEP 1.** STEP 0
  (Domenico, in parallelo): run di validazione 48h di una cella nota su L40S.
- 2026-06-29 ET: **STEP 1 IN CORSO** via Claude Code (VS Code). Prompt lanciato
  con i 5 workstream: (1) bring-up Dynamo aggregato+disaggregato + script/README
  + pin versione vLLM; (2) monitoring per-componente (PID->router/prefill/decode/
  KV-transfer + CSV per-componente + aggregato, backward-compatible coi sistemi
  single-process); (3) client prefix-repeat injection (0%/80%); (4) client burst
  arrival mode (poisson/bursty); (5) tooling calibrazione rate = frazione del
  ceiling. Chiusura STEP 1 = 2 run brevi (~20-30 min) su vLLM standalone +
  Dynamo disaggregato per validare harness+monitor+manifest+pipeline end-to-end.
- 2026-06-29 ET: **STEP 1 — decisioni di design LOCKED** (dopo review del piano
  Claude Code). Vincolo globale: **parallelismo minimo**, run SERIALI (un run per
  volta, niente run concorrenti che condividono GPU/host); campagna sequenziale e
  restartable, non parallela. Monitoring per-componente: un componente = *gruppo*
  di PID (somma entro il gruppo; PID singoli nel manifest); **fidarsi
  dell'aggregato solo su USS** (privata, no double-count, confrontabile coi
  single-process), RSS/PSS aggregati solo diagnostici; **topologia worker fissa,
  autoscaling Dynamo disattivato** (membership costante sulle 48h); KV-transfer
  (NIXL) come componente solo se ha PID proprio; etcd/NATS come componenti "infra"
  ma FUORI dall'aggregato engine; campionamento stesso-tick. Ceiling (WS5):
  definizione conservativa (achieved/offered ≥ 0.98 + backlog piatto + p99 sotto
  bound, non il bordo 0.95) per dare margine all'85% sulle 48h; **ceiling
  PER-CELLA** (dipende da prompt/output/prefix/burst): calibrare ognuna delle 19
  celle variando solo il rate, registrare ceiling/fraction/rate_calibrated nel
  manifest; calibrare a t0 e FISSARE il rate (l'erosione di capacità sulle 48h è
  un risultato da osservare, non assorbire); ricontrollare il budget ~150 GPU-h
  vs 19 celle × 3 sistemi. Versione vLLM = quella supportata da Dynamo, pinnata
  IDENTICA sui 3 sistemi via image digest. Deploy Dynamo locale via CLI (no k8s
  su singola box). `request_distribution` assorbito in `arrival_mode`; aggiunta
  colonna `shared_prefix_applied`. `attach_run.py` solo per i 2 run di
  validazione: la campagna 48h × 19 × 3 deve essere guidata da
  launch_cell/campaign.py (bring-up/readiness/teardown Dynamo, run non
  presidiati e serializzati). DA REGISTRARE a chiusura STEP 1: topologia worker
  scelta + versioni pinnate.
- 2026-06-29 ET: **STEP 1 — budget calibrazione ricontrollato (WS5).** Ottimizzazione
  chiave: i 5 fattori della DoW sono TUTTI di workload; il sistema di serving e la
  sua config sono FISSI per sistema. Quindi una sola istanza engine per sistema si
  riusa per tutte le 19 calibrazioni (load una volta, poi 19 sweep di solo-rate
  back-to-back, serial): 19 cold-load -> 1 per sistema. Stima: sweep per cella
  ~6-8 punti rate x ~4min + cooldown ~30min wallclock; per sistema ~1 cold-load +
  19 sweep ~10h. GPU-h: standalone/Triton (1 GPU) ~10 ciascuno, Dynamo (2 GPU) ~20
  -> **~40 GPU-h totali**, dentro i ~150 con margine (anche raddoppiando le finestre
  ~80). Coerente col vincolo seriale (le calibrazioni stesse girano una per volta).
  `calibrate_rate.py` supporterà sia "load-once + sweep N workload combos" sia il
  caso singola-cella; ceiling conservativo (achieved/offered >= 0.98 + backlog
  piatto + p99 sotto bound) scritto nel manifest con fraction e rate_calibrated.
- 2026-06-29 ET: **STEP 1 — vLLM PIN CONFERMATO e LOCKED a 0.16.0** (intersezione
  NON vuota dei tre stack). Immagini:
  - Dynamo: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13` (vLLM 0.16.0,
    NIXL 0.10.1, CUDA 13.0). Dynamo 1.0.1 stable (GA 2026-03-16).
  - Triton: `nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3` (vLLM 0.16.0).
    Si sposta dal 25.09 del preprint allo stack version-aligned (Triton 25.12 era
    ancora 0.11.1; arriva a 0.16.0 col 26.03). Da dichiarare nel paper come scelta
    deliberata verso lo stack corrente e allineato.
  - Standalone: `vllm/vllm-openai:v0.16.0-cu130` (Docker Hub upstream; esiste anche
    il tag bare `v0.16.0`, si usa `-cu130` per allineare CUDA 13). NON l'NGC
    `nvcr.io/nvidia/vllm:26.03` (quello e' 0.17.1).
  Fatto Triton confermato da DUE fonti: `server/build.py` @ branch r26.03
  version_map `"vllm_version": "0.16.0"`, e la riga container-contents delle note
  26.03 "vLLM version 0.16.0". Verifica finale al pull sulla box:
  `docker run --rm <img> pip show vllm`. Digest registrati al pull in
  `engines/*/image_pin*.json` (3 file creati; campo digest vuoto fino al pull;
  comandi di cattura nel file).
  **TTV (testo locked):** il SUBSTRATO per-immagine differisce (CUDA caching
  allocator, torch, glibc) - esattamente cio' che la metrica di stepness sonda.
  Questa differenza e' intrinseca a qualsiasi confronto cross-system (non si puo'
  eseguire "Triton" senza l'immagine di Triton); e' mitigata tenendo IDENTICO a
  0.16.0 il layer di orchestrazione vLLM, che e' strettamente meglio del confound
  da version-drift del preprint. NON dire "differisce solo il compute path".
  **Process tree Dynamo disaggregato (per WS2; regex da confermare sulla box):**
  frontend `python -m dynamo.frontend` (KV-router integrato, non separato);
  prefill `python -m dynamo.vllm --is-prefill-worker`; decode `python -m
  dynamo.vllm` (senza flag); NIXL in-worker (no PID proprio); infra `etcd` +
  `nats-server` (separati, FUORI dall'aggregato USS engine). Topologia worker
  fissa, autoscaling OFF.
- 2026-06-29 ET: **STEP 1 CODE-COMPLETE (9 commit), in attesa del GATE su box.**
  Implementati e testati in locale: WS3 prefix-repeat, WS4 burst arrival
  (arrival_mode assorbe request_distribution), WS5 calibrazione per-cella
  (`scripts/calibrate_rate.py` + `--calibration-file` in launch_cell), WS1 pin
  (3 image_pin*.json), WS2 monitor per-componente (`monitoring/multiproc_monitor.py`
  + multi-GPU in run_monitors + `proc_prefix` nel manifest + fix dict-manifest in
  aging_io), bring-up Dynamo (`deploy/dynamo/`), `scripts/attach_run.py`, celle di
  validazione (`campaigns/extension/cells/val_*.yaml`), gate checker
  (`analysis/validate_extension_run.py`). Gate checker validato su run sintetica
  Dynamo-shaped: PASS, incluso `aging_trends` che legge `agg_dynamo` come serie
  proc. NB locale: `.venv` ha httpx/pyyaml/psutil aggiunti per i test (gitignored).
  **GATE su cci-csgpu11 (precede il refactor seriale+lifecycle), con due
  affinamenti:**
  1. **Step 0 = verifica finale del pin.** `docker run --rm <img> pip show vllm`
     sui TRE: Dynamo, Triton, standalone. Devono dare TUTTI 0.16.0. **Triton e' il
     check critico** (da remoto confermato solo via build.py r26.03 + note 26.03;
     non con pip show). **Se Triton non da' 0.16.0 esatto: STOP**, l'assunzione di
     intersezione salta, riparliamone prima di proseguire. Registrare i 3 digest
     nei pin file.
  2. **Ordine invertito: prima vLLM standalone, poi Dynamo disaggregato.** Lo
     standalone e' il path single-process noto: se attach_run/pipeline hanno un bug
     di base lo si trova sul caso facile, non in mezzo alla complessita' multi-proc.
  3. Congelare i regex dei componenti in `val_dynamo_disagg.yaml` contro `ps`
     reale al primo bring-up (chiude "WS2 contro il process tree reale").
  Gate PASS = entrambi i run PASS + regex congelati + 3 pip show a 0.16.0. SOLO
  allora: refactor seriale (no fan-out per-slot in campaign.py) + lifecycle Dynamo
  (bring-up/readiness/teardown in launch_cell). Niente bozza lifecycle prima del
  gate: e' il pezzo che il gate deve validare su hardware reale.
  Backlog minore: `aging_trends.downsample_client` assume la colonna `ttft_s`
  (sempre presente nei CSV reali; guardia da una riga, non blocca).
- 2026-06-29 ET (sul server): **GATE STEP 0 ESEGUITO -> PIN RI-DERIVATO a 0.20.1.**
  Il `pip show vllm` reale ha smentito la ricerca remota su Triton: **Triton
  26.03 = vLLM 0.17.1, NON 0.16.0** (`build.py` r26.03 diceva 0.16.0 -> sbagliato;
  le note del *container vLLM NVIDIA* invece erano giuste, 26.03=0.17.1). Mappa
  reale (note container vLLM, affidabili): Triton/vLLM 26.01=0.11.1, 26.02=0.15.1,
  26.03=0.17.1, **26.04=0.19.0, 26.05=0.20.1**. Dynamo stable: 1.0.1=0.16.0,
  1.1.1=0.19.0, **1.2.0=0.20.1**. **A 0.16.0 l'intersezione e' VUOTA** (Triton
  salta 0.15.1->0.17.1). Intersezioni pulite a tre vie native: 0.19.0 (Dynamo
  1.1.1 + Triton 26.04) oppure 0.20.1 (Dynamo 1.2.0 + Triton 26.05).
  **DECISIONE (Domenico): pin = 0.20.1**:
  - Dynamo `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13`
  - Triton `nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3`
  - standalone `vllm/vllm-openai:v0.20.1-cu130`
  Aggiornati: 3 pin file, `deploy/dynamo/env.sh`, `val_vllm.yaml`,
  `val_dynamo_disagg.yaml`, README. **Validazione DOPO il re-pin** (sullo stack
  finale). Sul box: ground-truth i tre con `pip show vllm` PRIMA del gate
  (vllm-openai richiede `--entrypoint pip`); STOP se uno non e' 0.20.1.
- 2026-06-29 ET (sul server): **FINDING DISCO (blocker pre-campagna).**
  `docker info` -> DockerRootDir = **/var/lib/docker** su partizione **126G**
  (era al 98%, da cui il "no space left" malgrado /home abbia 6.3T liberi). Il
  move del data-root su /home di ADR-002 e' stato PERSO (era gia' nel
  technical-debt). Pulizia: rimosse le immagini vecchie del preprint (~135GB,
  campagna n=3 chiusa, digest gia' registrati) -> 70G liberi. **Prima della
  campagna 3-sistemi: rimettere il data-root su /home** (3 immagini ~75GB + dati
  + headroom non stanno comodi in 126G). I dati grezzi `runs/` e
  `runs_n1_baseline/` NON sono stati toccati (gia' sul Mac; non servono per lo
  spazio del data-root, che e' su /var/lib).
- 2026-05-20 evening ET: **Five-class taxonomy adopted (committed).**
  Pilot n=1 sanity check after the four-class patch surfaced two
  retrospective reclassifications:
  - **E1 pilot**: corr=0.32, K_trim_dRSS=0.0, K_trim_dVMS=789 →
    VAS-only step-wise, NOT continuous drift as the preprint assumed.
    The preprint missed this because it had no dVMS axis. The n=3
    paper recovers the finer mechanism class.
  - **A1 pilot**: corr=0.20, K_trim_dRSS=402, K_trim_dVMS=582 → does
    not fit any of the four classes (both axes step-wise but
    uncorrelated). Fell into border under the four-class rule.
  Decision: add a fifth class **"uncorrelated step-wise"**
  (corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10). Mechanism
  interpretation: heap-arena extension (RSS-side) and mmap-style
  block allocation (VMS-side) operating in parallel as two
  independent allocator events. The class is mechanism-justifiable
  a priori (glibc/CUDA-caching-allocator design does not constrain
  these to be synchronous), with A1 pilot as the empirical
  confirmation. Five-class taxonomy is now canonical for the paper.
- 2026-05-20 evening ET: **Stepness classifications of the pilot
  n=1 are retrospective only.** They are useful as a sanity check
  on the taxonomy and to demonstrate that the dVMS-axis refinement
  recovers signatures the preprint missed. The headline class
  assignment per cell in the camera-ready paper is from n=3 (r01,
  r02, r03 with majority rule), not from the pilot. A cell can
  legitimately fall into different classes in pilot vs n=3 because
  the host environment is not stationary across the 2-week window
  (system.mem_used host-side drift dropped 3x between r01 and r02
  of n=3, see Status snapshot). Pilot classes do not constrain n=3
  classes.
- 2026-05-25: **Per-cell aggregator `analysis/aggregate_slopes.py`
  shipped.** Implements TODO #5 (full n=3 analysis once all r03
  complete). Two estimators per (cell_id, indicator), computed
  entirely off the per-replica trend CSVs from `aging_trends.py`:
  - **Primary**: DerSimonian-Laird random-effects on the k Theil-Sen
    slopes, with per-replica SE derived from the upstream 95% CI via
    Gaussian-equivalent `(hi - lo) / (2 * 1.96)`. Reports `slope_RE`,
    `ci_RE`, `tau2`, `I^2`, `Q`, `Q_pvalue`, `k_used_RE`.
  - **Robustness**: sample median of the k per-replica slopes, with
    `[min, max]` reported as the exact non-parametric CI for the
    population median. Coverage by k from order statistics:
    `1 - 2 * (1/2)^k` → 75% at k=3, 87.5% at k=4, 93.75% at k=5.
    Note: a percentile bootstrap labelled "95%" on k=3 collapses to
    the same `[min, max]` interval and would mis-state coverage; we
    report the exact 75% directly.
  Per-cell BH-FDR at q=alpha is applied across the (cell_id,
  indicator) family. The per-cell MK p-value is the Stouffer z-score
  combination of the k per-replica MK z values (numerically stable
  via `scipy.stats.norm.sf`). Decision rule for `RE_significant`:
  (BH rejects cell-level Stouffer p) AND (RE 95% CI excludes 0) AND
  (n_replicas, k_used_RE, k_used_stouffer meet `--expected-replicas`,
  default 3). The pooled-median CI-excludes-zero flag is reported but
  does NOT gate any significance decision (75% CI too conservative).
- 2026-05-25: **Earlier pooled Theil-Sen on concatenated series was
  systematically biased toward zero.** First implementation pooled
  across replicas by per-replica time-reset + median-centering, then
  computed Theil-Sen on the concatenation. Empirically `slope_pooled`
  came in 7-30x below `slope_RE` on e2/e3 USS and exactly 0 on e1
  RSS. Root cause: Theil-Sen computes pairwise slopes across ALL
  pairs; on a median-centered concatenation, cross-replica pairs at
  similar relative times have `Δy ≈ 0` (both centered around their
  per-replica median) and outnumber within-replica pairs ~2:1. The
  median of pairwise slopes is therefore dragged toward 0. A
  follow-up patch added per-replica time-offset (i × 2T) to avoid
  near-zero `Δt` but only reduced the bias magnitude; the structural
  problem stayed. Replaced with the median-of-slopes estimator
  documented above, which never inspects raw data and so cannot
  suffer the cross-pair pathology. Sanity check on e1 USS:
  RE=4.6 KB/h, median=4.84 KB/h, agreement <6%; on e2 USS:
  RE=54.4 KB/h, median=58.6 KB/h, agreement <8%.
- 2026-05-25: **Audit findings on aggregate_slopes.py addressed.**
  - `load_proc(prefix, columns=None)` only loads `ts_unix +
    process_alive`. Was being called without `columns=[col]`, so all
    `proc.*` indicators returned None in the obsolete pooled-TS
    branch. Fixed before the branch was removed.
  - `k_used_RE` and `k_used_stouffer` propagated to CSV and human
    output. Both gate `RE_significant` against `--expected-replicas`
    (default 3); rows below threshold are excluded from significance
    AND marked with `*` in the `sig_RE` column.
  - High between-replica heterogeneity (`I^2 > 75%`) flagged as `!`
    next to `sig_RE`. Surfaces cells where the RE estimate is
    averaging substantially disagreeing replicas (e2 USS is the
    canonical case).
  - `--campaign-yaml` argument was speculative and unused; removed.
    `--runs-root` and `--downsample-seconds` removed in the
    median-of-slopes refactor because the new estimator no longer
    needs raw CSV access.
  - Numerical stability: `1 - stats.norm.cdf(abs(z))` and
    `1 - stats.chi2.cdf(Q, df)` replaced with `stats.norm.sf` and
    `stats.chi2.sf` so large z and large Q do not cancel to 0.
  - `se_from_ci_midrange` on degenerate `[c, c]` or inverted
    `[hi, lo]` CIs now returns NaN with a stderr warning instead of
    `SE_FLOOR`; the replica is dropped from the DL meta-analysis
    rather than receiving an effectively infinite weight.

---

## Open TODOs

In order of priority for the next session (2026-06-06):

0. **RESOLVED 2026-06-06: LaTeX sources sono su Overleaf** (canonical
   source). Domenico scrive il paper personalmente su Overleaf, in
   modalità conversazionale con l'assistente. Workflow:
   discussione-per-sezione → scrittura su Overleaf → paste in chat
   per review/feedback → iterazione. `paper/PAPER_UPDATE_PLAN.md` è
   il reference document per sapere cosa cambia in ogni sezione,
   non da incollare meccanicamente.

1. **RESOLVED 2026-06-07: paper-grade n=3 per-cell tables prodotte**
   (output in `paper/n3_analysis/`, vedi Decisions 2026-06-07). NB: girate
   con la copia patchata di aging_trends; il fix va ancora applicato in
   repo (vedi Decisions / TODO #9). Sequenza originale qui sotto per
   riferimento. Pipeline ready, basta lanciarla in locale:
   ```bash
   cd ~/Documents/Github/llm-serving-bench
   source .venv/bin/activate
   setopt interactive_comments  # zsh per evitare bug `#` argument
   # aging_trends già fatto ma con anomalia (405 vs 520 righe).
   # rifare con logging completo:
   rm -rf /tmp/wosar_n3 && mkdir -p /tmp/wosar_n3
   for cell in a1 a2 e1 e2 e3 e3b; do
     for r in r01 r02 r03; do
       python3 analysis/aging_trends.py \
         --run-dir runs/wosar2026_${cell}_${r} --csv \
         > /tmp/wosar_n3/${cell}_${r}_trends.csv \
         2>/tmp/wosar_n3/${cell}_${r}.log
     done
   done
   wc -l /tmp/wosar_n3/*_trends.csv | tail -1
   # se ancora 405 invece di 520: grep '[warn]' /tmp/wosar_n3/*.log
   # per capire cosa skipping per quali cells
   python3 analysis/fdr_aggregate.py \
     $(ls /tmp/wosar_n3/*_trends.csv | sed 's|^|--trends-csv |') \
     --csv > /tmp/wosar_n3/fdr_per_run.csv
   python3 analysis/aggregate_slopes.py \
     $(ls /tmp/wosar_n3/*_trends.csv | sed 's|^|--trends-csv |') \
     --alpha 0.10 --expected-replicas 3 \
     > /tmp/wosar_n3/per_cell.txt
   ```
   Output `/tmp/wosar_n3/per_cell.txt` = source of truth Section IV.C.

2. **RESOLVED 2026-06-07: stepness n=3 prodotta** (`paper/n3_analysis/
   stepness_all.csv`). Esito: E2 unica step-wise (r01/r02 mmap-style,
   r03 border; sanity e2_r99 border), altre 5 celle continuous drift 3/3.
   VAS-only su e3/e3b resta r02-only. Comando di riferimento:
   ```bash
   python3 analysis/stepness.py --logs-root runs --csv \
     > /tmp/wosar_n3/stepness_all.csv
   ```
   Confermare: E2 r03 ancora mmap-style step-wise? Le altre 5
   celle r03 ancora continuous drift? VAS-only su e3/e3b si
   manifesta su r03 o resta r02-only?

3. **DONE 2026-05-20: paper-grade r01/r02 slopes for e1, e2, e3.**
   See Status snapshot for the per-cell CI table (now USS-canonical).

4. **DONE 2026-05-23: paper-grade slopes on the full n=2 (12 runs).**
   USS adopted as canonical leak indicator. RSS reported alongside.
   VMS-only growth confirmed on e3/e3b r02 with empirical mechanism
   verification.

   ```bash
   # Reusable, full n=2 over all 6 cells:
   for cell in a1 a2 e1 e2 e3 e3b; do
     for r in r01 r02; do
       python3 analysis/aging_trends.py \
         --run-dir ~/wosar/runs/wosar2026_${cell}_${r} --csv \
         > /tmp/${cell}_${r}_trends.csv
     done
   done
   python3 analysis/fdr_aggregate.py \
     $(ls /tmp/*_r0[12]_trends.csv | sed 's|^|--trends-csv |') \
     --csv > /tmp/fdr_n2_all.csv

   # Inspect per indicator:
   awk -F, 'NR==1 || $3=="proc.uss_bytes" {print}' /tmp/fdr_n2_all.csv \
     | column -t -s,
   ```

3. **Wait for r03 batch 1 end (~2026-05-24 mattina ET).** Then run
   the same pipeline on `e1_r03`, `e2_r03`, `e3_r03` and extend the
   FDR aggregate to 15 runs.

4. **Launch r03 batch 2 (a1, a2, e3b r03).** Right after batch 1 ends.
   Each 36h. Expected end ~2026-05-26.

5. **Full n=3 analysis once r03 batch 2 completes.** Pipeline now
   wired end-to-end:
   ```bash
   # (a) per-run trends on all 18 runs
   for cell in a1 a2 e1 e2 e3 e3b; do
     for r in r01 r02 r03; do
       python3 analysis/aging_trends.py \
         --run-dir ~/wosar/runs/wosar2026_${cell}_${r} --csv \
         > /tmp/wosar2026_${cell}_${r}_trends.csv
     done
   done

   # (b) per-cell aggregation: RE-DL primary + median-of-3 robustness
   python3 analysis/aggregate_slopes.py \
     $(ls /tmp/wosar2026_*_r0[123]_trends.csv | sed 's|^|--trends-csv |') \
     --alpha 0.10 --expected-replicas 3 --csv \
     > /tmp/per_cell_n3.csv
   ```
   Decision rule for paper: per-cell `RE_significant` (BH on Stouffer
   p, RE CI excludes 0, replicas not degraded). See "Pipeline
   analytical details" for the formal definition. Median-of-3 with
   `[min, max]` reported as 75% non-parametric CI alongside for
   robustness cross-check. **Validated on batch 1 r03 (e1, e2, e3 ×
   r01/r02/r03) on 2026-05-25**: 9/9 runs USS paper-grade significant
   per-run, 2/3 cells RE-significant per-cell (e2 and e3; e1 fails RE
   CI excludes 0 because per-replica CIs are wide, individual MK
   strongly rejects), pooled-median in agreement with RE-DL within
   <10% on e1 and e2 (no outlier dragging the RE estimate).

6. **Paper writing on n=3 data.** The four content elements:
   (a) Campaign description, hardware, workload, stress regime
       (Section IV.A) on n=3.
   (b) Client-side stationarity check (Section IV.B) on n=3.
       Latency, TTFT, throughput, drop rate over the 36h window.
   (c) Process-side memory aging (Section IV.C) on n=3 with the
      stepness mechanism panel: per-cell RSS slope with
      BH-FDR-controlled significance, and (corr, K_trim_dRSS,
      K_trim_dVMS, steps/h) classification under the five-class
      taxonomy.
   (d) Low-load ablation (E3b) and 2x2 factorial (Section IV.D-E)
       on n=3.
   Threats-to-Validity (Section V) declares the parallel topology as
   a feature of the design (realistic multi-tenant deployment), the
   confounded factorial (vLLM version drift across cells), single
   hardware, single model.

7. **Stepness analysis sequencing.**
   - Run `stepness.py` on all completed n=3 runs (r01 + r02 + r03 as
     they land). Warmup is auto-resolved from the cell YAML; pass
     `--warmup-s 3600` only as an explicit override. Record per-cell
     (corr, K_trim_dRSS, K_trim_dVMS, steps/h) per replica.
   - After r03, compute between-run CI for each metric (3 replicates
     per cell). Robust class assignment requires the metric to be
     stable across replicas.
   - Decide the headline mechanism claim per cell from the n=3
     majority-class assignment.

8. **e3 drop rate analysis.** Both e3_r01 and e3_r02 show ~15.6-15.7%
   dropped requests at the saturated target rate; e3_r03 at 13% with
   ~15h left to potentially rise. Investigate:
   - Are drops time-clustered (bursty) or uniformly distributed?
   - Do they correlate with GPU 2 VRAM/util spikes or with the
     process scheduler queue depth?
   - Is the drop pattern stable across r01/r02/r03 in time, or only
     in aggregate fraction?
   The finding is reportable as a property of the PyTorch naive
   baseline at saturated load in the n=3 paper (Section IV.D rate
   ablation).

9. **Long-tail TODO (post-campaign).**
   - Update `docs/experimental_protocol.md` to reflect the actual
     executed protocol (n=3, 36h, parallel topology, Qwen2.5-7B).
   - Fix `run_monitors.py` manifest collision (currently a workaround).
   - Restore Docker data-root on /home per ADR-002.
   - Complete refactor of `analysis/aging_trends.py` onto
     `analysis/aging_io.py`. Warmup discard, --csv mode, process_alive
     parsing already done in the 2026-05-19 evening hardening commit.
     **PARTIALLY RESOLVED 2026-06-29:** the `proc_prefix` discovery now
     calls `aging_io.discover_proc_prefix` (reads manifest, excludes
     gpu*/system) and the GPU prefix is discovered dynamically instead
     of hardcoded `gpu0`. This closes the non-deterministic monitor
     discovery defect (the cause of the 405-vs-520-rows anomaly).
     Remaining code-quality debt: `load_csvs` is still local to
     aging_trends and does not sort rows by `ts_unix` after concat
     (see Code review CR-A5 below).

---

## Code review (2026-06-29) — long-running-systems + scientific-integrity pass

Full read of the operational core (campaign / launch_cell / monitors /
client) plus the analysis pipeline, with the two failure classes that
matter for this study in mind: (1) silent measurement bias on a 36h+
window, (2) supervision gaps when a process dies uncleanly. Findings
were verified against the source where they touch published numbers.
Severity: HIGH = can corrupt a paper number or break a long run;
MED = real defect, narrow blast radius; LOW = defensive / cosmetic.
"Verified" = read in source this session; "reported" = from the review
sub-agents, not re-confirmed line-by-line.

This is the actionable backlog. The HIGH/MED data-integrity items were
applied on 2026-06-29 (see "Applied" below); the remaining
supervision/timestamp items are deferred because they need server-side
testing.

### Applied 2026-06-29 (verified with a before/after pipeline run on the 18 local runs)

- **CR-A1 DONE.** `aging_trends` now computes slopes on a real
  elapsed-hours axis (per-bin `_t_hours` carried through
  `downsample_to_minutes`/`downsample_client`), not `np.arange(n)`.
  **Verified:** proc/gpu/system per-run indicators are byte-identical
  before/after on all 18 runs (the monitor bins have no gaps, as
  Domenico independently confirmed), so the USS/RSS/VMS headline slopes
  and the per-cell DL-RE table are unchanged. Only `client.*` indicators
  on cells with real client gaps (e3/e3b low-rate windows with zero
  requests) shift, and they were non-significant either way. On a
  gap-free run the new axis equals `arange(n)*window/3600` by
  construction, so the fix is exact, not approximate.
- **CR-A2 DONE.** BH rejection unified to `q <= alpha` (BH 1995) in
  `aging_trends` and `fdr_aggregate` (manual + statsmodels paths now
  derive reject from `q <= alpha`), matching `aggregate_slopes`. No
  shared helper was added: `aggregate_slopes` is intentionally
  standalone (it states so), so only the rule was unified.
- **CR-A8 DONE.** `downsample_client` uses `truthy_series(ok["streaming"])`
  instead of `== True`.
- **Triton throughput DONE (the MED tokens_per_sec finding).**
  `tokens_per_sec` is NaN ("unavailable") when no request reports a token
  count, instead of summing all-NaN to a fake 0. **Verified:** e2_r02
  went from `slope=0,n=2101` to `n=0,slope=nan`; same for a2. This keeps
  Triton's non-existent throughput out of the catalog rather than a false
  zero trend.
- **CR-O3 DONE.** `Watchdog` returns a fresh sentinel per call, so
  multiple watchdog timeouts no longer share/reuse the first timeout's
  timestamp.
- **CR-O4 DONE.** `steady_sampler` docstring corrected to describe the
  actual catch-up behavior (no false "deadlines are skipped" claim).
- **CR-C1 DONE** (committed 2026-06-29 in a prior commit): arrival
  accumulator.
- **Client restart DONE.** The client `CsvRotatingWriter` resumes past
  existing `requests_*.csv` instead of restarting at `_seq=0` in "w"
  mode, so a restart into the same output dir no longer overwrites prior
  data.

Net effect on the paper: **no number or conclusion changes.** The
per-cell `proc.uss_bytes` DL-RE slopes and every significance boolean are
identical before/after; only sub-threshold BH q-values shift by ~1%
because the corrected client family changed the BH set. The committed
`paper/n3_analysis/` outputs are now slightly stale on the `client.*`
rows and the BH q column; regenerate them with the fixed pipeline when
convenient (proc/USS rows will be unchanged).

### Still deferred (need server-side testing, not applied)

- **CR-O1 (HIGH). Orphan reaper.** Not applied: a correct reaper has to
  interact with `start_new_session`, the sudo/setuid proc_monitor (which
  clears PR_SET_PDEATHSIG on exec), and PID recycling, and it cannot be
  validated off the real host. Proposed design: have launch_cell write
  child PIDs/PGIDs to `run_dir/child_pids.json` on spawn, and reap any
  stale same-run group at launch/retry; optionally PR_SET_PDEATHSIG for
  the non-sudo children. Implement and verify on cci-csgpu11.
- **CR-C "NTP/monotonic" (MED).** Not applied: measuring durations with
  `time.monotonic()` instead of `time.time()` (with the silent
  `max(0.0, ...)` clamp) is a cross-adapter refactor of RequestResult and
  the three protocol clients; it does not help already-collected data and
  deserves its own pass with a re-run smoke test.

### Original backlog (for reference)

### Analysis / statistics (touches paper numbers)

- **CR-A1 (HIGH, verified). Slope-per-hour inflated by dropped time
  bins.** `aging_trends.trend_one_indicator` builds `x = np.arange(n)`
  and divides by `dt_hours` (`downsample_seconds/3600`), but
  `downsample_to_minutes` drops empty bins (`groupby().median()`), and
  the `process_alive=True` filter creates exactly such gaps on engine
  restarts. Slope is then overestimated proportional to the
  missing-bin fraction. Exact (~0 error) on clean gap-free runs; bites
  runs with restarts / sample_error stretches. **Recheck E2** (the
  r02>>r01 "disjoint" cell, Triton worker churn). `validation_check.py`
  already does the right thing (Theil-Sen on real `ts_unix`). Fix:
  carry per-bin median `ts_unix` through downsampling, pass real
  elapsed-hours as `x`, drop the `/dt_hours` rescale.

- **CR-A2 (MED, verified). BH rejection boundary inconsistent across
  the three aggregators.** `aging_trends.main` and `fdr_aggregate` use
  `q < alpha`; `aggregate_slopes` uses `q <= alpha`; statsmodels path
  uses its own. BH 1995 is `<=`. Only bites at the exact boundary or
  across machines (statsmodels present/absent), but it is
  non-determinism in a paper-grade pipeline. Fix: one shared
  `aging_io.bh_fdr` (`<=`), derive `reject` from q even on the
  statsmodels path.

- **CR-A3 (MED, verified). No PID segmentation in the slope pipeline.**
  `stepness.py` segments deltas per PID; `aging_trends` filters
  `process_alive` but runs Theil-Sen/MK across PID boundaries. Theil-Sen
  (median) is robust to a single restart jump; MK's S and the lag-1 rho
  are not, and level indicators (vram_used, num_fds) reset. Fix: apply
  the same pid-change mask before level trends, or document why levels
  tolerate restarts.

- **CR-A4 (MED, verified, methods-description). Docstring overclaims
  the autocorrelation correction.** The Theil-Sen CI uses a lag-1
  AR(1) inflation `(1+rho)/(1-rho)`; the MK test uses pymannkendall's
  Hamed-Rao. The docstring/header calls the AR(1) factor the "leading
  Hamed-Rao term", which it is not. The dual correction is defensible;
  the wording will draw a reviewer. Fix: state honestly (CI = AR(1)
  lag-1 approximation, test = Hamed-Rao), or recompute the CI variance
  from the rank autocorrelation.

- **CR-A5 (LOW, verified). Theil-Sen CI off-by-one on the order
  statistics.** `sen_slope_and_ci` uses the rank as a 0-based index
  without the `-1` conversion, shifting both CI bounds up by one
  order statistic (slightly anti-conservative on "CI excludes 0").
  Numerically negligible for headline indicators (M ~ 2e6 pairwise
  slopes at n~2160); only matters at small n (client indicators).
  Fix + add a unit test against a Gilbert worked example.

- **CR-A6 (LOW, verified). `load_csvs` does not sort rows by `ts_unix`
  after concatenating rotated files.** Safe today (zero-padded
  monotonic filenames) but fragile; `aging_io.load_proc` already sorts.
  Add `.sort_values(ts_col).reset_index(drop=True)`.

- **CR-A7 (LOW, verified). GPU prefix discovery still keys on the
  first-rotation file** (`gpu*_000000.csv`). If `_000000` is missing
  the whole GPU family is silently dropped (same 405-vs-520 class).
  Robust fix: mirror `discover_proc_prefix` (glob `gpu*_*.csv`, regex
  the `_\d{6}.csv` suffix, common stem).

- **CR-A8 (LOW, verified, defensive). `streaming == True` instead of
  `truthy_series`** in `downsample_client` (and `sweep_summary`).
  Probably harmless (pandas infers bool dtype) but inconsistent with
  the defense already adopted elsewhere; a mixed-dtype column would make
  the TTFT family vanish silently.

### Operational / long-running supervision

- **CR-O1 (HIGH, verified). No orphan reaper if `launch_cell` dies
  uncleanly.** Monitors + client run with `start_new_session=True`;
  cleanup lives only in `launch_cell`'s `finally`. On SIGKILL / OOM /
  crash they keep running to their own 36h deadline (proc_monitor under
  sudo) and the container is never torn down. Worse: monitors reopen
  output files by path with `mkdir(parents=True)` each rotation, so if
  the orchestrator archives/renames the run_dir, orphans recreate the
  original dir and collide with attempt 2's CSVs. Fix: write monitor/
  client PIDs to a file and reap stale same-run processes at launch /
  retry, or use PR_SET_PDEATHSIG / `systemd-run --scope`.

- **CR-O2 (MED, verified). Race on `current_proc` during shutdown.**
  `campaign.SlotWorker.interrupt()` (signal thread) reads
  `self.current_proc` while the worker nulls it in `finally`;
  `AttributeError` between the None-check and `.send_signal` can swallow
  the shutdown of the other slots. Fix: snapshot to a local, catch
  `AttributeError`.

- **CR-O3 (MED, verified). Watchdog sentinel is a shared, mutated
  dict.** `gpu/proc/system_monitor` pass one sentinel object;
  `steady_sampler` does `setdefault("ts_unix", ...)`, so after the
  first watchdog timeout every later timeout row reuses that first
  timestamp. Rare (NVML/psutil deadlock only) but real timestamp
  corruption. Fix: build a fresh sentinel per call, or `dict(sentinel)`
  inside `Watchdog.call`.

- **CR-O4 (LOW, verified). `steady_sampler` catch-up vs documented
  skip.** The docstring says overruns "skip deadlines (no backfill)";
  the code fires the backlog back-to-back (`n += 1` while
  `deadline = start + n*period`). With CR-O3 it produces a burst of
  degenerate-timestamp rows. Downsampling to 60s masks most of it.
  Fix: skip (`n = ceil((now-start)/period)`) or fix the docstring.

- **CR-O5 (LOW, verified, durability). No periodic flush in
  `CsvRotatingWriter`** (monitors and client): rows sit in the Python
  buffer until rotation/close, so a hard kill loses up to one rotation
  window. The "bounded data loss" claim only holds with a periodic
  flush. Add `flush()` per row (cheap).

### Client / load generator

- **CR-C1 (HIGH, verified). RESOLVED 2026-06-29.** Arrival process
  drifted: `benchmark.run` did `next_arrival = time.monotonic() + inter`
  instead of `next_arrival += inter`. Per-iteration cost (RNG, task
  creation, the synchronous CSV write on the drop path) was added to the
  inter-arrival gap, so the realized rate sat below target AND the
  deficit grew as the server aged (busier loop, more inline drop I/O),
  coupling the independent variable (offered load) to the dependent
  variable (aging). Fixed to the accumulator form `next_arrival += inter`
  (the existing `if now < next_arrival` guard handles catch-up correctly
  for open-loop Poisson). NB: the runs already on arXiv were generated
  with the drifting scheduler; the bias direction is "offered slightly
  below target", which is conservative for an aging claim (we under-, not
  over-, stressed), but quantify it before reusing those numbers in the
  journal extension.

- **CR-C2..C7 (reported, not re-verified line-by-line):** wall-clock vs
  monotonic for latency/TTFT with NTP steps over 36h silently clamped
  to 0 (`fill_derived_latencies`); scalar `httpx.Timeout` with no
  per-chunk read timeout (a stalled stream holds a slot 600s); restart
  overwrites `requests_000000.csv` (writer restarts at `_seq=0` in "w"
  mode); `asyncio.Event.set()` from an OS signal handler is not
  loop-safe (use `loop.add_signal_handler`); no per-row flush (= CR-O5);
  **to verify in analysis:** whether Triton `extras.output_chars` is
  converted to `actual_output_tokens` (else token/ITL metrics missing
  for one of three engines → cross-engine bias).

### Priority order before the next campaign

1. ~~**CR-C1** (arrival accumulator)~~ — DONE 2026-06-29.
2. ~~**CR-A1** (real time axis in slopes)~~ — DONE 2026-06-29, verified
   no change to USS/RSS/VMS headline numbers.
3. **CR-C2** (monotonic vs NTP) — silent TTFT bias. DEFERRED (cross-adapter
   refactor, server re-test needed).
4. **CR-O1** (orphan reaper) — DEFERRED (server-side test needed);
   ~~**CR-O2**~~ (shutdown race) still open, cheap, apply next.
5. ~~**CR-A2**~~ DONE (BH `<=` unified). **CR-A4 / CR-A5** still open —
   reviewer-proofing (AR(1) docstring, CI off-by-one unit test).
6. **CR-O5 / CR-C2 (read timeout)** — durability + no slot leak.

---

## Files and paths

On the laptop (this repo):
- `EXPERIMENT_STATE.md` (this file)
- `docs/project-wosar.md` (longer-form project doc)
- `docs/WOSAR_2026.pdf` (preprint as submitted; background only)
- `campaigns/wosar2026/{campaign.yaml, cells/*.yaml}` (campaign config)
- `scripts/{campaign.py, launch_cell.py, smoke_test_run.sh,
  campaign_health.sh}` (campaign machinery)
- `monitoring/{gpu_monitor.py, proc_monitor.py, system_monitor.py,
  run_monitors.py, find_engine_pid.py, _common.py}` (monitoring)
- `client/{run_client.py, config.yaml, prompts/arxiv_corpus.jsonl,
  protocols/*.py}` (workload)
- `analysis/{validation_check.py, aging_trends.py, fdr_aggregate.py,
  aggregate_slopes.py, stepness.py, aging_io.py}` (paper pipeline)
- `engines/{vllm_standalone, triton_vllm, pytorch_naive}/` (engine
  definitions, Dockerfiles, model_repository for Triton)

On the server (cci-csgpu11):
- `~/wosar/llm-serving-bench/` (this repo, checked out)
- `~/wosar/runs/wosar2026_<cell>_r<NN>/` (current campaign runs)
- `~/wosar/runs_n1_baseline/aging_pilot_24h_*/` (preprint pilot runs;
  not used in the camera-ready)
- `~/wosar/runs_aborted_20260516_052308/` (failed first attempt)
- `~/wosar/hf_cache/` (HuggingFace cache, mounted into all containers)

---

## Standard commands (server, in ~/wosar/llm-serving-bench)

```bash
# Periodic health check, run every 6-12h during the campaign
bash scripts/campaign_health.sh 2>&1 | tee /tmp/health.log
echo "exit code: ${PIPESTATUS[0]}"
# Exit 0=OK, 1=WARN (campaign OK, inspect when convenient), 2=FAIL (intervention needed)

# Per-run post-completion sanity verdict (NOT paper pipeline)
python3 analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_<cell>_r<NN>

# All r01 verdicts in batch
for cell in e1 e2 e3 a1 a2 e3b; do
  echo "=== ${cell}_r01 ==="
  python3 analysis/validation_check.py --run-dir ~/wosar/runs/wosar2026_${cell}_r01
done

# Paper pipeline: aging_trends per run (MK Hamed-Rao + Theil-Sen CI)
python3 analysis/aging_trends.py --run-dir ~/wosar/runs/wosar2026_<cell>_r<NN> --csv \
  > /tmp/<cell>_<NN>_trends.csv
# Stderr will show "warmup_s = 3600 (campaign)" for wosar2026_* runs.

# Aggregate across runs with BH-FDR at q=0.10
python3 analysis/fdr_aggregate.py \
  --trends-csv /tmp/*_trends.csv \
  > /tmp/fdr_results.csv
# Adds q_value and bh_reject columns. Decision rule for per-run analysis:
# significant trend iff mk_p<0.01 AND slope_ci excludes 0 AND bh_reject==True.

# Per-cell aggregation across the n=3 replicas (camera-ready source of truth).
# Primary: DL random-effects on the 3 per-replica TS slopes. Robustness:
# median of the 3 slopes with [min, max] = 75% non-parametric CI for the
# population median. Per-cell BH-FDR on the Stouffer-combined MK p.
python3 analysis/aggregate_slopes.py \
  $(ls /tmp/wosar2026_*_r0[123]_trends.csv | sed 's|^|--trends-csv |') \
  --alpha 0.10 --expected-replicas 3 \
  > /tmp/per_cell.txt
# Decision rule for paper: RE_significant iff (BH rejects Stouffer p)
# AND (RE 95% CI excludes 0) AND (n_replicas/k_used_RE/k_used_stouffer >= 3).
# Pooled-median CI is reported as a 75% robustness CI, NOT a 95% test.

# Stepness panel (corr, K_trim_dRSS, K_trim_dVMS, steps/h) per run.
# Warmup is auto-resolved from the campaign cell YAML.
python3 analysis/stepness.py --run-dir ~/wosar/runs/wosar2026_<cell>_r<NN>

# Or all campaign runs in one shot
python3 analysis/stepness.py --logs-root ~/wosar/runs

# Log a manual mitigation (e.g. after running docker prune by hand)
echo "$(date -Iseconds) | <category> | <free-text note>" \
  >> campaigns/wosar2026/state/mitigations.log
# categories: disk_prune, container_restart, engine_relaunch,
#             gpu_intervention, workload_param_change, host_intervention
```

---

## Pipeline analytical details (for paper)

The pipeline is fully in-repo as of 2026-05-19 evening, split across
four scripts. All four share warmup resolution and CSV parsing via
`analysis/aging_io.py`.

- **Trend detection**: Mann-Kendall with Hamed-Rao correction for
  autocorrelation. Significance at p < 0.01. Implemented in
  `analysis/aging_trends.py`.
- **Slope estimation**: Theil-Sen with 95% CI, computed on the real
  `ts_unix` axis (not sample indices). Variance inflated by lag-1
  AR(1) factor (1+rho)/(1-rho). Implemented in `aging_trends.py`.
- **Multi-test correction**: Benjamini-Hochberg FDR at q = 0.10.
  Two granularities are exposed:
  - Per-run (run_id, indicator) BH-FDR in `analysis/fdr_aggregate.py`,
    consuming `aging_trends.py --csv` outputs and adding `q_value`
    and `bh_reject` columns. Used for per-run diagnostics and for the
    intermediate "as runs come in" tables in this document.
  - Per-cell (cell_id, indicator) BH-FDR in
    `analysis/aggregate_slopes.py`, where the per-cell p is the
    Stouffer z-score combination of the k per-replica MK z values.
    Used for the camera-ready Section IV tables: one row per
    (cell, indicator) over the full n=3 family.
- **Per-cell aggregation across replicas**: `analysis/aggregate_slopes.py`
  emits two estimators per (cell_id, indicator) computed entirely on
  the per-replica trend CSVs:
  - Primary `slope_RE`: DerSimonian-Laird random-effects on the k
    Theil-Sen slopes, with per-replica SE derived from the upstream
    Theil-Sen 95% CI via Gaussian-equivalent `(hi - lo) / (2 * 1.96)`.
    Reports `slope_RE`, `ci_RE`, `tau2`, `I^2`, `Q`, `Q_pvalue`,
    `k_used_RE`. Used as the headline n=3 estimator.
  - Robustness `slope_pooled` (median): sample median of the k
    per-replica slopes, with `[min, max]` reported as the exact
    non-parametric CI for the population median. Coverage from order
    statistics: `1 - 2 * (1/2)^k` → 75% at k=3, 87.5% at k=4. Used
    only as a cross-check; NOT a 95% test.
  Headline `RE_significant`: (cell-level BH rejects Stouffer p) AND
  (RE 95% CI excludes 0) AND (n_replicas/k_used_RE/k_used_stouffer
  ≥ `--expected-replicas`, default 3). Rows below the replica
  threshold are excluded from `RE_significant` and marked with `*`
  in the human-readable table; rows with `I^2 > 75%` are marked with
  `!` to surface high between-replica heterogeneity.
- **Decision rule (per-run, pre-aggregation)**: a trend is declared
  significant when ALL of: (a) MK Hamed-Rao p < 0.01, (b) Theil-Sen
  95% CI excludes zero, (c) bh_reject is True. The first two come
  from `aging_trends.py`, the third from `fdr_aggregate.py`.
- **Decision rule (per-cell, camera-ready)**: `RE_significant` as
  defined above, from `aggregate_slopes.py`. Source of truth for
  Section IV tables on the n=3 campaign.
- **Canonical leak indicator: USS, not RSS.** Decided 2026-05-23
  based on the full n=2 (12 runs) decision-rule outcome:
  - USS = 12/12 paper-grade significant; RSS = 11/12 (a1_r02 fails
    on Theil-Sen CI lower bound = 0, an AR(1) inflation artifact
    isolated to that single run where rho_RSS = 0.99 vs
    rho_USS = 0.005).
  - Point estimates of USS and RSS agree within <10% on all 12
    runs (the underlying trend is the same; what differs is the
    autocorrelation structure of the residuals).
  - USS is semantically cleaner: it counts only process-private
    resident pages. RSS includes shared mappings (libraries, code,
    file-backed read-only segments) which are a confound for
    "leak rate per process".
  - VMS is the third indicator, reserved for the VAS-reservation
    pattern: it is the sum of all virtual memory regions of the
    process, including reserved-but-uncommitted anonymous mmaps.
    VMS - RSS measures address-space reservation rate without
    corresponding physical commitment, which is the signature of
    pool allocators with lazy commit (e.g. PyTorch CUDA caching
    allocator host-side metadata; see Status snapshot for the
    e3/e3b r02 finding).

  Camera-ready convention: USS is primary in all paper tables and
  figures. RSS appears in the same tables as a secondary column.
  VMS appears separately for the PyTorch+HF cells where it is
  paper-grade significant. RSS-vs-USS comparison is *not*
  load-bearing for any narrative claim; the divergence on a1_r02
  is reported as a methodological note in Section IV.B.

- **Stepness panel**: `corr` (RSS-VMS lag-0), `K_trim_dRSS`,
  `K_trim_dVMS`, `steps_per_h_1mb`, and `mean_top1_step_mb`
  (top 1% of positive ΔRSS jumps). Implemented in
  `analysis/stepness.py` with PID-segmented deltas and sparse-safe
  bootstrap CIs. Used to classify cells under the five-class taxonomy.
- **Per-run sanity gate**: `analysis/validation_check.py` is the
  lightweight per-run verdict tool (PASS/SOFT FAIL/HARD FAIL on
  RSS slope direction). Uses Theil-Sen on real time axis but does
  NOT compute Hamed-Rao or apply BH-FDR; explicitly NOT the paper
  pipeline. For paper-quality numbers always use
  `aging_trends.py + fdr_aggregate.py`.
- **Magnitude criterion**: open question whether to add an
  operationally meaningful threshold (e.g. slope > 1 MB/h) on top
  of the statistical significance. Discussed but not implemented.

---

## Step-wise mechanism panel (paper Section IV.E)

The three metrics, definitions, and classification rule.

1. **Primary — `rss_vms_corr`**: lag-0 cross-correlation of dRSS and
   dVMS post-warmup. Identifies the MECHANISM:
   - corr > 0.8 → lock-step → mmap-style allocation (RSS and VMS
     grow together: kernel-mapped blocks, never released)
   - corr < 0.5 → not lock-step → either continuous drift or
     sbrk-style heap-internal accumulation
2. **Secondary — `K_trim`**: excess kurtosis of ΔRSS after
   winsorization at the 99.9 percentile. Quantifies the INTENSITY
   of the step-wise pattern, robust to single outliers:
   - K_trim > 10 → tail-heavy → punctuated dynamics
   - K_trim < 5 → gaussian-like → continuous drift
3. **Operational — `steps_per_h_1mb`**: count of ΔRSS > 1 MB per
   hour of runtime. Reader-friendly descriptor: "this cell shows
   N step events of at least 1 MB per hour."

Raw `K` is reported alongside `K_trim` for transparency but does not
enter the classification rule (it is dominated by single outliers
and not cross-cell comparable).

**Current five-class classification rule.** Apply jointly:

| pattern                | condition                                           | mechanism interpretation |
|------------------------|-----------------------------------------------------|--------------------------|
| border (VMS missing/unusable) | `VMS_missing` or `VMS_unusable` in notes | Cannot classify on the three-axis panel without a usable VMS axis; flag for replica review. |
| continuous drift (low-step fallback) | both usable axes in low-step operational fallback | No significant step events on either axis; corr is micro-noise correlation, not mechanism. |
| mmap-style step-wise   | corr > 0.8 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10 | RSS and VMS step together at discrete events → kernel-mapped blocks never returned |
| sbrk-style step-wise   | corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS < 5 | RSS heap-arena extends without paired VMS step |
| VAS-only step-wise     | corr < 0.5 AND K_trim_dRSS < 5 AND K_trim_dVMS > 10 | VMS-only jumps, address space reserved without paging-in |
| uncorrelated step-wise | corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10 | RSS and VMS both jump but desynchronized |
| continuous drift       | corr < 0.5 AND K_trim_dRSS < 5 AND K_trim_dVMS < 5 | smooth small-grain accumulation, no big steps |
| border                 | mixed                                               | needs n=3 confirmation |

**Why this is a paper-worthy refinement.** The preprint Section IV.E
mentions both mmap and sbrk-extended heap as alternative hypotheses
but does not distinguish them. The (corr, K_trim) pair separates them
for the first time and is computable from the existing proc CSVs with
no new instrumentation. The headline mechanism claim of the n=3 paper
is the per-cell class assignment on n=3 data.

**Status of stepness metric panel:**
- Implemented in `analysis/stepness.py`.
- Current implementation reports dRSS and dVMS kurtosis, PID-segments
  deltas across engine restarts, treats missing/unusable VMS as
  `border`, and uses sparse-safe bootstrap CIs.
- Pilot n=1 retrospective under the five-class rule: E2 =
  mmap-style step-wise; E1 = VAS-only step-wise; A1 =
  uncorrelated step-wise; E3/E3b = continuous drift; A2 =
  continuous drift via both-axis low-step fallback.
- Campaign r01/r02 status (2026-05-21): E2 is mmap-style in both
  replicas; E1 and E3 are continuous drift in both replicas. E3 has a
  significant VMS slope, but `K_trim_dVMS` remains ~1, so it is smooth
  VMS drift rather than VAS-only step-wise.
- Current taxonomy:

  | pattern                                | condition                                                     | mechanism |
  |----------------------------------------|----------------------------------------------------------------|-----------|
  | border (VMS missing/unusable)          | `VMS_missing` or `VMS_unusable` in notes (cell breakage, monitor crash, no finite ΔVMS samples) | highest-priority short-circuit: cannot classify on (corr, K_trim_dRSS, K_trim_dVMS) without a usable VMS axis; returning drift would silently swallow missing data. |
  | continuous drift (low-step fallback)   | both usable axes in low-step operational fallback (steps/h < 0.01) | no significant step events on either axis; corr is noise correlation, not mechanism. Priority short-circuit before the metric-based rule (but yields to missing/unusable VMS). |
  | mmap-style step-wise                   | corr > 0.8 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10           | RSS+VMS lock-step, kernel-mapped blocks never returned |
  | sbrk-style step-wise                   | corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS < 5            | RSS heap-arena extends without kernel mmap |
  | VAS-only step-wise                     | corr < 0.5 AND K_trim_dRSS < 5  AND K_trim_dVMS > 10           | VMS-only jumps, address space reserved without paging-in |
  | uncorrelated step-wise                 | corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10           | RSS and VMS both jump but desynchronized; heap-arena + mmap operating independently |
  | continuous drift                       | corr < 0.5 AND K_trim_dRSS < 5  AND K_trim_dVMS < 5            | smooth small-grain accumulation everywhere |
  | (border)                               | any other combination                                          | mid-corr (0.5-0.8) with significant step events, or mixed K_trim; needs replica confirmation |

  The five-class taxonomy with priority short-circuits is implemented in
  `analysis/stepness.py` and documented in `analysis/README.md`. Change log
  on 2026-05-20 / 2026-05-21:
  - first commit: K_trim_dVMS metric added, K_trim=NaN math fallback,
    five-class rule.
  - fix n.1: low-step fallback made operational-driven (`steps/h < 0.01`)
    regardless of K_trim numeric value.
  - fix n.2 (commit `1c84e9e`): classify_stepness short-circuits to
    "continuous drift" when both axes are in low-step fallback, before
    the corr-based rule. Required because corr in the grey zone 0.5-0.8
    on a low-step run is correlation of micro-noise, not of mechanism.
  - fix n.4: PID-aware diff segmentation. `analyze_run` now loads
    `pid` and masks the diff row at every PID transition on both
    ΔRSS and ΔVMS. Without this, an engine restart (PID bump, RSS/VMS
    reset to the new process footprint) injects an O(GB) artifact
    step into the diff series. Synthetic test: a single PID change
    with no intra-PID jumps > 1 MB produced steps_per_h_1mb=120 and
    a +899.5 MB false delta under the unsegmented code; under the
    fix steps_per_h_1mb=0.000 as expected. Local pilot regression
    test: zero PID transitions detected (single-engine), all 6
    classes and numeric metrics identical to the pre-fix run.
    Diagnostic: stderr warning logs the number of PID transitions
    per run when > 0.
  - fix n.3: (a) `mean_top1_step_mb` recomputed on `arr[arr > 0]` with a
    top-N sort instead of `arr >= np.percentile(arr, 99)`. The old
    formula collapsed to ≈ 0 on zero-heavy sparse series because p99 of
    a mostly-zero series is 0 and the mask then admitted every
    non-negative sample. Sanity: E2 pilot top1% under the old code was
    0.005 MB (incompatible with steps>1MB/h=1.23), under the fix it is
    2.57 MB. The metric is descriptive only — does not enter the
    classification rule — but is paper-table material for Section IV.E.
    (b) `VMS_missing` now short-circuits to `border` ahead of the
    both-fallback rule. Safety net: when vms_bytes is absent the
    low-step fallback would otherwise fire spuriously on the empty
    array (0 < 0.01) and inject `VMS_low_step_operational_drift`,
    which the priority rule would then misread as drift. Currently
    psutil reports vms_bytes for all alive processes so the path is
    dormant on existing data, but a monitor crash in r03 would have
    been silently misclassified.
  - fix n.5: VMS present-but-unusable is also guarded. If `vms_bytes`
    exists but has no finite adjacent post-warmup ΔVMS samples after
    PID segmentation (for example all-NaN output), the row receives
    `VMS_unusable` and is classified as `border` instead of falling
    through to both-axis low-step drift. `--top-k` timestamps now use
    the same masked diff indices as `_diff_rss`, so PID-transition
    masking cannot shift event times. Bootstrap CIs now discard
    undefined constant-resample kurtosis values, reject
    `--bootstrap 0`, and recompute winsorization inside each K_trim
    bootstrap resample.

---

## Open questions for the next session

All within-n=3. No comparison with the preprint.

1. **Between-replica variance per cell. CI-AWARE READING ON n=2
   (USS, full 12-run picture as of 2026-05-23).**
   - **a1**: r01 CI [9.5, 16.0] KB/h and r02 CI [12.9, 13.4] KB/h.
     r02 CI is *inside* r01 CI. Highly reproducible. Most stable
     cell of the campaign on USS.
   - **a2**: r01 CI [15.6, 61.9] KB/h and r02 CI [1.4, 87.4] KB/h.
     r02 CI contains r01 CI. CI-compatible, but r02 CI very wide
     (factor 60 between lo and hi). Point estimate dropped 14x but
     within noise.
   - **e1**: r01 CI [1.5, 22.3] KB/h and r02 CI [0.7, 23.8] KB/h.
     Overlap massive; the apparent point ratio of 0.26 is
     consistent with Theil-Sen sample variance on low-step data.
   - **e2**: r01 CI [10.9, 29.0] KB/h and r02 CI [33.2, 228.1] KB/h.
     DISJOINT by 4.2 KB/h on USS. r02 slope ~8x higher. Real
     between-run effect on this cell.
   - **e3**: r01 CI [5.0, 19.5] KB/h and r02 CI [21.4, 87.9] KB/h.
     DISJOINT by 1.9 KB/h on USS. ~2.8x point ratio. Borderline
     real effect.
   - **e3b**: r01 CI [15.7, 132.2] KB/h and r02 CI [56.2, 206.1]
     KB/h. Overlap (intersection [56.2, 132.2]). CI-compatible
     despite 2.7x point ratio.

   Pattern: both CI-disjoint cells (e2, e3) are in slot batch 1
   (e1/e2/e3 ran in parallel) and both have r02 > r01 in the
   disjoint direction. e1 in the same batch is CI-compatible. All
   three batch-2 cells (a1, a2, e3b) are CI-compatible. Plausible
   causes if real: cumulative ambient state on the host, allocator
   path divergence between seeds, or batch-1-specific noise. The
   host-side `system.mem_used_bytes` was 41.5 MB/h in batch 1 of
   r01 and 14 MB/h in batch 1 of r02 (3x lower); batch 2 always
   shows 1.4-3.2 MB/h system slope (more than an order of magnitude
   below batch 1). System memory growth rate is driven by aggregate
   batch workload intensity, not by cell-specific behavior.

   Question for r03: does e2 (and e3) regress toward r01, or does
   r02 confirm a real drift in the slope estimate?

2. **Stepness class assignment per cell on n=3 — FULL 12-RUN PICTURE
   AFTER FIX P1 (2026-05-21 afternoon ET).** All 9 completed + 3
   in-progress runs analyzed via the five-class taxonomy with priority
   short-circuits. Headline: **only E2 is step-wise on n=3 campaign;
   all other 5 cells fall into continuous drift.**

   | cell | r01 | r02 | classe |
   |------|-----|-----|--------|
   | E1   | corr=0.64, K_raw=4053, steps/h=0, top1%=0.31 MB    | corr=0.31, K_raw=9527, steps/h=0, top1%=0.52 MB     | continuous drift (×2, via low-step short-circuit) |
   | E2   | corr=0.82, K_trim 284/239, top1%=1.73 MB, steps/h=0.23 | corr=0.83, K_trim 649/491, top1%=1.82 MB, steps/h=0.09 | **mmap-style step-wise (×2)** |
   | E3   | corr=0.37, K_trim 1.1/1.2, top1%=0.21 MB           | corr=0.24, K_trim 1.1/1.1, top1%=0.26 MB            | continuous drift (×2) |
   | A1   | corr=0.34, K_raw=5306, steps/h=0, top1%=0.094 MB   | corr=0.44, K_raw=3420, steps/h=0, top1%=0.082 MB (11h, in-progress) | continuous drift (×2, via low-step short-circuit) |
   | A2   | corr=0.55, K_raw=3178, steps/h=0, top1%=0.052 MB   | corr=0.25, K_raw=2907, steps/h=0, top1%=0.151 MB (11h, in-progress) | continuous drift (×2, via low-step short-circuit) |
   | E3b  | corr=0.36, K_trim 2.2/2.2, top1%=0.44 MB           | corr=0.35, K_trim 2.2/1.9, top1%=0.67 MB (11h, in-progress) | continuous drift (×2) |

   **E2 (Triton + vLLM V0) is the only mmap-style step-wise cell on
   n=3.** Both r01 and r02 with corr > 0.8, K_trim_dRSS > 280,
   K_trim_dVMS > 230, top1%_step ~1.7-1.8 MB, steps>1MB/h between
   0.09 and 0.23. Class and magnitude both reproducible across n=2.

   **The other 5 cells are all continuous drift.** A1 and A2 of the
   campaign have K_raw very high (3000-5300) but `steps>1MB/h=0` →
   low-step operational fallback fires → continuous drift via priority
   short-circuit. The high K_raw is driven by micro-noise outliers
   sub-MB, not by MB-scale step events. Mechanism interpretation:
   in the n=3 setting (parallel multi-tenant, 36h, 85% saturation)
   these cells exhibit fat-tailed sub-MB delta distributions, which
   the operational criterion (events > 1 MB) correctly classifies
   as drift.

   **Pilot-vs-campaign divergence on A1 and E1.** The pilot
   retrospective put A1 as "uncorrelated step-wise" (K_trim_dRSS=402,
   K_trim_dVMS=581, steps>1MB/h=0.17) and E1 as "VAS-only step-wise"
   (K_trim_dVMS=789, steps>1MB/h=0.085). The campaign puts both into
   continuous drift because all step events have collapsed below the
   1 MB operational threshold. Same direction as the slope-magnitude
   collapse we observed earlier (campaign slopes 3-260x smaller than
   pilot). The mechanism class shift is consistent: stepness events
   exist (K_raw is high) but no longer exceed the MB-scale threshold,
   so the cell is operationally drift on the campaign.

   **The headline mechanism finding of the paper.** Under the n=3
   setting, mmap-style step-wise allocation (RSS+VMS lock-step jumps
   of >1 MB, top 1% magnitude ~1.7-1.8 MB) emerges **exclusively in
   the Triton + vLLM V0 deployment (E2)**, with class and magnitude
   reproducible across n=2 replicas (n=3 pending r03). All other
   five deployments (vLLM standalone V1/V0 (E1, A1), Triton + vLLM
   V1 (A2), naive PyTorch+HF at saturated and low rate (E3, E3b))
   exhibit continuous drift under the same workload, with **no
   MB-scale step events**. This is a stronger finding than "five
   mechanism classes observed": a single canonical example, null
   findings on five contrast cells, replicable.

   **Implication for the 5-class taxonomy.** The classes
   "uncorrelated step-wise" and "VAS-only step-wise" are instantiated
   ONLY by pilot retrospective (A1 pilot and E1 pilot respectively),
   not by any cell on the n=3 campaign. For the paper:
   - The classes remain in the taxonomy table (mechanism-justifiable
     a priori; pilot retrospective is in-vitro empirical existence).
   - The paper text states explicitly: "in this n=3 setting no cell
     instantiates the uncorrelated step-wise or VAS-only step-wise
     classes; pilot retrospective observations are reported as
     supporting evidence that the classes are observable in principle,
     while the n=3 campaign data shows continuous drift is the
     dominant non-mmap pattern under realistic multi-tenant
     deployment."

   Pilot n=1 reclassification under the five-class rule + fix n.2
   (retrospective only, does not feed the paper headline):
   - E2 pilot → mmap-style step-wise
   - E1 pilot → VAS-only step-wise
   - A1 pilot → uncorrelated step-wise
   - E3, E3b pilot → continuous drift
   - A2 pilot → continuous drift (was border; reclassified after fix n.2)

3. **Stepness metric stability across replicas.** Within each cell,
   does the (corr, K_trim) point stay in one classification region
   across r01/r02/r03, or does it flicker across the boundary?
   Robust class assignment requires the metric to be stable.

4. **e3 drop rate stability and mechanism.** Two replicas at
   ~15.6-15.7%. Does r03 confirm? Is the drop mechanism client-side
   concurrency cap exhaustion, server-side asyncio.Lock starvation,
   or HTTP timeout from prolonged GPU occupation? Each gives a
   different one-line explanation in Section IV.D.

5. **Topology framing for Threats to Validity.** The n=3 campaign
   runs three cells in parallel on three GPUs. This is a realistic
   multi-tenant deployment topology, declared as such in the design
   (Section III). Section V should articulate explicitly that the
   reported aging signatures are properties of the deployment under
   parallel-tenant CPU contention, not of the engine in isolation.

---

## Archive (pre-reframing, kept for traceability)

The following items were active before the 2026-05-19 evening paper
framing decision. They assumed the camera-ready would include a
side-by-side comparison with the n=1 preprint Table IV. That
direction is no longer in scope. The work done under that framing
is retained here so that someone reading the git history (or
finding a `replicate_n1.py` script in the repo) understands what
it was for.

**Internal sanity check (no longer active).** Before the framing
decision, an internal sanity check compared r01 slopes against
preprint Table IV and observed differences of 3 to 260x. The
dominant working hypothesis was the parallel-topology effect: the
preprint ran one cell at a time on a single GPU; n=3 runs three
cells in parallel on three GPUs. The fact that e3b at sub-saturated
load (50 req/h) matched preprint magnitudes, while e3 at saturated
load (624 req/h) did not, was consistent with a topology- or
saturation-driven explanation. None of this analysis is required
for the standalone-n=3 paper.

**a1-isolated diagnostic (no longer planned as a paper deliverable).**
A plan to stop the campaign mid-way and run a1 in isolation on
gpu0 for 24h (matching preprint conditions) was on the table as
the cleanest topology-effect test. It is no longer required for
the camera-ready. If time permits post-campaign, it remains an
interesting mechanism question but does not block the paper.

**replicate_n1.py (frozen).** A one-shot script at the repo root
that reads the local n=1 CSVs and reproduces preprint Table IV
numbers via Theil-Sen within 5-20%. It was used to validate the
analytical pipeline against the published preprint numbers (ran
on 2026-05-19). The script is frozen and is not re-run for the
camera-ready. If rerun elsewhere, it has a hardcoded `BASE` path
tied to the original Cowork sandbox and needs to be parametrized
via a `--base PATH` CLI flag. Not a blocker.

**Preprint Table IV (background only, no longer a reference).** The
headline slope table from the preprint:

| ID  | Deployment         | RSS slope    | 95% CI               |
|-----|--------------------|--------------|----------------------|
| E1  | vLLM standalone V1 | +9.15 MB/h   | [+9.03, +9.24] MB/h  |
| E2  | Triton + vLLM V0   | +2.04 MB/h   | [+0.82, +3.12] MB/h  |
| E3  | PyTorch + HF naive | +170 KB/h    | [+85, +389] KB/h     |
| E3b | PyTorch + HF low   | +179 KB/h    | overlapping with E3  |
| A1  | vLLM V0 standalone | +530 KB/h    | (Table V of preprint)|
| A2  | Triton + vLLM V1   | +20 KB/h     | (Table V of preprint)|

These numbers do not appear in the camera-ready and are not the
target of any analysis here. Kept as a record of what the preprint
reported.
