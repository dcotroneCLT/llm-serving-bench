# EXPERIMENT_STATE.md

Living hand-off document for the WoSAR 2026 n=3 campaign. Updated by hand
whenever something material changes. Designed so a new chat session (or
a co-author) can pick up the thread in under five minutes.

Last updated: 2026-05-20 (afternoon ET).

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
- **Three-metric step-wise panel** (`rss_vms_corr`, `K_trim`,
  `steps_per_h_1mb`) with a three-category classification rule:
  mmap-style step-wise / sbrk-style step-wise / continuous drift.
  The preprint conflated mmap and sbrk under "step-wise". This is
  the paper-worthy mechanism refinement.
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

## TL;DR (operational, 2026-05-20)

n=3 campaign on day 4.5 of ~9.5. Round 1 (r01 of all 6 cells) complete.
Round 2 batch 1 (e1, e2, e3 r02) complete. Round 2 batch 2 (a1, a2, e3b
r02) currently running on gpu0/1/2, ~30h to go. Round 3 follows.

Two findings are already locked in on n=3 data (will appear in the
camera-ready regardless of r03 outcome):

- **e3 drop rate at saturated load.** PyTorch + HF naive at 0.174 rps
  drops ~15.6-15.7% of offered requests, confirmed on n=2 (e3_r01 and
  e3_r02). e3b at 0.050 rps drops <2%. The naive baseline at saturated
  load exhibits a capacity ceiling that the production-grade engines
  do not. Reportable as Section IV.D rate-sensitivity ablation.
- **Pipeline hardened and validated.** Paper pipeline is end-to-end
  in-repo: `aging_trends.py` with Theil-Sen on real time axis,
  `fdr_aggregate.py` for BH-FDR at q=0.10, `validation_check.py` as
  per-run sanity gate, `stepness.py` for the three-metric panel.
  Decision rule: trend significant iff MK Hamed-Rao p<0.01 AND
  Theil-Sen 95% CI excludes zero AND bh_reject=True.

Open headline questions for the paper (within-n=3, not vs preprint):

1. Per-cell **between-replica variance** of the RSS slope. Within
   ~30% across r01/r02/r03 means the cell's aging signature is
   reproducible.
2. Per-cell **step-wise classification** (mmap / sbrk / drift) on n=3,
   and whether the assignment is stable across replicas.
3. e3 drop rate **mechanism**: time-clustered vs uniform, correlated
   with GPU 2 VRAM/util spikes or not, behavior stable across r01/r02.

---

## Paper

- Venue: WoSAR 2026 (ISSRE workshop on Software Aging and Rejuvenation)
- Deadline: 30 June 2026
- Authors: Domenico Cotroneo (UniNa Federico II / UNCC), Bojan Cukic (UNCC)
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

## Status snapshot (2026-05-20 afternoon ET)

```
Campaign launched: 2026-05-16T11:12 UTC
state.summary    : completed=9, running=3, failed=0
```

Completed: all r01 of the 6 cells + r02 of e1/e2/e3. All MK-significant
at p<0.01, all PASS on validation_check.

**Paper-grade RSS slopes** (aging_trends + fdr_aggregate, all
significant=True per the decision rule MK p<0.01 AND CI excludes 0
AND bh_reject=True):

| cell | r01 slope | r01 CI [lo, hi]       | r02 slope  | r02 CI [lo, hi]        | CI relation        |
|------|-----------|------------------------|-------------|--------------------------|--------------------|
| e1   | 6.96 KB/h | [1.5, 22.3] KB/h       | 1.76 KB/h   | [0.74, 23.8] KB/h        | overlap massive    |
| e2   | 21.7 KB/h | [12.9, 31.6] KB/h      | 161.4 KB/h  | [40.3, 231.4] KB/h       | DISJOINT (+8.7 KB/h gap) |
| e3   | 11.0 KB/h | [4.9, 20.1] KB/h       | 31.1 KB/h   | [21.4, 88.0] KB/h        | barely disjoint (+1.3 KB/h gap) |

**Reading.** e1 r01 and r02 are CI-compatible; the apparent "17x
smaller" from validation_check was a point-estimate artifact, the
two replicas are reproducible within CI. e2 r02 slope is genuinely
larger than r01 (CIs disjoint by 8.7 KB/h, ~7x point ratio). e3 r02
slope is just-disjoint from r01 (~3x point ratio, gap 1.3 KB/h).

**Pipeline note (calibration gotcha).** validation_check.py and
aging_trends.py give the SAME point estimate on r02 (smooth-ish runs)
but DIFFER by ~5x on r01 (step-heavy runs). validation_check operates
on raw 5s samples; aging_trends.py downsamples to 60s windows before
Theil-Sen. On step-heavy series the two methods are not equivalent.
For the paper, only aging_trends + fdr_aggregate is paper-grade.
validation_check remains a per-run sanity gate (PASS/FAIL on trend
direction), not a slope source.

**Additional paper-grade finding on e3_r02.** proc.vms_bytes shows
significant growth at +52.2 MB/h (CI [20.9 KB, 77.3 MB]/h, MK p=7.7e-08,
bh_reject=True). RSS grows at only +31 KB/h on the same run. The
VMS-to-RSS ratio is ~1700x. This is a third pattern alongside the
mmap-lock-step and sbrk-RSS-only seen in the preprint, and the
current three-class stepness panel does NOT capture it (see Step-wise
mechanism panel section below). On e3_r01 the same indicator was
NOT significant (slope=0, p=0.097), so the VMS-only growth emerged
specifically in r02. r03 will settle whether it is a stable property
of the cell or a r02-specific event.

**Validation_check slopes (for the older runs, sanity-gate values only,
not paper-grade):**
- e1_r01: +0.035 MB/h, drop trivial
- e2_r01: +0.109 MB/h, drop trivial
- e3_r01: +0.058 MB/h, drop 15.7%
- a1_r01: +0.064 MB/h, drop trivial
- a2_r01: +0.217 MB/h, drop trivial
- e3b_r01: +0.201 MB/h, drop trivial
- e1_r02: +0.002 MB/h, drop trivial
- e2_r02: +0.161 MB/h, drop trivial
- e3_r02: +0.031 MB/h, drop 15.6%

Note that the validation_check r01 numbers are ~5x larger than the
paper-grade aging_trends numbers on the same data (step-event
sensitivity, see "Pipeline note" above).

Running (r02 batch 2, started ~19 May 11:14 UTC):
- a1_r02 on gpu0
- a2_r02 on gpu1
- e3b_r02 on gpu2

Expected r02 batch 2 end: ~21 May UTC.

Pending:
- All r03 batch (e1, a1, e2, a2, e3, e3b).
- Sanity run (e2 on gpu0, 6h).

Notable n=3 findings already locked (n>=2):
- e3 drop rate at saturated load: 15.7% (r01) and 15.6% (r02). The
  PyTorch naive baseline at 0.174 rps hits a capacity ceiling; e3b at
  0.050 rps drops <2% on the same engine and same GPU.

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

---

## Open TODOs

In order of priority for the next session:

1. **DONE 2026-05-20: paper-grade r01/r02 slopes for e1, e2, e3.**
   See Status snapshot for the per-cell CI table and reading. Headlines:
   - e1: r01 and r02 CIs overlap massively. Reproducible within CI.
   - e2: r01 and r02 CIs disjoint, r02 ~7x higher than r01.
   - e3: r01 and r02 CIs barely disjoint, r02 ~3x higher.
   - e3_r02 additionally shows paper-grade VMS growth at +52 MB/h
     (a fourth, "VAS-only" stepness class).

   Also DONE: stepness panel on e1/e2/e3 r02. E2 confirmed
   mmap-style on n=3. E1 confirmed continuous drift on n=3. E3
   needs a fourth class for VMS-only growth (see Open Q #2).

   ```bash
   # Reusable, for r01 + r02 of any three cells:
   for cell in e1 e2 e3; do
     python3 analysis/aging_trends.py \
       --run-dir ~/wosar/runs/wosar2026_${cell}_r01 --csv \
       > /tmp/${cell}_r01_trends.csv
     python3 analysis/aging_trends.py \
       --run-dir ~/wosar/runs/wosar2026_${cell}_r02 --csv \
       > /tmp/${cell}_r02_trends.csv
   done
   python3 analysis/fdr_aggregate.py \
     --trends-csv /tmp/e1_r01_trends.csv \
     --trends-csv /tmp/e2_r01_trends.csv \
     --trends-csv /tmp/e3_r01_trends.csv \
     --csv > /tmp/fdr_r01.csv
   python3 analysis/fdr_aggregate.py \
     --trends-csv /tmp/e1_r02_trends.csv \
     --trends-csv /tmp/e2_r02_trends.csv \
     --trends-csv /tmp/e3_r02_trends.csv \
     --csv > /tmp/fdr_r02.csv
   ```

2. **Wait for r02 batch 2 end (~21 May UTC).** Then run the same
   analysis on `a1_r02`, `a2_r02`, `e3b_r02`. After this, all 12 r02
   runs are available and the n=2 within-cell variance can be
   estimated per cell.

3. **r03 batch.** Continues per the campaign plan. Expected to end
   ~24-25 May UTC. Once r03 is in, full n=3 analysis: median slope per
   cell, between-run CI, BH-FDR across the joint family.

4. **Paper writing on n=3 data.** The four content elements:
   (a) Campaign description, hardware, workload, stress regime
       (Section IV.A) on n=3.
   (b) Client-side stationarity check (Section IV.B) on n=3.
       Latency, TTFT, throughput, drop rate over the 36h window.
   (c) Process-side memory aging (Section IV.C) on n=3 with the
       three-metric step-wise panel: per-cell RSS slope with
       BH-FDR-controlled significance, and (corr, K_trim, steps/h)
       classification (mmap-style / sbrk-style / continuous drift).
   (d) Low-load ablation (E3b) and 2x2 factorial (Section IV.D-E)
       on n=3.
   Threats-to-Validity (Section V) declares the parallel topology as
   a feature of the design (realistic multi-tenant deployment), the
   confounded factorial (vLLM version drift across cells), single
   hardware, single model.

5. **Stepness analysis sequencing.**
   - Run `stepness.py --warmup-s 3600` on all completed n=3 runs
     (r01 + r02 as they land). Record per-cell (corr, K_trim,
     steps/h) per replica.
   - After r03, compute between-run CI for each metric (3 replicates
     per cell). Robust class assignment requires the metric to be
     stable across replicas.
   - Decide the headline mechanism claim per cell from the n=3
     majority-class assignment.

6. **e3 drop rate analysis.** Both e3_r01 and e3_r02 show ~15.6-15.7%
   dropped requests at the saturated target rate. Investigate:
   - Are drops time-clustered (bursty) or uniformly distributed?
   - Do they correlate with GPU 2 VRAM/util spikes or with the
     process scheduler queue depth?
   - Is the drop pattern stable across r01/r02 in time, or only in
     aggregate fraction?
   The finding is reportable as a property of the PyTorch naive
   baseline at saturated load in the n=3 paper (Section IV.D rate
   ablation).

7. **Long-tail TODO (post-campaign).**
   - Update `docs/experimental_protocol.md` to reflect the actual
     executed protocol (n=3, 36h, parallel topology, Qwen2.5-7B).
   - Fix `run_monitors.py` manifest collision (currently a workaround).
   - Restore Docker data-root on /home per ADR-002.
   - Complete refactor of `analysis/aging_trends.py` onto
     `analysis/aging_io.py`. Warmup discard, --csv mode, process_alive
     parsing already done in the 2026-05-19 evening hardening commit;
     remaining: `load_csvs` and `proc_prefix` discovery still local.
     Code-quality debt, not correctness.

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
  stepness.py, aging_io.py}` (paper pipeline)
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
# Adds q_value and bh_reject columns. Decision rule for paper:
# significant trend iff mk_p<0.01 AND slope_ci excludes 0 AND bh_reject==True.

# Stepness panel (corr, K_trim, steps/h) per run, on campaign data
python3 analysis/stepness.py --run-dir ~/wosar/runs/wosar2026_<cell>_r<NN> --warmup-s 3600

# Or all campaign runs in one shot
python3 analysis/stepness.py --logs-root ~/wosar/runs --warmup-s 3600

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
- **Multi-test correction**: Benjamini-Hochberg FDR at q = 0.10
  across the joint family of (run_id, indicator) tests.
  Implemented in `analysis/fdr_aggregate.py`, which consumes
  `aging_trends.py --csv` output and adds `q_value` and `bh_reject`
  columns.
- **Decision rule**: a trend is declared significant when ALL of:
  (a) MK Hamed-Rao p < 0.01, (b) Theil-Sen 95% CI excludes zero,
  (c) bh_reject is True. The first two come from `aging_trends.py`,
  the third from `fdr_aggregate.py`.
- **Stepness panel**: `corr` (RSS-VMS lag-0), `K_trim` (winsorized
  excess kurtosis), `steps_per_h_1mb` (count of ΔRSS > 1 MB per
  hour). Implemented in `analysis/stepness.py`. Used to classify
  cells as mmap-style / sbrk-style / continuous drift.
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

**Three-category classification rule.** Apply jointly:

| pattern              | corr   | K_trim | mechanism interpretation                                                                  |
|----------------------|--------|--------|-------------------------------------------------------------------------------------------|
| mmap-style step-wise | > 0.8  | > 10   | RSS and VMS step together at discrete events → kernel-mapped blocks never returned        |
| sbrk-style step-wise | < 0.5  | > 10   | RSS steps without paired VMS steps → glibc heap-arena extension, no kernel mmap involved  |
| continuous drift     | < 0.5  | < 5    | smooth small-grain accumulation, no big steps                                             |
| (border)             | mixed  | mixed  | needs n=3 confirmation                                                                    |

**Why this is a paper-worthy refinement.** The preprint Section IV.E
mentions both mmap and sbrk-extended heap as alternative hypotheses
but does not distinguish them. The (corr, K_trim) pair separates them
for the first time and is computable from the existing proc CSVs with
no new instrumentation. The headline mechanism claim of the n=3 paper
is the per-cell class assignment on n=3 data.

**Status of stepness metric panel:**
- Implemented in `analysis/stepness.py`.
- Validated on n=1 pilot CSVs (E1 = continuous drift, E2 =
  mmap-style step-wise, A1 = sbrk-style step-wise; A2, E3, E3b border).
- **Confirmed on n=3 r02 for E1 and E2** (2026-05-20): E2 mmap-style
  (corr=0.83, K_trim=648.6) ✓; E1 continuous drift (corr=0.31,
  top1%_step=0.0001 MB) ✓ (K_trim=NaN is a script edge case to fix
  for low-step series).
- **Fourth class needed: "VAS-only growth".** e3_r02 is classified
  by the current rule as continuous drift (corr=0.24, K_trim_dRSS=1.1)
  because the panel looks at dRSS only. But e3_r02 also has paper-grade
  significant proc.vms_bytes growth at +52 MB/h with no matching RSS
  growth. Pattern: address space reserved (mmap) but never paged in
  → not visible in RSS. Plausible mechanism: PyTorch's CUDA caching
  allocator reserves host-side VAS for device-side mappings without
  touching CPU memory. To capture this class, stepness.py needs an
  additional metric K_trim_dVMS, and the classification rule needs
  a fourth row:

  | pattern                                | condition                                                     | mechanism |
  |----------------------------------------|----------------------------------------------------------------|-----------|
  | border (VMS_missing)                   | `VMS_missing` in notes (cell breakage, monitor crash)          | highest-priority short-circuit: cannot classify on (corr, K_trim_dRSS, K_trim_dVMS) without a VMS axis; returning drift would silently swallow missing data. |
  | continuous drift (low-step fallback)   | both axes in low-step operational fallback (steps/h < 0.01)    | no significant step events on either axis; corr is noise correlation, not mechanism. Priority short-circuit before the metric-based rule (but yields to `VMS_missing`). |
  | mmap-style step-wise                   | corr > 0.8 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10           | RSS+VMS lock-step, kernel-mapped blocks never returned |
  | sbrk-style step-wise                   | corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS < 5            | RSS heap-arena extends without kernel mmap |
  | VAS-only step-wise                     | corr < 0.5 AND K_trim_dRSS < 5  AND K_trim_dVMS > 10           | VMS-only jumps, address space reserved without paging-in |
  | uncorrelated step-wise                 | corr < 0.5 AND K_trim_dRSS > 10 AND K_trim_dVMS > 10           | RSS and VMS both jump but desynchronized; heap-arena + mmap operating independently |
  | continuous drift                       | corr < 0.5 AND K_trim_dRSS < 5  AND K_trim_dVMS < 5            | smooth small-grain accumulation everywhere |
  | (border)                               | any other combination                                          | mid-corr (0.5-0.8) with significant step events, or mixed K_trim; needs replica confirmation |

  The five-class taxonomy with priority short-circuits was committed to
  `analysis/stepness.py` and `analysis/README.md` across four commits
  on 2026-05-20 / 2026-05-21:
  - first commit: K_trim_dVMS metric added, K_trim=NaN math fallback,
    five-class rule.
  - fix n.1: low-step fallback made operational-driven (`steps/h < 0.01`)
    regardless of K_trim numeric value.
  - fix n.2 (commit `1c84e9e`): classify_stepness short-circuits to
    "continuous drift" when both axes are in low-step fallback, before
    the corr-based rule. Required because corr in the grey zone 0.5-0.8
    on a low-step run is correlation of micro-noise, not of mechanism.
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

---

## Open questions for the next session

All within-n=3. No comparison with the preprint.

1. **Between-replica variance per cell. CI-AWARE READING ON n=2
   (e1/e2/e3 only; a1/a2/e3b r02 still running).**
   - **e1**: r01 CI [1.5, 22.3] KB/h and r02 CI [0.74, 23.8] KB/h
     overlap completely. Reproducible within CI; the apparent point
     ratio of 0.25 is consistent with Theil-Sen sample variance on
     low-step data, not a real difference.
   - **e2**: r01 CI [12.9, 31.6] KB/h and r02 CI [40.3, 231.4] KB/h
     are disjoint by 8.7 KB/h. r02 slope is genuinely higher than
     r01, with a ~7x ratio. Real between-run effect on this cell.
   - **e3**: r01 CI [4.9, 20.1] KB/h and r02 CI [21.4, 88.0] KB/h
     are disjoint by 1.3 KB/h. Borderline; ~3x point ratio.
   Question for r03: does e2 (and e3) regress toward r01, or does
   r02 confirm a real drift in the slope estimate? Plausible causes
   if real: cumulative ambient state on the host (file cache, log
   growth, neighboring runs' residual effects). The host-side
   `system.mem_used_bytes` itself dropped 3x between r01 (41.5 MB/h)
   and r02 (14 MB/h), confirming the host environment was different.

2. **Stepness class assignment per cell on n=3. CONFIRMED ON n=2
   for E1, E2, E3 (2026-05-21, after fix n.1 + n.2):**
   - **E2** r01 and r02: corr ~0.82/0.83, K_trim_dRSS 284/649,
     K_trim_dVMS 239/491 → **mmap-style step-wise** in BOTH replicas.
     Mechanism class stable on n=2. Confirms preprint expectation of
     E2 as canonical mmap-style.
   - **E3** r01 and r02: corr ~0.37/0.24, K_trim_dRSS 1.1/1.1,
     K_trim_dVMS 1.2/1.1 → **continuous drift** in BOTH replicas.
     Note: e3_r02 has paper-grade VMS slope of +52 MB/h aggregate
     (from aging_trends), but K_trim_dVMS=1.1 says VMS grows smooth
     not step-wise. So the +52 MB/h is continuous drift on VMS, not
     VAS-only step-wise. Class is "continuous drift with VAS slope
     asymmetry" — drift on both axes but with RSS slope 31 KB/h and
     VMS slope 52 MB/h (ratio 1700x). The asymmetry itself is
     paper-relevant (Section IV.C commentary).
   - **E1** r01 and r02: corr 0.64/0.31, both K_trim under low-step
     operational fallback → **continuous drift** in BOTH replicas
     (after fix n.2). Stable on n=2.
   - A1, A2, E3b: r02 still running, classification pending. A1 in
     particular is the candidate for "uncorrelated step-wise"
     based on the pilot retrospective.
   - r03 (all cells): pending the campaign continuing.
   The headline mechanism claim of the camera-ready is the per-cell
   class assignment on n=3 with majority rule across r01/r02/r03, on
   the five-class taxonomy + priority short-circuit: mmap-style /
   sbrk-style / VAS-only / uncorrelated / continuous drift.

   Pilot n=1 reclassification under the five-class rule + fix n.2
   (retrospective only, does not feed the paper headline):
   - E2 pilot → mmap-style step-wise
   - E1 pilot → VAS-only step-wise (was: drift in preprint, refined
     by the dVMS axis; K_trim_dVMS=789 indicates real step events
     on VMS, fallback does not fire because steps/h_dRSS=0.085 > 0.01)
   - A1 pilot → uncorrelated step-wise
   - E3, E3b pilot → continuous drift
   - **A2 pilot** → continuous drift (was border; reclassified after
     fix n.2 because both axes scatter the low-step fallback. Coherent
     with the absence of step events anywhere on A2 pilot.)
   The pilot reclassification is paper-relevant only as evidence that
   the new taxonomy is non-trivial: it recovers mechanism distinctions
   the preprint could not make with its three-class panel.

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
