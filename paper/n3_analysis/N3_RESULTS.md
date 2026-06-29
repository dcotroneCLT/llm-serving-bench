# N=3 paper-grade results (generated 2026-06-06)

Source of truth for camera-ready Section IV. Produced by re-running the
in-repo pipeline on the 18 production runs (local mirror in `runs/`),
plus the e2_r99 sanity run for stepness.

Pipeline: `aging_trends.py` (per-run MK Hamed-Rao + Theil-Sen CI, 60s
downsample) -> `fdr_aggregate.py` (per-run BH-FDR) -> `aggregate_slopes.py`
(per-cell DerSimonian-Laird RE on k=3 Theil-Sen slopes + median-of-3
robustness; per-cell BH-FDR on Stouffer p; q=0.10, expected-replicas=3)
-> `stepness.py` (5-class mechanism panel).

Raw outputs in this directory: `per_cell.csv`, `per_cell.txt`,
`fdr_per_run.csv`, `stepness_all.csv`.

## IMPORTANT: a pipeline bug was found and worked around

`aging_trends.py` does NOT use `aging_io.discover_proc_prefix`. It has
its own local discovery (lines ~285 and ~293-297) with two defects:

1. GPU prefix hardcoded to `gpu0` -> GPU indicators silently dropped for
   all gpu1/gpu2 cells (e2, a2, e3, e3b).
2. The proc-prefix loop only excludes `("gpu0","system")` and iterates an
   UNSORTED glob. On gpu1/gpu2 cells it can pick `gpu1`/`gpu2` as the
   "process" monitor, so `proc.*` indicators (including USS, the canonical
   leak indicator) are computed on the wrong file or dropped. The outcome
   is filesystem-order-dependent, hence nondeterministic across machines.
   This is the real cause of the "405 vs ~520 rows" anomaly, and it can
   silently corrupt USS/RSS/VMS on the Triton and PyTorch cells.

These numbers were produced with a patched copy
(`aging_trends_FIXED_reference.py`) that (a) discovers the gpu prefix and
(b) excludes any `gpuN` + `system` from proc discovery via a sorted glob.
The patch reproduces the previously documented per-run values exactly
(e.g. a2_r01 USS 42.5 KB/h, e2_r01 USS 19.1 KB/h, e1_r01 USS 7.0 KB/h),
so the n=3 table below is trustworthy.

**To make this official/reproducible, apply the one-true-fix in the repo
via Claude Code**: replace the local discovery in `aging_trends.py` with
`aging_io.discover_proc_prefix(run_dir, manifest)` (already correct: reads
the manifest, excludes gpu*/system) plus an analogous gpu-prefix
discovery, then re-run. This closes long-tail TODO #9 in EXPERIMENT_STATE.
All 18 runs then yield the full 34-indicator catalog (612 rows total).

## Table IV source: per-cell USS leak rate, n=3 (DL-RE)

USS = canonical leak indicator. Slopes in KB/h.

| ID  | Deployment             | USS DL-RE | 95% CI (RE)      | I^2    | RE_sig | note |
|-----|------------------------|-----------|------------------|--------|--------|------|
| E1  | vLLM V1 standalone     |  4.73     | [-1.66, 11.11]   |  0%    | **NO** | CI includes 0 |
| A1  | vLLM V0 standalone     | 13.16     | [12.89, 13.43]   |  0%    | yes    | tightest, most reproducible |
| E2  | Triton + vLLM V0       | 55.75     | [11.02, 100.47]  | 89.2%  | yes    | huge between-replica spread |
| A2  | Triton + vLLM V1       | 22.74     | [3.18, 42.29]    | 60.2%  | yes    | high heterogeneity |
| E3  | PyTorch+HF saturated   | 19.95     | [4.79, 35.10]    | 53.6%  | yes    | |
| E3b | PyTorch+HF low load    | 61.65     | [27.33, 95.97]   |  0%    | yes    | |

RSS (secondary, KB/h): E1 4.73 (NS), A1 12.85, E2 64.08 (I^2 89.5%),
A2 17.61 (NS, I^2 78.8%), E3 19.93, E3b 64.93. USS and RSS point
estimates agree within ~10% except where heterogeneity dominates.

5/6 cells are RE-significant on USS. **E1 is NOT significant** (per-replica
slopes 7.0 / 1.8 / 5.0 KB/h; pooled CI crosses zero). A2 fails on RSS but
passes on USS (the USS-canonical argument again).

## VMS (VAS-only growth): r02-only, confirmed

VMS per-replica (MB/h): e3 = [0.0, 52.2, 0.0], e3b = [0.0, 126.7, 0.0].
The VAS-only growth appears only on r02 of both PyTorch+HF cells; the
per-cell DL-RE aggregate is NOT significant (1/3 replicas, high I^2). So
this stays a per-replica phenomenon, not a per-cell finding. Mechanism
(anonymous mmap by the CUDA caching allocator host-side) as documented.

## Stepness mechanism (n=3 + sanity)

| cell | r01 | r02 | r03 | sanity r99 |
|------|-----|-----|-----|------------|
| E2   | mmap-style | mmap-style | **border** | border |
| E1   | drift | drift | drift | - |
| A1   | drift | drift | drift | - |
| A2   | drift | drift | drift | - |
| E3   | drift | drift | drift | - |
| E3b  | drift | drift | drift | - |

E2 is the only cell with MB-scale step events (top 1% step 1.7-3.7 MB).
On r03 the corr drops to 0.55 (grey zone) so it classifies as "border"
not clean "mmap-style", but it still bears step events (top1% 2.8 MB) and
is NOT drift. The sanity run e2_r99 (e2 on gpu0) is also border with
top1% 3.7 MB, supporting cross-GPU presence of the step mechanism.

Headline (majority rule): **E2 step-wise on all 3 replicas (mmap-style
2/3, grey-zone 1/3); all other 5 cells continuous drift on all 3.** This
is the sharper-than-preprint claim, with the caveat that r03 is grey-zone
rather than textbook mmap.

## Client side

Drop rate (raw, over full run): e3 = 15.68 / 15.60 / 14.27 % (median
15.6%, range 1.4pp); e3b = 0.00 / 0.00 / 0.00 %. Drop rate is stationary
within each run (slope CI includes 0). Capacity-ceiling phenomenon of the
naive baseline at saturated load, not aging. All other client indicators
(latency, TTFT, throughput) stationary across all replicas.

GPU VRAM: stationary on e1, a1, a2; the gpu1/gpu2 cells (e2, e3, e3b)
report VRAM growth flags but these need scrutiny (shared-GPU multi-tenant
contention on gpu1/gpu2; the only clean preprint VRAM finding was A1 on
gpu0, which here is stationary). Do not promote GPU findings without a
per-run look; flagged for follow-up.

## The elephant: n=3 vs preprint Table IV/V

Preprint reported RSS slopes in MB/h; n=3 reports KB/h. Gap per cell
(preprint RSS -> n=3 RSS):

| cell | preprint | n=3      | factor    |
|------|----------|----------|-----------|
| E1   | 9.15 MB/h| 4.73 KB/h| ~1900x lower, now NON-significant |
| E2   | 2.04 MB/h| 64.1 KB/h| ~32x lower |
| E3   | 170 KB/h | 19.9 KB/h| ~8.5x lower |
| A1   | 530 KB/h | 12.9 KB/h| ~41x lower |
| A2   | 20 KB/h  | 17.6 KB/h| comparable |

Consequences for the camera-ready narrative:

- The preprint's central dramatic result ("the optimized vLLM standalone
  E1 is the heaviest leaker, 9.15 MB/h, ~6.6 GB over 30 days") **inverts**:
  at n=3, E1 shows NO statistically significant leak.
- The preprint's "leak rates span nearly three orders of magnitude"
  (20 KB/h to 9.15 MB/h) collapses to roughly one order at n=3
  (~5 to ~65 KB/h).
- Practical significance: worst-case n=3 leak ~44 MB / 30 days (E3b).
  Operationally negligible vs the preprint's GB/month framing.
- Direction of the gap is consistent with the documented pipeline
  corrections (validation_check ~5x inflation on step-heavy runs, USS vs
  RSS, the discovery bug above), but the MAGNITUDE for E1 (~1900x, and E1
  is continuous not step-heavy) is NOT explained by any single documented
  fix. Most likely also involves the vLLM "latest" version drift between
  the 7-May pilot image and the 15-May campaign image. To be understood
  before writing IV.C, and to be ready for a reviewer who cross-references
  the public preprint.

Open decision for Domenico: the camera-ready contribution likely has to
re-center from "aging is an operational threat" toward (a) the E2
step-wise mmap mechanism (reproducible, qualitative), (b) the methodology
(n=3 + DL-RE + corrected pipeline), and (c) an honest "leaks are real but
small in modern engines" message. The "standalone n=3" framing already
shields against having to reconcile preprint numbers, but the story's
weight shifts.
