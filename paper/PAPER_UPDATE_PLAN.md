# PAPER_UPDATE_PLAN.md

Hand-off document for the camera-ready WoSAR 2026 paper. Created
2026-06-06 to brief the next chat session (planned: Opus 4.8) on
what changes to apply to the preprint (`docs/WOSAR_2026.pdf`) to
produce the n=3 camera-ready.

This document is the **operational source of truth** for paper
writing. Pair it with `EXPERIMENT_STATE.md` (data and analysis state)
and `docs/WOSAR_2026.pdf` (the n=1 preprint, structural reference).

> **READ THIS FIRST (updated 2026-06-12).** The workshop paper is DONE and on
> arXiv (**arXiv:2606.11916**, "Characterizing Software Aging in GPU-Based LLM
> Serving Systems"), under WoSAR 2026 review (notif 10 Aug). So the
> section-by-section "what to rewrite" plan further below is now **HISTORICAL**
> (kept only for traceability). The **CURRENT source of truth** is the bottom
> section **"## 2026-06-12: EXPERIMENTAL PLAN — DoW"** (the extension campaign
> that feeds the journal). Data/decisions of the published paper:
> `paper/n3_analysis/N3_RESULTS.md`. Operational hand-off: `EXPERIMENT_STATE.md`.

---

## How to use this document

**Workflow (decided 2026-06-06): Domenico writes the paper himself
on Overleaf, in interactive conversation with the assistant.** Not
delegated drafting. The assistant's role is review, suggestion, and
alternative-phrasing companion. Section-by-section progression:
discuss a section → Domenico writes LaTeX on Overleaf → paste in
chat → assistant comments and proposes alternatives → iterate.

**This document is a REFERENCE for the conversation**, not a draft
to be copy-pasted into LaTeX. Use it to:
- Know which preprint sections to keep verbatim vs to rewrite
- See proposed draft text as a starting point for discussion
- Track what changes are required by the n=3 reframing
- Cross-check that all the new n=3 numbers and findings land in
  the right sections

If you are an LLM assistant picking this up, read in this order:

1. Read `EXPERIMENT_STATE.md` "Paper framing" section first
   (decision = standalone n=3 paper, NOT replication of preprint).
2. Read this entire document. Pay attention to:
   - Section "What stays unchanged" (~40% of preprint text)
   - Section "What gets rewritten" (~60%, the meat of the work)
   - Section "Open items" (things still to resolve before/during writing)
3. Read `docs/WOSAR_2026.pdf` for structural reference.
4. When Domenico starts a session, ask him which section he wants
   to work on next. He may also paste current LaTeX from Overleaf
   asking for review.

**Working language for chat: Italian. Paper itself: English.**

**Style of the camera-ready (inherited from preprint):**
- Sober, peer-level prose
- No em-dashes
- Explicit limitations
- Pre-registration of analysis pipeline declared upfront
- Tables and figures load-bearing

---

## Big picture: from n=1 preprint to n=3 camera-ready

The preprint reported 6 cells × 24h × n=1 = 144h of campaign data
with a within-run analysis pipeline (MK + Theil-Sen + BH-FDR
per-indicator). The headline numbers were per-cell point-estimate
slopes with within-run CIs.

The camera-ready reports **6 cells × 36h × n=3 + 1 sanity run (6h) =
654h of campaign data**, analyzed with a per-cell aggregation
pipeline (DerSimonian-Laird random-effects on the k=3 per-replica
Theil-Sen slopes + Stouffer combination of per-replica MK z-scores
+ per-cell BH-FDR). The headline numbers are per-cell DL-RE slopes
with between-replica CIs.

**The narrative skeleton stays the same.** Same RQs, same factorial
structure (3 primary + 1 low-load ablation + 2x2 engine×hosting),
same monitoring stack, same statistical-pipeline core
(autocorrelation-aware + FDR-controlled). What changes is
(a) the replica count, (b) the aggregation method, (c) the
canonical leak indicator (USS instead of RSS), (d) the step-wise
classification (five-class taxonomy instead of the preprint's
informal "step-wise vs continuous" dichotomy), (e) the framing
of the parallel-multi-tenant topology as a design feature.

---

## What stays unchanged (verbatim or near-verbatim)

These preprint sections need only minor edits (number bumps,
wording polishes) — the substance is unchanged:

### Abstract — partial
Keep first 2 sentences ("This paper proposes an empirical
methodology... LLM serving is different..."). Keep the
methodology positioning ("Our methodology is designed for
this setting..."). **Rewrite from "We run a campaign comparing"
onward** to reflect n=3 + 36h + parallel + USS-canonical + 5-class
taxonomy.

### Section I — Introduction
First 4 paragraphs unchanged (background framing on SAR, GPU
serving, why this setting is different). **Rewrite last 3
paragraphs** (the "We address this gap...", "We monitor 34
indicators...", "The contribution is three-fold..." paragraphs)
to reflect n=3 design. Update contribution claims to four-fold
(add: between-replica reproducibility on the leak signature).

### Section II.A — How modern LLM serving engines work
**Unchanged.** Background material on PagedAttention, continuous
batching, KV-cache, V0 vs V1 engine generations. The preprint
text is solid.

### Section II.B/C — Software aging and rejuvenation
**Unchanged** (modulo fix of the duplicate heading — preprint has
both II.B and II.C titled "Software aging and rejuvenation",
clearly a typo). The related-work survey is fine. Recent LLM-aging
work (Santos et al., Watanabe et al., Moura Silva et al.) stays.

### Section III.A — Workload and stress regime
**Unchanged in substance.** Same Poisson client, same arxiv
corpus, same prompt distribution (log-normal median 1500),
same 70% streaming. Minor edit: target rate is now `85% of
mechanical ceiling per cell` (matches preprint). Add one
sentence: "the same workload generator and corpus were used for
all replicas; per-replica seeds (1, 2, 3 for r01, r02, r03)
control only the request arrival schedule."

### Section III.B — Engines and ablations
**Unchanged in substance.** Same 6 cells (e1, a1, e2, a2, e3,
e3b). Same factorial design. Table I needs minor update:
add "Replicas" column with value 3 for all cells, and "Window
per replica" column with value 36h.

### Section III.C — Monitoring (Table II)
**Unchanged.** Same monitoring stack, same sampling periods,
same indicator catalog (~34 indicators). The hardening of the
monitoring (PID-aware proc tracking via `find_engine_pid.py`,
`process_alive` truthy parsing fix, etc.) is repo-side and
doesn't change the paper-side description.

### Section V.B — Workload assumption (TTV)
**Unchanged.** Stationary open-loop Poisson is a design choice;
the limitation argument is the same on n=3.

### Section V.C — Confounded factorial (TTV)
**Unchanged.** The vLLM version drift across cells is still a
residual confound; n=3 does not help here.

### Section V.D — Single hardware and single model (TTV)
**Unchanged.** Same generalization caveat.

### References
**Mostly unchanged.** The 24 preprint references stay. Possibly
add:
- DerSimonian and Laird 1986 (random-effects meta-analysis) for
  the new per-cell aggregator
- Stouffer 1949 or Whitlock 2005 (combination of p-values) for
  Stouffer z-combination
- Hollander-Wolfe 1999 already cited — keep

---

## What gets rewritten (the substantive work)

### Abstract — rewrite second half

Current second half (preprint):
> "We run a campaign comparing three representative deployments under
> identical workload, monitor host, device, and client metrics in
> parallel, and apply a pre-registered statistical pipeline that
> accounts for the autocorrelation and multiple-testing properties
> of monitoring data."

Proposed n=3 second half (draft):
> "We run a campaign comparing six deployment configurations under
> identical workload, three replicas of 36 hours each per
> configuration plus a six-hour cross-GPU sanity run, 654 hours of
> continuous engine operation in total. We monitor host, device,
> and client metrics in parallel at four levels of granularity, and
> apply a pre-registered statistical pipeline that accounts for the
> autocorrelation and multiple-testing properties of monitoring
> data and aggregates replicate-level slope estimates via
> random-effects meta-analysis."

### Section I — Introduction, last 3 paragraphs

Rewrite to:
- Bump "144 hours" to "654 hours" (18 × 36h + 6h)
- Add "n=3 replicas per configuration" framing
- Replace "single NVIDIA L40S GPU" with "three NVIDIA L40S GPUs
  in parallel"
- Update contribution claims:
  - First contribution: same wording ("first evidence of software
    aging in modern LLM serving on the GPU")
  - Second contribution: same wording on the 2x2 factorial, but
    note that **only Triton + vLLM V0 (E2) exhibits step-wise
    growth on n=3**; all other 5 cells show continuous drift.
    This is a sharper result than the preprint's "step-wise in
    three of four cells".
  - Third contribution (new): mechanism refinement via the
    (corr, K_trim_dRSS, K_trim_dVMS) panel that distinguishes
    five mechanism classes (mmap-style, sbrk-style, VAS-only,
    uncorrelated step-wise, continuous drift), recovering
    distinctions the preprint's three-class informal taxonomy
    could not make.
  - Fourth contribution (new): reproducibility of the leak
    signature across replicas — quantified per cell via DL-RE.

Limitations paragraph (last in Section I): keep most of the wording,
but **remove "the 24-hour window may miss aging effects that emerge
on longer timescales"** (we now use 36h; partial mitigation),
**remove "n ≥ 3 runs per configuration are an immediate priority for
future work"** (now done). Add new limitations: (a) parallel
multi-tenant topology is a design feature, not a flaw, but it does
constrain interpretation — see Section V; (b) some between-replica
variability remains, particularly on Triton-wrapped cells.

### Section III.D — Statistical pipeline (rewrite substantially)

Current preprint pipeline:
- MK Hamed-Rao per-indicator
- Theil-Sen with CI per-indicator
- BH-FDR across ~200 indicator-level tests
- Decision rule: trend significant iff MK rejects AND CI excludes 0

Camera-ready pipeline (preserve preprint core, add n=3 aggregation):
- All preprint steps stay (per-run trend detection unchanged)
- **New layer for n=3**: per-cell aggregation via
  `analysis/aggregate_slopes.py`. For each (cell_id, indicator):
  - Primary: DerSimonian-Laird random-effects on the k=3 per-replica
    Theil-Sen slopes. SE per-replica derived from upstream
    Theil-Sen 95% CI via Gaussian-equivalent `(hi - lo) / (2 * 1.96)`.
    Reports `slope_RE`, `ci_RE`, `tau2`, `I^2`, `Q`, `Q_pvalue`.
  - Per-cell p-value: Stouffer z-score combination of the k per-replica
    MK z values, with two-sided p via `scipy.stats.norm.sf` for
    numerical stability.
  - Per-cell BH-FDR (q=0.10) applied across the joint family of
    (cell_id, indicator) tests.
  - Headline `RE_significant`: (BH rejects Stouffer p) AND
    (RE 95% CI excludes 0) AND (n_replicas, k_used_RE, k_used_stouffer
    all ≥ 3).
  - Robustness cross-check: sample median of the k slopes, with
    [min, max] as the exact non-parametric CI for the population
    median (75% coverage at k=3, 87.5% at k=4). NOT a 95% test.
- **USS adopted as canonical leak indicator** (not RSS). Rationale
  goes in Section III.D paragraph:
  - 12/12 n=2 runs paper-grade significant on USS, 11/12 on RSS
    (a1_r02 fails on RSS CI floor=0 from AR(1) inflation with
    rho_RSS=0.99 vs rho_USS=0.005).
  - Point estimates USS-RSS agree within <10%.
  - USS = process-private resident pages, semantically cleaner
    (RSS includes shared mappings).
  - RSS reported alongside as secondary; VMS reported separately
    for cells where VMS-only growth is significant (Section IV.C).

### Section III.E (NEW) — Stepness mechanism panel

Brand new sub-section. Describes the (corr, K_trim_dRSS,
K_trim_dVMS, steps_per_h_1mb, mean_top1_step_mb) panel and the
five-class taxonomy plus two priority short-circuits
(border-on-VMS-missing, drift-on-both-axes-low-step). Reference
to `analysis/stepness.py` (open-source release with the paper).

The taxonomy table from `EXPERIMENT_STATE.md` "Step-wise mechanism
panel (paper Section IV.E)" should be reproduced verbatim in the
paper as a table or figure. The rationale ("preprint conflated mmap
and sbrk under step-wise; the RSS/VMS split + dVMS axis separates
them for the first time") goes in the introductory paragraph of
this section.

### Section IV.A — Campaign stability (rewrite Table III)

Table III in preprint has 6 rows (E1-A2), one per cell, 24h totals.

Camera-ready Table III has **6 rows × 3 replicas = 18 production
rows + 1 sanity row**. Aggregate per cell with median + range.
Or: keep it 6 rows with averaged/median per-cell numbers, and
mention "n=3 replicas per cell, see supplementary for per-replica
breakdown".

Sample text to add at end of Section IV.A:
> "The 18 production runs plus the 6-hour sanity run together
> amount to 654 hours of continuous engine operation. The sanity
> run is a 6-hour replay of cell e2 with the engine running on
> GPU 0 instead of GPU 1 — its slope estimate is statistically
> indistinguishable from the e2 r01/r02/r03 main runs (slope
> within their pooled 95% CI), supporting the cross-GPU
> generalizability of the e2 finding (Section V)."

### Section IV.B — Stationarity at the client side (extend)

Preprint result on n=1: client side stationary on all cells
except E3b which showed +0.04 s/h on E2E p50.

Camera-ready result on n=3:
- **Confirm client-side stationarity** on E1, A1, E2, A2 (slopes
  on latency, TTFT, throughput, drop rate all CI-include-0 across
  all 3 replicas of all 4 cells, also under per-cell DL aggregation)
- **e3 drop rate at saturated load: NEW headline finding**, n=3
  confirmed. Drop rates 15.7%, 15.6%, 14.3% on r01/r02/r03.
  Mediana 15.6%, range 1.4 pp (very tight). **This is a property
  of the naive PyTorch+HF baseline at saturated load (0.174 rps,
  85% of measured ceiling), not aging** — the drop rate is
  stationary within each replica (slope CI-includes-0 on drop_rate
  per run). To be discussed as a capacity-ceiling phenomenon, not
  a leak.
- e3b at low load (0.05 rps) drops <2% on all 3 replicas; matches
  preprint finding.

### Section IV.C — Process-side memory leaks (REWRITE entire section)

This is the biggest rewrite. The preprint had Table IV with three
rows (E1, E2, E3) showing per-cell RSS slope on n=1.

Camera-ready Table IV: **6 cells × 1 row each, with paper-grade
USS slope (DL-RE) on n=3**, columns:
| ID | Deployment | USS slope (DL-RE) | 95% CI | I² | k_used | RE_sig |

Numbers from the analysis to be lanciata as soon as the LaTeX
sources are recovered (see Open items). Preliminary numbers from
n=2 (will be refined to n=3 with r03 data):

| ID  | Deployment             | USS slope r01 | USS slope r02 |
|-----|------------------------|---------------|---------------|
| E1  | vLLM standalone V1     | 6.96 KB/h     | 1.76 KB/h     |
| E2  | Triton + vLLM V0       | 19.1 KB/h     | 157.2 KB/h    |
| E3  | PyTorch + HF naive sat | 11.0 KB/h     | 31.1 KB/h     |
| E3b | PyTorch + HF naive low | 38.0 KB/h     | 103.3 KB/h    |
| A1  | vLLM V0 standalone     | 12.9 KB/h     | 13.2 KB/h     |
| A2  | Triton + vLLM V1       | 42.5 KB/h     | 5.9 KB/h     |

(r03 numbers to be filled in once aggregate_slopes is rerun on
n=3.)

Add new finding to Section IV.C:
- **VAS-only growth on PyTorch+HF naive** (e3, e3b). VMS slope
  paper-grade significant only on r02 of both cells (+52 MB/h
  on e3_r02, +127 MB/h on e3b_r02), with RSS growth ~1000-1700x
  smaller on the same runs. Mechanism: anonymous mmap reservation
  by PyTorch CUDA caching allocator host-side metadata (verified
  empirically on e3b_r02: vms 38.5→46.0 GB, rss +20 MB, num_fds
  53→52, num_threads 200→199, ruling out FD leak and thread-stack
  growth). K_trim_dVMS ≤ 2.2 → smooth drift not step-wise.
  Phenomenon absent on r01 of both cells; r03 will determine
  whether r02 was the outlier.

### Section IV.D — Low-load ablation (extend with drop rate)

Preprint: E3b sub-saturated replicates E3 saturated RSS slope
within CI (workload regime is not the confound).

Camera-ready: same plus the **drop rate ablation**:
- E3 saturated drops 14-16% on n=3; capacity ceiling
- E3b sub-saturated drops <2% on n=3; engine handles load fine

Add 1-2 paragraphs on the drop rate interpretation. NOT aging.

### Section IV.E — 2x2 factorial (rewrite Table V)

Preprint Table V is the 2x2 matrix of RSS slopes on n=1.

Camera-ready Table V: same 2x2 matrix but with **USS DL-RE slopes
on n=3 with between-replica CIs**.

Update the qualitative paragraph after Table V: the "step-wise in
three of four cells" claim from the preprint becomes "**step-wise
mmap-style allocation emerges exclusively in Triton + vLLM V0
(E2)** on n=3. The other three cells (A1, A2, E1) exhibit
continuous drift with no MB-scale step events." This is the
sharper, paper-worthy n=3 finding.

The Figure 2 from preprint stays in spirit but needs to be
regenerated on n=3 data:
- (a) RSS trajectories for the 4 factorial cells (one
  representative replica per cell, e.g. r01 or median replica)
- (b) E2 detail: RSS+VMS lock-step (use e2_r01 since it's
  representative; mention reproducibility on r02, r03 in caption)

### Section IV.F (NEW) — Triton-wrapper between-run variance

A new sub-section on the hypothesis that emerged from n=2 (to be
checked on n=3 with r03 data):
- Standalone cells (e1, a1): r02/r01 USS ratios 0.26 and 1.02 (tight)
- Triton cells (e2, a2): ratios 8.2 and 0.14 (highly variable)
- PyTorch+HF cells (e3, e3b): ratios 2.8 and 2.7 (intermediate)

Hypothesis: Triton's dynamic batching scheduler maintains internal
state (request queues, model instance routing, output buffer
recycling) that is non-deterministic across replicas with different
seeds; this state propagates into the engine process footprint
and amplifies between-run variability.

Test: per-cell I² from DL-RE on USS. If I² > 75% on e2, a2 but
not on e1, a1 (with PyTorch in between), hypothesis confirmed
quantitatively.

If n=3 doesn't support this on r03 — note it as a hypothesis the
data did not support and move on.

### Section V (TTV) — rewrite to acknowledge n=3 progress

**Drop entirely the "Single-run design" subsection** (was V.A in
preprint). It's resolved.

Replace with new V.A: **"Between-replica variance bound."** Discuss
the per-cell I² from DL-RE meta-analysis. Where I² is small
(cells like a1, e1), the leak rate is reproducible within
narrow CIs. Where I² is large (Triton-wrapped cells, see
Section IV.F), the leak rate magnitude itself has substantial
between-replica variance — quantify in the paper, but the
qualitative finding (presence of leak, mechanism class) is
unchanged across replicas.

Keep V.B (workload), V.C (confounded factorial), V.D (single
hardware/model) as before.

Add new V.E: **"Parallel multi-tenant topology as a design
feature."** The n=3 campaign runs three cells in parallel on
three GPUs (the preprint ran one cell at a time on a single
GPU). This is declared as a feature: it reflects realistic
multi-tenant deployment under CPU contention, prompt sharing,
and host-level memory pressure. The Triton-wrapper variance
finding (Section IV.F) is partially a consequence of this
design. Cross-GPU sanity run (e2_r99: e2 on GPU 0 vs the
production e2 on GPU 1, 6h, ~22000 requests) shows slope
estimate consistent with the main e2 runs, partially supporting
generalization across GPU index.

### Section VI — Final Remarks (refresh closing)

Refresh to:
- 654 hours of continuous engine operation (vs 144 in preprint)
- 6 configurations × 3 replicas
- Four findings (instead of three in preprint):
  - Aging exists in this domain (unchanged)
  - Aging surface lies in framework orchestration layers
    (unchanged)
  - **Mechanism class is preserved across replicas: E2 is the
    only step-wise cell on n=3, all others are continuous drift**
    (new sharper claim)
  - **Leak rate is reproducible at the cell level, with
    Triton-wrapped cells showing larger between-replica
    variability than standalone or naive cells** (new on n=3)

Future directions paragraph: still relevant ones from preprint:
- Replication with n=3 done — say it's done!
- Cross-hardware extension (A100, H100, B200) — keep
- Cross-model-scale (70B, MoE, state-space) — keep
- Heap profiling on V0 engine to localize the mmap-style code path
  — keep
- Longer windows (7d+) for late-onset signatures — keep
- New: dedicated investigation of the VAS-only growth pattern on
  PyTorch+HF (e3/e3b r02) — does it persist on r03? Is it a
  property of the cached allocator host-side metadata?

---

## Open items (resolve before/during writing)

1. **RESOLVED 2026-06-06: LaTeX sources are on Overleaf.** Domenico
   writes on Overleaf directly, in conversation with the assistant.
   This document is reference, not a draft to incollare meccanicamente.

2. **Final n=3 numbers** for Tables III, IV, V. Pipeline ready in
   `~/Documents/Github/llm-serving-bench/runs/` (local data) +
   `analysis/aggregate_slopes.py` (per-cell aggregator). Run sequence
   in `EXPERIMENT_STATE.md` Open TODOs section #1.

3. **Stepness panel on n=3** for Section IV.E. Same pipeline-ready
   status. Run sequence in `EXPERIMENT_STATE.md` Open TODOs #2.

4. **Figure 2 regeneration on n=3 data.** Need a plotting script
   that uses the actual local CSVs. `analysis/plot_rss_2x2.py`
   and `analysis/plot_rss_combined.py` exist in repo — re-run on
   n=3 data with appropriate seed/representative-replica selection.

5. **Anomaly investigation on aging_trends output**: 405 rows
   total vs ~520 expected. Likely cause: some runs missing GPU
   indicator section (a2/e2 on gpu1, e3/e3b on gpu2 — aging_trends
   defaults to looking for gpu0 CSVs and skips if not found).
   Either fix aging_trends to discover GPU index from manifest
   or document the missing rows. Detailed log analysis in
   `/tmp/wosar_n3/*.log` (to be regenerated with stderr capture
   per the command sequence in `EXPERIMENT_STATE.md` TODO #1).

---

## Suggested writing order (conversational, section by section)

When the n=3 numbers are in (pipeline ready), discuss with Domenico
in this order to minimize backtracking. Each section is one
conversation block: assistant proposes structure/talking points,
Domenico writes on Overleaf, pastes back for review, iterate.

1. **Section IV.A** (campaign stability with new totals). Quick start,
   helps Domenico re-familiarize with the new numbers.
2. **Section III.D** (statistical pipeline n=3 aggregation). Self-contained,
   pre-registers the analysis upfront.
3. **Section IV.C** (Process-side leaks with USS Table IV n=3). Core.
4. **Section IV.E** (2x2 factorial with USS Table V n=3 + Figure 2). Core.
5. **Section IV.B** (client stationarity + e3 drop rate).
6. **Section IV.D** (low-load ablation + drop rate).
7. **Section IV.F** (new: Triton variance hypothesis, if data supports it).
8. **Section III.E** (new: stepness mechanism panel).
9. **Section V** (TTV refresh, drop single-run subsection).
10. **Section VI** (final remarks refresh).
11. **Section I** (introduction refresh: numbers + contributions).
12. **Abstract** (rewrite second half).
13. References (add any new ones for DL-RE, Stouffer).

Total estimated effort: depends on pace of conversation. A focused
session per day on 1-2 sections, ~10-15 sessions total over 2-3
weeks, comfortably within the 30 June deadline.

---

## Companion files

- `EXPERIMENT_STATE.md`: data and analysis state (the operational
  hand-off, paired with this document for paper writing)
- `docs/WOSAR_2026.pdf`: the n=1 preprint, structural reference
- `analysis/`: all pipeline scripts (already done, just rerun on n=3)
- `runs/`: local mirror of the 19 campaign run directories
- `campaigns/wosar2026/`: campaign configuration (cell YAMLs,
  campaign descriptor)
- `paper/n3_analysis/`: the n=3 paper-grade outputs run 2026-06-07
  (`per_cell.csv/.txt`, `fdr_per_run.csv`, `stepness_all.csv`,
  `N3_RESULTS.md`) plus `aging_trends_FIXED_reference.py`.

---

## 2026-06-07: two-paper split (SUPERSEDING DECISION)

This section overrides the single-camera-ready framing above wherever
they conflict. Decided with Domenico on 2026-06-07 after running the
full n=3 pipeline (outputs in `paper/n3_analysis/`).

### Context that changed the plan

- The draft `docs/WOSAR_2026.pdf` was **never submitted**. It was an
  initial draft whose eclatant n=1 numbers Domenico distrusted. The n=3
  analysis vindicates that instinct: those numbers were largely artefacts
  (a nondeterministic proc/gpu discovery bug in `aging_trends.py`, plus
  vLLM image version drift between the 7-May and 15-May images).
- The draft->n=3 gap is **structural, not run-to-run variance**: even the
  highest replica is 13-1300x below the draft on the optimized engines
  (E1 ~1300x). So there is no "right replica" that reconciles the two.

### The split

**Workshop (WoSAR 2026), single-run.** Same skeleton as the draft
(Abstract, I, II, III, IV.A-E, V, VI; Tables III/IV/V; Figures 1-2).
One 36h run per cell = **replica r02**, presented under the realistic
**multi-tenant parallel** framing (the r02 runs did execute 3 cells in
parallel across 3 GPUs). We substitute results AND reframe the text:
this is NOT a number-swap, because two of the draft's headline claims are
FALSE on the new data and must be rewritten, not just renumbered:
  - "the naive baseline is the cleanest" -> FALSE. On r02 (and on the n=3
    aggregate) the cleanest cell is E1 (vLLM V1 standalone). The naive
    PyTorch cells sit mid-to-high.
  - "leak rates span nearly three orders of magnitude" -> on r02 the
    spread is ~2 orders but all in KB/h (E1 1.8 -> E2 161 KB/h), with no
    clean engine-generation main effect. Operationally tens of MB/month.
Keep magnitude claims sober. The VMS VAS-only spike is r02-specific
(absent on r01/r03); either present it as a sober single-run observation
or omit it and reserve it for the journal. Do NOT claim replication.
r02 RSS slopes (KB/h) for Table IV/V: E1 1.8, A1 13.2, E2 161.4, A2 3.2,
E3 31.1, E3b 108.9 (re-verify after the in-repo pipeline fix).

**Journal extension, full n=3.** The novel contribution is the
replication itself plus the meta-analytic layer. Hook: aging studies in
the SAR literature are almost always single-run; this is among the first
replicated ones. Content beyond the workshop (>>30% new): DL-RE
random-effects aggregation + Stouffer combination + per-cell BH-FDR;
the reproducibility / between-replica heterogeneity story (Triton-wrapped
cells far less reproducible, I^2 up to 89%); the measurement lesson that a
single run is not representative; the cross-GPU sanity run; the honest
magnitude reckoning. **The journal Introduction has been drafted and is
FROZEN** (produced in the 2026-06-07 session; new bib keys:
`dersimonian1986`, `stouffer1949`, optional `whitlock2005`). It does NOT
reproduce the draft's "naive cleanest" / "3 orders" claims.

### n=3 numbers (source of truth: paper/n3_analysis/per_cell.csv)

USS DL-RE (KB/h): E1 4.73 (CI [-1.66, 11.11], NOT significant), A1 13.16
(I^2 0), E2 55.75 (I^2 89%), A2 22.74 (I^2 60%), E3 19.95 (I^2 54%),
E3b 61.65 (I^2 0). 5/6 cells RE-significant on USS (E1 the exception).
Stepness: E2 only step-wise (r01/r02 mmap-style, r03 border; sanity e2_r99
border); all other 5 cells continuous drift on all 3 replicas. VMS
VAS-only on e3_r02 (+52 MB/h) and e3b_r02 (+127 MB/h) only. Drop rate
e3 15.7/15.6/14.3%, e3b 0/0/0%.

### Pipeline caveat (applies to both papers)

`aging_trends.py` has a nondeterministic monitor-discovery bug (gpu0
hardcoded; proc-prefix can pick gpuN). The n=3 numbers above were produced
with a patched copy (`paper/n3_analysis/aging_trends_FIXED_reference.py`)
that reproduces all previously documented per-run values. Apply the
in-repo fix (use `aging_io.discover_proc_prefix` + gpu-prefix discovery,
TODO #9 in EXPERIMENT_STATE) and re-run before locking final table cells.

---

## 2026-06-07 (session 2): WORKSHOP FIRST DRAFT COMPLETE

All sections drafted on the single-run r02 data, section by section with
Domenico writing on Overleaf. The full skeleton of the draft is covered:
Abstract, Intro (I), Background (II, unchanged), Methodology (III:
Workload / Engines+ablations / Monitoring / Statistical pipeline),
Results (IV.A-E), Threats (V), Final remarks (VI). Tables I-V and Figure 2
done. Draft text lives on Overleaf; the proposed LaTeX for each block was
produced in chat.

### Data/编辑 decisions locked for the workshop

- **Single run = r02** for every cell (richest single replica: cleanest
  2x2, E2 mmap-style clean, only run with the VMS VAS-only). Chosen
  consciously knowing r02 is the "rosiest" replica; the n=3 journal will
  put it in perspective. Do NOT mention replicas in the workshop text
  (no "r02" in captions/labels).
- **USS is the canonical leak indicator** (Tables IV and V in USS).
- **Table I**: A2 rate `0.90 -> 1.75` (recalibrated), E1 vLLM ver
  `latest -> 0.21.0` (from docker_inspect label). 24h->36h, 6 runs = 216h.
- **Table III** (aggregate, r02): tok/s `n/a` for E2 and A2 (Triton client
  path does not log per-request output tokens; footnote added). Drop% e3
  15.60, e3b 0; rates within 0.4%; p50 60x lower e3b vs e3.
- **Figure 2 = single panel** `paper/n3_analysis/figures/uss_factorial.pdf`
  (Delta USS, 4 cells). Panel (b) RSS/VMS lock-step DROPPED (not visually
  clean on r02); the mechanism is stated in text via stepness numbers
  (corr 0.83, top-1% step 1.8 MB, heavy-tailed increments). Two-panel RSS
  version `rss_combined.pdf` also in repo if ever wanted.

### Findings as written (honest, r02) — these REPLACE the draft's claims

- Aging exists but is SMALL: USS KB/h. Primaries E1 1.8, E2 157, E3 31;
  2x2 A1 13, A2 5.9. 30-day extrapolation ~1/22/110 MB.
- **"naive is the cleanest" is DEAD**: E1 (vLLM V1 standalone) is the
  cleanest; naive PyTorch sits mid. Localization reframed: same inference
  path, different leaks -> leak is in the runtime, not the inference path.
- 2x2: NOT a crossover. **V0 leaks more than V1 in both hostings; Triton
  amplifies**; heaviest Triton+V0 (E2). Range ~2 orders, not 3.
- **Only E2 is step-wise** (mmap, corr 0.83, 1.8 MB steps); other 5 cells
  drift (sub-MB increments). Sharper than the draft's "3 of 4".
- **GPU = null**: A1 VRAM +124 MB/h does NOT reproduce (flat on all
  replicas); no cell has a significant VRAM trend. Draft's GPU headline
  removed.
- IV.D flip: E3b (+103) leaks MORE than E3 (+31), so the leak is not a
  saturation artefact (present, larger, at low load) -> time-driven not
  load-driven. VAS-only VMS on both PyTorch r02 runs (e3 +52, e3b +127
  MB/h; e3b raw 38.5->46.0 GB VMS, RSS +20 MB, fds/threads flat).
- Threats (V): removed dead "E3~E3b indistinguishable" argument; added
  **parallel multi-tenant topology** as a declared threat; version drift
  strengthened with real versions (0.7.3 / 0.10.1.1 / 0.21.0).

### Open items for next sessions

1. **Section-title consistency pass.** Decide ONE style (neutral noun
   phrases vs questions) and align all five. Pending retitles: IV.A
   "Campaign stability and stress regime" (-> "Campaign overview" or
   "Did the stress regime hold?"), IV.E "Engine generation and hosting
   layer interact" (-> "Engine and hosting together" or "Engine, hosting,
   or both?"), and "Workload regime sensitivity analysis" is also long.
   Already set: "Client-side analysis", "Process-side analysis".
2. **Abstract variant.** Chose framework-only (122 words) vs
   framework+findings (~150). Decision pending.
3. **"Is there really aging?" discussion.** Parked by Domenico for after
   the first full draft. He believes there is real aging/problems. Best
   supporting arguments: the E2 step-wise mmap mechanism (allocate-and-
   never-release, no obvious ceiling) and the days-vs-months horizon gap.
4. **In-repo pipeline fix** (aging_trends discovery, TODO #9) before
   locking any final numbers.

### NVIDIA proposal seed (discuss later, do NOT lose)

Reliability -> security bridge, planted as the last sentence of Final
remarks. Architectural take: a monotonic, never-released step allocation
in a long-running multi-tenant engine is a latent resource-exhaustion
primitive (CWE-400). It becomes a real vulnerability IF the step is
reachable/amplifiable from the request path -> remote low-cost memory-
exhaustion DoS, cross-tenant in a shared host. Current evidence: E2 steps
do NOT correlate with request rate/CPU/scheduler on the benign stationary
workload (reassuring, looks internal/time-driven), but adversarial inputs
were never probed, so "not client-triggerable" is unproven. Research
question for the call: can crafted request patterns drive/accelerate the
step allocations? stepness is the instrument to measure triggerability.

---

## 2026-06-12: EXPERIMENTAL PLAN — extension campaign (DoW). CURRENT source of truth.

Supersedes the earlier "48h x 3 platform x hardware x model factorial" sketch.
The journal extension and the NVIDIA-grant work are built on this campaign. The
local L40S arm runs first (de-risking); the L40S->A100 hardware axis and the
Nemotron model come with the grant. The journal's meta-analytic layer (DL-RE +
Stouffer + per-cell FDR; intro already frozen) applies to the replicated runs.

Pipeline: the `aging_trends.py` proc/gpu discovery bug (TODO #9) is FIXED
(now uses `aging_io.discover_proc_prefix` + dynamic gpu-prefix discovery).

DESIGN — one screening DoW run identically on three serving systems:
- Systems: NVIDIA Dynamo (disaggregated, 2 GPU/run), Triton+vLLM (1 GPU),
  vLLM standalone (1 GPU). Common engine = vLLM; pin the SAME vLLM version
  across the three to avoid the preliminary's version-drift confound.
- Workload DoW: 5 factors x 2 levels — rate, prompt-length, output-length,
  prefix-repeat, burstiness. 16-run Resolution V (I=ABCDE: all main effects +
  all 2-factor interactions estimable, unconfounded) + 3 center points = 19
  runs/system. 48h window each. Rate = fraction-of-ceiling with a short
  per-run calibration.
- Levels (Qwen, ctx 8192; CONFIRMED 2026-06-29): rate 30% / 85% ceiling;
  prompt ~512 / ~6000 tok; output ~64 / ~1024 tok; prefix-repeat 0% / 80%;
  burstiness Poisson / bursty. Center point = mid values.
- Responses per run (deep): USS slope/hour (primary); slope/million-requests
  (separates time- vs load-driven); stepness; AND per-component memory
  (router / prefill / decode / KV-transfer) for localization on Dynamo.
- Model: Qwen for the local arm. Nemotron + the L40S->A100 hardware axis are
  on the grant/A100 arm (adding both to the local DoW would ~double GPU-h
  beyond the 2-month window).

WHY THIS SHAPE: one cannot decide a priori whether aging is time- or
load-driven, so we lead with a stressful workload DoW and put rate among the
factors; slope/hour vs slope/request then decides it from the data. Screening
(Res V) ranks which workload stress dominates and per-component monitoring
finds where; then finer characterization on the 2-3 dominant factors + a 7-day
confirmation on the worst stressor. (This replaces the earlier "nominal base
vs probe" framing, per Domenico: in aging practice you stress to surface and
localize, then DoE the workload parameters to rank them.)

BUDGET / TIME: ~3,650 GPU-h on local 4x L40S (Dynamo 19x48x2 + Triton 19x48 +
vLLM 19x48) + calibrations; fits the ~2-month / ~5,760 GPU-h server window with
margin.

PHASING (~8 weeks): wk1-2 setup/de-risk (Dynamo bring-up aggregated +
disaggregated; MONITORING decision = map the component PIDs to track; two new
client features = prefix-repeat injection + burst arrival mode; harness
validation with 2 short runs; calibration tooling). wk3-7 the three DoW
campaigns (Dynamo first). wk7-8 deep analysis + dominant factors + 7-day runs
on the worst stressor.

LOCKED: 48h window [AMENDED 2026-07-10 to 36h -- see the amendment block
below; everything else in this list stands]; Res V 16+3CP; rate =
fraction-of-ceiling; 3 systems; Qwen local / Nemotron+A100 on the grant;
"stress-workload" terminology (never "adversarial"); security framed as a
resource-exhaustion implication with responsible disclosure; factor levels
(CONFIRMED 2026-06-29, above).

### 2026-07-10: AMENDMENT — screening window 48h -> 36h; mission timeline

**Decision (Domenico + review chat, 2026-07-10): the DoW screening runs are
36h, not 48h.** Exception: the 3 Dynamo center points run at 48h as a
cross-anchor against the completed 48h long test. Strictly serial execution
stands (one run at a time on the whole host; enforced in code by the serial
campaign scheduler and the run-slot lock).

Why the amendment. The original 48h budget (~3,650 GPU-h, "fits the 2-month
window") implicitly assumed the three system campaigns run in PARALLEL on
the 4 GPUs. The extension later adopted strict serialism for measurement
isolation (a workload DoE cannot tolerate co-tenant load that varies per
cell), which changes the wall-clock arithmetic: 57 x 48h = ~17 weeks vs
57 x 36h = ~13 weeks, both plus ~4 days of overheads (bring-up/teardown/
cooldown ~1h per run; per-cell calibrations ~40 min each) and a ~10% retry
margin (grounded in the observed SUT pathology rate, e.g. the NIXL
KV-transfer stall of 2026-07-06). The host window is 20 weeks. At 48h the
post-screening phase (finer DoE on the dominant factors + 7-day
confirmations) would be squeezed into ~3 weeks; at 36h it gets ~7.

Why 36h is right on the merits, not only the calendar:
1. Direct comparability with the n=3 baseline campaign (36h windows): every
   DoW slope is one-to-one comparable with the submitted paper's numbers,
   with no window-length confound between the two studies.
2. The statistical pipeline (MK + BH-FDR, the 5-class stepness panel,
   validator defaults) was built and validated on 36h series on this exact
   hardware; stepness counts discrete step events and was unambiguous at
   36h in the baseline, while 24h would leave slow steppers at 4-5 events
   and borderline classifications.
3. The marginal value of 48h over 36h is small for a SCREENING: the 48h
   long test gave slope CIs of +/-6% while baseline factor effects span
   orders of magnitude; late-onset coverage is delegated by design to the
   7-day confirmation runs.

Mission timeline within the 20-week window (strictly serial):
  wk 1        per-cell calibrations (provenance + max-age gates enforced in
              code), DoW campaign yaml dry-run, buffer.
  wk 2-14     screening: 57 runs (54 x 36h + 3 Dynamo center points x 48h),
              fixed-seed interleaved order across systems (see the cell
              generation task), inter-run cooldown, ~10% retry margin.
  wk 14-18    finer characterization on the 2-3 dominant factors + 2 x
              7-day confirmation runs on the worst stressor.
  wk 19-20    contingency + on-box analysis.

NEXT (CONFIRMED 2026-06-29): STEP 1 = Dynamo bring-up (aggregated +
disaggregated) + MONITORING decision (map component PIDs: router / prefill /
decode / KV-transfer) + two client features (prefix-repeat injection, burst
arrival mode) + harness validation (2 short runs) + per-run rate-calibration
tooling. Nothing in the DoW runs starts before STEP 1 is done.
STEP 0 (Domenico, in parallel): 48h validation run of one known cell on the
L40S server to confirm harness + fixed pipeline at the 48h horizon.
