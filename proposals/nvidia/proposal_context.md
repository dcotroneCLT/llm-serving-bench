# NVIDIA Academic Grant - Proposal Context and Narrative (working brief)

Created 2026-06-10. Purpose: capture the narrative and the experimental plan
for the NVIDIA Academic Grant proposal so the thread survives a lost chat.
Pairs with the template `nvidia_academic_grant_2026_template.docx` in this
folder.

## Status (end of session 2026-06-10)

DONE today: arXiv license guidance (use arXiv non-exclusive); confirmed WoSAR
is non-blind, submission 20 July, notif 10 Aug; locked the experimental plan +
run matrix + GPU-hours (below); built and filled the proposal `.docx`.

The `.docx` (`nvidia_academic_grant_2026_template.docx`) now has:
- **Project Title (locked):** "Does AI Infrastructure Age? Understanding
  Resource Exhaustion and Performance Degradation in NVIDIA LLM Serving"
- **Abstract (locked, ~157 words):** the plain-language version that defines
  aging in the first two sentences, then PI credibility (OS/JVM/edge AI +
  first GPU LLM serving characterization), then the NVIDIA extension plan.
- PI/collaborator: Domenico Cotroneo and Bojan Cukic, Professor, UNC Charlotte
  (all "University of Naples" references removed everywhere).

NEXT (tomorrow): fill **Methods** (DoE 2x2 + Dynamo + stress-workload probe)
and **Project Support Details** (drop in the GPU-hours matrix below). Working
rule from Domenico: a stated word count is an estimate, tolerance +/-10%,
unless he says "exact".

## The grant (facts)

- Program: NVIDIA Academic Grant Program. Submission deadline 30 June 2026,
  decisions ~September 2026, GPU access presumably from ~Oct 2026.
- Interest area we target: **AI Inference, Agents, and Systems Software**
  ("systems software for AI"). We are squarely in it: we study the systems
  software that serves LLMs, not model training.
- Requirement: incorporate pretrained models from ai.nvidia.com and/or make
  extensive use of NVIDIA software distributions. We satisfy this via
  **NVIDIA Triton** and its successor **NVIDIA Dynamo**, plus an NVIDIA
  **Nemotron** model as a served model.
- Resources (per CFP): up to 32,000 A100-80GB cloud hours, up to 8 concurrent
  GPUs, up to 32 TB storage. Physical hardware possible with justification.

## The narrative (the story to tell)

**1. PI credibility (the anchor).** Domenico Cotroneo is among the pioneers of
software aging research, with a track record across very different systems:
operating-system / Linux aging, JVM aging (measurement-based analysis of the
JVM), AI/ML aging (software aging in a real-time object detection system on an
edge server; aging-related bug prediction via graph-transformers), plus the
SAR survey/roadmap. The message to NVIDIA: this team knows how to design and
run rigorous long-window aging studies, and has done it repeatedly on new
classes of systems before they were mainstream. (Cite the relevant prior work
and the WoSAR preliminary paper.)

**2. The preliminary study (refer to the arXiv preprint).** "Characterizing
Software Aging in GPU-Based LLM Serving Systems" is, to our knowledge, the
first to bring SAR methodology to GPU LLM serving. On a single hardware class,
a single model, and a single 36-hour run per configuration, it already shows:
a small but statistically significant process-private memory leak in every
deployment; a leak rate that is a property of the full deployment (the
Triton + legacy-V0 combination is the heaviest, the optimized standalone engine
the cleanest); and a distinctive step-wise, allocate-and-never-release
signature unique to Triton + V0. The results are promising and motivate a
deeper, properly scaled study.

**3. Why extend, and the security twist.** Two reasons. (a) Generality: the
preliminary study is single-hardware, single-model, single-run; we need
replication and a range of NVIDIA hardware/models to know how the phenomenon
behaves. (b) Reliability -> security: the step-wise, never-released allocation
is a latent resource-exhaustion primitive. IF it is reachable or amplifiable
from the request path, it would be a new DoS-class vulnerability for
multi-tenant inference (uncontrolled resource consumption, CWE-400), where one
tenant's request stream could exhaust shared host memory and degrade
co-located tenants. IMPORTANT framing: this is a HYPOTHESIS to test, not a
proven finding. On our benign, stationary workload the step events did NOT
correlate with request rate, CPU, or scheduler pressure (reassuring), but we
never probed adversarial inputs. The proposed work is exactly to answer
"can crafted request patterns drive or accelerate the step allocations?".

## The proposed study (experimental plan)

- **Hardware.** Extend from the UNCC local GPUs [CONFIRM: the preliminary paper
  used L40S; Domenico mentioned "A40" - clarify which UNCC server] to
  **NVIDIA A100 80GB** via the grant cloud, and ideally **H100** (and Blackwell
  if available) to test hardware generality. A100 is the grant's assumed unit.
- **Serving platforms.** NVIDIA **Triton** (+ vLLM) AND **NVIDIA Dynamo**, the
  new open-source distributed inference framework and successor to Triton
  (disaggregated serving = a new orchestration layer = a new aging surface).
  Standalone vLLM as the reference. This is the strongest NVIDIA-software hook:
  we study the reliability and security of NVIDIA's newest serving stack.
- **Models.** NVIDIA **Nemotron** (from ai.nvidia.com) and Qwen2.5-7B for
  continuity; ideally a larger scale (e.g., 70B / MoE) to test model-scale
  generality.
- **Runs.** **48-hour runs, 3 replicas per configuration** (this fixes the
  single-run threat-to-validity of the preliminary study). A few extended
  7-day runs for late-onset effects.
- **Analysis.** The existing in-repo pipeline: Mann-Kendall (Hamed-Rao) +
  Theil-Sen with autocorrelation correction + BH-FDR, DerSimonian-Laird
  random-effects aggregation across replicas, and the increment-distribution
  "stepness" measure (kurtosis / concentration) to classify a leak's shape.
  Plus **Nsight** heap/allocator profiling to localize the step allocations,
  and an **adversarial-workload probe** (vary request size, repetition,
  burstiness) to test triggerability of the step allocations.
- **Outcomes.** A journal article extending the WoSAR preliminary paper; a
  responsible-disclosure report to the vLLM/Triton/Dynamo communities if the
  security probe is positive; open-source release of the measurement and
  analysis pipeline and a curated trace dataset.

## GPU-hours sketch (to refine for Project Support Details)

Key justification: aging is **time-driven**, so GPU-hours equal wall-clock
occupancy x concurrency, not high-FLOP utilization. Multi-day runs must be
**uninterrupted** (an ephemeral preemption destroys a run), which argues for
reserved / long-lived instances.

Rough arithmetic to refine so it sums within 32,000 A100-h:
`[platforms ~3: Triton, Dynamo, standalone] x [models ~2] x [configs/cells]
x [3 replicas] x [48 h] + [a few 7-day runs]`. Provide the explicit sum.

## Open items / to confirm

- UNCC hardware: A40 or L40S? (the preliminary paper says L40S).
- Lock the platform list: Triton + Dynamo (confirmed name) + standalone vLLM.
- PI eligibility: full-time faculty at an accredited institution (UNCC). Use
  the academic affiliation, not a company address.
- Keep the security claim a HYPOTHESIS ("we propose to test whether..."), never
  asserted as a known vulnerability.
- arXiv preprint: arXiv:2606.11916 (cs.SE, cs.AI), "Characterizing Software Aging in GPU-Based LLM Serving Systems", Cotroneo & Cukic, posted 2026-06-10.

## Pointers

- Template: `proposals/nvidia/nvidia_academic_grant_2026_template.docx`
- Preliminary paper: arXiv:2606.11916 (under WoSAR review). "Characterizing Software
  Aging in GPU-Based LLM Serving Systems".
- Analysis + decisions: `paper/n3_analysis/`, `paper/PAPER_UPDATE_PLAN.md`,
  `EXPERIMENT_STATE.md`.
- NVIDIA Dynamo: https://developer.nvidia.com/dynamo (successor to Triton,
  GTC 2025, supports vLLM/TensorRT-LLM/SGLang, disaggregated serving).

---

> **SUPERSEDED 2026-06-12.** The experimental plan below (base full-factorial
> + separate stress-workload probe) is HISTORICAL. The grant proposal `.docx`
> is already submitted and frozen with its own Methods text, so this brief is
> no longer the plan of record. The CURRENT experimental plan is the **DoW
> screening campaign** in `paper/PAPER_UPDATE_PLAN.md` -> section
> "## 2026-06-12: EXPERIMENTAL PLAN — DoW" (5-factor Res V 16+3CP, 48h, three
> systems, rate as a factor to separate time- vs load-driven aging). Keep the
> section below only for the proposal's GPU-hours arithmetic.

## Experimental plan (LOCKED 2026-06-10) + run matrix + GPU-hours [HISTORICAL]

Terminology: we call the demanding-workload experiment a **stress-workload
probe** (NOT "adversarial" - that word reads as offensive-security and can
alarm reviewers). Security stays as an *implication*, reported responsibly.

Rationale from our own 36h data (why these durations):
- 48h ~ 36h: continuous cells plateau (E1 by ~h8) or drift linearly; little new.
- E2 (Triton+V0) was still stepping at h33 and the PyTorch VAS-only (VMS) was
  still rising linearly at h36: these are the NON-CONVERGED configs -> worth
  long runs.
- Natural leak is too slow to be a DoS by itself (E2 ~3.8 MB/day -> ~9 months
  to 1 GB). So DoS = AMPLIFICATION under stress workloads, tested by the probe,
  NOT by running longer benign runs.

### Design (three tiers)

1. **Base full factorial**: platform {vLLM standalone, Triton+vLLM, NVIDIA
   Dynamo} x hardware {L40S local, A100 grant} x model {Qwen2.5-7B, NVIDIA
   Nemotron} x **3 replicas** at **48h** = **36 runs** (18 local on L40S + 18
   on A100). Crossing the model in tells us whether aging depends on the model.
   Co-located multi-tenant regime as in the preliminary.
2. **Exploratory long runs**: **7-day (168h), n=1** each, on the non-converged
   configs: E2/Triton and a PyTorch-VAS cell on L40S (local, free); Dynamo on
   A100. Goal: late-onset effects + unbounded-vs-saturating + natural baseline
   for the security comparison.
3. **Stress-workload probe**: targeted runs varying request size, repetition,
   burstiness, rate; metric = amplification factor (leak rate under stress /
   benign). On A100, Triton + Dynamo.
4. **Conditional confirmatory**: IF the probe shows amplification, a **14-day
   run on A100 under the stress workload** to quantify impact. Duration set
   from the observed stress rate, not a fixed percentage.

### GPU-hours (A100 = grant; L40S local = free). Dynamo disaggregated = 2 GPU.

| Tier (A100 only; L40S half is local/free) | GPUs/run x dur x reps x models | A100-hours |
|------|------|-----------:|
| Base A100 - vLLM std   | 1 x 48 x 3 x 2 | 288 |
| Base A100 - Triton     | 1 x 48 x 3 x 2 | 288 |
| Base A100 - Dynamo     | 2 x 48 x 3 x 2 | 576 |
| 7-day exploratory (A100) | ~2 runs x 168 (x2 GPU) | ~672 |
| Stress probe (A100)    | targeted | ~300 |
| Conditional 14-day A100| 2 x 336 x 1 | 672 |
| Calibration (rate-sweeps) | short | ~150 |
| **Subtotal** | | **~2,946** |
| Margin re-runs/preemption (~30%) | | ~884 |
| **Total** | | **~3,800** |

L40S (local, free, not charged to grant): base 3 platforms x 48h x 3 +
2 exploratory 7-day runs.

**Request sizing.** Single-model plan ~2,400 A100-h. Adding a second served
model (NVIDIA Nemotron) roughly doubles the base (~+600) and an H100 tier or
extra 7-day runs add more. **Recommended request ~5,000-6,000 A100-h** (full two-model factorial
+ margin + the conditional run), well under the 32k ceiling ->
credible, well-justified, not inflated. Concurrent GPUs: ~6 (<=8). Storage:
~1 TB (traces + container images), well under 32 TB.

### Pre-campaign blocker (do NOT skip before analysis)

The `aging_trends.py` proc/gpu discovery bug (TODO #9) MUST be fixed before
analyzing the new campaign - Dynamo and multi-GPU produce new CSV names and
the nondeterministic discovery will bite again. Code is left untouched for now
by Domenico's instruction; apply the fix (use `aging_io.discover_proc_prefix`
+ gpu-prefix discovery) when the campaign analysis starts.
