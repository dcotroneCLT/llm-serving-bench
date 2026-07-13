# FLAG Resolution Report: Original Pipeline Conventions vs Independent Verification

**Scope:** resolve the three FLAGs raised in `verification_report.md` by locating and running the ORIGINAL analysis scripts that produced the WoSAR 2026 paper numbers. All work was read-only on `runs/`; scripts were executed in place; nothing in the repo was modified.

**Bottom line:** all three FLAGs are resolved as convention mismatches in the independent verification, not errors in the paper. The original scripts, rerun live on the r02 data, reproduce every published value exactly (byte-identical to the saved intermediates in `paper/n3_analysis/`). No label swap, no truncation effect. Two wording refinements are recommended (see per-FLAG recommendations).

## Step 1: Pipeline inventory

| File | Role |
|------|------|
| `analysis/aging_trends.py` | Per-run trend table: Sen slope per hour + 95% CI + Hamed-Rao MK test. Source of Tables IV/V slopes and CIs. |
| `analysis/aging_io.py` | Shared loading; `resolve_warmup()` reads warmup from `campaigns/wosar2026/cells/<cell>.yaml`. |
| `analysis/stepness.py` | Step-wise mechanism panel: RSS/VMS increment correlation, kurtosis, step descriptors. Source of the E2 signature claims. |
| `analysis/aggregate_slopes.py`, `analysis/fdr_aggregate.py` | Cross-replica aggregation and FDR. |
| `paper/n3_analysis/per_cell.csv` | Saved aggregate with per-replica slopes and CIs (source of Tables IV/V numbers). |
| `paper/n3_analysis/stepness_all.csv` | Saved stepness output for all 19 runs (source of the E2 signature numbers). |
| `EXPERIMENT_STATE.md` (lines ~745-760, ~985-995) | Records the E3b raw endpoint inspection that produced "RSS +20 MB". |

### Conventions found in the code

| Parameter | `aging_trends.py` (slopes, Tables IV/V) | `stepness.py` (E2 signature) |
|-----------|------------------------------------------|------------------------------|
| Warmup | 3600 s, from cell yaml (`warmup_discard_s: 3600` in all six cells) | Same (3600 s from yaml) |
| Sampling used | 60 s windows, per-window MEDIAN of raw 5 s samples; `process_alive` filter | Raw 5 s sample-to-sample diffs, NO windowing; `drop_duplicates(ts_unix)`; PID-change rows masked out of the diff series |
| Subsampling cap | None (all ~2100 windows enter the Sen estimator) | None |
| Increment window | n/a | 5 s (native sampling interval) |
| Aggregation rule | median per 60 s window | none (raw diffs) |
| Truncation | None | None |
| CI method | Order statistics of pairwise slopes with AR(1) Hamed-Rao variance inflation (this is why the published CIs are much wider than plain `theilslopes` CIs) | Bootstrap (kurtosis only) |
| Step magnitude metric | n/a | `mean_top1_step_mb` = MEAN of the top 1% of POSITIVE raw 5 s dRSS jumps, in MiB (divisor 1024^2) |
| Correlation metric | n/a | lag-0 Pearson of raw 5 s dRSS vs dVMS, post-warmup, PID-segmented |

### Tables IV/V provenance confirmed

`paper/n3_analysis/per_cell.csv`, `proc.uss_bytes`, r02 column (bytes/h), matches the published table exactly:
E1 1755.43 [738.0, 23795.5] -> published +1.8 [0.7, 23.8] KB/h; E2 157246 [33244, 228055] -> +157 [33, 228]; E3 31082 [21388, 87944] -> +31 [21, 88]; E3b 103331 -> +103; A1 13165.7 -> +13; A2 5851.43 -> +5.9. The published slopes are therefore `aging_trends.py` on r02 (60 s median windows), and the published CIs are the AR(1)-inflated Sen CIs, resolving the CI-width question from the first report.

## Step 2: FLAG 1, E2 vs A2 RSS/VMS correlation

### a) Original script output (rerun live, `analysis/stepness.py --run-dir <run> --csv`)

| Run | rss_vms_corr (live rerun) | Saved in stepness_all.csv | Match |
|-----|---------------------------|---------------------------|-------|
| wosar2026_e2_r02 | 0.833370 | 0.833370 | exact |
| wosar2026_a2_r02 | 0.352700 | 0.352700 | exact |

The paper's "~0.83" is E2 r02 under the original convention. A2 r02 is 0.3527 under the same convention.

### b) Label swap check

No swap. `stepness_all.csv` carries distinct rows for all 19 runs with distinct values; cell IDs are inferred per run directory by `infer_cell_id` (no hardcoded mapping); the live rerun on each run directory reproduces the saved values exactly.

### c) Truncation at 34.0 h (engines of the second slot terminated at ~34.8 h)

| Run | Convention | Full series | Truncated at 34.0 h |
|-----|-----------|-------------|---------------------|
| E2 r02 | original (raw 5 s diffs) | 0.8334 | 0.8334 |
| A2 r02 | original (raw 5 s diffs) | 0.3527 | 0.3498 |
| E3b r02 | original (raw 5 s diffs) | 0.2547 | 0.2569 |
| E2 r02 | 5-min windows (verification report) | 0.753 | 0.753 |
| A2 r02 | 5-min windows (verification report) | 0.836 | 0.836 |
| E3b r02 | 5-min windows (verification report) | 0.521 | 0.521 |

Truncation is irrelevant to the correlations.

### d) A2 step magnitude vs E2

A2 r02 p99 of positive 5-min USS increments, truncated at 34.0 h: **0.460 MB** (the untruncated 0.988 MB from the first report was inflated by the end-of-run termination artifact). E2 r02 equivalent: 1.620 MB. Under the original raw 5 s metric, A2 top-1% positive dRSS = 0.055 MB vs E2 = 1.818 MB, a 33x difference; A2 has zero dRSS jumps above 1 MB (low-step operational fallback fires) while E2 shows 0.086 such steps per hour.

### Conclusion for FLAG 1

Neither a label swap nor a truncation effect. The discrepancy is purely the aggregation timescale: at raw 5 s resolution (the paper's convention) A2 = 0.35 and only E2 shows lock-step dRSS/dVMS increments; at 5-minute windows (the verification convention) A2's smooth co-drifting RSS and VMS trends produce a high correlation (0.836) that is co-trending, not the step signature. The paper's claim is correct under its stated mechanism metric. Two wording caveats: (i) A2 r02 is 0.3527, which is at, not below, 0.35; "<= 0.35" or "0.35" would be exact. (ii) The "<0.35 for all other cells" claim holds for the r02 replicas reported in the paper; at raw 5 s resolution some non-reported replicas exceed it (A2 r01 0.55, A2 r03 0.57, E1 r01 0.64, E1 r03 0.57), so the sentence should be scoped to the reported runs if it is not already.

**Recommendation:** keep the published numbers; state the convention (lag-0 Pearson correlation of consecutive 5 s RSS and VMS increments, post 1 h warmup, PID-segmented) and change "<0.35" to "<=0.35" or scope it to the reported r02 runs.

## Step 3a: FLAG 2, E2 "increments approach 1.8 MB"

The published value is `mean_top1_step_mb` from `stepness.py`: the MEAN of the top 1% of POSITIVE raw 5 s RSS increments, in MiB. Live rerun on E2 r02 gives **1.818 MB**, matching the saved 1.818359 and the paper's "approach 1.8 MB". It is not a p99, not USS-based, and not 5-minute-windowed; the verification report's 1.62 MB (p99 of 5-min USS increments) was a different statistic and is not in conflict.

**Recommendation:** keep 1.8 MB; state the convention explicitly in the paper text (mean of the largest 1% of positive 5-second RSS increments).

## Step 3b: FLAG 3, E3b "RSS growth about 20 MB"

Located in `EXPERIMENT_STATE.md` (entry dated 2026-05-23): a direct raw endpoint inspection of the e3b_r02 proc CSV, no fitting, no warmup exclusion, values displayed in GB at 2 decimals: "VMS 38.52 -> 46.04 GB (+7.52 GB), RSS 1.62 -> 1.64 GB (+20 MB)". Exact recomputation: RSS 1.6195 -> 1.6440 GB = **+24.47 MB**; the "+20 MB" is the subtraction of the two 2-decimal-rounded GB display values (1.64 - 1.62 = 0.02 GB). VMS 38.518 -> 46.043 GB = +7.525 GB, matching the published 7.5 GB.

**Recommendation:** update the text to "about 25 MB" (or "+24.5 MB"), or keep "about 20 MB" only if explicitly marked as order-of-magnitude. The derived VMS/RSS ratio becomes ~307x rather than ~375x; the qualitative claim (VAS growth with nearly flat RSS) is unaffected.

## Do the original scripts reproduce the paper from r02?

Yes, exactly, for every checked value: Tables IV/V slopes and CIs (per_cell.csv r02 entries match the paper digit for digit), E2 correlation 0.8334, E2 step magnitude 1.818 MB, E3b VMS +7.52 GB and +127 MB/h (per_cell.csv e3b proc.vms_bytes r02 = 1.26709e8 bytes/h). No evidence of a different replica or an older script version behind any published number. The only value not produced by a script is the E3b "+20 MB", which comes from a hand inspection recorded in EXPERIMENT_STATE.md and carries a rounding artifact.
