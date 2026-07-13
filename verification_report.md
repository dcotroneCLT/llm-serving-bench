# Numerical Verification Report: WoSAR 2026 Paper vs Raw Campaign Data

**Paper:** Characterizing Software Aging in GPU-Based LLM Serving Systems (WoSAR 2026)
**Campaign:** `runs/wosar2026_*_r02` (replica 2, attempt 1, host cci-csgpu11, confirmed via each `manifest.json`)
**Verification date:** 2026-07-05. All processing was read-only; no campaign file was modified.

## Reproducibility parameters

- Process-monitor chunks per run concatenated, sorted by `ts_unix`, converted to hours since first valid sample. Rows with a non-empty `sample_error`, non-finite values, or non-positive memory values were skipped.
- Warmup window: the first **1.0 h** after the first valid process sample was excluded before all slope, increment, and correlation computations.
- Theil-Sen: `scipy.stats.theilslopes` (v1.15.3), alpha = 0.95, on post-warmup USS vs hours, after uniform subsampling with stride `step = ceil(n / 2500)`. Actual strides: step 11 (2291 points) for E1, E2, E3; step 10 (2435 to 2438 points) for E3b, A1, A2.
- Units: KB = 1000 bytes, MB = 10^6 bytes, GB = 10^9 bytes. (KiB-based values would be 2.4% lower and do not change any verdict.)
- 5-minute windows: last valid sample per 300 s window, post-warmup; increments taken only between consecutive windows. p99 computed over positive USS increments. RSS/VMS correlation is the lag-zero Pearson coefficient over the same per-window increments.
- Drop rate: the paper's Table III values are reproduced exactly when only `status == "dropped"` is counted. Rows with `status == "error"` or `status == "timeout"` are reported separately below.

## Table 1. USS Theil-Sen slope (paper Tables IV/V), KB/h

| Cell | Published (95% CI) | Recomputed (95% CI) | Abs. dev. | Rel. dev. | Verdict |
|------|--------------------|---------------------|-----------|-----------|---------|
| E1  | +1.8 [0.7, 23.8] | +1.75 [1.67, 1.85] | -0.05 | 2.5% | PASS (inside published CI) |
| E2  | +157 [33, 228]   | +157.50 [152.95, 161.72] | +0.50 | 0.3% | PASS (inside published CI) |
| E3  | +31 [21, 88]     | +31.44 [30.60, 32.35] | +0.44 | 1.4% | PASS (inside published CI) |
| E3b | +103             | +104.39 [99.76, 109.46] | +1.39 | 1.4% | PASS |
| A1  | +13              | +13.16 [12.93, 13.38] | +0.16 | 1.2% | PASS |
| A2  | +5.9             | +5.76 [4.15, 8.83] | -0.14 | 2.4% | PASS |

The recomputed analytic CIs are much narrower than the published CIs for E1, E2, E3; the published intervals presumably reflect cross-replica or bootstrap variability rather than the single-run theilslopes interval. All recomputed point estimates fall inside the published intervals.

## Table 2. Workload summary (paper Table III)

Recomputed from `client/requests_*.csv` of each r02 run. Drop rate counts `status == "dropped"` only (this reproduces the published numbers; see notes).

| Cell | Requests pub / rec (dev) | Drop % pub / rec | p50 s pub / rec | p99 s pub / rec | Tok/s pub / rec | Achieved / target RPS | Verdict |
|------|--------------------------|------------------|-----------------|-----------------|-----------------|-----------------------|---------|
| E1  | 329k / 329,327 (0.1%) | 0.01 / 0.0091 | 7.93 / 7.928 | 51.92 / 51.919 | 686 / 686.4 | 2.541 / 2.545 | PASS |
| E2  | 280k / 280,438 (0.2%) | 0.01 / 0.0064 | 8.70 / 8.705 | 59.18 / 59.178 | n/a / no token data | 2.163 / 2.172 | PASS |
| E3  | 23k / 22,532 (2.0%)   | 15.60 / 15.5956 | 405 / 405.4 | 536 / 535.8 | 40 / 39.6 | 0.1738 / 0.174 | PASS |
| E3b | 6k / 6,308 (5.1%)     | 0.00 / 0.0000 | 6.59 / 6.588 | 43.05 / 43.054 | 13 / 12.6 | 0.0487 / 0.050 | PASS (see note 4) |
| A1  | 103k / 103,339 (0.3%) | 0.00 / 0.0000 | 5.17 / 5.168 | 34.27 / 34.275 | 211 / 210.5 | 0.7974 / 0.796 | PASS (see note 4) |
| A2  | 227k / 226,651 (0.2%) | 0.00 / 0.0000 | 6.32 / 6.317 | 42.82 / 42.817 | n/a / no token data | 1.749 / 1.753 | PASS (see note 4) |

E2 and A2 client logs contain no usable `actual_output_tokens` values, consistent with the blank throughput cells in the paper.

## Table 3. E2 memory signature

| Claim | Published | Recomputed | Rel. dev. | Verdict |
|-------|-----------|------------|-----------|---------|
| p99 of positive 5-min USS increments (E2) | approaches 1.8 MB | 1.620 MB | 10.0% | FLAG (marginal, see note 2) |
| RSS/VMS increment correlation, E2 | ~0.83 | 0.753 | 9.3% | PASS (marginal, see note 1) |
| RSS/VMS increment correlation, E1 | <0.35 | 0.295 | - | PASS |
| RSS/VMS increment correlation, A1 | <0.35 | 0.209 | - | PASS |
| RSS/VMS increment correlation, A2 | <0.35 | 0.836 | - | FLAG (see note 1) |

## Table 4. E3b memory claims

| Claim | Published | Recomputed | Rel. dev. | Verdict |
|-------|-----------|------------|-----------|---------|
| VMS total growth | ~7.5 GB | 7.52 GB (raw endpoint delta) | 0.3% | PASS |
| Sustained VMS Theil-Sen rate | ~127 MB/h | 126.75 MB/h | 0.2% | PASS |
| RSS growth | ~20 MB | 24.5 MB raw; 8.4 MB post-warmup | 22.4% (raw) | FLAG (see note 3) |

## Table 5. Monitored duration per run (target 36 h, tolerance 5%)

| Cell | Monitor duration (h) | Dev. from 36 h | Client wall (h) | Verdict |
|------|----------------------|----------------|-----------------|---------|
| E1  | 36.000 | 0.0% | 36.00 | PASS |
| E2  | 36.000 | 0.0% | 36.01 | PASS |
| E3  | 35.999 | 0.0% | 36.02 | PASS |
| E3b | 34.856 | 3.2% | 36.01 | PASS (see note 4) |
| A1  | 34.840 | 3.2% | 36.00 | PASS (see note 4) |
| A2  | 34.814 | 3.3% | 36.00 | PASS (see note 4) |

## Supplementary: endpoint deltas (MB; raw / post-warmup)

| Cell | USS | RSS | VMS |
|------|-----|-----|-----|
| E1  | 19.8 / 1.8 | 19.8 / 1.8 | 6.3 / 2.1 |
| E2  | 22.0 / 6.7 | 23.0 / 7.0 | 75.5 / 2.1 |
| E3  | 288.4 / 7.8 | 409.5 / 7.9 | 8383.0 / 4304.6 |
| E3b | 24.5 / 8.3 | 24.5 / 8.4 | 7524.6 / 5741.2 |
| A1  | 244.2 / 225.8 | 19.2 / 0.6 | 4511.7 / 1.0 |
| A2  | 84.9 / 12.2 | 74.3 / 2.6 | 47.2 / 2.1 |

A1 endpoint deltas are dominated by a terminal artifact (note 5); its Theil-Sen slope is unaffected.

## Supplementary: drop rate trend, 6-hour bins (% of all non-ok statuses)

| Cell | 0-6h | 6-12h | 12-18h | 18-24h | 24-30h | 30-36h | >36h |
|------|------|-------|--------|--------|--------|--------|------|
| E1  | 0.03 | 0.02 | 0.02 | 0.04 | 0.01 | 0.01 | 0.0 |
| E2  | 0.00 | 0.00 | 0.02 | 0.00 | 0.01 | 0.01 | 0.0 |
| E3  | 17.1 | 15.5 | 16.2 | 17.3 | 13.5 | 15.9 | 100 |
| E3b | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 17.2 | 100 |
| A1  | 0.01 | 0.02 | 0.01 | 0.02 | 0.02 | 19.8 | 100 |
| A2  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 19.9 | 100 |

No temporal trend in E1, E2, E3 (E3 fluctuates around its steady ~15-17% saturation-induced drop level). E3b, A1, A2 are clean until the 30-36 h bin, where failures spike (note 4). The >36 h bin contains only a handful of overflow requests.

## Notes on FLAGs and observations

1. **FLAG, A2 RSS/VMS correlation (Table 3).** The paper states only E2 shows high lag-zero RSS/VMS increment correlation (~0.83) and other cells are <0.35. Recomputed A2 = 0.836, robust to the window aggregation choice (0.79 to 0.86 across last/mean/first/max per-window variants), and it numerically matches the value the paper attributes to E2. Recomputed E2 = 0.753 with last-sample windows, reaching 0.829 only with max-per-window aggregation. Possible E2/A2 attribution mix-up, or the "<0.35" claim was meant to cover only the standalone cells (E1 = 0.295, A1 = 0.209, which do pass). Recommend re-checking the sentence before camera-ready.
2. **FLAG (marginal), E2 p99 positive USS increment.** Recomputed 1.620 MB vs published "approach 1.8 MB", 10.0% deviation, exactly at the threshold. Sensitive to windowing: mean-per-window gives 1.31 MB, max-per-window gives 2.22 MB. The qualitative claim (increments up to ~2 MB) holds; consider stating the aggregation rule or softening the number.
3. **FLAG, E3b RSS growth.** Published ~20 MB; recomputed raw endpoint delta 24.5 MB (22% high), post-warmup delta 8.4 MB. Neither warmup convention reproduces 20 MB exactly; the value may come from a different window or from another replica. The qualitative claim (RSS growth two orders of magnitude below VMS growth) is fully supported.
4. **Observation, end-of-run failures in E3b, A1, A2 (no rule violated).** In these three runs the process monitor stops recording valid samples at 34.81 to 34.86 h (the final ~1.2 h of monitor rows carry sample errors: 823 to 854 rows), and client requests fail at ~17 to 20% in the 30-36 h bin and 100% afterwards, consistent with the engine process dying at ~34.8 h while the client kept submitting. Total non-ok request fractions are E3b 2.74%, A1 3.27%, A2 3.30%, all with `status == "error"`, not `"dropped"`, so the published 0.00% drop rates are technically correct under the paper's drop definition but silently exclude these terminal errors. Durations remain within the 5% tolerance (3.2 to 3.3% short). Worth a one-line disclosure in the paper if not already present.
5. **Observation, A1 terminal USS artifact.** A1 shows a single +225 MB USS step at the very last valid sample (t = 34.84 h) with zero RSS change, almost certainly a page-sharing accounting artifact at process teardown. It inflates A1's endpoint deltas (raw USS delta 244 MB) and its p99 positive increment (198 MB), but the Theil-Sen slope (+13.16 KB/h, matching the published +13) is robust to it.
6. **Observation, published CIs.** For E1, E2, E3 the published CIs are far wider than the single-run analytic theilslopes CIs, consistent with the published intervals encoding across-replica variability. All point estimates verify.

## Overall verdict

All headline numbers of Tables III, IV and V reproduce from the r02 raw data within rounding (most latency and throughput values match to three or four significant digits). Three FLAGs, all in the qualitative memory-signature claims: the A2 RSS/VMS correlation contradicts the "<0.35 elsewhere" claim (most likely an E2/A2 mix-up), the E2 p99 increment is 10% below the stated 1.8 MB, and the E3b RSS growth figure is ~20% off. None affects the paper's main conclusions, but note 1 should be resolved before camera-ready.
