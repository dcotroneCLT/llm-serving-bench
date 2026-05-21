# Analysis Pipeline

This directory contains the post-processing scripts for pilot runs and
the WoSAR 2026 long-running software-aging campaign.

## Environment

Run the analysis scripts from an environment with the dependencies in
`analysis/requirements.txt` installed:

```bash
python3 -m pip install -r analysis/requirements.txt
```

## Inputs

Each run directory is self-contained:

```text
<runs_root>/<run_id>/
  manifest.json
  gpu<N>_*.csv
  <proc_label>_*.csv
  system_*.csv
  client/requests_*.csv
  logs/
```

Production run ids follow:

```text
wosar2026_<cell_id>_r<NN>
```

Examples: `wosar2026_e1_r01`, `wosar2026_a2_r03`.

For production analysis, prefer `--campaign-yaml
campaigns/wosar2026/campaign.yaml` plus `--runs-root`; the scripts then
derive cell labels, proc-monitor prefixes, replica ids, and warmup
discard from the campaign/cell definitions. Explicit `--run-dir` is
available for ad hoc runs.

## Standard Figures

2x2 factorial RSS figure across all available replicas:

```bash
python3 analysis/plot_rss_2x2.py \
  --campaign-yaml campaigns/wosar2026/campaign.yaml \
  --runs-root /home/dcotrone/wosar/runs \
  --cells e1,a1,e2,a2 \
  --replicas all
```

Combined paper figure with RSS panel plus RSS/VMS lock-step check:

```bash
python3 analysis/plot_rss_combined.py \
  --campaign-yaml campaigns/wosar2026/campaign.yaml \
  --runs-root /home/dcotrone/wosar/runs \
  --cells e1,a1,e2,a2 \
  --replicas 1 \
  --lockstep-cell e2
```

Diagnostic step-pattern plots:

```bash
python3 analysis/diagnose_step_patterns.py \
  --campaign-yaml campaigns/wosar2026/campaign.yaml \
  --runs-root /home/dcotrone/wosar/runs \
  --cells a1,e2 \
  --replicas all
```

Outputs default to `logs/figures/`. Use `--output-dir` and
`--output-prefix` when producing scratch figures.

## Trend Analysis

Run the statistical trend analysis per run directory:

```bash
python3 analysis/aging_trends.py \
  /home/dcotrone/wosar/runs/wosar2026_e1_r01 \
  --alpha 0.10 \
  --downsample-seconds 60
```

The analysis loads GPU, process, system, and client CSVs; downsamples
numeric indicators by time window; runs Hamed-Rao-corrected
Mann-Kendall tests; estimates Sen slopes; and applies Benjamini-Hochberg
FDR correction across indicators.

For one-off integrity checks after a run finishes:

```bash
python3 analysis/validation_check.py \
  --run-dir /home/dcotrone/wosar/runs/wosar2026_e1_r01
```

## Step-Wise Pattern Analysis

`analysis/stepness.py` quantifies the step-wise memory growth pattern
documented in paper Section IV.E (Figure 2). It emits these metrics per
run with bootstrap 95% CIs:

- `rss_vms_corr` (lag-0): **primary** signature of the allocation
  mechanism. High (> 0.8) means RSS and VMS jump together (mmap-style
  whole-page allocations); low (< 0.5) means RSS jumps without VMS
  (sbrk-style heap extension) or VMS jumps without RSS (address-space
  reservation without paging-in).
- `K_trim_dRSS`: **secondary** intensity metric on RSS. Excess kurtosis
  of ΔRSS after winsorizing the top 0.1 percent
  (`scipy.stats.mstats.winsorize(arr, limits=(0, 0.001))`). The raw
  `K_raw_dRSS` is reported alongside for reference; it is dominated by
  single extreme outliers and not comparable across runs. Bootstrap CIs
  filter undefined resamples from sparse/constant draws; for `K_trim`
  the winsorization is recomputed inside each bootstrap resample. The
  CSV column retains the historical name `K_trim` (same values).
- `K_trim_dVMS`: **secondary** intensity metric on VMS, computed with
  the same winsorize-then-kurtosis formula on ΔVMS. Added 2026-05-21 to
  capture VAS-only growth where address space is reserved by the
  allocator (e.g. PyTorch CUDA caching allocator on the host) but never
  paged in, so the dynamics are absent from dRSS.
- `steps_per_h_1mb`: **operational** descriptor. Count of ΔRSS events
  larger than 1 MiB per hour post-warmup. Diffs are PID-segmented:
  the row where `pid` changes vs. the previous sample is masked out
  of both ΔRSS and ΔVMS, so an engine restart does not inject an
  O(GB) artifact step into the count (synthetic test: a single PID
  transition with no intra-PID jumps > 1 MB inflated this metric to
  120 before the segmentation fix, correct value is 0).
- `mean_top1_step_mb`: **operational** descriptor. Mean of the top 1%
  of POSITIVE ΔRSS jumps, expressed in MB. The positive filter
  (`arr[arr > 0]` before the top-N selection) is required because on
  zero-heavy sparse series the 99th percentile of all diffs can be 0,
  in which case any `arr >= p99` mask admits every non-negative
  sample and the mean collapses to ≈ 0. Sanity: E2 pilot top1% under
  the old buggy formula was 0.005 MB, incompatible with its
  steps>1MB/h = 1.23; under the fix it is 2.57 MB.

### Five-class taxonomy

Implemented in `classify_stepness(corr, k_trim_drss, k_trim_dvms, notes)`:

| pattern                | condition                                           | K_trim_dRSS | K_trim_dVMS | mechanism interpretation |
|------------------------|-----------------------------------------------------|-------------|-------------|---------------------------|
| **border (VMS missing/unusable)** | `VMS_missing` or `VMS_unusable` in notes (cell breakage, monitor crash, `vms_bytes` column absent, or no finite ΔVMS samples post-warmup) | n/a | n/a | Highest priority: a run with no usable VMS axis cannot be classified on the (corr, K_trim_dRSS, K_trim_dVMS) panel; returning `continuous drift` would silently swallow missing data, so the row is flagged `border` for replica review. |
| continuous drift (low-step fallback) | both axes in low-step operational fallback (notes contain `RSS_low_step_operational_drift` AND `VMS_low_step_operational_drift`) | n/a (forced 0) | n/a (forced 0) | No significant step events on either axis; the measured corr is micro-noise correlation, not a mechanism signature. Takes precedence over the (corr, K_trim) rules below (but yields to missing/unusable VMS above). Canonical examples: `wosar2026_e1_r01` (corr=0.64 but both axes flat), A2 pilot (corr=0.58, K_raw_dRSS=3533 winsorize-artifact, K_raw_dVMS=1.08). |
| mmap-style step-wise   | corr `> 0.8`                                        | `> 10`      | `> 10`      | RSS and VMS step together at discrete events: kernel-mapped pages never returned. Canonical example: E2. |
| sbrk-style step-wise   | corr `< 0.5`                                        | `> 10`      | `< 5`       | RSS heap-arena extends without paired VMS step: glibc/jemalloc sbrk path, no kernel mmap involved. |
| VAS-only step-wise     | corr `< 0.5`                                        | `< 5`       | `> 10`      | VMS-only jumps with no resident component: address space reserved (anonymous mmap, MAP_NORESERVE-like) and never paged in. Canonical example: `wosar2026_e3_r02` (PyTorch CUDA caching allocator reserves host-side VAS for device-side mappings without touching CPU memory). |
| uncorrelated step-wise | corr `< 0.5`                                        | `> 10`      | `> 10`      | RSS and VMS both jump but desynchronized in time: heap-arena extension (RSS-side) and mmap-style allocation of large blocks (VMS-side) operating in parallel as two independent allocator phenomena. Canonical example: `aging_pilot_24h_vllm_v0_ablation_v2` (A1 pilot n=1) with corr=0.20, K_trim_dRSS=402, K_trim_dVMS=582. |
| continuous drift       | corr `< 0.5`                                        | `< 5`       | `< 5`       | Smooth small-grain accumulation everywhere, no large step events. Canonical example: E1. |
| border                 | mixed                                               | mixed       | mixed       | Out-of-bin or NaN on (corr, K_trim) with no fallback on either axis; needs replica confirmation. |

### Low-step fallback

Operational-driven, NOT math-driven. The rule keys off
`steps_per_h_1mb` on the same axis, independent of whether the raw
`K_trim` was computable numerically:

- if `steps_per_h_1mb < 0.01` on a given axis (≈ < 1 MB-scale jump
  per 100 hours), the series has no real step events on that axis
  and `K_trim` is overridden to `0.0`, even when the raw computation
  returned a large finite kurtosis (winsorize on a near-flat series
  inflates K from a handful of sub-MB outliers). A stderr warning is
  emitted and the output row gains a `notes` entry
  `RSS_low_step_operational_drift` or
  `VMS_low_step_operational_drift`. Threshold calibrated an order of
  magnitude below the lowest mmap-style cell observed in the n=3
  campaign (e2_r02 at ≈ 0.09 steps/h) so genuine sparse step-wise
  runs are not swept up by the fallback.
- otherwise, if `K_trim` is NaN/inf, the row is tagged
  `RSS_kurtosis_undefined` / `VMS_kurtosis_undefined`. The
  classification function maps any NaN component on
  (corr, K_trim_dRSS, K_trim_dVMS) to the `border` bucket so the
  downstream pipeline does not break.
- if `vms_bytes` is absent, all-NaN, or has no finite adjacent
  post-warmup deltas after PID segmentation, the row is tagged
  `VMS_missing` or `VMS_unusable` and classified as `border` before the
  low-step fallback can fire.

When the fallback fires on BOTH usable axes, the classification function
short-circuits to `continuous drift` regardless of corr; see the
priority rows at the top of the taxonomy table above.

Usage on the local pilot logs (auto-discovery from `aging_pilot_*` and
`wosar2026_*` subdirs):

```bash
python3 analysis/stepness.py --logs-root logs
```

On the production campaign, `stepness.py` auto-resolves warmup via
`aging_io.resolve_warmup`: `wosar2026_*` runs read
`campaigns/wosar2026/cells/<cell>.yaml` (`warmup_discard_s=3600`), and
pilot runs use the 1800 s convention. Pass `--warmup-s` only for an
explicit sensitivity check.

```bash
python3 analysis/stepness.py \
  --logs-root /home/dcotrone/wosar/runs
```

Add `--csv` for machine-readable output or `--top-k N` to also dump the
top-N ΔRSS event timestamps per run.

## Rate Sweeps

Aggregate a saturation sweep and write a reusable CSV table:

```bash
python3 analysis/sweep_curve.py \
  /home/dcotrone/wosar/runs/pilot_vllm_sweep_v2 \
  --output-csv /home/dcotrone/wosar/runs/pilot_vllm_sweep_v2/sweep_summary.csv
```

The script accepts integer and decimal rate directories such as
`client_04rps/` and `client_2.5rps/`, reports the sustainable knee, and
prints the recommended aging target rate as a fraction of effective RPS.

For a single-rate summary (one `requests_*.csv` directory), use
`analysis/sweep_summary.py <run_dir>`: it prints request counts, OK /
dropped breakdown, latency percentiles, and TTFT for streaming
requests. It is the per-level counterpart of `sweep_curve.py`.

## Defaults and Warmup

- No-argument plot scripts use the local 24h pilot logs under `logs/`.
- Campaign mode (via `--campaign-yaml`) uses each cell's
  `warmup_discard_s` field; for `wosar2026` this is 3600 seconds. This
  applies to `plot_rss_2x2.py`, `plot_rss_combined.py`, and
  `diagnose_step_patterns.py` (all of which go through `aging_io.py`).
- `--warmup-hours` overrides manifest/campaign warmup for sensitivity
  checks.
- `analysis/stepness.py` and `analysis/aging_trends.py` use run-root
  paths rather than the plotting scripts' `--campaign-yaml` interface,
  but both now resolve campaign warmup from project conventions. On
  production campaign data, omit `--warmup-s` for the cell-yaml value or
  pass it explicitly for sensitivity checks.
- Plot downsampling is time-based (`--plot-every-seconds`) rather than
  sample-count-based, so the figures remain valid if monitor sampling
  periods change.
