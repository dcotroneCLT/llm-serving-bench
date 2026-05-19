# Analysis Pipeline

This directory contains the post-processing scripts for pilot runs and
the WoSAR 2026 long-running software-aging campaign.

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

`analysis/stepness.py` quantifies the step-wise RSS growth pattern
documented in paper Section IV.E (Figure 2). It emits three metrics per
run with bootstrap 95% CIs:

- `rss_vms_corr` (lag-0): **primary** signature of the allocation
  mechanism. High (> 0.8) means RSS and VMS jump together (mmap-style
  whole-page allocations); low (< 0.5) means RSS jumps without VMS
  (sbrk-style heap extends).
- `K_trim`: **secondary** intensity metric. Excess kurtosis of ΔRSS
  after winsorizing the top 0.1 percent (`scipy.stats.mstats.winsorize(arr,
  limits=(0, 0.001))`). The raw `K` is reported alongside for
  reference; it is dominated by single extreme outliers and not
  comparable across runs.
- `steps_per_h_1mb`: **operational** descriptor. Count of ΔRSS events
  larger than 1 MiB per hour post-warmup.

Classification rule for the paper:

| corr | K_trim | category |
|---|---|---|
| `> 0.8` | `> 10` | mmap-style step-wise (canonical: E2) |
| `< 0.5` | `> 10` | RSS-only step-wise / sbrk-style (A1 candidate) |
| `< 0.5` | `< 5`  | continuous drift (canonical: E1) |

Usage on the local pilot logs (auto-discovery from `aging_pilot_*` and
`wosar2026_*` subdirs):

```bash
python3 analysis/stepness.py --logs-root logs
```

On the production campaign, **pass `--warmup-s 3600` explicitly** —
unlike the other analysis scripts the current implementation does not
load the per-cell `warmup_discard_s` from the campaign yaml and uses
its own 1800 s default.

```bash
python3 analysis/stepness.py \
  --logs-root /home/dcotrone/wosar/runs \
  --warmup-s 3600
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
- `analysis/stepness.py` and `analysis/aging_trends.py` do **not**
  currently read the cell yaml: their warmup defaults to 1800 s. On
  production campaign data pass `--warmup-s 3600` (stepness) or rely
  on the per-run timestamp window (aging_trends).
- Plot downsampling is time-based (`--plot-every-seconds`) rather than
  sample-count-based, so the figures remain valid if monitor sampling
  periods change.
