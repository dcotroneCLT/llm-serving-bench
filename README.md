# llm-serving-bench

Experimental benchmarking framework for comparing LLM serving engines under
sustained, long-duration GPU workloads. The project investigates software
aging in modern serving stacks, with a focus on resource-management
characteristics that are not visible in short-form throughput or latency
benchmarks.

This is a research codebase. It is being developed for an academic
publication and is currently in early alpha. APIs, configurations, and the
overall structure are expected to change.

## Current Campaign

The active experiment is the WoSAR 2026 **extension** (workload
day-of-week screening), a strictly serial follow-on to the completed n=3
replication baseline:

- Model: `Qwen/Qwen2.5-7B-Instruct`, BF16.
- Serving systems compared: Dynamo-disagg, Triton+vLLM, vLLM-standalone.
- Full screening campaign: 57 runs (54 x 36h + 3 Dynamo center points x 48h;
  window amended 2026-07-10), a Resolution V `2^(5-1)` workload DoW replicated on
  the three systems. It has its OWN descriptor
  `campaigns/extension/dow_campaign.yaml`, GENERATED from the design matrix in
  `scripts/generate_dow_cells.py` (see "DoW screening campaign" in
  `deploy/dynamo/README.md`). The separate `campaigns/extension/campaign.yaml`
  ships the three STEP-1 validation cells (one per serving system) that exercise
  the serial path end-to-end before the DoW runs and per-cell calibration.
- Warmup discard: first 1h excluded from slope and figure normalization.
- Topology: **one serial queue**, no parallel GPU slots — the
  measurement-isolation constraint forbids concurrency (the dynamo_disagg
  cells occupy both GPUs, and the other systems must not share the host).

The completed **baseline** was the n=3 replication over cells
`e1`, `a1`, `e2`, `a2`, `e3`, `e3b` (3 replicas each, 36h aging windows);
its retired parallel descriptor is noted under Legacy below.

The **active** campaign descriptor is
`campaigns/extension/campaign.yaml` (the WoSAR 2026 extension / workload
DoW screening). It is driven by `scripts/campaign.py`, which is **strictly
serial** by design — one global ordered queue of `(cell, replica)` runs on
the `launch_cell` production path, no parallel GPU slots (the
measurement-isolation constraint forbids concurrency). Each cell YAML is
the single source of truth for container image pin, GPU assignment, monitor
labels, workload target rate, duration, and warmup discard.

> **Legacy:** `campaigns/wosar2026/campaign.yaml` is the retired parallel
> (3-slot) descriptor from the original n=3 study, kept for provenance. The
> current serial orchestrator **rejects** it (it has no `mode: serial` and
> declares `slots:`), so it cannot be launched by accident.

## Repository layout

```
docs/         protocol, decision records, notes
client/       async benchmarking client (separate machine recommended)
monitoring/   metric collection agents (system / process / GPU)
engines/      configurations and Dockerfiles per serving engine
analysis/     statistical analysis scripts and notebooks
campaigns/    campaign descriptors and per-cell YAML definitions
paper/        manuscript sources (LaTeX)
runs/         experiment outputs (gitignored)
```

## Running

Use the smoke gate before burning a long GPU slot:

```bash
bash scripts/smoke_test.sh campaigns/extension/cells/val_vllm.yaml
```

Preview and launch the full campaign (exactly one of `--start`/`--resume`):

```bash
python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --dry-run
python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --start
python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --resume
```

`--start` wipes any existing state file and begins fresh; `--resume`
continues from the checkpoint (and refuses to resume a state file written
by a different `campaign_id`). For a single cell/replica, use
`scripts/launch_cell.py`; the campaign orchestrator is preferred for
production because it runs strictly serially on the production launch path,
checkpoints state after each run, and retries an ordinary failure once.

## Analysis

Install the analysis dependencies first if your Python environment does
not already have them:

```bash
python3 -m pip install -r analysis/requirements.txt
```

Pilot figures still run with no arguments. Production analysis points at a
campaign descriptor (or a specific run directory) and the run root.

### Active campaign (extension)

Per-run integrity checks. `validation_check.py` is for single-process cells
(e.g. `val_vllm`); a multi-process / Dynamo cell (`val_dynamo_disagg`) has an
aggregate proc series and must use the extension validator, which the
single-process checker also redirects to:

```bash
python3 analysis/validation_check.py --run-dir /home/dcotrone/wosar/runs/extension_dow_val_vllm_r01
python3 analysis/validate_extension_run.py --run-dir /home/dcotrone/wosar/runs/extension_dow_val_dynamo_disagg_r01
```

Cell-agnostic per-run trend / step analyses work directly on any run dir:

```bash
python3 analysis/aging_trends.py /home/dcotrone/wosar/runs/extension_dow_val_vllm_r01 --alpha 0.10 --downsample-seconds 60
python3 analysis/stepness.py --logs-root /home/dcotrone/wosar/runs
```

The campaign-aware plotters accept `--campaign-yaml`; pass extension cell ids
explicitly (their built-in defaults such as `--cells a1,e2` and
`--lockstep-cell e2` refer to retired baseline cells that do not exist in the
extension campaign):

```bash
python3 analysis/plot_rss_2x2.py --campaign-yaml campaigns/extension/campaign.yaml --runs-root /home/dcotrone/wosar/runs --cells val_vllm --replicas all
python3 analysis/diagnose_step_patterns.py --campaign-yaml campaigns/extension/campaign.yaml --runs-root /home/dcotrone/wosar/runs --cells val_vllm --replicas all
```

Note: `plot_rss_2x2` / `plot_rss_combined` were designed to overlay the six
baseline cells; they run on the extension campaign but their multi-cell /
lockstep framing (`--lockstep-cell`) is baseline-oriented, so on the two
current validation cells the single-panel overlay is the useful output.

### Baseline (n=3, retired)

These reproduce the completed n=3 WoSAR 2026 study from its archived runs
under `campaigns/wosar2026/`. Kept for provenance; the descriptor is not
launchable by the current serial orchestrator (see Legacy note above).

```bash
python3 analysis/plot_rss_2x2.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root /home/dcotrone/wosar/runs --replicas all
python3 analysis/plot_rss_combined.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root /home/dcotrone/wosar/runs --replicas 1
python3 analysis/diagnose_step_patterns.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root /home/dcotrone/wosar/runs --cells a1,e2 --replicas all
python3 analysis/aging_trends.py /home/dcotrone/wosar/runs/wosar2026_e1_r01 --alpha 0.10 --downsample-seconds 60
```

See `analysis/README.md` for the full analysis pipeline.

## Tests

The test suite runs off-box (no GPU, no docker). Install the dev
dependencies, then run it with the stdlib test runner:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m unittest discover -s tests
```

`requirements-dev.txt` pulls in the runtime packages the tests import
transitively (`PyYAML`, `psutil`, `httpx`) plus the analysis stack.

## Status

Production campaign tooling is in place: pinned cell descriptors,
single-cell launcher, campaign orchestrator, smoke gates, monitoring
agents, and campaign-aware analysis scripts. The repository remains a
research codebase, so paths and paper-facing labels may still change as
the data are finalized.

## Reproducibility

Once the experimental campaign is complete, the repository will include
pinned engine versions, Docker images, the full monitoring stack, raw data
links, and the analysis pipeline used to produce the figures and tables in
the paper.

## License

To be added.
