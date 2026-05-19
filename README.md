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

The active experiment is the WoSAR 2026 replication campaign:

- Model: `Qwen/Qwen2.5-7B-Instruct`, BF16.
- Production cells: `e1`, `a1`, `e2`, `a2`, `e3`, `e3b`.
- Replication: 3 replicas per cell, 18 production runs total.
- Duration: 36h measured aging window per production run, after engine
  readiness.
- Warmup discard: first 1h excluded from slope and figure normalization.
- Topology: 3 parallel GPU slots, two cells per slot, round-robin by
  replica.

The canonical campaign descriptor is
`campaigns/wosar2026/campaign.yaml`; each cell YAML is the single source
of truth for container image pin, GPU assignment, monitor labels,
workload target rate, duration, and warmup discard.

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
bash scripts/smoke_test.sh campaigns/wosar2026/cells/e1.yaml
```

Preview and launch the full campaign:

```bash
python3 scripts/campaign.py --campaign-yaml campaigns/wosar2026/campaign.yaml --dry-run
python3 scripts/campaign.py --campaign-yaml campaigns/wosar2026/campaign.yaml --start
python3 scripts/campaign.py --campaign-yaml campaigns/wosar2026/campaign.yaml --resume
```

For a single cell/replica, use `scripts/launch_cell.py`; the campaign
orchestrator is preferred for production because it checkpoints state,
retries failures once, and keeps GPU slots balanced.

## Analysis

Pilot figures still run with no arguments. Production analysis should
point at the campaign descriptor and the run root:

```bash
python3 analysis/plot_rss_2x2.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root /home/dcotrone/wosar/runs --replicas all
python3 analysis/plot_rss_combined.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root /home/dcotrone/wosar/runs --replicas 1
python3 analysis/diagnose_step_patterns.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root /home/dcotrone/wosar/runs --cells a1,e2 --replicas all
python3 analysis/aging_trends.py /home/dcotrone/wosar/runs/wosar2026_e1_r01 --alpha 0.10 --downsample-seconds 60
python3 analysis/stepness.py --logs-root /home/dcotrone/wosar/runs --warmup-s 3600
```

See `analysis/README.md` for the full analysis pipeline.

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
