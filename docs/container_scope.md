# Whole-container monitoring and Dynamo-only restart

Implemented 2026-11-06. Hardware validation on cci-csgpu11 is still required.
Local validation: 490 tests passed under Linux/WSL, including scope failure
cases, CSV-to-analysis integration, shell health checks and legacy Dynamo
regressions. Generated-cell freshness check passed.
The main campaign was stopped deliberately at 19:44:47 UTC, with 44 completed
runs and vLLM p05 interrupted. Preserve that run and the original campaign
state; do not resume the original 57-cell queue.

## Measurement contract

The extension Triton/vLLM validation templates now select `container_tree`.
The DoW generator copies this setting to all 38 corresponding cells. Legacy
baseline cells retain their historical single-process behavior. Dynamo retains
its existing PGID sampler, component definitions and CSV schema.

The launcher binds the container's host PID1 and creation time into
`monitors.components`, saved in both manifest.json and components.json.
The existing sudo multiproc_monitor entry point selects the new sampler only
for `scope_type: container_tree`; no additional sudo executable is introduced.

The sampler anchors the root PID/create-time, PID namespace and cgroup. It
enumerates namespace/cgroup membership, including PID1, recursively spawned
workers, reparented workers and processes started by docker exec. It refuses
host PID namespaces, non-isolating root cgroups and an engine child supplied
as PID1. Every process is sampled once per tick. USS/PSS must be readable;
unavailable memory is an incomplete sample, never a zero-byte contribution.

Components partition the measured scope:

- `*_engine`: EngineCore and GPU compute processes, plus their descendants.
- `*_orchestration`: PID1 and matching backend/frontend processes outside the
  engine subtrees.
- `*_auxiliary`: every remaining scoped process, including resource trackers.

Roles are process-level operational labels, not a causal diagnosis. When
orchestration and inference share one process, its memory cannot be separated
by this instrumentation; it belongs to the engine component if GPU-bound.
Inspect scope_membership.jsonl on the box to verify the partition empirically.

The total response is `agg_vllm` or `agg_triton` and includes ALL three roles.
USS is additive; summed RSS/VMS can double-count shared mappings and are
diagnostic. `aging_trends.py` keeps `proc.uss_bytes` as the canonical total and
also emits `component.<role>.*` memory indicators. The existing FDR aggregator
accepts these indicators in the same test family.

Every tick independently queries GPU compute PIDs on the assigned GPU. At
least one must exist and every reported compute PID must be in the measured
scope. Under the strictly serial design, an outside GPU process is a failure,
not something to ignore. `model_scope_complete` records this check.

Births during a sample invalidate that tick; stable new processes can enter the
next sample. Loss or replacement of a previously sampled identity is latched
as incomplete for the rest of the run: memory resets must not masquerade as
aging steps. The production launcher's existing consecutive-health-failure
policy aborts the run and preserves its evidence. Monitor stalls and missing
model-coverage evidence also fail the runtime health check. Calibration sweeps,
which do not start the memory monitors, retain container-liveness checks.

Additional evidence:

- scope_identity.json: namespace, cgroup, PID1/create-time, GPU devices.
- scope_membership.jsonl: initial membership and changes, cmdlines, ancestry,
  creation times, role assignment and GPU PIDs.
- Aggregate CSV: sampled identities, scoped/model PIDs, completeness flags,
  membership changes and error reason. Invalid totals are blank and
  `process_alive=False`, so existing analysis excludes them.

The extension validator checks all complete ticks for model/PID1 coverage,
disjoint components, process identity continuity and memory conservation. It
requires at least 95% complete ticks; an incomplete run is not paper-ready.
`--scope-only` runs this evidence gate plus GPU CSV presence for short checks,
without claiming that a few minutes suffice to test aging statistically.

## Server validation before reboot

First pull this commit in the stopped server checkout and activate `wosar`.
Save the original state (including ALERT.json), calibrations and pre-maintenance
host/image versions before changing the environment. Keep completed and
interrupted run directories intact. An example state snapshot:

```bash
conda activate wosar
cd ~/wosar/llm-serving-bench
git pull --ff-only
snapshot="$HOME/wosar/ops/$(date -u +%Y%m%dT%H%M%SZ)_before_reboot"
mkdir -p "$snapshot"
cp -a campaigns/extension/state "$snapshot/state"
git rev-parse HEAD > "$snapshot/git_sha.txt"
uname -a > "$snapshot/kernel.txt"
nvidia-smi > "$snapshot/nvidia-smi.txt"
docker images --digests > "$snapshot/images.txt"
systemctl list-timers --all > "$snapshot/timers.txt"
```

Run two short checks through the production lifecycle. Unique run IDs avoid
overwriting old evidence. These are validation runs, not a restart of the
Triton/vLLM screening. If either command fails, resolve it before long runs.

```bash
(
set -e
scope_epoch="scopecheck_$(date -u +%Y%m%dT%H%M%SZ)"
for cell in val_vllm val_triton; do
  python3 scripts/launch_cell.py \
    --cell-yaml "campaigns/extension/cells/${cell}.yaml" \
    --replica 1 --duration-s-override 300 \
    --campaign-id "$scope_epoch" \
    --runs-root "$HOME/wosar/runs" --repo-root "$PWD" \
    --hf-cache-host "$HOME/wosar/hf_cache"
  python3 analysis/validate_extension_run.py \
    --run-dir "$HOME/wosar/runs/${scope_epoch}_${cell}_r01" --scope-only
done
)
```

The regular smoke_test.sh also selects the scope gate for these new cells.
For manual attach_run.py use `--engine-pid` with the container's HOST PID1
from docker inspect, not the EngineCore child. Validate attached runs with the
same evidence gate.

## Reboot and Dynamo tail

Perform the planned host maintenance/reboot once short validation is complete.
Preserve the pinned images and common vLLM version. Do not alter the IPMI
watchdog. Keep the campaign's nightly-timer isolation policy during subsequent
runs. Record kernel, driver, images and timers again after reboot and confirm
GPU health, Docker storage, sudo monitor access and the `wosar` environment.
Run the existing short Dynamo validation path before its long-run tail.

The four pending Dynamo cells are p16, p09, p05 (36h) and cp3 (48h). The checked-in
dow_dynamo_tail.yaml already contains precisely this order, with its own state
file and the original campaign_id/calibrations. Verify that none completed
since the checkpoint and that their target run directories are fresh. Preview:

```bash
python3 scripts/campaign.py \
  --campaign-yaml campaigns/extension/dow_dynamo_tail.yaml --dry-run
```

A fresh tail baseline after reboot does not renew calibration provenance.
Driver/image changes require recalibration; kernel or other maintenance changes
also require a scientific comparability assessment. Preserve old calibrations
if new ones are needed and record the environment block explicitly. Check the
120-day age gate at the projected dispatch of cp3, not only at startup.

After preflight passes, start the tail inside tmux with the same command using
`--start` instead of `--dry-run`, only for a new tail state file. For an existing
tail checkpoint use `--resume`. The original state snapshot preserves the
intentional-stop alert, which a fresh campaign start may clear. Always select
the tail descriptor explicitly in longtest_status.sh and campaign_health.sh.
No other system is queued after cp3.

After the tail, collect all 19 Dynamo runs, validate and analyze the aggregate
and component USS slopes, workload effects, stepness and cp1/cp2/cp3 consistency.
Treat pre/post reboot as separate blocks. Additional matched bridge measurements
may be needed if the environment or sentinel results differ. Only after this
analysis should the corrected Triton/vLLM screening be recalibrated and started
under a new campaign/state identity, preserving the original evidence.
