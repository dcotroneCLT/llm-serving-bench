# Whole-container monitoring and Dynamo-only restart

Implemented 2026-11-06. Hardware validation on cci-csgpu11 is still required.
The initial 490-test validation was followed by a scope/integration audit.
Its seven findings are addressed by the version-2 corrections described in
[scope_review_fixes.md](scope_review_fixes.md), with regression tests spanning
the sampler, validators, analysis CLI, discovery and attach supervision.
The main campaign was stopped deliberately at 19:44:47 UTC, with 44 completed
runs and vLLM p05 interrupted. Preserve that run and the original campaign
state; do not resume the original 57-cell queue.

## Measurement contract

The extension Triton/vLLM validation templates select `container_tree` with
`scope_version: 2`. Regenerate old templates; version-1 role evidence is not
approved by the corrected validator.
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

- `*_engine`: processes matching the verified EngineCore pattern and their
  descendants, with stable attribution keyed by PID and creation time.
- `*_orchestration`: PID1 and matching backend/frontend processes outside the
  engine subtrees.
- `*_auxiliary`: every remaining scoped process, including resource trackers.

GPU contexts do not determine roles: PID1 can own a CUDA context while remaining
orchestration. At least one GPU process must belong to a verified engine subtree;
unclassified GPU processes make the map ambiguous and fail validation. An
in-process/mixed deployment requires explicit empirical role resolution before
long runs. Inspect the error CSV, live process map and scope_membership.jsonl on
the box. Reparenting preserves each previously sampled process's attribution;
ancestry and GPU-context changes are recorded in the audit.

The total response is `agg_vllm` or `agg_triton` and includes ALL three roles.
USS is additive; summed RSS/VMS can double-count shared mappings and are
diagnostic. `aging_trends.py` keeps `proc.uss_bytes` as the canonical total and
also emits `component.<role>.*` memory indicators. The existing FDR aggregator
accepts these indicators in the same test family.

Every tick independently queries GPU compute PIDs on the assigned GPU. At
least one must exist and every reported compute PID must be in the measured
scope. `gpu_pids` records every compute context; `model_pids` records its subset
inside verified engine subtrees. Under strict serialism an outside GPU process
is a failure. `model_scope_complete` requires coverage and an identified engine.

Births during a sample invalidate that tick; stable new processes can enter the
next sample. Loss or replacement of a previously sampled identity is latched
as incomplete for the rest of the run: memory resets must not masquerade as
aging steps. The production launcher's existing consecutive-health-failure
policy aborts the run and preserves its evidence. Managed and attached runs use
the same completeness/freshness policy, including a final evidence check after
monitor flush. Monitor stalls and missing model coverage fail that check. Calibration sweeps,
which do not start the memory monitors, retain container-liveness checks.

Additional evidence:

- scope_identity.json: namespace, cgroup, PID1/create-time, GPU devices.
- scope_membership.jsonl: initial membership and changes, cmdlines, ancestry,
  creation times, role assignment and GPU PIDs.
- Aggregate CSV: sampled identities, scoped/model PIDs, completeness flags,
  membership changes and error reason. Invalid totals are blank and
  `process_alive=False`, so existing analysis excludes them.

Both validator modes first reject unfinished/interrupted runs, failed drain or
teardown, missing streams, insufficient duration, sparse/duplicate/out-of-order
monitor data and incomplete client windows. Valid coverage is measured against
the declared duration and recorded sampling periods, with a 95% sample floor
and bounded continuity gaps. The scope gate then checks model/PID1 coverage,
disjoint stable roles, identity continuity and memory conservation.
`--scope-only` is for complete short checks and omits the aging significance
pipeline. The full gate requires usable USS statistics from at least ten
post-warmup bins, but never requires a significant aging trend: stationary data
are valid. Prefix-repeat expectations come from saved client configuration or
manifest (CLI override is explicit), including zero and center-point targets.

Stepness masks first differences across process-identity changes, including
births, so process-count growth is not classified as allocator growth. Batch
discovery includes finalized, non-interrupted DoW manifests whose run_id matches
their directory; it excludes live, failed, archived-copy and short validation
runs. Use explicit --run-dir for diagnostics on those excluded runs.

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
