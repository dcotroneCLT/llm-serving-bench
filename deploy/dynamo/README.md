# NVIDIA Dynamo local bring-up (extension campaign)

Local CLI deploy (no Kubernetes), single L40S box. Part of the 3-system
extension campaign that holds **vLLM 0.20.1 identical** across Dynamo,
Triton+vLLM, and standalone vLLM (see the pin files in `engines/*/image_pin*`).

Authoritative constraints: **`docs/extension_pin_constraint.md`** (the vLLM pin)
and EXPERIMENT_STATE.md "Standing constraints" SC-2 (disk-space management). The
docker runs here cap container logs (`--log-opt`); keep the docker data-root on
/home (not the 126G /var/lib).

Pin history: 0.16.0 was the first choice but the box gate found Triton ships
no 0.16.0 build (it skips from 0.15.1 at 26.02 to 0.17.1 at 26.03). The
three-way native intersection is **0.20.1** (Dynamo 1.2.0 + Triton 26.05 +
standalone v0.20.1). Always ground-truth with `pip show vllm` at pull.

The exact pinned image and component commands are encoded in `env.sh` and the
`serve_*.sh` scripts. Everything runs on the host network so the
OpenAI-compatible frontend and the etcd/NATS discovery are reachable on
localhost, and so the host `ps`/`/proc` sees the `python -m dynamo.*` processes
(which is what the per-component monitor matches).

## Pinned stack

| Piece | Value |
|---|---|
| Dynamo | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13` (1.2.0 stable) |
| Triton | `nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3` |
| Standalone | `vllm/vllm-openai:v0.20.1-cu129` |
| vLLM | 0.20.1 (identical across all three) |
| CUDA | Dynamo/Triton 13.x; standalone 12.9 (cu129). Both OK on driver 580.x |
| Model | Qwen/Qwen2.5-7B-Instruct, ctx 8192, BF16 |

## 0. Verify the vLLM version (the whole point of the pin)

The ground truth that all three systems share vLLM 0.20.1 (remote release notes
were wrong once, so pip show is authoritative). NOTE: vllm-openai's entrypoint is
the API server, so it needs `--entrypoint pip`:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 pip show vllm | grep -i '^Version'   # 0.20.1
docker run --rm nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3 pip show vllm | grep -i '^Version'    # 0.20.1 (gate-critical)
docker run --rm --entrypoint pip vllm/vllm-openai:v0.20.1-cu129 show vllm | grep -i '^Version'          # 0.20.1
# record each digest into the matching engines/*/image_pin*.json:
for img in nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 \
           nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3 \
           vllm/vllm-openai:v0.20.1-cu129; do
  docker inspect --format '{{index .RepoDigests 0}}' "$img"
done
```

STOP if any is not exactly 0.20.1.

## 1. Infrastructure (etcd + NATS)

```bash
bash deploy/dynamo/infra_up.sh
docker ps | grep -E 'dyn_etcd|dyn_nats'
```

## 2a. Aggregated (single GPU) — simplest, de-risk first

```bash
bash deploy/dynamo/serve_aggregated.sh
curl -s http://localhost:8400/v1/models   # must list the model, not {"data":[]}
```

## 2b. Disaggregated (2 GPU) — the campaign topology

```bash
N_PREFILL=1 N_DECODE=1 PREFILL_GPU=0 DECODE_GPU=1 bash deploy/dynamo/serve_disaggregated.sh
curl -s http://localhost:8400/v1/models   # must list the model
# end-to-end smoke (HTTP 200 + a completion):
curl -sS http://localhost:8400/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Reply with one word: hello"}],"max_tokens":16}'
```

Fixed topology, planner/autoscaler intentionally not launched, so the component
set is constant for the whole run (a moving worker set would inject fake leak
steps into the aggregate).

**Use `/v1/models` (not `/health`) as the readiness check.** `/health` reports
`healthy` even when no model is registered; only a non-empty `/v1/models` means
the stack can actually serve.

### Bring-up requirements baked into `serve_disaggregated.sh` (gate-2 findings)

All real-hardware issues, found by the STEP 1 gate before any 48h run:

- **etcd peer URLs** must be the literal `127.0.0.1` with an explicit
  `--initial-cluster` (see `infra_up.sh`): etcd rewrites a `localhost` advertise
  URL to the host IP but leaves `--initial-cluster` verbatim, so it exits 1.
- **`--user 0:0`** on the workers: the shared HF cache is root-owned (written by
  the root standalone arm); the image's default uid 1000 cannot write it.
- **`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`**
  on both workers: `--connector` is deprecated and the mode no longer defaults to
  nixl, so a prefill worker without it exits 1.
- **distinct `VLLM_NIXL_SIDE_CHANNEL_PORT` per worker** (5600, 5601, ...): all
  workers share the host network and otherwise clash on the default 5600
  ("Address already in use" in the NIXL handshake listener).
- **frontend start + restart-retry**: the frontend snapshots the model registry
  at startup and does NOT pick up a model that finalizes afterward (a frontend
  started right after the workers log "Registered base model" serves an empty
  `/v1/models` for 2+ min; a plain restart once the workers are fully ready
  serves immediately). So the script launches workers first, then (re)starts the
  frontend and polls `/v1/models`, restarting it until the model is listed. A
  frontend that serves an empty `/v1/models` 404s every request.

Known benign warning: `'EngineCoreProc' object has no attribute
get_kv_cache_group_metadata` — a Dynamo-1.2.0/vLLM-0.20.1 API drift that falls
back to `cache_config.block_size`; the worker still registers and serves.

## Registry flakiness — ROOT CAUSE (resolved)

**Root cause (found):** the frontend container was started WITHOUT the shared,
root-owned HF cache identity that the workers use (`--user 0:0` + the
`$HF_CACHE:/root/.cache/huggingface` mount). The frontend's discovery watcher
materializes the model card on the fly via `hub::from_hf()`, and without that
cache it fails:

```
ERROR dynamo_llm::discovery::watcher: Error adding model from discovery
  model_name="Qwen/Qwen2.5-7B-Instruct" namespace="dynamo"
  error="hub::from_hf(...): Failed to create cache directory
  \"/root/.cache/huggingface/hub\": Permission denied (os error 13)"
```

So `/v1/models` stayed `{"data":[]}` even though **etcd held all 9 `dynamo` keys
and worker registration was fine** — etcd/namespace/worker registration were
NEVER the problem, and the frontend's `/health` stayed 200 the whole time
(another reason `/health` is not a readiness check). It looked *flaky* only
because there were TWO divergent frontend start paths (`serve_disaggregated.sh`
and `diag_registry.sh`), started differently from run to run.

**Fix:** the frontend is now started by a single `start_frontend()` helper in
`env.sh`, using the SAME identity and cache mount as the workers (`--user 0:0`,
`COMMON_ENV`, `COMMON_MOUNT`, host network, no `--gpus` — the frontend does not
touch the GPU, so the "NVIDIA Driver was not detected" warning is expected). All
frontend starts — `serve_disaggregated.sh` (including its restart fallbacks),
`serve_aggregated.sh`, and `diag_registry.sh` — go through it; no duplicated
`docker run … dynamo.frontend` lines remain anywhere under `deploy/dynamo/`.
`/v1/models` remains the ONLY readiness check.

### `diag_registry.sh` (kept for regressions)

If the stack ever fails to serve again — workers log `Registered base model`,
the frontend is Up with HTTP 200 on `/health`, yet `/v1/models` stays
`{"data":[]}` — `diag_registry.sh` localizes the fault to one of two branches:

```bash
conda activate wosar
bash deploy/dynamo/diag_registry.sh          # ~6 min window, then auto-cleanup
WINDOW_S=600 bash deploy/dynamo/diag_registry.sh   # longer window
```

It brings up infra + **workers only** (via `serve_disaggregated.sh
FRONTEND_START=manual`, so the script's own frontend fallback logic is out of
the way), starts **exactly one** frontend itself at a controlled time with **no
churn**, and polls both etcd (`dynamo` key count) and `/v1/models` every 10 s.
On exit it captures evidence **before** cleanup into
`~/wosar/diag_registry/<ts>/` (full etcd key dump with values, full frontend /
prefill / decode logs, and a decode-log summary of `get_kv_cache_group_metadata`
and any de-registration / lease-expiry lines), then cleans up like the gate
(serve_down + infra_down + reaper + `docker rm -f dyn_*`). Exit code is 0 only on
`REGISTRY_OK`, 2 otherwise. Decision tree from the one-line `VERDICT`:

| VERDICT | signal | branch |
|---|---|---|
| `REGISTRY_OK` | `/v1/models` listed the model within the window | not flaky this run |
| `FRONTEND_DISCOVERY_SUSPECT` | etcd `dynamo` keys stayed >0 for ≥60 s while `/v1/models` stayed empty | frontend never reads the registration into `/v1/models` → look at frontend discovery / the etcd→frontend watch |
| `WORKER_REGISTRATION_SUSPECT` | etcd `dynamo` keys were 0 the whole window, or registered then dropped back to 0 | registration never persisted or the worker's lease expired → look at worker registration / lease keep-alive |

(A drop-to-zero outranks "keys present but not discovered", so a run that
registers then loses its keys reports `WORKER_REGISTRATION_SUSPECT`.)

A failed `repass_gate2.sh` bring-up is never evidence-free: on `bringup FAIL` the
gate dumps the `dyn_*` container logs and the etcd `dynamo` key list into
`$RUN_DIR/bringup_failure_evidence/` before cleanup.

## 3. Component identity = recorded PGIDs (NOT a host-wide regex)

The per-component monitor sums EXACTLY the processes the bring-up recorded, not
whatever a host-wide cmdline regex matches. At the end of
`serve_disaggregated.sh`, `record_component_pids.py` writes the host PGID of each
component's container init process to the identity file
(`$COMPONENT_PIDS_FILE`, default `~/wosar/dynamo_component_pids.json`):

```json
{"engine_group":"dynamo","components":{
  "dynamo_prefill":{"containers":["dyn_prefill_1"],"host_pids":[751535],"pgids":[751535],"expected_count":1}, ...}}
```

`attach_run.py --component-pids <file>` (and, in BATCH 2, `launch_cell`) merges
those pgids into the `components.json` the monitor reads. The monitor then:

- sums every process whose PGID is in the recorded set (so a vLLM EngineCore
  fork in the same group is captured; a single recorded PID would miss it);
- `membership_complete` requires EXACT membership (`n_pgids_alive == expected`,
  never `>=`): a dead instance (under) marks the tick incomplete;
- a process matching a component's cmdline regex but OUTSIDE its pgids is a stray
  (orphan / duplicate): it is NEVER summed and is counted into `n_pids_unexpected`.

The frozen `pattern` / `require` / `exclude` regexes in
`campaigns/extension/cells/val_dynamo_disagg.yaml` are now a SANITY CHECK and the
labeling aid (prefill = `--disaggregation-mode prefill`, decode = `decode`), not
the identity source. Sanity-eyeball the real tree once:

```bash
ps -eo pid,pgid,cmd | grep -E 'dynamo.frontend|dynamo.vllm' | grep -v grep
```

**Diagnostic vs red flag (locked):** `n_pids_unexpected` is a PURE PER-TICK
DIAGNOSTIC (the stray is outside the pgids, so the tick's aggregate is already
correct; never invalidate the tick on it). Enforcement is at RUN level:
`validate_extension_run.py` FAILS the run if any tick has `n_pids_unexpected>0`,
because a stray almost always means an orphan from a prior run that the BATCH 1
#2 reaper should have cleared.

**One-time completeness check (gate-2 re-pass).** PGID scoping's only failure
mode is a component child that `setsid`s into a different pgid, escaping the sum
without tripping `membership_complete`. Confirm none exists, live, under sudo:

```bash
sudo -E python3 deploy/dynamo/verify_scoping.py --component-pids "$COMPONENT_PIDS_FILE"
# VERDICT: COMPLETE  => every dynamo-related memory-holding PID is in a recorded pgid
```

## 4. First-bring-up flag check

The component commands in `serve_*.sh` use the explicit `--disaggregation-mode
{prefill,decode}` form (`python -m dynamo.frontend`, `python -m dynamo.vllm
--disaggregation-mode prefill|decode`). The legacy `--is-prefill-worker` flag
is deprecated in this image, and a worker with no mode flag defaults to `agg`
(aggregated), which would silently collapse the disaggregated topology.
Confirm the exact flags once. NOTE: `dynamo.vllm --help` initializes the device,
so it needs a GPU attached or it crashes with "Failed to infer device type":

```bash
docker run --rm --gpus '"device=0"' nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 python -m dynamo.vllm --help
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 python -m dynamo.frontend --help
```

Adjust env var names for etcd/NATS endpoints if `--help` shows different ones.

## 5. Teardown

```bash
bash deploy/dynamo/serve_down.sh     # engine components
bash deploy/dynamo/infra_down.sh     # etcd + nats
```

## Two ways to drive a run: `launch_cell` (unattended) vs `attach_run` (hand-started)

There are two entry points, and which one to use depends on who owns the engine
lifecycle:

- **`scripts/launch_cell.py` — full unattended lifecycle (campaign runs).** As of
  PHASE B item 2, `launch_cell` owns the whole Dynamo disaggregated stack:
  bring-up, readiness, component identity, and teardown, exactly as it does for
  single-container cells. Selected by `engine.lifecycle: dynamo_disagg` in the
  cell yaml (default `single_container`), with the topology under
  `engine.topology` (`n_prefill`, `n_decode`, `prefill_gpu`, `decode_gpu`). It
  runs `infra_up.sh` then `serve_disaggregated.sh` with those as env vars,
  **inheriting the scripts' fail-hard exit codes** (a non-zero exit aborts the
  run with the script output captured under `run_dir/logs/`); it re-verifies
  `/v1/models` lists the model, merges the recorded component PGID identity
  (`record_component_pids.py`), captures a VRAM baseline on BOTH GPUs, GPU-sanity
  -checks each worker against its assigned device, and on teardown saves logs for
  every stack container then runs `serve_down.sh` + `infra_down.sh` and sweeps
  any leftover `dyn_*`. The shell scripts remain the SINGLE source of truth for
  how containers start — `launch_cell` does not duplicate any `docker run`.

  ```bash
  python3 scripts/launch_cell.py \
      --cell-yaml campaigns/extension/cells/val_dynamo_disagg.yaml \
      --replica 1 --runs-root ~/wosar/runs --repo-root ~/wosar/llm-serving-bench \
      --hf-cache-host ~/wosar/hf_cache --campaign-id extension
  # identity file default: $WOSAR_COMPONENT_PIDS or ~/wosar/dynamo_component_pids.json
  ```

- **`scripts/attach_run.py` — hand-started stack (STEP 1 validation).** You bring
  the engine up by hand (`infra_up.sh` + `serve_disaggregated.sh`) and attach the
  monitors + client + manifest to the running frontend; `attach_run` does NOT
  manage the engine lifecycle. It ignores `engine.lifecycle`/`engine.topology`
  and keeps working unchanged. See `scripts/attach_run.py --help` and the
  `campaigns/extension/cells/val_*.yaml` cells.

Single-container cells (standalone vLLM, Triton) are unaffected: their
`launch_cell` path and manifests are byte-identical to before.

## Driving the whole extension campaign: `scripts/campaign.py` (strictly serial)

The extension campaign is ~57 runs of 48h each and is **strictly serial by
design** — measurement isolation demands one run at a time, and the
`dynamo_disagg` cells occupy both GPUs anyway. `scripts/campaign.py` dispatches
**one global ordered queue** of `(cell, replica)` runs, one `launch_cell.py`
subprocess at a time (the production path above), each waited to full
completion (teardown + VRAM quiescence happen *inside* `launch_cell`) plus a
configurable `inter_run_cooldown_s` before the next run. There is no thread
pool and no parallel-slot model; a campaign yaml that tries to express
parallelism (a `slots:` key, or `mode:` other than `serial`) is rejected at
load time. (The retired n=3 parallel scheduler lives in this file's git
history, not its runtime.)

```bash
# Pre-flight only: full schedule in order, per-cell calibration status,
# estimated wallclock, and the free-space gate — then exit.
python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --dry-run

# Start fresh (wipes the state file) — run it inside tmux; it also tees all
# output to campaigns/extension/state/logs/campaign_<ts>.log (stdout dies with
# the terminal, and the run must survive tmux loss):
tmux new -d -s ext_campaign \
  'python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --start'

# Stop and pick up across days: --resume skips completed runs and re-queues
# interrupted/failed ones under the retry policy.
python3 scripts/campaign.py --campaign-yaml campaigns/extension/campaign.yaml --resume
```

The campaign yaml schema is documented in `campaigns/extension/campaign.yaml`
(and the `scripts/campaign.py` module docstring). Behaviour that matters
operationally:

- **Retry policy.** An ordinary `launch_cell` failure is re-attempted **once**
  (with `--attempt` incremented, so the stale run_dir is archived and a fresh
  one is used), then marked `failed` and the queue moves on. `launch_cell`
  **exit 9** (run-slot flock held by another launcher), **exit 8** (orphan
  gate: a prior run is still active or has an unkillable orphan) and **exit 7**
  (free-space gate on runs-root or the docker data-root) are **NOT run
  failures** — they are host/precondition faults that a retry cannot fix, so
  the campaign stops **loudly** (exit 5) without burning attempts. Resolve the
  precondition (host ownership or free space), then `--resume`.
- **Calibration is a pre-flight gate.** Per-cell `--calibration-file` is passed
  through to `launch_cell`; if a required calibration is missing, or if any
  provided calibration file is invalid/rejected (`status != "ok"` without
  `allow_lower_bound_calibration`), the campaign fails **before run 1**, not at
  run N=37. The same `launch_cell` gate is reused so pre-flight accepts exactly
  what the run will accept.
- **Signals.** `SIGTERM`/`SIGINT`/`SIGHUP` forward `SIGTERM` to the current
  `launch_cell` child (graceful teardown), persist state (the in-flight run is
  marked `interrupted`), and exit non-zero (4).
- **State + resume.** `campaign_state.json` is written atomically (tmp +
  rename) after every status change; `--resume` re-queues everything that is
  not `completed`.

Exit codes: `0` complete · `2` usage · `3` pre-flight failed · `4` interrupted
by signal · `5` campaign-fatal (host ownership or free-space precondition).

## Re-pass gate (BATCH 1 + PHASE A): one command

`repass_gate2.sh` is the consolidated box-validation gate. It brings the
disaggregated stack up, drives a ~1500 s VALIDATION run (not a soak), runs the
multi-process validator and the empirical scoping check, proves the fail-loud
failure path, and confirms the disk check resolves the real docker data-root. It
cleans up (serve_down + infra_down + reaper) on exit, success or failure.

```bash
conda activate wosar
bash deploy/dynamo/repass_gate2.sh        # ~30 min; prints a PASS/FAIL summary
# overrides: DURATION_S=900 RUNS_ROOT=~/wosar/runs bash deploy/dynamo/repass_gate2.sh
```

It ends with a single summary; the gate is green only when OVERALL is PASS:

| check | meaning |
|---|---|
| `pip_pin` | vLLM 0.20.1 on all three images |
| `bringup` | disaggregated stack served the model |
| `validator` | `validate_extension_run.py` -> GATE: PASS |
| `no_orphans` | no tick had a stray outside the recorded pgids |
| `n_pids_unexpected_0` | explicit: max `n_pids_unexpected` over engine components is 0 |
| `verify_scoping` | smaps USS of all dynamo PIDs matches the recorded-pgid aggregate within tolerance (COMPLETE) |
| `fail_loud_negative` | bring-up against a bogus model EXITS NON-ZERO (does not proceed) |
| `disk_root` | the disk check resolves `DockerRootDir`, not `/var/lib` |

## Long test (final gate before the DoW campaign)

The long test is the unattended dress rehearsal for the ~57-run DoW campaign:
a 48h Dynamo-disagg production-duration run at the low (0.30 x ceiling) DoW
rate, followed by a short vLLM hand-off run that proves the serial cycle
(teardown -> VRAM quiescence -> cooldown -> opposite-lifecycle bring-up).

Launch it inside tmux and walk away; `scripts/run_longtest.sh` runs the whole
chain (guard -> calibration -> dry-run -> serial campaign -> verdict) and stops
hard on any failure. Calibration brings the Dynamo stack up via the `deploy/`
scripts, runs `calibrate_rate.py` at fraction 0.30, and tears it down; a
non-`ok` calibration is a HARD STOP (never a silent fallback to an uncalibrated
rate). The mid-run disk watchdog and 30-min heartbeat run inside `launch_cell`.

```bash
tmux new -s longtest
bash scripts/run_longtest.sh            # add --recalibrate to force a fresh sweep
```

Watch progress from a second SSH session (read-only, never touches the run):

```bash
bash scripts/longtest_status.sh
```

The runner tees everything to `campaigns/extension/state/logs/longtest_<UTC>.log`
and is safely re-runnable after an interruption (it detects an existing state
file and `--resume`s). On exit it prints the per-run status, the campaign exit
code (0 ok / 4 interrupted / 5 fatal), and the exact
`validate_extension_run.py` / `validation_check.py` / `aging_trends.py`
commands to run next.
