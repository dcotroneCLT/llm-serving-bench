# Running a Containerized Aging Experiment

This document codifies the launch procedure for any aging run that
targets a *containerized* serving engine. The production WoSAR 2026 path
is `scripts/campaign.py` -> `scripts/launch_cell.py`; this file is the
manual/debug runbook for operating the same components by hand.

The dynamic PID-tracking requirement comes from the 2026-05-12/13
ablation runs, where a static PID capture broke the per-process
indicators for the rest of the run.

The manual procedure uses four independent tmux sessions so each
component can be inspected or restarted in isolation.

For production, prefer:

```
python3 scripts/campaign.py --campaign-yaml campaigns/wosar2026/campaign.yaml --start
python3 scripts/campaign.py --campaign-yaml campaigns/wosar2026/campaign.yaml --resume
```

## Components

| Session | Role             | Script                                  |
| ------- | ---------------- | --------------------------------------- |
| 1       | engine container | `docker run …` (engine-specific)        |
| 2       | pid tracker      | `monitoring/find_engine_pid.py`         |
| 3       | host monitors    | `monitoring/run_monitors.py`            |
| 4       | client load      | `client/run_client.py`                  |

Sessions 2 and 3 share the same `<run_dir>/engine.pid`: session 2 keeps
it updated to point at the live engine worker; session 3 (specifically
`proc_monitor.py`) re-reads it every sample.

## Engine-specific process patterns

The `--process-pattern` is a regex matched against the joined cmdline of
each descendant of the container's init process. Use the pattern below
that matches the engine generation you are testing.

| Engine                                         | Pattern                            | Notes                                              |
| ---------------------------------------------- | ---------------------------------- | -------------------------------------------------- |
| Triton 25.09 + vLLM V1 (`a2`)                  | `EngineCore`                       | V1's dedicated async engine subprocess             |
| vLLM v0.7.3 standalone (`a1`)                  | `spawn_main`                       | Multiprocessing worker hosting V0                  |
| Triton + in-process V0 (`e2`)                  | `triton_python_backend_stub`       | V0 runs inside the Triton Python backend           |

If you add a new engine generation, identify the pattern by hand once:

```
docker exec -it <engine> ps -ef
# or, from the host:
pstree -p $(docker inspect --format '{{.State.Pid}}' <engine>)
```

Pick the deepest descendant doing real work. Avoid matching the
container init (PID 1 inside the container) or short-lived wrappers.

## Launch sequence

Below, replace placeholders with the values for the run you are
launching. `<run_dir>` is `<runs_root>/<run_id>` and is created
automatically by `run_monitors.py`.

### Session 1 — engine container

The engine is launched by whatever procedure the engine repo
documents. Two reference invocations:

```
# Triton 25.09 with vLLM V1
docker run -d --name triton_v1 --gpus device=1 \
  --shm-size=4g -e VLLM_USE_V1=1 \
  -p 8001:8000 \
  -v <model_repo>:/models \
  nvcr.io/nvidia/tritonserver:25.09-vllm-python-py3 \
  tritonserver --model-repository=/models
```

```
# vLLM 0.7.3 standalone (V0 engine)
docker run -d --name vllm_v0 --gpus device=0 \
  --shm-size=4g \
  -p 8000:8000 \
  vllm/vllm-openai:v0.7.3 \
  --model <hf_id> --port 8000
```

Wait for the engine to finish loading the model and to respond to a
single warm-up request before starting the other sessions, otherwise
the pid tracker may publish the loader PID and not the steady-state
worker PID.

### Session 2 — pid tracker

```
mkdir -p <run_dir>
python3 monitoring/find_engine_pid.py \
  --container-name <engine_container> \
  --process-pattern <pattern_from_table_above> \
  --pidfile <run_dir>/engine.pid \
  --poll-seconds 30 \
  --duration-seconds 129600
```

The script writes each resolved PID atomically to `<run_dir>/engine.pid`
and logs to stderr only when the PID changes. Keep this session
visible during the first hour so you can confirm at least one
resolution event.

### Session 3 — host monitors

```
python3 monitoring/run_monitors.py \
  --run-id <run_id> \
  --runs-root <runs_root> \
  --gpu-index <gpu_index> \
  --pidfile <run_dir>/engine.pid \
  --duration-seconds 129600 \
  --label-engine <engine_label>
```

`<gpu_index>` is the same GPU passed to the container via
`--gpus device=N`. `<engine_label>` is the base name for the
proc-monitor CSV (e.g. `triton_v1`, `vllm_v0`).

### Session 4 — client

```
python3 client/run_client.py \
  --config <config.yaml> \
  --output-dir <run_dir>/client \
  --duration-seconds 129600
```

The client config specifies target RPS, prompt corpus, and protocol.

For WoSAR 2026, the same duration value is stored in every production
cell YAML as `duration_s: 129600`, and the analysis warmup discard is
stored as `warmup_discard_s: 3600`.

## Validation, first 5 minutes

Before walking away from the run, verify:

1. `<run_dir>/engine.pid` exists and contains a non-zero PID.
2. `head <run_dir>/<engine_label>_000000.csv` shows
   `process_alive=True` and non-zero `rss_bytes`.
3. The pid tracker's stderr shows at least one
   `[find_engine_pid] resolved pid=… cmdline=…` line.
4. `nvidia-smi --id=<gpu_index>` reports the engine using GPU memory.
5. The client log shows non-zero successful requests within the first
   minute.

If item 2 fails (process_alive=False after 5 minutes), the most likely
causes are:

- `--process-pattern` does not match any descendant (check session 2 log).
- The proc monitor cannot read `/proc/<pid>` because it runs under a
  different user than the engine. Run both as the same user, or as a
  user that is a member of the `docker` group.
- The container is using PID namespace isolation in a way that hides
  the worker from the host `/proc`. This is rare with the standard
  `docker run` invocations above but can happen with custom
  `--security-opt` flags.

## Production Validation

Before a real 36h slot, run the campaign smoke gate:

```
bash scripts/smoke_test.sh campaigns/wosar2026/cells/<cell_id>.yaml
```

After a completed run, check the self-contained run directory:

```
python3 analysis/validation_check.py --run-dir <run_dir>
python3 analysis/aging_trends.py <run_dir> --alpha 0.10 --downsample-seconds 60
```

For the step-wise pattern panel (corr, K_trim, steps>1MB/h) across all
runs under a runs root:

```
python3 analysis/stepness.py --logs-root <runs_root> --warmup-s 3600
```

For campaign figures:

```
python3 analysis/plot_rss_2x2.py --campaign-yaml campaigns/wosar2026/campaign.yaml --runs-root <runs_root> --replicas all
```

## Why a static PID is not enough

A PID captured once at run start works only if the engine's worker
process never exits. In practice:

- vLLM V1 uses a separate `EngineCore` subprocess that the parent can
  respawn on internal recovery paths.
- vLLM V0 with the multiprocessing backend hosts the engine in a
  `spawn_main` subprocess; the parent supervisor can restart it.
- Triton's Python backend can reload a model on configuration change,
  changing the stub PID.

Any of these events will leave a static pidfile pointing at a dead
PID, after which `proc_monitor.py` records `process_alive=False`
forever. The dynamic tracker fixes this without changing the
proc-monitor contract.
