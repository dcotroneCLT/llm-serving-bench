# Monitoring agents

Three independent samplers plus an orchestrator. Designed to be run on
the same host as the serving engine, with output written to a per-run
directory.

## Files

- `_common.py` shared writer, watchdog, shutdown handling, steady sampler
- `gpu_monitor.py` NVML-based GPU metrics, default 1 Hz
- `proc_monitor.py` psutil-based per-process metrics, default 5 s
- `system_monitor.py` system-wide metrics, default 5 s
- `run_monitors.py` orchestrator that spawns all three and writes manifest
- `find_engine_pid.py` dynamic engine-PID resolver for containerized engines
- `RUN_AGING.md` 4-tmux-session launch procedure for 24-hour aging runs

## Prerequisites

```
pip install psutil nvidia-ml-py
```

`psutil` for process and system metrics. `nvidia-ml-py` (the official
NVIDIA Python bindings, package name on PyPI) for NVML access. No CUDA
toolkit needed for monitoring; the NVIDIA driver alone is sufficient.

For per-process I/O counters and FD counts, the monitor must run as the
same user as the engine, or as root. Otherwise some fields will be
empty.

## Quick smoke test

Standalone run of the GPU monitor for 10 seconds on GPU 0, writing to
`/tmp/mon`:

```
python gpu_monitor.py --gpu-index 0 --output-dir /tmp/mon --period-seconds 1
# in another terminal: send SIGTERM after 10 s
```

The system monitor needs no privileges and is the easiest to validate:

```
python system_monitor.py --output-dir /tmp/mon --period-seconds 1
```

After 30 s, kill it with Ctrl-C and inspect:

```
ls -la /tmp/mon
head /tmp/mon/system_000000.csv
```

You should see one CSV file per ~60 s of run time, each with a header
row and one data row per sample.

## Full run

The orchestrator handles a real run. The serving engine must be started
separately by the caller; once it is up, write its PID into a pidfile,
then launch:

```
python run_monitors.py \
    --run-id pilot_vllm_replicate1 \
    --runs-root ../runs \
    --gpu-index 1 \
    --pidfile ../runs/pilot_vllm_replicate1/engine.pid \
    --duration-seconds 7200 \
    --label-engine vllm_standalone
```

### Containerized engines: use find_engine_pid.py

For any engine running inside a Docker container, do **not** capture
the PID by hand with `docker top` and write it into `engine.pid`
statically. That breaks the moment the engine respawns a worker, and
proc_monitor will record `process_alive=False` for the remainder of
the run — this is exactly what happened to the 2026-05-12/13 aging
runs.

Instead, run `find_engine_pid.py` alongside `run_monitors.py`. It
inspects the container's descendant tree every N seconds and rewrites
`engine.pid` whenever the engine worker's PID changes; proc_monitor
re-reads the pidfile on each sample and follows the new PID
transparently.

See `RUN_AGING.md` for the full 4-tmux-session launch procedure
(engine container, pid tracker, host monitors, client) and a table of
per-engine `--process-pattern` regexes.

Output structure:

```
runs/pilot_vllm_replicate1/
  manifest.json
  gpu1_000000.csv ... gpu1_NNNNNN.csv
  vllm_standalone_000000.csv ... vllm_standalone_NNNNNN.csv
  system_000000.csv ... system_NNNNNN.csv
  logs/
    gpu_monitor.log
    proc_monitor.log
    system_monitor.log
```

## Robustness notes

Each monitor is independent. If one crashes the others continue, and the
orchestrator records the failure and shuts the rest down cleanly so the
run can be flagged in analysis.

CSV files rotate every 60 s by default. Crash loss is bounded by the
last unflushed sample plus the rotation interval.

Watchdog timeout (default 30 s) defends against NVML or psutil
deadlocks. A timed-out sample is still written to CSV with empty
measurement fields and `_overrun` set to True; the time series remains
aligned in time.

## Pre-flight before a long run

```
df -h ../runs        # disk space
ulimit -n            # FD limit at least 1024
free -g              # available memory
nvidia-smi           # GPU present and idle
```

Two-hour smoke run before each 24-hour campaign run is recommended.
