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
| Standalone | `vllm/vllm-openai:v0.20.1-cu130` |
| vLLM | 0.20.1 (identical across all three) |
| CUDA | 13.x (matches host driver 580.x) |
| Model | Qwen/Qwen2.5-7B-Instruct, ctx 8192, BF16 |

## 0. Verify the vLLM version (the whole point of the pin)

The ground truth that all three systems share vLLM 0.20.1 (remote release notes
were wrong once, so pip show is authoritative). NOTE: vllm-openai's entrypoint is
the API server, so it needs `--entrypoint pip`:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 pip show vllm | grep -i '^Version'   # 0.20.1
docker run --rm nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3 pip show vllm | grep -i '^Version'    # 0.20.1 (gate-critical)
docker run --rm --entrypoint pip vllm/vllm-openai:v0.20.1-cu130 show vllm | grep -i '^Version'          # 0.20.1
# record each digest into the matching engines/*/image_pin*.json:
for img in nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 \
           nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3 \
           vllm/vllm-openai:v0.20.1-cu130; do
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
curl -sf http://localhost:8400/health && echo OK
```

## 2b. Disaggregated (2 GPU) — the campaign topology

```bash
N_PREFILL=1 N_DECODE=1 PREFILL_GPU=0 DECODE_GPU=1 bash deploy/dynamo/serve_disaggregated.sh
curl -sf http://localhost:8400/health && echo OK
```

Fixed topology, planner/autoscaler intentionally not launched, so the component
set is constant for the whole run (a moving worker set would inject fake leak
steps into the aggregate).

## 3. Freeze the component regexes against the REAL process tree

This closes "WS2 against the real process tree". After 2b is healthy:

```bash
ps -eo pid,cmd | grep -E 'dynamo.frontend|dynamo.vllm' | grep -v grep
```

Confirm the cmdlines and freeze the `pattern` / `require` / `exclude` regexes in
`campaigns/extension/cells/val_dynamo_disagg.yaml` under `monitors.components`
so they match reality (decode = `dynamo.vllm` AND NOT `--is-prefill-worker`).

## 4. First-bring-up flag check

The component commands in `serve_*.sh` encode the researched Dynamo 1.0.1 CLI
(`python -m dynamo.frontend`, `python -m dynamo.vllm [--is-prefill-worker]`).
Confirm the exact flags once:

```bash
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 python -m dynamo.vllm --help
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13 python -m dynamo.frontend --help
```

Adjust env var names for etcd/NATS endpoints if `--help` shows different ones.

## 5. Teardown

```bash
bash deploy/dynamo/serve_down.sh     # engine components
bash deploy/dynamo/infra_down.sh     # etcd + nats
```

## Validation run (STEP 1 gate)

Drive monitors + client + manifest against the running frontend with
`scripts/attach_run.py` (it does NOT manage the engine lifecycle — that is the
later launch_cell/campaign work). See `scripts/attach_run.py --help` and the
`campaigns/extension/cells/val_*.yaml` cells.
