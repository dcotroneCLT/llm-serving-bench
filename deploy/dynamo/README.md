# NVIDIA Dynamo local bring-up (extension campaign)

Local CLI deploy (no Kubernetes), single L40S box. Part of the 3-system
extension campaign that holds **vLLM 0.16.0 identical** across Dynamo,
Triton+vLLM, and standalone vLLM (see the pin files in `engines/*/image_pin*`).

The exact pinned image and component commands are encoded in `env.sh` and the
`serve_*.sh` scripts. Everything runs on the host network so the
OpenAI-compatible frontend and the etcd/NATS discovery are reachable on
localhost, and so the host `ps`/`/proc` sees the `python -m dynamo.*` processes
(which is what the per-component monitor matches).

## Pinned stack

| Piece | Value |
|---|---|
| Image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13` |
| Dynamo | 1.0.1 (stable) |
| vLLM | 0.16.0 |
| NIXL | 0.10.1 |
| CUDA | 13.0 (matches host driver 580.x) |
| Model | Qwen/Qwen2.5-7B-Instruct, ctx 8192, BF16 |

## 0. Verify the vLLM version (the whole point of the pin)

The ground truth that all three systems share vLLM 0.16.0:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13 pip show vllm | grep Version   # 0.16.0
# record the digest into engines/dynamo_vllm/image_pin.json:
docker inspect --format '{{index .RepoDigests 0}}' nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13
```

Do the same `pip show vllm` for `nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3`
and `vllm/vllm-openai:v0.16.0-cu130` and fill in all three pin files.

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
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13 python -m dynamo.vllm --help
docker run --rm nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-cuda13 python -m dynamo.frontend --help
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
