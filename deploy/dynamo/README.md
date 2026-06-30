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

## 3. Freeze the component regexes against the REAL process tree

This closes "WS2 against the real process tree". After 2b is healthy:

```bash
ps -eo pid,cmd | grep -E 'dynamo.frontend|dynamo.vllm' | grep -v grep
```

Confirm the cmdlines and freeze the `pattern` / `require` / `exclude` regexes in
`campaigns/extension/cells/val_dynamo_disagg.yaml` under `monitors.components`
so they match reality (prefill = `dynamo.vllm --disaggregation-mode prefill`,
decode = `dynamo.vllm --disaggregation-mode decode`).

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

## Validation run (STEP 1 gate)

Drive monitors + client + manifest against the running frontend with
`scripts/attach_run.py` (it does NOT manage the engine lifecycle — that is the
later launch_cell/campaign work). See `scripts/attach_run.py --help` and the
`campaigns/extension/cells/val_*.yaml` cells.
