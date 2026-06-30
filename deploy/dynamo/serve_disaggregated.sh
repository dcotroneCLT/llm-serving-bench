#!/usr/bin/env bash
# Bring up NVIDIA Dynamo in DISAGGREGATED mode (prefill workers + decode workers
# + frontend with in-process KV router) on the local L40S box, vLLM 0.20.1.
#
# Topology is FIXED (N_PREFILL prefill, N_DECODE decode) and the planner /
# autoscaler is intentionally NOT launched, so the component membership the
# per-component monitor sees is constant over the run.
#
# Each component runs in its own container on the host network, GPU-pinned, so:
#   - the host sees distinct `python -m dynamo.*` processes (the monitor's
#     cmdline regexes match them);
#   - GPU assignment is explicit per worker.
#
# Prereq: bash deploy/dynamo/infra_up.sh   (etcd + NATS)
#
# Usage: bash deploy/dynamo/serve_disaggregated.sh
#
# FIRST BRING-UP: confirm the exact flags against `docker run --rm $DYNAMO_IMAGE
# python -m dynamo.vllm --help` and `python -m dynamo.frontend --help`, then
# freeze (a) the flags here and (b) the realized `ps` cmdlines into the cell
# yaml component regexes (campaigns/extension/cells/val_dynamo_disagg.yaml).
set -uo pipefail
source "$(dirname "$0")/env.sh"

COMMON_ENV=(
  -e "ETCD_ENDPOINTS=http://localhost:${ETCD_CLIENT_PORT}"
  -e "NATS_SERVER=nats://localhost:${NATS_PORT}"
  -e "HF_HOME=/root/.cache/huggingface"
)
COMMON_MOUNT=(-v "${HF_CACHE}:/root/.cache/huggingface")

# Clean any stale components.
docker rm -f "$DYN_FRONTEND_NAME" >/dev/null 2>&1 || true
for i in $(seq 1 "$N_PREFILL"); do docker rm -f "${DYN_PREFILL_PREFIX}_${i}" >/dev/null 2>&1 || true; done
for i in $(seq 1 "$N_DECODE");  do docker rm -f "${DYN_DECODE_PREFIX}_${i}"  >/dev/null 2>&1 || true; done

# ORDER: workers FIRST, frontend LAST. The frontend reads the model card at
# startup to populate /v1/models; if it starts before a worker has registered
# the model, /v1/models stays empty and every inference 404s. So we launch the
# workers, wait for each to register, then launch the frontend.
#
# Each worker:
#   --user 0:0                      root, to write the root-owned shared HF cache
#   --disaggregation-mode {prefill,decode}   explicit (mode defaults to 'agg')
#   --kv-transfer-config <NIXL>     explicit (--connector deprecated, no default)
#   VLLM_NIXL_SIDE_CHANNEL_PORT=N   distinct per worker (host-network port clash)

WORKER_NAMES=()
nixl_port="$NIXL_SIDE_CHANNEL_BASE_PORT"

# --- Prefill workers (GPU PREFILL_GPU); --disaggregation-mode=prefill ---
for i in $(seq 1 "$N_PREFILL"); do
  echo "[dynamo] prefill #$i on gpu $PREFILL_GPU (nixl port $nixl_port)"
  docker run -d --name "${DYN_PREFILL_PREFIX}_${i}" --network host --user "$WORKER_USER" \
    --gpus "\"device=${PREFILL_GPU}\"" "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" "${COMMON_MOUNT[@]}" \
    -e "VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port}" \
    "$DYNAMO_IMAGE" \
    python -m dynamo.vllm \
      --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --disaggregation-mode prefill \
      --kv-transfer-config "$KV_TRANSFER_CONFIG"
  WORKER_NAMES+=("${DYN_PREFILL_PREFIX}_${i}")
  nixl_port=$((nixl_port + 1))
done

# --- Decode workers (GPU DECODE_GPU); --disaggregation-mode=decode ---
# (Decode registers in Dynamo as the "backend" component; the process cmdline
# still carries --disaggregation-mode decode, which is what the monitor matches.)
for i in $(seq 1 "$N_DECODE"); do
  echo "[dynamo] decode #$i on gpu $DECODE_GPU (nixl port $nixl_port)"
  docker run -d --name "${DYN_DECODE_PREFIX}_${i}" --network host --user "$WORKER_USER" \
    --gpus "\"device=${DECODE_GPU}\"" "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" "${COMMON_MOUNT[@]}" \
    -e "VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port}" \
    "$DYNAMO_IMAGE" \
    python -m dynamo.vllm \
      --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --disaggregation-mode decode \
      --kv-transfer-config "$KV_TRANSFER_CONFIG"
  WORKER_NAMES+=("${DYN_DECODE_PREFIX}_${i}")
  nixl_port=$((nixl_port + 1))
done

# Wait for every worker to register its model card (logs "Registered base model")
# before starting the frontend. ~5 min cap covers cold model load.
echo "[dynamo] waiting for workers to register the model (model load ~1-3 min)..."
for c in "${WORKER_NAMES[@]}"; do
  for _ in $(seq 1 60); do
    if docker logs "$c" 2>&1 | grep -q "Registered base model"; then
      echo "[dynamo]   $c registered"; break
    fi
    sleep 5
  done
done

# --- Frontend (HTTP ingress + in-process KV router; no separate router PID) ---
echo "[dynamo] frontend on :$FRONTEND_HTTP_PORT"
docker run -d --name "$DYN_FRONTEND_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" \
  "$DYNAMO_IMAGE" \
  python -m dynamo.frontend --http-port "$FRONTEND_HTTP_PORT"

# Confirm the model is actually served before declaring the stack up.
echo "[dynamo] waiting for /v1/models to list the model..."
for _ in $(seq 1 24); do
  if curl -sf "http://localhost:${FRONTEND_HTTP_PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "[dynamo] model is served"; break
  fi
  sleep 5
done

echo "[dynamo] launched: ${N_PREFILL} prefill + ${N_DECODE} decode + frontend (planner/autoscaler NOT started)."
echo "[dynamo] readiness: curl -sf http://localhost:${FRONTEND_HTTP_PORT}/v1/models"
echo "[dynamo] host process tree (freeze these cmdlines into the cell yaml regexes):"
echo "         ps -eo pid,cmd | grep -E 'dynamo.frontend|dynamo.vllm' | grep -v grep"
