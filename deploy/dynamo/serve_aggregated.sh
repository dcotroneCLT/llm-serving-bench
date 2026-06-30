#!/usr/bin/env bash
# Bring up NVIDIA Dynamo in AGGREGATED mode (frontend + a single worker that does
# both prefill and decode) on one GPU, vLLM 0.20.1. Used as the simpler bring-up
# to de-risk the stack before the disaggregated path.
#
# Prereq: bash deploy/dynamo/infra_up.sh
# Usage:  bash deploy/dynamo/serve_aggregated.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"

COMMON_ENV=(
  -e "ETCD_ENDPOINTS=http://localhost:${ETCD_CLIENT_PORT}"
  -e "NATS_SERVER=nats://localhost:${NATS_PORT}"
  -e "HF_HOME=/root/.cache/huggingface"
)

docker rm -f "$DYN_FRONTEND_NAME" "$DYN_AGG_WORKER_NAME" >/dev/null 2>&1 || true

# Worker FIRST, frontend LAST (the frontend reads the model card at startup; if it
# starts before the worker registers, /v1/models is empty and inference 404s).
# --user 0:0: the shared HF cache is root-owned, the image's default uid 1000 can't
# write it. Aggregated mode needs no --kv-transfer-config / NIXL side channel.
echo "[dynamo] worker on gpu $AGG_GPU"
docker run -d --name "$DYN_AGG_WORKER_NAME" --network host --user "$WORKER_USER" \
  --gpus "\"device=${AGG_GPU}\"" "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "$DYNAMO_IMAGE" \
  python -m dynamo.vllm \
    --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL"

echo "[dynamo] waiting for the worker to register the model (model load ~1-3 min)..."
for _ in $(seq 1 60); do
  if docker logs "$DYN_AGG_WORKER_NAME" 2>&1 | grep -q "Registered base model"; then
    echo "[dynamo]   $DYN_AGG_WORKER_NAME registered"; break
  fi
  sleep 5
done

echo "[dynamo] frontend on :$FRONTEND_HTTP_PORT"
docker run -d --name "$DYN_FRONTEND_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" \
  "$DYNAMO_IMAGE" \
  python -m dynamo.frontend --http-port "$FRONTEND_HTTP_PORT"

echo "[dynamo] waiting for /v1/models to list the model..."
for _ in $(seq 1 24); do
  if curl -sf "http://localhost:${FRONTEND_HTTP_PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "[dynamo] model is served"; break
  fi
  sleep 5
done

echo "[dynamo] launched: 1 worker + frontend."
echo "[dynamo] readiness: curl -sf http://localhost:${FRONTEND_HTTP_PORT}/v1/models"
