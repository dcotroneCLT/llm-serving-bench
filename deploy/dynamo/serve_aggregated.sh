#!/usr/bin/env bash
# Bring up NVIDIA Dynamo in AGGREGATED mode (frontend + a single worker that does
# both prefill and decode) on one GPU, vLLM 0.16.0. Used as the simpler bring-up
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

echo "[dynamo] frontend on :$FRONTEND_HTTP_PORT"
docker run -d --name "$DYN_FRONTEND_NAME" --network host "${COMMON_ENV[@]}" \
  "$DYNAMO_IMAGE" \
  python -m dynamo.frontend --http-port "$FRONTEND_HTTP_PORT"

echo "[dynamo] worker on gpu $AGG_GPU"
docker run -d --name "$DYN_AGG_WORKER_NAME" --network host \
  --gpus "\"device=${AGG_GPU}\"" "${COMMON_ENV[@]}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "$DYNAMO_IMAGE" \
  python -m dynamo.vllm \
    --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL"

echo "[dynamo] launched: frontend + 1 worker."
echo "[dynamo] readiness: curl -sf http://localhost:${FRONTEND_HTTP_PORT}/health"
