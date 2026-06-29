#!/usr/bin/env bash
# Bring up NVIDIA Dynamo in DISAGGREGATED mode (prefill workers + decode workers
# + frontend with in-process KV router) on the local L40S box, vLLM 0.16.0.
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

# --- Frontend (HTTP ingress + in-process KV router; no separate router PID) ---
echo "[dynamo] frontend on :$FRONTEND_HTTP_PORT"
docker run -d --name "$DYN_FRONTEND_NAME" --network host "${COMMON_ENV[@]}" \
  "$DYNAMO_IMAGE" \
  python -m dynamo.frontend --http-port "$FRONTEND_HTTP_PORT"

# --- Prefill workers (GPU PREFILL_GPU); marked with --is-prefill-worker ---
for i in $(seq 1 "$N_PREFILL"); do
  echo "[dynamo] prefill #$i on gpu $PREFILL_GPU"
  docker run -d --name "${DYN_PREFILL_PREFIX}_${i}" --network host \
    --gpus "\"device=${PREFILL_GPU}\"" "${COMMON_ENV[@]}" "${COMMON_MOUNT[@]}" \
    "$DYNAMO_IMAGE" \
    python -m dynamo.vllm \
      --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --is-prefill-worker
done

# --- Decode workers (GPU DECODE_GPU); default vLLM behavior, NO prefill flag ---
for i in $(seq 1 "$N_DECODE"); do
  echo "[dynamo] decode #$i on gpu $DECODE_GPU"
  docker run -d --name "${DYN_DECODE_PREFIX}_${i}" --network host \
    --gpus "\"device=${DECODE_GPU}\"" "${COMMON_ENV[@]}" "${COMMON_MOUNT[@]}" \
    "$DYNAMO_IMAGE" \
    python -m dynamo.vllm \
      --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL"
done

echo "[dynamo] launched: frontend + ${N_PREFILL} prefill + ${N_DECODE} decode (planner/autoscaler NOT started)."
echo "[dynamo] readiness: curl -sf http://localhost:${FRONTEND_HTTP_PORT}/health"
echo "[dynamo] host process tree (freeze these cmdlines into the cell yaml regexes):"
echo "         ps -eo pid,cmd | grep -E 'dynamo.frontend|dynamo.vllm' | grep -v grep"
