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
set -euo pipefail
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
  registered=0
  for _ in $(seq 1 60); do
    if docker logs "$c" 2>&1 | grep -q "Registered base model"; then
      echo "[dynamo]   $c registered"; registered=1; break
    fi
    # Fail fast (and bound the wait) if the worker container has already exited:
    # a bad model / startup crash would otherwise stall here for the full timeout.
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null || echo false)" != "true" ]; then
      echo "[dynamo] FATAL: worker $c exited before registering the model; check 'docker logs $c'." >&2
      exit 1
    fi
    sleep 5
  done
  if [ "$registered" != 1 ]; then
    echo "[dynamo] FATAL: worker $c did not register within the timeout; check 'docker logs $c'." >&2
    exit 1
  fi
done

# --- Frontend (HTTP ingress + in-process KV router; no separate router PID) ---
# The frontend snapshots the model registry at startup and does NOT pick up a
# model that finalizes afterward, even minutes later (observed: a frontend
# started right after the workers log "Registered base model" serves an empty
# /v1/models for 2+ min, while a plain restart once the workers are fully ready
# serves immediately). So the reliable readiness signal is to (re)start the
# frontend and check /v1/models; if empty, restart and retry.
# Worker readiness lags the "Registered base model" log line: a frontend started
# in that early window snapshots an empty registry and never recovers, while a
# frontend started once the workers are fully discoverable serves /v1/models in
# ~5s (measured). So: SETTLE briefly, then keep a frontend up FRONTEND_POLL_TRIES
# x 5s per attempt (each attempt gives a fresh frontend enough uninterrupted time
# to discover), restarting up to FRONTEND_ATTEMPTS times. Generous + env-tunable.
sleep "${FRONTEND_SETTLE_S:-20}"
echo "[dynamo] frontend on :$FRONTEND_HTTP_PORT (settle=${FRONTEND_SETTLE_S:-20}s attempts=${FRONTEND_ATTEMPTS:-8} poll_tries=${FRONTEND_POLL_TRIES:-12})"
served=0
for attempt in $(seq 1 "${FRONTEND_ATTEMPTS:-8}"); do
  docker rm -f "$DYN_FRONTEND_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$DYN_FRONTEND_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" \
    "$DYNAMO_IMAGE" \
    python -m dynamo.frontend --http-port "$FRONTEND_HTTP_PORT" >/dev/null
  for _ in $(seq 1 "${FRONTEND_POLL_TRIES:-12}"); do
    if curl -sf "http://localhost:${FRONTEND_HTTP_PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
      served=1; break
    fi
    sleep 5
  done
  [ "$served" = 1 ] && { echo "[dynamo] model is served (frontend attempt $attempt)"; break; }
  echo "[dynamo] /v1/models still empty after ~$(( ${FRONTEND_POLL_TRIES:-12} * 5 ))s; restarting frontend (attempt $attempt)..."
done
if [ "$served" != 1 ]; then
  echo "[dynamo] FATAL: model not served after frontend restarts; workers did not register. " \
       "Check 'docker logs dyn_prefill_1 / dyn_decode_1'. Not recording identity; aborting." >&2
  exit 1
fi

# Record each component's process-group identity so the monitor scopes to EXACTLY
# these PGIDs (not a host-wide cmdline regex). One --component per label, with all
# instance containers of that label grouped together.
PREFILL_CONTAINERS=(); for i in $(seq 1 "$N_PREFILL"); do PREFILL_CONTAINERS+=("${DYN_PREFILL_PREFIX}_${i}"); done
DECODE_CONTAINERS=();  for i in $(seq 1 "$N_DECODE");  do DECODE_CONTAINERS+=("${DYN_DECODE_PREFIX}_${i}");  done
echo "[dynamo] recording component PGID identity -> $COMPONENT_PIDS_FILE"
python3 "$(dirname "$0")/record_component_pids.py" --engine-group "${DYN_ENGINE_GROUP:-dynamo}" \
  --out "$COMPONENT_PIDS_FILE" \
  --component dynamo_frontend "$DYN_FRONTEND_NAME" \
  --component dynamo_prefill  "${PREFILL_CONTAINERS[@]}" \
  --component dynamo_decode   "${DECODE_CONTAINERS[@]}" \
  --component etcd            "$ETCD_NAME" \
  --component nats            "$NATS_NAME"

echo "[dynamo] launched: ${N_PREFILL} prefill + ${N_DECODE} decode + frontend (planner/autoscaler NOT started)."
echo "[dynamo] readiness: curl -sf http://localhost:${FRONTEND_HTTP_PORT}/v1/models"
echo "[dynamo] component identity: $COMPONENT_PIDS_FILE"
echo "[dynamo] host process tree (freeze these cmdlines into the cell yaml regexes):"
echo "         ps -eo pid,cmd | grep -E 'dynamo.frontend|dynamo.vllm' | grep -v grep"
