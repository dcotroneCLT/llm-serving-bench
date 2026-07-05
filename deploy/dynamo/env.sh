#!/usr/bin/env bash
# Shared configuration for the local NVIDIA Dynamo bring-up (extension campaign).
#
# Source this from the serve_*.sh scripts: `source "$(dirname "$0")/env.sh"`.
# Single box, local CLI deploy (no Kubernetes). Pinned vLLM 0.20.1 stack.

set -euo pipefail

# --- Pinned image (vLLM 0.20.1; see engines/dynamo_vllm/image_pin.json) ---
# Pin re-derived after the box gate: 0.16.0 had no Triton release, so the
# three-way intersection is 0.20.1 (Dynamo 1.2.0 + Triton 26.05 + standalone).
DYNAMO_IMAGE="${DYNAMO_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13}"

# --- Model (identical to the Triton and standalone arms) ---
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# --- HuggingFace cache (reuse the campaign cache) ---
HF_CACHE="${HF_CACHE:-$HOME/wosar/hf_cache}"

# --- Networking. Dynamo components discover each other via etcd + NATS on
#     localhost; we run everything on the host network so the discovery and the
#     OpenAI-compatible frontend are reachable at localhost. ---
FRONTEND_HTTP_PORT="${FRONTEND_HTTP_PORT:-8400}"
ETCD_CLIENT_PORT="${ETCD_CLIENT_PORT:-2379}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-2380}"
NATS_PORT="${NATS_PORT:-4222}"

# --- GPU assignment (disaggregated uses two devices on the local L40S box) ---
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
AGG_GPU="${AGG_GPU:-0}"

# --- Fixed worker topology. DO NOT enable the Dynamo planner/autoscaler: launch
#     exactly these counts so the per-component aggregate membership is constant
#     across the 48h window (a moving worker set would inject fake leak steps). ---
N_PREFILL="${N_PREFILL:-1}"
N_DECODE="${N_DECODE:-1}"

# --- NIXL KV-transfer side channel. Every worker is on the host network, so each
#     must bind a DISTINCT side-channel port for the prefill<->decode handshake;
#     otherwise the second worker dies with "Address already in use" on the
#     default 5600. Ports are assigned base, base+1, ... across prefill then
#     decode workers (5600 prefill_1, 5601 decode_1 for the 1+1 topology). ---
NIXL_SIDE_CHANNEL_BASE_PORT="${NIXL_SIDE_CHANNEL_BASE_PORT:-5600}"

# --- Workers run as root (uid 0:0). The shared HF cache is root-owned (written by
#     the root-running standalone arm), and the Dynamo image's default uid 1000
#     cannot traverse/write it. Running as root matches the cache ownership, so no
#     chown/sudo is needed. The frontend does not touch the cache. ---
WORKER_USER="${WORKER_USER:-0:0}"

# --- vLLM disaggregated KV transfer connector (NIXL). Must be passed EXPLICITLY:
#     in this image --connector is deprecated and the mode no longer defaults to
#     nixl, so a prefill worker without it exits 1. Same value on both workers. ---
KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# --- Infra container images (etcd + NATS). Pin to the versions Dynamo 1.0.1
#     documents; bump only deliberately. ---
ETCD_IMAGE="${ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.21}"
NATS_IMAGE="${NATS_IMAGE:-nats:2.11-alpine}"

# Docker container log rotation (SC-2 #3): cap json-file logs so a 48h
# container cannot fill the OS/data-root disk. Used on every docker run.
DOCKER_LOG_OPTS=(--log-opt max-size=50m --log-opt max-file=3)

# Container names (so teardown is deterministic).
ETCD_NAME="${ETCD_NAME:-dyn_etcd}"
NATS_NAME="${NATS_NAME:-dyn_nats}"
DYN_FRONTEND_NAME="${DYN_FRONTEND_NAME:-dyn_frontend}"
DYN_PREFILL_PREFIX="${DYN_PREFILL_PREFIX:-dyn_prefill}"
DYN_DECODE_PREFIX="${DYN_DECODE_PREFIX:-dyn_decode}"
DYN_AGG_WORKER_NAME="${DYN_AGG_WORKER_NAME:-dyn_worker}"

# --- Shared container identity + env for EVERY Dynamo component (workers AND the
#     frontend). One definition so the frontend cannot drift from the workers.
#  - WORKER_USER (0:0, defined above): the shared HF cache is root-owned.
#  - COMMON_ENV: etcd/NATS discovery endpoints + HF_HOME under the mounted cache.
#  - COMMON_MOUNT: the root-owned shared HF cache. ---
COMMON_ENV=(
  -e "ETCD_ENDPOINTS=http://localhost:${ETCD_CLIENT_PORT}"
  -e "NATS_SERVER=nats://localhost:${NATS_PORT}"
  -e "HF_HOME=/root/.cache/huggingface"
)
COMMON_MOUNT=(-v "${HF_CACHE}:/root/.cache/huggingface")

# Start the OpenAI-compatible frontend container. It MUST share the workers'
# identity and cache: --user 0:0 + the root-owned shared HF cache mount. The
# frontend's discovery watcher materializes the model card via hub::from_hf()
# into /root/.cache/huggingface/hub; without the mount + root it fails with
# "Failed to create cache directory ... Permission denied (os error 13)", the
# discovery watcher drops the model, and /v1/models stays empty even though etcd
# holds all the registration keys (this was the "flaky registry" root cause).
# No --gpus: the frontend does not touch the GPU (the "NVIDIA Driver was not
# detected" warning it prints is expected and harmless). Callers poll
# /v1/models for readiness after this returns; this only launches the container.
start_frontend() {
  docker rm -f "$DYN_FRONTEND_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$DYN_FRONTEND_NAME" --network host --user "$WORKER_USER" \
    "${DOCKER_LOG_OPTS[@]}" "${COMMON_ENV[@]}" "${COMMON_MOUNT[@]}" \
    "$DYNAMO_IMAGE" python -m dynamo.frontend --http-port "$FRONTEND_HTTP_PORT" >/dev/null
}

# Engine aggregate name (-> agg_<group>) and the component PGID identity file the
# bring-up records and the monitor (via attach_run/launch_cell) scopes to. Keep
# DYN_ENGINE_GROUP in sync with monitors.components.engine_group in the cell yaml.
DYN_ENGINE_GROUP="${DYN_ENGINE_GROUP:-dynamo}"
COMPONENT_PIDS_FILE="${WOSAR_COMPONENT_PIDS:-$HOME/wosar/dynamo_component_pids.json}"

echo "[dynamo-env] image=$DYNAMO_IMAGE model=$MODEL ctx=$MAX_MODEL_LEN"
echo "[dynamo-env] frontend :$FRONTEND_HTTP_PORT  etcd :$ETCD_CLIENT_PORT  nats :$NATS_PORT"
echo "[dynamo-env] topology: prefill x$N_PREFILL (gpu $PREFILL_GPU), decode x$N_DECODE (gpu $DECODE_GPU)"
