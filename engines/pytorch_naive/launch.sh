#!/usr/bin/env bash
# Launcher for E3 (PyTorch+HF naive). Starts the container detached, waits
# for /readyz, then writes the host PID into <run_dir>/engine.pid so that
# monitoring/proc_monitor.py can attach to it.
#
# Usage:
#   ./launch.sh <run_dir>
#
# Env knobs (with defaults shown):
#   IMAGE=pytorch_naive:wosar2026
#   CONTAINER_NAME=pytorch_naive_e3
#   GPU_INDEX=2                       host GPU index (paper protocol: 2)
#   PORT=8002                         host-side port mapped to container :8000
#   HF_CACHE=$HOME/wosar/hf_cache
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
#   MAX_MODEL_LEN=8192
#   DTYPE=bfloat16
#   READYZ_TIMEOUT_S=600              cold loads of a 7B model can take a while
#
# Notes:
#   - We rely on `docker inspect -f '{{.State.Pid}}'` returning the host PID
#     of the container's PID 1. The Dockerfile uses the exec form of CMD
#     so PID 1 is uvicorn itself (not a shell wrapper).
#   - At the end we run nvidia-smi and check that the engine PID appears
#     among the compute apps of GPU $GPU_INDEX, as a sanity gate before
#     the long run starts. A mismatch typically means the container
#     bound to the wrong device or the model failed to allocate VRAM.

set -euo pipefail

RUN_DIR="${1:?usage: launch.sh <run_dir>}"
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)

HERE=$(cd "$(dirname "$0")" && pwd)

IMAGE="${IMAGE:-pytorch_naive:wosar2026}"
CONTAINER_NAME="${CONTAINER_NAME:-pytorch_naive_e3}"
GPU_INDEX="${GPU_INDEX:-2}"
PORT="${PORT:-8002}"
HF_CACHE="${HF_CACHE:-$HOME/wosar/hf_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
READYZ_TIMEOUT_S="${READYZ_TIMEOUT_S:-600}"

mkdir -p "$HF_CACHE"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[launch] building $IMAGE from $HERE"
  docker build -t "$IMAGE" "$HERE"
fi

# Tear down any stale container with the same name. We do not auto-remove
# containers from previous runs that have a different name; only the
# matching one.
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "[launch] removing existing container $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "[launch] docker run gpu=$GPU_INDEX port=$PORT image=$IMAGE"
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_INDEX\"" \
  --shm-size=16g \
  --ipc=host \
  -p "$PORT":8000 \
  -v "$HF_CACHE":/cache/huggingface \
  -e MODEL_NAME="$MODEL_NAME" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e DTYPE="$DTYPE" \
  -e GPU_DEVICE="cuda:0" \
  "$IMAGE" >/dev/null

# Give Docker a moment to assign the container its host PID.
sleep 2

PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER_NAME")
if [ -z "$PID" ] || [ "$PID" = "0" ]; then
  echo "[launch] ERROR: container $CONTAINER_NAME is not running"
  docker logs --tail 100 "$CONTAINER_NAME" || true
  exit 1
fi

echo "$PID" > "$RUN_DIR/engine.pid"
echo "[launch] container=$CONTAINER_NAME host_pid=$PID port=$PORT"
echo "[launch] wrote $RUN_DIR/engine.pid"

echo "[launch] waiting up to ${READYZ_TIMEOUT_S}s for /readyz on :$PORT ..."
ready=0
for i in $(seq 1 "$READYZ_TIMEOUT_S"); do
  if curl -sf "http://localhost:$PORT/readyz" >/dev/null 2>&1; then
    echo "[launch] /readyz OK after ${i}s"
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "[launch] ERROR: container exited during startup"
    docker logs --tail 200 "$CONTAINER_NAME" || true
    exit 1
  fi
  sleep 1
done
if [ "$ready" -eq 0 ]; then
  echo "[launch] ERROR: /readyz did not come up in ${READYZ_TIMEOUT_S}s"
  docker logs --tail 200 "$CONTAINER_NAME" || true
  exit 1
fi

echo "[launch] nvidia-smi compute apps for GPU $GPU_INDEX:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv -i "$GPU_INDEX" || true

if nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_INDEX" \
    | grep -Fxq "$PID"; then
  echo "[launch] OK: engine PID $PID is using GPU $GPU_INDEX"
else
  echo "[launch] WARNING: engine PID $PID NOT found on GPU $GPU_INDEX. Inspect before starting the run."
fi

echo "[launch] done. To stop: docker rm -f $CONTAINER_NAME"
