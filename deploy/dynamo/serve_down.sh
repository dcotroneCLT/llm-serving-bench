#!/usr/bin/env bash
# Tear down all Dynamo engine components (frontend + prefill + decode + agg
# worker). Leaves etcd/NATS up; run infra_down.sh to remove those too.
set -uo pipefail
source "$(dirname "$0")/env.sh"
names=("$DYN_FRONTEND_NAME" "$DYN_AGG_WORKER_NAME")
for i in $(seq 1 "${N_PREFILL:-4}"); do names+=("${DYN_PREFILL_PREFIX}_${i}"); done
for i in $(seq 1 "${N_DECODE:-4}");  do names+=("${DYN_DECODE_PREFIX}_${i}"); done
docker rm -f "${names[@]}" >/dev/null 2>&1 || true
echo "[dynamo] engine components removed (etcd/nats still up; infra_down.sh to remove)"
