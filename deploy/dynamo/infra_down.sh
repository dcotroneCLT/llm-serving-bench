#!/usr/bin/env bash
# Stop etcd + NATS.
set -uo pipefail
source "$(dirname "$0")/env.sh"
docker rm -f "$ETCD_NAME" "$NATS_NAME" >/dev/null 2>&1 || true
echo "[infra] etcd + nats removed"
