#!/usr/bin/env bash
# Start the infrastructure Dynamo depends on: etcd + NATS, on the host network.
# These are SEPARATE from the engine; the per-component monitor captures them in
# the "infra" aggregate, kept OUT of the engine USS aggregate.
#
# Usage: bash deploy/dynamo/infra_up.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"

docker rm -f "$ETCD_NAME" "$NATS_NAME" >/dev/null 2>&1 || true

echo "[infra] starting etcd ($ETCD_IMAGE)"
# --initial-advertise-peer-urls is pinned to localhost: without it etcd auto-detects
# the host's default-route IP for the peer URL, which then mismatches the default
# --initial-cluster (default=http://localhost:2380) and etcd exits 1 at startup.
docker run -d --name "$ETCD_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "$ETCD_IMAGE" \
  /usr/local/bin/etcd \
  --listen-client-urls "http://0.0.0.0:${ETCD_CLIENT_PORT}" \
  --advertise-client-urls "http://localhost:${ETCD_CLIENT_PORT}" \
  --listen-peer-urls "http://0.0.0.0:${ETCD_PEER_PORT}" \
  --initial-advertise-peer-urls "http://localhost:${ETCD_PEER_PORT}"

echo "[infra] starting NATS ($NATS_IMAGE) with JetStream"
docker run -d --name "$NATS_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "$NATS_IMAGE" \
  -js -p "$NATS_PORT"

echo "[infra] up. etcd :$ETCD_CLIENT_PORT  nats :$NATS_PORT"
echo "[infra] verify: docker ps | grep -E '$ETCD_NAME|$NATS_NAME'"
