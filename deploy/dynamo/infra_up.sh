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
# Single-node etcd: peer advertise + --initial-cluster MUST be set explicitly and
# match each other, using the LITERAL loopback IP 127.0.0.1 (not "localhost").
# etcd rewrites a "localhost" advertise URL to the host's default-route IP (peers
# can't reach localhost), but leaves --initial-cluster verbatim, so any localhost
# value there mismatches the rewritten advertise URL and etcd exits 1 at startup.
# A literal 127.0.0.1 is used as-is, so the two agree. Clients still reach etcd on
# localhost:2379 because it listens on 0.0.0.0 over the host network.
docker run -d --name "$ETCD_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "$ETCD_IMAGE" \
  /usr/local/bin/etcd \
  --name default \
  --listen-client-urls "http://0.0.0.0:${ETCD_CLIENT_PORT}" \
  --advertise-client-urls "http://127.0.0.1:${ETCD_CLIENT_PORT}" \
  --listen-peer-urls "http://0.0.0.0:${ETCD_PEER_PORT}" \
  --initial-advertise-peer-urls "http://127.0.0.1:${ETCD_PEER_PORT}" \
  --initial-cluster "default=http://127.0.0.1:${ETCD_PEER_PORT}"

echo "[infra] starting NATS ($NATS_IMAGE) with JetStream"
docker run -d --name "$NATS_NAME" --network host "${DOCKER_LOG_OPTS[@]}" "$NATS_IMAGE" \
  -js -p "$NATS_PORT"

echo "[infra] up. etcd :$ETCD_CLIENT_PORT  nats :$NATS_PORT"
echo "[infra] verify: docker ps | grep -E '$ETCD_NAME|$NATS_NAME'"
