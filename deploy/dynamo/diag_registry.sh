#!/usr/bin/env bash
# Diagnostic runbook for the flaky Dynamo disaggregated registry bring-up.
#
# Symptom (seen on repass_gate2.sh): prefill+decode workers log "Registered base
# model", a SINGLE long-lived frontend stays Up with HTTP 200 on /health, yet
# /v1/models stays {"data":[]} for 4+ minutes and through frontend restart
# fallbacks -- while an identical single-frontend probe in the same session
# served /v1/models in ~5 s. The fault is EITHER frontend discovery (etcd holds
# the worker's registration keys but the frontend never reads them into
# /v1/models) OR worker registration persistence (the keys are absent or a lease
# expires and they drop out). This script localizes which.
#
# It brings up infra + WORKERS ONLY (serve_disaggregated.sh FRONTEND_START=manual
# so there is NO internal frontend fallback in the way), then starts EXACTLY ONE
# frontend itself at a controlled time and polls both etcd and /v1/models on a
# fixed cadence. On exit it captures evidence BEFORE cleanup and prints a
# single-line VERDICT.
#
#   conda activate wosar
#   bash deploy/dynamo/diag_registry.sh            # ~6 min window, then cleanup
#   WINDOW_S=600 bash deploy/dynamo/diag_registry.sh   # longer window
#
# Exit code: 0 only on REGISTRY_OK, 2 otherwise.
#
# NOTE: deliberately NOT `set -e` -- we run every probe over the whole window and
# the EXIT trap captures evidence regardless of where we stop; each step is
# best-effort.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# env.sh gives us the container names, ports and image; it runs `set -euo
# pipefail`, so relax -e afterwards to keep the run-every-probe behavior.
source "$HERE/env.sh"
set +e

# --- diag configuration (all env-tunable) ---
WINDOW_S="${WINDOW_S:-360}"                 # total poll window (default 6 min)
POLL_INTERVAL_S="${POLL_INTERVAL_S:-10}"    # cadence
FRONTEND_DELAY_S="${FRONTEND_DELAY_S:-0}"   # wait after workers register before starting the ONE frontend
DIAG_ROOT="${DIAG_ROOT:-$HOME/wosar/diag_registry}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/wosar/runs}"  # for the orphan reaper on cleanup
TS="$(date +%Y%m%d_%H%M%S)"
EVID_DIR="$DIAG_ROOT/$TS"

# Container handles (1+1 topology, matching the diag bring-up below).
FE="$DYN_FRONTEND_NAME"          # dyn_frontend
PF="${DYN_PREFILL_PREFIX}_1"     # dyn_prefill_1
DEC="${DYN_DECODE_PREFIX}_1"     # dyn_decode_1

# --- verdict state (set defaults BEFORE the trap so an early exit still reports) ---
DIAG_VERDICT="WORKER_REGISTRATION_SUSPECT"
DIAG_REASON="bring-up did not reach the poll window"
DIAG_EXIT_CODE=2

echo "[diag] repo=$REPO evidence=$EVID_DIR window=${WINDOW_S}s interval=${POLL_INTERVAL_S}s"

# --- helpers ---
_ps() {  # docker ps status of one container by exact name (or 'absent')
  local s; s=$(docker ps -a --filter "name=^${1}$" --format '{{.Status}}' 2>/dev/null | head -1)
  echo "${s:-absent}"
}

_etcd_keycount() {  # count of dynamo keys currently in etcd
  docker exec -e ETCDCTL_API=3 "$ETCD_NAME" etcdctl get "" --prefix --keys-only 2>/dev/null | grep -ci dynamo
}

# The frontend is started via the shared start_frontend() from env.sh (same
# --user 0:0 + HF cache mount as the workers) so it can materialize the model
# card; no diag-local docker run, so this path cannot drift from the serve
# scripts (that divergence was what made the failure look flaky).

_capture_evidence() {
  mkdir -p "$EVID_DIR" 2>/dev/null || true
  # (1) full etcd key dump: all keys-only, then values for the dynamo-prefixed keys.
  {
    echo "### all keys (keys-only) ###"
    docker exec -e ETCDCTL_API=3 "$ETCD_NAME" etcdctl get "" --prefix --keys-only 2>/dev/null
    echo ""
    echo "### values for dynamo-prefixed keys ###"
    docker exec -e ETCDCTL_API=3 "$ETCD_NAME" etcdctl get "" --prefix --keys-only 2>/dev/null \
      | grep -i dynamo | while read -r k; do
          [ -n "$k" ] || continue
          echo "=== $k ==="
          docker exec -e ETCDCTL_API=3 "$ETCD_NAME" etcdctl get "$k" 2>/dev/null
          echo ""
        done
  } > "$EVID_DIR/etcd_dump.txt" 2>&1

  # (2) FULL docker logs (not tail) of frontend, prefill, decode.
  docker logs "$FE"  > "$EVID_DIR/frontend.log" 2>&1 || true
  docker logs "$PF"  > "$EVID_DIR/prefill.log"  2>&1 || true
  docker logs "$DEC" > "$EVID_DIR/decode.log"   2>&1 || true

  # (3) decode-log summary: the benign API-drift line + any de-registration /
  #     lease-expiry / disconnect signals, saved separately for a fast eyeball.
  {
    echo "### get_kv_cache_group_metadata (known-benign vLLM/Dynamo API drift) ###"
    grep -n "get_kv_cache_group_metadata" "$EVID_DIR/decode.log" 2>/dev/null || echo "(none)"
    echo ""
    echo "### de-registration / lease-expiry / disconnect signals ###"
    grep -niE "de-?register|unregister|lease.*(expir|revok|lost|grant)|expired|revoke|disconnect|removed|evict" \
      "$EVID_DIR/decode.log" 2>/dev/null || echo "(none)"
  } > "$EVID_DIR/decode_registration_summary.txt" 2>&1
}

# --- on-exit: evidence FIRST, then cleanup like repass_gate2.sh, VERDICT LAST ---
_on_exit() {
  echo "[diag] --- capturing evidence to $EVID_DIR (BEFORE cleanup) ---"
  _capture_evidence
  echo "[diag] --- cleanup (serve_down + infra_down + reaper + docker rm) ---"
  bash "$HERE/serve_down.sh"  >/dev/null 2>&1 || true
  bash "$HERE/infra_down.sh"  >/dev/null 2>&1 || true
  python3 -c "import sys; sys.path.insert(0,'$REPO/scripts'); import reaper; \
print('\n'.join(reaper.reap_orphans('$RUNS_ROOT')))" 2>/dev/null || true
  docker rm -f "$FE" "$PF" "$DEC" "$ETCD_NAME" "$NATS_NAME" >/dev/null 2>&1 || true
  echo "[diag] VERDICT: $DIAG_VERDICT ($DIAG_REASON)" | tee "$EVID_DIR/verdict.txt" 2>/dev/null
  exit "$DIAG_EXIT_CODE"
}
trap _on_exit EXIT

mkdir -p "$EVID_DIR"

# --- step 1: infra + workers only (NO frontend from serve_disaggregated) ---
bash "$HERE/infra_up.sh"
echo "[diag] bringing up WORKERS ONLY (FRONTEND_START=manual) ..."
if N_PREFILL=1 N_DECODE=1 PREFILL_GPU=0 DECODE_GPU=1 FRONTEND_START=manual \
     bash "$HERE/serve_disaggregated.sh"; then
  echo "[diag] workers registered; starting the SINGLE diagnostic frontend."
else
  # Workers never registered -> the fault is upstream of the frontend entirely.
  DIAG_VERDICT="WORKER_REGISTRATION_SUSPECT"
  DIAG_REASON="workers failed to register (serve_disaggregated FRONTEND_START=manual returned non-zero)"
  DIAG_EXIT_CODE=2
  echo "[diag] workers did NOT come up/register; skipping poll, going straight to evidence."
  exit 2
fi

# --- step 2: start EXACTLY ONE frontend at a controlled time, then poll ---
if [ "$FRONTEND_DELAY_S" -gt 0 ] 2>/dev/null; then
  echo "[diag] FRONTEND_DELAY_S=$FRONTEND_DELAY_S: waiting before starting the frontend ..."
  sleep "$FRONTEND_DELAY_S"
fi
start_frontend
echo "[diag] one frontend started on :$FRONTEND_HTTP_PORT (no churn from here); polling ..."

# --- step 3: poll etcd + /v1/models every POLL_INTERVAL_S for the window ---
served=0
max_keys=0
seen_keys=0            # etcd dynamo keys were >0 at some point
dropped_to_zero=0      # ... and later observed back at 0
empty_streak=0         # consecutive seconds with keys>0 while /v1/models empty
max_empty_streak=0

SECONDS=0
while [ "$SECONDS" -lt "$WINDOW_S" ]; do
  now="$(date +%H:%M:%S)"
  keycount="$(_etcd_keycount)"; keycount="${keycount:-0}"

  # /v1/models: HTTP code + body (truncated for stdout; keys list to a file).
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:${FRONTEND_HTTP_PORT}/v1/models" 2>/dev/null)
  code="${code:-000}"
  body=$(curl -s -m 5 "http://localhost:${FRONTEND_HTTP_PORT}/v1/models" 2>/dev/null)
  models_now=0
  if [ "$code" = "200" ] && printf '%s' "$body" | grep -q '"id"'; then models_now=1; served=1; fi
  body_trunc=$(printf '%s' "$body" | tr -d '\n' | cut -c1-120)

  # full key list to the evidence log (NOT stdout).
  {
    echo "=== $now (dynamo_keys=$keycount, models_http=$code) ==="
    docker exec -e ETCDCTL_API=3 "$ETCD_NAME" etcdctl get "" --prefix --keys-only 2>/dev/null
  } >> "$EVID_DIR/etcd_keys.log"

  # timestamped one-liner to stdout + poll log.
  echo "$(printf '[diag] %s  etcd_dynamo_keys=%-3s  models_http=%s models=%s | fe=[%s] pf=[%s] dec=[%s]' \
      "$now" "$keycount" "$code" "${body_trunc:-<empty>}" "$(_ps "$FE")" "$(_ps "$PF")" "$(_ps "$DEC")")" \
    | tee -a "$EVID_DIR/poll.log"

  # accumulate the discriminating signals.
  if [ "$keycount" -gt 0 ] 2>/dev/null; then
    seen_keys=1
    [ "$keycount" -gt "$max_keys" ] && max_keys="$keycount"
  else
    [ "$seen_keys" = 1 ] && dropped_to_zero=1
  fi
  if [ "$keycount" -gt 0 ] 2>/dev/null && [ "$models_now" != 1 ]; then
    empty_streak=$((empty_streak + POLL_INTERVAL_S))
    [ "$empty_streak" -gt "$max_empty_streak" ] && max_empty_streak="$empty_streak"
  else
    empty_streak=0
  fi

  [ "$served" = 1 ] && break
  sleep "$POLL_INTERVAL_S"
done

# --- step 4: verdict ---
# Ordering rationale: a drop-to-zero (or never-registered) is a stronger signal
# than "keys present but not discovered", so WORKER_REGISTRATION_SUSPECT is
# evaluated before FRONTEND_DISCOVERY_SUSPECT when both could apply.
if [ "$served" = 1 ]; then
  DIAG_VERDICT="REGISTRY_OK"
  DIAG_REASON="/v1/models listed the model within ${WINDOW_S}s"
  DIAG_EXIT_CODE=0
elif [ "$max_keys" -eq 0 ] || [ "$dropped_to_zero" = 1 ]; then
  DIAG_VERDICT="WORKER_REGISTRATION_SUSPECT"
  if [ "$max_keys" -eq 0 ]; then
    DIAG_REASON="etcd dynamo keys were 0 for the whole ${WINDOW_S}s window"
  else
    DIAG_REASON="etcd dynamo keys registered (max=$max_keys) then dropped back to 0 (lease expiry / de-registration)"
  fi
  DIAG_EXIT_CODE=2
elif [ "$max_empty_streak" -ge 60 ]; then
  DIAG_VERDICT="FRONTEND_DISCOVERY_SUSPECT"
  DIAG_REASON="etcd dynamo keys stayed >0 (max=$max_keys) for ${max_empty_streak}s but /v1/models stayed empty"
  DIAG_EXIT_CODE=2
else
  # Keys present and never dropped, but not sustained >=60s empty (only when the
  # window is very short). Registry has keys, so this leans frontend discovery.
  DIAG_VERDICT="FRONTEND_DISCOVERY_SUSPECT"
  DIAG_REASON="etcd dynamo keys present (max=$max_keys) but /v1/models empty; keys-empty streak ${max_empty_streak}s < 60s (weak signal)"
  DIAG_EXIT_CODE=2
fi

# _on_exit (EXIT trap) captures evidence, cleans up, prints the VERDICT and exits
# with DIAG_EXIT_CODE.
