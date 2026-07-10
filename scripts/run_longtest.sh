#!/usr/bin/env bash
# run_longtest.sh -- unattended LONG TEST for the WoSAR extension campaign.
#
# The final gate before the ~57-run DoW campaign. Launch it INSIDE tmux and walk
# away; it runs the whole chain and stops hard on any failure:
#
#   a. guard       -- shell/env/host preconditions (tmux, wosar env, git SHA,
#                     no dyn_* container, run-slot free)
#   b. calibration -- reuse an OK calibration JSON, else bring the Dynamo stack
#                     up via deploy/dynamo/*.sh, run calibrate_rate.py at
#                     fraction 0.30, tear the stack down (trap). status != ok is
#                     a HARD STOP -- never a silent fallback to an uncalibrated rate
#   c. dry-run     -- campaign.py --dry-run (pre-flight; prints the schedule)
#   d. campaign    -- campaign.py --start, or --resume iff a state file exists
#   e. verdict     -- per-run status, exit-code meaning, and the exact
#                     analysis command lines to run next (printed, NOT run)
#
# Usage:
#   tmux new -s longtest
#   bash scripts/run_longtest.sh [--recalibrate]
#
# Env:
#   LONGTEST_ALLOW_NOTMUX=1     bypass the tmux guard (discouraged)
#   LONGTEST_CALIB_RATES=...    ascending sweep rates (default 0.5,1,2,4,8,12)

set -euo pipefail

# --------------------------------------------------------------------------
# Paths + logging
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CAMPAIGN_YAML="$REPO_ROOT/campaigns/extension/longtest_campaign.yaml"
CELL_YAML="$REPO_ROOT/campaigns/extension/cells/longtest_dynamo_disagg.yaml"
STATE_DIR="$REPO_ROOT/campaigns/extension/state"
STATE_FILE="$STATE_DIR/longtest_campaign_state.json"
CALIB_JSON="$STATE_DIR/calibration/longtest_dynamo_disagg.json"
CALIB_WORK="$STATE_DIR/calibration/calib_work"
LOG_DIR="$STATE_DIR/logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/longtest_${TS}.log"

RECALIBRATE=0
[ "${1:-}" = "--recalibrate" ] && RECALIBRATE=1

mkdir -p "$LOG_DIR" "$(dirname "$CALIB_JSON")"
# Tee everything (this script + every child's stdout/stderr) to the durable log.
exec > >(tee -a "$LOG") 2>&1

log()  { echo "[longtest] $*"; }
banner() { echo ""; echo "[longtest] ======== $* ========"; }
die()  { echo "[longtest] HARD STOP: $*" >&2; exit 1; }

log "log file: $LOG"
log "campaign: $CAMPAIGN_YAML"

# runs_root comes from the campaign yaml (single source of truth).
RUNS_ROOT="$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CAMPAIGN_YAML'))['runs_root'])")" \
    || die "could not read runs_root from $CAMPAIGN_YAML"
log "runs_root: $RUNS_ROOT"

# --------------------------------------------------------------------------
# a. GUARD
# --------------------------------------------------------------------------
banner "PHASE a: guard"

if [ -z "${TMUX:-}" ] && [ "${LONGTEST_ALLOW_NOTMUX:-}" != "1" ]; then
    die "not inside tmux (\$TMUX unset). A 48h run must survive an SSH drop. Start 'tmux new -s longtest' first, or set LONGTEST_ALLOW_NOTMUX=1 to override."
fi
[ -n "${TMUX:-}" ] && log "tmux: OK ($TMUX)" || log "tmux: BYPASSED via LONGTEST_ALLOW_NOTMUX=1"

python3 -c "import yaml, psutil" 2>/dev/null \
    || die "wosar env not active (python3 -c 'import yaml, psutil' failed). Activate it, then retry."
log "wosar env: OK (yaml, psutil importable)"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY="DIRTY"
else
    GIT_DIRTY="clean"
fi
log "git: $GIT_SHA ($GIT_DIRTY)"

# No dyn_* container may already be running (a prior stack we would trample).
DYN_RUNNING="$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^dyn_' || true)"
if [ -n "$DYN_RUNNING" ]; then
    die "dyn_* container(s) already running: $(echo "$DYN_RUNNING" | tr '\n' ' '). Tear the stack down (bash deploy/dynamo/serve_down.sh && bash deploy/dynamo/infra_down.sh), then retry."
fi
log "dyn_* containers: none running"

# The run-slot lock must be free. acquire_run_slot in a throwaway subprocess
# returns the lock (released on exit) if free, or None if another launcher holds
# it; name the holder from the reaper ledger.
SLOT="$(PYTHONPATH="$REPO_ROOT/scripts" python3 - "$RUNS_ROOT" <<'PY'
import sys, reaper
runs_root = sys.argv[1]
f = reaper.acquire_run_slot(runs_root)
if f is None:
    try:
        holders = reaper.ledger_run_ids(runs_root) or ["<unknown; ledger empty>"]
    except Exception:
        holders = ["<unknown>"]
    print("HELD:" + ",".join(holders))
else:
    print("FREE")  # lock released when this process exits
PY
)" || die "could not check the run-slot lock"
case "$SLOT" in
    FREE) log "run-slot lock: free" ;;
    HELD:*) die "run-slot lock held by another launcher (active run(s): ${SLOT#HELD:}). Another campaign/launch_cell is live on this runs-root; wait for it or stop it." ;;
    *) die "unexpected run-slot check result: $SLOT" ;;
esac

# --------------------------------------------------------------------------
# b. CALIBRATION
# --------------------------------------------------------------------------
banner "PHASE b: calibration (fraction 0.30 of ceiling)"

calib_status() {
    # Print the status field of the calibration JSON, or empty if absent/unparseable.
    python3 -c "import json,sys; print(json.load(open('$CALIB_JSON')).get('status',''))" 2>/dev/null || true
}

if [ "$RECALIBRATE" -eq 0 ] && [ -f "$CALIB_JSON" ] && [ "$(calib_status)" = "ok" ]; then
    RATE="$(python3 -c "import json; print(json.load(open('$CALIB_JSON')).get('rate_calibrated_rps'))" 2>/dev/null || echo '?')"
    log "reusing existing OK calibration: $CALIB_JSON (rate_calibrated_rps=$RATE)"
else
    if [ "$RECALIBRATE" -eq 1 ]; then
        log "--recalibrate: forcing a fresh calibration sweep"
    else
        log "no OK calibration at $CALIB_JSON; running a fresh sweep"
    fi

    RATES="${LONGTEST_CALIB_RATES:-0.5,1,2,4,8,12}"
    BASE_URL="$(python3 -c "import yaml; print(yaml.safe_load(open('$CELL_YAML'))['workload']['client_config_overrides']['base_url'])")"
    PROTOCOL="$(python3 -c "import yaml; print(yaml.safe_load(open('$CELL_YAML'))['workload']['client_config_overrides']['protocol'])")"
    MODEL="$(python3 -c "import yaml; print(yaml.safe_load(open('$CELL_YAML'))['workload']['client_config_overrides']['model'])")"
    READY_URL="$(python3 -c "import yaml; print(yaml.safe_load(open('$CELL_YAML'))['engine']['readyz']['url'])")"
    READY_TIMEOUT="$(python3 -c "import yaml; print(yaml.safe_load(open('$CELL_YAML'))['engine']['readyz']['timeout_s'])")"

    _dynamo_down() {
        log "tearing down the Dynamo calibration stack"
        bash "$REPO_ROOT/deploy/dynamo/serve_down.sh"  >/dev/null 2>&1 || true
        bash "$REPO_ROOT/deploy/dynamo/infra_down.sh"  >/dev/null 2>&1 || true
    }
    # Teardown on ANY exit of the calibration span (success, failure, signal).
    trap '_dynamo_down' EXIT INT TERM

    log "bringing up the Dynamo stack (infra_up + serve_disaggregated)"
    bash "$REPO_ROOT/deploy/dynamo/infra_up.sh"
    bash "$REPO_ROOT/deploy/dynamo/serve_disaggregated.sh"

    log "waiting for $READY_URL (timeout ${READY_TIMEOUT}s)"
    elapsed=0
    until curl -sf --max-time 3 "$READY_URL" >/dev/null 2>&1; do
        [ "$elapsed" -ge "$READY_TIMEOUT" ] && die "engine not ready in ${READY_TIMEOUT}s"
        sleep 5; elapsed=$((elapsed + 5))
    done
    log "engine ready after ${elapsed}s"

    # Materialize the SAME client config the run will use (reuse launch_cell's
    # materializer -- do not hand-roll a config).
    rm -rf "$CALIB_WORK"; mkdir -p "$CALIB_WORK"
    PYTHONPATH="$REPO_ROOT/scripts" python3 - "$REPO_ROOT" "$CELL_YAML" "$CALIB_WORK" <<'PY'
import sys, yaml
from pathlib import Path
import launch_cell as lc
repo_root, cell_yaml, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cell = yaml.safe_load(cell_yaml.read_text())
lc.materialize_client_config(repo_root, work, cell, replica=1)
PY
    CLIENT_CFG="$CALIB_WORK/client_config.yaml"
    [ -f "$CLIENT_CFG" ] || die "failed to materialize client config for calibration"

    # Record the calibrated image tag+digest in the calibration provenance so the
    # run-time staleness gate can verify the ceiling was measured against the SAME
    # image (the pin is launch_cell's source of truth for the digest).
    IMAGE_TAG="$(python3 -c "import yaml; e=yaml.safe_load(open('$CELL_YAML'))['engine']; print(e['image_repo']+':'+e['image_tag'])" 2>/dev/null || true)"
    IMAGE_DIGEST="$(python3 - "$REPO_ROOT" "$CELL_YAML" <<'PY' 2>/dev/null || true
import json, sys, yaml
from pathlib import Path
repo_root, cell_yaml = Path(sys.argv[1]), Path(sys.argv[2])
eng = yaml.safe_load(cell_yaml.read_text())["engine"]
pin_rel = eng.get("digest_pin_file")
if pin_rel:
    p = Path(pin_rel)
    if not p.is_absolute():
        p = repo_root / pin_rel
    print(json.loads(p.read_text()).get("digest", ""))
PY
)"

    log "calibrate_rate.py: rates=$RATES fraction=0.30 -> $CALIB_JSON"
    set +e
    python3 "$REPO_ROOT/scripts/calibrate_rate.py" \
        --config "$CLIENT_CFG" \
        --base-url "$BASE_URL" \
        --protocol "$PROTOCOL" \
        --model "$MODEL" \
        --rates "$RATES" \
        --fraction 0.30 \
        --cell-id longtest_dynamo_disagg \
        --system dynamo_disagg \
        --image-tag "$IMAGE_TAG" \
        --image-digest "$IMAGE_DIGEST" \
        --output "$CALIB_JSON" \
        --sweep-dir "$CALIB_WORK/sweep"
    CAL_RC=$?
    set -e

    _dynamo_down
    trap - EXIT INT TERM

    STATUS="$(calib_status)"
    if [ "$CAL_RC" -ne 0 ] || [ "$STATUS" != "ok" ]; then
        die "calibration did not produce an OK ceiling (exit=$CAL_RC status='${STATUS:-none}'). A non-saturated / no-stable-point sweep makes the fraction-of-ceiling rate meaningless -- refusing to run 48h on an uncalibrated rate. Widen LONGTEST_CALIB_RATES and re-run with --recalibrate."
    fi
    RATE="$(python3 -c "import json; print(json.load(open('$CALIB_JSON')).get('rate_calibrated_rps'))")"
    log "calibration OK: rate_calibrated_rps=$RATE (0.30 x ceiling) -> $CALIB_JSON"
fi

# --------------------------------------------------------------------------
# c. DRY-RUN (pre-flight)
# --------------------------------------------------------------------------
banner "PHASE c: dry-run (pre-flight)"
python3 "$REPO_ROOT/scripts/campaign.py" --campaign-yaml "$CAMPAIGN_YAML" --dry-run \
    || die "pre-flight failed (see the schedule/calibration/free-space lines above)"

# --------------------------------------------------------------------------
# d. CAMPAIGN (start or resume)
# --------------------------------------------------------------------------
banner "PHASE d: campaign"
if [ -f "$STATE_FILE" ]; then
    log "state file present ($STATE_FILE) -> --resume (re-runnable after an interruption)"
    MODE=--resume
else
    log "no state file -> --start (fresh)"
    MODE=--start
fi
set +e
python3 "$REPO_ROOT/scripts/campaign.py" --campaign-yaml "$CAMPAIGN_YAML" "$MODE"
CAMPAIGN_RC=$?
set -e

# --------------------------------------------------------------------------
# e. VERDICT
# --------------------------------------------------------------------------
banner "PHASE e: verdict"
case "$CAMPAIGN_RC" in
    0) MEANING="OK (every scheduled run completed)" ;;
    4) MEANING="INTERRUPTED (signal; state persisted, re-run to resume)" ;;
    5) MEANING="CAMPAIGN-FATAL (non-retryable precondition: disk / host / image-pin)" ;;
    3) MEANING="PRE-FLIGHT ERROR (should have stopped in phase c)" ;;
    10) MEANING="COMPLETED_WITH_FAILURES (queue drained but >=1 run FAILED; inspect, then --resume --rerun-failed)" ;;
    *) MEANING="unexpected (see campaign log)" ;;
esac
log "campaign exit code: $CAMPAIGN_RC -- $MEANING"

log "per-run status (from $STATE_FILE):"
if [ -f "$STATE_FILE" ]; then
    python3 -c "
import json
s = json.load(open('$STATE_FILE'))
for k, v in sorted(s.get('runs', {}).items()):
    print(f\"[longtest]   {k}: {v.get('status','?')} (attempts={v.get('attempts','?')}, rc={v.get('last_rc','?')})\")
" || log "  (could not parse state file)"
else
    log "  (state file missing)"
fi

DYN_RUN_DIR="$RUNS_ROOT/extension_longtest_longtest_dynamo_disagg_r01"
VLLM_RUN_DIR="$RUNS_ROOT/extension_longtest_val_vllm_r01"
echo ""
log "NEXT STEPS -- run these by hand to validate the data (NOT run automatically):"
log "  # Dynamo 48h run (multi-process -> extension validator + trends):"
log "  python3 analysis/validate_extension_run.py --run-dir $DYN_RUN_DIR"
log "  python3 analysis/aging_trends.py $DYN_RUN_DIR --alpha 0.10 --downsample-seconds 60"
log "  # vLLM hand-off run (single-process validator):"
log "  python3 analysis/validation_check.py --run-dir $VLLM_RUN_DIR"

exit "$CAMPAIGN_RC"
