#!/usr/bin/env bash
# longtest_status.sh -- read-only status of the extension LONG TEST.
#
# Safe to run from any second SSH session while run_longtest.sh is live: it only
# READS state/manifest/log files and `docker ps`, never touches the run, and
# ALWAYS exits 0. Every step degrades gracefully (missing file -> say so, keep
# going) so a half-written run dir never makes the viewer fail.
#
# Usage:  bash scripts/longtest_status.sh

# Deliberately NOT set -e: a viewer must never abort on a missing/partial file.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAMPAIGN_YAML="$REPO_ROOT/campaigns/extension/longtest_campaign.yaml"
STATE_DIR="$REPO_ROOT/campaigns/extension/state"
STATE_FILE="$STATE_DIR/longtest_campaign_state.json"
LOG_DIR="$STATE_DIR/logs"

section() { echo ""; echo "== $* =="; }
note()    { echo "  $*"; }

echo "[longtest-status] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

RUNS_ROOT="$(python3 -c "import yaml; print(yaml.safe_load(open('$CAMPAIGN_YAML'))['runs_root'])" 2>/dev/null || true)"
CAMPAIGN_ID="$(python3 -c "import yaml; print(yaml.safe_load(open('$CAMPAIGN_YAML')).get('campaign_id',''))" 2>/dev/null || true)"

# --- per-run status ---
section "campaign state"
if [ -f "$STATE_FILE" ]; then
    python3 -c "
import json
s = json.load(open('$STATE_FILE'))
runs = s.get('runs', {})
if not runs:
    print('  (no runs recorded yet)')
for k, v in sorted(runs.items()):
    print(f\"  {k}: {v.get('status','?')} (attempts={v.get('attempts','?')}, rc={v.get('last_rc','?')})\")
" 2>/dev/null || note "(could not parse $STATE_FILE)"
else
    note "no state file yet ($STATE_FILE)"
fi

# --- current run (the one marked running) ---
section "current run"
RUN_KEY=""
ATTEMPT=""
if [ -f "$STATE_FILE" ]; then
    read -r RUN_KEY ATTEMPT < <(python3 -c "
import json
s = json.load(open('$STATE_FILE'))
for k, v in s.get('runs', {}).items():
    if v.get('status') == 'running':
        print(k, v.get('attempts', 1)); break
" 2>/dev/null || true)
fi

if [ -z "$RUN_KEY" ]; then
    note "no run currently marked 'running'"
else
    note "run_key=$RUN_KEY attempt=$ATTEMPT"
    RUN_DIR="$RUNS_ROOT/${CAMPAIGN_ID}_${RUN_KEY}"
    MANIFEST="$RUN_DIR/manifest.json"
    ATTEMPT_LOG="$LOG_DIR/${CAMPAIGN_ID}_${RUN_KEY}_attempt${ATTEMPT}.log"
    note "run_dir=$RUN_DIR"

    # elapsed vs duration_s from the live manifest.
    if [ -f "$MANIFEST" ]; then
        python3 -c "
import json, time
m = json.load(open('$MANIFEST'))
start = m.get('started_at_unix'); dur = m.get('duration_s')
if start and dur:
    el = time.time() - float(start); dur = float(dur)
    pct = (el / dur * 100.0) if dur > 0 else 0.0
    print(f'  elapsed {el/3600:.1f}h / {dur/3600:.1f}h ({pct:.0f}%)')
else:
    print('  manifest present but missing started_at_unix/duration_s')
" 2>/dev/null || note "(could not parse manifest)"
    else
        note "no live manifest yet ($MANIFEST)"
    fi

    # last heartbeat line from the per-attempt log.
    if [ -f "$ATTEMPT_LOG" ]; then
        HB="$(grep 'progress:' "$ATTEMPT_LOG" 2>/dev/null | tail -1 || true)"
        [ -n "$HB" ] && note "last heartbeat: $HB" || note "no heartbeat line yet (first is at ~30 min)"
    else
        note "no per-attempt log yet ($ATTEMPT_LOG)"
    fi

    # last disk_usage.csv row.
    DISK_CSV="$RUN_DIR/disk_usage.csv"
    if [ -f "$DISK_CSV" ]; then
        note "disk_usage header: $(head -1 "$DISK_CSV" 2>/dev/null || true)"
        note "disk_usage last:   $(tail -1 "$DISK_CSV" 2>/dev/null || true)"
    else
        note "no disk_usage.csv yet ($DISK_CSV)"
    fi

    # last 3 launch_cell log lines (streamed into the per-attempt log).
    section "last launch_cell lines"
    if [ -f "$ATTEMPT_LOG" ]; then
        tail -3 "$ATTEMPT_LOG" 2>/dev/null | sed 's/^/  /' || note "(could not read $ATTEMPT_LOG)"
    else
        note "no per-attempt log yet"
    fi
fi

# --- dyn_* containers (one-liner) ---
section "dyn_* containers"
if command -v docker >/dev/null 2>&1; then
    DYN="$(docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E '^dyn_' || true)"
    [ -n "$DYN" ] && echo "$DYN" | sed 's/^/  /' || note "none running"
else
    note "docker not available"
fi

exit 0
