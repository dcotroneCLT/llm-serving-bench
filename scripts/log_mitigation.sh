#!/usr/bin/env bash
#
# log_mitigation.sh — append an operator intervention to the campaign
# mitigations log. The campaign_health.sh script READS this log to surface
# recent interventions in its REPORT; nothing else writes to it.
#
# Usage:
#   scripts/log_mitigation.sh <category> "<note>"
#
# Fixed taxonomy (do not extend without also updating the paper's Threats
# to Validity section — keeping the set closed is the whole point):
#   disk_prune              docker prune / image cleanup / fs cleanup
#   container_restart       docker restart of a wosar2026_* container
#   engine_relaunch         engine process or service relaunched mid-run
#   gpu_intervention        nvidia-smi reset / MIG toggle / driver tweak
#   workload_param_change   client rate / max_tokens / prompt mix changed
#   host_intervention       reboot / kernel param / sudoers / network
#
# The log is append-only and lives at:
#   campaigns/wosar2026/state/mitigations.log
# (override path with MITIGATIONS_LOG=/path/to/file)
#
# Line format (pipe-delimited, one line per intervention):
#   <ISO-8601-timestamp> | <category> | <note>
#
# Examples:
#   scripts/log_mitigation.sh disk_prune "reclaimed 22.88GB after FAIL A.disk.var_lib"
#   scripts/log_mitigation.sh container_restart "wosar2026_e1_r01 OOM-killed, restarted same image"

set -euo pipefail

CATEGORIES="disk_prune container_restart engine_relaunch gpu_intervention workload_param_change host_intervention"

usage() {
    cat <<EOF
Usage: $(basename "$0") <category> "<note>"

Valid categories:
$(printf '  %s\n' $CATEGORIES)
Examples:
  $(basename "$0") disk_prune "reclaimed 22.88GB after FAIL A.disk.var_lib"
  $(basename "$0") container_restart "wosar2026_e1_r01 OOM-killed, restarted same image"
EOF
    exit 2
}

[ "$#" -eq 2 ] || usage
CATEGORY="$1"
NOTE="$2"

# Validate against the fixed taxonomy. Free-form would defeat the purpose:
# 3 months from now writing the paper, you need to grep/count categories
# cleanly, not reconcile 12 spellings of "prune".
if ! printf '%s\n' $CATEGORIES | grep -qx -- "$CATEGORY"; then
    echo "ERROR: unknown category '$CATEGORY'" >&2
    echo "Valid: $CATEGORIES" >&2
    exit 1
fi

[ -n "$NOTE" ] || { echo "ERROR: note is empty" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_LOG="$REPO_ROOT/campaigns/wosar2026/state/mitigations.log"
LOG="${MITIGATIONS_LOG:-$DEFAULT_LOG}"

mkdir -p "$(dirname "$LOG")"

# date "+%z" portable across GNU and BSD; -Iseconds is GNU-only.
TS=$(date "+%Y-%m-%dT%H:%M:%S%z")
printf '%s | %s | %s\n' "$TS" "$CATEGORY" "$NOTE" >> "$LOG"

echo "Logged to ${LOG}:"
echo "  $TS | $CATEGORY | $NOTE"
