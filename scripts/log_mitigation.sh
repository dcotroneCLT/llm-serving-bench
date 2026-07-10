#!/usr/bin/env bash
#
# log_mitigation.sh — append an operator intervention to the campaign
# mitigations log. The campaign_health.sh script READS this log to surface
# recent interventions in its REPORT; nothing else writes to it.
#
# Usage:
#   scripts/log_mitigation.sh [--campaign-yaml PATH] <category> "<note>"
#
# Fixed taxonomy (do not extend without also updating the paper's Threats
# to Validity section — keeping the set closed is the whole point):
#   disk_prune              docker prune / image cleanup / fs cleanup
#   container_restart       docker restart of a campaign container
#   engine_relaunch         engine process or service relaunched mid-run
#   gpu_intervention        nvidia-smi reset / MIG toggle / driver tweak
#   workload_param_change   client rate / max_tokens / prompt mix changed
#   host_intervention       reboot / kernel param / sudoers / network
#
# The log is append-only and lives beside the campaign's state file, DERIVED
# from --campaign-yaml exactly as campaign_health.sh derives it (default: the
# extension campaign). This keeps the writer and the reader pointed at the SAME
# file: a mitigation logged during the extension campaign must show up in that
# campaign's health report, not vanish into the retired wosar2026 state dir.
#   <campaign_dir>/<state_file dir>/mitigations.log
# (override the whole path with MITIGATIONS_LOG=/path/to/file)
#
# Line format (pipe-delimited, one line per intervention):
#   <ISO-8601-timestamp> | <category> | <note>
#
# Examples:
#   scripts/log_mitigation.sh disk_prune "reclaimed 22.88GB after FAIL A.disk.docker_root"
#   scripts/log_mitigation.sh --campaign-yaml campaigns/wosar2026/campaign.yaml \
#       container_restart "wosar2026_e1_r01 OOM-killed, restarted same image"

set -euo pipefail

CATEGORIES="disk_prune container_restart engine_relaunch gpu_intervention workload_param_change host_intervention"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Default to the extension (DoW) campaign, matching campaign_health.sh.
CAMPAIGN_YAML="$REPO_ROOT/campaigns/extension/campaign.yaml"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--campaign-yaml PATH] <category> "<note>"

Valid categories:
$(printf '  %s\n' $CATEGORIES)
Examples:
  $(basename "$0") disk_prune "reclaimed 22.88GB after FAIL A.disk.docker_root"
  $(basename "$0") --campaign-yaml campaigns/wosar2026/campaign.yaml \\
      container_restart "wosar2026_e1_r01 OOM-killed, restarted same image"
EOF
    exit 2
}

# --- CLI: extract --campaign-yaml, keep the two positionals (category, note) ---
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --campaign-yaml) CAMPAIGN_YAML="${2:?--campaign-yaml requires a path}"; shift 2 ;;
        --campaign-yaml=*) CAMPAIGN_YAML="${1#*=}"; shift ;;
        -h|--help) usage ;;
        --) shift; while [ "$#" -gt 0 ]; do POSITIONAL+=("$1"); shift; done ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
set -- ${POSITIONAL[@]+"${POSITIONAL[@]}"}

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

# Resolve the log path. An explicit MITIGATIONS_LOG wins and needs no campaign
# yaml; otherwise derive <campaign_dir>/<state_file dir>/mitigations.log the same
# way campaign_health.sh does, so the writer and reader never diverge.
yaml_top_scalar() {
    local file="$1" key="$2"
    awk -F': *' -v k="$key" '
        $1==k { v=$2; sub(/[[:space:]]+#.*/,"",v); gsub(/^["'\'']|["'\'']$/,"",v); print v; exit }
    ' "$file" 2>/dev/null
}
if [ -n "${MITIGATIONS_LOG:-}" ]; then
    LOG="$MITIGATIONS_LOG"
else
    if [ ! -f "$CAMPAIGN_YAML" ]; then
        echo "ERROR: campaign yaml not found: $CAMPAIGN_YAML" >&2
        echo "Pass --campaign-yaml PATH or set MITIGATIONS_LOG." >&2
        exit 2
    fi
    CAMPAIGN_YAML="$(cd "$(dirname "$CAMPAIGN_YAML")" && pwd)/$(basename "$CAMPAIGN_YAML")"
    CAMPAIGN_DIR="$(dirname "$CAMPAIGN_YAML")"
    STATE_REL=$(yaml_top_scalar "$CAMPAIGN_YAML" "state_file")
    STATE_REL="${STATE_REL:-state/campaign_state.json}"
    case "$STATE_REL" in
        /*) STATE_FILE="$STATE_REL" ;;
        *)  STATE_FILE="$CAMPAIGN_DIR/$STATE_REL" ;;
    esac
    LOG="$(dirname "$STATE_FILE")/mitigations.log"
fi

mkdir -p "$(dirname "$LOG")"

# date "+%z" portable across GNU and BSD; -Iseconds is GNU-only.
TS=$(date "+%Y-%m-%dT%H:%M:%S%z")
printf '%s | %s | %s\n' "$TS" "$CATEGORY" "$NOTE" >> "$LOG"

echo "Logged to ${LOG}:"
echo "  $TS | $CATEGORY | $NOTE"
