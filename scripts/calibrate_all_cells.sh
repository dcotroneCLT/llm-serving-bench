#!/usr/bin/env bash
# calibrate_all_cells.sh - run calibrate_rate.sh on every cell, sequentially.
#
# Calibrates e1, a1, e2, a2, e3 in order, then derives e3b's rate as
# 0.287 * e3_recommended (the n=1 ratio between e3b 0.050 rps and e3
# 0.174 rps, capturing the original protocol's "low rate" choice).
#
# Each cell: ~30 min (one container cold load + 6 rates x 4 min + cooldowns).
# Total wallclock: ~3 hours (sequential, single GPU per cell).
#
# Usage:
#   bash scripts/calibrate_all_cells.sh
#
# Run in tmux to survive SSH disconnect:
#   tmux new -d -s calib_all '
#     source ~/miniconda3/etc/profile.d/conda.sh && \
#     conda activate wosar && \
#     cd ~/wosar/llm-serving-bench && \
#     bash scripts/calibrate_all_cells.sh > ~/wosar/runs/calib_all.log 2>&1
#   '

set -uo pipefail

if [ -t 1 ]; then
    BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

CELLS=(e1 a1 e2 a2 e3)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$HOME/wosar/runs/calib_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
SUMMARY="$OUTPUT_DIR/all_cells_summary.tsv"
echo -e "cell\tsaturation_offered_rps\tsaturation_achieved_rps\trecommended_85pct_rps" > "$SUMMARY"

echo "${BOLD}[calib_all] starting full calibration sweep${RESET}"
echo "[calib_all] cells: ${CELLS[*]}"
echo "[calib_all] output: $OUTPUT_DIR"
echo "[calib_all] expected wallclock: ~3h sequential"
echo ""

START_TS=$(date +%s)

for cell in "${CELLS[@]}"; do
    echo ""
    echo "${BOLD}=========================================="
    echo "[calib_all] CELL $cell starting at $(date -Iseconds)"
    echo "==========================================${RESET}"
    LOG="$OUTPUT_DIR/calib_${cell}.log"

    if ! bash "$REPO_ROOT/scripts/calibrate_rate.sh" "$cell" 2>&1 | tee "$LOG"; then
        echo "${RED}[calib_all] $cell FAILED (rc=$?). Skipping summary line; continuing.${RESET}"
        echo -e "${cell}\tFAILED\tFAILED\tFAILED" >> "$SUMMARY"
        continue
    fi

    # Extract the saturation+recommended values from the log. The format is
    # produced by calibrate_rate.sh's final SATURATION ANALYSIS block.
    SAT_OFFERED=$(grep "offered = " "$LOG" | tail -1 | awk '{print $3}')
    SAT_ACHIEVED=$(grep "achieved = " "$LOG" | tail -1 | awk '{print $3}')
    RECOMMENDED=$(grep "target_rate_rps = " "$LOG" | tail -1 | awk '{print $3}')
    if [ -z "$SAT_OFFERED" ] || [ -z "$RECOMMENDED" ]; then
        echo "${YELLOW}[calib_all] $cell completed but could not parse SATURATION ANALYSIS lines${RESET}"
        echo -e "${cell}\tparse_error\tparse_error\tparse_error" >> "$SUMMARY"
        continue
    fi
    echo -e "${cell}\t${SAT_OFFERED}\t${SAT_ACHIEVED}\t${RECOMMENDED}" >> "$SUMMARY"
    echo "${GREEN}[calib_all] $cell done: recommended = ${RECOMMENDED} rps${RESET}"
done

# e3b: derive from e3 using the n=1 protocol ratio (0.050 / 0.174 = 0.287).
E3_REC=$(grep "^e3" "$SUMMARY" | tail -1 | cut -f4)
if [[ "$E3_REC" =~ ^[0-9.]+$ ]]; then
    E3B_REC=$(awk -v r="$E3_REC" 'BEGIN { printf "%.3f", r * 0.287 }')
    echo -e "e3b\tderived_from_e3\tderived_from_e3\t${E3B_REC}_(=_e3*0.287)" >> "$SUMMARY"
else
    echo -e "e3b\te3_unavailable\te3_unavailable\tcould_not_derive" >> "$SUMMARY"
fi

END_TS=$(date +%s)
DURATION_MIN=$(( (END_TS - START_TS) / 60 ))

echo ""
echo "${BOLD}=========================================="
echo "[calib_all] DONE. Total wallclock: ${DURATION_MIN} min"
echo "==========================================${RESET}"
echo ""
echo "${BOLD}RECOMMENDED RATES (apply to campaigns/wosar2026/cells/*.yaml workload.client_config_overrides.target_rate_rps):${RESET}"
column -t -s $'\t' "$SUMMARY"
echo ""
echo "Full per-cell logs and TSVs in: $OUTPUT_DIR"
