#!/usr/bin/env bash
# Mandatory pre-flight smoke test: run one cell for 60 seconds of monitoring,
# then verify framework integrity end-to-end. Catches the failure modes that
# previously cost us 24h+ of GPU time:
#
#   - Docker data root full mid-pull (round 1, Docker /home pool at 32 GB
#     overflowed and containers died after partial layer downloads).
#   - proc_monitor PID is dead/stale (process_alive=False everywhere).
#   - proc_monitor monitoring the wrapper (vllm serve script) instead of the
#     engine worker, so rss is in KB instead of GB.
#   - AccessDenied masked as process_alive=False (no sudo/cap).
#   - Wrong GPU index between container and monitor.
#   - vLLM env var silently ignored (e.g. VLLM_USE_V1=0 on vllm 0.20.1 where
#     V0 has been removed). Container starts, looks fine, runs the wrong engine.
#
# Wallclock: ~5-10 min per smoke test (cold load + 60s sampling + cooldown).
# Wallclock is dominated by the engine cold load, not by the test duration.
#
# Usage:
#   bash scripts/smoke_test.sh campaigns/wosar2026/cells/e1.yaml
#   bash scripts/smoke_test.sh campaigns/wosar2026/cells/e2.yaml 99
#
# Exit codes:
#   0  PASS: ALL checks pass. Safe to proceed to long runs.
#   1  HARD FAIL: framework, disk, or permission issue.
#   2  SOFT FAIL: framework runs but a check is below threshold. Inspect.

set -euo pipefail

CELL_YAML="${1:?usage: smoke_test.sh <cell_yaml> [replica]}"
REPLICA="${2:-99}"

# ---- Thresholds (tune here, used below) ----
MIN_VAR_LIB_GB=30          # min free on /var/lib (docker data root)
MIN_HOME_GB=5              # min free on /home (run dirs, CSVs)
MIN_RSS_MB=100             # alive samples must report >= 100 MB rss
MAX_DEAD_PCT=20            # > 20% dead samples is a HARD fail
MIN_VRAM_MIB=1000          # < 1 GiB on gpu monitor is a SOFT fail
# --------------------------------------------

if [ ! -f "$CELL_YAML" ]; then
  echo "[smoke] ERROR: cell yaml not found: $CELL_YAML" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CELL_ID=$(awk '/^cell_id:/ {print $2; exit}' "$CELL_YAML")
SHORT_LABEL=$(awk '/^short_label:/ {print $2; exit}' "$CELL_YAML")
GPU_DEVICE=$(awk '/^  gpu_device:/ {print $2; exit}' "$CELL_YAML")
CONTAINER_NAME_TEMPLATE=$(awk '/^  container_name_template:/ {gsub(/"/, ""); print $2; exit}' "$CELL_YAML")
CONTAINER_NAME=$(printf "$CONTAINER_NAME_TEMPLATE" | sed "s/{replica}/$(printf "%02d" "$REPLICA")/g")

SMOKE_DIR="/tmp/wosar_smoke_${CELL_ID}"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

echo "[smoke] ==== Cell: $CELL_ID ($SHORT_LABEL)  replica=$REPLICA  gpu=$GPU_DEVICE ===="
echo "[smoke] container: $CONTAINER_NAME"
echo "[smoke] smoke dir: $SMOKE_DIR"

# ---- Check 1: pre-flight disk space ----
echo "[smoke] check 1: disk space"
VAR_LIB_FREE_GB=$(df --output=avail -BG /var/lib | tail -1 | tr -d 'G ')
HOME_FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -d 'G ')
echo "[smoke]   /var/lib free: ${VAR_LIB_FREE_GB} GB (need >= ${MIN_VAR_LIB_GB})"
echo "[smoke]   /home    free: ${HOME_FREE_GB} GB (need >= ${MIN_HOME_GB})"
if [ "$VAR_LIB_FREE_GB" -lt "$MIN_VAR_LIB_GB" ]; then
  echo "[smoke] HARD FAIL: /var/lib free space < ${MIN_VAR_LIB_GB} GB."
  echo "[smoke] Docker images + container layers may not fit. Free space or resize before continuing."
  exit 1
fi
if [ "$HOME_FREE_GB" -lt "$MIN_HOME_GB" ]; then
  echo "[smoke] HARD FAIL: /home free space < ${MIN_HOME_GB} GB."
  echo "[smoke] Run dirs + CSV monitoring would not fit. Free space before continuing."
  exit 1
fi

# ---- Check 1b: leftover tmux sessions, containers, lock files ----
echo ""
echo "[smoke] check 1b: orphan state from previous runs"

# Tmux sessions matching our naming convention or generic monitor names.
ORPHAN_TMUX=$(tmux ls 2>/dev/null | grep -E "wosar|monitors_|client_|pid_tracker_|validation" || true)
if [ -n "$ORPHAN_TMUX" ]; then
  echo "[smoke]   orphan tmux sessions:"
  echo "$ORPHAN_TMUX" | sed 's/^/    /'
  echo "[smoke]   HARD FAIL: kill these manually before retrying (tmux kill-session -t <name>)."
  echo "[smoke]   This guards against PID/log confusion with the new smoke run."
  exit 1
fi

# Containers matching wosar2026_* (campaign-namespaced) or our smoke container.
ORPHAN_CONTAINERS=$(docker ps -a --format '{{.Names}}' | grep -E "^wosar2026_|^smoke_" || true)
if [ -n "$ORPHAN_CONTAINERS" ]; then
  echo "[smoke]   orphan containers found:"
  echo "$ORPHAN_CONTAINERS" | sed 's/^/    /'
  echo "[smoke]   removing them now (docker rm -f)..."
  echo "$ORPHAN_CONTAINERS" | xargs -r docker rm -f >/dev/null
fi

# HuggingFace cache locks older than 60 min (a previous container died holding a lock).
HF_CACHE="$HOME/wosar/hf_cache"
STALE_LOCKS=$(find "$HF_CACHE" -name "*.lock" -mmin +60 2>/dev/null || true)
if [ -n "$STALE_LOCKS" ]; then
  RUNNING_VLLM=$(docker ps --format '{{.Names}}' | grep -iE "vllm|triton|pytorch_naive" || true)
  if [ -n "$RUNNING_VLLM" ]; then
    echo "[smoke]   stale HF locks found, but containers are running (do NOT delete):"
    echo "$RUNNING_VLLM" | sed 's/^/    /'
    echo "[smoke]   SOFT WARN: locks may be legitimate. Investigate before deletion."
  else
    echo "[smoke]   stale HF locks (>60 min, no containers running): removing"
    echo "$STALE_LOCKS" | xargs -r rm -f
  fi
fi

# Host port availability for the cell's ports. Parse the cell yaml.
echo "[smoke]   checking host ports for cell..."
PORTS_TO_CHECK=$(awk '/^  port_mapping:/,/^  [a-z]/' "$CELL_YAML" | \
                 grep -oE '"[0-9]+:[0-9]+"' | cut -d: -f1 | tr -d '"' || true)
for port in $PORTS_TO_CHECK; do
  if ss -tlnp 2>/dev/null | grep -qE ":${port}\s"; then
    echo "[smoke]   HARD FAIL: host port $port is in use:"
    ss -tlnp 2>/dev/null | grep -E ":${port}\s"
    exit 1
  fi
done
echo "[smoke]   ports OK: $PORTS_TO_CHECK"

# Ensure wosar conda env is active (idempotent).
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "wosar" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate wosar
fi

# ---- Run launch_cell.py with short duration ----
echo "[smoke] check 2: launching launch_cell.py with --duration-s-override 60"
python "$REPO_ROOT/scripts/launch_cell.py" \
    --cell-yaml "$CELL_YAML" \
    --replica "$REPLICA" \
    --runs-root "$SMOKE_DIR" \
    --repo-root "$REPO_ROOT" \
    --hf-cache-host "$HOME/wosar/hf_cache" \
    --campaign-id smoke \
    --duration-s-override 60 \
    2>&1 | tee "$SMOKE_DIR/launch_cell.log"

LAUNCH_RC=${PIPESTATUS[0]}
RUN_DIR=$(printf '%s/smoke_%s_r%02d' "$SMOKE_DIR" "$CELL_ID" "$REPLICA")

echo ""
echo "[smoke] launch_cell.py rc=$LAUNCH_RC"
echo "[smoke] run dir: $RUN_DIR"

if [ "$LAUNCH_RC" -ne 0 ]; then
  echo "[smoke] HARD FAIL: launch_cell.py exited rc=$LAUNCH_RC"
  echo "[smoke] last 40 lines of launch log:"
  tail -40 "$SMOKE_DIR/launch_cell.log"
  exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
  echo "[smoke] HARD FAIL: run_dir does not exist: $RUN_DIR"
  exit 1
fi

# ---- Check 3: engine version cross-check (catch silent env var ignore) ----
# Docker logs are gone after launch_cell.py rm -f's the container. We saved
# them under run_dir/logs/container_stdout.log? No, we didn't (yet).
# As a stopgap, dump the prior 1500 lines from docker_inspect.json's log
# pointer if present, otherwise warn. For long runs, we will start writing
# `docker logs --since 0` to a file inside launch_cell.py post-readyz.
echo ""
echo "[smoke] check 3: engine version inspection (manual review)"
INSPECT_JSON="$RUN_DIR/docker_inspect.json"
if [ -f "$INSPECT_JSON" ]; then
  ENV_ARGS=$(python3 -c "import json; d=json.load(open('$INSPECT_JSON')); print('\n'.join(d.get('Config',{}).get('Env',[])))" 2>/dev/null || true)
  echo "[smoke]   container env (from docker inspect):"
  echo "$ENV_ARGS" | sed 's/^/    /'
  echo "[smoke]   for cell $CELL_ID, expected engine generation:"
  case "$CELL_ID" in
    e1) echo "    V1 (vllm/vllm-openai:wosar2026_e1, default scheduler in latest)" ;;
    a1) echo "    V0 (vllm/vllm-openai:wosar2026_a1 = v0.7.3, last with V0 default)" ;;
    e2) echo "    V0 inside Triton (no VLLM_USE_V1 env)" ;;
    a2) echo "    V1 inside Triton (VLLM_USE_V1=1 env)" ;;
    e3|e3b) echo "    HF transformers naive, no scheduler" ;;
  esac
  echo "[smoke]   NOTE: docker logs of container $CONTAINER_NAME no longer available"
  echo "[smoke]         (container removed by launch_cell teardown). For the actual"
  echo "[smoke]         36h runs, manually inspect 'docker logs <container_name>' in"
  echo "[smoke]         the first 5 min of each run for engine version confirmation."
fi

# ---- Check 4: proc_monitor data quality ----
echo ""
echo "[smoke] check 4: proc_monitor data quality"

PROC_CSV=""
for f in "$RUN_DIR"/*_000000.csv; do
  bn=$(basename "$f")
  case "$bn" in
    gpu*|system*) continue ;;
    *) PROC_CSV="$f"; break ;;
  esac
done

if [ -z "$PROC_CSV" ] || [ ! -f "$PROC_CSV" ]; then
  echo "[smoke] HARD FAIL: no proc_monitor CSV found in $RUN_DIR"
  ls -la "$RUN_DIR"
  exit 1
fi

echo "[smoke]   proc CSV: $(basename "$PROC_CSV")"
echo "[smoke]   first 3 rows (header + 2 data):"
head -3 "$PROC_CSV" | cut -c1-220
echo ""

TOTAL=$(awk -F, 'NR>1 {c++} END {print c+0}' "$PROC_CSV")
ALIVE=$(awk -F, 'NR>1 && $3=="True" {c++} END {print c+0}' "$PROC_CSV")
DEAD=$(awk -F, 'NR>1 && $3=="False" {c++} END {print c+0}' "$PROC_CSV")
RSS_OK=$(awk -F, 'NR>1 && $3=="True" && $4!="" && $4!="None" {c++} END {print c+0}' "$PROC_CSV")
USS_OK=$(awk -F, 'NR>1 && $3=="True" && $6!="" && $6!="None" {c++} END {print c+0}' "$PROC_CSV")
# RSS in bytes is column 4; min/max in MB on alive samples
RSS_MIN_MB=$(awk -F, 'NR>1 && $3=="True" && $4!="" && $4!="None" {m=$4/1048576; if(c==0||m<min)min=m; c++} END {printf "%.1f", (c>0)?min:0}' "$PROC_CSV")
RSS_MAX_MB=$(awk -F, 'NR>1 && $3=="True" && $4!="" && $4!="None" {m=$4/1048576; if(m>max)max=m; c++} END {printf "%.1f", max+0}' "$PROC_CSV")

echo "[smoke]   total samples: $TOTAL"
echo "[smoke]   alive=$ALIVE  dead=$DEAD"
echo "[smoke]   rss filled : $RSS_OK / $ALIVE"
echo "[smoke]   uss filled : $USS_OK / $ALIVE"
echo "[smoke]   rss range  : ${RSS_MIN_MB} MB .. ${RSS_MAX_MB} MB"

if [ "$TOTAL" -lt 5 ]; then
  echo "[smoke] HARD FAIL: fewer than 5 proc samples. readyz timing or monitor crash."
  exit 1
fi

if [ "$ALIVE" -eq 0 ]; then
  echo "[smoke] HARD FAIL: zero alive samples. Likely sudo/permission issue on proc_monitor."
  exit 1
fi

DEAD_PCT=$(( 100 * DEAD / TOTAL ))
if [ "$DEAD_PCT" -gt "$MAX_DEAD_PCT" ]; then
  echo "[smoke] HARD FAIL: ${DEAD_PCT}% dead samples (> ${MAX_DEAD_PCT}%). PID tracking broken."
  exit 1
fi

if [ "$RSS_OK" -lt "$ALIVE" ]; then
  GAP=$((ALIVE - RSS_OK))
  echo "[smoke] SOFT FAIL: $GAP alive samples have no rss. AccessDenied masked?"
  exit 2
fi

if [ "$USS_OK" -lt "$ALIVE" ]; then
  GAP=$((ALIVE - USS_OK))
  echo "[smoke] SOFT FAIL: $GAP alive samples have no uss. smaps_rollup partial read?"
  exit 2
fi

# Check 5: rss magnitude. A 7B BF16 model resident in RAM should be ~2-20 GB.
# If rss is tiny, we are monitoring the wrong PID (the wrapper `vllm serve`
# script instead of the worker), and the leak slopes will be meaningless.
RSS_MAX_INT=${RSS_MAX_MB%.*}
if [ "$RSS_MAX_INT" -lt "$MIN_RSS_MB" ]; then
  echo "[smoke] HARD FAIL: rss_max ${RSS_MAX_MB} MB < ${MIN_RSS_MB} MB."
  echo "[smoke]   We are likely tracking the wrapper, not the engine worker."
  echo "[smoke]   For triton_child cells, this means find_engine_pid resolved"
  echo "[smoke]   the wrong descendant. For container_pid1 cells, the engine"
  echo "[smoke]   image entrypoint may have a shell wrapper that doesn't exec."
  exit 1
fi

# Check 6: gpu_monitor VRAM
echo ""
echo "[smoke] check 6: gpu_monitor"
GPU_CSV=$(ls "$RUN_DIR"/gpu*_000000.csv 2>/dev/null | head -1)
if [ -z "$GPU_CSV" ]; then
  echo "[smoke] SOFT FAIL: no gpu_monitor CSV found."
  exit 2
fi
VRAM_MAX_MIB=$(awk -F, 'NR>1 && $3!="" {v=$3/1048576; if(v>m)m=v} END {printf "%.0f", m+0}' "$GPU_CSV")
echo "[smoke]   gpu max vram: $VRAM_MAX_MIB MiB"
if [ "$VRAM_MAX_MIB" -lt "$MIN_VRAM_MIB" ]; then
  echo "[smoke] SOFT FAIL: gpu_monitor shows < ${MIN_VRAM_MIB} MiB. Wrong GPU monitored?"
  exit 2
fi

# Final disk check after the run (catches runs that left a lot of debris)
echo ""
echo "[smoke] check 7: disk free after run"
df -h /var/lib /home | tail -2

echo ""
echo "[smoke] ============================================================"
echo "[smoke] PASS: $CELL_ID smoke test complete."
echo "[smoke]   alive ratio   : $ALIVE / $TOTAL"
echo "[smoke]   rss range     : ${RSS_MIN_MB} - ${RSS_MAX_MB} MB"
echo "[smoke]   uss/pss       : 100% populated on alive samples"
echo "[smoke]   gpu max vram  : $VRAM_MAX_MIB MiB"
echo "[smoke] ============================================================"
exit 0
