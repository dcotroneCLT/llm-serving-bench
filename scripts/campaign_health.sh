#!/usr/bin/env bash
# campaign_health.sh - exhaustive periodic health check for the WoSAR 2026 campaign.
#
# Read-only: safe to run while the campaign is in flight. No process is killed,
# no file is modified. Inspects only state files, CSVs, and docker/nvidia-smi.
#
# Run from any shell, any conda env:
#   bash scripts/campaign_health.sh
#
# Exit codes:
#   0  OK     - everything within thresholds. Safe to leave running.
#   1  WARN   - one or more soft issues; campaign is OK but inspect.
#   2  FAIL   - one or more hard issues; campaign integrity at risk.
#
# Sections:
#   A. Campaign-wide  (state file, disk, GPU pool, container count)
#   B. Per-run health (looped over wosar2026_*_rNN in runs_root)
#      B.1 Required output files
#      B.2 Manifest content
#      B.3 Container alive on correct GPU
#      B.4 PID resolution (engine.pid; find_engine_pid daemon for triton_child)
#      B.5 proc_monitor: alive ratio, field completeness, RSS magnitude
#      B.6 gpu_monitor:  VRAM plausible, sample continuity
#      B.7 system_monitor: sample presence, swap quiescent
#      B.8 client:       issued rate vs target, success ratio
#      B.9 Logs:         FATAL/error/exception inspection
#
# Tunable thresholds (env vars override):
#   HEALTH_MIN_RUNS_ROOT_GB=5     runs_root free, hard fail if below
#   HEALTH_MIN_VAR_LIB_GB=10      host /var/lib free, hard fail if below
#   HEALTH_MIN_ALIVE_PCT=95       proc_alive threshold, fail if below
#   HEALTH_MIN_RSS_MB=100         per-cell RSS floor, fail if below (wrong PID)
#   HEALTH_MIN_VRAM_MIB=1000      per-cell VRAM floor, fail if below
#   HEALTH_MAX_GPU_GAP_S=10       max gap between gpu samples, fail if above
#   HEALTH_MAX_PROC_GAP_S=60      max gap between proc samples, fail if above
#   HEALTH_RATE_TOLERANCE=0.40    issued/target rate must be within [1-T, 1+T]
#   HEALTH_PID_REFRESH_AGE_S=120  triton_child pidfile must be touched within N sec
#   HEALTH_FATAL_GREP="FATAL|Traceback|Exception|panic|out of memory|oom"

set -uo pipefail

# ---------- Colors ----------
if [ -t 1 ]; then
    RED=$'\033[31m'; YELLOW=$'\033[33m'; GREEN=$'\033[32m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    RED=""; YELLOW=""; GREEN=""; BOLD=""; DIM=""; RESET=""
fi

# ---------- Thresholds ----------
HEALTH_MIN_RUNS_ROOT_GB="${HEALTH_MIN_RUNS_ROOT_GB:-${HEALTH_MIN_HOME_GB:-5}}"
HEALTH_MIN_VAR_LIB_GB="${HEALTH_MIN_VAR_LIB_GB:-10}"
HEALTH_MIN_ALIVE_PCT="${HEALTH_MIN_ALIVE_PCT:-95}"
HEALTH_MIN_RSS_MB="${HEALTH_MIN_RSS_MB:-100}"
HEALTH_MIN_VRAM_MIB="${HEALTH_MIN_VRAM_MIB:-1000}"
HEALTH_MAX_GPU_GAP_S="${HEALTH_MAX_GPU_GAP_S:-10}"
HEALTH_MAX_PROC_GAP_S="${HEALTH_MAX_PROC_GAP_S:-60}"
HEALTH_RATE_TOLERANCE="${HEALTH_RATE_TOLERANCE:-0.40}"
HEALTH_PID_REFRESH_AGE_S="${HEALTH_PID_REFRESH_AGE_S:-120}"
HEALTH_FATAL_GREP="${HEALTH_FATAL_GREP:-FATAL|Traceback|Exception|panic|out of memory|oom|oom-killer|CUDA out of memory}"

# ---------- Paths ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAMPAIGN_YAML="$REPO_ROOT/campaigns/wosar2026/campaign.yaml"
CAMPAIGN_RUNS_ROOT=$(awk -F': *' '/^runs_root:/ {print $2; exit}' "$CAMPAIGN_YAML" 2>/dev/null | tr -d '"' || true)
RUNS_ROOT="${RUNS_ROOT:-${CAMPAIGN_RUNS_ROOT:-$HOME/wosar/runs}}"
STATE_FILE="$REPO_ROOT/campaigns/wosar2026/state/campaign_state.json"

# ---------- Findings accumulator ----------
declare -a FINDINGS
NUM_PASS=0; NUM_WARN=0; NUM_FAIL=0

record() {
    local status="$1" section="$2" msg="$3"
    case "$status" in
        PASS)
            FINDINGS+=("${GREEN}[PASS]${RESET} ${section} | ${msg}")
            NUM_PASS=$((NUM_PASS + 1))
            ;;
        WARN)
            FINDINGS+=("${YELLOW}[WARN]${RESET} ${section} | ${msg}")
            NUM_WARN=$((NUM_WARN + 1))
            ;;
        FAIL)
            FINDINGS+=("${RED}[FAIL]${RESET} ${section} | ${msg}")
            NUM_FAIL=$((NUM_FAIL + 1))
            ;;
    esac
}

# ---------- Helpers ----------
have_python() {
    command -v python3 >/dev/null 2>&1
}

json_get() {
    # json_get <file> <jq-style path>  (uses python3 stdlib)
    # Returns the value on stdout, or empty string on any error (missing
    # key, malformed JSON, file not found). Callers check with [ -n ... ].
    local file="$1" path="$2"
    JSON_FILE="$file" JSON_PATH="$path" python3 - <<'PY' 2>/dev/null
import json
import os
try:
    with open(os.environ["JSON_FILE"], encoding="utf-8") as f:
        d = json.load(f)
    for k in os.environ["JSON_PATH"].split("."):
        if k.isdigit():
            d = d[int(k)]
        else:
            d = d[k]
    print(d)
except Exception:
    pass
PY
}

is_int() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

disk_free_gb() {
    # disk_free_gb <path>; prints integer GiB available, or empty on error.
    local path="$1"
    [ -e "$path" ] || return 0
    df -Pk "$path" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

stat_mtime() {
    # GNU stat on Linux, BSD stat on macOS. Prints epoch seconds or empty.
    local path="$1"
    stat -c %Y "$path" 2>/dev/null || stat -f %m "$path" 2>/dev/null || true
}

run_cell_from_name() {
    local name="$1"
    if [[ "$name" =~ ^wosar2026_([^_]+)_r[0-9][0-9]$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

run_replica_from_name() {
    local name="$1"
    if [[ "$name" =~ _r([0-9][0-9])$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

cell_yaml() {
    local cell_id="$1"
    printf '%s/campaigns/wosar2026/cells/%s.yaml\n' "$REPO_ROOT" "$cell_id"
}

yaml_scalar() {
    # Fixed-format YAML helper for the campaign cell files.
    # yaml_scalar <file> <key>
    local file="$1" key="$2"
    awk -F': *' -v key="$key" '
        $1 ~ "^[[:space:]]*" key "$" {
            value = $2
            sub(/[[:space:]]+#.*/, "", value)
            gsub(/^["'\'']|["'\'']$/, "", value)
            print value
            exit
        }
    ' "$file" 2>/dev/null
}

manifest_kind() {
    local manifest="$1"
    if [ -n "$(json_get "$manifest" "cell_id")" ]; then
        printf 'launch\n'
    elif [ -n "$(json_get "$manifest" "args.label_engine")" ]; then
        printf 'monitor\n'
    else
        printf 'unknown\n'
    fi
}

# Find rotated proc CSVs in a run_dir (excludes gpu*, system*, client/)
find_proc_csvs() {
    local run_dir="$1"
    local out=()
    for f in "$run_dir"/*.csv; do
        [ -f "$f" ] || continue
        local bn
        bn=$(basename "$f")
        case "$bn" in
            gpu*|system*) continue ;;
        esac
        # must be <label>_NNNNNN.csv
        if [[ "$bn" =~ _[0-9]{6}\.csv$ ]]; then
            out+=("$f")
        fi
    done
    printf '%s\n' "${out[@]}"
}

# ============================================================
# Section A: campaign-wide
# ============================================================
echo "${BOLD}========================================================"
echo "WoSAR 2026 campaign health: $(date -Iseconds)"
echo "========================================================${RESET}"
echo ""

echo "${BOLD}== Section A: campaign-wide ==${RESET}"

# A.1 disk space
RUNS_ROOT_GB=$(disk_free_gb "$RUNS_ROOT")
VAR_LIB_GB=$(disk_free_gb /var/lib)
if ! is_int "$RUNS_ROOT_GB"; then
    record WARN "A.disk.runs_root" "could not read free space for $RUNS_ROOT"
elif [ "$RUNS_ROOT_GB" -lt "$HEALTH_MIN_RUNS_ROOT_GB" ]; then
    record FAIL "A.disk.runs_root" "free=${RUNS_ROOT_GB}GB < ${HEALTH_MIN_RUNS_ROOT_GB}GB at $RUNS_ROOT"
else
    record PASS "A.disk.runs_root" "free=${RUNS_ROOT_GB}GB at $RUNS_ROOT"
fi
if ! is_int "$VAR_LIB_GB"; then
    record WARN "A.disk.var_lib" "could not read free space for /var/lib"
elif [ "$VAR_LIB_GB" -lt "$HEALTH_MIN_VAR_LIB_GB" ]; then
    record FAIL "A.disk.var_lib" "free=${VAR_LIB_GB}GB < ${HEALTH_MIN_VAR_LIB_GB}GB"
else
    record PASS "A.disk.var_lib" "free=${VAR_LIB_GB}GB"
fi

# A.2 state file presence and parseable
if ! have_python; then
    record FAIL "A.python" "python3 not found; JSON checks cannot run"
elif [ ! -f "$STATE_FILE" ]; then
    record FAIL "A.state.file" "$STATE_FILE missing (campaign never started or path wrong)"
else
    if ! STATE_FILE="$STATE_FILE" python3 - <<'PY' 2>/dev/null
import json
import os
with open(os.environ["STATE_FILE"], encoding="utf-8") as f:
    json.load(f)
PY
    then
        record FAIL "A.state.file" "$STATE_FILE not parseable as JSON"
    else
        record PASS "A.state.file" "parseable"
        # A.2b status summary
        state_summary=$(STATE_FILE="$STATE_FILE" python3 - <<'PY'
import json
import os
try:
    with open(os.environ["STATE_FILE"], encoding="utf-8") as f:
        d = json.load(f)
    by_status = {}
    for k, v in d.get("runs", {}).items():
        s = v.get("status", "?")
        by_status.setdefault(s, []).append(k)
    print("STATUS_SUMMARY:" + " ".join([f"{s}={len(v)}" for s,v in sorted(by_status.items())]))
    for k, v in d.get("runs", {}).items():
        if v.get("status") == "failed":
            print(f"FAILED_RUN:{k} attempts={v.get('attempts')} rc={v.get('last_rc')}")
except Exception as e:
    print(f"PARSE_ERR:{e}")
PY
)
        SUM=$(printf '%s\n' "$state_summary" | grep "^STATUS_SUMMARY:" | sed 's/STATUS_SUMMARY://' || true)
        FAILED_RUNS=$(printf '%s\n' "$state_summary" | grep "^FAILED_RUN:" | sed 's/FAILED_RUN://' || true)
        record PASS "A.state.summary" "${SUM:-empty}"
        if [ -n "$FAILED_RUNS" ]; then
            while IFS= read -r line; do
                record FAIL "A.state.failed_run" "$line"
            done <<< "$FAILED_RUNS"
        fi
    fi
fi

# A.3 container counts
N_CONTAINERS=$(docker ps --filter "name=wosar2026_" --format '{{.Names}}' 2>/dev/null | wc -l | tr -d ' ')
if ! command -v docker >/dev/null 2>&1; then
    record FAIL "A.docker.containers" "docker command not found"
elif [ "$N_CONTAINERS" -eq 0 ]; then
    record FAIL "A.docker.containers" "no wosar2026 container running (campaign halted?)"
elif [ "$N_CONTAINERS" -gt 4 ]; then
    record WARN "A.docker.containers" "$N_CONTAINERS containers (expected up to 3 + sanity)"
else
    record PASS "A.docker.containers" "$N_CONTAINERS running"
fi

# A.4 GPU pool (snapshot)
GPU_LINES=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$GPU_LINES" ]; then
    record FAIL "A.gpu.nvidia_smi" "nvidia-smi returned nothing"
else
    while IFS=, read -r idx used total util temp; do
        idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' ')
        total=$(echo "$total" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
        temp=$(echo "$temp" | tr -d ' ')
        if is_int "$temp" && [ "$temp" -gt 85 ]; then
            record WARN "A.gpu${idx}.thermal" "temp=${temp}C above 85C threshold"
        fi
        if is_int "$used" && is_int "$total" && [ "$used" -gt "$total" ]; then
            record WARN "A.gpu${idx}.vram" "memory.used (${used}MiB) > total (${total}MiB)??"
        fi
        record PASS "A.gpu${idx}.snapshot" "vram=${used}/${total} MiB  util=${util}%  temp=${temp}C"
    done <<< "$GPU_LINES"
fi

# ============================================================
# Section B: per-run health
# ============================================================
echo ""
echo "${BOLD}== Section B: per-run health ==${RESET}"

# Build list of run dirs to inspect.
RUN_DIRS=""
if [ -d "$RUNS_ROOT" ]; then
    RUN_DIRS=$(find "$RUNS_ROOT" -maxdepth 1 -type d -name 'wosar2026_*_r[0-9][0-9]' 2>/dev/null | sort || true)
fi
if [ -z "$RUN_DIRS" ]; then
    record FAIL "B.runs" "no wosar2026_*_rNN in $RUNS_ROOT"
fi

NOW_TS=$(date +%s)

for run_dir in $RUN_DIRS; do
    name=$(basename "$run_dir")
    echo ""
    echo "${BOLD}-- $name --${RESET}"
    section="B.$name"

    manifest="$run_dir/manifest.json"
    if [ ! -f "$manifest" ]; then
        record FAIL "${section}.manifest" "manifest.json missing"
        continue
    fi

    # Determine cell_id, gpu_index, started_at, duration_s, expected container name, label.
    # Older run_monitors.py rewrote manifest.json while a run was active; in
    # that case infer stable fields from the run name and cell YAML so this
    # script remains useful during the campaign.
    MANIFEST_KIND=$(manifest_kind "$manifest")
    CELL_ID=$(json_get "$manifest" "cell_id")
    [ -n "$CELL_ID" ] || CELL_ID=$(run_cell_from_name "$name")
    REPLICA=$(json_get "$manifest" "replica")
    [ -n "$REPLICA" ] || REPLICA=$(run_replica_from_name "$name")
    CELL_YAML=$(cell_yaml "$CELL_ID")
    GPU_INDEX=$(json_get "$manifest" "engine.gpu_device")
    [ -n "$GPU_INDEX" ] || GPU_INDEX=$(json_get "$manifest" "args.gpu_index")
    [ -n "$GPU_INDEX" ] || GPU_INDEX=$(yaml_scalar "$CELL_YAML" "gpu_device")
    LABEL=$(json_get "$manifest" "monitors.proc.label")
    [ -n "$LABEL" ] || LABEL=$(json_get "$manifest" "args.label_engine")
    [ -n "$LABEL" ] || LABEL=$(yaml_scalar "$CELL_YAML" "label")
    CONTAINER_NAME=$(json_get "$manifest" "container.name")
    if [ -z "$CONTAINER_NAME" ]; then
        CONTAINER_NAME=$(yaml_scalar "$CELL_YAML" "container_name_template")
        CONTAINER_NAME=${CONTAINER_NAME//\{replica\}/$REPLICA}
    fi
    HOST_PID=$(json_get "$manifest" "container.host_pid")
    STARTED_AT_UNIX=$(json_get "$manifest" "started_at_unix")
    DURATION_S=$(json_get "$manifest" "duration_s")
    [ -n "$DURATION_S" ] || DURATION_S=$(json_get "$manifest" "args.duration_seconds")
    [ -n "$DURATION_S" ] || DURATION_S=$(yaml_scalar "$CELL_YAML" "duration_s")
    ENDED_AT=$(json_get "$manifest" "ended_at" 2>/dev/null)
    ENDED_AT_UNIX=$(json_get "$manifest" "ended_at_unix" 2>/dev/null)
    PID_STRATEGY=$(json_get "$manifest" "engine.pid_strategy.type")
    [ -n "$PID_STRATEGY" ] || PID_STRATEGY=$(yaml_scalar "$CELL_YAML" "type")

    if [ -z "$CELL_ID" ] || [ -z "$GPU_INDEX" ] || [ -z "$LABEL" ] || [ -z "$CONTAINER_NAME" ]; then
        record FAIL "${section}.manifest" "missing required fields after fallback (cell=$CELL_ID gpu=$GPU_INDEX label=$LABEL container=$CONTAINER_NAME kind=$MANIFEST_KIND)"
        continue
    fi
    if [ "$MANIFEST_KIND" = "monitor" ]; then
        record WARN "${section}.manifest.shape" "manifest.json is monitor-only; using run name/cell YAML fallbacks"
    fi

    # Is the run still nominally in progress?
    # ENDED_AT is empty when the manifest does not yet contain "ended_at".
    if [ -n "$ENDED_AT" ]; then
        IS_RUNNING=0
    else
        IS_RUNNING=1
    fi
    END_OR_NOW="$NOW_TS"
    [ -n "$ENDED_AT_UNIX" ] && END_OR_NOW="$ENDED_AT_UNIX"
    ELAPSED=$(awk -v s="${STARTED_AT_UNIX:-0}" -v n="$END_OR_NOW" -v d="${DURATION_S:-0}" 'BEGIN{if(s>0 && n>=s) print int(n-s); else print int(d)}')

    record PASS "${section}.context" "cell=$CELL_ID rep=$REPLICA gpu=$GPU_INDEX strategy=$PID_STRATEGY elapsed=${ELAPSED}s running=$IS_RUNNING manifest=$MANIFEST_KIND"

    # ---- B.1 required output files ----
    files_missing=()
    [ -s "$run_dir/engine.pid" ] || files_missing+=("engine.pid")
    [ -s "$run_dir/image_digest.txt" ] || files_missing+=("image_digest.txt")
    [ -s "$run_dir/docker_inspect.json" ] || files_missing+=("docker_inspect.json")
    [ -d "$run_dir/logs" ] || files_missing+=("logs/")
    [ -d "$run_dir/client" ] || files_missing+=("client/")
    if [ "${#files_missing[@]}" -gt 0 ]; then
        record FAIL "${section}.files" "missing: ${files_missing[*]}"
    else
        record PASS "${section}.files" "manifest/engine.pid/image_digest/docker_inspect/logs/client all present"
    fi

    # ---- B.2 image pin consistency ----
    if [ -f "$run_dir/image_digest.txt" ]; then
        digest_file=$(head -1 "$run_dir/image_digest.txt" | tr -d '\n')
        digest_manifest=$(json_get "$manifest" "image.digest")
        if [ -z "$digest_manifest" ]; then
            record PASS "${section}.image_digest" "$digest_file (manifest digest unavailable in $MANIFEST_KIND manifest)"
        elif [ "$digest_file" = "$digest_manifest" ]; then
            record PASS "${section}.image_digest" "$digest_file"
        else
            record FAIL "${section}.image_digest" "image_digest.txt and manifest disagree"
        fi
    fi

    # ---- B.3 container alive on correct GPU ----
    if [ "$IS_RUNNING" -eq 1 ]; then
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -Fxq "$CONTAINER_NAME"; then
            record PASS "${section}.container.alive" "$CONTAINER_NAME running"

            # GPU sanity: container PID OR any of its descendants must be on gpu_index
            compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null | tr -d ' ')
            current_pid=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null)
            on_gpu="no"
            if echo "$compute_pids" | grep -qE "^${current_pid}$"; then
                on_gpu="direct"
            else
                # Check descendants via psutil if available
                if have_python; then
                    on_gpu=$(python3 -c "
import sys
try:
    import psutil
except Exception:
    sys.exit(0)
try:
    root = psutil.Process($current_pid)
    pids = {p.pid for p in root.children(recursive=True)} | {$current_pid}
except Exception:
    print('no'); sys.exit(0)
compute = set()
for line in '''$compute_pids'''.split():
    try: compute.add(int(line))
    except: pass
print('descendant' if pids & compute else 'no')
" 2>/dev/null || echo "no")
                fi
            fi
            if [ "$on_gpu" = "no" ]; then
                record FAIL "${section}.gpu.binding" "container PID $current_pid (and descendants) NOT on gpu $GPU_INDEX"
            else
                record PASS "${section}.gpu.binding" "container on gpu $GPU_INDEX ($on_gpu match)"
            fi
        else
            record FAIL "${section}.container.alive" "$CONTAINER_NAME not in docker ps"
        fi
    fi

    # ---- B.4 PID resolution / find_engine_pid daemon ----
    if [ -s "$run_dir/engine.pid" ]; then
        engine_pid=$(cat "$run_dir/engine.pid" 2>/dev/null | tr -d '\n')
        if [[ "$engine_pid" =~ ^[0-9]+$ ]]; then
            if [ "$IS_RUNNING" -eq 1 ]; then
                if [ -e "/proc/$engine_pid" ]; then
                    record PASS "${section}.engine.pid_alive" "pid=$engine_pid present in /proc"
                else
                    record FAIL "${section}.engine.pid_alive" "pid=$engine_pid not in /proc (engine died)"
                fi
            fi
            if [ "$PID_STRATEGY" = "triton_child" ] && [ "$IS_RUNNING" -eq 1 ]; then
                # pidfile mtime must be recent (daemon polling 30s; allow 4x)
                pid_mtime=$(stat_mtime "$run_dir/engine.pid")
                if ! is_int "$pid_mtime"; then
                    record WARN "${section}.pid_daemon" "could not stat engine.pid mtime"
                else
                    pid_age=$(( NOW_TS - pid_mtime ))
                    if [ "$pid_age" -gt "$HEALTH_PID_REFRESH_AGE_S" ]; then
                        record WARN "${section}.pid_daemon" "engine.pid mtime ${pid_age}s ago > ${HEALTH_PID_REFRESH_AGE_S}s; find_engine_pid may have stalled"
                    else
                        record PASS "${section}.pid_daemon" "pidfile refreshed ${pid_age}s ago"
                    fi
                fi
            fi
        else
            record FAIL "${section}.engine.pid_alive" "engine.pid content '$engine_pid' is not a number"
        fi
    fi

    # ---- B.5 proc_monitor data quality ----
    proc_csvs=$(find_proc_csvs "$run_dir")
    if [ -z "$proc_csvs" ]; then
        record FAIL "${section}.proc.csv" "no proc CSV found"
    else
        n_proc_files=$(printf '%s\n' "$proc_csvs" | wc -l | tr -d ' ')
        # Aggregate alive/total across all rotated files, plus field completeness on alive samples.
        # Streamed in a single awk pass (no pandas required).
        proc_stats=$(awk -F, -v label="$LABEL" '
            BEGIN { OFS="," }
            FNR==1 {
                hcount=NF
                for(i=1;i<=NF;i++) {
                    header[i]=$i
                    if($i=="process_alive") alive_col=i
                }
                next
            }
            {
                tot++
                alive_value = alive_col ? $alive_col : $3
                if(alive_value=="True" || alive_value=="true" || alive_value=="1") {
                    alive++
                    for(i=1;i<=NF;i++) if($i!="" && $i!="None" && $i!="nan") pop[i]++
                }
            }
            END {
                # Output: total alive pop_for_each_column...
                printf "tot=%d alive=%d", tot, alive
                # We expect these key fields populated >=99% of alive
                # Use header to look up indices
                for(i=1;i<=hcount;i++) {
                    h = header[i]
                    if (h=="rss_bytes"||h=="vms_bytes"||h=="uss_bytes"||h=="pss_bytes"||h=="num_threads"||h=="num_fds"||h=="cpu_percent"||h=="voluntary_ctx_switches"||h=="io_read_bytes"||h=="io_write_bytes"||h=="num_children") {
                        v = (alive>0) ? (100*pop[i]/alive) : 0
                        printf " %s=%.1f%%", h, v
                    }
                }
                # Compute RSS min/max in MB on alive samples
                # (second pass would be needed; we keep simple by computing inline in END is tricky; skip min/max here)
                print ""
            }
        ' $proc_csvs)
        tot=$(echo "$proc_stats" | sed 's/.*tot=\([0-9]*\).*/\1/')
        alive=$(echo "$proc_stats" | sed 's/.*alive=\([0-9]*\).*/\1/')
        alive_pct=$(awk -v a=$alive -v t=$tot 'BEGIN{if(t>0) printf "%.1f", 100*a/t; else print "0"}')
        if awk -v p=$alive_pct -v m=$HEALTH_MIN_ALIVE_PCT 'BEGIN{exit !(p<m)}'; then
            record FAIL "${section}.proc.alive" "alive=${alive}/${tot} = ${alive_pct}% < ${HEALTH_MIN_ALIVE_PCT}%"
        else
            record PASS "${section}.proc.alive" "alive=${alive}/${tot} = ${alive_pct}% (${n_proc_files} rotated files)"
        fi
        sample_errors=$(awk -F, '
            FNR==1 {
                err_col=0
                for(i=1;i<=NF;i++) if($i=="sample_error") err_col=i
                next
            }
            err_col && $err_col!="" && $err_col!="None" {
                counts[$err_col]++
            }
            END {
                for (k in counts) printf "%s=%d ", k, counts[k]
            }
        ' $proc_csvs)
        if echo "$sample_errors" | grep -qE "access_denied|watchdog_timeout|pidfile_unreadable"; then
            record FAIL "${section}.proc.sample_error" "$sample_errors"
        elif [ -n "$sample_errors" ]; then
            record WARN "${section}.proc.sample_error" "$sample_errors"
        fi
        # Field completeness flagging
        for field in rss_bytes vms_bytes uss_bytes pss_bytes num_threads num_fds cpu_percent voluntary_ctx_switches io_read_bytes io_write_bytes num_children; do
            pct=$(echo "$proc_stats" | grep -oE "${field}=[0-9.]+%" | grep -oE "[0-9.]+")
            if [ -z "$pct" ]; then continue; fi
            if awk -v p=$pct 'BEGIN{exit !(p<95)}'; then
                record WARN "${section}.proc.field" "$field populated ${pct}% on alive (<95%)"
            fi
        done

        # RSS magnitude on most-recent rotated file (cheap)
        last_proc=$(echo "$proc_csvs" | tail -1)
        rss_stats=$(awk -F, '
            FNR==1 {
                for(i=1;i<=NF;i++) {
                    if($i=="rss_bytes") col=i
                    if($i=="process_alive") alive_col=i
                }
                next
            }
            {
                alive_value = alive_col ? $alive_col : $3
            }
            (alive_value=="True" || alive_value=="true" || alive_value=="1") && col && $col!="" && $col!="None" {
                v = $col / 1048576
                n++; if(n==1||v<mn) mn=v; if(v>mx) mx=v; sum+=v
            }
            END {
                if(n>0) printf "min=%.0f max=%.0f mean=%.0f n=%d", mn, mx, sum/n, n
                else printf "no_data"
            }
        ' "$last_proc")
        rss_max=$(echo "$rss_stats" | grep -oE "max=[0-9]+" | cut -d= -f2)
        if [ -n "$rss_max" ]; then
            if [ "$rss_max" -lt "$HEALTH_MIN_RSS_MB" ]; then
                record FAIL "${section}.proc.rss_size" "rss_max=${rss_max}MB < ${HEALTH_MIN_RSS_MB}MB (wrong PID monitored?)"
            else
                record PASS "${section}.proc.rss_size" "$rss_stats (in last rotated file, MB)"
            fi
        else
            record WARN "${section}.proc.rss_size" "rss column empty in last file"
        fi

        # Sample continuity: largest gap between proc samples across all rotated files (assumes ts_unix is col 1)
        max_gap=$(awk -F, '
            FNR==1 { next }
            { ts=$1+0 }
            prev>0 { g = ts - prev; if(g>maxg) maxg=g }
            { prev = ts }
            END { printf "%.1f", maxg+0 }
        ' $proc_csvs)
        if awk -v g=$max_gap -v m=$HEALTH_MAX_PROC_GAP_S 'BEGIN{exit !(g>m)}'; then
            record WARN "${section}.proc.continuity" "max gap ${max_gap}s > ${HEALTH_MAX_PROC_GAP_S}s"
        else
            record PASS "${section}.proc.continuity" "max gap ${max_gap}s"
        fi
    fi

    # ---- B.6 gpu_monitor data quality ----
    gpu_csvs=$(ls "$run_dir"/gpu*_*.csv 2>/dev/null)
    if [ -z "$gpu_csvs" ]; then
        record FAIL "${section}.gpu.csv" "no gpu CSV"
    else
        gpu_rows=$(awk -F, 'FNR>1 {n++} END {print n+0}' $gpu_csvs)
        if [ "$gpu_rows" -eq 0 ]; then
            record FAIL "${section}.gpu.csv" "gpu CSV files exist but contain no samples"
        fi
        # VRAM max + UUID
        gpu_stats=$(awk -F, '
            FNR==1 {
                for(i=1;i<=NF;i++) if($i=="vram_used_bytes") col=i
                next
            }
            col && $col!="" {
                v = $col / 1048576
                n++; if(n==1||v<mn) mn=v; if(v>mx) mx=v
            }
            END {
                if(n>0) printf "n=%d vram_min=%.0f vram_max=%.0f MiB", n, mn, mx
            }
        ' $gpu_csvs)
        vram_max=$(echo "$gpu_stats" | grep -oE "vram_max=[0-9]+" | cut -d= -f2)
        if [ -z "$vram_max" ]; then
            record FAIL "${section}.gpu.vram" "no populated vram_used_bytes samples"
        elif [ "$vram_max" -lt "$HEALTH_MIN_VRAM_MIB" ]; then
            record FAIL "${section}.gpu.vram" "vram_max=${vram_max}MiB < ${HEALTH_MIN_VRAM_MIB}MiB"
        else
            record PASS "${section}.gpu.vram" "$gpu_stats"
        fi
        # GPU sample continuity
        gpu_max_gap=$(awk -F, '
            FNR==1 { next }
            { ts=$1+0 }
            prev>0 { g = ts - prev; if(g>maxg) maxg=g }
            { prev = ts }
            END { printf "%.1f", maxg+0 }
        ' $gpu_csvs)
        if awk -v g=$gpu_max_gap -v m=$HEALTH_MAX_GPU_GAP_S 'BEGIN{exit !(g>m)}'; then
            record WARN "${section}.gpu.continuity" "max gap ${gpu_max_gap}s > ${HEALTH_MAX_GPU_GAP_S}s"
        else
            record PASS "${section}.gpu.continuity" "max gap ${gpu_max_gap}s"
        fi
    fi

    # ---- B.7 system_monitor presence + swap quiescence ----
    sys_csvs=$(ls "$run_dir"/system_*.csv 2>/dev/null)
    if [ -z "$sys_csvs" ]; then
        record FAIL "${section}.system.csv" "no system CSV"
    else
        sys_rows=$(awk -F, 'FNR>1 {n++} END {print n+0}' $sys_csvs)
        if [ "$sys_rows" -eq 0 ]; then
            record FAIL "${section}.system.csv" "system CSV files exist but contain no samples"
        else
            record PASS "${section}.system.csv" "${sys_rows} samples"
            # Check swap column (look for any non-zero non-empty value)
            swap_max=$(awk -F, '
                FNR==1 { for(i=1;i<=NF;i++) if($i ~ /swap.*used/) col=i; next }
                col && $col!="" && $col+0>m { m=$col+0 }
                END { print m+0 }
            ' $sys_csvs)
            if [ -n "$swap_max" ] && [ "$swap_max" -gt 1073741824 ]; then  # > 1 GB swap = bad
                record WARN "${section}.system.swap" "max swap used = ${swap_max} bytes (host memory pressure)"
            else
                record PASS "${section}.system.swap" "no significant swap (max=${swap_max} bytes)"
            fi
        fi
    fi

    # ---- B.8 client throughput vs target ----
    client_csvs=$(ls "$run_dir"/client/requests_*.csv 2>/dev/null)
    if [ -z "$client_csvs" ]; then
        if [ "$IS_RUNNING" -eq 0 ] || [ "${ELAPSED:-0}" -gt 120 ]; then
            record FAIL "${section}.client.csv" "no client CSV after elapsed=${ELAPSED}s"
        else
            record WARN "${section}.client.csv" "no client CSV yet (elapsed=${ELAPSED}s)"
        fi
    else
        client_stats=$(awk -F, '
            FNR==1 {
                for(i=1;i<=NF;i++) {
                    if($i=="status") status_col=i
                    if($i=="submitted_at_unix") submitted_col=i
                    if($i=="finished_at_unix") finished_col=i
                }
                next
            }
            {
                total++
                status = status_col ? $status_col : ""
                if(status=="ok" || status=="success") ok++
                else if(status=="timeout") timeout++
                else if(status=="dropped") dropped++
                else if(status!="") error++
                ts = 0
                if(finished_col && $finished_col!="") ts = $finished_col + 0
                else if(submitted_col && $submitted_col!="") ts = $submitted_col + 0
                if(ts>0) {
                    nts++
                    if(nts==1 || ts<mn) mn=ts
                    if(ts>mx) mx=ts
                }
            }
            END {
                span = (nts>1) ? (mx-mn) : 0
                printf "total=%d ok=%d error=%d timeout=%d dropped=%d span=%.3f", total, ok, error, timeout, dropped, span
            }
        ' $client_csvs)
        n_total=$(echo "$client_stats" | sed 's/.*total=\([0-9]*\).*/\1/')
        n_ok=$(echo "$client_stats" | sed 's/.*ok=\([0-9]*\).*/\1/')
        n_error=$(echo "$client_stats" | sed 's/.*error=\([0-9]*\).*/\1/')
        n_timeout=$(echo "$client_stats" | sed 's/.*timeout=\([0-9]*\).*/\1/')
        n_dropped=$(echo "$client_stats" | sed 's/.*dropped=\([0-9]*\).*/\1/')
        client_span=$(echo "$client_stats" | sed 's/.*span=\([0-9.]*\).*/\1/')
        if [ "$n_total" -eq 0 ]; then
            if [ "$IS_RUNNING" -eq 0 ] || [ "${ELAPSED:-0}" -gt 120 ]; then
                record FAIL "${section}.client.csv" "client CSV files exist but contain no request rows after elapsed=${ELAPSED}s"
            else
                record WARN "${section}.client.csv" "client CSV files exist but contain no request rows yet"
            fi
        elif [ "$n_ok" -eq 0 ] && [ "${ELAPSED:-0}" -gt 120 ]; then
            record FAIL "${section}.client.ok" "zero successful client rows: total=${n_total} error=${n_error} timeout=${n_timeout} dropped=${n_dropped}"
        fi
        issued_rate=$(awk -v n=$n_total -v s=$client_span 'BEGIN{if(s>0) printf "%.3f", n/s; else print 0}')
        # target from cell yaml
        target=$(awk '/target_rate_rps:/ {print $2; exit}' "$CELL_YAML")
        if [ -n "$target" ] && [ "$target" != "0" ] && awk -v s=$client_span 'BEGIN{exit !(s>0)}'; then
            ratio=$(awk -v a=$issued_rate -v t=$target 'BEGIN{if(t>0) printf "%.3f", a/t; else print 0}')
            tol=$HEALTH_RATE_TOLERANCE
            if awk -v r=$ratio -v t=$tol 'BEGIN{exit !(r<1-t || r>1+t)}'; then
                record WARN "${section}.client.rate" "issued=${issued_rate} target=${target} ratio=${ratio} outside [1-${tol},1+${tol}] span=${client_span}s"
            else
                record PASS "${section}.client.rate" "issued=${issued_rate} target=${target} ratio=${ratio}  ok=${n_ok}/${n_total} span=${client_span}s"
            fi
        elif [ -n "$target" ]; then
            record WARN "${section}.client.rate" "not enough timestamp span to estimate request rate yet (n=${n_total})"
        fi
        # Error rate
        if [ "$n_total" -gt 100 ]; then
            err_pct=$(awk -v e=$n_error -v to=$n_timeout -v t=$n_total 'BEGIN{printf "%.1f", 100*(e+to)/t}')
            if awk -v p=$err_pct 'BEGIN{exit !(p>5)}'; then
                record WARN "${section}.client.errors" "error/timeout responses = ${err_pct}% (>5%); dropped=${n_dropped}/${n_total}"
            fi
        fi
    fi

    # ---- B.9 log inspection ----
    for log in "$run_dir/logs"/*.log; do
        [ -f "$log" ] || continue
        bn=$(basename "$log")
        n_fatal=$(grep -cE "$HEALTH_FATAL_GREP" "$log" 2>/dev/null || true)
        n_fatal=${n_fatal:-0}
        if [ "$n_fatal" -gt 0 ]; then
            sample=$(grep -E "$HEALTH_FATAL_GREP" "$log" | head -1 | cut -c1-150)
            record WARN "${section}.log.$bn" "${n_fatal} suspicious lines; sample: $sample"
        fi
    done
    # find_engine_pid log for triton_child
    if [ "$PID_STRATEGY" = "triton_child" ]; then
        fep_log="$run_dir/logs/find_engine_pid.log"
        if [ -f "$fep_log" ]; then
            n_resolved=$(grep -cE "resolved pid=" "$fep_log" 2>/dev/null || true)
            n_no_match=$(grep -cE "no descendant" "$fep_log" 2>/dev/null || true)
            n_resolved=${n_resolved:-0}
            n_no_match=${n_no_match:-0}
            if [ "$n_resolved" -eq 0 ]; then
                record FAIL "${section}.fep.resolutions" "find_engine_pid never resolved a PID"
            else
                msg="resolved ${n_resolved} times (1=initial, >1=respawn events)"
                if [ "$n_no_match" -gt 0 ]; then
                    msg="$msg; $n_no_match 'no descendant' incidents"
                fi
                record PASS "${section}.fep.resolutions" "$msg"
            fi
        else
            record WARN "${section}.fep.log" "find_engine_pid.log absent for triton_child cell"
        fi
    fi
done

# ============================================================
# Final report
# ============================================================
echo ""
echo "${BOLD}========================================================"
echo "REPORT"
echo "========================================================${RESET}"
for line in "${FINDINGS[@]}"; do
    echo "$line"
done

echo ""
echo "${BOLD}========================================================"
echo "SUMMARY: ${GREEN}${NUM_PASS} PASS${RESET}${BOLD}, ${YELLOW}${NUM_WARN} WARN${RESET}${BOLD}, ${RED}${NUM_FAIL} FAIL${RESET}"
echo "========================================================${RESET}"

if [ "$NUM_FAIL" -gt 0 ]; then
    echo "${RED}${BOLD}HEALTH FAIL${RESET}: $NUM_FAIL hard issues. Campaign integrity at risk; investigate now."
    exit 2
elif [ "$NUM_WARN" -gt 0 ]; then
    echo "${YELLOW}${BOLD}HEALTH WARN${RESET}: $NUM_WARN soft issues. Campaign OK, inspect when convenient."
    exit 1
else
    echo "${GREEN}${BOLD}HEALTH OK${RESET}: all checks pass. Safe to leave running."
    exit 0
fi
