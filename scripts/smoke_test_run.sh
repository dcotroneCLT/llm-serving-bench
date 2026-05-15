#!/usr/bin/env bash
# smoke_test_run.sh - GO/NO-GO gate before each long aging run.
#
# Pure bash. Approx 5 minutes wall clock. Catches the failure modes that
# burned >24h of GPU in round 1: stale containers, wrong engine version
# (VLLM_USE_V1 silently ignored on vLLM 0.20.1), wrong PID monitored
# (wrapper instead of worker, rss tiny), missing sudo for proc_monitor.
#
# Usage:
#   bash scripts/smoke_test_run.sh <cell_id> <replica_n>
#     cell_id    : e1|e2|e3|e3b|a1|a2
#     replica_n  : 1|2|3 (replica index for the long run; used only for naming)
#
# Exit codes:
#   0  GO   : all hard checks PASS. Safe to launch the 24h/36h run.
#   1  NO-GO: at least one hard check FAILED. Details printed.
#
# This script is INDEPENDENT of campaign.py and launch_cell.py. It is meant
# to be run manually as the last check before each cell's first long run.

set -uo pipefail   # NOT -e: we collect all failures into a report.

# ---------- Color codes (TTY only) ----------
if [ -t 1 ]; then
    RED=$'\033[31m'; YELLOW=$'\033[33m'; GREEN=$'\033[32m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; YELLOW=""; GREEN=""; BOLD=""; RESET=""
fi

# ---------- Argument parsing ----------
if [ $# -ne 2 ]; then
    echo "usage: $0 <cell_id> <replica_n>"
    echo "  cell_id   : e1|e2|e3|e3b|a1|a2"
    echo "  replica_n : 1|2|3"
    exit 1
fi
CELL_ID="$1"
REPLICA="$2"
case "$CELL_ID" in
    e1|e2|e3|e3b|a1|a2) ;;
    *) echo "${RED}ERROR${RESET} invalid cell_id '$CELL_ID'"; exit 1 ;;
esac
case "$REPLICA" in
    1|2|3) ;;
    *) echo "${RED}ERROR${RESET} invalid replica_n '$REPLICA'"; exit 1 ;;
esac

# ---------- Cell config lookup ----------
declare -A IMAGE GPU_DEV PORT_HOST READY_PATH ETYPE LABEL TRITON_EXTRAS
IMAGE[e1]="vllm/vllm-openai:wosar2026_e1"
IMAGE[a1]="vllm/vllm-openai:wosar2026_a1"
IMAGE[e2]="tritonserver:wosar2026_e2_a2"
IMAGE[a2]="tritonserver:wosar2026_e2_a2"
IMAGE[e3]="pytorch_naive:wosar2026"
IMAGE[e3b]="pytorch_naive:wosar2026"

GPU_DEV[e1]=0;  GPU_DEV[a1]=0
GPU_DEV[e2]=1;  GPU_DEV[a2]=1
GPU_DEV[e3]=2;  GPU_DEV[e3b]=2

PORT_HOST[e1]=8100; PORT_HOST[a1]=8100
PORT_HOST[e2]=8200; PORT_HOST[a2]=8200
PORT_HOST[e3]=8300; PORT_HOST[e3b]=8300

READY_PATH[e1]="/health";              READY_PATH[a1]="/health"
READY_PATH[e2]="/v2/health/ready";     READY_PATH[a2]="/v2/health/ready"
READY_PATH[e3]="/readyz";              READY_PATH[e3b]="/readyz"

ETYPE[e1]="container_pid1"; ETYPE[a1]="container_pid1"
ETYPE[e2]="triton_child";   ETYPE[a2]="triton_child"
ETYPE[e3]="container_pid1"; ETYPE[e3b]="container_pid1"

LABEL[e1]="vllm_v1_standalone"
LABEL[a1]="vllm_v0_standalone"
LABEL[e2]="triton_vllm_v0"
LABEL[a2]="triton_vllm_v1"
LABEL[e3]="pytorch_naive"
LABEL[e3b]="pytorch_naive"

CONTAINER_NAME="smoke_run_${CELL_ID}_r${REPLICA}"
SMOKE_DIR="/tmp/wosar_smoke_run_${CELL_ID}_r${REPLICA}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_CACHE="$HOME/wosar/hf_cache"
WOSAR_PY="$HOME/miniconda3/envs/wosar/bin/python3.11"

GPU="${GPU_DEV[$CELL_ID]}"
PORT="${PORT_HOST[$CELL_ID]}"
READY_URL="http://localhost:${PORT}${READY_PATH[$CELL_ID]}"
IMG="${IMAGE[$CELL_ID]}"
LBL="${LABEL[$CELL_ID]}"

mkdir -p "$SMOKE_DIR"

# ---------- Report accumulator ----------
REPORT=()
NO_GO=0
record() {
    local status="$1" section="$2" detail="$3"
    case "$status" in
        PASS) REPORT+=("${GREEN}[PASS]${RESET} ${section} | ${detail}") ;;
        WARN) REPORT+=("${YELLOW}[WARN]${RESET} ${section} | ${detail}") ;;
        FAIL) REPORT+=("${RED}[FAIL]${RESET} ${section} | ${detail}"); NO_GO=1 ;;
    esac
    echo "  [${status}] ${section}: ${detail}"
}

# ---------- Cleanup (always runs) ----------
cleanup() {
    echo ""
    echo "${BOLD}[smoke_run] cleanup${RESET}"
    # Kill find_engine_pid daemon if still active
    if [ -n "${FEPID_BG:-}" ] && kill -0 "$FEPID_BG" 2>/dev/null; then
        kill "$FEPID_BG" 2>/dev/null || true
        wait "$FEPID_BG" 2>/dev/null || true
    fi
    # run_monitors.py handles its own duration via SIGTERM to child monitors
    # (gpu, proc-under-sudo, system). If we abort early via cleanup, the
    # run_monitors child inherits this script's process group and will be
    # SIGTERMed indirectly when this script exits.
    # The smoke container is always torn down here.
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    # Kill any tmux session we may have created (we don't, but defensive)
    tmux kill-session -t "smoke_run_${CELL_ID}_r${REPLICA}" 2>/dev/null || true

    echo ""
    echo "${BOLD}========== REPORT ==========${RESET}"
    for line in "${REPORT[@]}"; do
        echo "$line"
    done
    echo ""
    if [ "$NO_GO" -eq 1 ]; then
        echo "${RED}${BOLD}========== NO-GO ==========${RESET}"
        echo "Inspect: $SMOKE_DIR"
        exit 1
    else
        echo "${GREEN}${BOLD}========== GO ==========${RESET}"
        echo "Cell $CELL_ID replica $REPLICA is cleared for a long run."
        exit 0
    fi
}
trap cleanup EXIT INT TERM

echo "${BOLD}[smoke_run] cell=$CELL_ID  replica=$REPLICA  gpu=$GPU  port=$PORT${RESET}"
echo "[smoke_run] image=$IMG"
echo "[smoke_run] smoke dir=$SMOKE_DIR"

# ============================================================
# Section A. Pre-flight (system clean)
# ============================================================
check_a() {
    echo ""
    echo "${BOLD}=== Section A: pre-flight (system clean) ===${RESET}"

    # A.1 disk space
    local var_lib_gb home_gb
    var_lib_gb=$(df --output=avail -BG /var/lib | tail -1 | tr -d 'G ')
    home_gb=$(df --output=avail -BG /home | tail -1 | tr -d 'G ')
    if [ "$var_lib_gb" -lt 30 ]; then
        record FAIL "A.disk.var_lib" "free=${var_lib_gb}GB < 30GB (Docker layer storage)"
    else
        record PASS "A.disk.var_lib" "free=${var_lib_gb}GB"
    fi
    if [ "$home_gb" -lt 5 ]; then
        record FAIL "A.disk.home" "free=${home_gb}GB < 5GB (run dir + CSVs)"
    else
        record PASS "A.disk.home" "free=${home_gb}GB"
    fi

    # A.2 conflicting tmux sessions
    local tmux_conflicts
    tmux_conflicts=$(tmux ls 2>/dev/null | grep -E "monitors_${CELL_ID}|client_${CELL_ID}|pid_tracker_${CELL_ID}|smoke_run_${CELL_ID}" || true)
    if [ -n "$tmux_conflicts" ]; then
        record FAIL "A.tmux.conflict" "$(echo $tmux_conflicts | tr '\n' ' ')"
    else
        record PASS "A.tmux.conflict" "no conflicting sessions"
    fi

    # A.3 conflicting containers (same name or port)
    local same_name same_port
    same_name=$(docker ps -a --format '{{.Names}}' | grep -E "^${CONTAINER_NAME}$" || true)
    if [ -n "$same_name" ]; then
        record FAIL "A.docker.name" "container $CONTAINER_NAME exists, remove it first"
        return
    fi
    same_port=$(docker ps --format '{{.Names}} {{.Ports}}' | grep -E ":${PORT}->" || true)
    if [ -n "$same_port" ]; then
        record FAIL "A.docker.port" "host port ${PORT} bound: $same_port"
        return
    fi
    record PASS "A.docker.name_port" "no conflicting container or port binding"

    # A.4 no leftover server on the cell port
    if curl -sf --max-time 2 "$READY_URL" >/dev/null 2>&1; then
        record FAIL "A.health.leftover" "$READY_URL responded (a server is already up)"
        return
    fi
    record PASS "A.health.leftover" "no leftover server at $READY_URL"

    # A.5 stale HF cache lock files
    local stale_locks
    stale_locks=$(find "$HF_CACHE" -name "*.lock" -mmin +60 2>/dev/null | wc -l)
    if [ "$stale_locks" -gt 0 ]; then
        record FAIL "A.hf.locks" "$stale_locks stale *.lock in $HF_CACHE (>60min)"
    else
        record PASS "A.hf.locks" "no stale locks"
    fi
}

# ============================================================
# Section B. Container startup (engine sanity)
# ============================================================
check_b() {
    echo ""
    echo "${BOLD}=== Section B: container startup (engine sanity) ===${RESET}"

    # B.1 docker run (cell-specific args)
    local cmd=(docker run -d --name "$CONTAINER_NAME" --gpus "\"device=${GPU}\"" --shm-size 8g)
    case "$CELL_ID" in
        e1|a1)
            cmd+=(-p "${PORT}:8000")
            cmd+=(-v "${HF_CACHE}:/root/.cache/huggingface")
            cmd+=("$IMG")
            cmd+=(--model "Qwen/Qwen2.5-7B-Instruct"
                  --dtype "bfloat16"
                  --max-model-len "8192"
                  --gpu-memory-utilization "0.9"
                  --port "8000")
            ;;
        e2)
            cmd+=(-p "8200:8000" -p "8201:8001" -p "8202:8002")
            cmd+=(-v "${REPO_ROOT}/engines/triton_vllm/model_repository:/models")
            cmd+=(-v "${HF_CACHE}:/root/.cache/huggingface")
            cmd+=(-e "HF_HOME=/root/.cache/huggingface")
            cmd+=("$IMG")
            cmd+=(tritonserver --model-repository=/models --log-verbose=1)
            ;;
        a2)
            cmd+=(-p "8200:8000" -p "8201:8001" -p "8202:8002")
            cmd+=(-v "${REPO_ROOT}/engines/triton_vllm/model_repository:/models")
            cmd+=(-v "${HF_CACHE}:/root/.cache/huggingface")
            cmd+=(-e "HF_HOME=/root/.cache/huggingface" -e "VLLM_USE_V1=1")
            cmd+=("$IMG")
            cmd+=(tritonserver --model-repository=/models --log-verbose=1)
            ;;
        e3|e3b)
            cmd+=(-p "${PORT}:8000")
            cmd+=(-v "${HF_CACHE}:/cache/huggingface")
            cmd+=(-e "MODEL_NAME=Qwen/Qwen2.5-7B-Instruct"
                  -e "MAX_MODEL_LEN=8192"
                  -e "DTYPE=bfloat16"
                  -e "GPU_DEVICE=cuda:0"
                  -e "HF_HOME=/cache/huggingface")
            cmd+=("$IMG")
            ;;
    esac

    if ! "${cmd[@]}" >"$SMOKE_DIR/docker_run.out" 2>"$SMOKE_DIR/docker_run.err"; then
        record FAIL "B.docker.run" "$(head -3 $SMOKE_DIR/docker_run.err)"
        return
    fi
    record PASS "B.docker.run" "container $CONTAINER_NAME started"

    # B.2 poll /health
    local elapsed=0 ready=0
    while [ $elapsed -lt 180 ]; do
        if curl -sf --max-time 3 "$READY_URL" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! docker ps --format '{{.Names}}' | grep -qE "^${CONTAINER_NAME}$"; then
            record FAIL "B.health.died" "container exited during startup"
            docker logs --tail 30 "$CONTAINER_NAME" >"$SMOKE_DIR/container_died.log" 2>&1 || true
            return
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    if [ $ready -eq 0 ]; then
        record FAIL "B.health.timeout" "$READY_URL did not respond in 180s"
        return
    fi
    record PASS "B.health.ready" "ready after ${elapsed}s"

    # B.3 dump logs for inspection + engine version checks
    docker logs "$CONTAINER_NAME" >"$SMOKE_DIR/container_logs.txt" 2>&1 || true

    # B.3a critical: VLLM_USE_V1 was NOT rejected as unknown
    if grep -qiE "Unknown vLLM environment variable.*VLLM_USE_V1" "$SMOKE_DIR/container_logs.txt"; then
        record FAIL "B.engine.flag_ignored" "vLLM rejected VLLM_USE_V1: image version no longer supports the flag"
        return
    fi
    record PASS "B.engine.flag_check" "VLLM_USE_V1 not flagged as unknown"

    # B.3b expected engine generation per cell
    case "$CELL_ID" in
        e1)
            if grep -qiE "v1[^a-z0-9]*llm engine|v1 llm engine|engine.*v1" "$SMOKE_DIR/container_logs.txt"; then
                record PASS "B.engine.version" "V1 detected (expected for e1)"
            else
                record WARN "B.engine.version" "V1 marker not detected; inspect container_logs.txt"
            fi
            ;;
        a1)
            if grep -qiE "v1[^a-z0-9]*llm engine|v1 llm engine" "$SMOKE_DIR/container_logs.txt"; then
                record FAIL "B.engine.version" "V1 detected but a1 must be V0 (vllm 0.7.3)"
                return
            fi
            record PASS "B.engine.version" "no V1 marker (expected V0 for a1)"
            ;;
        a2)
            if grep -qiE "VLLM_USE_V1.*1|v1[^a-z0-9]*llm engine" "$SMOKE_DIR/container_logs.txt"; then
                record PASS "B.engine.version" "V1 effective (expected for a2 with VLLM_USE_V1=1)"
            else
                record WARN "B.engine.version" "V1 marker not detected for a2; inspect container_logs.txt"
            fi
            ;;
        e2)
            if grep -qiE "v1[^a-z0-9]*llm engine|VLLM_USE_V1.*1" "$SMOKE_DIR/container_logs.txt"; then
                record FAIL "B.engine.version" "V1 marker present but e2 must be V0 (Triton 25.09 default)"
                return
            fi
            record PASS "B.engine.version" "no V1 marker (expected V0 for e2)"
            ;;
        e3|e3b)
            record PASS "B.engine.version" "n/a (PyTorch HF naive, no scheduler)"
            ;;
    esac
}

# ============================================================
# Section C. Monitor smoke (60s sampling + 1 inference request)
# ============================================================
check_c() {
    echo ""
    echo "${BOLD}=== Section C: monitor smoke (60s + 1 request) ===${RESET}"

    local pid_file="$SMOKE_DIR/engine.pid"

    # C.1 resolve engine PID
    if [ "${ETYPE[$CELL_ID]}" = "container_pid1" ]; then
        local cpid
        cpid=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null)
        if [ -z "$cpid" ] || [ "$cpid" = "0" ]; then
            record FAIL "C.pid.resolve" "docker inspect returned no PID"
            return
        fi
        echo "$cpid" > "$pid_file"
        record PASS "C.pid.resolve" "container_pid1: $cpid"
    else
        # triton_child via find_engine_pid daemon (120s, more than enough for 60s sample)
        nohup "$WOSAR_PY" \
            "$REPO_ROOT/monitoring/find_engine_pid.py" \
            --container-name "$CONTAINER_NAME" \
            --process-pattern "EngineCore|triton_python_backend_stub" \
            --pidfile "$pid_file" \
            --poll-seconds 5 \
            --duration-seconds 120 \
            >"$SMOKE_DIR/find_engine_pid.log" 2>&1 &
        FEPID_BG=$!
        local wait=0
        while [ $wait -lt 60 ]; do
            if [ -f "$pid_file" ] && [ -s "$pid_file" ]; then
                local pid
                pid=$(cat "$pid_file" 2>/dev/null)
                if [[ "$pid" =~ ^[0-9]+$ ]]; then
                    record PASS "C.pid.resolve" "triton_child: $pid (find_engine_pid)"
                    break
                fi
            fi
            sleep 2
            wait=$((wait + 2))
        done
        if [ $wait -ge 60 ]; then
            record FAIL "C.pid.resolve" "find_engine_pid did not resolve in 60s; inspect $SMOKE_DIR/find_engine_pid.log"
            return
        fi
    fi

    # C.2 spawn run_monitors.py (orchestrator with built-in duration). It
    # spawns gpu/proc/system monitors, with proc wrapped in sudo -n. At
    # duration expiry run_monitors sends SIGTERM to the monitor process
    # groups; proc_monitor exits cleanly on its existing SIGTERM handler.
    # No new flags or signaling burdens for proc_monitor.
    local SMOKE_PARENT SMOKE_RUN_ID
    SMOKE_PARENT=$(dirname "$SMOKE_DIR")
    SMOKE_RUN_ID=$(basename "$SMOKE_DIR")
    "$WOSAR_PY" "$REPO_ROOT/monitoring/run_monitors.py" \
        --run-id "$SMOKE_RUN_ID" \
        --runs-root "$SMOKE_PARENT" \
        --gpu-index "$GPU" \
        --pidfile "$pid_file" \
        --duration-seconds 60 \
        --label-engine "$LBL" \
        >"$SMOKE_DIR/run_monitors.log" 2>&1 &
    local RUN_MON_BG=$!

    sleep 5
    if ! kill -0 "$RUN_MON_BG" 2>/dev/null; then
        record FAIL "C.run_monitors.start" "run_monitors exited early; see $SMOKE_DIR/run_monitors.log"
        return
    fi
    record PASS "C.run_monitors.start" "run_monitors orchestrator running (gpu+proc+system, proc via sudo)"

    # C.3 send one inference request mid-sampling
    local resp_status=""
    local resp_file="$SMOKE_DIR/inference_response.txt"
    case "$CELL_ID" in
        e1|a1)
            resp_status=$(curl -s -o "$resp_file" -w "%{http_code}" \
                --max-time 60 \
                -X POST "http://localhost:${PORT}/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Hello, world.","max_tokens":5,"temperature":0.1}')
            ;;
        e2|a2)
            resp_status=$(curl -s -o "$resp_file" -w "%{http_code}" \
                --max-time 60 \
                -X POST "http://localhost:${PORT}/v2/models/qwen/generate" \
                -H "Content-Type: application/json" \
                -d '{"text_input":"Hello, world.","parameters":{"max_tokens":5,"temperature":0.1,"stream":false}}')
            ;;
        e3|e3b)
            resp_status=$(curl -s -o "$resp_file" -w "%{http_code}" \
                --max-time 60 \
                -X POST "http://localhost:${PORT}/generate" \
                -H "Content-Type: application/json" \
                -d '{"prompt":"Hello, world.","max_tokens":5,"stream":false}')
            ;;
    esac
    if [[ "$resp_status" =~ ^2 ]]; then
        record PASS "C.inference.request" "HTTP $resp_status from cell endpoint"
    else
        record FAIL "C.inference.request" "HTTP $resp_status; body head: $(head -c 200 $resp_file 2>/dev/null)"
        return
    fi

    # C.4 wait for run_monitors to finish (it auto-exits at duration + grace)
    echo "[smoke_run] sampling for 60s (run_monitors will auto-exit)..."
    wait "$RUN_MON_BG" 2>/dev/null || true
    sleep 2

    # C.5 analyze the proc CSV
    local proc_csv
    proc_csv=$(ls "$SMOKE_DIR"/${LBL}_*.csv 2>/dev/null | head -1)
    if [ -z "$proc_csv" ]; then
        record FAIL "C.proc.csv_missing" "no proc CSV in $SMOKE_DIR"
        return
    fi

    local total alive rss_p50 cpu_p50
    total=$(awk -F, 'NR>1' "$proc_csv" | wc -l)
    alive=$(awk -F, 'NR>1 && $3=="True" {c++} END {print c+0}' "$proc_csv")

    if [ "$alive" -lt 8 ]; then
        record FAIL "C.proc.alive" "alive=$alive/$total (need >=8); permissions or wrong PID"
        return
    fi
    record PASS "C.proc.alive" "alive=$alive/$total"

    # rss_p50 in MB
    rss_p50=$(awk -F, 'NR>1 && $3=="True" && $4!="" && $4!="None" {print $4}' "$proc_csv" \
              | sort -n | awk '{a[NR]=$1} END {if (NR>0) printf "%.1f\n", a[int((NR+1)/2)]/1048576}')
    local rss_p50_int=${rss_p50%.*}
    if [ -z "$rss_p50_int" ] || [ "$rss_p50_int" -lt 100 ]; then
        record FAIL "C.proc.rss_size" "rss_p50=${rss_p50:-0}MB < 100MB; monitoring wrapper not engine"
        return
    fi
    record PASS "C.proc.rss_size" "rss_p50=${rss_p50}MB"

    # cpu_p50 (column 10)
    cpu_p50=$(awk -F, 'NR>1 && $3=="True" && $10!="" && $10!="None" {print $10}' "$proc_csv" \
              | sort -n | awk '{a[NR]=$1} END {if (NR>0) printf "%.2f\n", a[int((NR+1)/2)]}')
    if [ -z "$cpu_p50" ]; then
        record WARN "C.proc.cpu" "cpu_percent column empty"
    else
        local cpu_above_one
        cpu_above_one=$(awk -v c="$cpu_p50" 'BEGIN {print (c > 1) ? 1 : 0}')
        if [ "$cpu_above_one" -eq 1 ]; then
            record PASS "C.proc.cpu" "cpu_p50=${cpu_p50}%"
        else
            record WARN "C.proc.cpu" "cpu_p50=${cpu_p50}% <= 1% (low traffic in 60s window)"
        fi
    fi

    # C.6 file ownership and mode
    local file_owner file_mode
    file_owner=$(stat -c '%U:%G' "$proc_csv")
    file_mode=$(stat -c '%a' "$proc_csv")
    if [ "$file_owner" != "dcotrone:dcotrone" ]; then
        record WARN "C.proc.ownership" "owner=$file_owner (expected dcotrone:dcotrone)"
    else
        record PASS "C.proc.ownership" "owner=$file_owner"
    fi
    if [ "$file_mode" != "644" ]; then
        record WARN "C.proc.mode" "mode=$file_mode (expected 644)"
    else
        record PASS "C.proc.mode" "mode=644"
    fi
}

# ============================================================
# Section D. Aggregate sanity (live engine + gpu state)
# ============================================================
check_d() {
    echo ""
    echo "${BOLD}=== Section D: aggregate sanity ===${RESET}"

    # D.1 gpu util right now (one quick poll). Cell-side load just happened.
    local gpu_util
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    if [ -z "$gpu_util" ]; then
        record WARN "D.gpu.util" "nvidia-smi did not return utilization for gpu $GPU"
    else
        record PASS "D.gpu.util" "current utilization=${gpu_util}% on gpu $GPU"
    fi

    # D.2 vram usage on the cell's gpu (proxy for "model loaded")
    local vram_used
    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    if [ -z "$vram_used" ]; then
        record WARN "D.gpu.vram" "nvidia-smi did not return vram for gpu $GPU"
    elif [ "$vram_used" -lt 1000 ]; then
        record WARN "D.gpu.vram" "vram_used=${vram_used}MiB < 1000MiB on gpu $GPU"
    else
        record PASS "D.gpu.vram" "vram_used=${vram_used}MiB on gpu $GPU"
    fi

    # D.3 container still alive at end of smoke
    if ! docker ps --format '{{.Names}}' | grep -qE "^${CONTAINER_NAME}$"; then
        record FAIL "D.container.alive_end" "container died before end of smoke test"
    else
        record PASS "D.container.alive_end" "still running"
    fi
}

# ============================================================
# Execute
# ============================================================
check_a
if [ "$NO_GO" -eq 1 ]; then
    echo "${RED}[smoke_run] Section A failed, skipping B/C/D${RESET}"
else
    check_b
    if [ "$NO_GO" -eq 1 ]; then
        echo "${RED}[smoke_run] Section B failed, skipping C/D${RESET}"
    else
        check_c
        check_d
    fi
fi
# cleanup() runs via EXIT trap
