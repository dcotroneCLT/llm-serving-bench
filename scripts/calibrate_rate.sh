#!/usr/bin/env bash
# calibrate_rate.sh - saturation throughput sweep for one cell.
#
# Launches the cell's container once, then sends the client workload at
# increasing offered rates for short windows. For each rate, reports the
# achieved throughput (completed-OK requests / window) and the achieved/offered
# ratio. The saturation point is where achieved/offered drops below ~0.95
# AND latency takes off; 85% of saturation is the recommended steady-state
# rate for the long aging run.
#
# Usage:
#   bash scripts/calibrate_rate.sh <cell_id>
#
# Wallclock: ~30 min (cold load + 5 rates x 4min + cooldowns + teardown).

set -uo pipefail

CELL_ID="${1:?usage: calibrate_rate.sh <cell_id>}"
case "$CELL_ID" in
    e1|e2|e3|e3b|a1|a2) ;;
    *) echo "invalid cell_id '$CELL_ID'"; exit 1 ;;
esac

if [ -t 1 ]; then
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WOSAR_PY="$HOME/miniconda3/envs/wosar/bin/python3.11"
HF_CACHE="$HOME/wosar/hf_cache"
CELL_YAML="$REPO_ROOT/campaigns/wosar2026/cells/${CELL_ID}.yaml"
CALIB_DIR="$HOME/wosar/runs/calib_${CELL_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CALIB_DIR"

CONTAINER_NAME="calib_${CELL_ID}"

# Activate wosar env
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "wosar" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate wosar
fi

# ---------- Pull cell metadata via yaml parse ----------
read_yaml() {
    "$WOSAR_PY" -c "
import yaml
c = yaml.safe_load(open('$CELL_YAML'))
$1
"
}
PORT=$(read_yaml "print(c['engine']['port_mapping'][0].split(':')[0])")
GPU=$(read_yaml "print(c['engine']['gpu_device'])")
PROTOCOL=$(read_yaml "print(c['workload']['client_config_overrides']['protocol'])")
BASE_URL=$(read_yaml "print(c['workload']['client_config_overrides']['base_url'])")
MODEL=$(read_yaml "print(c['workload']['client_config_overrides']['model'])")
IMAGE_TAG=$(read_yaml "print(c['engine']['image_repo'] + ':' + c['engine']['image_tag'])")
READY_PATH=$(read_yaml "print(c['engine']['readyz']['url'].split('://')[1].split('/',1)[1])")
READY_URL="http://localhost:${PORT}/${READY_PATH#/}"
READY_TIMEOUT=$(read_yaml "print(c['engine']['readyz']['timeout_s'])")

# Per-cell docker run args
docker_run_cmd() {
    local cmd=(docker run -d --name "$CONTAINER_NAME" --gpus "\"device=${GPU}\"" --shm-size 8g)
    case "$CELL_ID" in
        e1|a1)
            cmd+=(-p "${PORT}:8000" -v "${HF_CACHE}:/root/.cache/huggingface")
            cmd+=("$IMAGE_TAG"
                  --model "Qwen/Qwen2.5-7B-Instruct"
                  --dtype "bfloat16"
                  --max-model-len "8192"
                  --gpu-memory-utilization "0.9"
                  --port "8000")
            ;;
        e2)
            cmd+=(-p "8200:8000" -p "8201:8001" -p "8202:8002"
                  -v "${REPO_ROOT}/engines/triton_vllm/model_repository:/models"
                  -v "${HF_CACHE}:/root/.cache/huggingface"
                  -e "HF_HOME=/root/.cache/huggingface"
                  "$IMAGE_TAG"
                  tritonserver --model-repository=/models --log-verbose=1)
            ;;
        a2)
            cmd+=(-p "8200:8000" -p "8201:8001" -p "8202:8002"
                  -v "${REPO_ROOT}/engines/triton_vllm/model_repository:/models"
                  -v "${HF_CACHE}:/root/.cache/huggingface"
                  -e "HF_HOME=/root/.cache/huggingface"
                  -e "VLLM_USE_V1=1"
                  "$IMAGE_TAG"
                  tritonserver --model-repository=/models --log-verbose=1)
            ;;
        e3|e3b)
            cmd+=(-p "${PORT}:8000"
                  -v "${HF_CACHE}:/cache/huggingface"
                  -e "MODEL_NAME=Qwen/Qwen2.5-7B-Instruct"
                  -e "MAX_MODEL_LEN=8192"
                  -e "DTYPE=bfloat16"
                  -e "GPU_DEVICE=cuda:0"
                  -e "HF_HOME=/cache/huggingface"
                  "$IMAGE_TAG")
            ;;
    esac
    printf '%s ' "${cmd[@]}"; echo
}

# ---------- Sweep parameters ----------
# Geometric-ish sweep covering a wide range. For vLLM engines V1 saturates
# around 3-5 rps on Qwen 7B + L40S, V0 around 1-2 rps. For PyTorch naive,
# saturation is ~0.2 rps. The sweep below covers all of these.
RATES=(0.25 0.5 1.0 2.0 4.0 8.0)
WINDOW_S=240            # 4 min per rate
COOLDOWN_S=30           # idle between rates so KV cache flushes
CONCURRENCY_CAP=64

echo "${BOLD}[calib] cell=$CELL_ID  port=$PORT  gpu=$GPU${RESET}"
echo "[calib] image=$IMAGE_TAG"
echo "[calib] sweep rates: ${RATES[*]} rps  window=${WINDOW_S}s  cooldown=${COOLDOWN_S}s"
echo "[calib] output dir: $CALIB_DIR"

# ---------- Cleanup hook ----------
cleanup() {
    echo ""
    echo "[calib] cleanup..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ---------- Section 1: container startup ----------
echo ""
echo "${BOLD}[calib] starting container $CONTAINER_NAME${RESET}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
DOCKER_CMD=$(docker_run_cmd)
echo "[calib] docker cmd: $DOCKER_CMD"
eval "$DOCKER_CMD" >"$CALIB_DIR/docker_run.out" 2>"$CALIB_DIR/docker_run.err" || {
    echo "${RED}[calib] FATAL: docker run failed${RESET}"
    cat "$CALIB_DIR/docker_run.err"
    exit 1
}

echo "[calib] waiting for ${READY_URL} (timeout ${READY_TIMEOUT}s)"
elapsed=0
while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
    if curl -sf --max-time 3 "$READY_URL" >/dev/null 2>&1; then
        echo "${GREEN}[calib] ready after ${elapsed}s${RESET}"
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qE "^${CONTAINER_NAME}$"; then
        echo "${RED}[calib] FATAL: container exited during startup${RESET}"
        docker logs --tail 30 "$CONTAINER_NAME" 2>&1 || true
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
    echo "${RED}[calib] FATAL: readyz did not respond in ${READY_TIMEOUT}s${RESET}"
    exit 1
fi

# ---------- Section 2: rate sweep ----------
SUMMARY="$CALIB_DIR/summary.tsv"
echo -e "offered_rate\tcompleted_ok\twindow_s\tachieved_rps\tachieved_over_offered\tmean_e2e_s\tp95_e2e_s" > "$SUMMARY"

for rate in "${RATES[@]}"; do
    sub="$CALIB_DIR/rate_${rate}"
    mkdir -p "$sub"
    echo ""
    echo "${BOLD}[calib] rate=${rate} rps for ${WINDOW_S}s${RESET}"

    "$WOSAR_PY" "$REPO_ROOT/client/run_client.py" \
        --config "$REPO_ROOT/client/config.yaml" \
        --output-dir "$sub" \
        --duration-seconds "$WINDOW_S" \
        --protocol "$PROTOCOL" \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --target-rate-rps "$rate" \
        --concurrency-cap "$CONCURRENCY_CAP" \
        >"$sub/client.log" 2>&1 || {
        echo "[calib] WARN: client exited non-zero at rate=$rate"
    }

    # Aggregate stats from the requests CSVs in this sub
    stats=$("$WOSAR_PY" - "$sub" "$WINDOW_S" <<'PY'
import csv, glob, sys, statistics
sub, window_s = sys.argv[1], float(sys.argv[2])
files = sorted(glob.glob(sub + "/requests_*.csv"))
ok = []
e2e = []
for f in files:
    with open(f) as fp:
        for r in csv.DictReader(fp):
            status = r.get("status", "")
            if status.lower() == "ok" or status == "success":
                ok.append(r)
                try:
                    e2e.append(float(r["e2e_latency_s"]))
                except (ValueError, KeyError, TypeError):
                    pass
n_ok = len(ok)
achieved_rps = n_ok / window_s
e2e.sort()
mean_e2e = statistics.mean(e2e) if e2e else 0.0
p95_e2e = e2e[int(0.95 * len(e2e))] if e2e else 0.0
print(f"{n_ok}\t{achieved_rps:.3f}\t{mean_e2e:.3f}\t{p95_e2e:.3f}")
PY
)
    n_ok=$(echo "$stats" | cut -f1)
    achieved=$(echo "$stats" | cut -f2)
    mean_e2e=$(echo "$stats" | cut -f3)
    p95_e2e=$(echo "$stats" | cut -f4)
    ratio=$(awk -v a="$achieved" -v o="$rate" 'BEGIN { if (o>0) printf "%.3f", a/o; else print "0.000" }')
    echo "[calib] rate=$rate  achieved=$achieved rps  ratio=$ratio  mean_e2e=${mean_e2e}s  p95_e2e=${p95_e2e}s  n_ok=$n_ok"
    echo -e "$rate\t$n_ok\t$WINDOW_S\t$achieved\t$ratio\t$mean_e2e\t$p95_e2e" >> "$SUMMARY"

    sleep "$COOLDOWN_S"
done

# ---------- Section 3: saturation analysis ----------
echo ""
echo "${BOLD}========== SUMMARY ==========${RESET}"
column -t -s $'\t' "$SUMMARY"

echo ""
echo "${BOLD}========== SATURATION ANALYSIS ==========${RESET}"
"$WOSAR_PY" - "$SUMMARY" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
# Saturation: largest offered rate where achieved/offered >= 0.95
sat_offered = 0.0
sat_achieved = 0.0
for r in rows:
    offered = float(r["offered_rate"])
    achieved = float(r["achieved_rps"])
    ratio = float(r["achieved_over_offered"])
    if ratio >= 0.95 and achieved > sat_achieved:
        sat_offered = offered
        sat_achieved = achieved
recommended = round(sat_achieved * 0.85, 3)
print(f"Saturation (last rate with achieved/offered >= 0.95):")
print(f"  offered = {sat_offered:.2f} rps")
print(f"  achieved = {sat_achieved:.3f} rps")
print(f"")
print(f"Recommended steady-state target (85% of saturation achieved):")
print(f"  target_rate_rps = {recommended}")
PY

echo ""
echo "[calib] full output in: $CALIB_DIR"
exit 0
