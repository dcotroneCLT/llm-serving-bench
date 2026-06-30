#!/usr/bin/env bash
# Consolidated box-validation gate for BATCH 1 (PGID scoping, orphan reaper,
# multi-GPU, multi-process validator) + BATCH 2 PHASE A (fail-loud bring-up,
# calibration enforcement, docker-root disk checks, monotonic durations).
#
# ONE command on cci-csgpu11. Brings the disaggregated stack up, drives a short
# VALIDATION run (not a soak), runs the validator + the empirical scoping check,
# proves the fail-loud failure path, and confirms the disk check resolves the
# real docker data-root. Cleans up (serve_down + infra_down + reaper) on exit,
# success OR failure, via a trap. Ends with a single PASS/FAIL summary.
#
#   conda activate wosar
#   bash deploy/dynamo/repass_gate2.sh            # ~30 min
#
# NOTE: deliberately NOT `set -e` -- we run every check and summarize; each step
# captures its own status.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$HOME/wosar/runs}"
DURATION_S="${DURATION_S:-1500}"
RUN_ID="${RUN_ID:-repass_val_dynamo_disagg}"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
CELL="$REPO/campaigns/extension/cells/val_dynamo_disagg.yaml"
PIDS_FILE="${WOSAR_COMPONENT_PIDS:-$HOME/wosar/dynamo_component_pids.json}"
DYNAMO_IMG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-cuda13"
TRITON_IMG="nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3"
VLLM_IMG="vllm/vllm-openai:v0.20.1-cu129"

# --- result accumulator ---
declare -A RESULT
mark() { RESULT["$1"]="$2"; printf '[repass] %-26s %s\n' "$1" "$2"; }

cleanup() {
  echo "[repass] --- cleanup (serve_down + infra_down + reaper) ---"
  bash "$HERE/serve_down.sh"  >/dev/null 2>&1 || true
  bash "$HERE/infra_down.sh"  >/dev/null 2>&1 || true
  python3 -c "import sys; sys.path.insert(0,'$REPO/scripts'); import reaper; \
print('\n'.join(reaper.reap_orphans('$RUNS_ROOT')))" 2>/dev/null || true
  docker rm -f dyn_frontend dyn_prefill_1 dyn_decode_1 dyn_etcd dyn_nats >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[repass] repo=$REPO run_dir=$RUN_DIR duration=${DURATION_S}s"

# ----- step 0: vLLM pin guard (all three images must ship 0.20.1) -----
pin_ok=1
v_dyn=$(docker run --rm "$DYNAMO_IMG" pip show vllm 2>/dev/null | awk -F': ' '/^Version/{print $2}')
v_tri=$(docker run --rm "$TRITON_IMG" pip show vllm 2>/dev/null | awk -F': ' '/^Version/{print $2}')
v_vllm=$(docker run --rm --entrypoint pip "$VLLM_IMG" show vllm 2>/dev/null | awk -F': ' '/^Version/{print $2}')
echo "[repass] vllm: dynamo=$v_dyn triton=$v_tri standalone=$v_vllm"
[ "$v_dyn" = "0.20.1" ] && [ "$v_tri" = "0.20.1" ] && [ "$v_vllm" = "0.20.1" ] || pin_ok=0
mark "pip_pin" "$([ $pin_ok = 1 ] && echo PASS || echo FAIL)"

# ----- step 1: bring up the disaggregated stack -----
bash "$HERE/infra_up.sh"
if N_PREFILL=1 N_DECODE=1 PREFILL_GPU=0 DECODE_GPU=1 bash "$HERE/serve_disaggregated.sh"; then
  mark "bringup" "PASS"
  echo "[repass] identity file ($PIDS_FILE):"; cat "$PIDS_FILE" || true
else
  mark "bringup" "FAIL"
fi

# ----- step 2-4: only if the stack came up -----
if [ "${RESULT[bringup]}" = "PASS" ]; then
  # step 2: attach a validation run
  rm -rf "$RUN_DIR"
  python3 "$REPO/scripts/attach_run.py" --cell-yaml "$CELL" \
    --base-url http://localhost:8400 --runs-root "$RUNS_ROOT" --run-id "$RUN_ID" \
    --repo-root "$REPO" --duration-seconds "$DURATION_S" --component-pids "$PIDS_FILE"

  # step 3: multi-process validator (must be PASS incl. no_orphans)
  val_log="$RUN_DIR/validate.out"
  python3 "$REPO/analysis/validate_extension_run.py" --run-dir "$RUN_DIR" --repo-root "$REPO" \
    | tee "$val_log"
  grep -q "GATE: PASS" "$val_log" && mark "validator" "PASS" || mark "validator" "FAIL"
  if grep -E "PASS +no_orphans" "$val_log" >/dev/null; then mark "no_orphans" "PASS"; else mark "no_orphans" "FAIL"; fi

  # explicit n_pids_unexpected over the engine components
  npu=$(python3 - "$RUN_DIR" <<'PY'
import csv, glob, sys
from pathlib import Path
run = Path(sys.argv[1]); worst = 0
for label in ("dynamo_frontend", "dynamo_prefill", "dynamo_decode"):
    for f in glob.glob(str(run / f"{label}_*.csv")):
        for r in csv.DictReader(open(f)):
            v = r.get("n_pids_unexpected")
            if v not in (None, ""):
                worst = max(worst, int(v))
print(worst)
PY
)
  echo "[repass] max n_pids_unexpected on engine components = $npu"
  [ "$npu" = "0" ] && mark "n_pids_unexpected_0" "PASS" || mark "n_pids_unexpected_0" "FAIL"

  # step 4: empirical scoping completeness (smaps total vs recorded-pgid aggregate).
  # Use the absolute interpreter path so sudo's sanitized PATH still finds the
  # conda python that has psutil (same pattern as the sudo'd monitor).
  PYBIN="$(python3 -c 'import sys; print(sys.executable)')"
  if sudo -E "$PYBIN" "$HERE/verify_scoping.py" --component-pids "$PIDS_FILE"; then
    mark "verify_scoping" "PASS"
  else
    mark "verify_scoping" "FAIL"   # exit(2) on INCOMPLETE
  fi

  # tear the engine down before the negative test re-uses the container names
  bash "$HERE/serve_down.sh" >/dev/null 2>&1 || true
else
  mark "validator" "SKIP"; mark "no_orphans" "SKIP"
  mark "n_pids_unexpected_0" "SKIP"; mark "verify_scoping" "SKIP"
fi

# ----- step 5: fail-loud NEGATIVE test (failure path must exit non-zero) -----
# A bogus model makes the workers exit before registering; serve_disaggregated
# must detect the dead worker and exit non-zero instead of proceeding.
echo "[repass] negative test: serve_disaggregated against a non-existent model ..."
if MODEL="wosar/does-not-exist-model" N_PREFILL=1 N_DECODE=1 PREFILL_GPU=0 DECODE_GPU=1 \
     bash "$HERE/serve_disaggregated.sh" >/dev/null 2>&1; then
  mark "fail_loud_negative" "FAIL"   # it returned 0 -> did NOT fail loud
else
  mark "fail_loud_negative" "PASS"   # non-zero exit as required
fi
bash "$HERE/serve_down.sh" >/dev/null 2>&1 || true

# ----- step 6: disk check resolves the real docker data-root -----
DR=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)
echo "[repass] DockerRootDir = '${DR:-<empty>}'"
if [ -n "$DR" ] && df -BG "$DR" >/dev/null 2>&1; then
  case "$DR" in
    /var/lib*) mark "disk_root" "WARN" ;;   # resolved, but still on /var/lib
    *)         mark "disk_root" "PASS" ;;
  esac
else
  mark "disk_root" "FAIL"
fi

# ----- summary -----
echo ""
echo "=================== REPASS GATE SUMMARY ==================="
overall=PASS
for k in pip_pin bringup validator no_orphans n_pids_unexpected_0 verify_scoping fail_loud_negative disk_root; do
  v="${RESULT[$k]:-MISSING}"
  printf '  %-22s %s\n' "$k" "$v"
  case "$v" in PASS|WARN|SKIP) ;; *) overall=FAIL ;; esac
done
echo "----------------------------------------------------------"
echo "  OVERALL: $overall"
echo "=========================================================="
[ "$overall" = "PASS" ] || exit 1
