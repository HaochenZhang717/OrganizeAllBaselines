#!/usr/bin/env bash
# run_all_baselines.sh
# Launches all baseline training jobs distributed across multiple GPUs.
#
# Usage:
#   bash run_all_baselines.sh [GPU_IDS]
#
#   GPU_IDS : comma-separated GPU indices to use (default: 0,1)
#
# Examples:
#   bash run_all_baselines.sh          # uses GPUs 0,1
#   bash run_all_baselines.sh 0,1,2,3  # spread across 4 GPUs
#   bash run_all_baselines.sh 0        # all jobs queued serially on GPU 0
#
# Each job's stdout/stderr is saved to logs/<baseline>__<script>.log
# TimeLDM's two stages (vae → ldm) run sequentially on the same GPU.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="${ROOT_DIR}/baselines"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── GPU list ──────────────────────────────────────────────────────────────────
GPU_STR="${1:-0,1}"
IFS=',' read -ra GPUS <<< "${GPU_STR}"
NUM_GPUS=${#GPUS[@]}
echo "GPUs to use : ${GPUS[*]}"
echo "# of tasks  : will be shown below"
echo ""

# ── Task definitions ──────────────────────────────────────────────────────────
# Format: "BaselineDir:script1.sh[:script2.sh:...]"
# Multiple scripts for the same entry run SEQUENTIALLY on one GPU (e.g. TimeLDM).
TASKS=(
  "Diffusion-TS:run_etth.sh"
  "Diffusion-TS:run_energy.sh"
  # "TimeGAN:run_etth.sh"
  # "TimeGAN:run_energy.sh"
  "TimeVAE:run_etth.sh"
  "TimeVAE:run_energy.sh"
  # "TimeLDM:run_energy_vae.sh:run_energy_ldm.sh"   # stage-1 then stage-2
  # "TimeLDM:run_etth1_vae.sh:run_etth1_ldm.sh"   # stage-1 then stage-2
)

# ── GPU-aware runner ──────────────────────────────────────────────────────────
# Each individual script hard-codes "export CUDA_VISIBLE_DEVICES=0".
# We strip that line so our GPU assignment (set before calling this function)
# takes effect instead.
# Note: Diffusion-TS also passes "--gpu 0" to Python; because we expose exactly
# one GPU via CUDA_VISIBLE_DEVICES, that GPU always appears as index 0 inside
# the process, so the flag remains correct for any physical GPU we assign.
run_scripts_on_gpu() {
  local work_dir="$1"
  local gpu="$2"
  shift 2
  local scripts=("$@")

  cd "${work_dir}"
  export CUDA_VISIBLE_DEVICES="${gpu}"

  for script in "${scripts[@]}"; do
    echo "[$(date '+%H:%M:%S')] Starting: ${script}  (GPU ${gpu})"
    bash <(sed '/^[[:space:]]*export[[:space:]]CUDA_VISIBLE_DEVICES=/d' "${script}")
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
      echo "[$(date '+%H:%M:%S')] FAILED: ${script} exited with code ${rc}"
      return ${rc}
    fi
    echo "[$(date '+%H:%M:%S')] Finished: ${script}"
  done
}

# Export so the function is available inside subshells launched with ( ... )
export -f run_scripts_on_gpu

# ── Launch all tasks in parallel ──────────────────────────────────────────────
PIDS=()
GPU_IDX=0

for TASK in "${TASKS[@]}"; do
  IFS=':' read -ra PARTS <<< "${TASK}"
  BASELINE="${PARTS[0]}"
  SCRIPTS=("${PARTS[@]:1}")
  GPU="${GPUS[$((GPU_IDX % NUM_GPUS))]}"
  GPU_IDX=$((GPU_IDX + 1))

  # Build a readable log filename, e.g. TimeGAN__run_etth.log
  SCRIPT_TAG="$(IFS=_; echo "${SCRIPTS[*]}")"
  LOG_FILE="${LOG_DIR}/${BASELINE}__${SCRIPT_TAG%.sh}.log"

  printf "  [GPU %s] %-14s  %-50s  log: %s\n" \
    "${GPU}" "${BASELINE}" "${SCRIPTS[*]}" "logs/$(basename "${LOG_FILE}")"

  (
    run_scripts_on_gpu \
      "${BASELINES_DIR}/${BASELINE}" \
      "${GPU}" \
      "${SCRIPTS[@]}"
  ) > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "All ${#TASKS[@]} jobs launched (PIDs: ${PIDS[*]})"
echo "Waiting for completion..."
echo ""

# ── Collect results ───────────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  TASK="${TASKS[$i]}"
  if wait "${PID}"; then
    printf "[DONE]   %s\n" "${TASK}"
  else
    printf "[FAILED] %s\n" "${TASK}"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [[ ${FAILED} -eq 0 ]]; then
  echo "All baselines completed successfully."
else
  echo "${FAILED} job(s) failed. Check logs in ${LOG_DIR}/"
  exit 1
fi