#!/usr/bin/env bash
# run_energy_sweep.sh
# Grid sweep over latent_dim, hidden_layer_sizes, kl_wt, lr, and custom_seas.
# Jobs are dispatched in parallel, one per GPU slot.
#
# Usage:
#   bash run_energy_sweep.sh [NUM_GPUS]   (default: 8)

NUM_GPUS="${1:-8}"
export WANDB_PROJECT="timevae_energy_sweep"

# ── Search grids ──────────────────────────────────────────────────────────────
LATENT_DIMS=(
#  32
#  64
  128
)
HIDDEN_CONFIGS=(
#  "32 64 128"
#  "64 128 256"
  "128 256 512"
  "256 256 512"
)
KL_WTS=(
  0.001
  0.01
#  0.1
)
LRS=(
  0.0001
  0.001
)
# Each entry is a string of flat (num_seasons len_per_season) pairs,
# or "none" to disable seasonal components.
CUSTOM_SEAS_LIST=(
  "none"
#  "24 1"          # daily only
#  "24 1 7 24"     # daily + weekly
)

# ── GPU slot tracker ──────────────────────────────────────────────────────────
# GPU_PIDS[i] holds the PID of the job currently running on GPU i (empty = free)
declare -a GPU_PIDS
for i in $(seq 0 $(( NUM_GPUS - 1 ))); do GPU_PIDS[$i]=""; done

# Returns the index of a free GPU slot, blocking until one is available.
get_free_gpu() {
  while true; do
    for i in $(seq 0 $(( NUM_GPUS - 1 ))); do
      local pid="${GPU_PIDS[$i]}"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$i"
        return
      fi
    done
    sleep 5
  done
}

# ── Build run list and dispatch ───────────────────────────────────────────────
TOTAL=$(( ${#LATENT_DIMS[@]} * ${#HIDDEN_CONFIGS[@]} * ${#KL_WTS[@]} * ${#LRS[@]} * ${#CUSTOM_SEAS_LIST[@]} ))
echo "Total runs : ${TOTAL}"
echo "GPUs in use: ${NUM_GPUS}"
echo ""

mkdir -p sweep_logs

RUN_ID=0
for LATENT_DIM in "${LATENT_DIMS[@]}"; do
for HIDDEN in "${HIDDEN_CONFIGS[@]}"; do
for KL_WT in "${KL_WTS[@]}"; do
for LR in "${LRS[@]}"; do
for SEAS in "${CUSTOM_SEAS_LIST[@]}"; do

  RUN_ID=$(( RUN_ID + 1 ))

  HIDDEN_TAG="${HIDDEN// /-}"
  SEAS_TAG="${SEAS// /-}"
  RUN_NAME="energy_ld${LATENT_DIM}_h${HIDDEN_TAG}_kl${KL_WT}_lr${LR}_s${SEAS_TAG}"
  RESULTS_FOLDER="./Checkpoints_sweep/${RUN_NAME}"
  LOG_FILE="sweep_logs/${RUN_NAME}.log"

  if [ "${SEAS}" = "none" ]; then
    SEAS_ARG=""
  else
    SEAS_ARG="--custom_seas ${SEAS}"
  fi

  SLOT=$(get_free_gpu)
  GPU_ID="${SLOT}"
  echo "[${RUN_ID}/${TOTAL}] GPU ${GPU_ID}  ${RUN_NAME}"

  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export WANDB_NAME="${RUN_NAME}"
    python main.py \
      --name           "${RUN_NAME}" \
      --config_file    Config/energy.yaml \
      --latent_dim     ${LATENT_DIM} \
      --hidden_layer_sizes ${HIDDEN} \
      --kl_wt          ${KL_WT} \
      --lr             ${LR} \
      --results_folder "${RESULTS_FOLDER}" \
      ${SEAS_ARG}
  ) > "${LOG_FILE}" 2>&1 &

  GPU_PIDS[$SLOT]=$!

done
done
done
done
done

# ── Wait for all remaining jobs ───────────────────────────────────────────────
echo ""
echo "All runs launched. Waiting for remaining jobs..."
wait
echo "Sweep complete."