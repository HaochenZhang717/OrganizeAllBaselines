#!/usr/bin/env bash
# run_energy_sweep.sh
# Grid sweep over lr, batch_size, d_model, n_layer_enc/dec, n_heads,
# and mlp_hidden_times for the energy dataset.
# Jobs are dispatched in parallel, one per GPU slot.
#
# Usage:
#   bash run_energy_sweep.sh [NUM_GPUS]   (default: 8)

NUM_GPUS="${1:-8}"
export WANDB_PROJECT="diffusion-ts_energy_sweep"

# ── Search grids ──────────────────────────────────────────────────────────────
LRS=(
  1e-4
  1e-5
)
BATCH_SIZES=(
  64
  128
)
D_MODELS=(
  64
  96
  128
)
# Paired (n_layer_enc, n_layer_dec) configs
ENC_LAYERS=(2 4 4)
DEC_LAYERS=(2 2 3)
N_HEADS_LIST=(
  4
  8
)
MLP_HIDDEN_TIMES_LIST=(
  4
  8
)

# ── GPU slot tracker ──────────────────────────────────────────────────────────
declare -a GPU_PIDS
for i in $(seq 0 $(( NUM_GPUS - 1 ))); do GPU_PIDS[$i]=""; done

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

# ── Count total runs ──────────────────────────────────────────────────────────
N_ARCH=${#ENC_LAYERS[@]}
TOTAL=$(( ${#LRS[@]} * ${#BATCH_SIZES[@]} * N_ARCH * ${#N_HEADS_LIST[@]} * ${#MLP_HIDDEN_TIMES_LIST[@]} ))
echo "Total runs : ${TOTAL}"
echo "GPUs in use: ${NUM_GPUS}"
echo ""

mkdir -p sweep_logs

RUN_ID=0
for LR in "${LRS[@]}"; do
for BS in "${BATCH_SIZES[@]}"; do
for (( ai=0; ai<N_ARCH; ai++ )); do
  N_ENC="${ENC_LAYERS[$ai]}"
  N_DEC="${DEC_LAYERS[$ai]}"
for D_MODEL in "${D_MODELS[@]}"; do
for N_HEADS in "${N_HEADS_LIST[@]}"; do
for MLP in "${MLP_HIDDEN_TIMES_LIST[@]}"; do

  RUN_ID=$(( RUN_ID + 1 ))

  RUN_NAME="energy_lr${LR}_bs${BS}_d${D_MODEL}_enc${N_ENC}_dec${N_DEC}_h${N_HEADS}_mlp${MLP}"
  RESULTS_FOLDER="./Checkpoints_sweep/${RUN_NAME}"
  LOG_FILE="sweep_logs/${RUN_NAME}.log"

  SLOT=$(get_free_gpu)
  echo "[${RUN_ID}/${TOTAL}] GPU ${SLOT}  ${RUN_NAME}"

  (
    export CUDA_VISIBLE_DEVICES="${SLOT}"
    export WANDB_NAME="${RUN_NAME}"
    python main.py \
      --name             "${RUN_NAME}" \
      --config_file      Config/neurips_baselines/energy.yaml \
      --gpu              0 \
      --train \
      --lr               ${LR} \
      --batch_size       ${BS} \
      --results_folder   "${RESULTS_FOLDER}" \
      --d_model          ${D_MODEL} \
      --n_layer_enc      ${N_ENC} \
      --n_layer_dec      ${N_DEC} \
      --n_heads          ${N_HEADS} \
      --mlp_hidden_times ${MLP}
  ) > "${LOG_FILE}" 2>&1 &

  GPU_PIDS[$SLOT]=$!

done
done
done
done
done
done

echo ""
echo "All runs launched. Waiting for remaining jobs..."
wait
echo "Sweep complete."