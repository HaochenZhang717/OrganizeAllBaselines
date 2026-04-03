#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="timevae"
export WANDB_NAME="energy"

# ── Tunable hyperparameters ───────────────────────────────────────────────────
LATENT_DIM=32
KL_WT=0.01
HIDDEN_LAYER_SIZES="64 128 256"
# custom_seas: flat pairs of (num_seasons, len_per_season)
# e.g. "24 1 7 24" for daily (24×1) + weekly (7×24) seasonality
# Leave empty to disable seasonal components
CUSTOM_SEAS=""

# ── Build optional args ───────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "${CUSTOM_SEAS}" ]; then
  EXTRA_ARGS="--custom_seas ${CUSTOM_SEAS}"
fi

python main.py \
  --name energy \
  --config_file Config/energy.yaml \
  --latent_dim     ${LATENT_DIM} \
  --kl_wt          ${KL_WT} \
  --hidden_layer_sizes ${HIDDEN_LAYER_SIZES} \
  ${EXTRA_ARGS}