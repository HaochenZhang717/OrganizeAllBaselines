#!/usr/bin/env bash
# Stage 1: Train the β-VAE on the energy dataset.

LR=1e-3
BS=64
DATASET=etth1
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="timeldm"
export WANDB_NAME="${DATASET}_vae_lr${LR}_bs${BS}"

python main.py \
  --name ${DATASET} \
  --config_file Config/${DATASET}.yaml \
  --stage vae \
  --train \
  --lr ${LR} \
  --batch_size ${BS}
