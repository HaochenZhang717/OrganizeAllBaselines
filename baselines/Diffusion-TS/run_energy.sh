#!/usr/bin/env bash

LR=1e-4
BS=64
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="diffusion-ts"
export WANDB_NAME="energy_lr${LR}_bs${BS}"

python main.py \
  --name energy \
  --config_file Config/neurips_baselines/energy.yaml \
  --gpu 0 \
  --train \
  --lr ${LR} \
  --batch_size ${BS} \
