#!/usr/bin/env bash

LR=1e-5
BS=128
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="diffusion-ts"
export WANDB_NAME="etth1_lr${LR}_bs${BS}"

python main.py \
  --name etth1 \
  --config_file Config/neurips_baselines/etth.yaml \
  --gpu 0 \
  --train \
  --lr ${LR} \
  --batch_size ${BS} \
  --fid_vae_ckpt "../../fid_vae/vae_ckpt/best.pt"
