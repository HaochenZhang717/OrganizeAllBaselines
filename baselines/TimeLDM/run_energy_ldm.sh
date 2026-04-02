#!/usr/bin/env bash
# Stage 2: Train the LDM denoiser using a pre-trained VAE.
# Update VAE_CKPT to the actual checkpoint path after Stage 1 completes.

LR=1e-4
BS=64
VAE_CKPT=./Checkpoints_energy/vae/LR1e-3-BS64/checkpoint-best.pt

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="timeldm"
export WANDB_NAME="energy_ldm_lr${LR}_bs${BS}"

python main.py \
  --name energy \
  --config_file Config/energy.yaml \
  --stage ldm \
  --train \
  --lr ${LR} \
  --batch_size ${BS} \
  --vae_ckpt ${VAE_CKPT}
