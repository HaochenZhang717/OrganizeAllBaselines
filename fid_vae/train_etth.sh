#!/usr/bin/env bash
# Train FID VAE feature extractor on ETTh1

export CUDA_VISIBLE_DEVICES=0

python train_fid_vae.py \
  --train_path ../processed/ETTh1/train_ts.npy \
  --val_path   ../processed/ETTh1/valid_ts.npy \
  --batch_size 128 \
  --epochs     200 \
  --lr         1e-3 \
  --hidden_size 128 \
  --num_layers  2 \
  --num_heads   8 \
  --latent_dim  128 \
  --beta        0.01 \
  --save_dir    ./vae_ckpts/etth1 \
  --device      cuda