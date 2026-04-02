#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="timegan"
export WANDB_NAME="etth1"

python main.py \
  --name etth1 \
  --config_file Config/etth.yaml
