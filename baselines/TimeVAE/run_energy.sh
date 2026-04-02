#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="timevae"
export WANDB_NAME="energy"

python main.py \
  --name energy \
  --config_file Config/energy.yaml
