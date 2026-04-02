"""
Entry point for TimeLDM training.

Two stages:
  --stage vae   Train the β-VAE (encoder + decoder).
  --stage ldm   Train the LDM denoiser with a frozen, pre-trained VAE.
                Requires --vae_ckpt pointing to a VAE checkpoint.
"""

import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import VAETrainer, LDMTrainer
from Data.build_dataloader import build_dataloader
from Utils.io_utils import (
    load_yaml_config,
    seed_everything,
    merge_opts_to_config,
    instantiate_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description='TimeLDM Training')

    parser.add_argument('--name',        type=str,   default=None)
    parser.add_argument('--config_file', type=str,   required=True)
    parser.add_argument('--output',      type=str,   default='OUTPUT')
    parser.add_argument('--tensorboard', action='store_true')

    # Stage selection
    parser.add_argument('--stage', type=str, required=True, choices=['vae', 'ldm'],
                        help='Training stage: "vae" to train the VAE, '
                             '"ldm" to train the LDM denoiser.')

    # For LDM stage: path to a trained VAE checkpoint
    parser.add_argument('--vae_ckpt', type=str, default=None,
                        help='Path to VAE checkpoint (required for --stage ldm).')

    # Training flags
    parser.add_argument('--train',     action='store_true', default=False)
    parser.add_argument('--milestone', type=str, default='best')

    # Hyper-parameters (can also be set via config)
    parser.add_argument('--lr',         type=float, required=True)
    parser.add_argument('--batch_size', type=int,   required=True)

    # Reproducibility
    parser.add_argument('--seed', type=int,  default=12345)
    parser.add_argument('--gpu',  type=int,  default=None)
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False)

    # Config overrides (key value pairs)
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help='Override config entries: key value ...')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == 'ldm' and args.vae_ckpt is None and args.train:
        raise ValueError('--vae_ckpt is required when --stage ldm and --train.')

    if args.seed is not None:
        seed_everything(args.seed, cudnn_deterministic=args.cudnn_deterministic)

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    # Inject CLI hyper-parameters into the correct solver stage block
    stage_cfg = config['solver'][args.stage]
    stage_cfg['base_lr'] = args.lr
    config['dataloader']['batch_size'] = args.batch_size

    # Derive save_dir (used by Logger and dataloader)
    base_folder = stage_cfg['results_folder']
    run_tag = f'LR{args.lr}-BS{args.batch_size}'
    stage_cfg['results_folder'] = os.path.join(base_folder, run_tag)
    args.save_dir = os.path.dirname(base_folder)   # parent for config/log artefacts

    logger = Logger(args)
    logger.save_config(config)

    dataloader_info = build_dataloader(config, args)

    # ── Stage: VAE ───────────────────────────────────────────────────────────
    if args.stage == 'vae':
        vae = instantiate_from_config(config['vae_model']).to(device)

        trainer = VAETrainer(config=config, args=args, vae=vae,
                             dataloader=dataloader_info)

        if args.train:
            trainer.train()
        else:
            trainer.load(args.milestone)
            # Optionally add eval-only logic here

    # ── Stage: LDM ───────────────────────────────────────────────────────────
    elif args.stage == 'ldm':
        vae          = instantiate_from_config(config['vae_model']).to(device)
        ldm_denoiser = instantiate_from_config(config['ldm_model']).to(device)

        # Load the trained VAE (use EMA weights)
        vae_data = torch.load(args.vae_ckpt, map_location=device)
        vae.load_state_dict(vae_data['vae'])
        print(f'Loaded VAE from {args.vae_ckpt}')

        trainer = LDMTrainer(config=config, args=args, vae=vae,
                             ldm_denoiser=ldm_denoiser,
                             dataloader=dataloader_info)

        if args.train:
            trainer.train()
        else:
            trainer.load(args.milestone)
            dataset   = dataloader_info['valid_dataset']
            window, var_num = dataset[0].shape
            samples = trainer.sample(
                num=len(dataset), size_every=128,
                shape=[window, var_num],
            )
            save_path = os.path.join(
                stage_cfg['results_folder'],
                f'timeldm_fake_{args.name}.npy',
            )
            np.save(save_path, samples)
            print(f'Saved {len(samples)} samples to {save_path}')


if __name__ == '__main__':
    main()
