import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')
    # training parameters
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    # args for random
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--milestone', type=str, default="best")
    parser.add_argument('--fid_vae_ckpt', type=str, required=True)


    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)



    args = parser.parse_args()
    # args.save_dir = os.path.join(args.output, f'{args.name}')
    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    args.save_dir = config['solver']['results_folder']
    config["solver"]["base_lr"] = args.lr
    config["dataloader"]["batch_size"] = args.batch_size
    config['solver']['results_folder'] = f"{config['solver']['results_folder']}/LR{config['solver']['base_lr']}-BS{config['dataloader']['batch_size']}"

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).to(device)

    # if args.sample == 1 and args.mode in ['infill', 'predict']:
    #     test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info)

    save_dir = config['solver']['results_folder']
    trainer.train()
    # trainer.load(args.milestone)
    # dataset = dataloader_info['valid_dataset']
    # window, var_num = dataset[0].shape
    # samples = trainer.sample(num=len(dataset), size_every=2001, shape=[window, var_num])
    # np.save(os.path.join(save_dir, f'ddpm_fake_{args.name}.npy'), samples)


if __name__ == '__main__':
    main()
