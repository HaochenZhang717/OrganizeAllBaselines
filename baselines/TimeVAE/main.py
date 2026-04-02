import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ── resolve project root so models/ is importable ────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.timevae import TimeVAE
from engine.solver import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',        type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--seed',        type=int, default=12345)
    return parser.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npy_loader(path: str, batch_size: int, shuffle: bool) -> DataLoader:
    data = np.load(path).astype(np.float32)       # (N, T, C)
    tensor = torch.from_numpy(data)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True,
                      drop_last=shuffle)


def main():
    args = parse_args()
    seed_everything(args.seed)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # ── dataloaders ───────────────────────────────────────────────────────────
    dl_cfg = config['dataloader']
    train_loader = load_npy_loader(dl_cfg['train_data'], dl_cfg['batch_size'], shuffle=True)
    valid_loader = load_npy_loader(dl_cfg['valid_data'], dl_cfg['batch_size'], shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    m_cfg = config['model']
    model = TimeVAE(
        seq_len=m_cfg['seq_len'],
        feat_dim=m_cfg['feat_dim'],
        latent_dim=m_cfg['latent_dim'],
        reconstruction_wt=m_cfg.get('reconstruction_wt', 3.0),
        hidden_layer_sizes=m_cfg.get('hidden_layer_sizes', [32, 64, 128]),
        trend_poly=m_cfg.get('trend_poly', 0),
        custom_seas=m_cfg.get('custom_seas', None),
        use_residual_conn=m_cfg.get('use_residual_conn', True),
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        config=config,
        args=args,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    trainer.train()


if __name__ == '__main__':
    main()