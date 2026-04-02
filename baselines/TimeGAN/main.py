import os
import sys
import argparse
import yaml
import torch
import numpy as np
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

# ── resolve project root so models/ is importable ────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.model import Encoder, Recovery, Generator, Supervisor, Discriminator
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

    # ── build opt namespace expected by model modules ─────────────────────────
    m_cfg = config['model']
    opt = SimpleNamespace(
        z_dim=m_cfg['feature_size'],
        hidden_dim=m_cfg['hidden_dim'],
        num_layer=m_cfg['num_layer'],
    )

    # ── networks ──────────────────────────────────────────────────────────────
    nets = {
        'encoder':      Encoder(opt),
        'recovery':     Recovery(opt),
        'generator':    Generator(opt),
        'supervisor':   Supervisor(opt),
        'discriminator': Discriminator(opt),
    }

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        config=config,
        args=args,
        nets=nets,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    trainer.train()


if __name__ == '__main__':
    main()
