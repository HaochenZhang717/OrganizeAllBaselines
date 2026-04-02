"""
Stage 1: Train the β-VAE for TimeLDM.

Usage:
    python train_vae.py --dataset sines --epochs 2000
    python train_vae.py --dataset stocks --data_path /path/to/stocks.csv --epochs 2000
    python train_vae.py --dataset etth   --data_path /path/to/ETTh1.csv  --seq_len 64
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.datasets import get_dataset, get_dataloader, DATASET_DEFAULTS
from models.vae import VAE
from losses import vae_loss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Train TimeLDM VAE (Stage 1)")

    # Data
    parser.add_argument("--dataset", type=str, default="sines",
                        choices=["sines", "mujoco", "stocks", "etth", "fmri"],
                        help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset file (not needed for sines)")
    parser.add_argument("--seq_len", type=int, default=24,
                        help="Sequence length / window size")
    parser.add_argument("--n_sines_samples", type=int, default=10000,
                        help="Number of samples for synthetic Sines dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/test split ratio")

    # Model architecture
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent space dimension m (num_heads * head_dim)")
    parser.add_argument("--num_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=None,
                        help="Transformer encoder layers (default: dataset preset)")
    parser.add_argument("--dec_layers", type=int, default=None,
                        help="Transformer decoder layers (default: dataset preset)")
    parser.add_argument("--dim_feedforward", type=int, default=None,
                        help="FFN hidden dim in transformer (default: 4*latent_dim)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")

    # Loss weights
    parser.add_argument("--lambda1", type=float, default=1.0, help="L2 loss weight")
    parser.add_argument("--lambda2", type=float, default=1.0, help="L1 loss weight")
    parser.add_argument("--lambda3", type=float, default=1.0, help="FFT loss weight")

    # β-VAE schedule
    parser.add_argument("--beta_max", type=float, default=1e-2,
                        help="Initial β for KL loss")
    parser.add_argument("--beta_min", type=float, default=1e-5,
                        help="Minimum β")
    parser.add_argument("--beta_lambda", type=float, default=0.7,
                        help="β decay factor when recon stagnates")
    parser.add_argument("--beta_patience", type=int, default=50,
                        help="Steps before decaying β")

    # Training
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: dataset preset)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for Adam")
    parser.add_argument("--lr_scheduler", action="store_true",
                        help="Use cosine annealing LR scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Print loss every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / mps / cpu (auto-detected if not set)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset defaults for per-dataset layer counts
# ---------------------------------------------------------------------------

ENC_LAYERS_DEFAULT = {"sines": 1, "mujoco": 1, "stocks": 2, "etth": 2, "fmri": 1}
DEC_LAYERS_DEFAULT = {"sines": 2, "mujoco": 2, "stocks": 3, "etth": 3, "fmri": 2}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    train_data, test_data, scaler = get_dataset(
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        n_sines_samples=args.n_sines_samples,
        seed=args.seed,
    )
    input_dim = train_data.shape[2]
    batch_size = args.batch_size or DATASET_DEFAULTS[args.dataset]["batch_size"]
    batch_size = min(batch_size, len(train_data))

    train_loader = get_dataloader(train_data, batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset} | train={len(train_data)} test={len(test_data)} "
          f"| input_dim={input_dim} | seq_len={args.seq_len}")

    # Model
    enc_layers = args.enc_layers or ENC_LAYERS_DEFAULT[args.dataset]
    dec_layers = args.dec_layers or DEC_LAYERS_DEFAULT[args.dataset]

    model = VAE(
        input_dim=input_dim,
        seq_len=args.seq_len,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        beta_max=args.beta_max,
        beta_min=args.beta_min,
        beta_lambda=args.beta_lambda,
        beta_patience=args.beta_patience,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = (CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.lr_scheduler else None)

    os.makedirs(args.save_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.save_dir, f"vae_{args.dataset}_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save scaler
    mins, maxs = scaler
    np.save(os.path.join(args.save_dir, f"scaler_{args.dataset}_mins.npy"), mins)
    np.save(os.path.join(args.save_dir, f"scaler_{args.dataset}_maxs.npy"), maxs)

    # Training
    best_loss = float("inf")
    history = {"total": [], "recon": [], "kl": [], "beta": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for x in train_loader:
            x = x.to(device)
            x_hat, mu, log_sigma, _ = model(x)

            total, recon, kl = vae_loss(
                x, x_hat, mu, log_sigma,
                beta=model.beta,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                lambda3=args.lambda3,
            )

            optimizer.zero_grad()
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Adaptive β update
            model.update_beta(recon.item())

            epoch_total += total.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()
            n_batches   += 1

        if scheduler:
            scheduler.step()

        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        history["beta"].append(model.beta)

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:5d}/{args.epochs} | "
                  f"total={avg_total:.5f}  recon={avg_recon:.5f}  "
                  f"kl={avg_kl:.5f}  beta={model.beta:.2e}")

        # Save best
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "beta": model.beta,
                "loss": best_loss,
            }, os.path.join(args.save_dir, f"vae_{args.dataset}_best.pt"))

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "beta": model.beta,
                "loss": avg_total,
            }, os.path.join(args.save_dir, f"vae_{args.dataset}_epoch{epoch}.pt"))

    # Final checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "beta": model.beta,
        "loss": avg_total,
    }, os.path.join(args.save_dir, f"vae_{args.dataset}_final.pt"))

    # Save loss history
    np.save(os.path.join(args.save_dir, f"vae_{args.dataset}_history.npy"),
            history)

    print(f"\nTraining complete. Best loss: {best_loss:.5f}")
    print(f"Checkpoints saved to: {args.save_dir}/")


if __name__ == "__main__":
    args = get_args()
    train(args)
