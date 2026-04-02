"""
Stage 2: Train the Latent Diffusion Model for TimeLDM.

Requires a pre-trained VAE checkpoint from train_vae.py.
The VAE encoder is frozen; only the LDM denoiser is trained.

Usage:
    python train_ldm.py --dataset sines --vae_ckpt checkpoints/vae_sines_best.pt
    python train_ldm.py --dataset stocks --vae_ckpt checkpoints/vae_stocks_best.pt \\
        --hidden_dim 1024 --t_max 80 --epochs 10000
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
from models.ldm import LDMDenoiser, add_noise, sample_time
from losses import ldm_loss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Train TimeLDM LDM (Stage 2)")

    # Data (must match VAE training)
    parser.add_argument("--dataset", type=str, default="sines",
                        choices=["sines", "mujoco", "stocks", "etth", "fmri"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--n_sines_samples", type=int, default=10000)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # VAE checkpoint (required)
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to trained VAE checkpoint (.pt)")

    # LDM architecture
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="MLP hidden dim (default: dataset preset from paper)")
    parser.add_argument("--num_mlp_layers", type=int, default=4,
                        help="Number of hidden MLP layers (paper: 4)")
    parser.add_argument("--time_emb_dim", type=int, default=None,
                        help="Sinusoidal time embedding dim (default: hidden_dim)")
    parser.add_argument("--ldm_dropout", type=float, default=0.0,
                        help="Dropout in LDM MLP")

    # Diffusion schedule
    parser.add_argument("--t_min", type=float, default=0.002,
                        help="Minimum noise level (t_min)")
    parser.add_argument("--t_max", type=float, default=80.0,
                        help="Maximum noise level T (sigma_max in EDM)")

    # Training
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1 (paper: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.96,
                        help="Adam beta2 (paper: 0.96)")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", action="store_true",
                        help="Use cosine annealing LR scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping (0 = disabled)")
    parser.add_argument("--num_workers", type=int, default=0)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


LDM_HIDDEN_DIM_DEFAULT = {
    "sines": 1024, "mujoco": 4096, "stocks": 1024, "etth": 1024, "fmri": 4096
}

ENC_LAYERS_DEFAULT = {"sines": 1, "mujoco": 1, "stocks": 2, "etth": 2, "fmri": 1}
DEC_LAYERS_DEFAULT = {"sines": 2, "mujoco": 2, "stocks": 3, "etth": 3, "fmri": 2}


# ---------------------------------------------------------------------------
# Load VAE from checkpoint
# ---------------------------------------------------------------------------

def load_vae(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]

    # Determine input_dim from saved args (dataset)
    dataset = saved_args["dataset"]
    input_dim = DATASET_DEFAULTS[dataset]["d"]
    seq_len   = saved_args["seq_len"]
    latent_dim = saved_args["latent_dim"]
    num_heads  = saved_args["num_heads"]
    enc_layers = saved_args.get("enc_layers") or ENC_LAYERS_DEFAULT[dataset]
    dec_layers = saved_args.get("dec_layers") or DEC_LAYERS_DEFAULT[dataset]

    vae = VAE(
        input_dim=input_dim,
        seq_len=seq_len,
        latent_dim=latent_dim,
        num_heads=num_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dim_feedforward=saved_args.get("dim_feedforward"),
        dropout=saved_args.get("dropout", 0.1),
    ).to(device)

    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return vae, latent_dim, seq_len, input_dim


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

    # Load frozen VAE
    print(f"Loading VAE from: {args.vae_ckpt}")
    vae, latent_dim, seq_len, input_dim = load_vae(args.vae_ckpt, device)
    print(f"VAE loaded | seq_len={seq_len}, latent_dim={latent_dim}, input_dim={input_dim}")

    # Override seq_len if provided
    if args.seq_len != seq_len:
        print(f"Warning: --seq_len {args.seq_len} overrides VAE seq_len {seq_len}. "
              "Make sure data matches VAE training.")
        seq_len = args.seq_len

    # Data
    train_data, test_data, _ = get_dataset(
        dataset_name=args.dataset,
        seq_len=seq_len,
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        n_sines_samples=args.n_sines_samples,
        seed=args.seed,
    )
    batch_size = args.batch_size or DATASET_DEFAULTS[args.dataset]["batch_size"]
    batch_size = min(batch_size, len(train_data))
    train_loader = get_dataloader(train_data, batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset} | train={len(train_data)}")

    # LDM model
    hidden_dim = args.hidden_dim or LDM_HIDDEN_DIM_DEFAULT[args.dataset]
    ldm = LDMDenoiser(
        seq_len=seq_len,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=args.num_mlp_layers,
        time_emb_dim=args.time_emb_dim,
        dropout=args.ldm_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    print(f"LDM parameters: {n_params:,}  |  hidden_dim={hidden_dim}")

    optimizer = optim.Adam(
        ldm.parameters(), lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = (CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.lr_scheduler else None)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"ldm_{args.dataset}_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        ldm.train()
        epoch_loss = 0.0
        n_batches = 0

        for x in train_loader:
            x = x.to(device)

            # Encode to latent (no gradient through encoder)
            with torch.no_grad():
                z0, _, _ = vae.encode(x)   # (B, seq_len, latent_dim)

            # Sample diffusion time
            t = sample_time(x.shape[0], args.t_min, args.t_max, device)

            # Forward diffusion
            z_t, eps = add_noise(z0, t)

            # Predict noise
            eps_pred = ldm(z_t, t)

            loss = ldm_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ldm.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:6d}/{args.epochs} | loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": ldm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "loss": best_loss,
                "latent_dim": latent_dim,
                "seq_len": seq_len,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
            }, os.path.join(args.save_dir, f"ldm_{args.dataset}_best.pt"))

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": ldm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "loss": avg_loss,
                "latent_dim": latent_dim,
                "seq_len": seq_len,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
            }, os.path.join(args.save_dir, f"ldm_{args.dataset}_epoch{epoch}.pt"))

    # Final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": ldm.state_dict(),
        "args": vars(args),
        "loss": avg_loss,
        "latent_dim": latent_dim,
        "seq_len": seq_len,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }, os.path.join(args.save_dir, f"ldm_{args.dataset}_final.pt"))

    np.save(os.path.join(args.save_dir, f"ldm_{args.dataset}_history.npy"),
            np.array(history))

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.save_dir}/")


if __name__ == "__main__":
    args = get_args()
    train(args)
