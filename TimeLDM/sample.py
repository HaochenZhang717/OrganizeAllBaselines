"""
Sampling / Inference for TimeLDM.

Loads a trained VAE decoder + LDM denoiser and generates synthetic time series
using the reverse SDE (stochastic) or ODE (deterministic) sampler.

Usage:
    python sample.py \\
        --vae_ckpt  checkpoints/vae_sines_best.pt \\
        --ldm_ckpt  checkpoints/ldm_sines_best.pt \\
        --n_samples 1000 \\
        --sampler   sde \\
        --n_steps   1000 \\
        --output    samples/sines_generated.npy
"""

import argparse
import os
import numpy as np
import torch

from models.vae import VAE
from models.ldm import LDMDenoiser, sigma
from data.datasets import DATASET_DEFAULTS


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Generate time series with TimeLDM")

    # Checkpoints
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--ldm_ckpt", type=str, required=True,
                        help="Path to trained LDM checkpoint")

    # Generation
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for generation")

    # Sampler
    parser.add_argument("--sampler", type=str, default="sde",
                        choices=["sde", "ode"],
                        help="sde: stochastic reverse SDE, ode: deterministic ODE")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of discretization steps for reverse process")
    parser.add_argument("--t_max", type=float, default=80.0,
                        help="Starting noise level (must match LDM training)")
    parser.add_argument("--t_min", type=float, default=0.002,
                        help="Final noise level (small positive value)")
    parser.add_argument("--stochasticity", type=float, default=1.0,
                        help="Stochasticity factor S_churn [0=ODE, 1=full SDE]")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save generated samples (.npy). "
                             "Default: samples/<dataset>_generated.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

ENC_LAYERS_DEFAULT = {"sines": 1, "mujoco": 1, "stocks": 2, "etth": 2, "fmri": 1}
DEC_LAYERS_DEFAULT = {"sines": 2, "mujoco": 2, "stocks": 3, "etth": 3, "fmri": 2}


def load_vae(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    dataset = a["dataset"]
    input_dim  = DATASET_DEFAULTS[dataset]["d"]
    seq_len    = a["seq_len"]
    latent_dim = a["latent_dim"]
    num_heads  = a["num_heads"]
    enc_layers = a.get("enc_layers") or ENC_LAYERS_DEFAULT[dataset]
    dec_layers = a.get("dec_layers") or DEC_LAYERS_DEFAULT[dataset]

    vae = VAE(
        input_dim=input_dim, seq_len=seq_len,
        latent_dim=latent_dim, num_heads=num_heads,
        enc_layers=enc_layers, dec_layers=dec_layers,
        dim_feedforward=a.get("dim_feedforward"),
        dropout=a.get("dropout", 0.1),
    ).to(device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, latent_dim, seq_len, input_dim


def load_ldm(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    seq_len    = ckpt["seq_len"]
    latent_dim = ckpt["latent_dim"]
    hidden_dim = ckpt["hidden_dim"]

    ldm = LDMDenoiser(
        seq_len=seq_len,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=a.get("num_mlp_layers", 4),
        time_emb_dim=a.get("time_emb_dim"),
        dropout=0.0,  # no dropout at inference
    ).to(device)
    ldm.load_state_dict(ckpt["model_state_dict"])
    ldm.eval()
    for p in ldm.parameters():
        p.requires_grad_(False)
    return ldm, seq_len, latent_dim


# ---------------------------------------------------------------------------
# Reverse samplers
# ---------------------------------------------------------------------------

def get_time_schedule(t_max: float, t_min: float, n_steps: int,
                      device: torch.device) -> torch.Tensor:
    """
    Linearly spaced time steps from t_max to t_min.
    Returns (n_steps+1,) tensor: [t_max, ..., t_min].
    """
    return torch.linspace(t_max, t_min, n_steps + 1, device=device)


@torch.no_grad()
def sample_sde(ldm, z_T, t_schedule, stochasticity=1.0):
    """
    Stochastic reverse SDE (Euler-Maruyama).

    For sigma(t) = t, sigdot(t) = 1:
        dz = 2 * eps_theta(z_t, t) * (-dt) + sqrt(2*t*|dt|) * noise
           = -2 * dt * eps_theta(z_t, t) + sqrt(2*t*|dt|) * noise

    Args:
        ldm:          trained denoiser
        z_T:          (B, seq_len, latent_dim) initial noise
        t_schedule:   (n_steps+1,) decreasing from t_max to t_min
        stochasticity: S in [0,1]; 0 = pure ODE, 1 = full SDE
    """
    z = z_T.clone()
    n_steps = len(t_schedule) - 1

    for i in range(n_steps):
        t_curr = t_schedule[i]
        t_next = t_schedule[i + 1]
        dt = t_next - t_curr          # negative (time decreases)

        t_batch = t_curr.expand(z.shape[0])
        eps_pred = ldm(z, t_batch)    # (B, T, latent_dim)

        # Deterministic drift
        z_det = z - 2.0 * (-dt) * eps_pred

        # Stochastic term (only if stochasticity > 0)
        if stochasticity > 0 and t_next > 0:
            noise_scale = stochasticity * (2.0 * t_curr.item() * (-dt.item())) ** 0.5
            z_det = z_det + noise_scale * torch.randn_like(z)

        z = z_det

    return z


@torch.no_grad()
def sample_ode(ldm, z_T, t_schedule):
    """
    Deterministic probability-flow ODE (Euler method).

    dz/dt = -score = eps_theta(z_t, t) / sigma(t) * sigma(t) = eps_theta(z_t, t)
    Using the drift-only reverse: dz = -2 * dt * eps_theta(z_t, t)
    """
    return sample_sde(ldm, z_T, t_schedule, stochasticity=0.0)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(vae, ldm, n_samples, batch_size, seq_len, latent_dim,
             t_max, t_min, n_steps, sampler, stochasticity, device):
    """Generate n_samples time series."""
    t_schedule = get_time_schedule(t_max, t_min, n_steps, device)

    all_samples = []
    n_generated = 0

    while n_generated < n_samples:
        bs = min(batch_size, n_samples - n_generated)

        # Sample initial noise z_T ~ N(0, t_max^2 * I)
        z_T = torch.randn(bs, seq_len, latent_dim, device=device) * t_max

        # Reverse diffusion
        if sampler == "sde":
            z_0 = sample_sde(ldm, z_T, t_schedule, stochasticity)
        else:
            z_0 = sample_ode(ldm, z_T, t_schedule)

        # Decode latent to time series
        x_hat = vae.decode(z_0)        # (bs, seq_len, input_dim)
        x_hat = x_hat.cpu().numpy()
        all_samples.append(x_hat)
        n_generated += bs

    return np.concatenate(all_samples, axis=0)[:n_samples]  # (n_samples, seq_len, d)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()

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

    # Load models
    print("Loading VAE...")
    vae, latent_dim, seq_len, input_dim = load_vae(args.vae_ckpt, device)

    print("Loading LDM...")
    ldm, seq_len_ldm, latent_dim_ldm = load_ldm(args.ldm_ckpt, device)

    assert seq_len == seq_len_ldm,     f"seq_len mismatch: VAE={seq_len}, LDM={seq_len_ldm}"
    assert latent_dim == latent_dim_ldm, f"latent_dim mismatch: VAE={latent_dim}, LDM={latent_dim_ldm}"

    print(f"Generating {args.n_samples} samples | "
          f"sampler={args.sampler}, n_steps={args.n_steps}, "
          f"t_max={args.t_max}, t_min={args.t_min}")

    samples = generate(
        vae=vae, ldm=ldm,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seq_len=seq_len,
        latent_dim=latent_dim,
        t_max=args.t_max,
        t_min=args.t_min,
        n_steps=args.n_steps,
        sampler=args.sampler,
        stochasticity=args.stochasticity,
        device=device,
    )

    print(f"Generated samples shape: {samples.shape}")

    # Save
    output = args.output
    if output is None:
        # Try to infer dataset name from ckpt path
        ckpt_name = os.path.basename(args.ldm_ckpt)
        dataset = ckpt_name.replace("ldm_", "").split("_")[0]
        os.makedirs("samples", exist_ok=True)
        output = f"samples/{dataset}_generated.npy"

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    np.save(output, samples)
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
