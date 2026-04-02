"""
Latent Diffusion Model (MLP denoiser) for TimeLDM.

Architecture (Fig. 2c of the paper):
  z_t (B, tau, latent_dim)
  -> Reshape to (B, tau*latent_dim)
  -> Linear -> hidden_dim
  -> + sinusoidal time embedding
  -> 4 x (Linear + SiLU)
  -> Linear output -> (B, tau*latent_dim)
  -> Reshape to (B, tau, latent_dim)

Noise schedule: sigma(t) = t  (Karras et al. 2022 / EDM)
Forward process: z_t = z_0 + sigma(t) * eps,   eps ~ N(0, I)
Training loss:   L = E || eps_theta(z_t, t) - eps ||^2
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Encode a scalar time value t into a sinusoidal feature vector.
    Follows the standard transformer positional encoding but for a single scalar.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even"
        self.embedding_dim = embedding_dim
        # Precompute frequency terms
        half = embedding_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)  # (half,)

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: (B,) scalar time values
        Returns:
            emb: (B, embedding_dim)
        """
        t = t.float().unsqueeze(1)               # (B, 1)
        args = t * self.freqs.unsqueeze(0)        # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embedding_dim)
        return emb


class LDMDenoiser(nn.Module):
    """
    MLP denoiser for the latent diffusion model.

    Args:
        seq_len:       tau, sequence length
        latent_dim:    m, latent dimension
        hidden_dim:    MLP hidden dimension (paper: 1024 or 4096)
        num_layers:    number of hidden linear layers (paper: 4)
        time_emb_dim:  dimension of time sinusoidal embedding (default = hidden_dim)
        dropout:       dropout probability
    """
    def __init__(
        self,
        seq_len: int,
        latent_dim: int = 32,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        time_emb_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        flat_dim = seq_len * latent_dim
        time_emb_dim = time_emb_dim or hidden_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(flat_dim, hidden_dim)

        # Hidden MLP layers
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        """
        Predict the noise epsilon given noisy latent z_t and time t.

        Args:
            z_t: (B, seq_len, latent_dim) - noisy latent
            t:   (B,) - continuous time values in [t_min, T]
        Returns:
            eps_pred: (B, seq_len, latent_dim) - predicted noise
        """
        B = z_t.shape[0]

        # Flatten
        h = z_t.reshape(B, -1)                # (B, flat_dim)

        # Project input
        h = self.input_proj(h)                # (B, hidden_dim)

        # Add time embedding
        t_emb = self.time_proj(self.time_embed(t))  # (B, hidden_dim)
        h = h + t_emb

        # MLP
        h = self.mlp(h)                       # (B, hidden_dim)

        # Output
        h = self.output_proj(h)               # (B, flat_dim)
        eps_pred = h.reshape(B, self.seq_len, self.latent_dim)
        return eps_pred


# ---------------------------------------------------------------------------
# Noise schedule: sigma(t) = t  (Karras EDM)
# ---------------------------------------------------------------------------

def sigma(t: torch.Tensor) -> torch.Tensor:
    """Noise level at time t. sigma(t) = t."""
    return t


def sample_time(batch_size: int, t_min: float, t_max: float,
                device: torch.device) -> torch.Tensor:
    """Sample t uniformly in [t_min, t_max]."""
    return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min


def add_noise(z0: torch.Tensor, t: torch.Tensor):
    """
    Forward diffusion: z_t = z_0 + sigma(t) * eps.

    Args:
        z0: (B, seq_len, latent_dim)
        t:  (B,)
    Returns:
        z_t: (B, seq_len, latent_dim)
        eps: (B, seq_len, latent_dim)
    """
    eps = torch.randn_like(z0)
    sig = sigma(t).view(-1, 1, 1)          # broadcast over seq and latent
    z_t = z0 + sig * eps
    return z_t, eps
