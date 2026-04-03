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
    Transformer-encoder denoiser for the latent diffusion model.

    Each position in the latent sequence (B, seq_len, latent_dim) is treated
    as a token.  A sinusoidal time embedding is projected to d_model and added
    to every token before the encoder stack, so the denoiser is conditioned on
    the diffusion time step.

    Args:
        seq_len:       tau, sequence length
        latent_dim:    m, latent dimension per position
        hidden_dim:    transformer d_model
        num_layers:    number of TransformerEncoderLayer blocks
        n_heads:       number of attention heads (hidden_dim must be divisible by n_heads)
        time_emb_dim:  sinusoidal embedding dim before projection (default = hidden_dim)
        dropout:       dropout used in attention and feed-forward sub-layers
    """
    def __init__(
        self,
        seq_len: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        time_emb_dim = hidden_dim

        # Time embedding: scalar t -> (B, hidden_dim), broadcast to all positions
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # Per-position input projection: latent_dim -> hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Per-position output projection: hidden_dim -> latent_dim
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        """
        Predict the noise epsilon given noisy latent z_t and time t.

        Args:
            z_t: (B, seq_len, latent_dim) - noisy latent
            t:   (B,) - continuous time values in [t_min, T]
        Returns:
            eps_pred: (B, seq_len, latent_dim) - predicted noise
        """
        # Project each position: (B, seq_len, latent_dim) -> (B, seq_len, hidden_dim)
        h = self.input_proj(z_t)
        breakpoint()
        # Add positional embedding
        h = h + self.pos_embed                               # (B, seq_len, hidden_dim)

        # Add time embedding (same value broadcast across all positions)
        t_emb = self.time_proj(self.time_embed(t))           # (B, hidden_dim)
        h = h + t_emb.unsqueeze(1)                           # (B, seq_len, hidden_dim)

        # Transformer encoder
        h = self.transformer(h)                              # (B, seq_len, hidden_dim)

        # Project back to latent space
        eps_pred = self.output_proj(h)                       # (B, seq_len, latent_dim)
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
