"""
VAE Encoder for TimeLDM.

Architecture (Fig. 2b of the paper):
  x (B, tau, d)
  -> Conv1d embedding  -> (B, tau, latent_dim)
  -> + learnable positional encoding
  -> Two parallel Transformer encoder stacks
  -> Linear heads -> mu (B, tau, latent_dim), log_sigma (B, tau, latent_dim)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe


class TransformerEncoderStack(nn.Module):
    """N-layer Transformer encoder (self-attention only)."""
    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # input shape: (B, T, d)
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.transformer(x)


class VAEEncoder(nn.Module):
    """
    Encodes time series x -> (mu, log_sigma) in latent space.

    Args:
        input_dim:      d, number of input features
        seq_len:        tau, number of time steps
        latent_dim:     m, latent space dimension
        num_heads:      number of attention heads (paper: 2)
        num_layers:     N, transformer encoder layers per stack
        dim_feedforward:hidden size of FFN in transformer (default 4*latent_dim)
        dropout:        dropout probability
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        latent_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, \
            f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        dim_feedforward = dim_feedforward or 4 * latent_dim

        # Input embedding: project from input_dim -> latent_dim per timestep
        self.embed = nn.Sequential(
            nn.Conv1d(input_dim, latent_dim, kernel_size=1),
        )

        # Learnable positional encoding
        self.pos_enc = PositionalEncoding(seq_len, latent_dim)

        # Two parallel transformer stacks for mu and log_sigma
        self.mu_encoder = TransformerEncoderStack(
            d_model=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.sigma_encoder = TransformerEncoderStack(
            d_model=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Final linear projections
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.sigma_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, seq_len, input_dim)
        Returns:
            mu:        (B, seq_len, latent_dim)
            log_sigma: (B, seq_len, latent_dim)
        """
        # Embedding: Conv1d expects (B, C, L)
        e = self.embed(x.transpose(1, 2)).transpose(1, 2)  # (B, seq_len, latent_dim)
        e_pe = self.pos_enc(e)                              # (B, seq_len, latent_dim)

        mu        = self.mu_head(self.mu_encoder(e_pe))
        log_sigma = self.sigma_head(self.sigma_encoder(e_pe))
        return mu, log_sigma

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        """Reparameterization trick: z = mu + sigma * eps."""
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(log_sigma)

    def encode(self, x: torch.Tensor):
        """Encode x and sample z via reparameterization."""
        mu, log_sigma = self.forward(x)
        z = self.reparameterize(mu, log_sigma)
        return z, mu, log_sigma
