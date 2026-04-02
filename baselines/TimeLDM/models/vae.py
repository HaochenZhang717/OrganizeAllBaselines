"""
Full β-VAE wrapper for TimeLDM.
Combines VAEEncoder and VAEDecoder with the adaptive β schedule.
"""

import torch
import torch.nn as nn
from .encoder import VAEEncoder
from .decoder import VAEDecoder


class VAE(nn.Module):
    """
    β-VAE for time series.

    Args:
        input_dim:       d, number of input/output features
        seq_len:         tau, sequence length
        latent_dim:      m, latent dimension
        num_heads:       attention heads (paper: 2)
        enc_layers:      N, transformer encoder layers
        dec_layers:      M, transformer decoder layers
        dim_feedforward: FFN hidden size (default 4*latent_dim)
        dropout:         dropout probability
        beta_max:        initial β for KL loss (paper: 1e-2)
        beta_min:        minimum β (paper: 1e-5)
        beta_lambda:     decay factor when recon loss stagnates (paper: 0.7)
        beta_patience:   steps to wait before decaying β
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        latent_dim: int = 32,
        num_heads: int = 2,
        enc_layers: int = 1,
        dec_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        beta_max: float = 1e-2,
        beta_min: float = 1e-5,
        beta_lambda: float = 0.7,
        beta_patience: int = 50,
    ):
        super().__init__()
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=enc_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = VAEDecoder(
            output_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Adaptive β state (not a nn.Parameter — managed manually)
        self.beta = beta_max
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.beta_lambda = beta_lambda
        self.beta_patience = beta_patience
        self._best_recon = float("inf")
        self._no_improve_steps = 0

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, seq_len, input_dim)
        Returns:
            x_hat, mu, log_sigma, z
        """
        z, mu, log_sigma = self.encoder.encode(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_sigma, z

    def encode(self, x: torch.Tensor):
        """Encode x -> z (sampled), mu, log_sigma."""
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor):
        """Decode z -> x_hat."""
        return self.decoder(z)

    def update_beta(self, recon_loss: float):
        """
        Adaptive β: decay if reconstruction loss is not improving.
        Should be called once per training step with the current recon loss value.
        """
        if recon_loss < self._best_recon:
            self._best_recon = recon_loss
            self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1
            if self._no_improve_steps >= self.beta_patience:
                self.beta = max(self.beta * self.beta_lambda, self.beta_min)
                self._no_improve_steps = 0
