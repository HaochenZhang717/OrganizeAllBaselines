"""
VAE Decoder for TimeLDM.

Architecture (Fig. 2d of the paper):
  z (B, tau, latent_dim)
  -> Embedding + learnable positional encoding
  -> M Transformer decoder layers (self-attn + cross-attn + FFN)
     (cross-attn keys/values come from the initial embedded z)
  -> Conv1d output projection -> x_hat (B, tau, input_dim)
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer:
      Self-Attention -> Add&Norm -> Cross-Attention -> Add&Norm -> FFN -> Add&Norm
    """
    def __init__(self, d_model: int, num_heads: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        """
        Args:
            x:      (B, T, d_model)  - current state
            memory: (B, T, d_model)  - latent representation (keys/values for cross-attn)
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Cross-attention (attend to latent memory)
        attn_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + attn_out)

        # Feed-forward
        x = self.norm3(x + self.ffn(x))
        return x


class VAEDecoder(nn.Module):
    """
    Decodes latent z back to time series x_hat.

    Args:
        output_dim:     d, number of output features
        seq_len:        tau, number of time steps
        latent_dim:     m, latent space dimension
        num_heads:      number of attention heads (paper: 2)
        num_layers:     M, number of decoder transformer layers
        dim_feedforward:hidden size of FFN (default 4*latent_dim)
        dropout:        dropout probability
    """
    def __init__(
        self,
        output_dim: int,
        seq_len: int,
        latent_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, \
            f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        dim_feedforward = dim_feedforward or 4 * latent_dim

        # Embedding for latent input
        self.embed = nn.Linear(latent_dim, latent_dim)

        # Learnable positional encoding
        self.pos_enc = PositionalEncoding(seq_len, latent_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=latent_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection back to data space
        self.out_proj = nn.Conv1d(latent_dim, output_dim, kernel_size=1)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, seq_len, latent_dim)
        Returns:
            x_hat: (B, seq_len, output_dim)
        """
        # Embed and add positional encoding
        h = self.pos_enc(self.embed(z))   # (B, seq_len, latent_dim)
        memory = h                        # keys/values for cross-attention

        # Transformer decoder layers
        for layer in self.layers:
            h = layer(h, memory)

        # Output projection: Conv1d expects (B, C, L)
        x_hat = self.out_proj(h.transpose(1, 2)).transpose(1, 2)  # (B, seq_len, output_dim)
        return x_hat
