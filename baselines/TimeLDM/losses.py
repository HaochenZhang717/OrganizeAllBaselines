"""
Loss functions for TimeLDM VAE training.

Reconstruction loss: L1 + L2 + FFT (frequency domain)
KL divergence loss:  beta-weighted KL(q(z|x) || N(0,I))
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
) -> torch.Tensor:
    """
    Combined reconstruction loss in data and frequency domains.

    L_recon = lambda1 * L2 + lambda2 * L1 + lambda3 * FFT_loss

    Args:
        x:       (B, T, d) ground truth
        x_hat:   (B, T, d) reconstruction
        lambda1: weight for L2 loss
        lambda2: weight for L1 loss
        lambda3: weight for FFT loss
    Returns:
        scalar loss
    """
    l2_loss = F.mse_loss(x_hat, x)
    l1_loss = F.l1_loss(x_hat, x)

    # FFT loss in the time dimension
    # torch.fft.fft returns complex tensor; we compare magnitudes
    fft_x    = torch.fft.fft(x,     dim=1)   # (B, T, d) complex
    fft_xhat = torch.fft.fft(x_hat, dim=1)   # (B, T, d) complex
    fft_loss = torch.mean(torch.abs(fft_x - fft_xhat))

    return lambda1 * l2_loss + lambda2 * l1_loss + lambda3 * fft_loss


def kl_divergence_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence: KL(q(z|x) || N(0,I)).

    For q(z|x) = N(mu, sigma^2*I):
      KL = -0.5 * sum(1 + 2*log_sigma - mu^2 - exp(2*log_sigma))

    Args:
        mu:        (B, T, latent_dim)
        log_sigma: (B, T, latent_dim)
    Returns:
        scalar KL (mean over batch and dimensions)
    """
    kl = -0.5 * torch.mean(
        1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()
    )
    return kl


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    beta: float = 1e-2,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
):
    """
    Total VAE loss: L_recon + beta * L_KL.

    Returns:
        total_loss, recon_loss, kl_loss (all scalars)
    """
    recon = reconstruction_loss(x, x_hat, lambda1, lambda2, lambda3)
    kl    = kl_divergence_loss(mu, log_sigma)
    total = recon + beta * kl
    return total, recon, kl


def ldm_loss(
    eps_pred: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    """
    LDM training loss: MSE between predicted and actual noise.

    L_LDM = E || eps_theta(z_t, t) - eps ||^2

    Args:
        eps_pred: (B, T, latent_dim) predicted noise
        eps:      (B, T, latent_dim) actual noise
    Returns:
        scalar MSE loss
    """
    return F.mse_loss(eps_pred, eps)
