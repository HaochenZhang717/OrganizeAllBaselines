"""
Predictive Score for time series generation evaluation.

Train-on-Synthetic, Test-on-Real (TSTR):
  1. Train a GRU predictor on synthetic data to predict the next timestep.
  2. Evaluate the trained predictor on real test data.
  3. Report MAE (lower is better).

Reference: TimeGAN (Yoon et al., 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GRUPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T-1, d)  -> predict (B, T-1, d) (next-step prediction)
        out, _ = self.gru(x)           # (B, T-1, hidden_dim)
        return self.head(out)          # (B, T-1, d)


def make_prediction_pairs(data: np.ndarray):
    """
    Split sequences into input (T-1 steps) and target (shifted by 1).
    data: (N, T, d)
    Returns: inputs (N, T-1, d), targets (N, T-1, d)
    """
    return data[:, :-1, :], data[:, 1:, :]


def predictive_score(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    hidden_dim: int = 64,
    num_layers: int = 2,
    n_epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_runs: int = 5,
    device: str = "cpu",
) -> dict:
    """
    Compute the predictive score via TSTR.

    Args:
        real_data: (N_real, T, d) real test data
        fake_data: (N_fake, T, d) generated samples (used for training)
        hidden_dim: GRU hidden size
        num_layers: GRU layers
        n_epochs:   training epochs
        batch_size: batch size
        lr:         learning rate
        n_runs:     independent runs
        device:     torch device string

    Returns:
        dict with 'mean', 'std', 'scores'
    """
    device = torch.device(device)
    input_dim = real_data.shape[2]

    X_fake, y_fake = make_prediction_pairs(fake_data)  # train on synthetic
    X_real, y_real = make_prediction_pairs(real_data)  # test on real

    scores = []
    for run in range(n_runs):
        rng = np.random.RandomState(run)
        idx = rng.permutation(len(X_fake))
        X_f = torch.tensor(X_fake[idx], dtype=torch.float32)
        y_f = torch.tensor(y_fake[idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_f, y_f),
                                  batch_size=batch_size, shuffle=True)

        model = GRUPredictor(input_dim, hidden_dim, num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        model.train()
        for _ in range(n_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on real data
        model.eval()
        X_r = torch.tensor(X_real, dtype=torch.float32).to(device)
        y_r = torch.tensor(y_real, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_r).cpu()
        mae = torch.mean(torch.abs(y_pred - y_r)).item()
        scores.append(mae)

    return {
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "scores": scores,
    }
