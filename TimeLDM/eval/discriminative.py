"""
Discriminative Score for time series generation evaluation.

Trains a GRU-based binary classifier to distinguish real vs. synthetic samples.
Reports |accuracy - 0.5| (lower is better — 0 means indistinguishable).

Reference: TimeGAN (Yoon et al., 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GRUDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, d)
        out, _ = self.gru(x)          # (B, T, hidden_dim)
        logits = self.classifier(out[:, -1, :])  # use last timestep
        return logits.squeeze(-1)     # (B,)


def discriminative_score(
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
    Compute the discriminative score.

    Args:
        real_data: (N_real, T, d) normalized real samples
        fake_data: (N_fake, T, d) generated samples
        hidden_dim: GRU hidden size
        num_layers: number of GRU layers
        n_epochs:   training epochs for the classifier
        batch_size: training batch size
        lr:         learning rate
        n_runs:     number of independent runs (report mean ± std)
        device:     torch device string

    Returns:
        dict with keys 'mean', 'std', 'scores' (list)
    """
    device = torch.device(device)
    input_dim = real_data.shape[2]
    n = min(len(real_data), len(fake_data))

    scores = []
    for run in range(n_runs):
        rng = np.random.RandomState(run)
        idx_r = rng.permutation(len(real_data))[:n]
        idx_f = rng.permutation(len(fake_data))[:n]

        X = np.concatenate([real_data[idx_r], fake_data[idx_f]], axis=0)
        y = np.concatenate([np.ones(n), np.zeros(n)], axis=0)

        # Shuffle
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        split = int(0.8 * len(X))
        X_train = torch.tensor(X[:split], dtype=torch.float32)
        y_train = torch.tensor(y[:split], dtype=torch.float32)
        X_test  = torch.tensor(X[split:], dtype=torch.float32)
        y_test  = torch.tensor(y[split:], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=batch_size, shuffle=True)

        model = GRUDiscriminator(input_dim, hidden_dim, num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(n_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device))
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu()
        acc = (preds == y_test).float().mean().item()
        scores.append(abs(acc - 0.5))

    return {
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "scores": scores,
    }
