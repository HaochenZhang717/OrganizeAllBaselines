"""
Dataset loading and preprocessing for TimeLDM.
Supports: sines, mujoco, stocks, etth, fmri
All datasets return windows of shape (N, seq_len, d).
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: numpy array of shape (N, seq_len, d)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normalize_minmax(data: np.ndarray):
    """Min-max normalize each feature to [0, 1] across the dataset.
    data: (N, T, d)
    Returns normalized data and (min, max) per feature for inversion.
    """
    mins = data.min(axis=(0, 1), keepdims=True)   # (1, 1, d)
    maxs = data.max(axis=(0, 1), keepdims=True)   # (1, 1, d)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    data_norm = (data - mins) / denom
    return data_norm, mins, maxs


def sliding_windows(series: np.ndarray, seq_len: int, stride: int = 1):
    """Slice a (T, d) series into (N, seq_len, d) windows."""
    T, d = series.shape
    windows = []
    for i in range(0, T - seq_len + 1, stride):
        windows.append(series[i: i + seq_len])
    return np.stack(windows, axis=0)  # (N, seq_len, d)


# ---------------------------------------------------------------------------
# Sines (simulated)
# ---------------------------------------------------------------------------

def generate_sines(n_samples: int, seq_len: int, n_features: int = 5,
                   seed: int = 42) -> np.ndarray:
    """Generate sinusoidal time series following TimeGAN setup."""
    rng = np.random.RandomState(seed)
    data = []
    t = np.linspace(0, 1, seq_len)
    for _ in range(n_samples):
        sample = []
        for _ in range(n_features):
            freq = rng.uniform(1.0, 5.0)
            phase = rng.uniform(0, 2 * np.pi)
            sample.append(np.sin(2 * np.pi * freq * t + phase))
        data.append(np.stack(sample, axis=-1))  # (seq_len, n_features)
    return np.stack(data, axis=0)  # (n_samples, seq_len, n_features)


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

def load_stocks(data_path: str, seq_len: int) -> np.ndarray:
    """
    Load Google stock CSV. Expected columns (after header):
    Date, Open, High, Low, Close, Adj Close, Volume  (or similar).
    The 6 numeric feature columns are used.
    """
    import pandas as pd
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number]).dropna()
    series = df.values.astype(np.float32)   # (T, d)
    windows = sliding_windows(series, seq_len, stride=1)
    return windows


# ---------------------------------------------------------------------------
# ETTh
# ---------------------------------------------------------------------------

def load_etth(data_path: str, seq_len: int) -> np.ndarray:
    """Load ETTh1 or ETTh2 CSV."""
    import pandas as pd
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number]).dropna()
    series = df.values.astype(np.float32)   # (T, 7)
    windows = sliding_windows(series, seq_len, stride=1)
    return windows


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------

def load_mujoco(data_path: str, seq_len: int) -> np.ndarray:
    """
    Load MuJoCo dataset from .npy file.
    Expected shape: (N, T, 14) or a (T, 14) series.
    """
    data = np.load(data_path)
    if data.ndim == 2:
        # Single long series -> window it
        windows = sliding_windows(data, seq_len, stride=1)
    elif data.ndim == 3:
        T = data.shape[1]
        if T == seq_len:
            windows = data
        else:
            windows = np.concatenate(
                [sliding_windows(data[i], seq_len, stride=1)
                 for i in range(len(data))], axis=0)
    else:
        raise ValueError(f"Unexpected MuJoCo data shape: {data.shape}")
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# fMRI
# ---------------------------------------------------------------------------

def load_fmri(data_path: str, seq_len: int) -> np.ndarray:
    """
    Load fMRI dataset from .npy file.
    Expected shape: (N, T, 50) or (T, 50).
    """
    data = np.load(data_path)
    if data.ndim == 2:
        windows = sliding_windows(data, seq_len, stride=1)
    elif data.ndim == 3:
        T = data.shape[1]
        if T == seq_len:
            windows = data
        else:
            windows = np.concatenate(
                [sliding_windows(data[i], seq_len, stride=1)
                 for i in range(len(data))], axis=0)
    else:
        raise ValueError(f"Unexpected fMRI data shape: {data.shape}")
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DATASET_DEFAULTS = {
    "sines":  {"d": 5,  "batch_size": 1024},
    "mujoco": {"d": 14, "batch_size": 1024},
    "stocks": {"d": 6,  "batch_size": 512},
    "etth":   {"d": 7,  "batch_size": 1024},
    "fmri":   {"d": 50, "batch_size": 1024},
}


def get_dataset(
    dataset_name: str,
    seq_len: int = 24,
    data_path: str = None,
    train_ratio: float = 0.8,
    n_sines_samples: int = 10000,
    seed: int = 42,
):
    """
    Returns (train_data, test_data, scaler_info) where data arrays are
    (N, seq_len, d) numpy arrays normalized to [0, 1].

    scaler_info = (mins, maxs) for inverse normalization.
    """
    name = dataset_name.lower()

    if name == "sines":
        d = DATASET_DEFAULTS["sines"]["d"]
        raw = generate_sines(n_sines_samples, seq_len, n_features=d, seed=seed)

    elif name == "stocks":
        assert data_path is not None, "data_path required for stocks"
        raw = load_stocks(data_path, seq_len)

    elif name == "etth":
        assert data_path is not None, "data_path required for etth"
        raw = load_etth(data_path, seq_len)

    elif name == "mujoco":
        assert data_path is not None, "data_path required for mujoco"
        raw = load_mujoco(data_path, seq_len)

    elif name == "fmri":
        assert data_path is not None, "data_path required for fmri"
        raw = load_fmri(data_path, seq_len)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(DATASET_DEFAULTS.keys())}")

    # Normalize to [0, 1]
    raw_norm, mins, maxs = normalize_minmax(raw)

    # Train / test split
    n = len(raw_norm)
    split = int(n * train_ratio)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    train_data = raw_norm[idx[:split]]
    test_data  = raw_norm[idx[split:]]

    return train_data, test_data, (mins, maxs)


def get_dataloader(data: np.ndarray, batch_size: int, shuffle: bool = True,
                   num_workers: int = 0) -> DataLoader:
    dataset = TimeSeriesDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True)
