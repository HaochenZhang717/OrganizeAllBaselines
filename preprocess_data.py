"""
Preprocess CSV files into sliding window numpy arrays split into train/valid/test.

Each channel is normalized to [-1, 1] using per-channel min/max computed on the
full time series.  Normalization stats are saved alongside the .npy splits so
that the transform can be inverted later.

Output shape: (num_windows, window_size, num_features)
Saved files:
  processed/<dataset>/train_ts.npy
  processed/<dataset>/valid_ts.npy
  processed/<dataset>/test_ts.npy
  processed/<dataset>/norm_stats.npz   ← keys: 'min', 'max'  shape (num_features,)
"""

import os
import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

DATASETS = {
    "energy_data": {
        "path": "raw_data/energy_data.csv",
        "date_col": None,          # no date column
    },
    "ETTh1": {
        "path": "raw_data/ETTh1.csv",
        "date_col": "date",        # column to drop before windowing
    },
}

WINDOW_SIZE = 128       # length of each sliding window (time steps)
STRIDE = 1             # step size between consecutive windows

# Train / valid / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO  = 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str, date_col: str | None) -> np.ndarray:
    """Load CSV and return a float32 array of shape (T, num_features)."""
    df = pd.read_csv(path)
    if date_col is not None and date_col in df.columns:
        df = df.drop(columns=[date_col])
    return df.values.astype(np.float32)


def normalize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each channel to [-1, 1] using per-channel min/max over the full
    time series.

    Returns:
        data_norm : normalized array, same shape as input
        ch_min    : (num_features,) per-channel minimum
        ch_max    : (num_features,) per-channel maximum
    """
    ch_min = data.min(axis=0)   # (F,)
    ch_max = data.max(axis=0)   # (F,)
    span = ch_max - ch_min
    span[span == 0] = 1.0       # avoid division by zero for constant channels
    data_norm = 2.0 * (data - ch_min) / span - 1.0
    return data_norm, ch_min, ch_max


def sliding_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Convert (T, F) array into (num_windows, window_size, F) via a sliding window.
    """
    t, f = data.shape
    indices = range(0, t - window_size + 1, stride)
    windows = np.stack([data[i : i + window_size] for i in indices], axis=0)
    return windows  # (num_windows, window_size, F)


def split_windows(
    windows: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split windows array into train / valid / test along axis 0."""
    n = len(windows)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return windows[:train_end], windows[train_end:valid_end], windows[valid_end:]


# ── Main ──────────────────────────────────────────────────────────────────────

def process_dataset(name: str, cfg: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")

    # Load
    data = load_csv(cfg["path"], cfg["date_col"])
    print(f"  Raw data shape : {data.shape}  (timesteps × features)")

    # Normalize each channel to [-1, 1] (stats from full series)
    data, ch_min, ch_max = normalize(data)
    print(f"  Normalized     : channel range [{data.min():.3f}, {data.max():.3f}]")

    # Sliding windows
    windows = sliding_windows(data, WINDOW_SIZE, STRIDE)
    print(f"  Windows shape  : {windows.shape}  (num_windows × window_size × features)")

    # Split
    train, valid, test = split_windows(windows, TRAIN_RATIO, VALID_RATIO)
    print(f"  Train          : {train.shape}")
    print(f"  Valid          : {valid.shape}")
    print(f"  Test           : {test.shape}")

    # Save
    out_dir = os.path.join("processed", name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_ts.npy"), train)
    np.save(os.path.join(out_dir, "valid_ts.npy"), valid)
    np.save(os.path.join(out_dir, "test_ts.npy"),  test)
    np.savez(os.path.join(out_dir, "norm_stats.npz"), min=ch_min, max=ch_max)
    print(f"  Saved to       : {out_dir}/")


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        process_dataset(name, cfg)
    print("\nDone.")
