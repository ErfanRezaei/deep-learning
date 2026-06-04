

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ["Open", "High", "Low", "Close"]
CLOSE_IDX = FEATURES.index("Close")

# Chronological split boundaries (inclusive end dates).
TRAIN_END = "2024-07-31"
VAL_END = "2024-12-31"
# Everything after VAL_END belongs to the test set (Jan-Dec 2025).

DEFAULT_T = 20
DEFAULT_HORIZONS = (1, 2, 3, 4, 5)
DEFAULT_ROLLING_L = 3
DEFAULT_GAMMA = 1.1  # turning-point buy threshold (Part 1d, required value)


def rolling_weights(l: int) -> np.ndarray:
    """Linearly decaying weights w_0..w_l that sum to 1.

    w_j is proportional to (l + 1 - j), giving the most recent (target-day)
    price the largest weight. For l=3 this is [0.4, 0.3, 0.2, 0.1].
    """
    w = np.arange(l + 1, 0, -1, dtype=np.float64)
    return (w / w.sum()).astype(np.float32)


@dataclass
class Normalizer:
    """Per-feature standardization fitted on training windows only."""
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _label_split(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    if date <= pd.Timestamp(VAL_END):
        return "val"
    return "test"


def _target_for_horizon(close, t, d, target, weights):
    """Return ratio at horizon d for anchor t, for the chosen target type."""
    p_t = close[t]
    if target == "exact":
        return (close[t + d] - p_t) / p_t
    # rolling: weighted average of prices p^{t+d}, p^{t+d-1}, ..., p^{t+d-l}
    l = len(weights) - 1
    avg = sum(weights[j] * close[t + d - j] for j in range(l + 1))
    return (avg - p_t) / p_t


def build_windows(df: pd.DataFrame, T: int, horizons,
                  target="exact", weights=None, gamma=DEFAULT_GAMMA):
    """Return arrays X (n,T,F), y, and the split label per window.

    A window anchored at index t is assigned to a split by the date of its
    *anchor* day t (the last day of the input window), so that no window
    leaks future test prices into training.

    Target types:
      * "exact"/"rolling": y is a length-D regression vector of return ratios.
      * "turning" (Part 1d): y is a single binary label, 1 (buy) if the
        max-price d-day return  (pmax^{t+d} - p^t) / p^t  exceeds ``gamma``
        for any d, else 0 (pass). The max price pmax^{t+d} is the daily High.
    """
    feats = df[FEATURES].to_numpy(dtype=np.float32)
    close = df["Close"].to_numpy(dtype=np.float32)
    high = df["High"].to_numpy(dtype=np.float32)
    dates = df.index
    max_h = max(horizons)

    X, y, splits = [], [], []
    n = len(df)
    for t in range(T - 1, n - max_h):
        window = feats[t - T + 1: t + 1]                 # (T, F)
        if target == "turning":
            p_t = close[t]
            max_ret = max((high[t + d] - p_t) / p_t for d in horizons)
            label = [1.0 if max_ret > gamma else 0.0]
            y.append(label)
        else:
            y.append([_target_for_horizon(close, t, d, target, weights)
                      for d in horizons])
        X.append(window)
        splits.append(_label_split(dates[t]))

    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(splits))


def load_all(data_dir: str, tickers, T=DEFAULT_T, horizons=DEFAULT_HORIZONS,
             target="exact", rolling_l=DEFAULT_ROLLING_L, gamma=DEFAULT_GAMMA):
    """Load every ticker CSV, build windows, concatenate, and split.

    Returns a dict with X/y arrays per split plus a fitted Normalizer.
    Normalization statistics are computed on training windows only.
    ``target`` selects the target type ("exact", "rolling", or "turning").
    """
    weights = rolling_weights(rolling_l) if target == "rolling" else None
    all_X, all_y, all_split = [], [], []
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}.csv")
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        X, y, splits = build_windows(df, T, horizons, target, weights, gamma)
        all_X.append(X)
        all_y.append(y)
        all_split.append(splits)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    splits = np.concatenate(all_split)

    train_mask = splits == "train"
    train_X = X[train_mask].reshape(-1, X.shape[-1])
    norm = Normalizer(mean=train_X.mean(axis=0), std=train_X.std(axis=0) + 1e-8)

    data = {}
    for name in ("train", "val", "test"):
        mask = splits == name
        data[name] = (norm.transform(X[mask]).astype(np.float32), y[mask])
    data["normalizer"] = norm
    return data


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
