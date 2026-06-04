
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import (DEFAULT_HORIZONS, DEFAULT_T, FEATURES, WindowDataset,
                     load_all)
from models import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM"]


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    loader = DataLoader(WindowDataset(X, np.zeros((len(X), 1), np.float32)),
                        batch_size=batch_size)
    preds = [model(xb.to(device)).cpu().numpy() for xb, _ in loader]
    return np.concatenate(preds)


def metrics(pred, true):
    err = pred - true
    per_h = (err ** 2).mean(axis=0)               # MSE per horizon
    return {
        "mse": float((err ** 2).mean()),
        "mae": float(np.abs(err).mean()),
        "per_horizon_mse": {f"d={d}": float(m)
                            for d, m in zip(DEFAULT_HORIZONS, per_h)},
    }


def plot_history(log_path, tag):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        hist = json.load(f)["history"]
    epochs = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_mse"] for h in hist], label="train")
    plt.plot(epochs, [h["val_mse"] for h in hist], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.yscale("log")
    plt.title(f"{tag} training curve"); plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{tag}_curve.png")
    plt.savefig(out, dpi=120); plt.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecaster.")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--target", choices=["exact", "rolling"],
                        default="exact")
    parser.add_argument("--rolling-l", type=int, default=3)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{args.model}_{args.target}"
    ckpt_path = args.checkpoint or os.path.join(RESULTS_DIR, f"{tag}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    ck = ckpt["args"]

    model = build_model(
        args.model,
        input_size=len(FEATURES),
        hidden_size=ck["hidden_size"],
        num_layers=ck["num_layers"],
        output_size=len(DEFAULT_HORIZONS),
        dropout=ck["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    data = load_all(DATA_DIR, args.tickers, T=DEFAULT_T,
                    horizons=DEFAULT_HORIZONS, target=args.target,
                    rolling_l=args.rolling_l)

    summary = {}
    rows = []
    for split in ("train", "val", "test"):
        X, y = data[split]
        m = metrics(predict(model, X, device), y)
        summary[split] = m
        print(f"[{split:5s}] MSE {m['mse']:.6e} | MAE {m['mae']:.6e} | "
              f"n={len(X)}")
        for d, v in m["per_horizon_mse"].items():
            rows.append({"split": split, "horizon": d, "mse": v})

    # Persist metrics: JSON (full) + CSV (per-horizon, tidy for the report).
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    csv_path = os.path.join(RESULTS_DIR, f"{tag}_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    curve = plot_history(
        os.path.join(RESULTS_DIR, f"{tag}_train_log.json"), tag)

    print(f"\nSaved metrics JSON/CSV to results/ ({tag}_metrics.*)")
    if curve:
        print(f"Saved training curve to {curve}")


if __name__ == "__main__":
    main()
