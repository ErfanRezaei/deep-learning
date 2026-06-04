
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (DEFAULT_HORIZONS, DEFAULT_T, FEATURES, WindowDataset,
                     load_all)
from models import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM"]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / count


def main():
    parser = argparse.ArgumentParser(description="Train return forecaster.")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--target", choices=["exact", "rolling"],
                        default="exact",
                        help="exact d-day returns (1b) or weighted "
                             "rolling-average returns (1c)")
    parser.add_argument("--rolling-l", type=int, default=3,
                        help="rolling-average window l (used when "
                             "--target rolling)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"{args.model}_{args.target}"
    data = load_all(DATA_DIR, args.tickers, T=DEFAULT_T,
                    horizons=DEFAULT_HORIZONS, target=args.target,
                    rolling_l=args.rolling_l)
    train_loader = DataLoader(WindowDataset(*data["train"]),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(*data["val"]),
                            batch_size=args.batch_size)

    model = build_model(
        args.model,
        input_size=len(FEATURES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(DEFAULT_HORIZONS),
        dropout=args.dropout,
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    print(f"Device: {device} | model: {args.model} | target: {args.target} | "
          f"train/val windows: {len(data['train'][0])}/{len(data['val'][0])}")

    history = []
    best_val = float("inf")
    ckpt_path = os.path.join(RESULTS_DIR, f"{tag}_best.pt")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
        if va < best_val:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args)}, ckpt_path)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d} | train MSE {tr:.6e} | val MSE {va:.6e}")
    elapsed = time.time() - start

    log_path = os.path.join(RESULTS_DIR, f"{tag}_train_log.json")
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "best_val_mse": best_val,
                   "elapsed_sec": elapsed, "history": history}, f, indent=2)
    print(f"\nBest val MSE: {best_val:.6e} | {elapsed:.1f}s")
    print(f"Checkpoint: {ckpt_path}\nLog: {log_path}")


if __name__ == "__main__":
    main()
