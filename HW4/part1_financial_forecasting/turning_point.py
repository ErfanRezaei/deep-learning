
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import (DEFAULT_GAMMA, DEFAULT_HORIZONS, DEFAULT_T, FEATURES,
                     WindowDataset, load_all)
from models import build_classifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_balance(y):
    n = len(y)
    pos = int(y.sum())
    return {"n": n, "buy": pos, "pass": n - pos,
            "buy_rate": (pos / n) if n else 0.0}


def classification_metrics(probs, true, threshold=0.5):
    pred = (probs >= threshold).astype(np.int64)
    true = true.astype(np.int64)
    tp = int(((pred == 1) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    acc = (tp + tn) / max(len(true), 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "buy_rate_true": float(true.mean()) if len(true) else 0.0,
        "buy_rate_pred": float(pred.mean()) if len(pred) else 0.0,
        "support": len(true),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(1)            # (batch,)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / max(count, 1)


@torch.no_grad()
def predict_probs(model, X, device, batch_size=256):
    model.eval()
    loader = DataLoader(WindowDataset(X, np.zeros((len(X), 1), np.float32)),
                        batch_size=batch_size)
    out = [torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in loader]
    return np.concatenate(out) if out else np.array([])


def get_args():
    p = argparse.ArgumentParser(description="Turning-point detector (Part 1d).")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model(args, device):
    return build_classifier(
        args.rnn, input_size=len(FEATURES), hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout).to(device)


def train(args, data, device, tag):
    Xtr, ytr = data["train"]
    balance = {s: class_balance(data[s][1]) for s in ("train", "val", "test")}
    print("Class balance (buy rate):  " + "  ".join(
        f"{s}={balance[s]['buy']}/{balance[s]['n']} "
        f"({balance[s]['buy_rate']:.4%})" for s in balance))

    train_loader = DataLoader(WindowDataset(Xtr, ytr),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(*data["val"]),
                            batch_size=args.batch_size)

    # Counter class imbalance with a positive-class weight (if any positives).
    n_pos = int(ytr.sum())
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos else 1.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    history, best_val = [], float("inf")
    ckpt_path = os.path.join(RESULTS_DIR, f"{tag}_best.pt")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val:
            best_val = va
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args)}, ckpt_path)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d} | train BCE {tr:.4f} | val BCE {va:.4f}")

    with open(os.path.join(RESULTS_DIR, f"{tag}_train_log.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_loss": best_val,
                   "class_balance": balance, "history": history}, f, indent=2)
    print(f"\nBest val BCE: {best_val:.4f}\nCheckpoint: {ckpt_path}")


def evaluate(args, data, device, tag):
    ckpt_path = os.path.join(RESULTS_DIR, f"{tag}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    ck = ckpt["args"]
    model = build_classifier(
        args.rnn, input_size=len(FEATURES), hidden_size=ck["hidden_size"],
        num_layers=ck["num_layers"], dropout=ck["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    summary = {"gamma": args.gamma, "splits": {}}
    for split in ("train", "val", "test"):
        X, y = data[split]
        probs = predict_probs(model, X, device)
        m = classification_metrics(probs, y.squeeze(1))
        m["class_balance"] = class_balance(y)
        summary["splits"][split] = m
        cm = m["confusion_matrix"]
        print(f"[{split:5s}] acc {m['accuracy']:.4f} | P {m['precision']:.4f} | "
              f"R {m['recall']:.4f} | F1 {m['f1']:.4f} | "
              f"buy(true/pred) {m['buy_rate_true']:.4%}/{m['buy_rate_pred']:.4%} "
              f"| TP{cm['tp']} FP{cm['fp']} FN{cm['fn']} TN{cm['tn']}")

    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Training-curve plot (if a log exists).
    log_path = os.path.join(RESULTS_DIR, f"{tag}_train_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            hist = json.load(f)["history"]
        ep = [h["epoch"] for h in hist]
        plt.figure(figsize=(6, 4))
        plt.plot(ep, [h["train_loss"] for h in hist], label="train")
        plt.plot(ep, [h["val_loss"] for h in hist], label="val")
        plt.xlabel("epoch"); plt.ylabel("BCE loss")
        plt.title(f"{tag} training curve"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{tag}_curve.png"), dpi=120)
        plt.close()

    if summary["splits"]["test"]["class_balance"]["buy"] == 0:
        print("\nWARNING: zero buy samples at gamma="
              f"{args.gamma}. Classification metrics are degenerate; this "
              "reflects the assignment threshold, not a code bug.")
    print(f"\nSaved metrics to results/{tag}_metrics.json")


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Official gamma keeps the plain tag; supplementary thresholds get a suffix
    # so they never overwrite the required gamma=1.1 results.
    suffix = "" if args.gamma == DEFAULT_GAMMA else f"_g{args.gamma:g}"
    tag = f"turning_{args.rnn}{suffix}"

    data = load_all(DATA_DIR, args.tickers, T=DEFAULT_T,
                    horizons=DEFAULT_HORIZONS, target="turning",
                    gamma=args.gamma)
    print(f"Device: {device} | rnn: {args.rnn} (bidirectional) | "
          f"gamma: {args.gamma} | mode: {args.mode}")

    if args.mode == "train":
        train(args, data, device, tag)
    else:
        evaluate(args, data, device, tag)


if __name__ == "__main__":
    main()
