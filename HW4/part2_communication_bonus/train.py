
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import CommunicationSystem, sample_messages

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def symbol_error_rate(logits, msg):
    pred = logits.argmax(dim=-1)
    return (pred != msg).float().mean().item()


@torch.no_grad()
def evaluate_ser(model, n_msgs, batch, device):
    model.eval()
    wrong_sym, total_sym = 0, 0
    for _ in range(max(1, n_msgs // batch)):
        msg = sample_messages(batch, model.n_symbols, model.alphabet, device)
        pred = model(msg).argmax(dim=-1)
        wrong_sym += (pred != msg).sum().item()
        total_sym += msg.numel()
    model.train()
    return wrong_sym / total_sym


def main():
    p = argparse.ArgumentParser(description="Train Part 2 comm system.")
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--sigma2", type=float, default=0.25, help="noise variance")
    p.add_argument("--alphabet", type=int, default=8)
    p.add_argument("--n-symbols", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--no-feedback", action="store_true",
                   help="open-loop ablation: TX does not see the relay feedback")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="comm",
                   help="prefix for saved artifacts (use a distinct tag for "
                        "supplementary runs so they do not overwrite)")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CommunicationSystem(
        n_symbols=args.n_symbols, alphabet=args.alphabet, T=args.T,
        sigma=args.sigma2 ** 0.5, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff,
        use_feedback=not args.no_feedback).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | params: {n_params} | T={args.T} "
          f"sigma^2={args.sigma2} (SNR per symbol ~ "
          f"{10 * np.log10((1 / args.n_symbols) / args.sigma2):.1f} dB)")

    history, best_ser = [], 1.0
    ckpt_path = os.path.join(RESULTS_DIR, f"{args.tag}_best.pt")
    start = time.time()
    for step in range(1, args.steps + 1):
        msg = sample_messages(args.batch_size, args.n_symbols, args.alphabet,
                              device)
        logits = model(msg)
        loss = F.cross_entropy(logits.reshape(-1, args.alphabet),
                               msg.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if step % 100 == 0 or step == 1:
            tr_ser = symbol_error_rate(logits, msg)
            val_ser = evaluate_ser(model, 20000, 2000, device)
            history.append({"step": step, "loss": loss.item(),
                            "train_ser": tr_ser, "val_ser": val_ser})
            if val_ser < best_ser:
                best_ser = val_ser
                torch.save({"state_dict": model.state_dict(),
                            "args": vars(args)}, ckpt_path)
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"train SER {tr_ser:.4f} | val SER {val_ser:.4f}")
    elapsed = time.time() - start

    with open(os.path.join(RESULTS_DIR, f"{args.tag}_train_log.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_ser": best_ser,
                   "elapsed_sec": elapsed, "history": history}, f, indent=2)

    steps = [h["step"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, [h["train_ser"] for h in history], label="train SER")
    plt.plot(steps, [h["val_ser"] for h in history], label="val SER")
    plt.xlabel("step"); plt.ylabel("symbol error rate"); plt.yscale("log")
    plt.title("Part 2: training curve"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{args.tag}_curve.png"), dpi=120)
    plt.close()

    print(f"\nBest val SER: {best_ser:.4f} | {elapsed:.1f}s")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
