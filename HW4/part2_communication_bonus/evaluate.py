
import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import (CommunicationSystem, awgn_capacity_bits,
                   fano_bler_lower_bound, power_normalize, sample_messages)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Evaluate Part 2 comm system.")
    p.add_argument("--tag", default="comm")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n-messages", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ckpt_path = args.checkpoint or os.path.join(RESULTS_DIR, f"{args.tag}_best.pt")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    ck = ckpt["args"]

    use_feedback = not ck.get("no_feedback", False)
    model = CommunicationSystem(
        n_symbols=ck["n_symbols"], alphabet=ck["alphabet"], T=ck["T"],
        sigma=ck["sigma2"] ** 0.5, d_model=ck["d_model"], n_heads=ck["n_heads"],
        n_layers=ck["n_layers"], d_ff=ck["d_ff"],
        use_feedback=use_feedback).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Measured per-round transmit power, to confirm the constraint E||x||^2<=1.
    with torch.no_grad():
        _msg = sample_messages(4096, ck["n_symbols"], ck["alphabet"], device)
        oh = torch.nn.functional.one_hot(_msg, ck["alphabet"]).float()
        tx_cols, fb_cols, powers = [], [], []
        for t in range(ck["T"]):
            th = model._pad(tx_cols, ck["T"] - 1, oh)
            fh = model._pad(fb_cols, ck["T"] - 1, oh)
            ridx = oh.new_full((oh.size(0), ck["n_symbols"], 1),
                               t / max(ck["T"] - 1, 1))
            raw = torch.cat([oh, th, fh, ridx], dim=-1)
            x = power_normalize(model.encoder(raw).squeeze(-1))
            powers.append(x.pow(2).sum(dim=1).mean().item())  # E||x^(t)||^2
            if t < ck["T"] - 1:
                tx_cols.append(x)
                if use_feedback:
                    fb_cols.append(x + model.sigma * torch.randn_like(x))
    mean_round_power = float(np.mean(powers))

    wrong_sym = total_sym = 0
    wrong_blk = total_blk = 0
    pos_wrong = np.zeros(ck["n_symbols"])
    for _ in range(max(1, args.n_messages // args.batch_size)):
        msg = sample_messages(args.batch_size, ck["n_symbols"], ck["alphabet"],
                              device)
        pred = model(msg).argmax(dim=-1)
        mism = (pred != msg)
        wrong_sym += mism.sum().item()
        total_sym += msg.numel()
        wrong_blk += mism.any(dim=1).sum().item()
        total_blk += msg.size(0)
        pos_wrong += mism.float().sum(dim=0).cpu().numpy()

    ser = wrong_sym / total_sym
    bler = wrong_blk / total_blk
    pos_acc = (1 - pos_wrong / total_blk).tolist()
    summary = {
        "T": ck["T"], "sigma2": ck["sigma2"], "alphabet": ck["alphabet"],
        "n_symbols": ck["n_symbols"], "use_feedback": use_feedback,
        "n_messages": total_blk,
        "ser": ser, "symbol_accuracy": 1 - ser, "bler": bler,
        "per_position_accuracy": pos_acc,
        "mean_round_power": mean_round_power,
        "capacity_bits": awgn_capacity_bits(ck["sigma2"], ck["T"], ck["n_symbols"]),
        "message_entropy_bits": ck["n_symbols"] * np.log2(ck["alphabet"]),
        "fano_bler_lower_bound": fano_bler_lower_bound(
            ck["sigma2"], ck["T"], ck["n_symbols"], ck["alphabet"]),
    }
    print(f"messages: {total_blk}  T={ck['T']}  sigma^2={ck['sigma2']}  "
          f"feedback={use_feedback}")
    print(f"SER {ser:.5f} | symbol acc {1 - ser:.5f} | BLER {bler:.5f} | "
          f"mean round power {mean_round_power:.4f}")
    print("per-position acc: " +
          "  ".join(f"s{i+1}={a:.4f}" for i, a in enumerate(pos_acc)))

    metrics_path = os.path.join(RESULTS_DIR, f"{args.tag}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(5, 4))
    plt.bar(range(1, ck["n_symbols"] + 1), pos_acc)
    plt.ylim(0, 1); plt.xlabel("symbol position"); plt.ylabel("accuracy")
    plt.title("Part 2: per-position symbol accuracy"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{args.tag}_per_position.png"), dpi=120)
    plt.close()
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
