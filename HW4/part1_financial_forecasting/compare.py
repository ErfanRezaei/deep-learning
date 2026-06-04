
import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import DEFAULT_HORIZONS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TARGETS = ["exact", "rolling"]


def _load(tag, kind):
    path = os.path.join(RESULTS_DIR, f"{tag}_{kind}")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stability(history):
    """Smoothness metrics of the validation-MSE curve."""
    val = np.array([h["val_mse"] for h in history])
    half = val[len(val) // 2:]
    jitter = float(np.abs(np.diff(val)).mean()) if len(val) > 1 else 0.0
    return {
        "final_val_mse": float(val[-1]),
        "val_std_second_half": float(half.std()),
        "val_jitter_mean_abs_step": jitter,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare exact vs rolling.")
    parser.add_argument("--models", nargs="+", default=["lstm", "gru"])
    args = parser.parse_args()

    rows, summary, curves = [], {}, {}
    for model in args.models:
        for target in TARGETS:
            tag = f"{model}_{target}"
            metrics = _load(tag, "metrics.json")
            log = _load(tag, "train_log.json")
            if metrics is None or log is None:
                print(f"skip {tag}: missing results (train+evaluate it first)")
                continue

            for split in ("train", "val", "test"):
                m = metrics[split]
                rows.append({"model": model, "target": target, "split": split,
                             "scope": "overall", "mse": m["mse"],
                             "mae": m["mae"]})
                for d, v in m["per_horizon_mse"].items():
                    rows.append({"model": model, "target": target,
                                 "split": split, "scope": d, "mse": v,
                                 "mae": ""})

            summary[tag] = {
                "test_mse": metrics["test"]["mse"],
                "val_mse": metrics["val"]["mse"],
                "train_mse": metrics["train"]["mse"],
                "best_val_mse": log.get("best_val_mse"),
                **stability(log["history"]),
            }
            curves[tag] = [h["val_mse"] for h in log["history"]]

    if not rows:
        print("Nothing to compare. Train+evaluate at least one run first.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-horizon test MSE: grouped bars, exact vs rolling per model.
    present = [m for m in args.models
               if f"{m}_exact" in summary or f"{m}_rolling" in summary]
    horizons = [f"d={d}" for d in DEFAULT_HORIZONS]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4),
                                 squeeze=False)
        x = np.arange(len(horizons))
        for ax, model in zip(axes[0], present):
            for i, target in enumerate(TARGETS):
                sub = df[(df.model == model) & (df.target == target)
                         & (df.split == "test") & (df.scope.isin(horizons))]
                if sub.empty:
                    continue
                vals = [sub[sub.scope == h]["mse"].values[0] for h in horizons]
                ax.bar(x + (i - 0.5) * 0.4, vals, width=0.4, label=target)
            ax.set_xticks(x); ax.set_xticklabels(horizons)
            ax.set_title(f"{model.upper()} test MSE per horizon")
            ax.set_ylabel("MSE"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "comparison_per_horizon.png"),
                    dpi=120)
        plt.close(fig)

    # Validation curves overlaid.
    if curves:
        plt.figure(figsize=(6, 4))
        for tag, vals in curves.items():
            plt.plot(range(1, len(vals) + 1), vals, label=tag)
        plt.xlabel("epoch"); plt.ylabel("val MSE"); plt.yscale("log")
        plt.title("Validation curves: exact vs rolling"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "comparison_curves.png"), dpi=120)
        plt.close()

    print("Comparison summary (overall):")
    print(df[df.scope == "overall"].to_string(index=False))
    print(f"\nSaved: comparison_metrics.csv, comparison_summary.json, "
          f"comparison_per_horizon.png, comparison_curves.png  (in results/)")


if __name__ == "__main__":
    main()
