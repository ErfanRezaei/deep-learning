
import argparse
import csv
import json
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import awgn_capacity_bits, capacity_threshold_sigma2

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")

# (sigma^2, use_feedback) configurations. SNR sweep uses feedback; two points
# are repeated without feedback as an ablation.
SNR_POINTS = [0.25, 0.15, 0.10, 0.05, 0.02, 0.01]
ABLATION_POINTS = [0.25, 0.10]


def run(sigma2, feedback, steps):
    tag = f"sweep_s{sigma2:g}" + ("" if feedback else "_nofb")
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--sigma2", str(sigma2), "--steps", str(steps), "--tag", tag]
    if not feedback:
        cmd.append("--no-feedback")
    subprocess.run(cmd, check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"),
                    "--tag", tag, "--n-messages", "100000"],
                   check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(os.path.join(RESULTS_DIR, f"{tag}_metrics.json")) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Part 2 SNR sweep + ablation.")
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()

    rows = []
    for s2 in SNR_POINTS:
        print(f"[sweep] sigma^2={s2}  feedback=True ...")
        rows.append(run(s2, True, args.steps))
    for s2 in ABLATION_POINTS:
        print(f"[ablation] sigma^2={s2}  feedback=False ...")
        rows.append(run(s2, False, args.steps))

    # Tidy CSV.
    csv_path = os.path.join(RESULTS_DIR, "sweep_results.csv")
    fields = ["sigma2", "use_feedback", "snr_db", "capacity_bits",
              "message_entropy_bits", "ser", "bler", "symbol_accuracy",
              "mean_round_power", "fano_bler_lower_bound"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            snr_db = 10 * math.log10((1.0 / r["n_symbols"]) / r["sigma2"])
            w.writerow({k: r.get(k) for k in fields} | {"snr_db": round(snr_db, 2)})

    # SER vs SNR plot: feedback-on sweep + ablation points + capacity threshold.
    def snr(r):
        return 10 * math.log10((1.0 / r["n_symbols"]) / r["sigma2"])

    fb = sorted([r for r in rows if r["use_feedback"]], key=snr)
    nofb = sorted([r for r in rows if not r["use_feedback"]], key=snr)
    H = rows[0]["message_entropy_bits"]
    # sigma^2 where capacity == H  ->  threshold SNR in dB (verified helper).
    s2_thr = capacity_threshold_sigma2(rows[0]["T"], rows[0]["n_symbols"],
                                       rows[0]["alphabet"])
    snr_thr = 10 * math.log10((1.0 / rows[0]["n_symbols"]) / s2_thr)

    plt.figure(figsize=(6.5, 4.5))
    plt.semilogy([snr(r) for r in fb], [max(r["ser"], 1e-5) for r in fb],
                 "o-", label="feedback (relay)")
    if nofb:
        plt.semilogy([snr(r) for r in nofb], [max(r["ser"], 1e-5) for r in nofb],
                     "s--", label="no feedback (open loop)")
    plt.axvline(snr_thr, color="gray", ls=":",
                label=f"capacity = {H:.0f} bits (SNR={snr_thr:.1f} dB)")
    plt.gca().invert_xaxis()  # higher noise (lower SNR) on the right
    plt.xlabel("SNR per channel use (dB)"); plt.ylabel("symbol error rate")
    plt.title("Part 2 (supplementary): SER vs SNR"); plt.legend()
    plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sweep_ser_vs_snr.png"), dpi=120)
    plt.close()

    print(f"\nSaved {csv_path} and sweep_ser_vs_snr.png")
    print(f"Capacity crosses the {H:.0f}-bit message at sigma^2~{s2_thr:.3f} "
          f"(SNR~{snr_thr:.1f} dB).")


if __name__ == "__main__":
    main()
