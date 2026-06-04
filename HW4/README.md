# CS515 Deep Learning — Homework 4

Sequence modeling. **Part 1 (Financial Forecasting)** is complete and
report-ready; **Part 2 (the communication-protocol bonus)** has a working
end-to-end baseline.

## Repository layout

```
CS515-HW4/
├── part1_financial_forecasting/
│   ├── data_download.py   # download daily OHLC data via yfinance
│   ├── dataset.py         # sliding windows + chronological split + normalization
│   ├── models.py          # StockLSTM / StockGRU
│   ├── train.py           # MSE training loop (AdamW), checkpoint + logs
│   ├── evaluate.py        # train/val/test metrics, CSV + training-curve plot
│   ├── compare.py         # exact vs rolling-average comparison artifacts
│   ├── turning_point.py   # Part 1d: bidirectional buy/pass detector
│   └── results/           # logs, metrics, checkpoints, plots
├── part2_communication_bonus/
│   ├── model.py           # transformer TX encoder + AWGN channel + RX decoder
│   ├── train.py           # end-to-end training (cross-entropy over symbols)
│   ├── evaluate.py        # SER / BLER / per-position acc / power check
│   ├── sweep.py           # supplementary SNR sweep + feedback ablation
│   ├── summarize.py       # aggregate metrics -> PART2_SUMMARY.md
│   └── results/           # logs, metrics, checkpoints, plots
├── report.tex / report.pdf  # Part 1 report
├── requirements.txt
└── README.md
```

## Part 1 — current implementation

Two target types share the same data pipeline and the same LSTM/GRU models,
selected with `--target`:

- **`exact`** (Part 1b): exact d-day return `r^{t+d} = (p^{t+d} - p^t) / p^t`.
- **`rolling`** (Part 1c): weighted rolling-average return
  `r^{t+d} = (Σ_{j=0}^{l} w_j p^{t+d-j} - p^t) / p^t` with window `l = 3`.
  Weights decay linearly and sum to 1 (`[0.4, 0.3, 0.2, 0.1]`), keeping the
  target on the same scale as the exact return.

Common settings for both, `d = 1..5`:

- **Tickers:** `AAPL`, `MSFT`, `JPM` (three S&P 500 names; configurable).
- **Period:** Jan 2020 – Dec 2025, daily OHLC (`auto_adjust=True`).
- **Chronological split:**
  - train: Jan 2020 – Jul 2024
  - val:   Aug 2024 – Dec 2024
  - test:  Jan 2025 – Dec 2025
- **Features (F=4):** Open, High, Low, Close.
- **Windows:** lookback `T = 20`; target is the 5-vector of return ratios
  `r^{t+d} = (p^{t+d} - p^t) / p^t` for `d = 1..5`.
- **Normalization:** per-feature standardization fitted on **training windows
  only** (no leakage).
- **Models:** stacked `nn.LSTM` / `nn.GRU` → dropout → linear head, input
  `(batch, T, F)` → output `5`.
- **Training:** MSE loss, AdamW, best checkpoint by validation MSE.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

All commands are run from inside `part1_financial_forecasting/`. Every run is
identified by a tag `<model>_<target>` (e.g. `lstm_exact`, `gru_rolling`), so
results never overwrite each other.

```bash
cd part1_financial_forecasting

# 1. Download data  -> data/{AAPL,MSFT,JPM}.csv
python data_download.py

# 2. Train: choose --model {lstm,gru} and --target {exact,rolling}
python train.py --model lstm --target exact
python train.py --model gru  --target exact
python train.py --model lstm --target rolling
python train.py --model gru  --target rolling

# 3. Evaluate (same flags as training)
python evaluate.py --model lstm --target exact
python evaluate.py --model gru  --target exact
python evaluate.py --model lstm --target rolling
python evaluate.py --model gru  --target rolling

# 4. Compare exact vs rolling across both models
python compare.py
```

### Part 1d — turning-point detection (buy / pass)

A separate **bidirectional** LSTM/GRU classifier. Label = buy (1) if the
max-price d-day return `(pmax^{t+d} - p^t)/p^t` exceeds `gamma` for any
`d=1..5` (pmax = daily High); otherwise pass (0). Uses `BCEWithLogitsLoss`
with a positive-class weight to counter imbalance. Same data, split, and
`T=20` as the regression tasks. Results are tagged `turning_<rnn>`.

```bash
cd part1_financial_forecasting

# train then evaluate (gamma defaults to the required 1.1)
python turning_point.py --mode train --rnn lstm
python turning_point.py --mode eval  --rnn lstm
python turning_point.py --mode train --rnn gru
python turning_point.py --mode eval  --rnn gru

# supplementary sanity check at an achievable threshold (separate tag)
python turning_point.py --mode train --rnn lstm --gamma 0.05
python turning_point.py --mode eval  --rnn lstm --gamma 0.05
python turning_point.py --mode train --rnn gru  --gamma 0.05
python turning_point.py --mode eval  --rnn gru  --gamma 0.05
```

Non-default `--gamma` values are saved under a suffixed tag
(`turning_<rnn>_g<gamma>`) so they never overwrite the official `gamma=1.1`
results.

Flags: `--rnn {lstm,gru}`, `--gamma` (default `1.1`), `--epochs`, `--lr`,
`--hidden-size`, `--num-layers`, `--dropout`, `--tickers`.

**Class balance at gamma = 1.1 (important):** the assignment defines the
return ratio as `(pmax - p^t)/p^t`, so `gamma = 1.1` means a **>110% gain
within 5 trading days** — which never occurs for AAPL/MSFT/JPM over
2020–2025. Hence the buy class is **empty in all splits** and the metrics are
degenerate (accuracy 1.0, precision/recall/F1 = 0); the code prints an
explicit warning. We keep `gamma = 1.1` as the required default and report
this honestly. For reference, observed buy-rates at lower thresholds:

| gamma | train | val   | test  |
|------:|------:|------:|------:|
| 1.1   | 0.00% | 0.00% | 0.00% |
| 0.20  | 0.09% | 0.00% | 0.14% |
| 0.10  | 3.09% | 1.57% | 2.46% |
| 0.05  | 19.6% | 9.43% | 15.3% |

Running e.g. `--gamma 0.05` yields a non-degenerate problem on which the
detector learns real signal (val recall ≈ 0.70, F1 ≈ 0.39), confirming the
pipeline is correct; the degeneracy at 1.1 is purely the threshold.

Common flags: `--tickers AAPL MSFT JPM`, `--epochs`, `--lr`, `--hidden-size`,
`--num-layers`, `--dropout`, and `--rolling-l` (default 3). See `--help`.

## Outputs (in `results/`)

Per run (`<tag>` = `<model>_<target>`):

- `<tag>_train_log.json` — per-epoch train/val MSE and run config.
- `<tag>_best.pt` — best checkpoint (by val MSE) + its hyperparameters.
- `<tag>_metrics.{json,csv}` — overall + per-horizon MSE/MAE on all splits.
- `<tag>_curve.png` — training/validation loss curve.

Comparison (`compare.py`, exact vs rolling):

- `comparison_metrics.csv` — tidy overall + per-horizon MSE for every run.
- `comparison_summary.json` — overall MSE plus training-stability metrics
  (val-MSE std over the 2nd half of training, and mean epoch-to-epoch jitter;
  lower = more stable).
- `comparison_per_horizon.png` — per-horizon test MSE, exact vs rolling.
- `comparison_curves.png` — validation-loss curves, exact vs rolling.

Turning-point detection (`turning_point.py`, tag `turning_<rnn>`):

- `turning_<rnn>_best.pt` — best checkpoint (by val BCE) + hyperparameters.
- `turning_<rnn>_train_log.json` — per-epoch BCE, run config, class balance.
- `turning_<rnn>_metrics.json` — accuracy, precision, recall, F1, confusion
  matrix, and true/pred buy-rate on all splits (plus the gamma used).
- `turning_<rnn>_curve.png` — training/validation BCE curve.

## Final summary and report

After running the experiments above, aggregate everything and build the report:

```bash
# from part1_financial_forecasting/  -> results/PART1_SUMMARY.md
python summarize.py

# from the repo root CS515-HW4/  -> report.pdf
latexmk -pdf report.tex
```

- `results/PART1_SUMMARY.md` — single Markdown summary of every metric/table,
  auto-generated from the JSON artifacts (source of truth for the report).
- `report.tex` / `report.pdf` — the Part 1 academic report.

# Part 2 — Communication protocol (bonus): baseline

An interactive two-node system where a transformer **TX encoder** and **RX
decoder** are trained end-to-end to communicate a message `m ∈ {1..8}^4`
(four 8-ary symbols) over `T=4` rounds of an AWGN forward channel
(`σ²=0.25`), with a noiseless feedback relay.

- **Per round** the TX emits 4 coded symbols `x^(t) ∈ R⁴` (one per message
  symbol), power-normalised so `E‖x^(t)‖² = 1` (per-round average power).
- **Channel:** `y^(t) = x^(t) + ε`, `ε ~ N(0, σ²)`.
- **Feedback (Hint 1):** noiseless relay `f^(t) = y^(t)`; the TX conditions the
  next round on the one-hot message, its past transmissions, the relayed noisy
  feedback, and a round index.
- **Coder (Hint 3):** input MLP → positional encoding → standard transformer
  blocks (`LN(h+MHA)`, `LN(h+FFN)`) → output MLP.
- **Decoder (Hint 2):** runs once at the end on the collected `y^(1..T)` and
  classifies each of the 4 symbols. Loss = cross-entropy; optimiser AdamW.

Messages are sampled fresh each step (infinite data), so training is measured
in gradient steps.

```bash
cd part2_communication_bonus

# official setting (T=4, sigma^2=0.25) -- the main result
python train.py --steps 4000
python evaluate.py                 # -> results/comm_metrics.json, plots

# supplementary: SNR sweep + feedback ablation (trains several small models)
python sweep.py --steps 3000       # -> sweep_results.csv, sweep_ser_vs_snr.png

# aggregate everything into one Markdown summary
python summarize.py                # -> results/PART2_SUMMARY.md
```

Flags: `--T`, `--sigma2`, `--steps`, `--batch-size`, `--lr`, `--d-model`,
`--n-layers`, `--n-heads`, `--no-feedback` (open-loop ablation), `--tag`.
Outputs (tagged, default `comm`): `<tag>_best.pt`, `<tag>_train_log.json`,
`<tag>_curve.png`, `<tag>_metrics.json`, `<tag>_per_position.png`. The metrics
JSON also records the measured per-round power and the capacity reference.

**Baseline result.** At the required `σ²=0.25` the system reaches
**SER ≈ 0.28** (symbol accuracy ≈ 0.72, BLER ≈ 0.73), with measured per-round
power ≈ 1.0 (constraint satisfied). This regime is **capacity-limited, not a
bug**: with per-round power 1 split over 4 symbols the SNR is 0 dB/use, so the
16 channel uses carry ≈ 8 bits while the message is 12 bits — and since
feedback does not increase AWGN capacity, error-free transmission is not
achievable here (Fano floor BLER ≳ 0.25). The SNR sweep confirms this: SER
drops sharply once `σ² < 0.137` (where capacity exceeds 12 bits) and reaches
≈ 0 at high SNR, validating the implementation. The feedback ablation shows
the relay gives a modest reliability gain. All claims are stated cautiously in
`results/PART2_SUMMARY.md`.

## Not implemented yet

- Part 2 report section / integration into `report.tex` (the experiments and
  analysis are done; only the write-up remains).
