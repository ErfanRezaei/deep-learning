
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Standard post-LN block: LN(h + MHA(h)) then LN(h + FFN(h))."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h):
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ffn(h))
        return h


class TransformerCoder(nn.Module):
    """MLP -> (+positional encoding) -> transformer blocks -> MLP.

    Used for both TX encoder and RX decoder (Hint 3: an MLP before and after
    the transformer module). Operates on the fixed sequence of 4 symbol tokens.
    """

    def __init__(self, in_dim, out_dim, n_tokens=4, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(),
                                     nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)])
        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                      nn.Linear(d_model, out_dim))

    def forward(self, raw):                 # raw: (B, n_tokens, in_dim)
        h = self.in_proj(raw) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)             # (B, n_tokens, out_dim)


def power_normalize(x):
    """Scale a per-round code (B, n) so E[||x||^2] = 1 (avg power constraint).

    Uses the batch mean power so the whole 4-symbol round vector has unit
    expected energy (each symbol then carries power 1/n).
    """
    n = x.size(-1)
    power = x.pow(2).mean()                 # E[per-symbol power] over the batch
    return x / torch.sqrt(n * power + 1e-9)


class CommunicationSystem(nn.Module):
    """End-to-end TX encoder + AWGN channel + noiseless relay + RX decoder."""

    def __init__(self, n_symbols=4, alphabet=8, T=4, sigma=0.5,
                 d_model=64, n_heads=4, n_layers=2, d_ff=128,
                 use_feedback=True):
        super().__init__()
        self.n_symbols = n_symbols
        self.alphabet = alphabet
        self.T = T
        self.sigma = sigma
        self.use_feedback = use_feedback   # False = open-loop ablation

        # TX raw features per token: one-hot symbol | past tx | past feedback |
        # round index. Past buffers have width T-1 (rounds seen so far).
        tx_in = alphabet + 2 * (T - 1) + 1
        self.encoder = TransformerCoder(tx_in, 1, n_symbols, d_model, n_heads,
                                        n_layers, d_ff)
        # RX sees the T noisy symbols collected per token.
        self.decoder = TransformerCoder(T, alphabet, n_symbols, d_model,
                                        n_heads, n_layers, d_ff)

    def _pad(self, cols, width, ref):
        """Stack history columns and right-pad with zeros to a fixed width."""
        B = ref.size(0)
        h = torch.stack(cols, dim=-1) if cols else ref.new_zeros(B, self.n_symbols, 0)
        if h.size(-1) < width:
            pad = ref.new_zeros(B, self.n_symbols, width - h.size(-1))
            h = torch.cat([h, pad], dim=-1)
        return h

    def forward(self, msg):                 # msg: (B, n_symbols) long in [0,A)
        onehot = F.one_hot(msg, self.alphabet).float()
        tx_cols, fb_cols, received = [], [], []
        for t in range(self.T):
            tx_hist = self._pad(tx_cols, self.T - 1, onehot)
            fb_hist = self._pad(fb_cols, self.T - 1, onehot)
            ridx = onehot.new_full((onehot.size(0), self.n_symbols, 1),
                                   t / max(self.T - 1, 1))
            raw = torch.cat([onehot, tx_hist, fb_hist, ridx], dim=-1)
            r = self.encoder(raw).squeeze(-1)            # (B, n_symbols)
            x = power_normalize(r)
            y = x + self.sigma * torch.randn_like(x)     # AWGN forward channel
            received.append(y)
            if t < self.T - 1:
                tx_cols.append(x)            # keep grad: backprop through rounds
                if self.use_feedback:
                    fb_cols.append(y)        # noiseless relay feedback (Hint 1)
                # else: feedback buffer stays zero -> open-loop code
        Y = torch.stack(received, dim=-1)    # (B, n_symbols, T)
        return self.decoder(Y)               # (B, n_symbols, alphabet) logits


def sample_messages(batch_size, n_symbols=4, alphabet=8, device="cpu"):
    return torch.randint(0, alphabet, (batch_size, n_symbols), device=device)


# --- Channel information-theoretic references (for cautious analysis) ---

def awgn_capacity_bits(sigma2, T=4, n_symbols=4):
    """Shannon capacity (bits) of the whole T-round interaction.

    There are ``T * n_symbols`` real channel uses. The per-round average power
    constraint E||x||^2 <= 1 over ``n_symbols`` parallel uses is maximised by
    equal allocation, giving power ``1/n_symbols`` per use. For an AWGN use,
    C = 0.5*log2(1 + SNR). Feedback does NOT increase AWGN capacity, so this
    is an upper bound on reliably communicable bits for either system.
    """
    p_per_use = 1.0 / n_symbols
    snr = p_per_use / sigma2
    return 0.5 * math.log2(1.0 + snr) * (T * n_symbols)


def capacity_threshold_sigma2(T=4, n_symbols=4, alphabet=8):
    """Noise variance at which capacity exactly equals the message entropy.

    Solves H = T*n_symbols * 0.5*log2(1 + (1/n_symbols)/sigma2) for sigma2,
    where H = n_symbols*log2(alphabet). Below this noise level reliable
    transmission becomes information-theoretically possible.
    """
    H = n_symbols * math.log2(alphabet)
    snr = 2 ** (H / (0.5 * T * n_symbols)) - 1
    return (1.0 / n_symbols) / snr


def fano_bler_lower_bound(sigma2, T=4, n_symbols=4, alphabet=8):
    """Fano lower bound on block (message) error rate.

    With H(M) = n_symbols*log2(alphabet) bits and I(M;Y) <= capacity,
    H(M|Y) >= H(M) - C, and Fano gives
        H(M|Y) <= 1 + P_e * log2(|M| - 1).
    This is a loose but valid floor; it never proves a given SER is optimal.
    """
    H = n_symbols * math.log2(alphabet)
    C = awgn_capacity_bits(sigma2, T, n_symbols)
    H_cond = max(H - C, 0.0)
    M = alphabet ** n_symbols
    return max((H_cond - 1.0) / math.log2(M - 1), 0.0)
