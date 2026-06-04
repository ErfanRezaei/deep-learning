
import torch
import torch.nn as nn


class _RecurrentForecaster(nn.Module):
    """Shared scaffolding for the LSTM and GRU variants."""

    rnn_cls = None  # set by subclasses

    def __init__(self, input_size=4, hidden_size=64, num_layers=2,
                 output_size=5, dropout=0.2):
        super().__init__()
        self.rnn = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)          # (batch, T, hidden)
        last = out[:, -1, :]          # final time step
        return self.head(self.dropout(last))


class StockLSTM(_RecurrentForecaster):
    rnn_cls = nn.LSTM


class StockGRU(_RecurrentForecaster):
    rnn_cls = nn.GRU


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return StockLSTM(**kwargs)
    if name == "gru":
        return StockGRU(**kwargs)
    raise ValueError(f"Unknown model '{name}' (expected 'lstm' or 'gru').")


class TurningPointClassifier(nn.Module):
    """Bidirectional LSTM/GRU buy-vs-pass detector (Part 1d).

    Kept separate from the regression forecasters: it is bidirectional and
    outputs a single logit. The summary fed to the head concatenates the
    final hidden states of the forward and backward directions of the last
    layer (the canonical bi-RNN sequence representation).
    """

    def __init__(self, rnn_type="lstm", input_size=4, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, 1)  # buy/pass logit

    def forward(self, x):
        out, hidden = self.rnn(x)
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden  # LSTM vs GRU
        fwd, bwd = h_n[-2], h_n[-1]                # last layer, both directions
        summary = torch.cat([fwd, bwd], dim=1)     # (batch, 2*hidden)
        return self.head(self.dropout(summary)).squeeze(1)  # (batch,)


def build_classifier(rnn_type: str, **kwargs) -> nn.Module:
    if rnn_type.lower() not in ("lstm", "gru"):
        raise ValueError(f"Unknown rnn '{rnn_type}' (expected 'lstm'/'gru').")
    return TurningPointClassifier(rnn_type=rnn_type, **kwargs)
