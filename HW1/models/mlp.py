from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.hidden_blocks = nn.ModuleList()

        in_features = input_size

        for hidden_dim in hidden_dims:
            layers = [nn.Linear(in_features, hidden_dim)]

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self._get_activation(activation))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            block = nn.Sequential(*layers)
            self.hidden_blocks.append(block)

            in_features = hidden_dim

        self.output_layer = nn.Linear(in_features, num_classes)

    def _get_activation(self, activation: str) -> nn.Module:
        activation = activation.lower()

        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()

        raise ValueError("Activation must be either 'relu' or 'gelu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        for block in self.hidden_blocks:
            x = block(x)

        x = self.output_layer(x)
        return x
    