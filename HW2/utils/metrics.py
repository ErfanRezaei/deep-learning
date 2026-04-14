from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:

    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return 100.0 * correct / total