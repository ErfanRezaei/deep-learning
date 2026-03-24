from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import AverageMeter, compute_accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a model on a validation or test dataloader."""

    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        accuracy = compute_accuracy(logits, labels)

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy, batch_size)

    return {
        "loss": loss_meter.average,
        "accuracy": acc_meter.average,
    }