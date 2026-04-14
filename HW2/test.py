from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
) -> dict[str, float]:
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


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Alias wrapper used by HW2 evaluation pipeline."""
    return evaluate(
        model=model,
        data_loader=dataloader,
        criterion=criterion,
        device=device,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def save_metrics_json(metrics: dict[str, Any], output_path: str) -> None:
    """Save metrics dictionary as JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Metrics saved to: {output_file}")