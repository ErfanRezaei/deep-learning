import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import evaluate


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    return evaluate(model, test_loader, criterion, device)