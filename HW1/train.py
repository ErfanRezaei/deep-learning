from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from parameters import TrainingConfig


def get_mnist_dataloaders(
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=config.download,
    )

    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        transform=transform,
        download=config.download,
    )

    val_size = int(len(full_train_dataset) * config.val_split)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(config.random_seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for parameter in model.parameters():
        l1_penalty += parameter.abs().sum()
    return l1_penalty


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l1_lambda: float = 0.0,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if l1_lambda > 0.0:
            loss = loss + l1_lambda * compute_l1_penalty(model)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return epoch_loss, epoch_accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return epoch_loss, epoch_accuracy


def plot_training_history(
    history: dict[str, list[float]],
    report_dir: str,
) -> None:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(report_path / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(report_path / "accuracy_curve.png")
    plt.close()