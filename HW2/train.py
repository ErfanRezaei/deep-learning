from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from models import build_model
from parameters import ExperimentConfig
from test import evaluate
from utils.losses import compute_distillation_loss
from utils.metrics import AverageMeter, compute_accuracy


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> optim.Optimizer:
    """Build optimizer from experiment configuration."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    if config.optim.optimizer_name == "adam":
        return optim.Adam(
            trainable_parameters,
            lr=config.optim.learning_rate,
            weight_decay=config.optim.weight_decay,
        )

    if config.optim.optimizer_name == "sgd":
        return optim.SGD(
            trainable_parameters,
            lr=config.optim.learning_rate,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {config.optim.optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Build scheduler from experiment configuration."""

    if config.optim.scheduler_name == "none":
        return None

    if config.optim.scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=config.optim.step_size,
            gamma=config.optim.gamma,
        )

    if config.optim.scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.optim.epochs,
        )

    raise ValueError(f"Unsupported scheduler: {config.optim.scheduler_name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
) -> Dict[str, float]:
    """Train the model for one epoch with standard supervised learning."""

    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch [{epoch_index}/{total_epochs}] - Train",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(logits, labels)
        batch_size = labels.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy, batch_size)

        progress_bar.set_postfix(
            loss=f"{loss_meter.average:.4f}",
            acc=f"{acc_meter.average:.2f}%",
        )

    return {
        "loss": loss_meter.average,
        "accuracy": acc_meter.average,
    }


def train_one_epoch_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    hard_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
    alpha: float,
    temperature: float,
) -> Dict[str, float]:
    """Train the student model for one epoch using knowledge distillation."""

    student_model.train()
    teacher_model.eval()

    total_loss_meter = AverageMeter()
    hard_loss_meter = AverageMeter()
    soft_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch [{epoch_index}/{total_epochs}] - KD Train",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)

        total_loss, hard_loss, soft_loss = compute_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=labels,
            hard_criterion=hard_criterion,
            alpha=alpha,
            temperature=temperature,
        )

        total_loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(student_logits, labels)
        batch_size = labels.size(0)

        total_loss_meter.update(total_loss.item(), batch_size)
        hard_loss_meter.update(hard_loss.item(), batch_size)
        soft_loss_meter.update(soft_loss.item(), batch_size)
        acc_meter.update(accuracy, batch_size)

        progress_bar.set_postfix(
            loss=f"{total_loss_meter.average:.4f}",
            hard=f"{hard_loss_meter.average:.4f}",
            kd=f"{soft_loss_meter.average:.4f}",
            acc=f"{acc_meter.average:.2f}%",
        )

    return {
        "loss": total_loss_meter.average,
        "hard_loss": hard_loss_meter.average,
        "soft_loss": soft_loss_meter.average,
        "accuracy": acc_meter.average,
    }


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    config: ExperimentConfig,
) -> None:
    """Save a training checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "experiment_name": config.runtime.experiment_name,
        "model_name": config.runtime.model_name,
    }

    torch.save(checkpoint, checkpoint_path)


def save_history(history_path: str, history: Dict[str, Any]) -> None:
    """Save training history as JSON."""

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def load_checkpoint_weights(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
) -> None:
    """Load model weights from checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def load_teacher_model(
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    """Build and load the teacher model from a saved checkpoint."""

    if not config.distill.teacher_checkpoint:
        raise ValueError("Teacher checkpoint must be provided for distillation.")

    teacher_model = build_model(
        model_name=config.distill.teacher_model_name,
        num_classes=config.data.num_classes,
    ).to(device)

    checkpoint = torch.load(config.distill.teacher_checkpoint, map_location=device)
    teacher_model.load_state_dict(checkpoint["model_state_dict"])
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    return teacher_model


def train_model(
    model: nn.Module,
    dataloaders,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[Dict[str, Any], Dict[str, float], str]:
    """Train, validate, save best checkpoint, and finally test the model."""

    set_seed(config.data.seed)

    use_distillation = config.distill.use_distillation
    teacher_model = load_teacher_model(config, device) if use_distillation else None

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    if use_distillation:
        history["train_hard_loss"] = []
        history["train_soft_loss"] = []

    best_val_accuracy = -1.0
    checkpoint_path = str(
        Path(config.runtime.checkpoint_dir)
        / f"{config.runtime.experiment_name}_best.pt"
    )
    history_path = str(
        Path(config.runtime.logs_dir)
        / f"{config.runtime.experiment_name}_history.json"
    )

    for epoch in range(1, config.optim.epochs + 1):
        if use_distillation:
            train_metrics = train_one_epoch_distillation(
                student_model=model,
                teacher_model=teacher_model,
                data_loader=dataloaders.train_loader,
                hard_criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_index=epoch,
                total_epochs=config.optim.epochs,
                alpha=config.distill.alpha,
                temperature=config.distill.temperature,
            )
        else:
            train_metrics = train_one_epoch(
                model=model,
                data_loader=dataloaders.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch_index=epoch,
                total_epochs=config.optim.epochs,
            )

        val_metrics = evaluate(
            model=model,
            data_loader=dataloaders.val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if use_distillation:
            history["train_hard_loss"].append(train_metrics["hard_loss"])
            history["train_soft_loss"].append(train_metrics["soft_loss"])
            print(
                f"Epoch {epoch:02d}/{config.optim.epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Hard Loss: {train_metrics['hard_loss']:.4f} | "
                f"KD Loss: {train_metrics['soft_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{config.optim.epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                config=config,
            )
            print(f"Best checkpoint saved to: {checkpoint_path}")

    save_history(history_path, history)
    print(f"Training history saved to: {history_path}")

    load_checkpoint_weights(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
    )

    test_metrics = evaluate(
        model=model,
        data_loader=dataloaders.test_loader,
        criterion=criterion,
        device=device,
    )

    return history, test_metrics, checkpoint_path