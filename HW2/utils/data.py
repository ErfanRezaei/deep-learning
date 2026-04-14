from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from parameters import DataConfig
from utils.augmentations import build_eval_transform, build_train_transform
from utils.cifar10c import CIFAR10CDataset


@dataclass
class DataLoaders:
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: DataLoader
    classes: tuple[str, ...]


CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _build_train_val_indices(
    dataset_size: int,
    val_ratio: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def _apply_optional_subset(
    dataset,
    max_samples: Optional[int],
    seed: int,
):
    if max_samples is None:
        return dataset

    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def build_cifar10_dataloaders(config: DataConfig) -> DataLoaders:
    """Build train/val/test dataloaders for clean CIFAR-10."""
    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    train_dataset_full = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=train_transform,
        download=config.download,
    )

    val_dataset_full = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=eval_transform,
        download=False,
    )

    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        transform=eval_transform,
        download=config.download,
    )

    train_indices, val_indices = _build_train_val_indices(
        dataset_size=len(train_dataset_full),
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_dataset = _apply_optional_subset(
        train_dataset,
        config.max_train_samples,
        config.seed,
    )
    val_dataset = _apply_optional_subset(
        val_dataset,
        config.max_eval_samples,
        config.seed + 1,
    )
    test_dataset = _apply_optional_subset(
        test_dataset,
        config.max_eval_samples,
        config.seed + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    classes = tuple(train_dataset_full.classes)
    return DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
    )


def build_cifar10c_test_dataloaders(config: DataConfig) -> DataLoaders:
    """Build test-only dataloader for CIFAR-10-C."""
    eval_transform = build_eval_transform(config)

    test_dataset = CIFAR10CDataset(
        root=config.cifar10c_dir,
        corruption_name=config.corruption_name,
        severity=config.corruption_severity,
        transform=eval_transform,
    )

    test_dataset = _apply_optional_subset(
        test_dataset,
        config.max_eval_samples,
        config.seed + 3,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return DataLoaders(
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        classes=CIFAR10_CLASSES,
    )