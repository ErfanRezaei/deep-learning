from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from parameters import DataConfig


@dataclass
class DataLoaders:

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: tuple[str, ...]


def _build_train_transform(config: DataConfig) -> transforms.Compose:

    transform_list = []

    target_size = config.resize_to if config.use_imagenet_size else config.image_size

    if config.use_imagenet_size:
        transform_list.append(transforms.Resize((target_size, target_size)))

    transform_list.extend(
        [
            transforms.RandomCrop(target_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )

    return transforms.Compose(transform_list)


def _build_eval_transform(config: DataConfig) -> transforms.Compose:

    transform_list = []

    target_size = config.resize_to if config.use_imagenet_size else config.image_size

    if config.use_imagenet_size:
        transform_list.append(transforms.Resize((target_size, target_size)))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )

    return transforms.Compose(transform_list)


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


def build_cifar10_dataloaders(config: DataConfig) -> DataLoaders:

    train_transform = _build_train_transform(config)
    eval_transform = _build_eval_transform(config)

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

    return DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=tuple(train_dataset_full.classes),
    )