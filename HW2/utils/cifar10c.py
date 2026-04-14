from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CIFAR10C_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


class CIFAR10CDataset(Dataset):
    """CIFAR-10-C test dataset for a single corruption and severity."""

    def __init__(
        self,
        root: str,
        corruption_name: str,
        severity: int,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.corruption_name = corruption_name
        self.severity = severity
        self.transform = transform

        if corruption_name not in CIFAR10C_CORRUPTIONS:
            raise ValueError(
                f"Unknown corruption '{corruption_name}'. "
                f"Supported corruptions: {CIFAR10C_CORRUPTIONS}"
            )

        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("severity must be one of {1, 2, 3, 4, 5}")

        data_path = self.root / f"{corruption_name}.npy"
        labels_path = self.root / "labels.npy"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Could not find corruption file: {data_path}"
            )

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Could not find labels file: {labels_path}"
            )

        all_images = np.load(data_path)
        all_labels = np.load(labels_path)

        start = (severity - 1) * 10000
        end = severity * 10000

        self.images = all_images[start:end]
        self.labels = all_labels[start:end].astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        label = int(self.labels[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, label