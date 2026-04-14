from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms

from parameters import DataConfig


_AFFINE = Image.Transform.AFFINE if hasattr(Image, "Transform") else Image.AFFINE
_BILINEAR = (
    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
)


def _target_size(config: DataConfig) -> int:
    return config.resize_to if config.use_imagenet_size else config.image_size


def _int_parameter(level: float, maxval: int) -> int:
    return int(level * maxval / 10)


def _float_parameter(level: float, maxval: float) -> float:
    return float(level) * maxval / 10.0


def _sample_level(severity: int) -> float:
    return np.random.uniform(low=0.1, high=float(severity))


def _randomly_negate(value: float) -> float:
    return -value if random.random() > 0.5 else value


def _autocontrast(image: Image.Image, severity: int) -> Image.Image:
    del severity
    return ImageOps.autocontrast(image)


def _equalize(image: Image.Image, severity: int) -> Image.Image:
    del severity
    return ImageOps.equalize(image)


def _posterize(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    bits = max(1, 4 - _int_parameter(level, 4))
    return ImageOps.posterize(image, bits)


def _solarize(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    threshold = 256 - _int_parameter(level, 256)
    threshold = max(0, min(255, threshold))
    return ImageOps.solarize(image, threshold)


def _rotate(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    degrees = _int_parameter(level, 30)
    degrees = _randomly_negate(float(degrees))
    return image.rotate(degrees, resample=_BILINEAR)


def _shear_x(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    shear = _float_parameter(level, 0.3)
    shear = _randomly_negate(shear)
    return image.transform(
        image.size,
        _AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=_BILINEAR,
    )


def _shear_y(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    shear = _float_parameter(level, 0.3)
    shear = _randomly_negate(shear)
    return image.transform(
        image.size,
        _AFFINE,
        (1, 0, 0, shear, 1, 0),
        resample=_BILINEAR,
    )


def _translate_x(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    shift = _float_parameter(level, image.size[0] / 3.0)
    shift = _randomly_negate(shift)
    return image.transform(
        image.size,
        _AFFINE,
        (1, 0, shift, 0, 1, 0),
        resample=_BILINEAR,
    )


def _translate_y(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    shift = _float_parameter(level, image.size[1] / 3.0)
    shift = _randomly_negate(shift)
    return image.transform(
        image.size,
        _AFFINE,
        (1, 0, 0, 0, 1, shift),
        resample=_BILINEAR,
    )


def _color(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    factor = 0.1 + _float_parameter(level, 1.8)
    return ImageEnhance.Color(image).enhance(factor)


def _contrast(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    factor = 0.1 + _float_parameter(level, 1.8)
    return ImageEnhance.Contrast(image).enhance(factor)


def _brightness(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    factor = 0.1 + _float_parameter(level, 1.8)
    return ImageEnhance.Brightness(image).enhance(factor)


def _sharpness(image: Image.Image, severity: int) -> Image.Image:
    level = _sample_level(severity)
    factor = 0.1 + _float_parameter(level, 1.8)
    return ImageEnhance.Sharpness(image).enhance(factor)


AUGMENTATION_OPS: tuple[Callable[[Image.Image, int], Image.Image], ...] = (
    _autocontrast,
    _equalize,
    _posterize,
    _solarize,
    _rotate,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
    _color,
    _contrast,
    _brightness,
    _sharpness,
)


class AugMixTransform:
    """
    AugMix transform for training.
    Applies random crop/flip first, then mixes several augmentation chains,
    and finally converts the result to a normalized tensor.
    """

    def __init__(
        self,
        pre_augment: transforms.Compose,
        post_augment: transforms.Compose,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        severity: int = 3,
    ) -> None:
        self.pre_augment = pre_augment
        self.post_augment = post_augment
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.severity = severity

    def _apply_single_chain(self, image: Image.Image) -> Image.Image:
        image_aug = image.copy()
        depth = self.depth if self.depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):
            op = random.choice(AUGMENTATION_OPS)
            image_aug = op(image_aug, self.severity)

        return image_aug

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self.pre_augment(image)

        clean_tensor = self.post_augment(image)

        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = float(np.random.beta(self.alpha, self.alpha))

        mix = torch.zeros_like(clean_tensor)

        for i in range(self.width):
            image_aug = self._apply_single_chain(image)
            aug_tensor = self.post_augment(image_aug)
            mix += ws[i] * aug_tensor

        mixed = (1.0 - m) * clean_tensor + m * mix
        return mixed


def _build_spatial_train_transform(config: DataConfig) -> transforms.Compose:
    target_size = _target_size(config)

    transform_list: list[Callable] = []

    if config.use_imagenet_size:
        transform_list.append(transforms.Resize((target_size, target_size)))

    transform_list.extend(
        [
            transforms.RandomCrop(target_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )

    return transforms.Compose(transform_list)


def _build_post_tensor_transform(config: DataConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )


def build_clean_train_transform(config: DataConfig) -> transforms.Compose:
    """Standard clean training transform."""
    spatial_transform = _build_spatial_train_transform(config)
    post_tensor_transform = _build_post_tensor_transform(config)

    return transforms.Compose(
        [
            spatial_transform,
            post_tensor_transform,
        ]
    )


def build_clean_eval_transform(config: DataConfig) -> transforms.Compose:
    """Standard evaluation transform for clean and CIFAR-10-C."""
    target_size = _target_size(config)

    transform_list: list[Callable] = []

    if config.use_imagenet_size:
        transform_list.append(transforms.Resize((target_size, target_size)))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )

    return transforms.Compose(transform_list)


def build_augmix_train_transform(config: DataConfig) -> AugMixTransform:
    """
    Real AugMix training transform.
    For CPU-friendliness, severity is fixed to a moderate value (3).
    """
    spatial_transform = _build_spatial_train_transform(config)
    post_tensor_transform = _build_post_tensor_transform(config)

    return AugMixTransform(
        pre_augment=spatial_transform,
        post_augment=post_tensor_transform,
        width=config.augmix_width,
        depth=config.augmix_depth,
        alpha=config.augmix_alpha,
        severity=3,
    )


def build_train_transform(config: DataConfig):
    """Dispatch training transform based on use_augmix flag."""
    if config.use_augmix:
        return build_augmix_train_transform(config)
    return build_clean_train_transform(config)


def build_eval_transform(config: DataConfig) -> transforms.Compose:
    """Evaluation transform shared by clean and corrupted test sets."""
    return build_clean_eval_transform(config)