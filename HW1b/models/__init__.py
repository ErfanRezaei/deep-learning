from __future__ import annotations

from torch import nn

from .mobilenet_student import MobileNetStudent
from .resnet_cifar import resnet18_cifar
from .simple_cnn import SimpleCNN
from .transfer_learning import TransferResNet18


def build_model(
    model_name: str,
    num_classes: int,
    use_pretrained: bool = False,
    freeze_early_layers: bool = False,
) -> nn.Module:

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "resnet_cifar":
        return resnet18_cifar(num_classes=num_classes)

    if model_name == "transfer_resnet18_resize":
        return TransferResNet18(
            mode="resize_freeze",
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            freeze_early_layers=freeze_early_layers,
        )

    if model_name == "transfer_resnet18_cifar":
        return TransferResNet18(
            mode="cifar_finetune",
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            freeze_early_layers=False,
        )

    if model_name == "mobilenet_student":
        return MobileNetStudent(
            num_classes=num_classes,
            use_pretrained=use_pretrained,
        )

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        "Currently implemented: "
        "['simple_cnn', 'resnet_cifar', 'transfer_resnet18_resize', "
        "'transfer_resnet18_cifar', 'mobilenet_student']"
    )