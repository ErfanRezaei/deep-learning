from __future__ import annotations

import torch
from torch import nn


def _load_resnet18(use_pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        model = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18

        model = resnet18(pretrained=use_pretrained)

    return model


def _replace_conv1_with_cifar_stem(model: nn.Module) -> None:

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )


    with torch.no_grad():
        if old_conv.weight.shape[-1] == 7:
            new_conv.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])

    model.conv1 = new_conv
    model.maxpool = nn.Identity()


class TransferResNet18(nn.Module):


    def __init__(
        self,
        mode: str,
        num_classes: int = 10,
        use_pretrained: bool = True,
        freeze_early_layers: bool = False,
    ) -> None:
        super().__init__()

        if mode not in {"resize_freeze", "cifar_finetune"}:
            raise ValueError(f"Unsupported transfer learning mode: {mode}")

        self.mode = mode
        self.freeze_early_layers = freeze_early_layers and mode == "resize_freeze"

        self.model = _load_resnet18(use_pretrained=use_pretrained)

        if self.mode == "cifar_finetune":
            _replace_conv1_with_cifar_stem(self.model)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if self.freeze_early_layers:
            self._freeze_early_backbone()

    def _freeze_early_backbone(self) -> None:
        frozen_modules = [
            self.model.conv1,
            self.model.bn1,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
        ]

        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "TransferResNet18":

        super().train(mode)

        if mode and self.freeze_early_layers:
            self.model.conv1.eval()
            self.model.bn1.eval()
            self.model.layer1.eval()
            self.model.layer2.eval()
            self.model.layer3.eval()

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)