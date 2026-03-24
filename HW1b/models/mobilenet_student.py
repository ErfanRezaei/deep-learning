from __future__ import annotations

from torch import nn


def _load_mobilenet_v3_small(use_pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        model = mobilenet_v3_small(weights=weights)
    except Exception:
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(pretrained=use_pretrained)

    return model


class MobileNetStudent(nn.Module):

    def __init__(self, num_classes: int = 10, use_pretrained: bool = False) -> None:
        super().__init__()

        self.model = _load_mobilenet_v3_small(use_pretrained=use_pretrained)

        first_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)