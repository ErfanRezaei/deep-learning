from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from parameters import AttackConfig, DataConfig, VisualizationConfig
from utils.attacks import denormalize_images, pgd_attack


def _resolve_target_layer(model: nn.Module, target_layer_name: str) -> nn.Module:
    """
    Resolve a target layer by name.
    Tries:
    1) exact named_modules() match
    2) direct attribute on model
    3) attribute under model.backbone
    4) suffix match in named_modules()
    """
    named_modules = dict(model.named_modules())

    if target_layer_name in named_modules:
        return named_modules[target_layer_name]

    if hasattr(model, target_layer_name):
        return getattr(model, target_layer_name)

    if hasattr(model, "backbone"):
        backbone = getattr(model, "backbone")
        if hasattr(backbone, target_layer_name):
            return getattr(backbone, target_layer_name)

        backbone_named = dict(backbone.named_modules())
        if target_layer_name in backbone_named:
            return backbone_named[target_layer_name]

    for name, module in named_modules.items():
        if name.endswith(target_layer_name):
            return module

    available = list(named_modules.keys())[:50]
    raise ValueError(
        f"Could not resolve target layer '{target_layer_name}'. "
        f"Some available module names: {available}"
    )


class GradCAM:
    """Simple Grad-CAM implementation."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.backward_handle = self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        else:
            self.backward_handle = self.target_layer.register_backward_hook(
                self._backward_hook
            )

    def _forward_hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        del module, inputs
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        del module, grad_input
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Grad-CAM heatmap for the given input.
        Returns:
            cams: [B, 1, H, W] in [0, 1]
            logits: [B, num_classes]
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        selected_scores = logits[torch.arange(logits.size(0), device=logits.device), class_idx]
        score = selected_scores.sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = (weights * self.activations).sum(dim=1, keepdim=True)
        cams = F.relu(cams)

        cams = F.interpolate(
            cams,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cams_flat = cams.view(cams.size(0), -1)
        cam_min = cams_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        cam_max = cams_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        cams = (cams - cam_min) / (cam_max - cam_min + 1e-8)

        return cams.detach(), logits.detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def _tensor_to_display_image(
    image_norm: torch.Tensor,
    data_config: DataConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Convert one normalized CHW tensor to HWC image in [0, 1].
    """
    image = denormalize_images(image_norm.unsqueeze(0).to(device), data_config, device)[0]
    image = torch.clamp(image, 0.0, 1.0)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return image


def _cam_to_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a CAM heatmap on top of an RGB image.
    """
    cmap = plt.colormaps["jet"]
    heatmap = cmap(cam)[..., :3]
    overlay = (1.0 - alpha) * image + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def collect_misclassified_adversarial_examples(
    model: nn.Module,
    dataloader: DataLoader,
    attack_config: AttackConfig,
    data_config: DataConfig,
    device: torch.device,
    num_examples: int,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Find samples that are correctly classified on clean input but misclassified after PGD.
    """
    model.eval()

    collected: list[dict[str, Any]] = []
    total_seen = 0

    for images, labels in dataloader:
        if max_samples is not None and total_seen >= max_samples:
            break

        if max_samples is not None:
            remaining = max_samples - total_seen
            if remaining <= 0:
                break
            images = images[:remaining]
            labels = labels[:remaining]

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)

        adv_images = pgd_attack(
            model=model,
            images_norm=images,
            labels=labels,
            attack_config=attack_config,
            data_config=data_config,
            device=device,
        )

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_preds = adv_logits.argmax(dim=1)

        failure_mask = (clean_preds == labels) & (adv_preds != labels)
        failure_indices = torch.where(failure_mask)[0]

        for idx in failure_indices.tolist():
            collected.append(
                {
                    "clean_image": images[idx].detach().cpu(),
                    "adv_image": adv_images[idx].detach().cpu(),
                    "label": int(labels[idx].item()),
                    "clean_pred": int(clean_preds[idx].item()),
                    "adv_pred": int(adv_preds[idx].item()),
                }
            )

            if len(collected) >= num_examples:
                return collected

        total_seen += labels.size(0)

    return collected


def save_gradcam_comparison_figure(
    clean_image: np.ndarray,
    clean_overlay: np.ndarray,
    adv_image: np.ndarray,
    adv_overlay: np.ndarray,
    true_label_name: str,
    clean_pred_name: str,
    adv_pred_name: str,
    output_path: Path,
) -> None:
    """Save a 1x4 comparison figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(clean_image)
    axes[0].set_title(f"Clean Image\nTrue: {true_label_name}")
    axes[0].axis("off")

    axes[1].imshow(clean_overlay)
    axes[1].set_title(f"Clean Grad-CAM\nPred: {clean_pred_name}")
    axes[1].axis("off")

    axes[2].imshow(adv_image)
    axes[2].set_title(f"Adversarial Image\nTrue: {true_label_name}")
    axes[2].axis("off")

    axes[3].imshow(adv_overlay)
    axes[3].set_title(f"Adv Grad-CAM\nPred: {adv_pred_name}")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_gradcam_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    attack_config: AttackConfig,
    data_config: DataConfig,
    vis_config: VisualizationConfig,
    device: torch.device,
    class_names: tuple[str, ...],
    output_dir: str,
    output_prefix: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Find 1-2 adversarial failure samples and save clean/adv Grad-CAM figures.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    examples = collect_misclassified_adversarial_examples(
        model=model,
        dataloader=dataloader,
        attack_config=attack_config,
        data_config=data_config,
        device=device,
        num_examples=vis_config.num_gradcam_samples,
        max_samples=max_samples,
    )

    metadata: dict[str, Any] = {
        "num_requested_examples": vis_config.num_gradcam_samples,
        "num_found_examples": len(examples),
        "target_layer": vis_config.gradcam_target_layer,
        "saved_figures": [],
        "examples": [],
    }

    if len(examples) == 0:
        metadata_path = output_dir_path / f"{output_prefix}_gradcam_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print("No suitable clean->adversarial failure examples were found.")
        print(f"Metadata saved to: {metadata_path}")
        return metadata

    target_layer = _resolve_target_layer(model, vis_config.gradcam_target_layer)
    gradcam = GradCAM(model=model, target_layer=target_layer)

    try:
        for idx, example in enumerate(examples, start=1):
            clean_input = example["clean_image"].unsqueeze(0).to(device)
            adv_input = example["adv_image"].unsqueeze(0).to(device)

            clean_target = torch.tensor([example["clean_pred"]], device=device)
            adv_target = torch.tensor([example["adv_pred"]], device=device)

            clean_cam, _ = gradcam.generate(clean_input, class_idx=clean_target)
            adv_cam, _ = gradcam.generate(adv_input, class_idx=adv_target)

            clean_cam_np = clean_cam[0, 0].detach().cpu().numpy()
            adv_cam_np = adv_cam[0, 0].detach().cpu().numpy()

            clean_image_np = _tensor_to_display_image(example["clean_image"], data_config, device)
            adv_image_np = _tensor_to_display_image(example["adv_image"], data_config, device)

            clean_overlay = _cam_to_overlay(clean_image_np, clean_cam_np)
            adv_overlay = _cam_to_overlay(adv_image_np, adv_cam_np)

            true_label_name = class_names[example["label"]]
            clean_pred_name = class_names[example["clean_pred"]]
            adv_pred_name = class_names[example["adv_pred"]]

            figure_path = output_dir_path / f"{output_prefix}_sample{idx:02d}.png"

            save_gradcam_comparison_figure(
                clean_image=clean_image_np,
                clean_overlay=clean_overlay,
                adv_image=adv_image_np,
                adv_overlay=adv_overlay,
                true_label_name=true_label_name,
                clean_pred_name=clean_pred_name,
                adv_pred_name=adv_pred_name,
                output_path=figure_path,
            )

            metadata["saved_figures"].append(str(figure_path))
            metadata["examples"].append(
                {
                    "sample_index": idx,
                    "true_label": true_label_name,
                    "clean_prediction": clean_pred_name,
                    "adversarial_prediction": adv_pred_name,
                    "figure_path": str(figure_path),
                }
            )

            print(f"Saved Grad-CAM figure: {figure_path}")

    finally:
        gradcam.close()

    metadata_path = output_dir_path / f"{output_prefix}_gradcam_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Grad-CAM metadata saved to: {metadata_path}")
    return metadata