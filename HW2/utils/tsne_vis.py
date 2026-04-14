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
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader

from parameters import AttackConfig, DataConfig, VisualizationConfig
from utils.attacks import pgd_attack


def _resolve_feature_layer(model: nn.Module, target_layer_name: str) -> nn.Module:
    """Resolve a feature layer by name."""
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


class FeatureHook:
    """Capture activations from a chosen layer."""

    def __init__(self, layer: nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        del module, inputs
        self.activations = output.detach()

    def close(self) -> None:
        self.handle.remove()


def _pool_features(features: torch.Tensor) -> torch.Tensor:
    """
    Convert hooked activations into [B, D] feature vectors.
    """
    if features.ndim == 4:
        features = F.adaptive_avg_pool2d(features, output_size=1)
        features = features.view(features.size(0), -1)
        return features

    if features.ndim == 2:
        return features

    return features.view(features.size(0), -1)


def _extract_features(
    model: nn.Module,
    hook: FeatureHook,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a forward pass and return pooled hooked features + logits.
    """
    logits = model(inputs)

    if hook.activations is None:
        raise RuntimeError("Feature hook did not capture activations.")

    features = _pool_features(hook.activations)
    return features.detach(), logits.detach()


def collect_tsne_features(
    model: nn.Module,
    dataloader: DataLoader,
    attack_config: AttackConfig,
    data_config: DataConfig,
    vis_config: VisualizationConfig,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Collect clean and adversarial features for the same samples.
    """
    model.eval()

    target_layer = _resolve_feature_layer(model, vis_config.gradcam_target_layer)
    hook = FeatureHook(target_layer)

    clean_features_all: list[torch.Tensor] = []
    adv_features_all: list[torch.Tensor] = []

    true_labels_all: list[torch.Tensor] = []
    clean_preds_all: list[torch.Tensor] = []
    adv_preds_all: list[torch.Tensor] = []

    total = 0

    try:
        for images, labels in dataloader:
            if max_samples is not None and total >= max_samples:
                break

            if max_samples is not None:
                remaining = max_samples - total
                if remaining <= 0:
                    break
                images = images[:remaining]
                labels = labels[:remaining]

            images = images.to(device)
            labels = labels.to(device)

            adv_images = pgd_attack(
                model=model,
                images_norm=images,
                labels=labels,
                attack_config=attack_config,
                data_config=data_config,
                device=device,
            )

            with torch.no_grad():
                clean_features, clean_logits = _extract_features(model, hook, images)
                adv_features, adv_logits = _extract_features(model, hook, adv_images)

            clean_preds = clean_logits.argmax(dim=1)
            adv_preds = adv_logits.argmax(dim=1)

            clean_features_all.append(clean_features.cpu())
            adv_features_all.append(adv_features.cpu())
            true_labels_all.append(labels.cpu())
            clean_preds_all.append(clean_preds.cpu())
            adv_preds_all.append(adv_preds.cpu())

            total += labels.size(0)

    finally:
        hook.close()

    if total == 0:
        raise ValueError("No samples were collected for t-SNE.")

    clean_features = torch.cat(clean_features_all, dim=0).numpy()
    adv_features = torch.cat(adv_features_all, dim=0).numpy()

    true_labels = torch.cat(true_labels_all, dim=0).numpy()
    clean_preds = torch.cat(clean_preds_all, dim=0).numpy()
    adv_preds = torch.cat(adv_preds_all, dim=0).numpy()

    return {
        "clean_features": clean_features,
        "adv_features": adv_features,
        "true_labels": true_labels,
        "clean_preds": clean_preds,
        "adv_preds": adv_preds,
        "num_samples": int(total),
    }


def _compute_safe_perplexity(num_points: int) -> float:
    """
    t-SNE requires perplexity < n_samples.
    Keep it small and CPU-friendly.
    """
    if num_points <= 6:
        return max(2.0, float(num_points - 1))
    return min(20.0, max(5.0, float((num_points - 1) // 3)))


def run_tsne_analysis(
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
    Create a t-SNE plot showing clean vs adversarial sample locations.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    collected = collect_tsne_features(
        model=model,
        dataloader=dataloader,
        attack_config=attack_config,
        data_config=data_config,
        vis_config=vis_config,
        device=device,
        max_samples=max_samples or vis_config.tsne_max_samples,
    )

    clean_features = collected["clean_features"]
    adv_features = collected["adv_features"]
    true_labels = collected["true_labels"]
    clean_preds = collected["clean_preds"]
    adv_preds = collected["adv_preds"]

    all_features = np.concatenate([clean_features, adv_features], axis=0)
    domain_labels = np.array(
        ["clean"] * len(clean_features) + ["adversarial"] * len(adv_features)
    )
    repeated_true_labels = np.concatenate([true_labels, true_labels], axis=0)
    repeated_pred_labels = np.concatenate([clean_preds, adv_preds], axis=0)

    num_points = all_features.shape[0]
    perplexity = _compute_safe_perplexity(num_points)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=42,
    )
    embedding = tsne.fit_transform(all_features)

    fig, ax = plt.subplots(figsize=(8, 6))

    clean_mask = domain_labels == "clean"
    adv_mask = domain_labels == "adversarial"

    ax.scatter(
        embedding[clean_mask, 0],
        embedding[clean_mask, 1],
        s=28,
        alpha=0.75,
        marker="o",
        label="Clean",
    )

    ax.scatter(
        embedding[adv_mask, 0],
        embedding[adv_mask, 1],
        s=34,
        alpha=0.75,
        marker="x",
        label="Adversarial",
    )

    ax.set_title("t-SNE of Clean and Adversarial Features")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend()
    fig.tight_layout()

    figure_path = output_dir_path / f"{output_prefix}_tsne.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "num_clean_samples": int(len(clean_features)),
        "num_adversarial_samples": int(len(adv_features)),
        "num_total_points": int(num_points),
        "feature_layer": vis_config.gradcam_target_layer,
        "tsne_perplexity": float(perplexity),
        "figure_path": str(figure_path),
        "class_names": list(class_names),
        "num_label_changes": int((clean_preds != adv_preds).sum()),
        "num_clean_correct": int((clean_preds == true_labels).sum()),
        "num_adv_correct": int((adv_preds == true_labels).sum()),
    }

    metadata_path = output_dir_path / f"{output_prefix}_tsne_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved t-SNE figure: {figure_path}")
    print(f"Saved t-SNE metadata: {metadata_path}")

    return metadata