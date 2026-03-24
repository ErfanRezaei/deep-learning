from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


TaskType = Literal[
    "transfer_learning",
    "simple_cnn",
    "resnet_scratch",
    "distill_simple_cnn",
    "mobilenet_student",
]


@dataclass
class DataConfig:

    data_dir: str = "data"
    num_classes: int = 10
    train_batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    pin_memory: bool = True
    download: bool = True
    seed: int = 42

    image_size: int = 32
    resize_to: int = 224
    use_imagenet_size: bool = False
    val_ratio: float = 0.1

    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


@dataclass
class OptimizationConfig:

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    optimizer_name: Literal["sgd", "adam"] = "adam"
    scheduler_name: Literal["none", "step", "cosine"] = "none"
    step_size: int = 10
    gamma: float = 0.1
    label_smoothing: float = 0.0


@dataclass
class DistillationConfig:

    use_distillation: bool = False
    alpha: float = 0.7
    temperature: float = 4.0
    teacher_checkpoint: Optional[str] = None
    teacher_model_name: str = "resnet_cifar"


@dataclass
class RuntimeConfig:

    task: TaskType = "simple_cnn"
    model_name: str = "simple_cnn"
    experiment_name: str = "simple_cnn_baseline"
    device: str = "cuda"
    use_pretrained: bool = False
    freeze_early_layers: bool = False

    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    logs_dir: str = "results/logs"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"


@dataclass
class ExperimentConfig:

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)