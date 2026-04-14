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

ModeType = Literal[
    "sanity_check",
    "train",
    "eval_cifar10c",
    "eval_pgd",
    "analyze_adv",
    "train_kd_augmix",
    "transfer_attack",
]

AttackNormType = Literal["linf", "l2"]


@dataclass
class DataConfig:
    data_dir: str = "data"
    cifar10c_dir: str = "data/CIFAR-10-C"

    num_classes: int = 10
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True
    seed: int = 42

    image_size: int = 32
    resize_to: int = 224
    use_imagenet_size: bool = False
    val_ratio: float = 0.1

    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    # HW2 additions
    use_augmix: bool = False
    augmix_width: int = 3
    augmix_depth: int = -1
    augmix_alpha: float = 1.0

    corruption_name: str = "gaussian_noise"
    corruption_severity: int = 1

    # CPU-friendly debug / quick runs
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 256


@dataclass
class OptimizationConfig:
    epochs: int = 5
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
    student_checkpoint: Optional[str] = None


@dataclass
class AttackConfig:
    enabled: bool = False
    norm: AttackNormType = "linf"
    epsilon: float = 4.0 / 255.0
    step_size: float = 1.0 / 255.0
    steps: int = 20
    random_start: bool = True
    max_adv_samples: int = 256


@dataclass
class VisualizationConfig:
    gradcam_target_layer: str = "layer4"
    num_gradcam_samples: int = 2
    tsne_max_samples: int = 256


@dataclass
class RuntimeConfig:
    mode: ModeType = "sanity_check"
    task: TaskType = "transfer_learning"
    model_name: str = "transfer_resnet18_cifar"
    experiment_name: str = "hw2_debug"

    device: str = "cpu"
    use_pretrained: bool = True
    freeze_early_layers: bool = False

    checkpoint_path: Optional[str] = None

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
    attack: AttackConfig = field(default_factory=AttackConfig)
    vis: VisualizationConfig = field(default_factory=VisualizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)