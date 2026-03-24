from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import torch

from models import build_model
from parameters import (
    DataConfig,
    DistillationConfig,
    ExperimentConfig,
    OptimizationConfig,
    RuntimeConfig,
)
from train import build_optimizer, build_scheduler, train_model
from utils.data import DataLoaders, build_cifar10_dataloaders
from utils.losses import build_classification_criterion


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="HW2 - Deep Learning Experiments")

    parser.add_argument(
        "--mode",
        type=str,
        default="sanity_check",
        choices=["sanity_check", "train"],
        help="Whether to run only sanity checks or actual training.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="simple_cnn",
        choices=[
            "transfer_learning",
            "simple_cnn",
            "resnet_scratch",
            "distill_simple_cnn",
            "mobilenet_student",
        ],
        help="Which experiment family to run.",
    )
    parser.add_argument("--model-name", type=str, default="simple_cnn")
    parser.add_argument("--experiment-name", type=str, default="debug_run")

    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--scheduler-name", type=str, default="none", choices=["none", "step", "cosine"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--use-imagenet-size", action="store_true")
    parser.add_argument("--resize-to", type=int, default=224)
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--freeze-early-layers", action="store_true")

    parser.add_argument("--use-distillation", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--teacher-model-name", type=str, default="resnet_cifar")

    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:

    data_config = DataConfig(
        data_dir=args.data_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_imagenet_size=args.use_imagenet_size,
        resize_to=args.resize_to,
    )

    optim_config = OptimizationConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer_name,
        scheduler_name=args.scheduler_name,
        label_smoothing=args.label_smoothing,
    )

    distill_config = DistillationConfig(
        use_distillation=args.use_distillation,
        alpha=args.alpha,
        temperature=args.temperature,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_model_name=args.teacher_model_name,
    )

    runtime_config = RuntimeConfig(
        task=args.task,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        device=args.device,
        use_pretrained=args.use_pretrained,
        freeze_early_layers=args.freeze_early_layers,
    )

    return ExperimentConfig(
        data=data_config,
        optim=optim_config,
        distill=distill_config,
        runtime=runtime_config,
    )


def prepare_directories(config: ExperimentConfig) -> None:

    Path(config.runtime.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.runtime.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.runtime.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(config.runtime.tables_dir).mkdir(parents=True, exist_ok=True)


def resolve_device(device_name: str) -> torch.device:

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total_params, trainable_params


def run_data_sanity_check(config: ExperimentConfig) -> DataLoaders:

    dataloaders = build_cifar10_dataloaders(config.data)

    train_batch = next(iter(dataloaders.train_loader))
    images, labels = train_batch

    print("\n===== DATA SANITY CHECK =====")
    print(f"Classes: {dataloaders.classes}")
    print(f"Train batches: {len(dataloaders.train_loader)}")
    print(f"Val batches: {len(dataloaders.val_loader)}")
    print(f"Test batches: {len(dataloaders.test_loader)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print("=============================\n")

    return dataloaders


def run_model_sanity_check(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:

    model = build_model(
    model_name=config.runtime.model_name,
    num_classes=config.data.num_classes,
    use_pretrained=config.runtime.use_pretrained,
    freeze_early_layers=config.runtime.freeze_early_layers,
).to(device)

    images, _ = next(iter(dataloaders.train_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)

    total_params, trainable_params = count_parameters(model)

    print("===== MODEL SANITY CHECK =====")
    print(f"Model name: {config.runtime.model_name}")
    print(f"Input batch shape: {images.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("==============================\n")


def run_training_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:

    model = build_model(
    model_name=config.runtime.model_name,
    num_classes=config.data.num_classes,
    use_pretrained=config.runtime.use_pretrained,
    freeze_early_layers=config.runtime.freeze_early_layers,
).to(device)

    criterion = build_classification_criterion(
        label_smoothing=config.optim.label_smoothing
    )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history, test_metrics, checkpoint_path = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print("==============================\n")


def main() -> None:

    args = parse_args()
    config = build_config(args)

    prepare_directories(config)
    device = resolve_device(config.runtime.device)

    if device.type == "cpu":
        config.data.pin_memory = False

    print("\n===== EXPERIMENT CONFIG =====")
    pprint(asdict(config))
    print(f"Resolved device: {device}")
    print("=============================\n")

    dataloaders = run_data_sanity_check(config)
    run_model_sanity_check(config, dataloaders, device)

    if args.mode == "sanity_check":
        print("Project skeleton + selected model are ready.")
        print("Next step: run actual training with --mode train.")
        return

    run_training_pipeline(config, dataloaders, device)


if __name__ == "__main__":
    main()