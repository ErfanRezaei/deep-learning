from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import torch

from models import build_model
from parameters import (
    AttackConfig,
    DataConfig,
    DistillationConfig,
    ExperimentConfig,
    OptimizationConfig,
    RuntimeConfig,
    VisualizationConfig,
)
from test import evaluate_model, load_checkpoint, save_metrics_json
from train import build_optimizer, build_scheduler, train_model
from utils.attacks import evaluate_transfer_attack, evaluate_under_pgd
from utils.cifar10c import CIFAR10C_CORRUPTIONS
from utils.data import (
    DataLoaders,
    build_cifar10_dataloaders,
    build_cifar10c_test_dataloaders,
)
from utils.gradcam import run_gradcam_analysis
from utils.losses import build_classification_criterion
from utils.tsne_vis import run_tsne_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW2 - Deep Learning Experiments")

    parser.add_argument(
        "--mode",
        type=str,
        default="sanity_check",
        choices=[
            "sanity_check",
            "train",
            "train_kd_augmix",
            "eval_clean",
            "eval_cifar10c",
            "eval_cifar10c_full",
            "eval_pgd",
            "analyze_adv",
            "visualize_tsne",
            "transfer_attack",
        ],
        help="Execution mode for HW2 pipeline.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="transfer_learning",
        choices=[
            "transfer_learning",
            "simple_cnn",
            "resnet_scratch",
            "distill_simple_cnn",
            "mobilenet_student",
        ],
        help="Which experiment family to run.",
    )

    parser.add_argument("--model-name", type=str, default="transfer_resnet18_cifar")
    parser.add_argument("--experiment-name", type=str, default="hw2_debug")

    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="none",
        choices=["none", "step", "cosine"],
    )
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
    parser.add_argument("--cifar10c-dir", type=str, default="data/CIFAR-10-C")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--use-augmix", action="store_true")
    parser.add_argument("--corruption-name", type=str, default="gaussian_noise")
    parser.add_argument(
        "--corruption-severity",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=256)

    parser.add_argument("--attack-norm", type=str, default="linf", choices=["linf", "l2"])
    parser.add_argument("--attack-epsilon", type=float, default=4.0 / 255.0)
    parser.add_argument("--attack-step-size", type=float, default=1.0 / 255.0)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--random-start", action="store_true")

    parser.add_argument("--checkpoint-path", type=str, default=None)

    parser.add_argument("--gradcam-target-layer", type=str, default="layer4")
    parser.add_argument("--num-gradcam-samples", type=int, default=2)
    parser.add_argument("--tsne-max-samples", type=int, default=256)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    data_config = DataConfig(
        data_dir=args.data_dir,
        cifar10c_dir=args.cifar10c_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_imagenet_size=args.use_imagenet_size,
        resize_to=args.resize_to,
        use_augmix=args.use_augmix,
        corruption_name=args.corruption_name,
        corruption_severity=args.corruption_severity,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
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

    attack_config = AttackConfig(
        enabled=args.mode in {"eval_pgd", "analyze_adv", "visualize_tsne", "transfer_attack"},
        norm=args.attack_norm,
        epsilon=args.attack_epsilon,
        step_size=args.attack_step_size,
        steps=args.attack_steps,
        random_start=args.random_start,
        max_adv_samples=args.max_eval_samples,
    )

    vis_config = VisualizationConfig(
        gradcam_target_layer=args.gradcam_target_layer,
        num_gradcam_samples=args.num_gradcam_samples,
        tsne_max_samples=args.tsne_max_samples,
    )

    runtime_config = RuntimeConfig(
        mode=args.mode,
        task=args.task,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        device=args.device,
        use_pretrained=args.use_pretrained,
        freeze_early_layers=args.freeze_early_layers,
        checkpoint_path=args.checkpoint_path,
    )

    return ExperimentConfig(
        data=data_config,
        optim=optim_config,
        distill=distill_config,
        attack=attack_config,
        vis=vis_config,
        runtime=runtime_config,
    )


def prepare_directories(config: ExperimentConfig) -> None:
    Path(config.runtime.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.runtime.results_dir).mkdir(parents=True, exist_ok=True)
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
    if config.runtime.mode in {"eval_cifar10c", "eval_cifar10c_full"}:
        dataloaders = build_cifar10c_test_dataloaders(config.data)
        test_batch = next(iter(dataloaders.test_loader))
        images, labels = test_batch

        print("\n===== CIFAR-10-C DATA SANITY CHECK =====")
        print(f"Corruption: {config.data.corruption_name}")
        print(f"Severity: {config.data.corruption_severity}")
        print(f"Test batches: {len(dataloaders.test_loader)}")
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print("=======================================\n")
        return dataloaders

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

    batch_loader = dataloaders.train_loader if dataloaders.train_loader is not None else dataloaders.test_loader
    images, _ = next(iter(batch_loader))
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
    if dataloaders.train_loader is None or dataloaders.val_loader is None:
        raise ValueError("Training mode requires train_loader and val_loader.")

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


def run_kd_training_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if not config.distill.use_distillation:
        raise ValueError("KD training requires --use-distillation.")
    if config.distill.teacher_checkpoint is None:
        raise ValueError("KD training requires --teacher-checkpoint.")

    print("Running KD training pipeline...")
    print(f"Teacher model: {config.distill.teacher_model_name}")
    print(f"Teacher checkpoint: {config.distill.teacher_checkpoint}")

    run_training_pipeline(config, dataloaders, device)


def run_clean_evaluation_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("Clean evaluation requires --checkpoint-path.")

    model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)

    model = load_checkpoint(model, config.runtime.checkpoint_path, device)

    criterion = build_classification_criterion(
        label_smoothing=config.optim.label_smoothing
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders.test_loader,
        criterion=criterion,
        device=device,
    )

    result = {
        "experiment_name": config.runtime.experiment_name,
        "mode": "eval_clean",
        "model_name": config.runtime.model_name,
        "checkpoint_path": config.runtime.checkpoint_path,
        "metrics": metrics,
    }

    output_path = Path(config.runtime.tables_dir) / f"{config.runtime.experiment_name}_clean.json"
    save_metrics_json(result, str(output_path))

    print("\n===== CLEAN EVALUATION RESULTS =====")
    print(f"Checkpoint: {config.runtime.checkpoint_path}")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print("====================================\n")


def run_cifar10c_evaluation_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("CIFAR-10-C evaluation requires --checkpoint-path.")

    model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)

    model = load_checkpoint(model, config.runtime.checkpoint_path, device)

    criterion = build_classification_criterion(
        label_smoothing=config.optim.label_smoothing
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders.test_loader,
        criterion=criterion,
        device=device,
    )

    result = {
        "experiment_name": config.runtime.experiment_name,
        "mode": "eval_cifar10c",
        "model_name": config.runtime.model_name,
        "checkpoint_path": config.runtime.checkpoint_path,
        "corruption_name": config.data.corruption_name,
        "corruption_severity": config.data.corruption_severity,
        "metrics": metrics,
    }

    filename = (
        f"{config.runtime.experiment_name}_"
        f"{config.data.corruption_name}_sev{config.data.corruption_severity}.json"
    )
    output_path = Path(config.runtime.tables_dir) / filename
    save_metrics_json(result, str(output_path))

    print("\n===== CIFAR-10-C EVALUATION RESULTS =====")
    print(f"Checkpoint: {config.runtime.checkpoint_path}")
    print(f"Corruption: {config.data.corruption_name}")
    print(f"Severity: {config.data.corruption_severity}")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    print("=========================================\n")


def run_cifar10c_full_sweep(
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("Full CIFAR-10-C sweep requires --checkpoint-path.")

    model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)

    model = load_checkpoint(model, config.runtime.checkpoint_path, device)

    criterion = build_classification_criterion(
        label_smoothing=config.optim.label_smoothing
    )

    all_results: list[dict[str, object]] = []

    for corruption_name in CIFAR10C_CORRUPTIONS:
        for severity in [1, 2, 3, 4, 5]:
            config.data.corruption_name = corruption_name
            config.data.corruption_severity = severity

            dataloaders = build_cifar10c_test_dataloaders(config.data)

            metrics = evaluate_model(
                model=model,
                dataloader=dataloaders.test_loader,
                criterion=criterion,
                device=device,
            )

            row = {
                "corruption_name": corruption_name,
                "severity": severity,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
            }
            all_results.append(row)

            print(
                f"[CIFAR-10-C] {corruption_name:20s} | "
                f"severity={severity} | "
                f"acc={metrics['accuracy']:.2f}% | "
                f"loss={metrics['loss']:.4f}"
            )

    mean_accuracy = sum(float(row["accuracy"]) for row in all_results) / len(all_results)

    result = {
        "experiment_name": config.runtime.experiment_name,
        "mode": "eval_cifar10c_full",
        "model_name": config.runtime.model_name,
        "checkpoint_path": config.runtime.checkpoint_path,
        "mean_accuracy": mean_accuracy,
        "results": all_results,
    }

    output_path = Path(config.runtime.tables_dir) / f"{config.runtime.experiment_name}_cifar10c_full.json"
    save_metrics_json(result, str(output_path))

    print("\n===== CIFAR-10-C FULL SWEEP SUMMARY =====")
    print(f"Mean Corrupted Accuracy: {mean_accuracy:.2f}%")
    print("=========================================\n")


def run_pgd_evaluation_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("PGD evaluation requires --checkpoint-path.")

    model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)

    model = load_checkpoint(model, config.runtime.checkpoint_path, device)

    metrics = evaluate_under_pgd(
        model=model,
        dataloader=dataloaders.test_loader,
        attack_config=config.attack,
        data_config=config.data,
        device=device,
        max_samples=config.data.max_eval_samples,
    )

    result = {
        "experiment_name": config.runtime.experiment_name,
        "mode": "eval_pgd",
        "model_name": config.runtime.model_name,
        "checkpoint_path": config.runtime.checkpoint_path,
        "metrics": metrics,
    }

    safe_eps = str(config.attack.epsilon).replace(".", "p")
    filename = (
        f"{config.runtime.experiment_name}_pgd_"
        f"{config.attack.norm}_eps{safe_eps}.json"
    )
    output_path = Path(config.runtime.tables_dir) / filename
    save_metrics_json(result, str(output_path))

    print("\n===== PGD EVALUATION RESULTS =====")
    print(f"Checkpoint: {config.runtime.checkpoint_path}")
    print(f"Norm: {config.attack.norm}")
    print(f"Epsilon: {config.attack.epsilon}")
    print(f"Step size: {config.attack.step_size}")
    print(f"Steps: {config.attack.steps}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Clean Loss: {metrics['clean_loss']:.4f}")
    print(f"Clean Accuracy: {metrics['clean_accuracy']:.2f}%")
    print(f"Adversarial Loss: {metrics['adv_loss']:.4f}")
    print(f"Adversarial Accuracy: {metrics['adv_accuracy']:.2f}%")
    print(
        "Attack Success Rate on Initially Correct Samples: "
        f"{metrics['attack_success_rate_on_clean']:.2f}%"
    )
    print("==================================\n")


def run_analyze_adv_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("Adversarial analysis requires --checkpoint-path.")

    model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)

    model = load_checkpoint(model, config.runtime.checkpoint_path, device)

    metadata = run_gradcam_analysis(
        model=model,
        dataloader=dataloaders.test_loader,
        attack_config=config.attack,
        data_config=config.data,
        vis_config=config.vis,
        device=device,
        class_names=dataloaders.classes,
        output_dir=config.runtime.figures_dir,
        output_prefix=config.runtime.experiment_name,
        max_samples=config.data.max_eval_samples,
    )

    metadata_path = Path(config.runtime.tables_dir) / f"{config.runtime.experiment_name}_gradcam_summary.json"
    save_metrics_json(metadata, str(metadata_path))

    print("\n===== ADVERSARIAL ANALYSIS RESULTS =====")
    print(f"Checkpoint: {config.runtime.checkpoint_path}")
    print(f"Requested examples: {config.vis.num_gradcam_samples}")
    print(f"Found examples: {metadata['num_found_examples']}")
    print(f"Target layer: {config.vis.gradcam_target_layer}")
    print(f"Figures directory: {config.runtime.figures_dir}")
    print("========================================\n")


def run_tsne_visualization_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("t-SNE visualization requires --checkpoint-path.")

    model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)

    model = load_checkpoint(model, config.runtime.checkpoint_path, device)

    metadata = run_tsne_analysis(
        model=model,
        dataloader=dataloaders.test_loader,
        attack_config=config.attack,
        data_config=config.data,
        vis_config=config.vis,
        device=device,
        class_names=dataloaders.classes,
        output_dir=config.runtime.figures_dir,
        output_prefix=config.runtime.experiment_name,
        max_samples=config.vis.tsne_max_samples,
    )

    metadata_path = Path(config.runtime.tables_dir) / f"{config.runtime.experiment_name}_tsne_summary.json"
    save_metrics_json(metadata, str(metadata_path))

    print("\n===== t-SNE VISUALIZATION RESULTS =====")
    print(f"Checkpoint: {config.runtime.checkpoint_path}")
    print(f"Feature layer: {config.vis.gradcam_target_layer}")
    print(f"t-SNE max samples: {config.vis.tsne_max_samples}")
    print(f"Saved figure: {metadata['figure_path']}")
    print("=======================================\n")


def run_transfer_attack_pipeline(
    config: ExperimentConfig,
    dataloaders: DataLoaders,
    device: torch.device,
) -> None:
    if config.runtime.checkpoint_path is None:
        raise ValueError("Transfer attack requires student --checkpoint-path.")
    if config.distill.teacher_checkpoint is None:
        raise ValueError("Transfer attack requires --teacher-checkpoint.")

    student_model = build_model(
        model_name=config.runtime.model_name,
        num_classes=config.data.num_classes,
        use_pretrained=config.runtime.use_pretrained,
        freeze_early_layers=config.runtime.freeze_early_layers,
    ).to(device)
    student_model = load_checkpoint(student_model, config.runtime.checkpoint_path, device)

    teacher_use_pretrained = config.distill.teacher_model_name.startswith("transfer_")
    teacher_model = build_model(
        model_name=config.distill.teacher_model_name,
        num_classes=config.data.num_classes,
        use_pretrained=teacher_use_pretrained,
        freeze_early_layers=False,
    ).to(device)
    teacher_model = load_checkpoint(teacher_model, config.distill.teacher_checkpoint, device)

    metrics = evaluate_transfer_attack(
        teacher_model=teacher_model,
        student_model=student_model,
        dataloader=dataloaders.test_loader,
        attack_config=config.attack,
        data_config=config.data,
        device=device,
        max_samples=config.data.max_eval_samples,
    )

    result = {
        "experiment_name": config.runtime.experiment_name,
        "mode": "transfer_attack",
        "teacher_model_name": config.distill.teacher_model_name,
        "teacher_checkpoint": config.distill.teacher_checkpoint,
        "student_model_name": config.runtime.model_name,
        "student_checkpoint": config.runtime.checkpoint_path,
        "metrics": metrics,
    }

    output_path = Path(config.runtime.tables_dir) / f"{config.runtime.experiment_name}_transfer_attack.json"
    save_metrics_json(result, str(output_path))

    print("\n===== TRANSFER ATTACK RESULTS =====")
    print(f"Teacher checkpoint: {config.distill.teacher_checkpoint}")
    print(f"Student checkpoint: {config.runtime.checkpoint_path}")
    print(f"Teacher Clean Accuracy: {metrics['teacher_clean_accuracy']:.2f}%")
    print(f"Teacher Adv Accuracy: {metrics['teacher_adv_accuracy']:.2f}%")
    print(f"Student Clean Accuracy: {metrics['student_clean_accuracy']:.2f}%")
    print(f"Student Adv Accuracy on Teacher-crafted Attacks: {metrics['student_adv_accuracy']:.2f}%")
    print(
        "Transfer Success Rate on Student Clean-Correct Samples: "
        f"{metrics['transfer_success_rate_on_student_clean']:.2f}%"
    )
    print("===================================\n")


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
        print("Clean/CIFAR-10-C, AugMix, PGD, Grad-CAM, and t-SNE paths are ready.")
        print("KD with robust teacher and transfer attack path are now available.")
        return

    if args.mode == "train":
        run_training_pipeline(config, dataloaders, device)
        return

    if args.mode == "train_kd_augmix":
        run_kd_training_pipeline(config, dataloaders, device)
        return

    if args.mode == "eval_clean":
        run_clean_evaluation_pipeline(config, dataloaders, device)
        return

    if args.mode == "eval_cifar10c":
        run_cifar10c_evaluation_pipeline(config, dataloaders, device)
        return

    if args.mode == "eval_cifar10c_full":
        run_cifar10c_full_sweep(config, device)
        return

    if args.mode == "eval_pgd":
        run_pgd_evaluation_pipeline(config, dataloaders, device)
        return

    if args.mode == "analyze_adv":
        run_analyze_adv_pipeline(config, dataloaders, device)
        return

    if args.mode == "visualize_tsne":
        run_tsne_visualization_pipeline(config, dataloaders, device)
        return

    if args.mode == "transfer_attack":
        run_transfer_attack_pipeline(config, dataloaders, device)
        return

    print(f"Mode '{args.mode}' is registered in the config skeleton.")


if __name__ == "__main__":
    main()