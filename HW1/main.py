import argparse
from pathlib import Path

import torch
import torch.nn as nn

from models import MLP
from parameters import TrainingConfig
from test import test_model
from train import (
    evaluate,
    get_mnist_dataloaders,
    plot_training_history,
    train_one_epoch,
)


def str_to_bool(value: str) -> bool:
    return value.lower() in ["true", "1", "yes", "y"]


def parse_hidden_dims(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="MNIST MLP homework")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--download", type=str, default="true")

    parser.add_argument("--input_size", type=int, default=28 * 28)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_batchnorm", type=str, default="false")

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--report_dir", type=str, default="report")

    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--l2_lambda", type=float, default=0.0)

    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--use_scheduler", type=str, default="false")
    parser.add_argument("--scheduler_step_size", type=int, default=3)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        random_seed=args.random_seed,
        download=str_to_bool(args.download),
        input_size=args.input_size,
        num_classes=args.num_classes,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        dropout=args.dropout,
        use_batchnorm=str_to_bool(args.use_batchnorm),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        report_dir=args.report_dir,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        early_stopping_patience=args.early_stopping_patience,
        use_scheduler=str_to_bool(args.use_scheduler),
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
    )
    return config


def main() -> None:
    config = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config.report_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(config)

    model = MLP(
        input_size=config.input_size,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        activation=config.activation,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_lambda,
    )

    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            l1_lambda=config.l1_lambda,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), config.save_path)
            print(f"Best model saved to {config.save_path}")
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    plot_training_history(history, config.report_dir)
    print(f"Plots saved in: {config.report_dir}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(config.save_path, map_location=device))

    test_loss, test_acc = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print("Final Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()