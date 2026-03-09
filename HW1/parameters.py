from dataclasses import dataclass, field


@dataclass
class TrainingConfig:

    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 0
    val_split: float = 0.1
    random_seed: int = 42
    download: bool = True

    input_size: int = 28 * 28
    num_classes: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.0
    use_batchnorm: bool = False

    learning_rate: float = 0.001
    num_epochs: int = 8
    save_path: str = "best_model.pt"
    report_dir: str = "report"

    l1_lambda: float = 0.0
    l2_lambda: float = 0.0

    early_stopping_patience: int = 3

    use_scheduler: bool = False
    scheduler_step_size: int = 3
    scheduler_gamma: float = 0.5