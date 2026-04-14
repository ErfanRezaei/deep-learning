from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_classification_criterion(label_smoothing: float = 0.0) -> nn.Module:

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    hard_criterion: nn.Module,
    alpha: float = 0.7,
    temperature: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


    hard_loss = hard_criterion(student_logits, targets)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)

    total_loss = (1.0 - alpha) * hard_loss + alpha * soft_loss
    return total_loss, hard_loss.detach(), soft_loss.detach()