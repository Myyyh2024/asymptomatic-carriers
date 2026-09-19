import torch.nn.functional as F


def kldiv(logits, targets, temperature=1.0, reduction="batchmean"):
    """Temperature-scaled KL divergence used for knowledge distillation."""
    student_log_probabilities = F.log_softmax(logits / temperature, dim=1)
    teacher_probabilities = F.softmax(targets / temperature, dim=1)
    return F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction=reduction,
    ) * (temperature * temperature)
