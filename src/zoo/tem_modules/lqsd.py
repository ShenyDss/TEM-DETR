"""Layer-wise Query Self-Distillation (LQSD) losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor


def _validate_stack(name: str, value: Tensor) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [L, B, Q, D]")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least the final decoder layer")


def _per_layer_mean(values: Tensor, zero: Tensor) -> Tensor:
    """Reduce [layers, batch, queries] without producing NaN on empty batches."""
    if values.numel() == 0:
        return zero.expand(values.shape[0])
    return values.reshape(values.shape[0], -1).mean(dim=1)


def lqsd_loss(
    normal_hidden: Tensor,
    anomaly_hidden: Tensor,
    normal_logits: Tensor,
    anomaly_logits: Tensor,
    temperature: float | Tensor,
) -> dict[str, Tensor]:
    """Distil layers 1..L-1 from detached final-layer branch teachers.

    Each branch contributes independently; depth weights rise linearly from
    the earliest layer to the penultimate layer and sum to one.
    """
    _validate_stack("normal_hidden", normal_hidden)
    _validate_stack("anomaly_hidden", anomaly_hidden)
    _validate_stack("normal_logits", normal_logits)
    _validate_stack("anomaly_logits", anomaly_logits)
    if normal_hidden.shape != anomaly_hidden.shape:
        raise ValueError("normal_hidden and anomaly_hidden must have identical shapes")
    if normal_logits.shape != anomaly_logits.shape:
        raise ValueError("normal_logits and anomaly_logits must have identical shapes")
    if normal_logits.shape[:3] != normal_hidden.shape[:3]:
        raise ValueError("logit stacks must align with hidden stacks in [L, B, Q]")
    if normal_hidden.shape[-1] == 0:
        raise ValueError("hidden feature dimension must be positive")
    if normal_logits.shape[-1] == 0:
        raise ValueError("logit class dimension must be positive")
    if isinstance(temperature, Tensor):
        if temperature.numel() != 1:
            raise ValueError("temperature must be a positive scalar")
        temperature_value = float(temperature.detach().cpu())
    else:
        temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be positive")

    layers = normal_hidden.shape[0]
    zero = (
        normal_hidden[:-1].sum() * 0.0
        + anomaly_hidden[:-1].sum() * 0.0
        + normal_logits[:-1].sum() * 0.0
        + anomaly_logits[:-1].sum() * 0.0
    )
    if layers == 1:
        return {"feature": zero, "kl": zero}

    weights = torch.arange(
        1, layers, device=normal_hidden.device, dtype=normal_hidden.dtype
    )
    weights = weights / weights.sum()

    normal_feature = 1.0 - functional.cosine_similarity(
        normal_hidden[:-1], normal_hidden[-1].detach().unsqueeze(0), dim=-1
    )
    anomaly_feature = 1.0 - functional.cosine_similarity(
        anomaly_hidden[:-1], anomaly_hidden[-1].detach().unsqueeze(0), dim=-1
    )
    feature = (
        _per_layer_mean(normal_feature, zero) + _per_layer_mean(anomaly_feature, zero)
    ).mul(weights).sum()

    normal_student_log_probs = functional.log_softmax(normal_logits[:-1] / temperature_value, dim=-1)
    anomaly_student_log_probs = functional.log_softmax(anomaly_logits[:-1] / temperature_value, dim=-1)
    normal_teacher_probs = functional.softmax(normal_logits[-1].detach() / temperature_value, dim=-1)
    anomaly_teacher_probs = functional.softmax(anomaly_logits[-1].detach() / temperature_value, dim=-1)
    normal_kl = functional.kl_div(
        normal_student_log_probs, normal_teacher_probs.unsqueeze(0), reduction="none"
    ).sum(dim=-1)
    anomaly_kl = functional.kl_div(
        anomaly_student_log_probs, anomaly_teacher_probs.unsqueeze(0), reduction="none"
    ).sum(dim=-1)
    kl = (
        _per_layer_mean(normal_kl, zero) + _per_layer_mean(anomaly_kl, zero)
    ).mul(weights).sum() * (temperature_value**2)

    return {"feature": feature, "kl": kl}
