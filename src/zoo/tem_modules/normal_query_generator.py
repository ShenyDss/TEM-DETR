"""Single-image normal query synthesis for TEM-DETR."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ..rtdetr.rtdetrv2_decoder import MLP, MSDeformableAttention


def _validate_spatial_shapes(spatial_shapes: Sequence[Sequence[int]], token_count: int) -> None:
    if not spatial_shapes:
        raise ValueError("spatial_shapes must contain at least one feature level")
    total = 0
    for shape in spatial_shapes:
        if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
            raise ValueError("spatial_shapes must contain positive (height, width) pairs")
        total += int(shape[0]) * int(shape[1])
    if total != token_count:
        raise ValueError("spatial_shapes token count does not match memory length")


class NormalQueryGenerator(nn.Module):
    """Generate paired normal query features from a single encoded image memory."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_feature_levels: int = 3,
        num_queries: int = 100,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_points: int = 4,
        source: str = "encoded_memory",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or nhead <= 0 or hidden_dim % nhead:
            raise ValueError("hidden_dim must be positive and divisible by nhead")
        if num_feature_levels <= 0 or num_queries <= 0 or dim_feedforward <= 0:
            raise ValueError("feature levels, queries and feedforward dimension must be positive")
        if source not in {"encoded_memory", "learned_slots"}:
            raise ValueError("normal query source must be encoded_memory or learned_slots")

        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.source = source
        self.normal_slots = nn.Parameter(torch.empty(num_queries, hidden_dim))
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        self.cross_attn = MSDeformableAttention(
            hidden_dim, nhead, num_feature_levels, num_points
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.normal_slots, std=0.02)
        for layer in self.query_pos_head.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        memory: Tensor,
        spatial_shapes: Sequence[Sequence[int]],
        reference_points: Tensor,
    ) -> Tensor:
        """Return one normal query feature for each shared decoder proposal."""
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_dim:
            raise ValueError("memory must have shape [B, N, C] with the configured hidden_dim")
        _validate_spatial_shapes(spatial_shapes, memory.shape[1])
        if len(spatial_shapes) != self.num_feature_levels:
            raise ValueError("spatial_shapes feature-level count does not match NQG configuration")
        if reference_points.shape != (memory.shape[0], self.num_queries, 4):
            raise ValueError("reference_points must have shape [B, num_queries, 4]")
        if reference_points.device != memory.device:
            raise ValueError("reference_points and memory must be on the same device")

        slots = self.normal_slots.unsqueeze(0).expand(memory.shape[0], -1, -1)
        query_position = self.query_pos_head(reference_points)
        if self.source == "learned_slots":
            output = self.norm1(slots + query_position)
        else:
            cross_update = self.cross_attn(
                slots + query_position,
                reference_points.unsqueeze(2),
                memory,
                spatial_shapes,
            )
            output = self.norm1(slots + self.dropout1(cross_update))
        ffn_update = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        return self.norm2(output + self.dropout3(ffn_update))


def _weighted_mean(values: Tensor, *, zero: Tensor) -> Tensor:
    return values.mean() if values.numel() else zero


def normal_query_loss(
    generated_normal: Tensor,
    generated_anomaly: Tensor,
    teacher: Tensor,
    cosine_weight: float,
    l1_weight: float,
) -> dict[str, Tensor]:
    """Distil explicit final normal queries into both generated normal views."""
    if generated_normal.ndim != 3 or generated_anomaly.ndim != 3 or teacher.ndim != 3:
        raise ValueError("generated queries and teacher must have shape [B, Q, D]")
    if generated_normal.shape != generated_anomaly.shape or generated_normal.shape != teacher.shape:
        raise ValueError("generated_normal, generated_anomaly and teacher must have identical shapes")
    if not math.isfinite(float(cosine_weight)) or not math.isfinite(float(l1_weight)):
        raise ValueError("cosine_weight and l1_weight must be finite")

    detached_teacher = teacher.detach()
    zero = generated_normal.sum() * 0.0 + generated_anomaly.sum() * 0.0
    cosine = _weighted_mean(
        1.0 - functional.cosine_similarity(generated_normal, detached_teacher, dim=-1),
        zero=zero,
    ) + _weighted_mean(
        1.0 - functional.cosine_similarity(generated_anomaly, detached_teacher, dim=-1),
        zero=zero,
    )
    smooth_l1 = _weighted_mean(
        functional.smooth_l1_loss(generated_normal, detached_teacher, reduction="none"), zero=zero
    ) + _weighted_mean(
        functional.smooth_l1_loss(generated_anomaly, detached_teacher, reduction="none"), zero=zero
    )
    return {
        "cosine": cosine * float(cosine_weight),
        "smooth_l1": smooth_l1 * float(l1_weight),
    }
