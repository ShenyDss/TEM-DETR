"""Discrepancy-guided candidate selection and subset Hungarian matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from ..rtdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@dataclass(frozen=True)
class TopKSelection:
    """Global decoder-query positions selected by decreasing discrepancy."""

    indices: Tensor
    scores: Tensor


def _validate_query_pair(anomaly_queries: Tensor, generated_normal_queries: Tensor) -> None:
    if anomaly_queries.ndim != 3 or generated_normal_queries.ndim != 3:
        raise ValueError("query features must have shape [B, Q, D]")
    if anomaly_queries.shape != generated_normal_queries.shape:
        raise ValueError("anomaly and generated-normal queries must have identical shapes")
    if anomaly_queries.shape[-1] == 0:
        raise ValueError("query feature dimension must be positive")


def select_discrepancy_topk(
    anomaly_queries: Tensor,
    generated_normal_queries: Tensor,
    k: int,
    *,
    eps: float = 1e-8,
) -> TopKSelection:
    """Select exactly ``k`` globally indexed queries with largest cosine gap."""
    _validate_query_pair(anomaly_queries, generated_normal_queries)
    if not isinstance(k, int) or isinstance(k, bool) or not 0 < k <= anomaly_queries.shape[1]:
        raise ValueError("k must satisfy 0 < k <= query count")
    if eps <= 0:
        raise ValueError("eps must be positive")

    anomaly = functional.normalize(anomaly_queries, dim=-1, eps=eps)
    generated_normal = functional.normalize(generated_normal_queries, dim=-1, eps=eps)
    discrepancy = 1.0 - (anomaly * generated_normal).sum(dim=-1)
    scores, indices = discrepancy.topk(k, dim=1, largest=True, sorted=True)
    return TopKSelection(indices=indices, scores=scores)


def validate_target_capacity(targets: Sequence[Mapping[str, Tensor]], top_k: int) -> None:
    """Reject samples whose target count exceeds the detector candidate capacity."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    for batch_index, target in enumerate(targets):
        labels = target.get("labels")
        if not isinstance(labels, Tensor) or labels.ndim != 1:
            raise ValueError(f"targets[{batch_index}].labels must be a one-dimensional Tensor")
        if labels.numel() > top_k:
            raise ValueError(
                f"targets[{batch_index}] has {labels.numel()} objects, exceeding top_k={top_k}"
            )


class SubsetHungarianMatcher(nn.Module):
    """Run RT-DETR's assignment costs only over a discrepancy-selected subset."""

    def __init__(
        self,
        class_cost: float = 2.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        *,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("at least one matching cost must be nonzero")
        self.class_cost = float(class_cost)
        self.bbox_cost = float(bbox_cost)
        self.giou_cost = float(giou_cost)
        self.use_focal_loss = use_focal_loss
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    @staticmethod
    def _validate_outputs(outputs: Mapping[str, Tensor], selection: TopKSelection) -> tuple[Tensor, Tensor]:
        logits = outputs.get("pred_logits")
        boxes = outputs.get("pred_boxes")
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError("outputs.pred_logits must have shape [B, Q, C]")
        if not isinstance(boxes, Tensor) or boxes.shape != (*logits.shape[:2], 4):
            raise ValueError("outputs.pred_boxes must have shape [B, Q, 4]")
        if selection.indices.shape != selection.scores.shape or selection.indices.ndim != 2:
            raise ValueError("selection indices and scores must have matching shape [B, K]")
        if selection.indices.shape[0] != logits.shape[0]:
            raise ValueError("selection batch size must match outputs")
        if selection.indices.numel() and (
            selection.indices.min().item() < 0 or selection.indices.max().item() >= logits.shape[1]
        ):
            raise ValueError("selection indices must refer to output queries")
        return logits, boxes

    def _classification_cost(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.use_focal_loss:
            probabilities = logits.sigmoid()[:, labels]
            negative = (1 - self.alpha) * probabilities.pow(self.gamma) * (-(1 - probabilities + 1e-8).log())
            positive = self.alpha * (1 - probabilities).pow(self.gamma) * (-(probabilities + 1e-8).log())
            return positive - negative
        return -logits.softmax(-1)[:, labels]

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
        selection: TopKSelection,
    ) -> list[tuple[Tensor, Tensor]]:
        logits, boxes = self._validate_outputs(outputs, selection)
        if len(targets) != logits.shape[0]:
            raise ValueError("targets batch size must match outputs")
        validate_target_capacity(targets, selection.indices.shape[1])

        matches: list[tuple[Tensor, Tensor]] = []
        for batch_index, target in enumerate(targets):
            labels = target.get("labels")
            target_boxes = target.get("boxes")
            if not isinstance(labels, Tensor) or labels.ndim != 1:
                raise ValueError(f"targets[{batch_index}].labels must be a one-dimensional Tensor")
            if not isinstance(target_boxes, Tensor) or target_boxes.shape != (labels.numel(), 4):
                raise ValueError(f"targets[{batch_index}].boxes must have shape [targets[{batch_index}], 4]")
            if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= logits.shape[-1]):
                raise ValueError(f"targets[{batch_index}].labels contains an invalid class index")

            global_indices = selection.indices[batch_index].to(device=logits.device, dtype=torch.long)
            if labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                matches.append((empty, empty))
                continue

            local_logits = logits[batch_index, global_indices]
            # Decoder boxes are sigmoid-normalized, but mixed-precision overflow
            # can transiently produce NaN/Inf values.  Keep matcher geometry
            # finite so generalized IoU cannot abort the training loop.
            local_boxes = torch.nan_to_num(
                boxes[batch_index, global_indices].float(), nan=0.5, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
            target_labels = labels.to(device=logits.device, dtype=torch.long)
            target_boxes = target_boxes.to(device=boxes.device, dtype=local_boxes.dtype).clamp(0.0, 1.0)
            costs = (
                self.class_cost * self._classification_cost(local_logits, target_labels)
                + self.bbox_cost * torch.cdist(local_boxes, target_boxes, p=1)
                - self.giou_cost
                * generalized_box_iou(
                    box_cxcywh_to_xyxy(local_boxes), box_cxcywh_to_xyxy(target_boxes)
                )
            )
            costs = torch.nan_to_num(costs, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6).clamp(-1.0e6, 1.0e6)
            local_source, target_index = linear_sum_assignment(costs.cpu())
            local_source_tensor = torch.as_tensor(local_source, dtype=torch.int64, device=logits.device)
            target_tensor = torch.as_tensor(target_index, dtype=torch.int64, device=logits.device)
            matches.append((global_indices[local_source_tensor], target_tensor))
        return matches
