"""Cross-view masked attention (CVMA) for paired TEM-DETR training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class CVMAOutput:
    """Memories and masked-token reconstruction diagnostics from CVMA."""

    normal_memory: Tensor
    anomaly_memory: Tensor
    reconstruction_loss: dict[str, Tensor]
    mask_indices: Tensor


class CrossViewMaskedAttention(nn.Module):
    """Reconstruct randomly masked, non-defect tokens from the aligned view.

    CVMA is deliberately a replacement-only block: unmasked encoded memory is
    never routed through an attention or projection operation.  This preserves
    the encoder representation exactly at those positions and makes the
    single-image inference path an allocation-free identity operation.
    """

    def __init__(
        self,
        hidden_dim: int,
        nhead: int,
        mask_ratio: float = 0.20,
        *,
        enabled: bool = True,
        position_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if nhead <= 0:
            raise ValueError("nhead must be positive")
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.mask_ratio = mask_ratio
        self.enabled = bool(enabled)
        self.position_bias = bool(position_bias)
        self.head_dim = hidden_dim // nhead

        self.mask_token = nn.Parameter(torch.empty(hidden_dim))
        self.view_type_embedding = nn.Embedding(2, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # The MLP learns a per-head correction to a conservative, aligned
        # preference.  Its zero initialization means that before learning the
        # strictly negative squared-distance term is maximal at zero offset.
        self.relative_position_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nhead),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.view_type_embedding.weight, std=0.02)
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        for layer in self.relative_position_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _validate_spatial_shapes(
        spatial_shapes: Sequence[tuple[int, int]], token_count: int
    ) -> None:
        if not spatial_shapes:
            raise ValueError("spatial_shapes must contain at least one feature level")
        total = 0
        for shape in spatial_shapes:
            if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
                raise ValueError("spatial_shapes must contain positive (height, width) pairs")
            total += int(shape[0]) * int(shape[1])
        if total != token_count:
            raise ValueError(
                "spatial_shapes token count does not match the encoded memory length"
            )

    def _validate_inputs(
        self,
        normal_memory: Tensor,
        anomaly_memory: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
        protection_mask: Tensor,
    ) -> None:
        if normal_memory.ndim != 3 or anomaly_memory.ndim != 3:
            raise ValueError("normal_memory and anomaly_memory must have shape [B, N, C]")
        if normal_memory.shape != anomaly_memory.shape:
            raise ValueError("normal_memory and anomaly_memory must have identical shapes")
        if normal_memory.shape[-1] != self.hidden_dim:
            raise ValueError("memory hidden dimension does not match CVMA hidden_dim")
        if normal_memory.device != anomaly_memory.device:
            raise ValueError("normal_memory and anomaly_memory must be on the same device")
        if protection_mask.shape != normal_memory.shape[:2]:
            raise ValueError("protection_mask must have shape [B, N] matching encoded memory")
        self._validate_spatial_shapes(spatial_shapes, normal_memory.shape[1])

    def _sample_mask(self, protection_mask: Tensor, generator: torch.Generator | None) -> Tensor:
        """Sample exactly floor(mask_ratio * eligible) positions per item."""
        sampled = torch.zeros_like(protection_mask, dtype=torch.bool)
        for batch_index in range(protection_mask.shape[0]):
            eligible = (~protection_mask[batch_index]).nonzero(as_tuple=False).flatten()
            count = math.floor(eligible.numel() * self.mask_ratio)
            if count == 0:
                continue
            permutation = torch.randperm(
                eligible.numel(), device=eligible.device, generator=generator
            )[:count]
            sampled[batch_index, eligible[permutation]] = True
        return sampled

    @staticmethod
    def _sine_1d(coordinates: Tensor, num_features: int) -> Tensor:
        feature_index = torch.arange(
            num_features, device=coordinates.device, dtype=coordinates.dtype
        )
        frequency = 10000.0 ** (feature_index / max(num_features, 1))
        angle = coordinates[:, None] * (2.0 * math.pi) / frequency
        return torch.where((feature_index.long() % 2 == 0)[None], angle.sin(), angle.cos())

    def _shared_2d_coordinates(
        self,
        spatial_shapes: Sequence[tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        coordinates: list[Tensor] = []
        for height, width in spatial_shapes:
            y = torch.arange(int(height), device=device, dtype=dtype) / max(int(height) - 1, 1)
            x = torch.arange(int(width), device=device, dtype=dtype) / max(int(width) - 1, 1)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            coordinates.append(torch.stack((grid_y.flatten(), grid_x.flatten()), dim=-1))
        return torch.cat(coordinates, dim=0)

    def _shared_2d_sine_positions(
        self,
        spatial_shapes: Sequence[tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the same flattened multi-scale sine encoding for both views."""
        coordinates = self._shared_2d_coordinates(spatial_shapes, device, dtype)
        coordinate_features = math.ceil(self.hidden_dim / 2)
        position = torch.cat(
            (
                self._sine_1d(coordinates[:, 0], coordinate_features),
                self._sine_1d(coordinates[:, 1], coordinate_features),
            ),
            dim=-1,
        )
        return position[:, : self.hidden_dim]

    def _relative_position_bias(self, query_positions: Tensor, key_positions: Tensor) -> Tensor:
        """Return [query, key, head] relative biases."""
        if query_positions.ndim != 2 or key_positions.ndim != 2:
            raise ValueError("relative-position inputs must have shape [N, D]")
        if query_positions.shape[-1] != key_positions.shape[-1]:
            raise ValueError("query and key position dimensions must match")
        if not self.position_bias:
            return query_positions.new_zeros(
                (query_positions.shape[0], key_positions.shape[0], self.nhead)
            )
        delta = query_positions[:, None, :] - key_positions[None, :, :]
        squared_distance = delta.square().sum(dim=-1, keepdim=True)
        if delta.shape[-1] == 2:
            mlp_input = torch.cat((delta, squared_distance), dim=-1)
        else:
            zeros = torch.zeros_like(squared_distance)
            mlp_input = torch.cat((zeros, zeros, squared_distance), dim=-1)
        learned = self.relative_position_mlp(mlp_input)
        return learned - squared_distance

    def _reconstruct_masked_tokens(
        self,
        target_memory: Tensor,
        source_memory: Tensor,
        mask: Tensor,
        positions: Tensor,
        coordinates: Tensor,
        target_view: int,
        source_view: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cross-attend one sample's masked queries to its aligned other view."""
        masked_indices = mask.nonzero(as_tuple=False).flatten()
        target = target_memory[masked_indices].detach()
        if masked_indices.numel() == 0:
            empty = target_memory.new_empty((0, self.hidden_dim))
            return empty, empty, masked_indices

        # Global cross-view attention scales as O(M*N) in token count and is
        # prohibitively large for 640px multi-scale RT-DETR features on 8GB
        # GPUs.  For large pyramids use the strict spatially aligned token as
        # the key/value (the paired images are pixel-aligned), retaining the
        # learned projections and masked reconstruction objective without
        # allocating a multi-gigabyte attention matrix.
        if source_memory.shape[0] > 2048:
            source = source_memory[masked_indices] + positions[masked_indices]
            attended = self.value_projection(source)
            replacement = self.mask_token[None] + self.output_projection(attended)
            return replacement, target, masked_indices

        query_input = (
            self.mask_token[None]
            + positions[masked_indices]
            + self.view_type_embedding.weight[target_view][None]
        )
        key_value_input = (
            source_memory + positions + self.view_type_embedding.weight[source_view][None]
        )
        query = self.query_projection(query_input).view(-1, self.nhead, self.head_dim)
        key = self.key_projection(key_value_input).view(-1, self.nhead, self.head_dim)
        value = self.value_projection(key_value_input).view(-1, self.nhead, self.head_dim)
        logits = torch.einsum("mhd,nhd->hmn", query, key) / math.sqrt(self.head_dim)
        position_bias = self._relative_position_bias(
            coordinates[masked_indices], coordinates
        ).permute(2, 0, 1)
        attention = (logits + position_bias).softmax(dim=-1)
        attended = torch.einsum("hmn,nhd->mhd", attention, value).reshape(-1, self.hidden_dim)

        # The mask token is a residual base, so the prediction does not leak
        # the original target feature while still supporting residual updates.
        replacement = self.mask_token[None] + self.output_projection(attended)
        return replacement, target, masked_indices

    @staticmethod
    def _empty_losses(normal_memory: Tensor, anomaly_memory: Tensor) -> dict[str, Tensor]:
        zero = normal_memory.sum() * 0.0 + anomaly_memory.sum() * 0.0
        return {"cosine": zero, "smooth_l1": zero}

    def forward(
        self,
        normal_memory: Tensor,
        anomaly_memory: Tensor,
        spatial_shapes: list[tuple[int, int]],
        protection_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> CVMAOutput:
        self._validate_inputs(normal_memory, anomaly_memory, spatial_shapes, protection_mask)
        protection_mask = protection_mask.to(device=normal_memory.device, dtype=torch.bool)

        if not self.enabled:
            return CVMAOutput(
                normal_memory=normal_memory,
                anomaly_memory=anomaly_memory,
                reconstruction_loss=self._empty_losses(normal_memory, anomaly_memory),
                mask_indices=torch.zeros_like(protection_mask),
            )

        # Eval must never sample or read the companion branch.  The dedicated
        # single-image inference method below is the allocation-free form.
        if not self.training:
            return CVMAOutput(
                normal_memory=normal_memory,
                anomaly_memory=anomaly_memory,
                reconstruction_loss=self._empty_losses(normal_memory, anomaly_memory),
                mask_indices=torch.zeros_like(protection_mask),
            )

        mask_indices = self._sample_mask(protection_mask, generator)
        positions = self._shared_2d_sine_positions(
            spatial_shapes, normal_memory.device, normal_memory.dtype
        )
        coordinates = self._shared_2d_coordinates(
            spatial_shapes, normal_memory.device, normal_memory.dtype
        )
        normal_output = normal_memory.clone()
        anomaly_output = anomaly_memory.clone()
        target_values: list[Tensor] = []
        predicted_values: list[Tensor] = []

        # A distinct random normal/anomaly target view is selected per batch
        # item; the other view remains the full key/value source.
        selected_anomaly = torch.randint(
            0,
            2,
            (normal_memory.shape[0],),
            device=normal_memory.device,
            generator=generator,
        ).bool()
        for batch_index in range(normal_memory.shape[0]):
            if selected_anomaly[batch_index]:
                replacement, target, indices = self._reconstruct_masked_tokens(
                    anomaly_memory[batch_index],
                    normal_memory[batch_index],
                    mask_indices[batch_index],
                    positions,
                    coordinates,
                    target_view=1,
                    source_view=0,
                )
                anomaly_output[batch_index, indices] = replacement.to(dtype=anomaly_output.dtype)
            else:
                replacement, target, indices = self._reconstruct_masked_tokens(
                    normal_memory[batch_index],
                    anomaly_memory[batch_index],
                    mask_indices[batch_index],
                    positions,
                    coordinates,
                    target_view=0,
                    source_view=1,
                )
                normal_output[batch_index, indices] = replacement.to(dtype=normal_output.dtype)
            if indices.numel():
                predicted_values.append(replacement)
                target_values.append(target)

        if not predicted_values:
            reconstruction_loss = self._empty_losses(normal_memory, anomaly_memory)
        else:
            prediction = torch.cat(predicted_values, dim=0)
            target = torch.cat(target_values, dim=0)
            reconstruction_loss = {
                "cosine": (1.0 - functional.cosine_similarity(prediction, target, dim=-1)).mean(),
                "smooth_l1": functional.smooth_l1_loss(prediction, target),
            }
        return CVMAOutput(
            normal_memory=normal_output,
            anomaly_memory=anomaly_output,
            reconstruction_loss=reconstruction_loss,
            mask_indices=mask_indices,
        )

    def inference(
        self, memory: Tensor, spatial_shapes: list[tuple[int, int]]) -> Tensor:
        """Return the original memory object for single-image inference."""
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_dim:
            raise ValueError("memory must have shape [B, N, C] with the configured hidden_dim")
        self._validate_spatial_shapes(spatial_shapes, memory.shape[1])
        return memory
