"""TEM-DETR integration point for the upstream RT-DETRv2 graph."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn

from .rtdetr.rtdetr import RTDETR
from ..core import register

from tem_detr.models.cvma import CrossViewMaskedAttention
from tem_detr.models.lqsd import lqsd_loss
from tem_detr.models.normal_query_generator import NormalQueryGenerator
from tem_detr.losses.subset_matcher import select_discrepancy_topk


@register()
class TEMRTDETR(RTDETR):
    """Upstream-compatible anomaly-only TEM detector scaffold.

    The default is the official anomaly-only RT-DETRv2 graph.  When enabled,
    paired training runs the same upstream backbone/encoder and decoder on
    both views, inserting CVMA on encoder memory and exposing LQSD losses to
    the upstream solver through ``tem_losses``.
    """

    # Keep upstream dependency injection active when the model is registered
    # under a new config section name (TEMRTDETR instead of RTDETR).
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        paired_query_enabled: bool = False,
        cvma_enabled: bool = False,
        lqsd_enabled: bool = False,
        cvma_mask_ratio: float = 0.20,
        cvma_position_bias: bool = True,
        lqsd_temperature: float = 2.0,
        nqg_enabled: bool = False,
        top_k: int = 150,
        selection_start_epoch: int = 5,
        cvma_loss_weight: float = 0.1,
        lqsd_loss_weight: float = 0.1,
        nqg_loss_weight: float = 0.05,
    ) -> None:
        super().__init__(backbone, encoder, decoder)
        self.paired_query_enabled = bool(paired_query_enabled)
        self.cvma_enabled = bool(cvma_enabled)
        self.lqsd_enabled = bool(lqsd_enabled)
        self.lqsd_temperature = float(lqsd_temperature)
        self.nqg_enabled = bool(nqg_enabled)
        self.top_k = int(top_k)
        self.selection_start_epoch = int(selection_start_epoch)
        self.cvma_loss_weight = float(cvma_loss_weight)
        self.lqsd_loss_weight = float(lqsd_loss_weight)
        self.nqg_loss_weight = float(nqg_loss_weight)
        hidden_dim = int(getattr(decoder, "hidden_dim", 256))
        nhead = int(getattr(decoder, "nhead", 8))
        self.cvma = CrossViewMaskedAttention(
            hidden_dim,
            nhead,
            cvma_mask_ratio,
            enabled=self.cvma_enabled,
            position_bias=cvma_position_bias,
        )
        self.nqg = NormalQueryGenerator(
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_feature_levels=int(getattr(decoder, "num_levels", 3)),
            num_queries=int(getattr(decoder, "num_queries", 300)),
        )

    @staticmethod
    def _gather_queries(values, indices):
        return values.gather(1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

    def _token_protection(self, protection_mask, spatial_shapes, batch_size, device):
        token_count = sum(int(h) * int(w) for h, w in spatial_shapes)
        if protection_mask is None:
            return torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        if protection_mask.ndim == 2 and protection_mask.shape == (batch_size, token_count):
            return protection_mask.to(device=device, dtype=torch.bool)
        if protection_mask.ndim == 3:
            protection_mask = protection_mask.unsqueeze(1)
        if protection_mask.ndim != 4 or protection_mask.shape[0] != batch_size:
            raise ValueError("protection_mask must have shape [B,H,W] or [B,N]")
        mask = protection_mask.to(device=device, dtype=torch.float32)
        levels = [
            F.interpolate(mask, size=(int(h), int(w)), mode="nearest").to(torch.bool).flatten(1)
            for h, w in spatial_shapes
        ]
        return torch.cat(levels, dim=1)

    def _decode(self, memory, spatial_shapes, targets):
        del targets
        decoder = self.decoder
        content, ref_points, enc_boxes, enc_logits = decoder._get_decoder_input(
            memory, spatial_shapes
        )
        decoded = decoder.decoder(
            content,
            ref_points,
            memory,
            spatial_shapes,
            decoder.dec_bbox_head,
            decoder.dec_score_head,
            decoder.query_pos_head,
            return_hidden=True,
        )
        out_bboxes, out_logits, hidden = decoded
        output = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training and decoder.aux_loss:
            output["aux_outputs"] = decoder._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            output["enc_aux_outputs"] = decoder._set_aux_loss(enc_logits, enc_boxes)
            output["enc_meta"] = {"class_agnostic": decoder.query_select_method == "agnostic"}
        # Keep the proposal references used to initialize the decoder. NQG in
        # the original TEM-DETR is conditioned on these shared proposals,
        # rather than on the already-refined final boxes.
        references = ref_points.sigmoid().detach()
        return output, hidden, out_logits, references

    def _forward_paired(self, anomaly, targets, normal, protection_mask):
        # Run both views through the official graph.  Some upstream transforms
        # (padding/rounding in multi-scale resize) can produce a one-pixel
        # discrepancy between the two feature pyramids; normalize the normal
        # branch to the anomaly branch before flattening into encoder tokens.
        anomaly_features = self.encoder(self.backbone(anomaly))
        normal_features = self.encoder(self.backbone(normal))
        if len(normal_features) != len(anomaly_features):
            raise ValueError("normal and anomaly feature pyramid levels must match")
        aligned_normal = []
        for normal_level, anomaly_level in zip(normal_features, anomaly_features):
            if normal_level.shape[-2:] != anomaly_level.shape[-2:]:
                normal_level = F.interpolate(
                    normal_level,
                    size=anomaly_level.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_normal.append(normal_level)
        normal_features = aligned_normal
        normal_memory, spatial_shapes = self.decoder._get_encoder_input(normal_features)
        anomaly_memory, anomaly_shapes = self.decoder._get_encoder_input(anomaly_features)
        if spatial_shapes != anomaly_shapes:
            raise ValueError("normal and anomaly encoder spatial shapes must match after alignment")
        token_mask = self._token_protection(
            protection_mask, spatial_shapes, anomaly.shape[0], anomaly.device
        )
        cvma_output = self.cvma(normal_memory, anomaly_memory, spatial_shapes, token_mask)
        anomaly_output, anomaly_hidden, anomaly_logits, references = self._decode(
            cvma_output.anomaly_memory, spatial_shapes, targets
        )
        tem_losses = {}
        if self.cvma_enabled:
            tem_losses.update(
                {
                    "loss_cvma_cos": cvma_output.reconstruction_loss["cosine"] * self.cvma_loss_weight,
                    "loss_cvma_smooth_l1": cvma_output.reconstruction_loss["smooth_l1"] * self.cvma_loss_weight,
                }
            )
        if self.paired_query_enabled or self.lqsd_enabled or self.nqg_enabled:
            normal_output, normal_hidden, normal_logits, _ = self._decode(
                cvma_output.normal_memory, spatial_shapes, targets
            )
        if self.lqsd_enabled:
            distill = lqsd_loss(
                normal_hidden,
                anomaly_hidden,
                normal_logits,
                anomaly_logits,
                self.lqsd_temperature,
            )
            tem_losses.update(
                {
                    "loss_lqsd_feat": distill["feature"] * self.lqsd_loss_weight,
                    "loss_lqsd_kl": distill["kl"] * self.lqsd_loss_weight,
                }
            )
        if self.nqg_enabled:
            generated_normal = self.nqg(
                cvma_output.normal_memory,
                spatial_shapes,
                references,
            )
            normal_teacher = normal_hidden[-1].detach()
            nqg_cos = (1.0 - F.cosine_similarity(generated_normal, normal_teacher, dim=-1)).mean()
            nqg_l1 = F.smooth_l1_loss(generated_normal, normal_teacher)
            tem_losses.update(
                {
                    "loss_nqg_cos": nqg_cos * self.nqg_loss_weight,
                    "loss_nqg_smooth_l1": nqg_l1 * self.nqg_loss_weight,
                }
            )
        if (self.paired_query_enabled or self.nqg_enabled) and self._tem_epoch >= self.selection_start_epoch:
            selection = select_discrepancy_topk(anomaly_hidden[-1], normal_hidden[-1], self.top_k)
            anomaly_output["pred_logits"] = self._gather_queries(anomaly_output["pred_logits"], selection.indices)
            anomaly_output["pred_boxes"] = self._gather_queries(anomaly_output["pred_boxes"], selection.indices)
            anomaly_output["query_indices"] = selection.indices
        if tem_losses:
            anomaly_output["tem_losses"] = tem_losses
        return anomaly_output

    def _forward_explicit_reference(self, anomaly, normal):
        """Rank anomaly queries against queries decoded from a normal reference."""
        anomaly_features = self.encoder(self.backbone(anomaly))
        normal_features = self.encoder(self.backbone(normal))
        if len(normal_features) != len(anomaly_features):
            raise ValueError("normal and anomaly feature pyramid levels must match")
        aligned_normal = []
        for normal_level, anomaly_level in zip(normal_features, anomaly_features):
            if normal_level.shape[-2:] != anomaly_level.shape[-2:]:
                normal_level = F.interpolate(
                    normal_level,
                    size=anomaly_level.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_normal.append(normal_level)
        normal_memory, normal_shapes = self.decoder._get_encoder_input(aligned_normal)
        anomaly_memory, anomaly_shapes = self.decoder._get_encoder_input(anomaly_features)
        if normal_shapes != anomaly_shapes:
            raise ValueError("normal and anomaly encoder spatial shapes must match")

        anomaly_output, anomaly_hidden, _, _ = self._decode(
            anomaly_memory, anomaly_shapes, None
        )
        _, normal_hidden, _, _ = self._decode(normal_memory, normal_shapes, None)
        selection = select_discrepancy_topk(
            anomaly_hidden[-1], normal_hidden[-1], self.top_k
        )
        anomaly_output["pred_logits"] = self._gather_queries(
            anomaly_output["pred_logits"], selection.indices
        )
        anomaly_output["pred_boxes"] = self._gather_queries(
            anomaly_output["pred_boxes"], selection.indices
        )
        anomaly_output["query_indices"] = selection.indices
        return anomaly_output

    def forward(
        self,
        x,
        targets: Any = None,
        normal=None,
        protection_mask=None,
        tem_epoch=None,
        inference_mode="generated",
    ):
        self._tem_epoch = int(tem_epoch) if tem_epoch is not None else 0
        if self.training and normal is not None and (
            self.paired_query_enabled or self.cvma_enabled or self.lqsd_enabled or self.nqg_enabled
        ):
            return self._forward_paired(x, targets, normal, protection_mask)
        if not self.training and inference_mode == "explicit":
            if normal is None:
                raise ValueError("explicit-reference inference requires a normal image")
            return self._forward_explicit_reference(x, normal)
        if inference_mode != "generated":
            raise ValueError("inference_mode must be 'generated' or 'explicit'")
        del normal, protection_mask
        if not self.training and self.nqg_enabled:
            feats = self.encoder(self.backbone(x))
            memory, spatial_shapes = self.decoder._get_encoder_input(feats)
            output, hidden, _, references = self._decode(memory, spatial_shapes, None)
            generated = self.nqg(memory, spatial_shapes, references)
            selection = select_discrepancy_topk(hidden[-1], generated, self.top_k)
            output["pred_logits"] = self._gather_queries(output["pred_logits"], selection.indices)
            output["pred_boxes"] = self._gather_queries(output["pred_boxes"], selection.indices)
            return output
        return super().forward(x, targets)
