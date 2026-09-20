"""Paired MVTec COCO dataset adapter for upstream RT-DETRv2."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torchvision.transforms import functional as tvf

from ...core import register
from .coco_dataset import CocoDetection
from .._misc import Mask


DEFAULT_PAIR_PATTERN = r"^(?P<product>.+?)__(?P<index>\d+)__.*\.(?:png|PNG)$"


@register()
class PairedCocoDetection(CocoDetection):
    """Return ``(anomaly, target, normal, protection_mask)`` for TEM training.

    The adapter intentionally keeps the upstream COCO target contract and adds
    aligned template data as extra tuple fields for the TEM solver.
    """

    __inject__ = ["transforms"]

    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        normal_root: str | Path,
        pair_template: str = "{product}/train/good/{index}.png",
        pair_pattern: str = DEFAULT_PAIR_PATTERN,
        transforms=None,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
    ) -> None:
        super().__init__(
            img_folder,
            ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
        )
        self.normal_root = Path(normal_root).expanduser().resolve(strict=False)
        self.pair_template = pair_template
        self.pair_pattern = re.compile(pair_pattern)

    def _normal_path(self, file_name: str, image_record: Mapping[str, Any]) -> Path:
        """Resolve an explicit COCO pair field, falling back to the filename rule."""
        normal_file_name = image_record.get("normal_file_name")
        if not isinstance(normal_file_name, str):
            match = self.pair_pattern.fullmatch(Path(file_name).name)
            if match is None:
                raise ValueError(f"cannot resolve paired template for {file_name!r}")
            try:
                normal_file_name = self.pair_template.format(**match.groupdict())
            except (KeyError, ValueError) as error:
                raise ValueError(f"cannot render paired template for {file_name!r}: {error}") from error
        relative = Path(normal_file_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"normal template escapes normal_root for {file_name!r}")
        candidate = (self.normal_root / relative).resolve(strict=False)
        try:
            candidate.relative_to(self.normal_root)
        except ValueError as error:
            raise ValueError(f"normal template escapes normal_root for {file_name!r}") from error
        if not candidate.is_file():
            raise FileNotFoundError(f"normal template does not exist for {file_name!r}: {candidate}")
        return candidate

    @staticmethod
    def _protection_mask(annotations: list[Mapping[str, Any]], width: int, height: int) -> Mask:
        """Build a conservative pixel mask from COCO boxes for CVMA protection."""
        mask = torch.zeros((height, width), dtype=torch.uint8)
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x, y, box_width, box_height = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            left = max(0, min(width, int(x)))
            top = max(0, min(height, int(y)))
            right = max(0, min(width, int(x + box_width + 0.999999)))
            bottom = max(0, min(height, int(y + box_height + 0.999999)))
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 1
        return Mask(mask)

    def __getitem__(self, idx: int):
        image, target = self.load_item(idx)
        image_record = self.coco.loadImgs([self.ids[idx]])[0]
        normal_path = self._normal_path(image_record["file_name"], image_record)
        from PIL import Image

        with Image.open(normal_path) as normal_source:
            normal = normal_source.convert("RGB").copy()
        if normal.size != image.size:
            raise ValueError(
                f"unaligned pair for {image_record['file_name']!r}: "
                f"anomaly={image.size}, normal={normal.size}"
            )
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[self.ids[idx]]))
        protection = self._protection_mask(annotations, image.width, image.height)
        if self._transforms is not None:
            transformed = self._transforms(image, target, normal, protection, self)
            if not isinstance(transformed, tuple) or len(transformed) != 5:
                raise TypeError(
                    "paired upstream transforms must return (image, target, normal, protection_mask, dataset)"
                )
            image, target, normal, protection, _ = transformed
        image = tvf.pil_to_tensor(image) if not isinstance(image, Tensor) else image
        normal = tvf.pil_to_tensor(normal) if not isinstance(normal, Tensor) else normal
        image = tvf.convert_image_dtype(image, torch.float32)
        normal = tvf.convert_image_dtype(normal, torch.float32)
        if image.shape[-2:] != normal.shape[-2:]:
            raise ValueError(
                f"paired transform changed image sizes differently for {image_record['file_name']!r}: "
                f"anomaly={tuple(image.shape[-2:])}, normal={tuple(normal.shape[-2:])}"
            )
        protection = torch.as_tensor(protection, dtype=torch.bool)
        if protection.shape != image.shape[-2:]:
            raise ValueError(
                f"protection mask size differs from transformed image for {image_record['file_name']!r}: "
                f"mask={tuple(protection.shape)}, image={tuple(image.shape[-2:])}"
            )
        target["normal"] = normal
        target["protection_mask"] = protection
        return image, target
