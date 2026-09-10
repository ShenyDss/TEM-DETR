"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
import inspect
from typing import Iterable, Mapping

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _unpack_tem_batch(batch, device):
    """Move an upstream batch and peel paired TEM metadata from targets.

    The standard collate path still returns ``(samples, targets)``.  The
    paired dataset stores the aligned normal image and protection mask on each
    target so that upstream dataloaders/evaluators remain usable.  This helper
    removes those private fields before the official criterion sees targets and
    returns them as model-only keyword arguments.  A four-tuple is accepted as
    well for custom collate functions that keep pair fields outside targets.
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError("upstream TEM loader must yield a tuple or list")
    if len(batch) == 2:
        samples, raw_targets = batch
        explicit_normal, explicit_protection = None, None
    elif len(batch) == 4:
        samples, raw_targets, explicit_normal, explicit_protection = batch
    else:
        raise ValueError("upstream TEM loader must yield (samples, targets) or four paired fields")
    if not isinstance(raw_targets, (tuple, list)):
        raise TypeError("upstream TEM targets must be a sequence of mappings")

    targets = []
    normals = []
    protections = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            raise TypeError("upstream TEM target entries must be dictionaries")
        target = {key: value.to(device) if hasattr(value, "to") else value
                  for key, value in raw_target.items()}
        normal = target.pop("normal", None)
        protection = target.pop("protection_mask", None)
        if normal is not None:
            normals.append(normal)
        if protection is not None:
            protections.append(protection)
        targets.append(target)

    samples = samples.to(device)
    normal = explicit_normal.to(device) if hasattr(explicit_normal, "to") else explicit_normal
    protection = (
        explicit_protection.to(device)
        if hasattr(explicit_protection, "to") else explicit_protection
    )
    if normal is None and normals:
        normal = torch.stack(normals, dim=0)
    if protection is None and protections:
        protection = torch.stack(protections, dim=0)
    return samples, targets, normal, protection


def _merge_tem_losses(loss_dict, outputs):
    """Append model-side CVMA/LQSD losses to official detector losses."""
    if not isinstance(outputs, Mapping):
        return
    tem_losses = outputs.get("tem_losses")
    if tem_losses is None:
        return
    if not isinstance(tem_losses, Mapping):
        raise TypeError("outputs['tem_losses'] must be a mapping")
    for name, value in tem_losses.items():
        if not isinstance(name, str) or not torch.is_tensor(value) or value.ndim != 0:
            raise TypeError("TEM auxiliary losses must be scalar tensors keyed by strings")
        loss_dict[name] = value


def _forward_model(model, samples, targets, normal=None, protection_mask=None, epoch=None):
    """Call vanilla RT-DETR or paired TEMRTDETR without changing either API."""
    if normal is None and protection_mask is None:
        return model(samples, targets=targets)
    kwargs = dict(
        targets=targets,
        normal=normal,
        protection_mask=protection_mask,
    )
    if "tem_epoch" in inspect.signature(model.forward).parameters:
        kwargs["tem_epoch"] = epoch
    return model(
        samples,
        **kwargs,
    )


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets, normal, protection_mask = _unpack_tem_batch(batch, device)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = _forward_model(model, samples, targets, normal, protection_mask, epoch)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
                _merge_tem_losses(loss_dict, outputs)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = _forward_model(model, samples, targets, normal, protection_mask, epoch)
            loss_dict = criterion(outputs, targets, **metas)
            _merge_tem_losses(loss_dict, outputs)
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets, _, _ = _unpack_tem_batch(batch, device)

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator
