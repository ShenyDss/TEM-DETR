"""Check transformer pyramids, paired TEM backward and 448px evaluation on CPU."""
from pathlib import Path
import sys
from PIL import Image  # Preload Pillow DLLs on Windows.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig


def main():
    torch.set_num_threads(2)
    for name in ('tem_swin_t', 'tem_vit_b16'):
        cfg = YAMLConfig(str(ROOT / 'configs/tem' / f'{name}.yml'),
                         SwinTBackbone={'pretrained': False}, ViTB16Backbone={'pretrained': False},
                         eval_spatial_size=[448, 448])
        model = cfg.model.train()
        x, normal = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
        targets = [dict(labels=torch.tensor([1]), boxes=torch.tensor([[.5, .5, .2, .2]])) for _ in range(2)]
        out = model(x, targets, normal=normal, protection_mask=torch.zeros(2, 128, 128, dtype=torch.bool))
        losses = cfg.criterion(out, targets)
        losses.update(out['tem_losses'])
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()
        grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            image = torch.rand(1, 3, 448, 448)
            pyramid = model.backbone(image)
            assert [tuple(p.shape[-2:]) for p in pyramid] == [(56, 56), (28, 28), (14, 14)]
            prediction = model(image)
            assert prediction['pred_boxes'].shape == (1, 20, 4)
            assert torch.isfinite(prediction['pred_logits']).all()
        print(name, 'PASS: paired backward, finite backbone gradients, 448px pyramid and inference', flush=True)
        del model, cfg, out, losses, total, grads, pyramid, prediction


if __name__ == '__main__':
    main()
