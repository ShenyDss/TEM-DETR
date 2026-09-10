"""Torchvision transformer backbones with RT-DETR stride-8/16/32 outputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import swin_t, Swin_T_Weights, vit_b_16, ViT_B_16_Weights
from ...core import register


class ImageNetInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x):
        return (x - self.mean) / self.std


@register()
class SwinTBackbone(nn.Module):
    """Swin-T stages 2/3/4; channels 192/384/768, strides 8/16/32."""
    def __init__(self, pretrained=True):
        super().__init__()
        self.normalize = ImageNetInput()
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = model.features
        self.norms = nn.ModuleList(nn.LayerNorm(c) for c in (192, 384, 768))
        self.norms[-1].load_state_dict(model.norm.state_dict())

    def forward(self, x):
        x = self.normalize(x)
        outputs = []
        for i, stage in enumerate(self.features):
            x = stage(x)
            if i in (3, 5, 7):
                outputs.append(self.norms[len(outputs)](x).permute(0, 3, 1, 2).contiguous())
        return outputs


@register()
class ViTB16Backbone(nn.Module):
    """ViT-B/16 layers 4/8/12 with an explicit multiscale detection adapter.

    ViT has a single patch scale: these are adapted, not native pyramid maps.
    Interpolated pretrained position embeddings support variable input sizes.
    """
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        self.normalize = ImageNetInput()
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.patch_embed = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding
        self.dropout = model.encoder.dropout
        self.layers = model.encoder.layers
        self.norms = nn.ModuleList(nn.LayerNorm(768) for _ in range(3))
        for norm in self.norms:
            norm.load_state_dict(model.encoder.ln.state_dict())
        self.projections = nn.ModuleList(nn.Conv2d(768, out_channels, 1) for _ in range(3))

    def forward(self, x):
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError('ViT-B/16 detection inputs must be divisible by 32')
        patches = self.patch_embed(self.normalize(x))
        b, c, h, w = patches.shape
        x = torch.cat([self.class_token.expand(b, -1, -1), patches.flatten(2).transpose(1, 2)], 1)
        side = math.isqrt(self.pos_embedding.shape[1] - 1)
        pos = self.pos_embedding[:, 1:].transpose(1, 2).reshape(1, c, side, side)
        pos = F.interpolate(pos, size=(h, w), mode='bicubic', align_corners=False)
        pos = torch.cat([self.pos_embedding[:, :1], pos.flatten(2).transpose(1, 2)], 1)
        x = self.dropout(x + pos)
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in (3, 7, 11):
                j = len(outputs)
                feature = self.norms[j](x[:, 1:]).transpose(1, 2).reshape(b, c, h, w)
                feature = self.projections[j](feature)
                if j == 0:
                    feature = F.interpolate(feature, size=(h * 2, w * 2), mode='bilinear', align_corners=False)
                elif j == 2:
                    feature = F.avg_pool2d(feature, 2)
                outputs.append(feature)
        return outputs
