"""
Swin-S backbone extracting 4-scale feature maps (res2–res5).

Swin-S stage outputs (for 512x512 input):
  res2: (B,  96, H/4,  W/4)   — after stage 1 (features[1])
  res3: (B, 192, H/8,  W/8)   — after stage 2 (features[3])
  res4: (B, 384, H/16, W/16)  — after stage 3 (features[5])
  res5: (B, 768, H/32, W/32)  — after stage 4 (features[7])

Note: torchvision Swin outputs (B, H, W, C); we permute to (B, C, H, W) for FPN.
"""

import torch
import torch.nn as nn
from torchvision.models import swin_s, Swin_S_Weights


class SwinSBackbone(nn.Module):
    out_channels = {
        "res2":  96,
        "res3": 192,
        "res4": 384,
        "res5": 768,
    }

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
        base = swin_s(weights=weights)

        # features is a Sequential of 8 blocks:
        # [0] patch embed, [1] stage1, [2] merge, [3] stage2,
        # [4] merge,       [5] stage3, [6] merge, [7] stage4
        f = base.features
        self.patch_embed = f[0]   # stride 4
        self.stage1      = f[1]   # → res2 (96 ch)
        self.merge1      = f[2]
        self.stage2      = f[3]   # → res3 (192 ch)
        self.merge2      = f[4]
        self.stage3      = f[5]   # → res4 (384 ch)
        self.merge3      = f[6]
        self.stage4      = f[7]   # → res5 (768 ch)

    def forward(self, x: torch.Tensor) -> dict:
        # Swin outputs (B, H, W, C); permute each to (B, C, H, W) for FPN
        x    = self.patch_embed(x)
        x    = self.stage1(x)
        res2 = x.permute(0, 3, 1, 2).contiguous()
        x    = self.stage2(self.merge1(x))
        res3 = x.permute(0, 3, 1, 2).contiguous()
        x    = self.stage3(self.merge2(x))
        res4 = x.permute(0, 3, 1, 2).contiguous()
        x    = self.stage4(self.merge3(x))
        res5 = x.permute(0, 3, 1, 2).contiguous()
        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}