"""
ResNet-152 backbone that extracts 4-scale feature maps (res2–res5).

Output dict keys and shapes (for input H×W):
  res2: (B,  256, H/4,  W/4)
  res3: (B,  512, H/8,  W/8)
  res4: (B, 1024, H/16, W/16)
  res5: (B, 2048, H/32, W/32)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNet18Backbone(nn.Module):
    # Channel dimensions per stage (bottleneck blocks)
    out_channels = {
        "res2":  256,
        "res3":  512,
        "res4": 1024,
        "res5": 2048,
    }

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet152(weights=weights)

        # Stem: conv1 + bn1 + relu + maxpool → stride 4
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1  # res2: stride 4  →  256 ch
        self.layer2 = base.layer2  # res3: stride 8  →  512 ch
        self.layer3 = base.layer3  # res4: stride 16 → 1024 ch
        self.layer4 = base.layer4  # res5: stride 32 → 2048 ch

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}
