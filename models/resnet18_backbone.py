"""
True ResNet-18 backbone extracting 4-scale feature maps (res2–res5).

Output dict keys and shapes (for input H×W):
  res2: (B,  64, H/4,  W/4)
  res3: (B, 128, H/8,  W/8)
  res4: (B, 256, H/16, W/16)
  res5: (B, 512, H/32, W/32)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    out_channels = {
        "res2":  64,
        "res3": 128,
        "res4": 256,
        "res5": 512,
    }

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1  # res2: stride 4  →  64 ch
        self.layer2 = base.layer2  # res3: stride 8  → 128 ch
        self.layer3 = base.layer3  # res4: stride 16 → 256 ch
        self.layer4 = base.layer4  # res5: stride 32 → 512 ch

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}