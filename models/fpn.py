"""
FPN Pixel Decoder for Mask2Former.

Takes 4-scale backbone features and produces:
  - mask_features  : (B, hidden_dim, H/4, W/4)  — full-res mask feature map
  - multi_scale_features : list of 3 tensors at H/8, H/16, H/32  — for cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FPNPixelDecoder(nn.Module):
    def __init__(self, in_channels: dict, hidden_dim: int = config.HIDDEN_DIM):
        """
        Args:
            in_channels: dict mapping stage name → channel count
                e.g. {'res2': 64, 'res3': 128, 'res4': 256, 'res5': 512}
            hidden_dim: output channel dimension (default 256)
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Lateral 1×1 projections (high-res → low-res order for top-down)
        self.lateral_convs = nn.ModuleDict({
            name: nn.Conv2d(ch, hidden_dim, kernel_size=1)
            for name, ch in in_channels.items()
        })

        # 3×3 output convolutions after fusion
        self.output_convs = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            )
            for name in in_channels
        })

        # Final conv on the highest-resolution map (res2) → mask_features
        self.mask_features_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                            padding=1)

        self.stages = ["res2", "res3", "res4", "res5"]  # low-res → high-res

    def forward(self, features: dict):
        """
        Args:
            features: dict {'res2', 'res3', 'res4', 'res5'}
        Returns:
            mask_features       : (B, hidden_dim, H/4, W/4)
            multi_scale_features: [feat_res5, feat_res4, feat_res3]  (3 levels)
        """
        # Project all lateral connections
        laterals = {s: self.lateral_convs[s](features[s]) for s in self.stages}

        # Top-down fusion: start from res5 (coarsest)
        fused = {}
        fused["res5"] = self.output_convs["res5"](laterals["res5"])

        for coarse, fine in [("res5", "res4"), ("res4", "res3"), ("res3", "res2")]:
            upsampled = F.interpolate(fused[coarse], size=laterals[fine].shape[-2:],
                                      mode="nearest")
            fused[fine] = self.output_convs[fine](laterals[fine] + upsampled)

        # mask_features: highest resolution (res2)
        mask_features = self.mask_features_conv(fused["res2"])

        # multi-scale for cross-attention: 3 levels (res5, res4, res3)
        multi_scale_features = [fused["res5"], fused["res4"], fused["res3"]]

        return mask_features, multi_scale_features
