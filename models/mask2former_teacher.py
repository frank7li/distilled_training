"""
Mask2Former with Swin-S backbone (teacher model for distillation).
Identical architecture to mask2former.py but uses the SwinSBackbone.
"""

import torch
import torch.nn as nn

import config
from .swin_s_backbone import SwinSBackbone
from .fpn import FPNPixelDecoder
from .transformer_decoder import MaskedTransformerDecoder


class Mask2FormerTeacher(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 num_queries: int = config.NUM_QUERIES,
                 hidden_dim: int = config.HIDDEN_DIM,
                 num_decoder_layers: int = config.NUM_DECODER_LAYERS,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.backbone = SwinSBackbone(pretrained=pretrained_backbone)
        self.pixel_decoder = FPNPixelDecoder(
            in_channels=SwinSBackbone.out_channels,
            hidden_dim=hidden_dim,
        )
        self.transformer_decoder = MaskedTransformerDecoder(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
        )

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            List of dicts per decoder layer, each with:
                'pred_logits': (B, Q, num_classes+1)
                'pred_masks' : (B, Q, H/4, W/4)
        """
        features = self.backbone(images)
        mask_features, multi_scale = self.pixel_decoder(features)
        layer_outputs = self.transformer_decoder(multi_scale, mask_features)
        return [
            {"pred_logits": cls_logits, "pred_masks": mask_logits}
            for cls_logits, mask_logits in layer_outputs
        ]

    def forward_features(self, images: torch.Tensor):
        """
        Like forward() but also returns mask_features for distillation.

        Returns:
            layer_outputs: list of dicts (same as forward)
            mask_features: (B, hidden_dim, H/4, W/4)
        """
        features = self.backbone(images)
        mask_features, multi_scale = self.pixel_decoder(features)
        layer_outputs = self.transformer_decoder(multi_scale, mask_features)
        outputs = [
            {"pred_logits": cls_logits, "pred_masks": mask_logits}
            for cls_logits, mask_logits in layer_outputs
        ]
        return outputs, mask_features