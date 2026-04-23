"""
Full Mask2Former model: Backbone → FPN Pixel Decoder → Transformer Decoder.
"""

import torch
import torch.nn as nn

import config
from .backbone import ResNet18Backbone
from .fpn import FPNPixelDecoder
from .transformer_decoder import MaskedTransformerDecoder


class Mask2Former(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 num_queries: int = config.NUM_QUERIES,
                 hidden_dim: int = config.HIDDEN_DIM,
                 num_decoder_layers: int = config.NUM_DECODER_LAYERS,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.pixel_decoder = FPNPixelDecoder(
            in_channels=ResNet18Backbone.out_channels,
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
