"""
Masked Transformer Decoder for Mask2Former.

Architecture per layer:
  1. Masked Cross-Attention  (queries attend to multi-scale features, gated by mask)
  2. Self-Attention          (queries attend to each other)
  3. Feed-Forward Network

Outputs per layer (for auxiliary losses):
  class_logits : (B, num_queries, num_classes + 1)
  mask_logits  : (B, num_queries, H/4, W/4)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class MaskedTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Masked cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,
                                                 batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 2. Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, query_pos: torch.Tensor,
                memory: torch.Tensor, memory_pos: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries    : (B, Q, C)
            query_pos  : (B, Q, C)  positional embeddings for queries
            memory     : (B, HW, C) flattened multi-scale feature map
            memory_pos : (B, HW, C) positional embeddings for memory
            attn_mask  : (B*nhead, Q, HW) or (B, Q, HW) — binary mask (True = ignore)
        """
        q = queries + query_pos
        k = memory + memory_pos

        # Expand attn_mask from (B, Q, HW) to (B*nhead, Q, HW) as required by PyTorch
        expanded_mask = None
        if attn_mask is not None:
            nhead = self.cross_attn.num_heads
            B = attn_mask.shape[0]
            expanded_mask = (attn_mask.unsqueeze(1)
                             .expand(-1, nhead, -1, -1)
                             .reshape(B * nhead, attn_mask.shape[1], attn_mask.shape[2]))

        # Masked cross-attention
        attended, _ = self.cross_attn(q, k, memory, attn_mask=expanded_mask)
        queries = self.norm1(queries + attended)

        # Self-attention
        q2 = queries + query_pos
        self_out, _ = self.self_attn(q2, q2, queries)
        queries = self.norm2(queries + self_out)

        # FFN
        queries = self.norm3(queries + self.ffn(queries))
        return queries


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 num_queries: int = config.NUM_QUERIES,
                 hidden_dim: int = config.HIDDEN_DIM,
                 num_layers: int = config.NUM_DECODER_LAYERS,
                 nhead: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Learnable query content and position embeddings
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_pos  = nn.Embedding(num_queries, hidden_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(hidden_dim, nhead=nhead)
            for _ in range(num_layers)
        ])

        # Per-layer prediction heads (class + mask)
        self.class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes + 1) for _ in range(num_layers)
        ])
        self.mask_embed_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=3)
            for _ in range(num_layers)
        ])

        # Positional encoding for multi-scale memory
        self.level_embed = nn.Embedding(3, hidden_dim)  # 3 FPN levels

    def _get_memory_and_pos(self, multi_scale_features):
        """
        Flatten 3 FPN levels into one memory sequence with level positional encodings.
        Returns:
            memory     : (B, sum(H_i*W_i), C)
            memory_pos : (B, sum(H_i*W_i), C)
        """
        B = multi_scale_features[0].shape[0]
        memory_parts, pos_parts = [], []
        for lvl, feat in enumerate(multi_scale_features):
            # feat: (B, C, H, W)
            h, w = feat.shape[-2:]
            flat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            # Simple 2D sinusoidal pos encoding
            pos = self._sinusoidal_pos(h, w, self.hidden_dim, feat.device)  # (H*W, C)
            pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, C)
            # Add level embedding
            level_emb = self.level_embed.weight[lvl].view(1, 1, -1)
            pos = pos + level_emb
            memory_parts.append(flat)
            pos_parts.append(pos)
        memory = torch.cat(memory_parts, dim=1)
        memory_pos = torch.cat(pos_parts, dim=1)
        return memory, memory_pos

    @staticmethod
    def _sinusoidal_pos(h: int, w: int, dim: int, device) -> torch.Tensor:
        """Returns (H*W, dim) sinusoidal position encodings."""
        assert dim % 4 == 0
        d = dim // 4
        y_pos = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)
        x_pos = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, dtype=torch.float32, device=device)
                        * (-math.log(10000.0) / d))
        sin_y = torch.sin(y_pos * div)
        cos_y = torch.cos(y_pos * div)
        sin_x = torch.sin(x_pos * div)
        cos_x = torch.cos(x_pos * div)
        # Encode y and x separately then interleave
        y_enc = torch.cat([sin_y, cos_y], dim=-1).unsqueeze(1).expand(-1, w, -1)  # (H,W,dim/2)
        x_enc = torch.cat([sin_x, cos_x], dim=-1).unsqueeze(0).expand(h, -1, -1)  # (H,W,dim/2)
        pos = torch.cat([y_enc, x_enc], dim=-1)  # (H, W, dim)
        return pos.reshape(h * w, dim)

    def forward(self, multi_scale_features: list, mask_features: torch.Tensor):
        """
        Args:
            multi_scale_features: [feat_res5, feat_res4, feat_res3] each (B, C, Hi, Wi)
            mask_features       : (B, C, H/4, W/4)

        Returns:
            List of (class_logits, mask_logits) per decoder layer:
                class_logits: (B, Q, num_classes+1)
                mask_logits : (B, Q, H/4, W/4)
        """
        B = mask_features.shape[0]
        memory, memory_pos = self._get_memory_and_pos(multi_scale_features)

        # Init queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)  # (B,Q,C)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)  # (B,Q,C)

        outputs = []
        prev_mask_logits = None

        for i, layer in enumerate(self.layers):
            # Build attention mask from previous mask predictions
            attn_mask = None
            if prev_mask_logits is not None:
                attn_mask = self._build_attn_mask(prev_mask_logits, multi_scale_features)

            queries = layer(queries, query_pos, memory, memory_pos, attn_mask=attn_mask)

            # Predict class and mask
            cls_logits = self.class_heads[i](queries)  # (B, Q, C+1)
            mask_emb = self.mask_embed_heads[i](queries)  # (B, Q, C)
            # Dot product with mask_features → (B, Q, H/4, W/4)
            mask_logits = torch.einsum("bqc,bchw->bqhw", mask_emb, mask_features)

            prev_mask_logits = mask_logits.detach()
            outputs.append((cls_logits, mask_logits))

        return outputs

    def _build_attn_mask(self, mask_logits: torch.Tensor,
                         multi_scale_features: list) -> torch.Tensor:
        """
        Build binary attention mask from mask predictions.
        For each FPN level, downsample mask logits and threshold.
        Returns: (B, Q, total_HW) bool tensor — True means IGNORE.
        """
        B, Q, Hm, Wm = mask_logits.shape
        masks = []
        for feat in multi_scale_features:
            h, w = feat.shape[-2:]
            m = F.interpolate(mask_logits, size=(h, w), mode="bilinear",
                              align_corners=False)  # (B, Q, h, w)
            m = (m.sigmoid() < 0.5).flatten(2)  # (B, Q, h*w) bool
            masks.append(m)
        attn_mask = torch.cat(masks, dim=2)  # (B, Q, total_HW)
        # If a query attends nowhere, unmask everything for that query
        all_masked = attn_mask.all(dim=2, keepdim=True)
        attn_mask = attn_mask & ~all_masked
        return attn_mask
