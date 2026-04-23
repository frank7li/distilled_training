"""
Semantic segmentation metrics: mIoU computation and mask→semantic map conversion.
"""

import torch
import torch.nn.functional as F


def predictions_to_semantic_map(outputs: dict, image_size: tuple,
                                 num_classes: int) -> torch.Tensor:
    """
    Convert Mask2Former outputs (from the last decoder layer) to a semantic
    segmentation map via argmax over class-weighted mask predictions.

    Args:
        outputs    : dict with 'pred_logits' (B, Q, C+1) and 'pred_masks' (B, Q, H', W')
        image_size : (H, W) — target output resolution
        num_classes: number of foreground classes

    Returns:
        semantic_map: (B, H, W) int64, values 0 = background/no-data, 1–num_classes = classes
    """
    pred_logits = outputs["pred_logits"]   # (B, Q, C+1)
    pred_masks  = outputs["pred_masks"]    # (B, Q, H', W')

    B, Q, C1 = pred_logits.shape
    H, W = image_size

    # Softmax over classes, drop the "no-object" class (last)
    cls_prob = pred_logits.softmax(-1)[..., :-1]  # (B, Q, C)

    # Upsample mask logits to image resolution
    pred_masks_up = F.interpolate(pred_masks, size=(H, W), mode="bilinear",
                                   align_corners=False)  # (B, Q, H, W)
    mask_prob = pred_masks_up.sigmoid()  # (B, Q, H, W)

    # Semantic logits: for each pixel, score for each class = sum over queries of
    #   cls_prob[q, c] * mask_prob[q, pixel]
    # Shape: (B, C, H, W)
    semantic_logits = torch.einsum("bqc,bqhw->bchw", cls_prob, mask_prob)

    # Argmax → class index (0-indexed within foreground, then shift +1 for 1-indexed output)
    semantic_map = semantic_logits.argmax(dim=1) + 1  # (B, H, W), values 1..C

    return semantic_map.long()


def compute_miou(pred_maps: torch.Tensor, gt_maps: torch.Tensor,
                 num_classes: int, ignore_index: int = 0) -> float:
    """
    Compute mean IoU over classes present in the ground truth.

    Args:
        pred_maps  : (B, H, W) int64, predicted class indices (1-indexed)
        gt_maps    : (B, H, W) int64, ground truth (1-indexed, 0=ignore)
        num_classes: number of classes (7 for LoveDA)
        ignore_index: GT value to ignore (default 0)

    Returns:
        mIoU: float
    """
    iou_per_class = []
    for cls in range(1, num_classes + 1):
        pred_cls = (pred_maps == cls)
        gt_cls   = (gt_maps   == cls) & (gt_maps != ignore_index)

        intersection = (pred_cls & gt_cls).sum().item()
        union        = (pred_cls | gt_cls).sum().item()

        if union == 0:
            continue  # class not present, skip
        iou_per_class.append(intersection / union)

    if not iou_per_class:
        return 0.0
    return sum(iou_per_class) / len(iou_per_class)
