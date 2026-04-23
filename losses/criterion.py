"""
Hungarian Matcher + Set Criterion for Mask2Former.

Target format per image:
  targets[i] = {
      'labels': LongTensor (K,)         — class ids (1-indexed, 1–7)
      'masks' : BoolTensor  (K, H, W)   — binary masks per instance
  }
where K = number of unique non-background classes present in the GT mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import config


# ---------------------------------------------------------------------------
# Focal loss helper
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "mean") -> torch.Tensor:
    """Binary sigmoid focal loss."""
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Dice loss on sigmoid predictions.
    inputs, targets: (...,) or (N, HW) flattened.
    """
    prob = inputs.sigmoid()
    numerator = 2 * (prob * targets).sum(-1)
    denominator = prob.sum(-1) + targets.sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)


# ---------------------------------------------------------------------------
# Build per-image targets from semantic segmentation masks
# ---------------------------------------------------------------------------

def build_targets_from_semantic(semantic_masks: torch.Tensor, num_classes: int):
    """
    Convert a batch of semantic segmentation masks to per-image instance targets.

    Args:
        semantic_masks: (B, H, W) int64, values 0=ignore, 1–num_classes=classes
        num_classes: number of semantic classes (7 for LoveDA)

    Returns:
        List of B dicts, each {'labels': (K,), 'masks': (K, H, W) bool}
    """
    targets = []
    for sem in semantic_masks:  # sem: (H, W)
        unique_classes = sem.unique()
        unique_classes = unique_classes[unique_classes > 0]  # drop ignore (0)
        if len(unique_classes) == 0:
            targets.append({"labels": torch.zeros(0, dtype=torch.long, device=sem.device),
                             "masks": torch.zeros(0, *sem.shape, dtype=torch.bool,
                                                  device=sem.device)})
            continue
        labels = unique_classes.long()
        masks = torch.stack([sem == c for c in unique_classes])  # (K, H, W) bool
        targets.append({"labels": labels, "masks": masks})
    return targets


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    def __init__(self, cost_cls: float = config.LOSS_WEIGHT_CLS,
                 cost_focal: float = config.LOSS_WEIGHT_FOCAL,
                 cost_dice: float = config.LOSS_WEIGHT_DICE):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """
        Args:
            outputs: dict with
                'pred_logits': (B, Q, C+1)
                'pred_masks' : (B, Q, H', W')
            targets: list of B dicts {'labels': (K,), 'masks': (K, H, W)}

        Returns:
            List of (row_ind, col_ind) index pairs per image.
        """
        B, Q, _ = outputs["pred_logits"].shape
        pred_logits = outputs["pred_logits"]   # (B, Q, C+1)
        pred_masks  = outputs["pred_masks"]    # (B, Q, H', W')

        indices = []
        for b in range(B):
            tgt = targets[b]
            K = len(tgt["labels"])
            if K == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                 torch.tensor([], dtype=torch.long)))
                continue

            # Class cost: negative softmax probability of target class
            logits_b = pred_logits[b].softmax(-1)  # (Q, C+1)
            cls_cost = -logits_b[:, tgt["labels"] - 1]  # (Q, K) — 0-indexed classes

            # Downsample GT masks to match prediction resolution
            h, w = pred_masks.shape[-2:]
            gt_masks = tgt["masks"].float()  # (K, H, W)
            gt_masks_down = F.interpolate(gt_masks.unsqueeze(0), size=(h, w),
                                          mode="nearest").squeeze(0)  # (K, h, w)

            pred_m = pred_masks[b]  # (Q, h, w)

            # Focal cost: (Q, K)
            focal_cost = torch.stack([
                sigmoid_focal_loss(pred_m.flatten(1),
                                   gt_masks_down[k].flatten().unsqueeze(0).expand(Q, -1),
                                   reduction="none").mean(-1)
                for k in range(K)
            ], dim=1)

            # Dice cost: (Q, K)
            dice_cost = torch.stack([
                dice_loss(pred_m.flatten(1),
                          gt_masks_down[k].flatten().unsqueeze(0).expand(Q, -1))
                for k in range(K)
            ], dim=1)

            cost_matrix = (self.cost_cls * cls_cost +
                           self.cost_focal * focal_cost +
                           self.cost_dice * dice_cost)  # (Q, K)

            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            indices.append((torch.tensor(row_ind, dtype=torch.long),
                            torch.tensor(col_ind, dtype=torch.long)))
        return indices


# ---------------------------------------------------------------------------
# Set Criterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 weight_cls: float = config.LOSS_WEIGHT_CLS,
                 weight_focal: float = config.LOSS_WEIGHT_FOCAL,
                 weight_dice: float = config.LOSS_WEIGHT_DICE):
        super().__init__()
        self.num_classes = num_classes
        self.weight_cls = weight_cls
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.matcher = HungarianMatcher(weight_cls, weight_focal, weight_dice)

        # Class weights: up-weight foreground
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = 0.1  # "no object" class gets lower weight
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, all_layer_outputs: list, semantic_masks: torch.Tensor):
        """
        Args:
            all_layer_outputs: list of dicts per decoder layer, each with
                'pred_logits': (B, Q, C+1)
                'pred_masks' : (B, Q, H', W')
            semantic_masks: (B, H, W) int64 GT

        Returns:
            Total scalar loss and dict of components.
        """
        targets = build_targets_from_semantic(semantic_masks, self.num_classes)

        total_loss = 0.0
        loss_dict = {"loss_cls": 0.0, "loss_focal": 0.0, "loss_dice": 0.0}

        for layer_idx, outputs in enumerate(all_layer_outputs):
            l_cls, l_focal, l_dice = self._compute_layer_loss(outputs, targets)
            total_loss = total_loss + l_cls + l_focal + l_dice
            loss_dict["loss_cls"] = loss_dict["loss_cls"] + l_cls.item()
            loss_dict["loss_focal"] = loss_dict["loss_focal"] + l_focal.item()
            loss_dict["loss_dice"] = loss_dict["loss_dice"] + l_dice.item()

        n_layers = len(all_layer_outputs)
        loss_dict = {k: v / n_layers for k, v in loss_dict.items()}
        return total_loss, loss_dict

    def _compute_layer_loss(self, outputs: dict, targets: list):
        B, Q, _ = outputs["pred_logits"].shape
        indices = self.matcher(outputs, targets)

        # --- Classification loss ---
        # Build target class index for all queries (default = no-object)
        target_classes = torch.full((B, Q), self.num_classes,
                                    dtype=torch.long,
                                    device=outputs["pred_logits"].device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]  # 1-indexed
            # Shift to 0-indexed for the loss head (label 1→0, ..., 7→6)
            target_classes[b, src_idx] -= 1

        # For unmatched queries, keep as num_classes (no-object)
        # Re-map no-object to num_classes (last position in weight vector)
        loss_cls = F.cross_entropy(
            outputs["pred_logits"].reshape(B * Q, -1),
            target_classes.reshape(B * Q),
            weight=self.empty_weight.to(outputs["pred_logits"].device),
        )
        loss_cls = self.weight_cls * loss_cls

        # --- Mask losses ---
        pred_masks = outputs["pred_masks"]  # (B, Q, H', W')
        h, w = pred_masks.shape[-2:]

        focal_losses, dice_losses = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            gt_masks = targets[b]["masks"][tgt_idx].float()  # (M, H, W)
            gt_masks_down = F.interpolate(gt_masks.unsqueeze(0), size=(h, w),
                                          mode="nearest").squeeze(0)  # (M, h, w)
            pred_m = pred_masks[b, src_idx]  # (M, h, w)

            focal_losses.append(
                sigmoid_focal_loss(pred_m.flatten(1), gt_masks_down.flatten(1))
            )
            dice_losses.append(dice_loss(pred_m.flatten(1), gt_masks_down.flatten(1)).mean())

        if focal_losses:
            loss_focal = self.weight_focal * torch.stack(focal_losses).mean()
            loss_dice  = self.weight_dice  * torch.stack(dice_losses).mean()
        else:
            device = pred_masks.device
            loss_focal = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dice  = torch.tensor(0.0, device=device, requires_grad=True)

        return loss_cls, loss_focal, loss_dice
