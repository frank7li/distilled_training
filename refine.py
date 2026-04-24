"""
Selective teacher refinement for semantic segmentation on LoveDA.

Usage:
    python refine.py \
        --student-ckpt checkpoints/distill_best.pth \
        --teacher-ckpt checkpoints/swin_s_best.pth

Outputs:
    Inference-only script (no files saved by default)
    Returns refined predictions for all images in the dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mask2former_student import Mask2FormerStudent
from models.mask2former_teacher import Mask2FormerTeacher

PATCH_SIZE = 128


# Model loading
def load_model(model_cls: type, ckpt_path: str, device: torch.device) -> nn.Module:
    model = model_cls(pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# Output conversion
def _outputs_to_semantic_logits(outputs, image_size):
    last = outputs[-1]
    cls_prob = last["pred_logits"].softmax(-1)[..., :-1]
    masks_up = F.interpolate(last["pred_masks"], size=image_size,
                             mode="bilinear", align_corners=False)
    mask_prob = masks_up.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", cls_prob, mask_prob)


# Patch extraction
def extract_patches(images, patch_size):
    B, C, H, W = images.shape
    n_h, n_w = H // patch_size, W // patch_size

    imgs = images[:, :, :n_h * patch_size, :n_w * patch_size]

    patches = (imgs
               .view(B, C, n_h, patch_size, n_w, patch_size)
               .permute(0, 2, 4, 1, 3, 5))

    return patches, (n_h, n_w)

# Uncertainty
def get_uncertainty(logits):
    probs = logits.softmax(dim=1)
    return -(probs * (probs + 1e-9).log()).sum(dim=1)

def get_uncertain_indices(entropy, grid, patch_size, threshold):
    B = entropy.shape[0]
    n_h, n_w = grid

    indices = []
    for b in range(B):
        for i in range(n_h):
            for j in range(n_w):
                patch = entropy[b,
                                i * patch_size:(i + 1) * patch_size,
                                j * patch_size:(j + 1) * patch_size]

                if patch.mean() > threshold:
                    indices.append((b, i, j))

    return indices


# Refinement
def refine_batch(images, student, teacher, device, quantile=0.7):
    B, _, H, W = images.shape
    images = images.to(device, non_blocking=True)

    with torch.no_grad():
        # Student full inference
        student_logits = _outputs_to_semantic_logits(student(images), (H, W))
        student_probs = student_logits.softmax(dim=1)

        student_conf, student_pred = student_probs.max(dim=1)
        student_pred = student_pred + 1  # 1-indexed

        # Uncertainty
        entropy = get_uncertainty(student_logits)
        threshold = torch.quantile(entropy.flatten(), quantile).item()

        # Patch selection
        patches, grid = extract_patches(images, PATCH_SIZE)
        uncertain = get_uncertain_indices(entropy, grid, PATCH_SIZE, threshold)

        if not uncertain:
            return student_pred

        # Gather patches
        patch_batch = torch.stack([patches[b, i, j] for (b, i, j) in uncertain])

        # Teacher inference
        teacher_logits = _outputs_to_semantic_logits(
            teacher(patch_batch), (PATCH_SIZE, PATCH_SIZE)
        )
        teacher_probs = teacher_logits.softmax(dim=1)

        teacher_conf, teacher_pred = teacher_probs.max(dim=1)
        teacher_pred = teacher_pred + 1

        # Merge predictions (confidence-based)
        refined = student_pred.clone()

        for k, (b, i, j) in enumerate(uncertain):
            r0, r1 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            c0, c1 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE

            s_conf_patch = student_conf[b, r0:r1, c0:c1]
            t_conf_patch = teacher_conf[k]

            replace_mask = t_conf_patch > s_conf_patch

            refined_patch = refined[b, r0:r1, c0:c1]
            refined_patch[replace_mask] = teacher_pred[k][replace_mask]

            refined[b, r0:r1, c0:c1] = refined_patch

    return refined