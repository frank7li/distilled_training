"""
Knowledge distillation training: Mask2Former + Swin-S (teacher, frozen)
→ Mask2Former + ResNet-18 (student).

Distillation losses:
  L_total = L_supervised + lambda_kd * L_soft_kd + lambda_feat * L_feat_kd

  L_supervised : standard Mask2Former loss (focal + dice + CE) against GT
  L_soft_kd    : KL divergence between teacher/student output class distributions
                 averaged over all decoder layers
  L_feat_kd    : MSE between teacher and student mask_features (after a learned
                 projection since channel dims differ: teacher=256, student=256)

Usage:
    # With a Swin-S teacher checkpoint:
    python train_distill.py --teacher-ckpt checkpoints/swin_s_best.pth

    # Without a teacher checkpoint (supervised-only, for debugging):
    python train_distill.py --no-teacher

    # Full dataset, more epochs:
    python train_distill.py --teacher-ckpt checkpoints/swin_s_best.pth --full --epochs 50

Outputs:
    checkpoints/distill_best.pth   — best val mIoU student checkpoint
    distill_log.csv                — per-epoch losses and mIoU
"""

import argparse
import os
import csv
import random
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from data.loveda_dataset import get_datasets
from models.mask2former_teacher import Mask2FormerTeacher
from models.mask2former_student import Mask2FormerStudent
from losses import SetCriterion
from utils import predictions_to_semantic_map, compute_miou


# ---------------------------------------------------------------------------
# Distillation loss helpers
# ---------------------------------------------------------------------------

def soft_kd_loss(student_outputs: list, teacher_outputs: list,
                 temperature: float = 4.0) -> torch.Tensor:
    """
    KL divergence between teacher and student class probability distributions,
    averaged over all decoder layers.

    Both outputs are lists of dicts with 'pred_logits': (B, Q, C+1).
    Temperature scaling softens the distributions so the student learns
    from the teacher's relative class confidences, not just hard peaks.
    """
    total = torch.tensor(0.0, device=student_outputs[0]["pred_logits"].device)
    for s_out, t_out in zip(student_outputs, teacher_outputs):
        s_log = F.log_softmax(s_out["pred_logits"] / temperature, dim=-1)  # (B, Q, C+1)
        t_prob = F.softmax(t_out["pred_logits"] / temperature, dim=-1)     # (B, Q, C+1)
        # KL(T || S) averaged over batch and queries; scale by T^2 per Hinton et al.
        kl = F.kl_div(s_log, t_prob, reduction="batchmean")
        total = total + kl * (temperature ** 2)
    return total / len(student_outputs)


def feature_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                    projector: nn.Module) -> torch.Tensor:
    """
    MSE between projected student mask_features and teacher mask_features.
    Both are (B, hidden_dim, H/4, W/4) — same shape since hidden_dim=256 for both.
    The projector adapts student features before comparison.
    """
    student_proj = projector(student_feat)
    # Detach teacher features — we only want to push the student toward them
    return F.mse_loss(student_proj, teacher_feat.detach())


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_student_optimizer(student: Mask2FormerStudent, projector: nn.Module) -> AdamW:
    backbone_params = list(student.backbone.parameters())
    other_params = [p for p in student.parameters()
                    if not any(p is bp for bp in backbone_params)]
    return AdamW(
        [
            {"params": backbone_params,      "lr": config.LR_BACKBONE},
            {"params": other_params,         "lr": config.LR_DECODER},
            {"params": projector.parameters(), "lr": config.LR_DECODER},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def load_teacher(ckpt_path: Optional[str], device: torch.device) -> Optional[Mask2FormerTeacher]:
    """Load frozen teacher. Returns None if no checkpoint provided."""
    if ckpt_path is None:
        return None
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    teacher = Mask2FormerTeacher(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        pretrained_backbone=False,  # weights come from checkpoint
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"Teacher loaded from {ckpt_path}  "
          f"(epoch {ckpt.get('epoch', '?')}, mIoU {ckpt.get('val_miou', '?'):.4f})")
    return teacher


def teacher_forward(teacher: Mask2FormerTeacher, images: torch.Tensor):
    """Run teacher forward pass + extract mask_features without grad."""
    features = teacher.backbone(images)
    mask_features, multi_scale = teacher.pixel_decoder(features)
    layer_outputs = teacher.transformer_decoder(multi_scale, mask_features)
    outputs = [
        {"pred_logits": cls_logits, "pred_masks": mask_logits}
        for cls_logits, mask_logits in layer_outputs
    ]
    return outputs, mask_features


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(student, teacher, projector, loader, criterion,
                    optimizer, device, lambda_kd, lambda_feat, temperature):
    student.train()
    projector.train()

    total_loss = total_sup = total_kd = total_feat = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        # Student forward (returns outputs + mask_features for feat KD)
        student_outputs, student_feat = student.forward_features(images)

        # Supervised loss (same Mask2Former criterion as baseline)
        loss_sup, _ = criterion(student_outputs, masks)

        loss_kd   = torch.tensor(0.0, device=device)
        loss_feat = torch.tensor(0.0, device=device)

        if teacher is not None:
            with torch.no_grad():
                teacher_outputs, teacher_feat = teacher_forward(teacher, images)

            loss_kd   = soft_kd_loss(student_outputs, teacher_outputs, temperature)
            loss_feat = feature_kd_loss(student_feat, teacher_feat, projector)

        loss = loss_sup + lambda_kd * loss_kd + lambda_feat * loss_feat

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(projector.parameters()), max_norm=0.1
        )
        optimizer.step()

        total_loss += loss.item()
        total_sup  += loss_sup.item()
        total_kd   += loss_kd.item()
        total_feat += loss_feat.item()

    n = len(loader)
    return total_loss / n, total_sup / n, total_kd / n, total_feat / n


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = config.NUM_CLASSES):
    model.eval()
    all_preds, all_gts = [], []
    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        all_outputs = model(images)
        last_output = all_outputs[-1]
        pred_map = predictions_to_semantic_map(
            last_output, image_size=(images.shape[2], images.shape[3]),
            num_classes=num_classes,
        )
        all_preds.append(pred_map.cpu())
        all_gts.append(masks)

    pred_cat = torch.cat(all_preds, dim=0)
    gt_cat   = torch.cat(all_gts,   dim=0)
    return compute_miou(pred_cat, gt_cat, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distillation training for Mask2Former")
    parser.add_argument("--teacher-ckpt", type=str, default=None,
                        help="Path to frozen Swin-S teacher checkpoint (.pth)")
    parser.add_argument("--no-teacher", action="store_true",
                        help="Run supervised-only (no distillation), useful for debugging")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Training epochs (default: 40)")
    parser.add_argument("--full", action="store_true",
                        help="Use full training set instead of subset")
    parser.add_argument("--lambda-kd", type=float, default=1.0,
                        help="Weight for soft KD loss (default: 1.0)")
    parser.add_argument("--lambda-feat", type=float, default=0.5,
                        help="Weight for feature KD loss (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Softmax temperature for soft KD (default: 4.0)")
    args = parser.parse_args()

    use_teacher = (not args.no_teacher) and (args.teacher_ckpt is not None)
    num_epochs = args.epochs
    subset_fraction = 1.0 if args.full else config.SUBSET_FRACTION

    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_ds, val_ds = get_datasets(root=config.DATA_ROOT, download=True,
                                    subset_fraction=subset_fraction)
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Teacher (frozen)
    teacher = None
    if use_teacher:
        teacher = load_teacher(args.teacher_ckpt, device)
    elif not args.no_teacher:
        print("Warning: no --teacher-ckpt provided. Running supervised-only.")

    # Student (ResNet-18)
    student = Mask2FormerStudent(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        pretrained_backbone=True,
    ).to(device)

    # Feature projector: adapts student mask_features before comparing to teacher's.
    # Both teacher and student use hidden_dim=256, so this is a small 1x1 conv refinement.
    projector = nn.Sequential(
        nn.Conv2d(config.HIDDEN_DIM, config.HIDDEN_DIM, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(config.HIDDEN_DIM, config.HIDDEN_DIM, kernel_size=1),
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = SetCriterion(num_classes=config.NUM_CLASSES).to(device)
    optimizer = build_student_optimizer(student, projector)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = "distill_log.csv"
    best_miou = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "loss_supervised",
                         "loss_kd", "loss_feat", "val_miou"])

    print(f"\nDistillation config:")
    print(f"  Teacher: {'enabled' if teacher else 'disabled (supervised only)'}")
    print(f"  lambda_kd={args.lambda_kd}, lambda_feat={args.lambda_feat}, "
          f"temperature={args.temperature}")
    print(f"  Epochs: {num_epochs}, Subset: {subset_fraction}\n")

    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        t0 = time.time()

        total_loss, loss_sup, loss_kd, loss_feat = train_one_epoch(
            student, teacher, projector, train_loader, criterion,
            optimizer, device,
            lambda_kd=args.lambda_kd if teacher else 0.0,
            lambda_feat=args.lambda_feat if teacher else 0.0,
            temperature=args.temperature,
        )
        val_miou = evaluate(student, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        eta_secs = (sum(epoch_times) / len(epoch_times)) * (num_epochs - epoch)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))

        print(f"  total={total_loss:.4f}  sup={loss_sup:.4f}  "
              f"kd={loss_kd:.4f}  feat={loss_feat:.4f}  "
              f"val_mIoU={val_miou:.4f}  ETA={eta_str}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{total_loss:.6f}", f"{loss_sup:.6f}",
                             f"{loss_kd:.6f}", f"{loss_feat:.6f}", f"{val_miou:.6f}"])

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "distill_config": {
                    "lambda_kd": args.lambda_kd,
                    "lambda_feat": args.lambda_feat,
                    "temperature": args.temperature,
                    "teacher_ckpt": args.teacher_ckpt,
                },
            }, "checkpoints/distill_best.pth")
            print(f"  New best mIoU: {best_miou:.4f} -> checkpoints/distill_best.pth")

    print(f"\nDone. Best val mIoU: {best_miou:.4f}  Log: {log_path}")


if __name__ == "__main__":
    main()