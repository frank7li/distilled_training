"""
Training script for Mask2Former + ResNet-18 on 10% LoveDA subset.

Usage:
    python train.py [--epochs N]

Outputs:
    checkpoints/best.pth   — best val mIoU checkpoint
    training_log.csv       — per-epoch loss and mIoU
"""

import argparse
import os
import csv
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from data.loveda_dataset import get_datasets
from models import Mask2Former
from losses import SetCriterion
from utils import predictions_to_semantic_map, compute_miou


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


def build_optimizer(model: Mask2Former) -> AdamW:
    backbone_params = list(model.backbone.parameters())
    other_params = [p for p in model.parameters()
                    if not any(p is bp for bp in backbone_params)]
    return AdamW(
        [
            {"params": backbone_params, "lr": config.LR_BACKBONE},
            {"params": other_params,   "lr": config.LR_DECODER},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        all_outputs = model(images)
        loss, _ = criterion(all_outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs (overrides config.EPOCHS)")
    parser.add_argument("--full", action="store_true",
                        help="Use full training set instead of 10%% subset")
    args = parser.parse_args()
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

    # Model
    model = Mask2Former(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        pretrained_backbone=True,
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = SetCriterion(num_classes=config.NUM_CLASSES).to(device)
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = "training_log.csv"
    best_miou = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_miou"])

    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_miou   = evaluate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta_secs = avg_time * (num_epochs - epoch)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))

        print(f"  train_loss={train_loss:.4f}  val_mIoU={val_miou:.4f}  "
              f"epoch_time={epoch_time:.1f}s  ETA={eta_str}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_miou:.6f}"])

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
            }, "checkpoints/best.pth")
            print(f"  ✓ New best mIoU: {best_miou:.4f} — checkpoint saved")

    print(f"\nTraining complete ({num_epochs} epoch(s)). Best val mIoU: {best_miou:.4f}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
