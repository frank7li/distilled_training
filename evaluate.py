"""
Evaluation script for selective teacher refinement (quantiles = 0.9, 0.8, 0.7) on LoveDA.

Usage:
    python evaluate.py [--student-ckpt PATH] [--teacher-ckpt PATH] [--batch-size N] [--num-workers N] [--log-file PATH]
    
Outputs:
    Prints per-class IoU and mean IoU (mIoU) on the validation set
    refine_log.csv       — per-quantile mIoU and time cost

"""

import argparse
import csv
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import config
from data.loveda_dataset import get_datasets
from models.mask2former_student import Mask2FormerStudent
from models.mask2former_teacher import Mask2FormerTeacher
from refine import load_model, refine_batch

NUM_CLASSES = config.NUM_CLASSES  # 7

CLASS_NAMES = [
    "Background", "Building", "Road",
    "Water", "Barren", "Forest", "Agriculture"
]


# Confusion Matrix
def update_confusion_matrix(cm, preds, labels):
    mask = labels != 0
    preds = preds[mask] - 1
    labels = labels[mask] - 1

    preds = preds.clamp(0, NUM_CLASSES - 1)
    labels = labels.clamp(0, NUM_CLASSES - 1)

    idx = labels * NUM_CLASSES + preds
    cm += torch.bincount(idx, minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES)


# IoU
def compute_iou(cm):
    tp = cm.diagonal()
    fn = cm.sum(dim=1) - tp
    fp = cm.sum(dim=0) - tp

    iou = tp.float() / (tp + fn + fp).float().clamp(min=1)
    present = cm.sum(dim=1) > 0
    miou = iou[present].mean().item()

    return iou, miou


# CSV Logging
def save_csv(log_file, results):
    file_exists = False
    try:
        with open(log_file, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                "timestamp", "quantile", "mIoU",
                "total_time", "time_per_image"
            ] + CLASS_NAMES
            writer.writerow(header)

        for r in results:
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                r["quantile"],
                f"{r['miou']:.4f}",
                f"{r['total_time']:.2f}",
                f"{r['time_per_image']:.4f}",
            ] + [f"{x:.4f}" for x in r["iou"]]

            writer.writerow(row)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt", default="checkpoints/distill_best.pth")
    parser.add_argument("--teacher-ckpt", default="checkpoints/swin_s_best.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-file", default="refine_log.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_ds = get_datasets()
    loader = DataLoader(val_ds,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True)

    student = load_model(Mask2FormerStudent, args.student_ckpt, device)
    teacher = load_model(Mask2FormerTeacher, args.teacher_ckpt, device)

    quantiles = [0.9, 0.8, 0.7]

    results = []

    print("\n===== Running Experiments =====")

    for q in quantiles:
        print(f"\n--- Quantile {q} ---")

        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

        start = time.time()
        num_images = 0

        for images, labels in loader:
            preds = refine_batch(images, student, teacher, device, quantile=q)
            update_confusion_matrix(cm, preds.cpu(), labels.cpu())
            num_images += images.size(0)

        total_time = time.time() - start
        time_per_image = total_time / num_images

        iou, miou = compute_iou(cm)

        print(f"mIoU: {miou:.4f}")
        print(f"Time: {total_time:.2f}s | Per image: {time_per_image:.4f}s")

        results.append({
            "quantile": q,
            "miou": miou,
            "iou": iou.tolist(),
            "total_time": total_time,
            "time_per_image": time_per_image
        })

    # Save CSV
    save_csv(args.log_file, results)
    print(f"\nSaved results to {args.log_file}")

    # Comparison Table
    print("\n===== Summary Table =====")
    print(f"{'Quantile':>10} | {'mIoU':>6} | {'Time(s)':>8} | {'Time/img':>10}")
    print("-" * 44)

    for r in results:
        print(f"{r['quantile']:>10} | "
              f"{r['miou']:.4f} | "
              f"{r['total_time']:.2f} | "
              f"{r['time_per_image']:.4f}")

    print("\n===== Per-class IoU (last run) =====")
    last = results[-1]

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>12}: {last['iou'][i]:.4f}")


if __name__ == "__main__":
    main()