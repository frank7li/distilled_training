"""
End-to-end inference pipeline: student-teacher selective fallback.

Given an input image, the student model produces a prediction and an aggregate
confidence score.  If the mean confidence exceeds the threshold the student
prediction is kept; otherwise the teacher model is called on the full image
and its prediction is used instead.

Usage:
    python pipeline.py --image path/to/image.png
    python pipeline.py --image path/to/image.png --threshold 0.7
    python pipeline.py --image path/to/image.png --save output.png
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from models.mask2former_student import Mask2FormerStudent
from models.mask2former_teacher import Mask2FormerTeacher
from refine import load_model, _outputs_to_semantic_logits

CLASS_NAMES = [
    "Background", "Building", "Road",
    "Water", "Barren", "Forest", "Agriculture",
]

CLASS_COLORS = [
    (0, 0, 0),        # Background — black
    (255, 0, 0),      # Building — red
    (255, 255, 0),    # Road — yellow
    (0, 0, 255),      # Water — blue
    (128, 128, 128),  # Barren — gray
    (0, 255, 0),      # Forest — green
    (0, 128, 0),      # Agriculture — dark green
]


def preprocess(image_path: str, image_size: int = config.IMAGE_SIZE) -> torch.Tensor:
    """Load an image from disk and return a (1, 3, H, W) normalised tensor."""
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def predict(model, image_tensor: torch.Tensor, device: torch.device):
    """Run a model and return (pred_map, mean_confidence).

    pred_map : (1, H, W) int64, 1-indexed class labels
    mean_conf: float, average per-pixel max softmax probability
    """
    image_tensor = image_tensor.to(device)
    H, W = image_tensor.shape[2], image_tensor.shape[3]

    with torch.no_grad():
        logits = _outputs_to_semantic_logits(model(image_tensor), (H, W))
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)
        pred = pred + 1  # 1-indexed to match LoveDA convention

    return pred, conf.mean().item()


def colorize(pred_map: torch.Tensor) -> np.ndarray:
    """Convert a (H, W) label map to an RGB numpy array."""
    h, w = pred_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[pred_map == cls_id + 1] = color  # classes are 1-indexed
    return rgb


def run_pipeline(image_path: str,
                 student_ckpt: str = "checkpoints/distill_best.pth",
                 teacher_ckpt: str = "checkpoints/swin_s_best.pth",
                 threshold: float = 0.25,
                 device: Optional[torch.device] = None,
                 save_path: Optional[str] = None):
    """Full pipeline: student first, teacher fallback if confidence < threshold."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = load_model(Mask2FormerStudent, student_ckpt, device)
    teacher = load_model(Mask2FormerTeacher, teacher_ckpt, device)

    image_tensor = preprocess(image_path)

    student_pred, student_conf = predict(student, image_tensor, device)

    used_teacher = False
    if student_conf >= threshold:
        final_pred = student_pred
        final_conf = student_conf
    else:
        teacher_pred, teacher_conf = predict(teacher, image_tensor, device)
        final_pred = teacher_pred
        final_conf = teacher_conf
        used_teacher = True

    pred_map = final_pred.squeeze(0).cpu()

    # Print results
    model_used = "teacher" if used_teacher else "student"
    print(f"Image        : {image_path}")
    print(f"Student conf : {student_conf:.4f}  (threshold={threshold})")
    print(f"Model used   : {model_used}")
    print(f"Final conf   : {final_conf:.4f}")

    unique_classes = torch.unique(pred_map).tolist()
    print("Classes found:")
    for cls_id in sorted(unique_classes):
        if 1 <= cls_id <= len(CLASS_NAMES):
            pixel_count = (pred_map == cls_id).sum().item()
            pct = 100.0 * pixel_count / pred_map.numel()
            print(f"  {CLASS_NAMES[cls_id - 1]:>12}: {pct:5.1f}%")

    if save_path:
        rgb = colorize(pred_map)
        Image.fromarray(rgb).save(save_path)
        print(f"Saved to     : {save_path}")

    return pred_map, final_conf, used_teacher


def main():
    parser = argparse.ArgumentParser(
        description="Student-teacher inference pipeline with confidence gating")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--student-ckpt", default="checkpoints/distill_best.pth")
    parser.add_argument("--teacher-ckpt", default="checkpoints/swin_s_best.pth")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Confidence threshold for student fallback (default: 0.25)")
    parser.add_argument("--save", default=None,
                        help="Optional path to save colourised prediction as PNG")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    run_pipeline(
        image_path=args.image,
        student_ckpt=args.student_ckpt,
        teacher_ckpt=args.teacher_ckpt,
        threshold=args.threshold,
        device=device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
