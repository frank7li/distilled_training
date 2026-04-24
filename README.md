# CSCI 566 Project: Mask2Former Knowledge Distillation on LoveDA


---

## Experiments & Results

### Experiment 1: ResNet-18 Supervised Baseline (Midterm)

- Model: Mask2Former + ResNet-18, 30 epochs, batch size 8, full dataset
- Script: `train.py`
- Checkpoint: `checkpoints/best.pth`
- **Val mIoU: 0.4154** (epoch 18)

Per-class IoU:

| Class | IoU |
|-------|-----|
| Background | 0.4702 |
| Building | 0.4795 |
| Road | 0.4444 |
| Water | 0.5476 |
| Barren | 0.1888 |
| Forest | 0.3403 |
| Agriculture | 0.4373 |

---

### Experiment 2: Swin-S Teacher

- Model: Mask2Former + Swin-S (ImageNet pretrained backbone)
- Script: `train_teacher.py --full`
- Epochs: 30, batch size 4, full dataset
- Optimizer: AdamW, backbone lr=1e-4, decoder lr=1e-3, weight decay=0.05, CosineAnnealingLR
- Checkpoint: `checkpoints/swin_s_best.pth`
- **Val mIoU: 0.5265** (epoch 11) — reported in midterm: 0.5282 ✓

---

### Experiment 3: Knowledge Distillation

- Student: Mask2Former + ResNet-18 (ImageNet pretrained)
- Teacher: Mask2Former + Swin-S (frozen, loaded from `checkpoints/swin_s_best.pth`)
- Script: `train_distill.py --teacher-ckpt checkpoints/swin_s_best.pth --full`
- Epochs: 40, full dataset
- Loss: `L_total = L_supervised + 1.0 * L_soft_kd + 0.5 * L_feat_kd` (temperature T=4)
- Checkpoint: `checkpoints/distill_best.pth`
- **Val mIoU: 0.4855** (epoch 3)

| Model | mIoU |
|-------|------|
| ResNet-18 supervised (midterm) | 0.4154 |
| ResNet-18 distilled (ours) | **0.4855** |
| Swin-S teacher | 0.5265 |

Distillation closed **~62% of the teacher-student gap** (+0.070 over supervised baseline).

---


### Experiment 4: Selective Teacher Refinement 
A region is considered uncertain if its entropy exceeds a threshold τ
| Quantile | mIoU | Time(s) | Time/img(s) |
|----------|------|---------|-------------|
| 0.9 | 0.4892 | 66.48 | 0.0398 |
| 0.8 | 0.4905 | 73.63 | 0.0441 |
| 0.7 | **0.4940** | 79.01 | 0.0473 |

---

## Saved Weights

| File | Model | Val mIoU |
|------|-------|---------|
| `checkpoints/swin_s_best.pth` | Mask2Former + Swin-S (teacher) | 0.5265 |
| `checkpoints/distill_best.pth` | Mask2Former + ResNet-18 (distilled) | 0.4855 |
| `checkpoints/best.pth` | Mask2Former + ResNet-18 (supervised) | 0.4154 |


Load the weights for this:

```python
from models.mask2former_student import Mask2FormerStudent
from models.mask2former_teacher import Mask2FormerTeacher
import torch

student = Mask2FormerStudent(pretrained_backbone=False)
student.load_state_dict(torch.load("checkpoints/distill_best.pth")["model_state_dict"])

teacher = Mask2FormerTeacher(pretrained_backbone=False)
teacher.load_state_dict(torch.load("checkpoints/swin_s_best.pth")["model_state_dict"])
```