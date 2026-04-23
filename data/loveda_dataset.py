import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchgeo.datasets import LoveDA

import config


def get_transforms(split: str):
    """Return image + mask transforms for a given split."""
    size = config.IMAGE_SIZE
    if split == "train":
        img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    mask_tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    return img_tf, mask_tf


class LoveDAWrapped(Dataset):
    """
    Wraps torchgeo LoveDA to return (image_tensor, mask_tensor) pairs.

    LoveDA mask values: 0 = no-data/ignore, 1-7 = semantic classes.
    We keep them as-is; the criterion ignores index 0.
    """

    def __init__(self, root: str, split: str, download: bool = True):
        # torchgeo LoveDA uses scene="urban"/"rural", not train/val split directly.
        # The dataset constructor accepts split="train"/"val"/"test".
        self.base = LoveDA(root=root, split=split, scene=["urban", "rural"],
                           download=download)
        self.img_tf, self.mask_tf = get_transforms(split)
        self.split = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # torchgeo returns dict with 'image' (C, H, W) uint8 tensor and 'mask' (H, W) tensor
        image = sample["image"]  # torch.uint8, shape (C, H, W)
        mask = sample["mask"]    # torch.int64 or torch.uint8, shape (H, W) or (1, H, W)

        # Convert image tensor → PIL for transforms, then back
        from PIL import Image as PILImage
        img_np = image[:3].permute(1, 2, 0).numpy().astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)
        image_t = self.img_tf(pil_img)  # (3, H, W) float

        # Mask handling
        if mask.dim() == 3:
            mask = mask[0]  # (H, W)
        mask = mask.long()
        mask_pil = PILImage.fromarray(mask.numpy().astype(np.uint8))
        mask_pil = self.mask_tf(mask_pil)
        mask_t = torch.from_numpy(np.array(mask_pil)).long()  # (H, W)

        return image_t, mask_t


def get_datasets(root: str = config.DATA_ROOT, download: bool = True,
                 subset_fraction: float = config.SUBSET_FRACTION):
    """
    Returns (train_dataset, val_dataset).
    subset_fraction=1.0 uses the full training set; <1.0 uses a random subset.
    """
    full_train = LoveDAWrapped(root=root, split="train", download=download)
    val_ds = LoveDAWrapped(root=root, split="val", download=download)

    if subset_fraction >= 1.0:
        train_ds = full_train
    else:
        rng = random.Random(config.SEED)
        indices = list(range(len(full_train)))
        rng.shuffle(indices)
        n_subset = max(1, int(len(indices) * subset_fraction))
        subset_indices = sorted(indices[:n_subset])
        train_ds = Subset(full_train, subset_indices)

    return train_ds, val_ds
