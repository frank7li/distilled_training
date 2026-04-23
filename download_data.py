"""
Downloads the LoveDA dataset to ./data/loveda.

Usage:
    python download_data.py
"""

import config
from data.loveda_dataset import LoveDAWrapped

print("Downloading LoveDA train split...")
LoveDAWrapped(root=config.DATA_ROOT, split="train", download=True)

print("Downloading LoveDA val split...")
LoveDAWrapped(root=config.DATA_ROOT, split="val", download=True)

print(f"Done. Dataset saved to {config.DATA_ROOT}")
