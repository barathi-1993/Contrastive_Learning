"""utils/dataset.py — Dataset classes for contrastive pretraining and downstream tasks."""
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation factory
# ─────────────────────────────────────────────────────────────────────────────
def build_contrastive_transform(cfg) -> transforms.Compose:
    """Build the standard two-view augmentation pipeline from config."""
    a = cfg.Augmentation
    t = [
        transforms.RandomResizedCrop(
            cfg.Dataset.image_size,
            scale=tuple(a.crop_scale)
        ) if a.random_resized_crop else transforms.Resize((cfg.Dataset.image_size, cfg.Dataset.image_size)),
        transforms.RandomHorizontalFlip() if a.horizontal_flip else None,
        transforms.RandomApply(
            [transforms.ColorJitter(
                brightness=a.color_jitter_strength,
                contrast=a.color_jitter_strength,
                saturation=a.color_jitter_strength,
                hue=a.color_jitter_hue
            )], p=a.color_jitter_prob
        ) if a.color_jitter else None,
        transforms.RandomGrayscale(p=a.grayscale_prob),
        transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=int(0.1 * cfg.Dataset.image_size) | 1,
                sigma=tuple(a.gaussian_blur_sigma)
            )], p=a.gaussian_blur_prob
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(a.normalize_mean), std=list(a.normalize_std)),
    ]
    return transforms.Compose([x for x in t if x is not None])

def build_eval_transform(cfg) -> transforms.Compose:
    """Deterministic eval/inference transform."""
    a = cfg.Augmentation
    sz = cfg.Dataset.image_size
    return transforms.Compose([
        transforms.Resize(int(sz * 1.14)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(a.normalize_mean), std=list(a.normalize_std)),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def _collect_images(root: str) -> List[dict]:
    """Recursively collect all image files under root."""
    root = os.path.abspath(root)
    if not os.path.exists(root):
        print(f"❌ Path does not exist: {root}")
        sys.exit(1)
    records = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in VALID_EXTS:
                records.append({
                    "path":   os.path.join(dirpath, fname),
                    "folder": os.path.basename(os.path.dirname(dirpath)),
                    "part":   os.path.basename(dirpath),
                    "label":  os.path.basename(dirpath),   # default: folder = class
                })
    print(f"✅ Collected {len(records)} images from {root}")
    return records

class ContrastiveDataset(Dataset):
    """
    Returns two augmented views of the same image for contrastive pretraining.
    Supports multi-crop (DINO / SwAV) when n_views > 2.
    """
    def __init__(
        self,
        root: str,
        transform: Callable,
        n_views: int = 2,
        local_transform: Optional[Callable] = None,
        n_local_views: int = 0,
    ):
        self.records        = _collect_images(root)
        self.transform      = transform
        self.n_views        = n_views
        self.local_transform = local_transform
        self.n_local_views  = n_local_views

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception as e:
            print(f"⚠️  Cannot open {rec['path']}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        global_views = [self.transform(img) for _ in range(self.n_views)]
        local_views  = []
        if self.local_transform and self.n_local_views > 0:
            local_views = [self.local_transform(img) for _ in range(self.n_local_views)]

        views = global_views + local_views
        return views, rec["folder"], rec["part"]

class EvalDataset(Dataset):
    """
    Single-view dataset for embedding extraction, classification, and clustering.
    Labels are inferred from the immediate parent folder name unless a label_map
    dict {filename_stem: int} is provided.
    """
    def __init__(
        self,
        root: str,
        transform: Callable,
        label_map: Optional[dict] = None,
    ):
        self.records   = _collect_images(root)
        self.transform = transform

        # Build class → id mapping
        if label_map:
            self.label_map = label_map
        else:
            classes        = sorted({r["label"] for r in self.records})
            self.label_map = {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple:
        rec = self.records[idx]
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception as e:
            print(f"⚠️  Cannot open {rec['path']}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        tensor = self.transform(img)
        label  = self.label_map.get(rec["label"], -1)
        return tensor, label, rec["path"], rec["folder"], rec["part"]
