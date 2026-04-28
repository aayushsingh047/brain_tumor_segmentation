"""
BraTS Dataset Module
Handles loading and preprocessing of BraTS .nii files for training.

BraTS dataset structure (per patient folder):
    <case_id>/
        t1.nii  (or t1.nii.gz)
        t1ce.nii
        t2.nii
        flair.nii
        seg.nii   ← ground truth labels

Label mapping (BraTS convention):
    0 = Background
    1 = Necrotic Core / Non-enhancing tumor
    2 = Edema
    4 = Enhancing Tumor   (NOTE: BraTS uses 4, not 3)

We remap label 4 → 3 so output is 0-3 (4 classes).
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    """Load a single .nii / .nii.gz file → float32 numpy array."""
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


def normalise_volume(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalisation restricted to non-zero (brain) voxels.
    Zero voxels (background / skull-stripped) stay at 0.
    """
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std  = volume[mask].std() + 1e-8
    out  = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / std
    return out


def resize_volume(volume: np.ndarray,
                  target_shape=(128, 128, 128),
                  order: int = 1) -> np.ndarray:
    """
    Resize a 3-D volume to target_shape using scipy zoom.
    order=1  → bilinear  (for image data)
    order=0  → nearest   (for label maps – no spurious classes)
    """
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order).astype(np.float32)


def remap_brats_labels(seg: np.ndarray) -> np.ndarray:
    """
    BraTS ground-truth uses labels {0, 1, 2, 4}.
    Remap 4 → 3 so the network sees {0, 1, 2, 3} (4 classes).
    """
    out = seg.copy().astype(np.int64)
    out[seg == 4] = 3
    return out


# ─────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the BraTS Brain Tumor Segmentation challenge.

    Expects a root directory where each sub-folder is one patient case:

        root/
            BraTS20_Training_001/
                BraTS20_Training_001_t1.nii.gz
                BraTS20_Training_001_t1ce.nii.gz
                BraTS20_Training_001_t2.nii.gz
                BraTS20_Training_001_flair.nii.gz
                BraTS20_Training_001_seg.nii.gz
            BraTS20_Training_002/
                ...

    The class also supports a simpler layout where files are just named
    t1.nii, t1ce.nii, t2.nii, flair.nii, seg.nii inside each sub-folder.

    Args:
        root_dir    : path to dataset root
        target_shape: (D, H, W) to resize all volumes to
        augment     : whether to apply random flips during training
        mode        : 'train' | 'val' | 'test'  (test → no label returned)
    """

    MODALITIES = ["t1", "t1ce", "t2", "flair"]

    def __init__(self,
                 root_dir: str,
                 target_shape=(128, 128, 128),
                 augment: bool = False,
                 mode: str = "train"):
        self.root_dir     = Path(root_dir)
        self.target_shape = target_shape
        self.augment      = augment
        self.mode         = mode

        # Collect patient case folders
        self.cases = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        if len(self.cases) == 0:
            raise ValueError(f"No patient folders found in {root_dir}")

        logger.info(f"[BraTSDataset] Found {len(self.cases)} cases "
                    f"(mode={mode}, augment={augment})")

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _find_file(self, case_dir: Path, suffix: str) -> str:
        """
        Locate a modality file inside case_dir.
        Supports both naming conventions:
          1. <case_id>_<suffix>.nii.gz
          2. <suffix>.nii  /  <suffix>.nii.gz
        """
        # Style 1: prefix_suffix.nii.gz  (official BraTS naming)
        for ext in [".nii.gz", ".nii"]:
            pattern = str(case_dir / f"*_{suffix}{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        # Style 2: bare suffix.nii.gz
        for ext in [".nii.gz", ".nii"]:
            p = case_dir / f"{suffix}{ext}"
            if p.exists():
                return str(p)

        raise FileNotFoundError(
            f"Could not find '{suffix}' modality in {case_dir}"
        )

    def _load_case(self, case_dir: Path):
        """
        Load all 4 modalities + seg for one patient.
        Returns:
            image : np.float32 (4, D, H, W) – normalised, resized
            label : np.int64   (D, H, W)    – remapped to 0-3, resized
        """
        modality_volumes = []
        for mod in self.MODALITIES:
            path   = self._find_file(case_dir, mod)
            vol    = load_nifti(path)           # (H, W, D) or (D, H, W) – nib default
            vol    = resize_volume(vol, self.target_shape, order=1)
            vol    = normalise_volume(vol)
            modality_volumes.append(vol)

        image = np.stack(modality_volumes, axis=0)  # (4, D, H, W)

        if self.mode != "test":
            seg_path = self._find_file(case_dir, "seg")
            seg      = load_nifti(seg_path)
            seg      = resize_volume(seg, self.target_shape, order=0)
            seg      = remap_brats_labels(seg).astype(np.int64)
        else:
            seg = None

        return image, seg

    def _augment(self, image: np.ndarray, label: np.ndarray):
        """
        Random flips along each spatial axis.
        image : (4, D, H, W)
        label : (D, H, W)
        """
        for axis in range(1, 4):          # skip channel axis
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis - 1).copy()
        return image, label

    # ──────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        image, label = self._load_case(case_dir)

        if self.augment and label is not None:
            image, label = self._augment(image, label)

        image_tensor = torch.from_numpy(image).float()

        if label is not None:
            label_tensor = torch.from_numpy(label).long()
            return image_tensor, label_tensor
        else:
            return image_tensor


# ─────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────

def get_dataloaders(root_dir: str,
                    target_shape=(128, 128, 128),
                    batch_size: int = 2,
                    val_split: float = 0.2,
                    num_workers: int = 2,
                    seed: int = 42) -> tuple:
    """
    Build train / validation DataLoaders from a single root directory.

    Args:
        root_dir    : BraTS root folder
        target_shape: resize target for all volumes
        batch_size  : samples per batch
        val_split   : fraction of data for validation
        num_workers : DataLoader workers
        seed        : random seed for reproducible split

    Returns:
        train_loader, val_loader
    """
    np.random.seed(seed)

    # Full dataset (no augmentation – we set it per-subset below)
    full_dataset = BraTSDataset(root_dir, target_shape=target_shape,
                                augment=False, mode="train")

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    indices  = np.random.permutation(n_total)
    train_idx = indices[:n_train].tolist()
    val_idx   = indices[n_train:].tolist()

    # Subset wrappers
    from torch.utils.data import Subset

    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(full_dataset, val_idx)

    # For the training subset we want augmentation – patch __getitem__
    # The cleanest way: create a second dataset object with augment=True
    train_dataset = BraTSDataset(root_dir, target_shape=target_shape,
                                  augment=True, mode="train")
    train_dataset.cases = [full_dataset.cases[i] for i in train_idx]

    val_dataset = BraTSDataset(root_dir, target_shape=target_shape,
                                augment=False, mode="train")
    val_dataset.cases = [full_dataset.cases[i] for i in val_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,          # validation: 1 sample at a time (memory)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}  |  "
                f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader
