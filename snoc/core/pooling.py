"""Shared pooling primitives and multi-scale feature extraction."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

PHYSICAL_SCALES_MM: List[int] = [10, 50, 200, 500]


def avg_pool(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape
    return cv2.resize(img, (max(1, w // block), max(1, h // block)),
                      interpolation=cv2.INTER_AREA)


def max_pool(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape
    hc, wc = (h // block) * block, (w // block) * block
    return img[:hc, :wc].reshape(h // block, block,
                                  w // block, block).max(axis=(1, 3))


def extract_features(
    img: np.ndarray,
    mm_per_pixel: float = 1.0,
    physical_scales_mm: Optional[List[int]] = None,
    return_images: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """4 features per scale: [avg_mean, avg_var, max_mean, max_sparsity] × S scales.

    If return_images=True, also returns a list of (mm, pool_type, pooled_img) tuples.
    """
    if physical_scales_mm is None:
        physical_scales_mm = PHYSICAL_SCALES_MM
    features: list = []
    collected: list = []
    for mm in physical_scales_mm:
        block = max(1, int(mm / mm_per_pixel))
        ap = avg_pool(img, block)
        mp = max_pool(img, block)
        if return_images:
            collected.append((mm, "avg_pool", ap))
            collected.append((mm, "max_pool", mp))
        ap_flat = ap.flatten()
        mp_flat = mp.flatten()
        features.extend([
            float(ap_flat.mean()),
            float(ap_flat.var()),
            float(mp_flat.mean()),
            float(np.mean(mp_flat > 0)),
        ])
    if return_images:
        return np.array(features), collected
    return np.array(features)


def extract_features_12(
    img: np.ndarray,
    mm_per_pixel: float = 1.0,
    physical_scales_mm: Optional[List[int]] = None,
) -> np.ndarray:
    """3 features per scale for weight search: [density, variance, fragmentation] × S."""
    if physical_scales_mm is None:
        physical_scales_mm = PHYSICAL_SCALES_MM
    features: list = []
    for mm in physical_scales_mm:
        block = max(1, int(mm / mm_per_pixel))
        ap = avg_pool(img, block)
        mp = max_pool(img, block)
        features.extend([
            float(ap.mean()),
            float(ap.var()),
            float(np.mean(mp > 0)),
        ])
    return np.array(features)


try:
    import torch
    import torch.nn.functional as F

    def extract_features_batch(
        images: "torch.Tensor",
        mm_per_pixel: float = 1.0,
        physical_scales_mm: Optional[List[int]] = None,
    ) -> "torch.Tensor":
        """Batch extract_features() via PyTorch. Returns shape (B, 4*S)."""
        if physical_scales_mm is None:
            physical_scales_mm = PHYSICAL_SCALES_MM
        x = images.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.ndim == 3:
            x = x.unsqueeze(1)
        B, _, H, W = x.shape
        scale_feats = []
        for mm in physical_scales_mm:
            block = max(1, int(mm / mm_per_pixel))
            Hc = (H // block) * block
            Wc = (W // block) * block
            xc = x[:, :, :Hc, :Wc]
            ap = F.avg_pool2d(xc, kernel_size=block, stride=block)
            mp = F.max_pool2d(xc, kernel_size=block, stride=block)
            avg_mean     = ap.mean(dim=[1, 2, 3])
            avg_var      = ap.pow(2).mean(dim=[1, 2, 3]) - avg_mean.pow(2)
            max_mean     = mp.mean(dim=[1, 2, 3])
            max_sparsity = (mp > 0).float().mean(dim=[1, 2, 3])
            scale_feats.append(
                torch.stack([avg_mean, avg_var, max_mean, max_sparsity], dim=1)
            )
        return torch.cat(scale_feats, dim=1)

except ImportError:
    pass
