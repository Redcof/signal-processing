"""Synthetic binary image generators and theory-grounded ground-truth labels."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import label as cc_label

EPS = 1e-8


def bernoulli(
    h: int = 700, w: int = 600, p: float = 0.10, seed: int = 42
) -> np.ndarray:
    """Independent Bernoulli occupancy field."""
    return (np.random.default_rng(seed).random((h, w)) < p).astype(np.float32)


def make_blob_field(
    h: int = 700, w: int = 600,
    n_blobs: int = 5, radius: int = 40, seed: int = 42,
) -> np.ndarray:
    """Concentrated foreground: n_blobs circles of given radius."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for _ in range(n_blobs):
        cy = rng.integers(radius, h - radius)
        cx = rng.integers(radius, w - radius)
        cv2.circle(img, (int(cx), int(cy)), radius, 1.0, -1)
    return np.clip(img, 0, 1)


def make_fragmented_field(
    h: int = 700, w: int = 600,
    n_dots: int = 200, dot_r: int = 4, seed: int = 42,
) -> np.ndarray:
    """Scattered small dots — same total area as make_blob_field but fragmented."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for _ in range(n_dots):
        cy = rng.integers(0, h)
        cx = rng.integers(0, w)
        cv2.circle(img, (int(cx), int(cy)), dot_r, 1.0, -1)
    return np.clip(img, 0, 1)


def gt_label(img: np.ndarray) -> float:
    """Theory-grounded label: 0.4·entropy + 0.35·CC density + 0.25·radial variance."""
    p   = float(img.mean()) + EPS
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    _, n = cc_label((img > 0).astype(np.uint8))
    norm_cc = min(n / (img.size / 100.0 + EPS) / 10.0, 1.0)
    h, w = img.shape
    idx  = np.mgrid[0:h, 0:w]
    r    = np.sqrt((idx[0] - h // 2) ** 2 + (idx[1] - w // 2) ** 2)
    rmax = r.max() + EPS
    rings = [img[((k / 8) * rmax <= r) & (r < ((k + 1) / 8) * rmax)].mean()
             for k in range(8)]
    rad_var = min(float(np.var(rings)) / 0.1, 1.0)
    return float(np.clip(0.4 * ent + 0.35 * norm_cc + 0.25 * rad_var, 0, 1))
