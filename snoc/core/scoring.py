"""Clutter scoring functions — feature-vector and image-to-score variants."""

from __future__ import annotations

# from typing import List, Optional

import numpy as np

# from core.pooling import avg_pool, max_pool, PHYSICAL_SCALES_MM

EPS = 1e-8


# ── Feature-vector scorers (used with extract_features / extract_features_12) ──

def clutter_score(
    features: np.ndarray,
    w_density: float = 0.40,
    w_irregularity: float = 0.35,
    w_fragmentation: float = 0.25,
) -> float:
    """Weighted score from 16-dim feature vector (4 features × S scales)."""
    f = features.reshape(-1, 4)
    return (w_density      * f[:, 0].mean() +
            w_irregularity * f[:, 1].mean() +
            w_fragmentation * f[:, 3].mean())


def score_from_weights(
    features_12: np.ndarray,
    w1: float,
    w2: float,
    w3: float,
) -> np.ndarray:
    """Weighted score from 12-dim feature array (N, 12). Returns (N,)."""
    f  = features_12.reshape(len(features_12), 4, 3)
    D  = f[:, :, 0].mean(axis=1)
    I  = f[:, :, 1].mean(axis=1)
    Fr = f[:, :, 2].mean(axis=1)
    return w1 * D + w2 * I + w3 * Fr


# # ── Image-to-score scorers (used in clutter_unified / clutter_hybrid) ──────────

# def clutter_score_new(
#     img: np.ndarray,
#     mm_per_pixel: float = 1.0,
#     scales: Optional[List[int]] = None,
# ) -> float:
#     """Parameter-free: C = (1/S) Σ [σ²/(μ(1-μ)+ε)] × φ."""
#     if scales is None:
#         scales = PHYSICAL_SCALES_MM
#     total = 0.0
#     for mm in scales:
#         block  = max(1, int(mm / mm_per_pixel))
#         ap     = avg_pool(img, block)
#         mp     = max_pool(img, block)
#         mu     = float(ap.mean())
#         sigma2 = float(ap.var())
#         phi    = float(np.mean(mp > 0))
#         total += (sigma2 / (mu * (1.0 - mu) + EPS)) * phi
#     return total / len(scales)


# def clutter_score_old(
#     img: np.ndarray,
#     mm_per_pixel: float = 1.0,
#     scales: Optional[List[int]] = None,
#     w_density: float = 0.40,
#     w_irregularity: float = 0.35,
#     w_fragmentation: float = 0.25,
# ) -> float:
#     """Original weighted linear combination (image-to-score)."""
#     if scales is None:
#         scales = PHYSICAL_SCALES_MM
#     D, V, F = [], [], []
#     for mm in scales:
#         block = max(1, int(mm / mm_per_pixel))
#         ap = avg_pool(img, block).flatten()
#         mp = max_pool(img, block).flatten()
#         D.append(float(ap.mean()))
#         V.append(float(ap.var()))
#         F.append(float(np.mean(mp > 0)))
#     return w_density * np.mean(D) + w_irregularity * np.mean(V) + w_fragmentation * np.mean(F)


# def clutter_score_hybrid(
#     img: np.ndarray,
#     mm_per_pixel: float = 1.0,
#     scales: Optional[List[int]] = None,
# ) -> float:
#     """H = (1/3)[D̄ + ND̄ + F̄] — equal-weight, no tunable coefficients."""
#     if scales is None:
#         scales = PHYSICAL_SCALES_MM
#     D_vals, ND_vals, F_vals = [], [], []
#     for mm in scales:
#         block  = max(1, int(mm / mm_per_pixel))
#         ap     = avg_pool(img, block)
#         mp     = max_pool(img, block)
#         mu     = float(ap.mean())
#         sigma2 = float(ap.var())
#         phi    = float(np.mean(mp > 0))
#         D_vals.append(mu)
#         ND_vals.append(sigma2 / (mu * (1.0 - mu) + EPS))
#         F_vals.append(phi)
#     return (float(np.mean(D_vals)) + float(np.mean(ND_vals)) + float(np.mean(F_vals))) / 3.0


try:
    import torch

    def clutter_score_batch(
        features: "torch.Tensor",
        w_density: float = 0.40,
        w_irregularity: float = 0.35,
        w_fragmentation: float = 0.25,
    ) -> "torch.Tensor":
        """Batch clutter_score for output of extract_features_batch(). Returns (B,)."""
        B, _ = features.shape
        f = features.view(B, -1, 4)
        return (w_density       * f[:, :, 0].mean(dim=1) +
                w_irregularity  * f[:, :, 1].mean(dim=1) +
                w_fragmentation * f[:, :, 3].mean(dim=1))

except ImportError:
    pass
