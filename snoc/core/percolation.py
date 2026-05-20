"""Physical ground-truth metrics and gradient analysis for percolation validation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.signal import savgol_filter

PC_THEORY = 0.5927  # 2D square-lattice site percolation threshold


def largest_cluster_fraction(img: np.ndarray) -> float:
    """Size of largest connected component / total foreground pixels."""
    binary = (img > 0).astype(np.uint8)
    labeled, n = cc_label(binary)
    if n == 0:
        return 0.0
    sizes    = np.bincount(labeled.ravel())[1:]
    total_fg = binary.sum()
    return float(sizes.max() / total_fg) if total_fg > 0 else 0.0


def spans_image(img: np.ndarray) -> bool:
    """True if any component touches both the left and right edges."""
    binary = (img > 0).astype(np.uint8)
    labeled, _ = cc_label(binary)
    left  = set(labeled[:, 0].ravel()) - {0}
    right = set(labeled[:, -1].ravel()) - {0}
    return bool(left & right)


def smooth(y: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing — preserves peak shape better than a moving average."""
    n   = len(y)
    win = min(window, n - (0 if n % 2 == 1 else 1))
    win = win if win % 2 == 1 else win - 1
    win = max(win, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    return savgol_filter(y, window_length=win, polyorder=poly)


def first_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(smooth(y), x)


def second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(smooth(first_derivative(y, x)), x)


def find_inflection(scores: np.ndarray, ps: np.ndarray) -> float:
    """p value of d²S/dp² zero-crossing closest to PC_THEORY."""
    d2 = second_derivative(scores, ps)
    zc = np.where(np.diff(np.sign(d2)))[0]
    if len(zc) == 0:
        return float(ps[np.argmax(first_derivative(scores, ps))])
    inf_ps = ps[zc]
    return float(inf_ps[np.argmin(np.abs(inf_ps - PC_THEORY))])


def find_peak_derivative(scores: np.ndarray, ps: np.ndarray) -> float:
    """p value where dS/dp is maximum — steepest score transition."""
    return float(ps[np.argmax(first_derivative(scores, ps))])
