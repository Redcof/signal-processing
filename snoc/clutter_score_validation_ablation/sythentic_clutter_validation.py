"""
Empirical Validation — Three-Stage Protocol
Pooling-Based Clutter Estimation for Binary Spatial Fields

Stage 1: Synthetic controlled experiments
Stage 2: Real image evaluation (TUM / DLR / custom)
Stage 3: Ablation study

Dependencies: numpy, opencv-python, scipy, matplotlib, scikit-learn
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, kendalltau
from itertools import product
import os
import warnings

from utils import copy_figure, save_fig
from core.pooling import extract_features, PHYSICAL_SCALES_MM as PHYSICAL_SCALES_MM
from core.scoring import clutter_score
from core.synthesis import bernoulli as make_random_scatter, make_blob_field, make_fragmented_field
from core.synthesis import gt_label as gt_clutter_label
warnings.filterwarnings("ignore")

NUM_OCCUPANCY_FIELDS = 20
PC_THEORY = 0.5927

class PlotContextManager:
    def __init__(self, figsize, save_path: str):
        self.figsize = figsize
        self.save_path = save_path

    def __enter__(self):
        plt.figure(figsize=self.figsize, dpi=120)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(plt, self.save_path, bw=False)
        plt.close()

def make_gradient_occupancy(h: int = 700, w: int = 600,
                             p_start: float = 0.02,
                             p_end: float = 0.50) -> list:
    """Series of Bernoulli fields with occupancy linearly increasing."""
    ps = np.linspace(p_start, p_end, NUM_OCCUPANCY_FIELDS)
    return [(p, make_random_scatter(h, w, p, seed=int(p * 1000))) for p in ps]


# ─────────────────────────────────────────────
# STAGE 1  —  SYNTHETIC CONTROLLED EXPERIMENTS
# ─────────────────────────────────────────────

def stage1_synthetic(mm_per_pixel: float = 1.0, save_dir: str = "results"):
    """
    Three controlled series:
      A. Vary occupancy p → score must be monotone
      B. Fix area, increase fragmentation → score must rise
      C. Three archetype images for visual sanity check
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("STAGE 1: SYNTHETIC CONTROLLED EXPERIMENTS")
    print("=" * 60)

    # ── Series A: vary occupancy ──────────────────────────────
    print("\n[A] Occupancy series  (p: 0.02 → 0.50)")
    occ_series = make_gradient_occupancy()
    ps, scores_A, gts_A = [], [], []
    for p, img in occ_series:
        f = extract_features(img, mm_per_pixel)
        s = clutter_score(f)
        g = p # GT is simply the occupancy p for this series
        ps.append(p)
        scores_A.append(s)
        gts_A.append(g)
        print(f"  p={p:.3f}  score={s:.4f}  gt={g:.4f}")

    rho_A, _ = spearmanr(ps, scores_A)
    kendalltau_A, _ = kendalltau(ps, scores_A)
    print(f"  → Spearman ρ (score vs p): {rho_A:.4f}  "
          f"{'✓ monotone' if rho_A > 0.9 else '✗ check weights'}")
    print(f"  → Kendall's τ (score vs p): {kendalltau_A:.4f}")

    # ── Series B: fix total area, vary fragmentation ──────────
    print("\n[B] Fragmentation series  (same total white area)")
    frag_blob_configs = [
        (1, "1 large blob",      make_blob_field(n_blobs=1,  radius=100)),
        (3, "3 medium blobs",    make_blob_field(n_blobs=3,  radius=58)),
        (10, "10 small blobs",    make_blob_field(n_blobs=10, radius=32)),
        (50, "50 tiny blobs",     make_blob_field(n_blobs=50, radius=14)),
        (200, "200 dots",          make_fragmented_field(n_dots=200, dot_r=4)),
        (1000, "1000 dots",         make_fragmented_field(n_dots=1000, dot_r=2)),
    ]
    labels_B, scores_B, gts_B = [], [], []
    frag_score = {}
    for gt, label, img in frag_blob_configs:
        f = extract_features(img, mm_per_pixel)
        s = clutter_score(f)
        g = gt # gt_clutter_label(img)
        frag_score[label] = s
        labels_B.append(label)
        scores_B.append(s)
        gts_B.append(g)
        print(f"  {label:<22} score={s:.4f}  gt={g:.4f}")

    rho_B, _ = spearmanr(scores_B, gts_B)
    kendalltau_B, _ = kendalltau(scores_B, gts_B)
    print(f"  → Spearman ρ (score vs gt): {rho_B:.4f}")
    print(f"  → Kendall's τ (score vs gt): {kendalltau_B:.4f}")

    # ── Series C: archetypal images ───────────────────────────
    print("\n[C] Archetype images")
    archetypes = {
        "empty (p=0.01)":          make_random_scatter(p=0.01),
        "sparse scatter (p=0.05)": make_random_scatter(p=0.05),
        "dense scatter (p=0.30)":  make_random_scatter(p=0.30),
        "single blob":             make_blob_field(n_blobs=1, radius=120),
        "many dots":               make_fragmented_field(n_dots=500, dot_r=3),
    }
    arch_scores = {}
    for name, img in archetypes.items():
        f = extract_features(img, mm_per_pixel)
        s = clutter_score(f)
        arch_scores[name] = s
        print(f"  {name:<28} score={s:.4f}")
        
    # combine plot with occupancy sweep and fragmentation series for 
    # side-by-side comparison, plus flagmentation all 6 blob thumbnels 
    # and all 5 archetype thumbnails in one big figure.

    # 1. Create the main figure
    # constrained_layout is essential when nesting subfigures
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1.2]) 

    # --- ROW 1: THE THREE PLOTS ---
    axs_top = subfigs[0].subplots(1, 3)
    subfigs[0].suptitle("Synthetic Analytics", fontsize=16, fontweight='bold')

    # OCCUPANCY SWEEP
    ax = axs_top[0]
    ax.plot(ps, scores_A, "o", color="green", label="predicted score")
    ax.plot(ps, gts_A,    "s-", color="lightgray", label="GT=occupancy p")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Occupancy p")
    ax.set_ylabel("Clutter score - C")
    ax.set_title(f"A: Occupancy sweep ({NUM_OCCUPANCY_FIELDS} images)\nSpearman ρ = {rho_A:.3f}")
    
    # FRAGMENTATION SERIES
    ax = axs_top[1]
    names = list(frag_score.keys())
    vals  = list(frag_score.values())
    ax.bar(range(len(names)), vals, color="lightseagreen")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_xlabel("GT = [1..6] (log scale)")
    ax.set_ylabel("Clutter score=C")
    ax.set_title(f"B: Fragmentation series ({len(gts_B)} images)\nSpearman ρ = {rho_B:.3f}")
    ax.grid(True, alpha=0.3)
    
    # ARCHETYPES SERIES
    ax = axs_top[2]
    names = list(arch_scores.keys())
    vals  = list(arch_scores.values())
    # colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))
    ax.bar(range(len(names)), vals, color="cornflowerblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Clutter score - C")
    ax.set_title(f"C: Archetypes ({len(names)} images)")
    ax.grid(True, alpha=0.3)

    # --- ROW 2: THE IMAGE GROUPS ---
    # Split the bottom subfigure into two columns (Left: 6 images, Right: 5 images)
    subfigs_bottom = subfigs[1].subfigures(1, 2, width_ratios=[1.2, 1])

    # Group A: 6 Images (2x3 Grid)
    # FRAGMENTATION SERIES IMAGES
    subfigs_bottom[0].set_facecolor('#f9f9f9') # Light tint to visually group them
    subfigs_bottom[0].suptitle(f"Dataset: Fragmentation Series ({len(frag_blob_configs)}) Images", fontsize=12)
    axs_left = subfigs_bottom[0].subplots(2, 3)
    for i, (ax, (gt, label, img)) in enumerate(zip(axs_left.flat, frag_blob_configs)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}\nClutter score={scores_B[frag_blob_configs.index((gt, label, img))]:.4f}")
        ax.axis("off")

    # Group B: 5 Images (2x3 Grid with one empty spot)
    # ARCHETYPES IMAGES
    subfigs_bottom[1].set_facecolor('#f0f0f0') 
    subfigs_bottom[1].suptitle(f"Dataset: Archetypes Series ({len(archetypes)}) Images", fontsize=12)
    axs_right = subfigs_bottom[1].subplots(2, 3)
    for i, ax in enumerate(axs_right.flat):
        if i == 5:
            ax.remove() # Removes the 6th empty subplot
            break
        name, img = list(archetypes.items())[i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\nClutter score={arch_scores[name]:.4f}")
        ax.axis("off")
    path = os.path.join(save_dir, "sythentic_evaluation.tiff")
    save_fig(plt, path, bw=False)
    plt.close(fig)
    
    print(f"\n  → Figure saved: {save_dir}/sythentic_evaluation.tiff")
    copy_figure(path, "Figure_5.1.tiff")

    return {
        "series_A": {"p": ps, "scores": scores_A, "gt": gts_A, "rho": rho_A},
        "series_B": {"labels": labels_B, "scores": scores_B, "gt": gts_B, "rho": rho_B},
        "archetypes": arch_scores,
    }


# ─────────────────────────────────────────────
# STAGE 3  —  ABLATION STUDY
# ─────────────────────────────────────────────

def stage3_ablation(mm_per_pixel: float = 1.0,
                    save_dir: str = "results") -> dict:
    """
    Three ablation axes:
      A. Drop each pooling scale one at a time
      B. Replace max pool with avg pool (and vice versa)
      C. Vary mm/px resolution factor
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("STAGE 3: ABLATION STUDY")
    print("=" * 60)

    # Build reference dataset (n=40 synthetic images with known GT)
    num_occupancy = 100
    print(f" {num_occupancy=}")
    ps = np.linspace(0.02, 0.50, num_occupancy)
    ref_imgs = [make_random_scatter(p=p, seed=i, h=1024, w=1024) for i, p in enumerate(ps)]
    ref_gt   = ps

    def eval_rho(feat_fn):
        preds = [clutter_score(feat_fn(img)) for img in ref_imgs]
        rho, _ = spearmanr(preds, ref_gt)
        tau, _ = kendalltau(preds, ref_gt)
        return rho, tau
    
    def eval_rho_4scale(feat_fn):
        frag_blob_configs = [
            (1, "1 large blob",      make_blob_field(n_blobs=1,  radius=100)),
            (3, "3 medium blobs",    make_blob_field(n_blobs=3,  radius=58)),
            (10, "10 small blobs",    make_blob_field(n_blobs=10, radius=32)),
            (50, "50 tiny blobs",     make_blob_field(n_blobs=50, radius=14)),
            (200, "200 dots",          make_fragmented_field(n_dots=200, dot_r=4)),
            (1000, "1000 dots",         make_fragmented_field(n_dots=1000, dot_r=2)),
        ]
        gt, lbl, imgs = zip(*frag_blob_configs)
        preds = [clutter_score(feat_fn(img)) for img in imgs]
        rho, _ = spearmanr(preds, gt)
        tau, _ = kendalltau(preds, gt)
        return rho, tau

    # ── Ablation A: drop each scale ───────────────────────────
    print("\n[A] Scale ablation")
    all_scales  = PHYSICAL_SCALES_MM
    baseline_rho, baseline_tau = eval_rho_4scale(lambda img: extract_features(img, mm_per_pixel, all_scales))
    print(f"  Baseline (all scales) ρ = {baseline_rho:.4f}, τ = {baseline_tau:.4f}")

    scale_rhos = {}
    for drop in all_scales:
        remaining = [s for s in all_scales if s != drop]
        rho, tau = eval_rho_4scale(lambda img, r=remaining: extract_features(img, mm_per_pixel, r))
        delta = rho - baseline_rho
        scale_rhos[f"{drop}mm"] = rho
        print(f"  Drop {drop:4d}mm  ρ={rho:.4f}  τ={tau:.4f}  Δρ={delta:.4f}")

    # ── Ablation B: feature type substitution ────────────────
    print("\n[B] Feature-type ablation")

    def features_avg_only(img):
        feats = []
        for mm in PHYSICAL_SCALES_MM:
            block = max(1, int(mm / mm_per_pixel))
            ap = avg_pool(img, block).flatten()
            feats.extend([ap.mean(), ap.var(), ap.mean(), float(np.mean(ap > 0))])
        return np.array(feats)

    def features_max_only(img):
        feats = []
        for mm in PHYSICAL_SCALES_MM:
            block = max(1, int(mm / mm_per_pixel))
            mp = max_pool(img, block).flatten()
            feats.extend([mp.mean(), mp.var(), mp.mean(), float(np.mean(mp > 0))])
        return np.array(feats)

    variants = {
        "Full (avg + max)":    lambda img: extract_features(img, mm_per_pixel),
        "Avg pooling only":    features_avg_only,
        "Max pooling only":    features_max_only,
    }
    feat_rhos = {}
    for name, fn in variants.items():
        rho, tau = eval_rho_4scale(fn)
        feat_rhos[name] = rho
        print(f"  {name:<26} ρ = {rho:.4f}, τ = {tau:.4f}")

    # ── Ablation C: resolution factor ────────────────────────
    print("\n[C] Resolution (mm/px) ablation")
    # Fine resolutions (< 1 mm/px) produce very large blocks; coarse resolutions
    # (> 1 mm/px) collapse to 1px blocks. Both extremes are physically valid.
    # Clamp block to [1, min(h,w)//2] inside extract_features via max(1,...).
    res_factors = [0.5, 1.0, 2.0, 4.0, 8.0]
    res_rhos = {}
    for r in res_factors:
        rho, tau = eval_rho_4scale(lambda img, res=r: extract_features(img, res))
        res_rhos[f"{r} mm/px"] = rho
        print(f"  {r} mm/px  ρ = {rho:.4f}, τ = {tau:.4f}")

    # ── Plot Stage 3 ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Ablation Study ({num_occupancy} barnouli fields)", fontsize=14, fontweight="bold")

    # A: scale drop
    ax = axes[0]
    labels = ["Baseline"] + list(scale_rhos.keys())
    values = [baseline_rho] + list(scale_rhos.values())
    colors = ["lightseagreen"] + ["red" if v < baseline_rho else "cornflowerblue"
                             for v in scale_rhos.values()]
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("A: Scale ablation")
    ax.set_ylim(max(0, min(values) - 0.05), min(1.05, max(values) + 0.05))
    ax.axhline(baseline_rho, color="gray", lw=1, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # B: feature type
    ax = axes[1]
    labels_B = list(feat_rhos.keys())
    values_B = list(feat_rhos.values())
    colors_B = ["lightseagreen" if "Full" in l else "cornflowerblue" for l in labels_B]
    ax.bar(range(len(labels_B)), values_B, color=colors_B)
    ax.set_xticks(range(len(labels_B)))
    ax.set_xticklabels(labels_B, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("B: Feature-type ablation")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # C: resolution
    ax = axes[2]
    res_labels = list(res_rhos.keys())
    res_values = list(res_rhos.values())
    res_x = [float(k.split()[0]) for k in res_labels]
    ax.plot(res_x, res_values, "o-", color="#533AB7", lw=2, ms=8)
    ax.axvline(mm_per_pixel, color="red", lw=1, ls="--", alpha=0.7, label=f"baseline ({mm_per_pixel} mm/px)")
    ax.set_xscale("log")
    ax.set_xlabel("mm / pixel (log scale)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("C: Resolution ablation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "stage3_ablation.tiff")
    save_fig(fig, path, bw=False)
    plt.close(fig)
    print(f"\n  → Figure saved: {path}")

    return {
        "A_scale_drop":      {"baseline": baseline_rho, "per_scale": scale_rhos},
        "B_feature_type":    feat_rhos,
        "C_resolution":      res_rhos,
    }


# ─────────────────────────────────────────────
# STAGE 4  —  Occupancy vs score
# ─────────────────────────────────────────────
def generate_binary_patch(p, seed=1, h=700, w=600):
    """
    p: float between 0.0 and 1.0
    Returns: (h, w) numpy array of 0s and 1s
    """
    total_pixels = h * w
    # Calculate exact number of white pixels (1s)
    num_ones = int(round(total_pixels * p))
    
    # Create flat array: num_ones of '1's, the rest '0's
    patch = np.zeros(total_pixels, dtype=int)
    patch[:num_ones] = 1
    
    # Shuffle to scatter the white pixels randomly
    np.random.seed(seed)  # for reproducibility
    np.random.shuffle(patch)
    
    im = patch.reshape((h, w))
    return im.astype(np.float32)


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def print_summary(r1): #, r2, r3):
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nStage 1 — Synthetic")
    print(f"  Occupancy sweep  ρ = {r1['series_A']['rho']:.4f}")
    print(f"  Fragmentation    ρ = {r1['series_B']['rho']:.4f}")

    # print(f"\nStage 2 — Real images")
    # for name, res in r2.items():
    #     print(f"  {name:<30} ρ={res['spearman_rho']:.4f}  τ={res['kendall_tau']:.4f}")

    # print(f"\nStage 3 — Ablation")
    # print(f"  Scale ablation  baseline ρ = {r3['A_scale_drop']['baseline']:.4f}")
    # worst_scale = min(r3['A_scale_drop']['per_scale'],
    #                   key=r3['A_scale_drop']['per_scale'].get)
    # if len(np.unique(list(r3['A_scale_drop']['per_scale'].values()))) == 1:
    #     print("  All scales equally important (no single worst scale)")
    # else:
    #     print(f"  Most important scale: {worst_scale}")
    # print(f"  Best feature type: "
    #       f"{max(r3['B_feature_type'], key=r3['B_feature_type'].get)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR     = pathlib.Path(__file__).parent / "results"
    MM_PER_PIXEL = 1.0      # change for your sensor  (e.g. 0.5 for 2px per mm)

    # ── Stage 1: synthetic ──────────────────────────────────
    r1 = stage1_synthetic(mm_per_pixel=MM_PER_PIXEL, save_dir=SAVE_DIR)

    # ── Summary ─────────────────────────────────────────────
    print_summary(r1)

    print("\nAll figures saved to:", os.path.abspath(SAVE_DIR))
