"""
Percolation Threshold Validation
=================================
For a 2D square lattice Bernoulli field, percolation theory
predicts a phase transition at p_c ≈ 0.5927.

Below p_c : only finite clusters exist  → low spatial disorder
Above p_c : a spanning cluster appears  → high spatial disorder

We validate that our pooling-based clutter score detects this
transition via:
  1. Inflection point in score vs p curve
  2. Peak in first derivative (dS/dp) near p_c
  3. Peak in second derivative (d²S/dp²) crossing zero at p_c
  4. Largest-cluster-size curve for physical ground truth
  5. Statistical robustness across multiple random seeds

Dependencies: numpy, opencv-python, scipy, matplotlib
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from utils import copy_figure, save_fig
from core.synthesis import make_blob_field, make_fragmented_field
from core.timing import timeit, TimerContext, get_device
from core.pooling import avg_pool, extract_features, PHYSICAL_SCALES_MM
from core.scoring import clutter_score
from core.percolation import PC_THEORY

ALWAYS_CPU = False  # set True to force CPU (for timing comparison)

def get_torch_device():
    return get_device(always_cpu=ALWAYS_CPU)


def figure_analysis_grid(mm_per_pixel, save_dir, main_images, desc_s, colors, file_name, title):
    def get_8_filtered_images(img):
        f, collected_images = extract_features(img, mm_per_pixel, return_images=True)
        return clutter_score(f), collected_images

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(5, 9, figure=fig)
    
    # 1. Create Pairwise Background Tints (Cols 1-2, 3-4, 5-6, 7-8)
    bg_colors = ['#f2f2f5', '#e5e5eaff'] * 2  # Slightly deeper tones for visibility
    for g in range(4):
        bg = fig.add_subplot(gs[:, 1 + 2*g : 3 + 2*g])
        bg.set_facecolor(bg_colors[g])
        bg.set_zorder(-1)             # Push to the very back
        bg.set_in_layout(False)       # Prevent tight_layout clipping
        
        # Hide ticks and spines manually so the facecolor stays visible
        bg.set(xticks=[], yticks=[])
        for spine in bg.spines.values():
            spine.set_visible(False)
        
    # 2. Populate Grid
    for r in range(5):
        score, collected = get_8_filtered_images(main_images[r])
        
        # Column 0: Main Image
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(main_images[r], cmap="gray", vmin=0, vmax=1)
        ax.set_ylabel(f"{desc_s[r]}\nscore = {score:.4f}", color=colors[r])
        ax.set(xticks=[], yticks=[])
        if r == 0: ax.set_title("Main", fontweight="bold")
        if r == 4: ax.set_xlabel(f"{main_images[r].shape}", fontsize=10)
        
        # Columns 1-8: Filtered Variations
        for c in range(8):
            ax = fig.add_subplot(gs[r, c + 1])
            mm, _type, pooled_img = collected[c]
            ax.imshow(pooled_img, cmap="gray", vmin=0, vmax=1)
            ax.set(xticks=[], yticks=[])
            
            # CRITICAL: Make the image subplot patch transparent so the background shines through
            ax.patch.set_visible(False)
            
            if r == 0: ax.set_title(f"({mm}mm\n{_type}_pool)", fontsize=10)
            if r == 4: ax.set_xlabel(f"{pooled_img.shape}", fontsize=10)

    fig.tight_layout()
    path3 = os.path.join(save_dir, file_name)
    save_fig(fig, path3, bw=False)
    plt.close()
    print(f"  → Figure 3 saved: {path3}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def visualize_example_images(save_dir: str, mm_per_pixel: float):
    # Generate 5 example images at different occupancy levels
    # ── FIGURE 3: sample images at key p values ──
    # ############### 
    # Creating 5 random main images
    key_ps = [0.35, 0.50, PC_THEORY, 0.65, 0.75]
    rng = np.random.default_rng(0)
    main_images = [(rng.random((700, 600)) < p).astype(np.float32) for k, p in enumerate(key_ps)]
    desc_s = []
    colors = []
    for p in key_ps:
        if p == PC_THEORY:
            desc_s.append(f"p = {p:.4f} ← p_c")
            colors.append("red")
        else:
            colors.append("black")
            desc_s.append(f"p = {p:.4f}")
    file_name="figure3_percolation_samples.tiff"
    figure_analysis_grid(mm_per_pixel, save_dir, main_images, desc_s, colors,
            file_name=file_name,
        title="Sample Images and Pooled Maps at Key Occupancy Values")
    path = save_dir / file_name
    copy_figure(path, "figure 5.4.tiff")
    
    frag_blob_configs = [
        ("1 large blob",  'black',    make_blob_field(n_blobs=1,  radius=100)),
        ("3 medium blobs",'black',    make_blob_field(n_blobs=3,  radius=58)),
        ("10 small blobs",'black',    make_blob_field(n_blobs=10, radius=32)),
        ("50 tiny blobs", 'black',    make_blob_field(n_blobs=50, radius=14)),
        ("200 dots",      'black',    make_fragmented_field(n_dots=200, dot_r=4)),
        ("1000 dots",     'black',    make_fragmented_field(n_dots=1000, dot_r=2)),
    ]
    
    desc_s, colors, main_images = map(list, zip(*frag_blob_configs))
    file_name="figure3_sythentic_fragmented_blobs.tiff"
    figure_analysis_grid(mm_per_pixel, save_dir, main_images, desc_s, colors,
            file_name=file_name,
        title="Fragmented Blobs and Pooled Maps")
    path = save_dir / file_name
    copy_figure(path, "figure 5.2.tiff")

if __name__ == "__main__":
    save_dir = pathlib.Path(__file__).parent / "results"
    os.makedirs(save_dir, exist_ok=True)
    visualize_example_images(save_dir, mm_per_pixel=1.0)