"""
Weight Search for Clutter Score
=================================
12-dim feature set: [f1_density, f2_variance, f3_fragmentation] × 4 scales

Steps:
  1. Exhaustive grid search (step 0.05)
  2. Bayesian optimisation (Gaussian process surrogate, 80 iterations)
  3. Analytical inverse-variance weighting (post-hoc justification)

Ground truth: theory-based GT label on synthetic Bernoulli fields
"""

import numpy as np
import matplotlib

from utils import save_fig
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from itertools import product as iproduct
import warnings
warnings.filterwarnings("ignore")

from core.pooling import extract_features_12, PHYSICAL_SCALES_MM as PHYSICAL_SCALES
from core.scoring import score_from_weights
from core.synthesis import gt_label

MM_PER_PIXEL = 1.0
EPS          = 1e-8
SAVE_DIR     = "."

# ── build reference dataset ───────────────────────────────────────────────────

print("Building reference dataset (n=80 synthetic images)...")
N  = 80
ps = np.linspace(0.02, 0.50, N)
rng = np.random.default_rng(42)

all_feats = []
all_gt    = []
for p in ps:
    img = (rng.random((700, 600)) < p).astype(np.float32)
    all_feats.append(extract_features_12(img))
    all_gt.append(gt_label(img))

feats_arr = np.array(all_feats)   # (80, 12)
gt_arr    = np.array(all_gt)      # (80,)
print(f"  Feature matrix: {feats_arr.shape}   GT range: [{gt_arr.min():.3f}, {gt_arr.max():.3f}]")

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — EXHAUSTIVE GRID SEARCH
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 1: Grid search (step=0.05, w1+w2+w3=1)")
print("="*55)

steps  = np.round(np.arange(0.05, 0.96, 0.05), 2)
best_rho_grid = -1
best_w_grid   = None
grid_results  = []

for w1 in steps:
    for w2 in steps:
        w3 = round(1.0 - w1 - w2, 10)
        if w3 < 0.04 or w3 > 0.91:
            continue
        preds = score_from_weights(feats_arr, w1, w2, w3)
        rho, _ = spearmanr(preds, gt_arr)
        grid_results.append((w1, w2, w3, rho))
        if rho > best_rho_grid:
            best_rho_grid = rho
            best_w_grid   = (w1, w2, w3)

grid_results.sort(key=lambda x: -x[3])
print(f"\n  Combinations evaluated: {len(grid_results)}")
print(f"\n  Top 10 results:")
print(f"  {'w1':>6} {'w2':>6} {'w3':>6} {'rho':>8}")
print("  " + "-"*30)
for w1,w2,w3,rho in grid_results[:10]:
    print(f"  {w1:>6.2f} {w2:>6.2f} {w3:>6.2f} {rho:>8.4f}")

print(f"\n  BEST: w1={best_w_grid[0]:.2f}  w2={best_w_grid[1]:.2f}  "
      f"w3={best_w_grid[2]:.2f}  rho={best_rho_grid:.4f}")

# ════════════════════════════════════════════════════════════════════════
# STEP 2 — BAYESIAN OPTIMISATION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 2: Bayesian optimisation (GP surrogate, 80 iters)")
print("="*55)

from skopt import gp_minimize
from skopt.space import Real

call_count  = 0
bayes_trace = []

def objective(params):
    global call_count
    w1, w2 = params
    w3 = 1.0 - w1 - w2
    if w3 < 0.02 or w3 > 0.98:
        return 0.0
    preds = score_from_weights(feats_arr, w1, w2, w3)
    rho, _ = spearmanr(preds, gt_arr)
    bayes_trace.append((w1, w2, w3, rho))
    call_count += 1
    return -rho

space = [Real(0.05, 0.90, name='w1'),
         Real(0.05, 0.90, name='w2')]

result = gp_minimize(
    objective, space,
    n_calls=80, n_initial_points=15,
    random_state=42, verbose=False
)

w1_b, w2_b = result.x
w3_b = 1.0 - w1_b - w2_b
rho_b = -result.fun

print(f"\n  Iterations: {len(bayes_trace)}")
print(f"  BEST: w1={w1_b:.4f}  w2={w2_b:.4f}  w3={w3_b:.4f}  rho={rho_b:.4f}")

# ════════════════════════════════════════════════════════════════════════
# STEP 3 — ANALYTICAL INVERSE-VARIANCE WEIGHTING
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 3: Analytical inverse-variance weighting")
print("="*55)

f = feats_arr.reshape(N, 4, 3)
D_vals  = f[:, :, 0].mean(axis=1)
I_vals  = f[:, :, 1].mean(axis=1)
Fr_vals = f[:, :, 2].mean(axis=1)

var_D  = float(np.var(D_vals))
var_I  = float(np.var(I_vals))
var_Fr = float(np.var(Fr_vals))

inv_v  = np.array([1/var_D, 1/var_I, 1/var_Fr])
w_anal = inv_v / inv_v.sum()

preds_anal = w_anal[0]*D_vals + w_anal[1]*I_vals + w_anal[2]*Fr_vals
rho_anal, _ = spearmanr(preds_anal, gt_arr)

print(f"\n  Feature variances:  Var(D)={var_D:.5f}  Var(I)={var_I:.5f}  Var(F)={var_Fr:.5f}")
print(f"  Analytical weights: w1={w_anal[0]:.4f}  w2={w_anal[1]:.4f}  w3={w_anal[2]:.4f}")
print(f"  Spearman rho:       {rho_anal:.4f}")

# ════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("FINAL COMPARISON")
print("="*55)
print(f"\n  {'Method':<28} {'w1':>7} {'w2':>7} {'w3':>7} {'rho':>8}")
print("  " + "-"*55)
print(f"  {'Prior (heuristic)':<28} {'0.40':>7} {'0.35':>7} {'0.25':>7}  ", end="")
preds_prior = score_from_weights(feats_arr, 0.40, 0.35, 0.25)
rho_prior, _ = spearmanr(preds_prior, gt_arr)
print(f"{rho_prior:.4f}")
print(f"  {'Grid search':<28} {best_w_grid[0]:>7.4f} {best_w_grid[1]:>7.4f} {best_w_grid[2]:>7.4f}  {best_rho_grid:.4f}")
print(f"  {'Bayesian optimisation':<28} {w1_b:>7.4f} {w2_b:>7.4f} {w3_b:>7.4f}  {rho_b:.4f}")
print(f"  {'Inverse-variance (analytical)':<28} {w_anal[0]:>7.4f} {w_anal[1]:>7.4f} {w_anal[2]:>7.4f}  {rho_anal:.4f}")

# ════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Weight Search Results — 12-dim Feature Set", fontsize=13, fontweight="bold")

# Panel 1: grid search heatmap (w1 vs w2, best w3 implied)
ax = axes[0]
grid_w1 = np.array([r[0] for r in grid_results])
grid_w2 = np.array([r[1] for r in grid_results])
grid_rho = np.array([r[3] for r in grid_results])
sc = ax.scatter(grid_w1, grid_w2, c=grid_rho, cmap="RdYlGn", s=40,
                vmin=grid_rho.min(), vmax=grid_rho.max(), alpha=0.8)
ax.scatter(*best_w_grid[:2], c="red", s=120, zorder=5, marker="*", label=f"best ρ={best_rho_grid:.3f}")
plt.colorbar(sc, ax=ax, label="Spearman ρ")
ax.set_xlabel("w₁ (density)"); ax.set_ylabel("w₂ (irregularity)")
ax.set_title("Grid search — ρ landscape"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# Panel 2: Bayesian convergence
ax = axes[1]
bayes_rhos = [r[3] for r in bayes_trace]
running_best = np.maximum.accumulate(bayes_rhos)
ax.plot(bayes_rhos, color="#B4B2A9", lw=0.8, alpha=0.7, label="iteration ρ")
ax.plot(running_best, color="#1D9E75", lw=2, label="best so far")
ax.axhline(best_rho_grid, color="#E24B4A", lw=1.5, ls="--", label=f"grid best={best_rho_grid:.3f}")
ax.set_xlabel("Iteration"); ax.set_ylabel("Spearman ρ")
ax.set_title("Bayesian optimisation convergence"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 3: weight comparison bar chart
ax = axes[2]
methods = ["Prior\n(heuristic)", "Grid\nsearch", "Bayesian\nopt.", "Inverse\nvariance"]
w1s = [0.40, best_w_grid[0], w1_b, w_anal[0]]
w2s = [0.35, best_w_grid[1], w2_b, w_anal[1]]
w3s = [0.25, best_w_grid[2], w3_b, w_anal[2]]
x   = np.arange(len(methods)); bw = 0.25
ax.bar(x - bw,   w1s, bw, color="#1D9E75", alpha=0.85, label="w₁ density")
ax.bar(x,        w2s, bw, color="#534AB7", alpha=0.85, label="w₂ irregularity")
ax.bar(x + bw,   w3s, bw, color="#D85A30", alpha=0.85, label="w₃ fragmentation")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Weight value"); ax.set_title("Weight comparison across methods")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, 0.75)

plt.tight_layout()
# fig.savefig(f"{SAVE_DIR}/weight_search_results.tiff", dpi=130, bbox_inches="tight")
save_fig(fig, f"{SAVE_DIR}/weight_search_results.png", bw=False)
plt.close(fig)
print(f"\n  Figure saved: {SAVE_DIR}/weight_search_results.png")

# save final weights
np.save(f"{SAVE_DIR}/optimal_weights.npy",
        np.array([w1_b, w2_b, w3_b]))
print(f"  Weights saved: {SAVE_DIR}/optimal_weights.npy")
print("\nDone.")
