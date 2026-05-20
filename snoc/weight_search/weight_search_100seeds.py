"""
Weight Search — 100 Seeds × 80 Images
All three steps per seed, then aggregate statistics.
"""

import numpy as np
import matplotlib

from utils import save_fig
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from skopt import gp_minimize
from skopt.space import Real
import warnings
warnings.filterwarnings("ignore")

from core.pooling import extract_features_12, PHYSICAL_SCALES_MM as PHYSICAL_SCALES
from core.scoring import score_from_weights as score_from_w
from core.synthesis import gt_label

MM_PER_PIXEL = 1.0
EPS          = 1e-8
N_SEEDS      = 100
N_IMAGES     = 80

def grid_search(feats, gt):
    steps = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_rho, best_w = -1, None
    for w1 in steps:
        for w2 in steps:
            w3 = round(1.0 - w1 - w2, 10)
            if w3 < 0.04 or w3 > 0.91:
                continue
            rho, _ = spearmanr(score_from_w(feats, w1, w2, w3), gt)
            if rho > best_rho:
                best_rho, best_w = rho, (w1, w2, w3)
    return best_w, best_rho

def bayesian_search(feats, gt, n_calls=50):
    def obj(params):
        w1, w2 = params
        w3 = 1.0 - w1 - w2
        if w3 < 0.02 or w3 > 0.98:
            return 0.0
        rho, _ = spearmanr(score_from_w(feats, w1, w2, w3), gt)
        return -rho
    res = gp_minimize(obj, [Real(0.05, 0.90), Real(0.05, 0.90)],
                      n_calls=n_calls, n_initial_points=10,
                      random_state=0, verbose=False)
    w1, w2 = res.x
    w3 = 1.0 - w1 - w2
    return (w1, w2, w3), -res.fun

def inv_var_weights(feats):
    f   = feats.reshape(len(feats), 4, 3)
    vs  = np.array([np.var(f[:,:,i].mean(axis=1)) for i in range(3)])
    vs  = np.where(vs < 1e-12, 1e-12, vs)
    inv = 1.0 / vs
    return inv / inv.sum()

# ── run across 100 seeds ─────────────────────────────────────────────────────

print(f"Running weight search: {N_SEEDS} seeds × {N_IMAGES} images")
print("="*55)

ps = np.linspace(0.02, 0.50, N_IMAGES)

# storage
grid_w  = np.zeros((N_SEEDS, 3))
grid_r  = np.zeros(N_SEEDS)
bayes_w = np.zeros((N_SEEDS, 3))
bayes_r = np.zeros(N_SEEDS)
anal_w  = np.zeros((N_SEEDS, 3))
anal_r  = np.zeros(N_SEEDS)
prior_r = np.zeros(N_SEEDS)

for seed in range(N_SEEDS):
    rng   = np.random.default_rng(seed)
    feats = np.array([extract_features_12(
                        (rng.random((700, 600)) < p).astype(np.float32))
                      for p in ps])
    gt    = np.array([gt_label(
                        (np.random.default_rng(seed*1000+i).random((700, 600)) < p
                         ).astype(np.float32))
                      for i, p in enumerate(ps)])

    # grid
    gw, gr = grid_search(feats, gt)
    grid_w[seed]  = gw
    grid_r[seed]  = gr

    # bayesian
    bw, br = bayesian_search(feats, gt, n_calls=50)
    bayes_w[seed] = bw
    bayes_r[seed] = br

    # analytical
    aw = inv_var_weights(feats)
    anal_w[seed] = aw
    pr, _ = spearmanr(score_from_w(feats, *aw), gt)
    anal_r[seed] = pr

    # prior
    pr2, _ = spearmanr(score_from_w(feats, 0.40, 0.35, 0.25), gt)
    prior_r[seed] = pr2

    if (seed+1) % 10 == 0:
        print(f"  seed {seed+1:3d}/100 done  "
              f"grid_rho={grid_r[:seed+1].mean():.4f}  "
              f"bayes_rho={bayes_r[:seed+1].mean():.4f}")

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print(f"{'Method':<28} {'w1 mean±std':>14} {'w2 mean±std':>14} {'w3 mean±std':>14}")
print("-"*65)
for label, w, r in [
    ("Prior (heuristic)",    np.tile([0.40,0.35,0.25],(N_SEEDS,1)), prior_r),
    ("Grid search",          grid_w,  grid_r),
    ("Bayesian optimisation",bayes_w, bayes_r),
    ("Inverse-variance",     anal_w,  anal_r),
]:
    print(f"  {label:<26} "
          f"{w[:,0].mean():.4f}±{w[:,0].std():.4f}  "
          f"{w[:,1].mean():.4f}±{w[:,1].std():.4f}  "
          f"{w[:,2].mean():.4f}±{w[:,2].std():.4f}")

print("\n" + "="*65)
print(f"{'Method':<28} {'rho mean':>10} {'rho std':>10} {'rho min':>10} {'rho max':>10}")
print("-"*65)
for label, r in [
    ("Prior (heuristic)",    prior_r),
    ("Grid search",          grid_r),
    ("Bayesian optimisation",bayes_r),
    ("Inverse-variance",     anal_r),
]:
    print(f"  {label:<26} "
          f"{r.mean():>10.6f} {r.std():>10.6f} "
          f"{r.min():>10.6f} {r.max():>10.6f}")

# ── figures ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(f"Weight Search — {N_SEEDS} Seeds × {N_IMAGES} Images",
             fontsize=13, fontweight="bold")

methods  = ["Grid search", "Bayesian opt.", "Inv-variance"]
w_arrays = [grid_w, bayes_w, anal_w]
r_arrays = [grid_r, bayes_r, anal_r]
colors   = ["#1D9E75", "#534AB7", "#BA7517"]

# Row 1: weight distributions (violin)
ax = axes[0, 0]
data_w1 = [grid_w[:,0], bayes_w[:,0], anal_w[:,0]]
data_w2 = [grid_w[:,1], bayes_w[:,1], anal_w[:,1]]
data_w3 = [grid_w[:,2], bayes_w[:,2], anal_w[:,2]]
x = np.arange(3)
bw = 0.25
for i, (d, lbl, col) in enumerate([
    (data_w1, "w₁ density",     "#1D9E75"),
    (data_w2, "w₂ irregularity","#534AB7"),
    (data_w3, "w₃ fragmentation","#D85A30")
]):
    bp = ax.boxplot(d, positions=x + (i-1)*bw, widths=bw*0.85,
                    patch_artist=True,
                    boxprops=dict(facecolor=col, alpha=0.6),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=col),
                    capprops=dict(color=col),
                    flierprops=dict(marker=".", color=col, alpha=0.4, markersize=4))
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Weight value"); ax.set_title("Weight distributions (box plots)")
from matplotlib.patches import Patch
ax.legend([Patch(facecolor=c, alpha=0.7) for c in ["#1D9E75","#534AB7","#D85A30"]],
          ["w₁ density","w₂ irregularity","w₃ fragmentation"], fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(0.40, color="#1D9E75", lw=0.8, ls=":", alpha=0.5)
ax.axhline(0.35, color="#534AB7", lw=0.8, ls=":", alpha=0.5)
ax.axhline(0.25, color="#D85A30", lw=0.8, ls=":", alpha=0.5)

# Row 1, col 2: rho distribution violin
ax = axes[0, 1]
all_rhos = [prior_r, grid_r, bayes_r, anal_r]
labels_r = ["Prior", "Grid", "Bayesian", "Inv-var"]
cols_r   = ["#888780","#1D9E75","#534AB7","#BA7517"]
parts = ax.violinplot(all_rhos, positions=range(4), showmedians=True, showextrema=True)
for i, (pc, col) in enumerate(zip(parts['bodies'], cols_r)):
    pc.set_facecolor(col); pc.set_alpha(0.6)
parts['cmedians'].set_colors(cols_r)
parts['cbars'].set_colors(["#888888"]*4)
parts['cmins'].set_colors(["#888888"]*4)
parts['cmaxes'].set_colors(["#888888"]*4)
ax.set_xticks(range(4)); ax.set_xticklabels(labels_r, fontsize=9)
ax.set_ylabel("Spearman ρ"); ax.set_title("ρ distribution across 100 seeds")
ax.grid(True, alpha=0.3, axis="y")

# Row 1, col 3: mean weights bar comparison
ax = axes[0, 2]
x4  = np.arange(4)
bw2 = 0.22
all_w = [np.tile([0.40,0.35,0.25],(N_SEEDS,1)), grid_w, bayes_w, anal_w]
labels4 = ["Prior","Grid","Bayesian","Inv-var"]
for i, (col, lbl) in enumerate([
    ("#1D9E75","w₁ density"),
    ("#534AB7","w₂ irregularity"),
    ("#D85A30","w₃ fragmentation")
]):
    means = [w[:,i].mean() for w in all_w]
    stds  = [w[:,i].std()  for w in all_w]
    ax.bar(x4 + (i-1)*bw2, means, bw2, yerr=stds,
           color=col, alpha=0.80, capsize=3, label=lbl)
ax.set_xticks(x4); ax.set_xticklabels(labels4, fontsize=9)
ax.set_ylabel("Mean weight ± std"); ax.set_title("Mean weights (mean ± std)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# Row 2: per-seed ρ traces
for col_idx, (label, r_arr, col) in enumerate([
    ("Grid search",          grid_r,  "#1D9E75"),
    ("Bayesian optimisation",bayes_r, "#534AB7"),
    ("Inverse-variance",     anal_r,  "#BA7517"),
]):
    ax = axes[1, col_idx]
    ax.plot(range(N_SEEDS), r_arr, color=col, lw=0.8, alpha=0.6)
    ax.axhline(r_arr.mean(), color=col, lw=2, ls="--",
               label=f"mean={r_arr.mean():.4f}")
    ax.axhline(prior_r.mean(), color="#888780", lw=1.2, ls=":",
               label=f"prior mean={prior_r.mean():.4f}")
    ax.fill_between(range(N_SEEDS),
                    r_arr.mean()-r_arr.std(),
                    r_arr.mean()+r_arr.std(),
                    alpha=0.15, color=col)
    ax.set_xlabel("Seed"); ax.set_ylabel("Spearman ρ")
    ax.set_title(f"{label}\nper-seed ρ  (mean±std={r_arr.mean():.4f}±{r_arr.std():.4f})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, f"weight_search_s{N_SEEDS}_p{N_IMAGES}.png", bw=False)
plt.close(fig)
print(f"\n  Figure saved: weight_search_s{N_SEEDS}_p{N_IMAGES}.png")
print("Done.")
