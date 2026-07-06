"""
Ablation Study — Complete
==========================
Scale ablation      : fragmentation series  GT = [1..6]
Feature type        : fragmentation series  GT = [1..6]
                    + occupancy sweep       GT = p
Resolution ablation : occupancy sweep       GT = p
                      image size 1024x1024 (ensures 0.5 mm/px block >= 1px)

All using SNOCS 12-dim features (f1 density, f2 variance, f3 fragmentation)
Plots: column-wise grouping — scale | feature type | resolution
"""

import pathlib

import numpy as np
import matplotlib
from utils import copy_figure, save_fig

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings("ignore")

from core.pooling import avg_pool, max_pool, PHYSICAL_SCALES_MM as PHYSICAL_SCALES
from core.synthesis import bernoulli, make_blob_field as blob_field, make_fragmented_field as dot_field

MM_PER_PIXEL = 1.0
IMG_H, IMG_W = 1024, 1024    # large enough for 0.5 mm/px (block=5px min)
save_dir = pathlib.Path(__file__).parent / "results"
save_dir.mkdir(exist_ok=True)

# ── SNOCS variants ────────────────────────────────────────────────────────────

def snocs_full(img, scales=None, mm_per_pixel=MM_PER_PIXEL,
               w1=0.40, w2=0.35, w3=0.25):
    if scales is None: scales = PHYSICAL_SCALES
    D_v, I_v, F_v = [], [], []
    for mm in scales:
        k  = max(1, int(mm / mm_per_pixel))
        ap = avg_pool(img, k)
        mp = max_pool(img, k)
        D_v.append(float(ap.mean()))
        I_v.append(float(ap.var()))
        F_v.append(float(np.mean(mp > 0)))
    return w1*np.mean(D_v) + w2*np.mean(I_v) + w3*np.mean(F_v)

def snocs_avg_only(img, scales=None, mm_per_pixel=MM_PER_PIXEL):
    """f1 + f2 from avg-pool only."""
    if scales is None: scales = PHYSICAL_SCALES
    D_v, I_v = [], []
    for mm in scales:
        k  = max(1, int(mm / mm_per_pixel))
        ap = avg_pool(img, k)
        D_v.append(float(ap.mean()))
        I_v.append(float(ap.var()))
    return 0.40*np.mean(D_v) + 0.60*np.mean(I_v)

def snocs_max_only(img, scales=None, mm_per_pixel=MM_PER_PIXEL):
    """f3 from max-pool only."""
    if scales is None: scales = PHYSICAL_SCALES
    F_v = []
    for mm in scales:
        k  = max(1, int(mm / mm_per_pixel))
        mp = max_pool(img, k)
        F_v.append(float(np.mean(mp > 0)))
    return float(np.mean(F_v))


def metrics(preds, gt):
    p, g = np.array(preds), np.array(gt)
    if p.std() < 1e-10: return 0.0, 0.0
    rho, _ = spearmanr(p, g)
    tau, _ = kendalltau(p, g)
    return float(rho), float(tau)

# ═════════════════════════════════════════════════════════════════════════════
# DATASETS
# ═════════════════════════════════════════════════════════════════════════════

# Fragmentation series — fixed total area, increasing scatter
frag_configs = [
    ("1 large blob",   blob_field(h=IMG_H, w=IMG_W, n_blobs=1,    radius=150)),
    ("3 medium blobs", blob_field(h=IMG_H, w=IMG_W, n_blobs=3,    radius=87)),
    ("10 small blobs", blob_field(h=IMG_H, w=IMG_W, n_blobs=10,   radius=48)),
    ("50 tiny blobs",  blob_field(h=IMG_H, w=IMG_W, n_blobs=50,   radius=21)),
    ("200 dots",       dot_field( h=IMG_H, w=IMG_W, n_dots=200,   dot_r=6)),
    ("1000 dots",      dot_field( h=IMG_H, w=IMG_W, n_dots=1000,  dot_r=3)),
]
frag_labels = [c[0] for c in frag_configs]
frag_imgs   = [c[1] for c in frag_configs]
frag_gt     = list(range(1, 7))          # GT = [1..6] by construction

# Occupancy sweep
occ_ps   = np.linspace(0.02, 0.50, 12)
occ_imgs = [bernoulli(h=IMG_H, w=IMG_W, p=p, seed=i) for i, p in enumerate(occ_ps)]
occ_gt   = list(occ_ps)

# ═════════════════════════════════════════════════════════════════════════════
# A — SCALE ABLATION  (fragmentation GT=[1..6])
# ═════════════════════════════════════════════════════════════════════════════

print("="*64)
print("A  SCALE ABLATION  (fragmentation series  GT=[1..6])")
print("="*64)
print(f"\n  {'Configuration':<28} {'ρ':>8} {'τ':>8} {'Δρ':>8}")
print("  "+"-"*54)

base_f  = [snocs_full(img) for img in frag_imgs]
rho_b, tau_b = metrics(base_f, frag_gt)
print(f"  {'Baseline (all scales)':<28} {rho_b:>8.4f} {tau_b:>8.4f} {'—':>8}")

scale_res = {"Baseline": (rho_b, tau_b, 0.0)}
for drop in PHYSICAL_SCALES:
    rem    = [s for s in PHYSICAL_SCALES if s != drop]
    sc     = [snocs_full(img, scales=rem) for img in frag_imgs]
    rho, tau = metrics(sc, frag_gt)
    delta  = rho - rho_b
    lbl    = f"Drop {drop} mm"
    scale_res[lbl] = (rho, tau, delta)
    print(f"  {lbl:<28} {rho:>8.4f} {tau:>8.4f} {delta:>+8.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# B — FEATURE TYPE ABLATION  (both datasets)
# ═════════════════════════════════════════════════════════════════════════════

print("\n"+"="*64)
print("B  FEATURE TYPE ABLATION")
print("="*64)

feat_fns = {
    "Full (avg + max)": snocs_full,
    "Avg-pool only":    snocs_avg_only,
    "Max-pool only":    snocs_max_only,
}

print(f"\n  Fragmentation series  GT=[1..6]  (fixed total area)")
print(f"  {'Configuration':<28} {'ρ':>8} {'τ':>8}")
print("  "+"-"*46)
feat_frag = {}
for name, fn in feat_fns.items():
    sc = [fn(img) for img in frag_imgs]
    rho, tau = metrics(sc, frag_gt)
    feat_frag[name] = (rho, tau, sc)
    print(f"  {name:<28} {rho:>8.4f} {tau:>8.4f}")

print(f"\n  Occupancy sweep  GT=p  (p varies 0.02→0.50)")
print(f"  {'Configuration':<28} {'ρ':>8} {'τ':>8}")
print("  "+"-"*46)
feat_occ = {}
for name, fn in feat_fns.items():
    sc = [fn(img) for img in occ_imgs]
    rho, tau = metrics(sc, occ_gt)
    feat_occ[name] = (rho, tau, sc)
    print(f"  {name:<28} {rho:>8.4f} {tau:>8.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# C — RESOLUTION ABLATION  (occupancy sweep GT=p)
# ═════════════════════════════════════════════════════════════════════════════

print("\n"+"="*64)
print("C  RESOLUTION ABLATION  (occupancy sweep  GT=p)")
print(f"   Image size: {IMG_H}×{IMG_W}  (min block at 0.5mm/px = {max(1,int(10/0.5))}px)")
print("="*64)
print(f"\n  {'mm/px':<10} {'min block':>10} {'ρ':>8} {'τ':>8}")
print("  "+"-"*38)

res_factors = [0.5, 1.0, 2.0, 4.0, 8.0]
res_res = {}
for r in res_factors:
    min_blk = max(1, int(min(PHYSICAL_SCALES) / r))
    sc = [snocs_full(img, mm_per_pixel=r) for img in occ_imgs]
    rho, tau = metrics(sc, occ_gt)
    res_res[r] = (rho, tau, min_blk)
    print(f"  {r:<10.1f} {min_blk:>10d} {rho:>8.4f} {tau:>8.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURES — column-wise: A | B | C
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 10))
fig.suptitle(
    "Ablation Study — SNOCS  (1024×1024 images)\n"
    "Column A: Scale Ablation  |  Column B: Feature Type  |  Column C: Resolution",
    fontsize=12, fontweight="bold")
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── A1: Scale ablation bar (ρ) ────────────────────────────
ax = fig.add_subplot(gs[0, 0])
cfgs   = list(scale_res.keys())
rhos_a = [scale_res[c][0] for c in cfgs]
taus_a = [scale_res[c][1] for c in cfgs]
deltas = [scale_res[c][2] for c in cfgs]
cols_a = ["lightseagreen"] + [
    "#E24B4A" if d < -0.05 else
    "#BA7517" if d < 0     else
    "#3B8BD4" for d in deltas[1:]]
width = 0.35
ax.bar(np.array(range(len(cfgs)))-width/2, rhos_a, width, color=cols_a, label="Δρ")
ax.bar(np.array(range(len(cfgs)))+width/2, taus_a, width, color=cols_a, label="Δτ", alpha=0.45, hatch='///', edgecolor='white')
ax.set_xticks(range(len(cfgs)))
ax.set_xticklabels(cfgs, rotation=35, ha="right", fontsize=8)
ax.axhline(rho_b, color="gray", lw=1.2, ls="--",
           label=f"baseline ρ={rho_b:.3f}")
ax.set_ylabel("Spearman ρ")
ax.set_title("A1: Scale ablation — ρ\n(fragmentation GT=[1..6])")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(max(0, min(rhos_a)-0.15), 1.05)

# ── A2: Scale Δρ bar ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
dk   = [k for k in scale_res if k != "Baseline"]
dv   = [scale_res[k][2] for k in dk]
dtau = [scale_res[k][1] - tau_b for k in dk]
dc   = ["#E24B4A" if d < 0 else "#3B8BD4" for d in dv]
x_d  = np.arange(len(dk)); bw_d = 0.35
ax.bar(x_d - bw_d/2, dv,   bw_d, color=dc, label="Δρ")
ax.bar(x_d + bw_d/2, dtau, bw_d, color=dc, alpha=0.45, label="Δτ", hatch='///', edgecolor='white')
ax.set_xticks(x_d)
ax.set_xticklabels(dk, fontsize=9, rotation=20, ha="right")
ax.axhline(0, color="gray", lw=1.2)
ax.set_ylabel("Δ vs baseline"); ax.set_xlabel("Dropped scale")
ax.set_title("A2: Scale importance (Δρ, Δτ)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# ── B1: Feature type — fragmentation ─────────────────────
ax = fig.add_subplot(gs[0, 1])
fn   = list(feat_fns.keys())
frho = [feat_frag[n][0] for n in fn]
ftau = [feat_frag[n][1] for n in fn]
x_f  = np.arange(len(fn)); bw_f = 0.35
ax.bar(x_f - bw_f/2, frho, bw_f, color="lightseagreen", label="ρ")
ax.bar(x_f + bw_f/2, ftau, bw_f, color="cornflowerblue", alpha=0.45, label="τ", hatch='///', edgecolor='white')
ax.set_xticks(x_f); ax.set_xticklabels(fn, rotation=20, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("Correlation"); ax.set_ylim(-1.1, 1.1)
ax.set_title("B1: Feature type\n(fragmentation GT=[1..6])")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── B2: Feature type — occupancy ─────────────────────────
ax = fig.add_subplot(gs[1, 1])
orho = [feat_occ[n][0] for n in fn]
otau = [feat_occ[n][1] for n in fn]
ax.bar(x_f - bw_f/2, orho, bw_f, color="lightseagreen", label="ρ")
ax.bar(x_f + bw_f/2, otau, bw_f, color="cornflowerblue", alpha=0.45, label="τ", hatch='///', edgecolor='white')
ax.set_xticks(x_f); ax.set_xticklabels(fn, rotation=20, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("Correlation"); ax.set_ylim(-0.1, 1.1)
ax.set_title("B2: Feature type\n(occupancy sweep GT=p)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── C1: Resolution ρ and τ line ───────────────────────────
ax = fig.add_subplot(gs[0, 2])
rv   = list(res_res.keys())
rrho = [res_res[r][0] for r in rv]
rtau = [res_res[r][1] for r in rv]
ax.plot(rv, rrho, "o-", color="lightseagreen", lw=2, ms=8, label="Spearman ρ")
ax.plot(rv, rtau, "s--", color="cornflowerblue", lw=2, ms=7, label="Kendall τ")
ax.axvline(1.0, color="gray", lw=1.5, ls=":",
           label="baseline 1.0 mm/px")
ax.set_xscale("log"); ax.set_xlabel("mm / pixel (log scale)")
ax.set_ylabel("Correlation")
ax.set_title("C1: Resolution ablation\n(occupancy sweep GT=p)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# ── C2: Resolution — min block size ──────────────────────
ax = fig.add_subplot(gs[1, 2])
rblk = [res_res[r][2] for r in rv]
cols_r = ["#E24B4A" if b < 5 else "lightseagreen" for b in rblk]
ax.bar(range(len(rv)), rblk, color=cols_r, edgecolor="white")
for i, (b, r) in enumerate(zip(rblk, rrho)):
    ax.text(i, b+0.3, f"ρ={r:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(rv)))
ax.set_xticklabels([f"{r} mm/px" for r in rv], fontsize=8, rotation=20, ha="right")
ax.axhline(5, color="#E24B4A", lw=1.2, ls="--",
           label="min recommended block (5px)")
ax.set_ylabel("Min block size (px)")
ax.set_title("C2: Min block size per resolution\n(with ρ labels)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# ── Panel 4: score profiles per feature type — fragmentation
ax = fig.add_subplot(gs[0, 3])
for name, fn, col, ls, marker in [
    ("Full",     snocs_full, "lightseagreen", "-", "o"),
    ("Avg only", snocs_avg_only,  "cornflowerblue", "--", "s"),
    ("Max only", snocs_max_only,  "#E24B4A", ":", "^"),
]:
    scores = [fn(img) for img in frag_imgs]
    # normalise to [0,1] for visual comparison
    s = np.array(scores)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    ax.plot(range(6), s, color=col, ls=ls, lw=2, marker=marker, ms=6, label=name)
ax.set_xticks(range(6))
ax.set_xticklabels([l[:10] for l in frag_labels], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("Clutter Score")
ax.set_title("D1: Score profiles\n(fragmentation series)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 6: score profiles — occupancy sweep ────────────
ax = fig.add_subplot(gs[1, 3])
for name, fn, col, ls, marker in [
    ("Full",     snocs_full, "lightseagreen", "-", "o"),
    ("Avg only", snocs_avg_only,  "cornflowerblue", "--", "s"),
    ("Max only", snocs_max_only,  "#E24B4A", ":", "^"),
]:
    scores = [fn(img) for img in occ_imgs]
    ax.plot(occ_ps, scores, color=col, ls=ls, lw=2, marker=marker, ms=6, label=name)
ax.set_xlabel("Occupancy p")
ax.set_ylabel("Clutter score")
ax.set_title("D2: Score profiles\n(occupancy sweep)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

path = save_dir / "ablation_complete.tiff"
save_fig(fig, path, bw=False)
copy_figure(path, "figure 5.5.tiff")

plt.close(fig)
print(f"\n  Figure saved: {path}")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS FOR DOCX TABLE
# ═════════════════════════════════════════════════════════════════════════════

import json
results = {
    "scale": {k: {"rho": round(v[0],4), "tau": round(v[1],4),
                  "delta": round(v[2],4)} for k,v in scale_res.items()},
    "feat_frag": {k: {"rho": round(v[0],4), "tau": round(v[1],4)}
                  for k,v in feat_frag.items()},
    "feat_occ":  {k: {"rho": round(v[0],4), "tau": round(v[1],4)}
                  for k,v in feat_occ.items()},
    "resolution": {str(r): {"rho": round(v[0],4), "tau": round(v[1],4),
                             "min_block": v[2]} for r,v in res_res.items()},
}
with open(save_dir / "ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {save_dir / 'ablation_results.json'}")
print("Done.")
