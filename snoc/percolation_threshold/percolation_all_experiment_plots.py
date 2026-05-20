"""
Percolation Validation — Plots
Reads percolation_summary.csv and produces figure
"""

import pathlib

import numpy as np
import matplotlib

from utils import copy_figure, save_fig
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import csv, os

# ── data ──────────────────────────────────────────────────────────────────────
root_dir = pathlib.Path(__file__).parent / "results"
root_dir.mkdir(exist_ok=True)
CSV_PATH = root_dir / "percolation_summary.csv"
OUT_PNG  = root_dir / "percolation_summary.png"
PC       = 0.5927   # theoretical threshold

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

rows = load(CSV_PATH)

def v(row, key):
    return float(row[key.strip()])

data = []
for row in rows:
    sz_raw = row["image_size"].strip().strip('"()').replace(" ","")
    h, w   = [int(x) for x in sz_raw.split(",")]
    data.append({
        "label"     : f"{h}×{w}",
        "pixels"    : h * w / 1e6,
        "n_ps"      : int(row["n_ps"]),
        "n_seeds"   : int(row["n_seeds"]),
        "inf"       : float(row["Inflection point (p̂c=d²C/dp²) "]),
        "error"     : float(row["Error (Δpc=|p̂c-pc|)"]),
        "error_pct" : float(row["Error% (Δpc/pc*100)"]),
        "peak"      : float(row["Peak gradient (dC/dp)"]),
        "rho"       : float(row["spearman ρ largest-cluster fraction"]),
        "rho_p"     : float(row["spearman ρ p-val"]),
        "tau"       : float(row["kendall τ largest-cluster fraction"]),
        "tau_p"     : float(row["kendall τ p-val"]),
        "ms"        : float(row["time_millisec"]),
    })

# ── BW-safe style elements ────────────────────────────────────────────────────
# Each experiment gets: marker, hatch, linestyle, grayscale shade
STYLES = [
    {"marker":"o", "hatch":"///",  "ls":"-", 'color':"cornflowerblue",  "gray":"0.15", "fill":"0.75"},
    {"marker":"s", "hatch":"||",  "ls":"--", 'color':"orange",  "gray":"0.40", "fill":"0.88"},
    {"marker":"^", "hatch":"...",  "ls":":", 'color':"lightseagreen",   "gray":"0.65", "fill":"0.60"},
]

# ── FIGURE ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "Percolation Threshold Validation — SNOCS Clutter Score\n"
    f"Theoretical $p_c$ = {PC}  (2D square lattice, infinite limit)",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

labels = [d["label"] for d in data]
x      = np.arange(len(data))
bw     = 0.28

# ── Panel 1: Inflection point bar + error bars ────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for i, d in enumerate(data):
    s = STYLES[i]
    ax.bar(i, d["inf"], color=s["color"], edgecolor="white",
           hatch=s["hatch"], lw=1.2, zorder=2, width=0.55,
           label=d["label"])
    ax.errorbar(i, d["inf"], yerr=0, fmt=s["marker"],
                color="white", ms=8, zorder=4, capsize=4)
    ax.text(i, d["inf"] + 0.001, f"{d['inf']:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.axhline(PC, color="red", lw=2, ls="--", label=f"$p_c$ = {PC}", zorder=5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Detected $\\hat{p}_c$")
ax.set_title("A: Inflection $\\hat{p}_c$\n(zero-crossing of $d^2C/dp^2$)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0.56, 0.62)

# ── Panel 2: Error bar chart ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
for i, d in enumerate(data):
    s = STYLES[i]
    ax.bar(i, d["error"], color=s["color"], edgecolor="white",
           hatch=s["hatch"], lw=1.2, width=0.55)
    ax.text(i, d["error"] + 2e-4,
            f"{d['error']:.4f}\n({d['error_pct']:.2f}%)",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("$\\Delta p=|\\hat{p}_c - p_c|$")
ax.set_title("B: Absolute error vs $p_c$ (lower = better)\n")
ax.grid(True, alpha=0.3, axis="y")

# ── Panel 3: ρ and τ grouped bars ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for i, d in enumerate(data):
    s = STYLES[i]
    ax.bar(i - bw/2 - 0.02, d["rho"], bw,
           color=s["color"], edgecolor="white",
           hatch=s["hatch"], lw=1.2, label=d["label"])
    ax.bar(i + bw/2 + 0.02, d["tau"], bw,
           color="white",   edgecolor="white",
           hatch=s["hatch"], lw=1.2)
    ax.text(i - bw/2 - 0.02, d["rho"] + 5e-5,
            f"ρ={d['rho']:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(i + bw/2 + 0.02, d["tau"] + 5e-5,
            f"τ={d['tau']:.4f}", ha="center", va="bottom", fontsize=7)

ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Correlation vs LCF")
ax.set_title("C: Spearman $\\rho$ (filled) and\nKendall $\\tau$ (open)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0.996, 1.001)

# ── Panel 4: Inflection vs image size (line) ──────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
pixels = [d["pixels"] for d in data]
infs   = [d["inf"]    for d in data]
for i, d in enumerate(data):
    s = STYLES[i]
    ax.plot(d["pixels"], d["inf"], marker=s["marker"],
            ms=10, color=s["color"], zorder=4,
            linestyle="none", markerfacecolor=s["color"],
            markeredgewidth=1.5, label=d["label"])
ax.plot(pixels, infs, ls="--", color="0.5", lw=1.2, zorder=2)
ax.axhline(PC, color="red", lw=1.8, ls="--", label=f"$p_c$={PC}")
ax.set_xlabel("Image size (Megapixel-MPx)")
ax.set_ylabel("Detected $\\hat{p}_c$")
ax.set_title("D: Inflection vs image size\n(convergence to $p_c$)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(0.56, 0.62)

# ── Panel 5: Runtime bar ──────────────────────────────────────────────────────
ax  = fig.add_subplot(gs[1, 1])
ax2 = ax.twinx()
for i, d in enumerate(data):
    s = STYLES[i]
    ax.bar(i, d["ms"], color=s["color"], edgecolor="white",
           hatch=s["hatch"], lw=1.2, width=0.55)
    ax.text(i, d["ms"] - 0.5, f"{d['ms']:.2f} ms",
            ha="center", va="bottom", fontsize=9)
ax2.plot(x, pixels, marker="D", ms=7, color="black",
         ls="--", lw=1.2, label="Image Mpx")
for i, p in enumerate(pixels):
    ax2.text(i, p + 0.1, f"{p:.2f} MPx", ha="center",
             va="bottom", fontsize=7.5, color="black")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Processing time (ms/image)")
ax2.set_ylabel("Image size (Mpx)")
ax.set_title("E: Runtime vs image size")
ax2.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

# ── Panel 6: Scatter error vs correlation ─────────────────────────────────────
# ax = fig.add_subplot(gs[1, 2])
# ax2 = ax.twinx()
# for i, d in enumerate(data):
#     s = STYLES[i]
#     ax.scatter(d["error"], d["rho"], s=200,
#                marker='s', hatch=s['hatch'], facecolors=s["color"],
#                edgecolors="white", linewidths=2, zorder=4)
#     ax.annotate(d["rho"], (d["error"], d["rho"]),
#                 textcoords="offset points", xytext=(5, -12),
#                 fontsize=8.5, color="black")
# ax.set_yticks([])
# ax2.set_ylim(0.99996, 1.0001)
# ax.set_xlabel("$|\\hat{p}_c - p_c|$ (absolute error)")
# ax2.set_ylabel("Spearman $\\rho$ vs LCF")
# ax.set_title("F: Error vs correlation\n(ideal: bottom-left)")
# ax.grid(True, alpha=0.3)


# global legend — BW print key
patches = []
for i, (d, s) in enumerate(zip(data, STYLES)):
    patches.append(mpatches.Patch(
        facecolor=s["color"], edgecolor="white",
        hatch=s["hatch"], lw=1,
        label=f"{d['label']}  ({d['n_seeds']} seeds, {d['n_ps']} p-values)"
    ))
patches.append(mpatches.Patch(facecolor="white", edgecolor="red",
                               hatch="", lw=2, ls="--",
                               label=f"Theoretical $p_c$ = {PC}"))
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=8.5, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.015))

save_fig(fig, OUT_PNG, bw=True)
copy_figure(OUT_PNG, "figure 5.5.png")
plt.close(fig)
print(f"Figure saved: {OUT_PNG}")
