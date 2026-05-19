#!/usr/bin/env python3
"""
plot_top10_disruptors.py
========================
Figure: Top 10 Vpr mutants that disrupt Vpr–KRAB interaction.

Panel A: Box plot of ipTM across 39 KZFPs for WT + top-10 mutants,
         with individual KZFP points overlaid.
Panel B: Dot plot (mean ± SD) for each mutant vs. WT, ranked by mean ipTM.

Output: analysis/top10_disruptors.png
"""

import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
PRED_DIR = os.path.join(BASE_DIR, "predictions")
OUT_DIR  = os.path.join(BASE_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

def get_wt_iptms():
    wt = {}
    for d in sorted(glob.glob(os.path.join(PRED_DIR, "Vpr_*-KRAB"))):
        gene = os.path.basename(d).replace("Vpr_", "").replace("-KRAB", "")
        f = sorted(glob.glob(os.path.join(d, "*scores*rank_001*.json")))
        if f:
            v = json.load(open(f[0])).get("iptm", np.nan)
            if v >= 0.7:
                wt[gene] = v
    return wt

wt_iptms = get_wt_iptms()   # {gene: wt_iptm}  39 KZFPs
kzfps    = sorted(wt_iptms.keys())

# Load raw ipTM matrix (20 mutants × 39 KZFPs)
mat = pd.read_csv(os.path.join(BASE_DIR, "analysis", "mutant_scan", "iptm_matrix.tsv"),
                  sep="\t", index_col=0)

# Top 10 by lowest mean ipTM across KZFPs
mean_iptm = mat.mean(axis=1).sort_values()
top10 = mean_iptm.head(10).index.tolist()

# Build tidy frame: WT + top10 mutants
wt_series = pd.Series(wt_iptms, name="WT")[kzfps]
plot_labels = ["WT"] + top10
data_dict   = {"WT": wt_series.values}
for m in top10:
    data_dict[m] = mat.loc[m, kzfps].values

# ── Colour scheme ──────────────────────────────────────────────────────────────
WT_COLOR  = "#2166AC"
MUT_CMAP  = plt.cm.RdYlGn_r   # red = most disruptive (left)
mut_colors = [MUT_CMAP(i / (len(top10) - 1)) for i in range(len(top10))]
colors = [WT_COLOR] + mut_colors

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                         gridspec_kw={"width_ratios": [2, 1]})
fig.subplots_adjust(wspace=0.35)

# ── Panel A: Box + strip ───────────────────────────────────────────────────────
ax = axes[0]
n  = len(plot_labels)

for i, (label, color) in enumerate(zip(plot_labels, colors)):
    vals = data_dict[label]
    # Box
    bp = ax.boxplot(vals, positions=[i], widths=0.55,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="", linestyle="none"),
                    showfliers=False)
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.75)

    # Jittered strip of individual KZFP points
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               color=color, alpha=0.55, s=22, zorder=3,
               edgecolors="white", linewidths=0.4)

# WT mean reference line
wt_mean = np.mean(data_dict["WT"])
ax.axhline(wt_mean, color=WT_COLOR, linewidth=1.2, linestyle="--",
           alpha=0.6, zorder=1, label=f"WT mean = {wt_mean:.2f}")

ax.set_xticks(range(n))
ax.set_xticklabels(plot_labels, rotation=40, ha="right", fontsize=10)
ax.set_ylabel("ipTM (AlphaFold2-Multimer)", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("Top 10 Vpr mutants disrupting KRAB binding\n"
             f"(distribution across {len(kzfps)} KZFPs with WT ipTM ≥ 0.70)",
             fontsize=11)
ax.tick_params(axis="x", which="both", length=0)
ax.grid(axis="y", alpha=0.25, linewidth=0.7)
ax.legend(fontsize=9, loc="lower right")

# Annotate mean ipTM value above each box
for i, label in enumerate(plot_labels):
    m = np.mean(data_dict[label])
    ax.text(i, m + 0.035, f"{m:.2f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold",
            color="black" if label == "WT" else "dimgray")

# ── Panel B: Mean ± SD dot plot ────────────────────────────────────────────────
ax2 = axes[1]

# Only mutants (no WT), ranked most→least disruptive
means = [np.mean(data_dict[m]) for m in top10]
sds   = [np.std(data_dict[m])  for m in top10]

y_pos = np.arange(len(top10))

ax2.barh(y_pos, means, xerr=sds, color=mut_colors, alpha=0.75,
         edgecolor="white", linewidth=0.5,
         error_kw=dict(elinewidth=1.2, capsize=3, ecolor="gray"))

# WT reference
ax2.axvline(wt_mean, color=WT_COLOR, linewidth=1.5, linestyle="--",
            label=f"WT mean ({wt_mean:.2f})")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(top10, fontsize=10)
ax2.set_xlabel("Mean ipTM across 39 KZFPs", fontsize=11)
ax2.set_xlim(0, 0.90)
ax2.set_title("Mean ipTM ± SD\n(ranked most → least disruptive)", fontsize=11)
ax2.grid(axis="x", alpha=0.25, linewidth=0.7)
ax2.legend(fontsize=9, loc="lower right")

# Annotate mean values
for i, (m, sd) in enumerate(zip(means, sds)):
    ax2.text(m + sd + 0.01, i, f"{m:.2f}", va="center", fontsize=8)

fig.suptitle("Vpr alanine-scan hotspot mutants: effect on KRAB binding",
             fontsize=13, fontweight="bold", y=1.01)

out_path = os.path.join(OUT_DIR, "top10_disruptors.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
