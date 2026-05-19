#!/usr/bin/env python3
"""
analyze_vpr_ala_scan.py
=======================
Summarize and visualize the alanine-scan ipTM results for all 96 Vpr single
mutants vs ZNF430 KRAB, and compare to WT Vpr (ipTM = 0.76).

Outputs written to analysis/ala_scan/:
  ala_scan_scores.tsv       -- full table: pos, wt_aa, mut_aa, ipTM, ΔipTM
  ala_scan_waterfall.png    -- waterfall bar chart of ΔipTM, colored by residue class
  ala_scan_linear.png       -- linear plot along sequence with secondary structure hint
  hotspots.tsv              -- top 20 most disruptive mutations (ΔipTM < -0.15)

Usage:
    conda run -n colabfold python analyze_vpr_ala_scan.py
"""

import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
PRED_DIR = os.path.join(BASE_DIR, "predictions", "ala_scan")
WT_DIR   = os.path.join(BASE_DIR, "predictions", "Vpr_ZNF430-KRAB")
OUT_DIR  = os.path.join(BASE_DIR, "analysis", "ala_scan")
os.makedirs(OUT_DIR, exist_ok=True)

VPR_WT   = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
WT_IPTM  = 0.76   # pre-computed WT best-model ipTM

# Approximate Vpr secondary structure (HXB2 / NMR PDB 1M8L)
# H = helix, L = loop/turn, format: (start1, end1, label), 1-based
SS_REGIONS = [
    (17, 33,  "α1"),
    (38, 50,  "α2"),
    (55, 77,  "α3"),
]

# Residue class → color
AA_CLASSES = {
    "positive":   {"R", "K", "H"},
    "negative":   {"D", "E"},
    "aromatic":   {"F", "Y", "W"},
    "polar":      {"S", "T", "N", "Q", "C"},
    "hydrophobic":{"L", "I", "V", "M", "A"},
    "special":    {"G", "P"},
}
CLASS_COLORS = {
    "positive":    "#E41A1C",
    "negative":    "#377EB8",
    "aromatic":    "#984EA3",
    "polar":       "#4DAF4A",
    "hydrophobic": "#FF7F00",
    "special":     "#A65628",
}

def aa_class(aa):
    for cls, members in AA_CLASSES.items():
        if aa in members:
            return cls
    return "special"


def best_iptm(pred_dir):
    best = np.nan
    for f in glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")):
        try:
            d = json.load(open(f))
            v = d.get("iptm", np.nan)
            if np.isnan(best) or v > best:
                best = v
        except Exception:
            pass
    return best


def main():
    rows = []
    n_missing = 0

    for pos0, wt_aa in enumerate(VPR_WT):
        pos1   = pos0 + 1
        mut_aa = "G" if wt_aa == "A" else "A"
        label  = f"Vpr_{wt_aa}{pos1}{mut_aa}_ZNF430-KRAB"
        out_dir = os.path.join(PRED_DIR, label)

        iptm = best_iptm(out_dir)
        if np.isnan(iptm):
            n_missing += 1

        rows.append({
            "pos":      pos1,
            "wt_aa":    wt_aa,
            "mut_aa":   mut_aa,
            "mutation": f"{wt_aa}{pos1}{mut_aa}",
            "iptm":     iptm,
            "delta_iptm": iptm - WT_IPTM,
            "aa_class": aa_class(wt_aa),
            "complete": not np.isnan(iptm),
        })

    df = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
    tsv = os.path.join(OUT_DIR, "ala_scan_scores.tsv")
    df.to_csv(tsv, sep="\t", index=False, float_format="%.4f")
    print(f"Total: {len(df)}  |  Complete: {df['complete'].sum()}  |  Missing: {n_missing}")
    if n_missing:
        missing = df[~df["complete"]]["mutation"].tolist()
        print(f"  Missing: {missing[:10]}{'...' if len(missing)>10 else ''}")

    df_done = df[df["complete"]].copy()
    if df_done.empty:
        print("No completed predictions yet.")
        return

    # ── Hotspots table ─────────────────────────────────────────────────────────
    hotspots = df_done.sort_values("delta_iptm").head(20)
    hotspots.to_csv(os.path.join(OUT_DIR, "hotspots.tsv"), sep="\t",
                    index=False, float_format="%.4f")
    print("\nTop 10 most disruptive mutations (ΔipTM):")
    print(hotspots[["mutation","wt_aa","aa_class","iptm","delta_iptm"]]
          .head(10).to_string(index=False))

    # ── Waterfall bar chart ────────────────────────────────────────────────────
    df_sorted = df_done.sort_values("delta_iptm")
    colors = [CLASS_COLORS[c] for c in df_sorted["aa_class"]]

    fig, ax = plt.subplots(figsize=(max(12, len(df_done) * 0.18), 5))
    bars = ax.bar(range(len(df_sorted)), df_sorted["delta_iptm"],
                  color=colors, edgecolor="none", width=0.85)

    # Label the most disruptive ones
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        if row["delta_iptm"] < -0.15:
            ax.text(i, row["delta_iptm"] - 0.01, row["mutation"],
                    ha="center", va="top", fontsize=5.5, rotation=90)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-0.15, color="gray", linestyle="--", linewidth=0.7,
               label="ΔipTM = −0.15 (hotspot threshold)")
    ax.set_xlabel("Vpr single-Ala mutant (ranked by ΔipTM)")
    ax.set_ylabel("ΔipTM vs. WT")
    ax.set_title(f"Alanine scan: Vpr × ZNF430-KRAB  (WT ipTM = {WT_IPTM:.2f})\n"
                 f"{df['complete'].sum()} / {len(df)} positions complete")
    ax.set_xticks([])

    legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c)
                      for c in CLASS_COLORS]
    legend_patches.append(mpatches.Patch(color="gray", label="ΔipTM = −0.15"))
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ala_scan_waterfall.png"), dpi=150)
    plt.close()
    print("\nSaved: ala_scan_waterfall.png")

    # ── Linear sequence plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))

    # WT ipTM reference line
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-0.15, color="gray", linestyle="--", linewidth=0.7)

    # Secondary structure shading
    for (s, e, name) in SS_REGIONS:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.10, color="steelblue", zorder=0)
        ax.text((s + e) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.1 else -0.05,
                name, ha="center", va="top", fontsize=8, color="steelblue")

    for _, row in df_done.iterrows():
        color = CLASS_COLORS[row["aa_class"]]
        ax.bar(row["pos"], row["delta_iptm"], color=color, width=0.8,
               edgecolor="none", zorder=3)

    # Label hotspots
    for _, row in df_done[df_done["delta_iptm"] < -0.15].iterrows():
        ax.text(row["pos"], row["delta_iptm"] - 0.01, row["mutation"],
                ha="center", va="top", fontsize=6, rotation=90)

    ax.set_xlabel("Vpr residue position")
    ax.set_ylabel("ΔipTM vs. WT")
    ax.set_title(f"Alanine scan along Vpr sequence (WT ipTM = {WT_IPTM:.2f})\n"
                 "Blue shading = predicted helices (α1/α2/α3)")
    ax.set_xlim(0, 97)
    ax.set_xticks(range(1, 97, 5))
    ax.grid(axis="y", alpha=0.2)

    legend_patches = [mpatches.Patch(color=CLASS_COLORS[c], label=c)
                      for c in CLASS_COLORS]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ala_scan_linear.png"), dpi=150)
    plt.close()
    print("Saved: ala_scan_linear.png")
    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
