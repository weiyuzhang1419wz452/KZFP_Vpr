#!/usr/bin/env python3
"""
analyze_mutant_kzfp_scan.py
============================
Summarize and visualize the 20-mutant × 39-KZFP cross-scan results.
For each mutant, compares its ipTM against each KZFP KRAB to the WT ipTM
for the same KZFP, yielding a ΔipTM matrix.

Outputs written to analysis/mutant_scan/:
  delta_iptm_matrix.tsv   -- ΔipTM(mutant, KZFP) matrix
  delta_iptm_heatmap.png  -- clustered heatmap
  per_mutant_boxplot.png  -- distribution of ΔipTM per Vpr mutant (across KZFPs)
  per_kzfp_boxplot.png    -- distribution of ΔipTM per KZFP (across mutants)
  universal_hotspots.tsv  -- mutants that disrupt ALL KZFPs (mean ΔipTM < -0.2)

Usage:
    conda run -n colabfold python analyze_mutant_kzfp_scan.py
"""

import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

BASE_DIR  = "/local/workdir/wz452/Claude/KZFP_Vpr"
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
OUT_DIR   = os.path.join(BASE_DIR, "analysis", "mutant_scan")
os.makedirs(OUT_DIR, exist_ok=True)

VPR_WT = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
IPTM_THRESHOLD = 0.7


def best_iptm(pred_dir):
    for f in glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")):
        try:
            return json.load(open(f)).get("iptm", np.nan)
        except Exception:
            pass
    return np.nan


def get_wt_iptms():
    """Return {gene: wt_iptm} for all KZFPs with WT ipTM >= threshold."""
    wt = {}
    for d in glob.glob(os.path.join(PRED_DIR, "Vpr_*-KRAB")):
        gene = os.path.basename(d).replace("Vpr_", "").replace("-KRAB", "")
        v = best_iptm(d)
        if not np.isnan(v) and v >= IPTM_THRESHOLD:
            wt[gene] = v
    return wt


def get_top_mutant_labels(n=20):
    tsv = os.path.join(BASE_DIR, "analysis", "ala_scan", "ala_scan_scores.tsv")
    df  = pd.read_csv(tsv, sep="\t")
    return df[df["complete"]].sort_values("delta_iptm").head(n)["mutation"].tolist()


def main():
    wt_iptms = get_wt_iptms()
    mutants  = get_top_mutant_labels(20)
    kzfps    = sorted(wt_iptms.keys(), key=lambda g: -wt_iptms[g])

    print(f"Mutants: {len(mutants)}  |  KZFPs: {len(kzfps)}")

    scan_dir = os.path.join(PRED_DIR, "mutant_scan")

    # Build raw ipTM matrix and ΔipTM matrix
    raw_mat   = pd.DataFrame(np.nan, index=mutants, columns=kzfps)
    delta_mat = pd.DataFrame(np.nan, index=mutants, columns=kzfps)
    n_done = n_miss = 0

    for mut in mutants:
        for gene in kzfps:
            d = os.path.join(scan_dir, f"Vpr_{mut}_{gene}-KRAB")
            v = best_iptm(d)
            if np.isnan(v):
                n_miss += 1
            else:
                n_done += 1
                raw_mat.loc[mut, gene]   = v
                delta_mat.loc[mut, gene] = v - wt_iptms[gene]

    total = len(mutants) * len(kzfps)
    print(f"Complete: {n_done}/{total}  |  Missing: {n_miss}")

    if n_done == 0:
        print("No completed predictions yet.")
        return

    # Save tables
    raw_mat.to_csv(os.path.join(OUT_DIR, "iptm_matrix.tsv"), sep="\t",
                   float_format="%.4f")
    delta_mat.to_csv(os.path.join(OUT_DIR, "delta_iptm_matrix.tsv"), sep="\t",
                     float_format="%.4f")
    print("Saved: delta_iptm_matrix.tsv")

    # ── Heatmap ────────────────────────────────────────────────────────────────
    # Only cluster rows/cols with enough data
    dmat = delta_mat.copy()
    dmat_filled = dmat.fillna(0)

    # Hierarchical clustering of rows (mutants) and cols (KZFPs)
    try:
        row_order = leaves_list(linkage(pdist(dmat_filled.values), method="average"))
        col_order = leaves_list(linkage(pdist(dmat_filled.values.T), method="average"))
    except Exception:
        row_order = list(range(len(mutants)))
        col_order = list(range(len(kzfps)))

    dmat_clust = dmat_filled.iloc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(max(14, len(kzfps) * 0.38),
                                    max(7,  len(mutants) * 0.42)))
    im = ax.imshow(dmat_clust.values, aspect="auto", cmap="RdYlGn",
                   vmin=-0.6, vmax=0.2, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="ΔipTM vs. WT Vpr", fraction=0.02, pad=0.02)

    ax.set_xticks(range(len(dmat_clust.columns)))
    ax.set_xticklabels(dmat_clust.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(dmat_clust.index)))
    ax.set_yticklabels(dmat_clust.index, fontsize=8)

    # Annotate each cell with ΔipTM value
    for i in range(len(dmat_clust.index)):
        for j in range(len(dmat_clust.columns)):
            v = dmat_clust.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=5, color="black" if abs(v) < 0.35 else "white")

    ax.set_title(f"ΔipTM: Top-20 Vpr hotspot mutants × {len(kzfps)} KZFPs\n"
                 f"({n_done}/{total} predictions complete)  Red = disrupted  Green = enhanced",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delta_iptm_heatmap.png"), dpi=150)
    plt.close()
    print("Saved: delta_iptm_heatmap.png")

    # ── Per-mutant boxplot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    data_per_mut = [dmat.loc[m].dropna().values for m in mutants]
    means = [np.mean(d) if len(d) else np.nan for d in data_per_mut]
    order = np.argsort(means)

    bp = ax.boxplot([data_per_mut[i] for i in order],
                    labels=[mutants[i] for i in order],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4878CF")
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-0.2, color="gray", linestyle="--", linewidth=0.7,
               label="ΔipTM = −0.2 threshold")
    ax.set_ylabel("ΔipTM vs. WT Vpr")
    ax.set_title("Per-mutant ΔipTM distribution across all 39 KZFPs\n"
                 "(ranked by mean ΔipTM)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_mutant_boxplot.png"), dpi=150)
    plt.close()
    print("Saved: per_mutant_boxplot.png")

    # ── Per-KZFP boxplot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(kzfps) * 0.4), 5))
    data_per_kzfp = [dmat[g].dropna().values for g in kzfps]
    means_k = [np.mean(d) if len(d) else np.nan for d in data_per_kzfp]
    order_k = np.argsort(means_k)

    bp = ax.boxplot([data_per_kzfp[i] for i in order_k],
                    labels=[kzfps[i] for i in order_k],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, i in zip(bp["boxes"], order_k):
        wt_v = wt_iptms.get(kzfps[i], 0)
        cmap = plt.cm.YlOrRd
        patch.set_facecolor(cmap((wt_v - 0.7) / 0.15))
        patch.set_alpha(0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔipTM vs. WT Vpr")
    ax.set_title("Per-KZFP ΔipTM distribution across all 20 Vpr hotspot mutants\n"
                 "(ranked by mean ΔipTM; color = WT ipTM)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_kzfp_boxplot.png"), dpi=150)
    plt.close()
    print("Saved: per_kzfp_boxplot.png")

    # ── Universal hotspots ────────────────────────────────────────────────────
    mean_delta = dmat.mean(axis=1).sort_values()
    frac_disrupt = (dmat < -0.2).sum(axis=1) / dmat.notna().sum(axis=1)
    summary = pd.DataFrame({
        "mean_delta_iptm": mean_delta,
        "frac_kzfps_disrupted": frac_disrupt,
        "n_kzfps_tested": dmat.notna().sum(axis=1),
    }).sort_values("mean_delta_iptm")
    summary.to_csv(os.path.join(OUT_DIR, "universal_hotspots.tsv"), sep="\t",
                   float_format="%.4f")

    print("\nUniversal hotspots (mean ΔipTM across all KZFPs):")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
