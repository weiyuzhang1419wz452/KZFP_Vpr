#!/usr/bin/env python3
"""
analyze_vpr_mutants.py
======================
Compare AlphaFold2-Multimer predictions for wild-type Vpr vs. E24R, R36P,
and E24R+R36P double mutant, each paired with ZNF430 KRAB domain (aa35-107).

Outputs (written to analysis/vpr_mutants/):
  scores_table.tsv        -- ipTM / ptm / max_PAE for all 4 variants
  iptm_barplot.png        -- bar chart of best-model ipTM
  pae_comparison.png      -- 2x2 inter-chain PAE heatmaps (all 5 models averaged)
  contacts_wt.tsv         -- interface contacts in WT best model
  contacts_lost.tsv       -- WT contacts absent in the double mutant
  interface_plddt.png     -- per-residue pLDDT at the interface for WT vs. dbl

Usage:
    conda run -n colabfold python analyze_vpr_mutants.py
"""

import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = "/local/workdir/wz452/Claude/KZFP_Vpr"
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
OUT_DIR   = os.path.join(BASE_DIR, "analysis", "vpr_mutants")
os.makedirs(OUT_DIR, exist_ok=True)

# Vpr is 96 aa, ZNF430-KRAB is 73 aa → split index in concatenated coords
VPR_LEN  = 96
KRAB_LEN = 73   # aa35-107 = 73 residues

VARIANTS = {
    "WT":        "Vpr_ZNF430-KRAB",
    "E24R":      "VprE24R_ZNF430-KRAB",
    "R36P":      "VprR36P_ZNF430-KRAB",
    "E24R+R36P": "VprE24R-R36P_ZNF430-KRAB",
}

COLORS = {
    "WT":        "#4878CF",
    "E24R":      "#F0A500",
    "R36P":      "#E8735A",
    "E24R+R36P": "#D62728",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_scores(pred_dir):
    """Return list of dicts (one per ranked model) with iptm, ptm, max_pae."""
    models = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*scores*rank_*.json"))):
        try:
            d = json.load(open(f))
            rank = int(os.path.basename(f).split("rank_")[1].split("_")[0])
            models.append({
                "rank":    rank,
                "iptm":    d.get("iptm", np.nan),
                "ptm":     d.get("ptm",  np.nan),
                "max_pae": d.get("max_pae", np.nan),
                "pae":     np.array(d["pae"]) if "pae" in d else None,
                "plddt":   np.array(d["plddt"]) if "plddt" in d else None,
                "file":    f,
            })
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    models.sort(key=lambda x: x["rank"])
    return models


def best_model(models):
    """Return the rank-001 model dict, or None."""
    for m in models:
        if m["rank"] == 1:
            return m
    return models[0] if models else None


def mean_interchain_pae(pae_matrix, vpr_len=VPR_LEN):
    """
    Return the mean off-diagonal (inter-chain) PAE block.
    In ColabFold multimer output, chain A = Vpr (rows/cols 0:vpr_len),
    chain B = KRAB (rows/cols vpr_len:).
    """
    if pae_matrix is None:
        return np.nan
    a_to_b = pae_matrix[:vpr_len, vpr_len:]
    b_to_a = pae_matrix[vpr_len:, :vpr_len]
    return float(np.mean(np.concatenate([a_to_b.ravel(), b_to_a.ravel()])))


def interchain_pae_block(pae_matrix, vpr_len=VPR_LEN):
    """Return the two off-diagonal blocks averaged: shape (vpr_len, krab_len)."""
    if pae_matrix is None:
        return None
    a_to_b = pae_matrix[:vpr_len, vpr_len:]
    b_to_a = pae_matrix[vpr_len:, :vpr_len].T
    return (a_to_b + b_to_a) / 2.0


def parse_pdb_contacts(pdb_path, vpr_len=VPR_LEN, dist_cutoff=5.0):
    """
    Parse a ColabFold PDB and return a list of (vpr_resnum, krab_resnum, dist_A)
    interface contacts where at least one heavy-atom pair is within dist_cutoff Å.
    Chain A = Vpr, Chain B = KRAB.
    """
    from collections import defaultdict
    coords = defaultdict(list)   # (chain, resnum) → [(x,y,z), ...]
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                chain  = line[21]
                resnum = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords[(chain, resnum)].append(np.array([x, y, z]))
    except Exception as e:
        print(f"  PDB parse error: {e}")
        return []

    a_res = sorted(r for (c, r) in coords if c == "A")
    b_res = sorted(r for (c, r) in coords if c == "B")

    contacts = []
    for ra in a_res:
        ca = np.array(coords[("A", ra)])
        for rb in b_res:
            cb = np.array(coords[("B", rb)])
            # min heavy-atom distance
            dists = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
            if dists.min() <= dist_cutoff:
                contacts.append((ra, rb, float(dists.min())))
    return contacts


def best_pdb(pred_dir):
    pdbs = sorted(glob.glob(os.path.join(pred_dir, "*rank_001*.pdb")))
    return pdbs[0] if pdbs else None


# ── Main analysis ──────────────────────────────────────────────────────────────

def main():
    # ── 1. Load scores ────────────────────────────────────────────────────────
    data = {}
    for label, dirname in VARIANTS.items():
        d = os.path.join(PRED_DIR, dirname)
        if not os.path.isdir(d):
            print(f"  Missing prediction dir: {d}")
            data[label] = []
            continue
        models = load_scores(d)
        if not models:
            print(f"  No score files found for {label}")
        data[label] = models

    # ── 2. Summary table ──────────────────────────────────────────────────────
    rows = []
    for label, models in data.items():
        if not models:
            rows.append({"variant": label, "best_iptm": np.nan,
                         "mean_iptm": np.nan, "best_ptm": np.nan,
                         "best_max_pae": np.nan, "mean_interchain_pae": np.nan})
            continue
        bm = best_model(models)
        iptms = [m["iptm"] for m in models if not np.isnan(m["iptm"])]
        rows.append({
            "variant":            label,
            "best_iptm":          bm["iptm"],
            "mean_iptm":          float(np.mean(iptms)) if iptms else np.nan,
            "best_ptm":           bm["ptm"],
            "best_max_pae":       bm["max_pae"],
            "mean_interchain_pae": mean_interchain_pae(bm["pae"]),
        })

    df = pd.DataFrame(rows).set_index("variant")
    tsv_path = os.path.join(OUT_DIR, "scores_table.tsv")
    df.to_csv(tsv_path, sep="\t", float_format="%.3f")
    print("\nScores table:")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {tsv_path}")

    # ── 3. ipTM bar plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    labels   = list(VARIANTS.keys())
    iptm_vals = [df.loc[l, "best_iptm"] if l in df.index else np.nan for l in labels]
    icp_vals  = [df.loc[l, "mean_interchain_pae"] if l in df.index else np.nan
                 for l in labels]
    colors = [COLORS[l] for l in labels]

    bars = ax.bar(labels, iptm_vals, color=colors, edgecolor="black", linewidth=0.8,
                  width=0.6, zorder=3)
    for bar, v in zip(bars, iptm_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="ipTM = 0.5 (interaction threshold)")
    ax.set_ylabel("Best-model ipTM")
    ax.set_title("Vpr × ZNF430-KRAB: WT vs. Vpr mutants")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "iptm_barplot.png"), dpi=150)
    plt.close()
    print("Saved: iptm_barplot.png")

    # ── 4. Inter-chain PAE heatmaps ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    vmax = 30.0

    for ax, (label, models) in zip(axes, data.items()):
        if not models:
            ax.set_title(f"{label}\n(no data)")
            continue
        bm = best_model(models)
        block = interchain_pae_block(bm["pae"])
        if block is None:
            ax.set_title(f"{label}\n(no PAE)")
            continue
        im = ax.imshow(block.T, origin="lower", aspect="auto",
                       cmap="RdYlGn_r", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="PAE (Å)")
        ax.set_xlabel("Vpr residue")
        ax.set_ylabel("ZNF430-KRAB residue")
        iptm_v = bm["iptm"]
        ax.set_title(f"{label}  (ipTM = {iptm_v:.3f})")

        # Mark mutated positions on x-axis (Vpr residues 24 and 36, 0-indexed: 23, 35)
        for pos, name in [(23, "E24R"), (35, "R36P")]:
            ax.axvline(pos, color="white", linewidth=1.5, alpha=0.8)
            ax.text(pos, block.shape[1] * 0.95, name, color="white",
                    fontsize=7, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5))

    fig.suptitle("Inter-chain PAE: Vpr (x-axis) vs ZNF430-KRAB (y-axis)\nLower = more confident interaction",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pae_comparison.png"), dpi=150)
    plt.close()
    print("Saved: pae_comparison.png")

    # ── 5. Interface contacts ──────────────────────────────────────────────────
    wt_pdb  = best_pdb(os.path.join(PRED_DIR, VARIANTS["WT"]))
    dbl_pdb = best_pdb(os.path.join(PRED_DIR, VARIANTS["E24R+R36P"]))

    wt_contacts = []
    if wt_pdb:
        wt_contacts = parse_pdb_contacts(wt_pdb)
        wt_df = pd.DataFrame(wt_contacts,
                             columns=["Vpr_resnum", "KRAB_resnum", "min_dist_A"])
        wt_df.to_csv(os.path.join(OUT_DIR, "contacts_wt.tsv"), sep="\t", index=False)
        print(f"WT contacts: {len(wt_contacts)}")
    else:
        print("WT PDB not found — skipping contact analysis")

    if dbl_pdb and wt_contacts:
        dbl_contacts = parse_pdb_contacts(dbl_pdb)
        dbl_pairs = {(c[0], c[1]) for c in dbl_contacts}
        lost = [c for c in wt_contacts if (c[0], c[1]) not in dbl_pairs]
        lost_df = pd.DataFrame(lost, columns=["Vpr_resnum", "KRAB_resnum", "min_dist_A"])
        lost_df.to_csv(os.path.join(OUT_DIR, "contacts_lost.tsv"), sep="\t", index=False)
        print(f"Contacts lost in E24R+R36P double mutant: {len(lost)}")

    # ── 6. Interface pLDDT comparison (WT vs double mutant) ───────────────────
    wt_bm  = best_model(data.get("WT", []))
    dbl_bm = best_model(data.get("E24R+R36P", []))

    if wt_bm and wt_bm["plddt"] is not None:
        # interface residues = Vpr residues that make at least one contact
        iface_vpr = sorted({c[0] - 1 for c in wt_contacts}) if wt_contacts else []
        iface_krab = sorted({c[1] - 1 + VPR_LEN for c in wt_contacts}) if wt_contacts else []
        iface_idx  = sorted(set(iface_vpr) | set(iface_krab))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, (label, bm) in zip(axes, [("WT", wt_bm), ("E24R+R36P", dbl_bm)]):
            if bm is None or bm["plddt"] is None:
                ax.set_title(f"{label}\n(no data)")
                continue
            plddt = bm["plddt"]
            x = np.arange(len(plddt))
            vpr_mask  = x < VPR_LEN
            krab_mask = x >= VPR_LEN

            ax.fill_between(x[vpr_mask],  plddt[vpr_mask],  alpha=0.3,
                            color=COLORS[label if label in COLORS else "WT"],
                            label="Vpr")
            ax.fill_between(x[krab_mask], plddt[krab_mask], alpha=0.3,
                            color="#2ca02c", label="ZNF430-KRAB")
            ax.plot(x, plddt, linewidth=0.7,
                    color=COLORS[label if label in COLORS else "WT"])

            # Highlight Vpr interface residues
            for idx in iface_vpr:
                if idx < len(plddt):
                    ax.axvline(idx, color="red", alpha=0.3, linewidth=0.8)

            # Mark mutation sites
            for pos, mname in [(23, "E24R"), (35, "R36P")]:
                ax.axvline(pos, color="black", linestyle="--", linewidth=1.2)
                ax.text(pos + 0.5, 102, mname, fontsize=7, va="top")

            ax.axvline(VPR_LEN - 0.5, color="gray", linewidth=1.5)
            ax.text(VPR_LEN / 2, 5, "Vpr", ha="center", fontsize=9, color="gray")
            ax.text(VPR_LEN + KRAB_LEN / 2, 5, "ZNF430-KRAB",
                    ha="center", fontsize=9, color="gray")
            ax.set_xlim(-1, VPR_LEN + KRAB_LEN)
            ax.set_ylim(0, 110)
            ax.set_xlabel("Residue index")
            ax.set_ylabel("pLDDT")
            ax.set_title(f"{label}  (ipTM = {bm['iptm']:.3f})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.2)

        fig.suptitle("Per-residue pLDDT — WT vs E24R+R36P\n"
                     "Red lines = WT interface residues; dashed = mutation sites",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "interface_plddt.png"), dpi=150)
        plt.close()
        print("Saved: interface_plddt.png")

    # ── 7. ΔipTM summary ──────────────────────────────────────────────────────
    wt_iptm = df.loc["WT", "best_iptm"] if "WT" in df.index else np.nan
    print("\n── ΔipTM relative to WT ──────────────────────────────────────")
    for label in ["E24R", "R36P", "E24R+R36P"]:
        if label in df.index:
            delta = df.loc[label, "best_iptm"] - wt_iptm
            print(f"  {label:12s}: Δ ipTM = {delta:+.3f}  "
                  f"(WT={wt_iptm:.3f}  mut={df.loc[label,'best_iptm']:.3f})")
    print("\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
