#!/usr/bin/env python3
"""
5_analyze.py
============
Full analysis pipeline for HIV-1 Vpr x KZFP ColabFold predictions.

Sections (all run by default; pass --only <name> to run one):
  summary   -- ipTM table for all domains (KRAB / ZNF / KRAB-A / KRAB-B)
  phylogeny -- KRAB-domain NJ tree colored by ipTM (MAFFT + BioPython)
  features  -- Sequence / AA-composition / MSA-positional analysis
  ml        -- ML feature importance (RF + GB, sequence + conservation + structure)
  krab_b    -- KRAB-B conservation: key positions mediating Vpr interaction

Outputs written to: analysis/

Usage:
    conda run -n colabfold python 5_analyze.py
    conda run -n colabfold python 5_analyze.py --only summary
    conda run -n colabfold python 5_analyze.py --only krab_b
"""

import argparse, glob, json, os, re, subprocess, warnings
warnings.filterwarnings("ignore")
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from Bio import AlignIO, Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import chi2_contingency, linregress, mannwhitneyu, spearmanr, fisher_exact

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score

import logomaker

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = "/local/workdir/wz452/script/project/KZFP_Vpr"
PRED_DIR     = os.path.join(BASE_DIR, "predictions")
SEQ_DIR      = os.path.join(BASE_DIR, "sequences")
OUT_DIR      = os.path.join(BASE_DIR, "analysis")
ALIGNED_KRAB = os.path.join(OUT_DIR, "krab_aligned.fasta")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
HIGH_THR = 0.5
LOW_THR  = 0.3
AAS      = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBICITY = dict(
    A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5,  Q=-3.5, E=-3.5, G=-0.4,
    H=-3.2, I=4.5, L=3.8,  K=-3.9, M=1.9,  F=2.8,  P=-1.6, S=-0.8,
    T=-0.7, W=-0.9, Y=-1.3, V=4.2,
)
BLOCK_COLORS = {"seq": "#4878CF", "cons": "#6ACC65", "struct": "#D65F5F"}


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_iptm_scores(domain):
    """Return {gene: max_iptm} for all Vpr_*-{domain} prediction directories."""
    scores = {}
    for d in glob.glob(os.path.join(PRED_DIR, f"Vpr_*-{domain}")):
        gene  = os.path.basename(d).replace("Vpr_", "").replace(f"-{domain}", "")
        iptms = []
        for f in glob.glob(os.path.join(d, "*scores*.json")):
            try:
                data = json.load(open(f))
                if "iptm" in data:
                    iptms.append(data["iptm"])
            except Exception:
                pass
        if iptms:
            scores[gene] = max(iptms)
    return scores


def extract_krab_seq(gene):
    """Return the KRAB portion of sequence from Vpr_{gene}-KRAB.fasta."""
    fasta = os.path.join(SEQ_DIR, f"Vpr_{gene}-KRAB.fasta")
    if not os.path.exists(fasta):
        return None
    with open(fasta) as fh:
        content = fh.read().strip()
    seq_line = "".join(l for l in content.split("\n") if not l.startswith(">"))
    return seq_line.split(":")[1] if ":" in seq_line else seq_line


def safe_id(gene):
    return re.sub(r"[^A-Za-z0-9_]", "_", gene)


def sig_stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def sequence_features(seq):
    seq_clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())
    if len(seq_clean) < 5:
        return None
    pa    = ProteinAnalysis(seq_clean)
    h, sh, co = pa.secondary_structure_fraction()
    feats = {
        "seq_len":        len(seq_clean),
        "pI":             pa.isoelectric_point(),
        "charge_pH7":     pa.charge_at_pH(7.0),
        "hydrophobicity": float(np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq_clean])),
        "aromaticity":    pa.aromaticity(),
        "instability":    pa.instability_index(),
        "helix_frac":     h,
        "sheet_frac":     sh,
        "coil_frac":      co,
    }
    total = len(seq_clean)
    for aa in AAS:
        feats[f"aa_{aa}"] = seq_clean.count(aa) / total
    return feats


def a3m_conservation(a3m_path, vpr_len):
    """Shannon entropy features from the KRAB portion of a paired a3m MSA."""
    seqs = []
    with open(a3m_path) as fh:
        cur = ""
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append(cur)
                cur = ""
            else:
                cur += line
        if cur:
            seqs.append(cur)

    nan_result = dict(cons_mean=np.nan, cons_std=np.nan, cons_min=np.nan,
                      cons_nterm=np.nan, cons_cterm=np.nan, n_homologs=len(seqs))
    if len(seqs) < 2:
        return nan_result

    def remove_insertions(s):
        return "".join(c for c in s if c == "-" or c.isupper())

    clean     = [remove_insertions(s) for s in seqs]
    krab_cols = [s[vpr_len:] for s in clean if len(s) >= vpr_len + 5]
    if not krab_cols:
        return nan_result

    entropies = []
    for pos in range(len(krab_cols[0])):
        col    = [s[pos] for s in krab_cols if pos < len(s)]
        nongap = [c for c in col if c != "-"]
        if len(nongap) < 3:
            continue
        counts = Counter(nongap)
        total  = sum(counts.values())
        probs  = [v / total for v in counts.values()]
        entropies.append(-sum(p * np.log2(p) for p in probs if p > 0))

    if not entropies:
        return nan_result

    e = np.array(entropies)
    n = int(min(20, len(e) // 3))
    return dict(cons_mean=float(e.mean()), cons_std=float(e.std()),
                cons_min=float(e.min()),
                cons_nterm=float(e[:n].mean()) if n else np.nan,
                cons_cterm=float(e[-n:].mean()) if n else np.nan,
                n_homologs=len(seqs))


def structural_features(pred_dir):
    """pLDDT, PAE, and interface-contact features from the rank_001 model."""
    score_files = sorted(glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")))
    pdb_files   = sorted(glob.glob(os.path.join(pred_dir, "*rank_001*.pdb")))
    if not score_files or not pdb_files:
        return None

    data  = json.load(open(score_files[0]))
    plddt = np.array(data["plddt"])
    pae   = np.array(data["pae"])
    ptm   = data.get("ptm", np.nan)

    ca = {"A": [], "B": []}
    with open(pdb_files[0]) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ch = line[21]
                if ch in ca:
                    ca[ch].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not ca["A"] or not ca["B"]:
        return None

    vpr_ca  = np.array(ca["A"])
    krab_ca = np.array(ca["B"])
    vl, kl  = len(vpr_ca), len(krab_ca)

    vpr_plddt  = plddt[:vl]
    krab_plddt = plddt[vl:vl + kl]
    dists      = np.linalg.norm(vpr_ca[:, None] - krab_ca[None], axis=-1)
    vpr_mask   = np.any(dists < 8.0, axis=1)
    krab_mask  = np.any(dists < 8.0, axis=0)
    n_contacts = int((dists < 8.0).sum())

    pae_vk = pae[:vl, vl:vl + kl]
    pae_kv = pae[vl:vl + kl, :vl]
    iface_pae_mean = float((pae_vk.mean() + pae_kv.mean()) / 2)
    iface_pae_min  = float(min(pae_vk.min(), pae_kv.min()))

    n_vpr_iface  = int(vpr_mask.sum())
    n_krab_iface = int(krab_mask.sum())
    if n_vpr_iface > 0 and n_krab_iface > 0:
        focused_pae = float(pae[np.ix_(np.where(vpr_mask)[0],
                                       vl + np.where(krab_mask)[0])].mean())
    else:
        focused_pae = iface_pae_mean

    return dict(
        ptm=float(ptm), max_pae=float(data.get("max_pae", np.nan)),
        vpr_plddt_mean=float(vpr_plddt.mean()),
        krab_plddt_mean=float(krab_plddt.mean()), krab_plddt_std=float(krab_plddt.std()),
        krab_plddt_min=float(krab_plddt.min()),
        iface_pae_mean=iface_pae_mean, iface_pae_min=iface_pae_min,
        focused_pae_mean=focused_pae,
        n_contacts=n_contacts, n_vpr_iface=n_vpr_iface, n_krab_iface=n_krab_iface,
        iface_vpr_plddt=float(vpr_plddt[vpr_mask].mean()) if n_vpr_iface > 0 else np.nan,
        iface_krab_plddt=float(krab_plddt[krab_mask].mean()) if n_krab_iface > 0 else np.nan,
        vpr_len=vl, krab_len=kl,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — ipTM summary table
# ══════════════════════════════════════════════════════════════════════════════

def run_summary():
    print("\n" + "=" * 60)
    print("SECTION 1 — ipTM summary")
    print("=" * 60)

    krab_scores   = load_iptm_scores("KRAB")
    znf_scores    = load_iptm_scores("ZNF")
    krab_a_scores = load_iptm_scores("KRAB_A")
    krab_b_scores = load_iptm_scores("KRAB_B")

    all_genes = sorted(set(krab_scores) | set(znf_scores) |
                       set(krab_a_scores) | set(krab_b_scores))

    rows = []
    for gene in all_genes:
        rows.append(dict(
            gene        = gene,
            krab_iptm   = krab_scores.get(gene),
            krab_a_iptm = krab_a_scores.get(gene) if krab_a_scores else None,
            krab_b_iptm = krab_b_scores.get(gene) if krab_b_scores else None,
            znf_iptm    = znf_scores.get(gene),
        ))
    df = pd.DataFrame(rows).sort_values("krab_iptm", ascending=False, na_position="last")

    # Always save the 2-domain summary
    df[["gene", "krab_iptm", "znf_iptm"]].to_csv(
        os.path.join(OUT_DIR, "iptm_summary.tsv"), sep="\t", index=False)
    print(f"  Saved: iptm_summary.tsv  ({len(df)} genes)")

    # Save 4-domain table when KRAB-A/B data is available
    if krab_a_scores or krab_b_scores:
        df.to_csv(os.path.join(OUT_DIR, "iptm_all_domains.tsv"),
                  sep="\t", index=False, float_format="%.3f")
        print(f"  Saved: iptm_all_domains.tsv  (4 domains)")

    # Bar chart: top 40 KRAB vs ZNF
    top = df.dropna(subset=["krab_iptm"]).head(40)
    x, w = np.arange(len(top)), 0.4
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - w/2, top["krab_iptm"], width=w, color="#e74c3c", label="KRAB domain", alpha=0.85)
    ax.bar(x + w/2, top["znf_iptm"].fillna(0), width=w, color="#3498db", label="ZNF domain", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top["gene"], rotation=45, ha="right", fontsize=7)
    ax.axhline(0.5, color="black", ls="--", lw=0.8, label="ipTM = 0.5")
    ax.set_ylabel("max ipTM with Vpr"); ax.set_ylim(0, 1)
    ax.set_title("Top 40 KZFPs: Vpr interaction confidence by domain")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "iptm_krab_vs_znf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: iptm_krab_vs_znf.png")

    print(f"\n  KRAB:   {(df['krab_iptm'] >= HIGH_THR).sum()} / "
          f"{df['krab_iptm'].notna().sum()} genes with ipTM >= {HIGH_THR}")
    print(f"  ZNF:    {(df['znf_iptm']  >= HIGH_THR).sum()} / "
          f"{df['znf_iptm'].notna().sum()} genes with ipTM >= {HIGH_THR}")
    if krab_a_scores:
        a = pd.Series(krab_a_scores)
        print(f"  KRAB-A: {(a >= HIGH_THR).sum()} / {len(a)} genes with ipTM >= {HIGH_THR}")
    if krab_b_scores:
        b = pd.Series(krab_b_scores)
        print(f"  KRAB-B: {(b >= HIGH_THR).sum()} / {len(b)} genes with ipTM >= {HIGH_THR}")

    print("\n  Top 10 KRAB interactors:")
    for _, row in df.head(10).iterrows():
        parts = [f"KRAB={row.krab_iptm:.3f}"]
        if krab_a_scores and pd.notna(row.get("krab_a_iptm")):
            parts.append(f"A={row.krab_a_iptm:.3f}")
        if krab_b_scores and pd.notna(row.get("krab_b_iptm")):
            parts.append(f"B={row.krab_b_iptm:.3f}")
        if pd.notna(row.znf_iptm):
            parts.append(f"ZNF={row.znf_iptm:.3f}")
        print(f"    {row.gene:<15}  {'  '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — KRAB phylogeny colored by ipTM
# ══════════════════════════════════════════════════════════════════════════════

def run_phylogeny():
    print("\n" + "=" * 60)
    print("SECTION 2 — KRAB phylogeny")
    print("=" * 60)

    iptm_scores = load_iptm_scores("KRAB")
    records, missing = [], []
    for gene, iptm in iptm_scores.items():
        seq = extract_krab_seq(gene)
        if not seq or len(seq) < 10:
            missing.append(gene)
            continue
        records.append(SeqRecord(Seq(seq), id=safe_id(gene), description=""))
    if missing:
        print(f"  Warning: missing sequence for {len(missing)} genes")
    print(f"  {len(records)} sequences ready")

    # MAFFT alignment
    combined = os.path.join(OUT_DIR, "krab_sequences.fasta")
    SeqIO.write(records, combined, "fasta")
    print(f"  Running MAFFT...")
    with open(ALIGNED_KRAB, "w") as fh:
        result = subprocess.run(
            ["mafft", "--auto", "--thread", "8", "--quiet", combined],
            stdout=fh, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("  MAFFT error:", result.stderr.decode()); return

    # NJ tree
    aln         = AlignIO.read(ALIGNED_KRAB, "fasta")
    constructor = DistanceTreeConstructor(DistanceCalculator("identity"), "nj")
    tree        = constructor.build_tree(aln)
    tree.root_at_midpoint()

    id_to_iptm = {safe_id(g): v for g, v in iptm_scores.items()}
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=1)

    n_leaves = len(tree.get_terminals())
    fig, ax  = plt.subplots(figsize=(14, max(12, n_leaves * 0.18)))
    Phylo.draw(tree, axes=ax, do_show=False,
               label_func=lambda c: c.name if c.is_terminal() else "",
               label_colors=lambda name: mcolors.to_hex(cmap(norm(id_to_iptm.get(name, 0)))),
               show_confidence=False)
    ax.set_title("KZFP KRAB domain phylogeny\n"
                 "(color = max ipTM with Vpr; darker = stronger interaction)", fontsize=11)
    ax.set_xlabel("Branch length (substitutions/site)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.01, aspect=20, label="max ipTM with Vpr")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"krab_phylogeny_iptm.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("  Saved: krab_phylogeny_iptm.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Sequence / AA-composition / MSA-positional features
# ══════════════════════════════════════════════════════════════════════════════

def run_features():
    print("\n" + "=" * 60)
    print("SECTION 3 — Sequence feature analysis")
    print("=" * 60)

    iptm_scores = load_iptm_scores("KRAB")
    pairs = [(g, extract_krab_seq(g)) for g in sorted(iptm_scores, key=iptm_scores.get)]
    pairs = [(g, s) for g, s in pairs if s]
    genes, seqs = zip(*pairs)
    iptms = np.array([iptm_scores[g] for g in genes])

    high_idx = np.where(iptms >= HIGH_THR)[0]
    low_idx  = np.where(iptms <= LOW_THR)[0]
    print(f"  {len(genes)} genes  (high>={HIGH_THR}: {len(high_idx)}, low<={LOW_THR}: {len(low_idx)})")

    # Physicochemical correlations
    prop_names = ["length", "pI", "charge_pH7", "hydrophobicity",
                  "aromaticity", "instability", "helix_frac", "sheet_frac", "coil_frac"]
    props_list = []
    for g, seq in zip(genes, seqs):
        seq_clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())
        if len(seq_clean) < 5:
            continue
        pa = ProteinAnalysis(seq_clean)
        h, sh, co = pa.secondary_structure_fraction()
        props_list.append(dict(
            gene=g, iptm=iptm_scores[g], length=len(seq_clean),
            pI=pa.isoelectric_point(), charge_pH7=pa.charge_at_pH(7.0),
            hydrophobicity=float(np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq_clean])),
            aromaticity=pa.aromaticity(), instability=pa.instability_index(),
            helix_frac=h, sheet_frac=sh, coil_frac=co,
        ))
    prop_data = {k: np.array([p[k] for p in props_list]) for k in prop_names}
    iptm_arr  = np.array([p["iptm"] for p in props_list])
    correlations = {k: spearmanr(prop_data[k], iptm_arr) for k in prop_names}

    print("\n  Spearman r with ipTM:")
    for k, (r, pv) in sorted(correlations.items(), key=lambda x: -abs(x[1][0])):
        print(f"    {k:<20} r={r:+.3f}  p={pv:.2e} {sig_stars(pv)}")

    # AA composition (high vs low)
    hi_seqs = [seqs[i] for i in high_idx]
    lo_seqs = [seqs[i] for i in low_idx]

    def aa_freq(seq_list):
        counts = Counter(aa for seq in seq_list for aa in seq.upper() if aa in AAS)
        total  = sum(counts.values())
        return {aa: counts[aa] / total for aa in AAS} if total else {aa: 0 for aa in AAS}

    hi_freq   = aa_freq(hi_seqs)
    lo_freq   = aa_freq(lo_seqs)
    diff_freq = {aa: hi_freq[aa] - lo_freq[aa] for aa in AAS}
    aa_pvals  = {aa: mannwhitneyu(
                    [s.upper().count(aa) / max(len(s), 1) for s in hi_seqs],
                    [s.upper().count(aa) / max(len(s), 1) for s in lo_seqs],
                    alternative="two-sided")[1]
                 for aa in AAS}

    # MSA positional Chi²
    if not os.path.exists(ALIGNED_KRAB):
        print("  krab_aligned.fasta not found -- run section 'phylogeny' first")
        chi2_scores = chi2_pvals = np.array([])
        enriched_aa = []
        aln_seqs_hi = aln_seqs_lo = []
    else:
        aln_dict    = {rec.id: str(rec.seq) for rec in AlignIO.read(ALIGNED_KRAB, "fasta")}
        aln_seqs_hi = [aln_dict[safe_id(g)] for g in [genes[i] for i in high_idx] if safe_id(g) in aln_dict]
        aln_seqs_lo = [aln_dict[safe_id(g)] for g in [genes[i] for i in low_idx]  if safe_id(g) in aln_dict]
        aln_len     = len(next(iter(aln_dict.values())))
        chi2_scores = np.zeros(aln_len)
        chi2_pvals  = np.ones(aln_len)
        enriched_aa = [""] * aln_len
        for pos in range(aln_len):
            hi_col = [s[pos] for s in aln_seqs_hi]
            lo_col = [s[pos] for s in aln_seqs_lo]
            if hi_col.count("-") / max(len(hi_col), 1) > 0.7:
                continue
            table = [[hi_col.count(aa), lo_col.count(aa)]
                     for aa in AAS if hi_col.count(aa) + lo_col.count(aa) > 0]
            if len(table) < 2:
                continue
            try:
                chi2, pv, *_ = chi2_contingency(table)
                chi2_scores[pos], chi2_pvals[pos] = chi2, pv
                best_aa = max(AAS, key=lambda aa: hi_col.count(aa)/max(len(hi_col),1)
                                                 - lo_col.count(aa)/max(len(lo_col),1))
                enriched_aa[pos] = best_aa
            except Exception:
                pass
        sig_pos = np.where(chi2_pvals < 0.01)[0]
        print(f"\n  Significant MSA positions (p<0.01): {len(sig_pos)}")

    # Plot
    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(iptms, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(HIGH_THR, color="red",    ls="--", lw=1.5, label=f"high >={HIGH_THR}")
    ax.axvline(LOW_THR,  color="orange", ls="--", lw=1.5, label=f"low <={LOW_THR}")
    ax.set_xlabel("max ipTM"); ax.set_ylabel("Count")
    ax.set_title("A  ipTM distribution", fontweight="bold"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1:])
    keys  = list(correlations)
    rvals = [correlations[k][0] for k in keys]
    pvals = [correlations[k][1] for k in keys]
    ax.barh(keys, rvals, color=["#d62728" if r > 0 else "#1f77b4" for r in rvals], alpha=0.8)
    for i, (r, pv) in enumerate(zip(rvals, pvals)):
        ax.text(r + (0.01 if r >= 0 else -0.01), i, sig_stars(pv),
                va="center", ha="left" if r >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman r with ipTM"); ax.set_xlim(-0.8, 0.8)
    ax.set_title("B  Physicochemical correlates", fontweight="bold")

    top2 = sorted(correlations.items(), key=lambda x: -abs(x[1][0]))[:2]
    for ci, (prop, (r, pv)) in enumerate(top2):
        ax = fig.add_subplot(gs[1, ci])
        ax.scatter(prop_data[prop], iptm_arr, c=iptm_arr, cmap="YlOrRd", s=15, alpha=0.7, vmin=0, vmax=1)
        sl, ic, *_ = linregress(prop_data[prop], iptm_arr)
        xr = np.linspace(prop_data[prop].min(), prop_data[prop].max(), 100)
        ax.plot(xr, sl * xr + ic, "k--", lw=1)
        ax.set_xlabel(prop); ax.set_ylabel("ipTM")
        ax.set_title(f"{'CD'[ci]}  {prop}  r={r:+.3f}", fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    sorted_aas = sorted(AAS, key=lambda aa: -diff_freq[aa])
    diff_vals  = [diff_freq[aa] for aa in sorted_aas]
    ax.bar(sorted_aas, diff_vals,
           color=["#d62728" if d > 0 else "#1f77b4" for d in diff_vals], alpha=0.8)
    for i, aa in enumerate(sorted_aas):
        if aa_pvals[aa] < 0.05:
            ax.text(i, diff_vals[i] + (0.001 if diff_vals[i] >= 0 else -0.003),
                    "*", ha="center", fontsize=11)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Amino acid"); ax.set_ylabel("Freq(high) - Freq(low)")
    ax.set_title("E  AA enrichment", fontweight="bold"); ax.tick_params(axis="x", labelsize=8)

    if len(chi2_scores) > 0:
        sig_pos = np.where(chi2_pvals < 0.01)[0]
        ax = fig.add_subplot(gs[2, :])
        ax.plot(chi2_scores, color="gray", lw=0.6, alpha=0.7)
        ax.scatter(sig_pos, chi2_scores[sig_pos], color="red", s=20, zorder=3,
                   label=f"p<0.01 ({len(sig_pos)} pos)")
        for p in sorted(sig_pos, key=lambda x: chi2_scores[x], reverse=True)[:15]:
            ax.annotate(enriched_aa[p], (p, chi2_scores[p]),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7, color="darkred")
        ax.set_xlabel("Alignment position"); ax.set_ylabel("Chi2")
        ax.set_title("F  Position-specific differences (high vs low ipTM)", fontweight="bold")
        ax.legend(fontsize=9)

        top_pos = sorted(sorted(sig_pos, key=lambda p: chi2_scores[p], reverse=True)[:25])
        if top_pos:
            ax = fig.add_subplot(gs[3, :])
            freq_mat = np.zeros((len(AAS), len(top_pos)))
            for j, pos in enumerate(top_pos):
                hi_col = [aln_seqs_hi[k][pos] for k in range(len(aln_seqs_hi))]
                lo_col = [aln_seqs_lo[k][pos] for k in range(len(aln_seqs_lo))]
                for i, aa in enumerate(AAS):
                    freq_mat[i, j] = (hi_col.count(aa) / max(len(hi_col), 1)
                                     - lo_col.count(aa) / max(len(lo_col), 1))
            vmax = np.abs(freq_mat).max()
            im = ax.imshow(freq_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(top_pos)))
            ax.set_xticklabels([f"{p}\n({enriched_aa[p]})" for p in top_pos], fontsize=7)
            ax.set_yticks(range(len(AAS))); ax.set_yticklabels(AAS, fontsize=8)
            ax.set_title("G  AA frequency difference at discriminating positions", fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.6, label="Freq(high) - Freq(low)")

    fig.suptitle(f"KRAB domain features — Vpr interaction\n"
                 f"(n={len(genes)}, high>={HIGH_THR}: n={len(high_idx)}, low<={LOW_THR}: n={len(low_idx)})",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"krab_features.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("  Saved: krab_features.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Machine learning
# ══════════════════════════════════════════════════════════════════════════════

def run_ml():
    print("\n" + "=" * 60)
    print("SECTION 4 — Machine learning")
    print("=" * 60)

    rows, skipped = [], 0
    for pred_dir in sorted(glob.glob(os.path.join(PRED_DIR, "Vpr_*-KRAB"))):
        gene = os.path.basename(pred_dir).replace("Vpr_", "").replace("-KRAB", "")
        iptms = []
        for f in glob.glob(os.path.join(pred_dir, "*scores*.json")):
            try:
                data = json.load(open(f))
                if "iptm" in data:
                    iptms.append(data["iptm"])
            except Exception:
                pass
        if not iptms:
            skipped += 1; continue
        iptm = max(iptms)

        seq = extract_krab_seq(gene)
        if not seq:
            skipped += 1; continue
        sf = sequence_features(seq)
        if sf is None:
            skipped += 1; continue

        a3m_files = glob.glob(os.path.join(pred_dir, "*.a3m"))
        if a3m_files:
            try:
                with open(a3m_files[0]) as fh:
                    hdr = fh.readline().strip()
                vpr_len = int(hdr[1:].split("\t")[0].split(",")[0]) if hdr.startswith("#") else 96
            except Exception:
                vpr_len = 96
            cf = a3m_conservation(a3m_files[0], vpr_len)
        else:
            cf = dict(cons_mean=np.nan, cons_std=np.nan, cons_min=np.nan,
                      cons_nterm=np.nan, cons_cterm=np.nan, n_homologs=0)

        stf = structural_features(pred_dir)
        if stf is None:
            skipped += 1; continue

        row = {"gene": gene, "iptm": iptm, "label": int(iptm >= HIGH_THR)}
        row.update({f"seq_{k}": v for k, v in sf.items()})
        row.update({f"cons_{k}": v for k, v in cf.items()})
        row.update({f"struct_{k}": v for k, v in stf.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} samples ({skipped} skipped)")
    df.to_csv(os.path.join(OUT_DIR, "krab_feature_matrix.csv"), index=False)

    feature_cols = [c for c in df.columns if c not in ("gene", "iptm", "label")]
    X_df = df[feature_cols].copy()
    drop = X_df.isna().mean()[lambda s: s > 0.2].index.tolist()
    if drop:
        print(f"  Dropping {len(drop)} high-NaN features")
    X_df = X_df.drop(columns=drop).fillna(X_df.median())
    feature_names = X_df.columns.tolist()
    X, y_reg, y_cls = X_df.values, df["iptm"].values, df["label"].values
    print(f"  Feature matrix: {X.shape[0]} x {X.shape[1]}")

    kf  = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=3, random_state=42, n_jobs=8)
    gb_reg = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                       learning_rate=0.05, random_state=42)
    rf_cls = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=3, random_state=42, n_jobs=8,
                                    class_weight="balanced")

    r2_rf  = cross_val_score(rf_reg, X, y_reg, cv=kf,  scoring="r2")
    r2_gb  = cross_val_score(gb_reg, X, y_reg, cv=kf,  scoring="r2")
    auc_rf = cross_val_score(rf_cls, X, y_cls, cv=skf, scoring="roc_auc")
    print(f"\n  RF regression   R2  : {r2_rf.mean():.3f} +/- {r2_rf.std():.3f}")
    print(f"  GB regression   R2  : {r2_gb.mean():.3f} +/- {r2_gb.std():.3f}")
    print(f"  RF classifier   AUC : {auc_rf.mean():.3f} +/- {auc_rf.std():.3f}")

    rf_reg.fit(X, y_reg)
    rf_cls.fit(X, y_cls)
    perm = permutation_importance(rf_reg, X, y_reg, n_repeats=20, random_state=42, n_jobs=8)
    sp_r = np.array([spearmanr(X[:, i], y_reg)[0] for i in range(X.shape[1])])
    sp_p = np.array([spearmanr(X[:, i], y_reg)[1] for i in range(X.shape[1])])

    feat_df = pd.DataFrame({
        "feature":      feature_names,
        "block":        [f.split("_")[0] for f in feature_names],
        "perm_imp":     perm.importances_mean,
        "perm_std":     perm.importances_std,
        "impurity_imp": rf_reg.feature_importances_,
        "spearman_r":   sp_r,
        "spearman_p":   sp_p,
    }).sort_values("perm_imp", ascending=False)
    feat_df.to_csv(os.path.join(OUT_DIR, "krab_feature_importance.csv"), index=False)

    block_imp = feat_df.groupby("block")["perm_imp"].sum()
    top30 = feat_df.head(30)

    print("\n  Block importance:")
    for b, v in block_imp.sort_values(ascending=False).items():
        print(f"    {b:8s}: {v:.4f} ({100*v/block_imp.sum():.1f}%)")
    print("\n  Top 10 features:")
    for _, row in feat_df.head(10).iterrows():
        print(f"    [{row.block:6s}] {row.feature:<35} "
              f"perm={row.perm_imp:.4f}  r={row.spearman_r:+.3f} {sig_stars(row.spearman_p)}")

    # Plot
    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    legend_elements = [Patch(facecolor=c, label=b) for b, c in BLOCK_COLORS.items()]

    ax = fig.add_subplot(gs[0, 0])
    means = [r2_rf.mean(), r2_gb.mean(), auc_rf.mean()]
    stds  = [r2_rf.std(),  r2_gb.std(),  auc_rf.std()]
    bars  = ax.bar(["RF\nRegression", "GB\nRegression", "RF\nClassifier\n(AUC)"],
                   means, yerr=stds, capsize=6,
                   color=["#4878CF", "#6ACC65", "#D65F5F"], alpha=0.85, width=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.02, f"{m:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score (R2 or AUC)")
    ax.set_title("A  5-fold CV performance", fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)

    ax = fig.add_subplot(gs[0, 1:])
    top_n  = min(30, len(top30))
    y_pos  = np.arange(top_n)
    colors = [BLOCK_COLORS.get(r["block"], "gray") for _, r in top30.head(top_n).iterrows()]
    ax.barh(y_pos, top30.head(top_n)["perm_imp"].values,
            xerr=top30.head(top_n)["perm_std"].values, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(y_pos); ax.set_yticklabels(top30.head(top_n)["feature"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (mean delta R2)")
    ax.set_title("B  Top feature importances", fontweight="bold")
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(block_imp.index, block_imp.values,
           color=[BLOCK_COLORS.get(b, "gray") for b in block_imp.index], alpha=0.85)
    for i, (b, v) in enumerate(block_imp.items()):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Sum permutation importance")
    ax.set_title("C  Block importance", fontweight="bold")

    for fi, feat in enumerate(top30["feature"].head(2)):
        ax   = fig.add_subplot(gs[1, fi + 1])
        fidx = feature_names.index(feat)
        xv   = X[:, fidx]
        sc   = ax.scatter(xv, y_reg, c=y_reg, cmap="YlOrRd", s=12, alpha=0.6, vmin=0, vmax=1)
        sl, ic, *_ = linregress(xv, y_reg)
        xr = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xr, sl * xr + ic, "k--", lw=1)
        r, p  = spearmanr(xv, y_reg)
        block = feat.split("_")[0]
        ax.set_title(f"{'DE'[fi]}  [{block}] {feat.replace(block+'_','')}\nr={r:+.3f}{sig_stars(p)}",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel(feat.replace(block + "_", ""), fontsize=8); ax.set_ylabel("ipTM")
        fig.colorbar(sc, ax=ax, shrink=0.7)

    ax = fig.add_subplot(gs[2, :2])
    top_sp = pd.concat([feat_df.sort_values("spearman_r", ascending=False).head(15),
                        feat_df.sort_values("spearman_r").head(10)])
    yp = np.arange(len(top_sp))
    ax.barh(yp, top_sp["spearman_r"].values,
            color=[BLOCK_COLORS.get(b, "gray") for b in top_sp["block"]], alpha=0.85)
    ax.set_yticks(yp); ax.set_yticklabels(top_sp["feature"].values, fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman r with ipTM")
    ax.set_title("F  Top positive and negative correlates", fontweight="bold")
    ax.legend(handles=legend_elements, fontsize=8)

    ax     = fig.add_subplot(gs[2, 2])
    y_pred = cross_val_predict(rf_reg, X, y_reg, cv=kf)
    r2_cv  = r2_score(y_reg, y_pred)
    ax.scatter(y_reg, y_pred, c=y_reg, cmap="YlOrRd", s=12, alpha=0.6, vmin=0, vmax=1)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Actual ipTM"); ax.set_ylabel("Predicted ipTM (CV)")
    ax.set_title(f"G  Predicted vs actual\nCV R2={r2_cv:.3f}", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle(f"ML: KRAB features determining Vpr interaction\n"
                 f"(n={len(df)}, {X.shape[1]} features: seq + conservation + structure)",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"krab_ml_results.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("  Saved: krab_ml_results.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — KRAB-B conservation analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_krab_b():
    print("\n" + "=" * 60)
    print("SECTION 5 — KRAB-B conservation analysis")
    print("=" * 60)

    aln_b_fa  = os.path.join(OUT_DIR, "krab_b_aligned.fasta")
    box_iptm  = os.path.join(OUT_DIR, "krab_box_iptm.tsv")

    if not os.path.exists(aln_b_fa):
        print("  krab_b_aligned.fasta not found.")
        print("  Run:  python 3_batch_submit.py --domain krab_boxes --no-submit")
        return
    if not os.path.exists(box_iptm):
        print("  krab_box_iptm.tsv not found -- run section 'summary' first,")
        print("  then generate it with:  python 5_analyze.py --only summary")
        return

    # Load alignment
    seqs, name = {}, None
    with open(aln_b_fa) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif name:
                seqs[name] = seqs.get(name, "") + line
    aln_len = len(next(iter(seqs.values())))
    print(f"  Alignment: {len(seqs)} sequences x {aln_len} positions")

    # Load ipTM scores
    iptm_df = pd.read_csv(box_iptm, sep="\t")
    iptm_b  = iptm_df[iptm_df["box"] == "KRAB_B"].set_index("gene")["iptm"]
    safe_to_iptm = {safe_id(g): v for g, v in iptm_b.items()}

    valid  = [(n, seqs[n], safe_to_iptm[n]) for n in seqs if n in safe_to_iptm]
    seqs_v = [x[1] for x in valid]
    iptm_v = np.array([x[2] for x in valid])
    print(f"  Sequences with ipTM: {len(valid)}")

    high_idx  = np.where(iptm_v >= HIGH_THR)[0]
    low_idx   = np.where(iptm_v <  HIGH_THR)[0]
    high_seqs = [seqs_v[i] for i in high_idx]
    low_seqs  = [seqs_v[i] for i in low_idx]
    n_h, n_l  = len(high_seqs), len(low_seqs)
    print(f"  High (>={HIGH_THR}): {n_h}   Low (<{HIGH_THR}): {n_l}")

    # Frequency and count matrices
    def freq_mat(seq_list):
        mat = np.zeros((aln_len, len(AAS)))
        for seq in seq_list:
            for i, aa in enumerate(seq):
                if aa in AAS:
                    mat[i, AAS.index(aa)] += 1
        col = mat.sum(axis=1, keepdims=True); col[col == 0] = 1
        return mat / col

    def cnt_mat(seq_list):
        mat = np.zeros((aln_len, len(AAS)), dtype=int)
        for seq in seq_list:
            for i, aa in enumerate(seq):
                if aa in AAS:
                    mat[i, AAS.index(aa)] += 1
        return mat

    def gap_frac(seq_list):
        return np.array([sum(1 for s in seq_list if s[i] == "-") / len(seq_list)
                         for i in range(aln_len)])

    freq_h = freq_mat(high_seqs);  freq_l = freq_mat(low_seqs)
    cnt_h  = cnt_mat(high_seqs);   cnt_l  = cnt_mat(low_seqs)
    gap_h  = gap_frac(high_seqs);  gap_all = gap_frac(seqs_v)

    # Fisher's exact test, Bonferroni corrected
    pvals = np.ones((aln_len, len(AAS)));  odds = np.ones((aln_len, len(AAS)))
    for i in range(aln_len):
        for j in range(len(AAS)):
            a, c = cnt_h[i, j], cnt_l[i, j]
            if a + c == 0:
                continue
            try:
                or_, p = fisher_exact([[a, n_h - a], [c, n_l - c]], alternative="greater")
                pvals[i, j] = p
                odds[i, j]  = or_ if np.isfinite(or_) else 50.0
            except Exception:
                pass
    pvals_adj = np.minimum(pvals * aln_len * len(AAS), 1.0)

    sig_rows = [
        dict(aln_pos=i+1, aa=AAS[j],
             freq_high=round(freq_h[i,j], 3), freq_low=round(freq_l[i,j], 3),
             odds_ratio=round(odds[i,j], 2),  p_adj=round(pvals_adj[i,j], 5))
        for i in range(aln_len) for j in range(len(AAS))
        if pvals_adj[i,j] < 0.05 and freq_h[i,j] >= 0.2
    ]
    sig_df = pd.DataFrame(sig_rows).sort_values("p_adj") if sig_rows else pd.DataFrame()
    sig_df.to_csv(os.path.join(OUT_DIR, "krab_b_key_positions.tsv"), sep="\t", index=False)
    print(f"\n  Significant positions (Bonferroni p<0.05): {len(sig_df)}")
    if len(sig_df):
        print(sig_df.head(15).to_string(index=False))

    # Non-gappy columns for logos / heatmaps
    keep = np.where(gap_h < 0.5)[0]
    print(f"\n  Non-gappy columns: {len(keep)}")

    def make_logo_df(freq_mat_full, cols):
        sub = freq_mat_full[cols, :]
        ic  = np.clip(np.sum(sub * np.log2(sub + 1e-9), axis=1) + np.log2(20), 0, None)
        return pd.DataFrame(sub * ic[:, None], columns=AAS)

    for df_logo, title, fname in [
        (make_logo_df(freq_h, keep),
         f"KRAB-B logo -- High ipTM (>={HIGH_THR}, n={n_h})", "krab_b_logo_high.png"),
        (make_logo_df(freq_l, keep),
         f"KRAB-B logo -- Low ipTM (<{HIGH_THR}, n={n_l})",   "krab_b_logo_low.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, len(df_logo) * 0.25), 3))
        try:
            logomaker.Logo(df_logo, ax=ax, color_scheme="chemistry")
        except Exception:
            logomaker.Logo(df_logo, ax=ax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Alignment position (non-gappy columns)")
        ax.set_ylabel("Information content (bits)")
        step = max(1, len(df_logo) // 20)
        ax.set_xticks(range(0, len(df_logo), step))
        ax.set_xticklabels(keep[range(0, len(df_logo), step)] + 1, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150); plt.close()
        print(f"  Saved: {fname}")

    # Enrichment heatmap (freq_high - freq_low)
    diff = freq_h[keep, :] - freq_l[keep, :]
    fig, ax = plt.subplots(figsize=(max(12, len(keep) * 0.22), 6))
    im = ax.imshow(diff.T, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_yticks(range(len(AAS))); ax.set_yticklabels(AAS, fontsize=8)
    step = max(1, len(keep) // 20)
    ax.set_xticks(range(0, len(keep), step))
    ax.set_xticklabels(keep[range(0, len(keep), step)] + 1, fontsize=7)
    ax.set_xlabel("Alignment position"); ax.set_ylabel("Amino acid")
    ax.set_title("KRAB-B: AA frequency difference (High - Low ipTM)\n"
                 "Red = enriched in high-ipTM; Blue = depleted", fontsize=10)
    plt.colorbar(im, ax=ax, label="Delta frequency")
    for row in (sig_df.itertuples() if len(sig_df) else []):
        col_in = np.where(keep == row.aln_pos - 1)[0]
        if len(col_in):
            ax.add_patch(plt.Rectangle((col_in[0] - 0.5, AAS.index(row.aa) - 0.5),
                                       1, 1, fill=False, edgecolor="black", lw=1.2))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "krab_b_enrichment.png"), dpi=150); plt.close()
    print("  Saved: krab_b_enrichment.png")

    # Spearman correlation heatmap
    corr_mat = np.zeros((aln_len, len(AAS)))
    for i in range(aln_len):
        for j, aa in enumerate(AAS):
            presence = np.array([1 if s[i] == aa else 0 for s in seqs_v])
            if presence.sum() >= 5:
                r, _ = spearmanr(presence, iptm_v)
                corr_mat[i, j] = r if np.isfinite(r) else 0

    fig, ax = plt.subplots(figsize=(max(12, len(keep) * 0.22), 6))
    im = ax.imshow(corr_mat[keep, :].T, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_yticks(range(len(AAS))); ax.set_yticklabels(AAS, fontsize=8)
    ax.set_xticks(range(0, len(keep), step)); ax.set_xticklabels(keep[range(0, len(keep), step)] + 1, fontsize=7)
    ax.set_xlabel("Alignment position"); ax.set_ylabel("Amino acid")
    ax.set_title("KRAB-B: Spearman r of AA presence with Vpr ipTM\n"
                 "Red = positive correlation", fontsize=10)
    plt.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "krab_b_correlation.png"), dpi=150); plt.close()
    print("  Saved: krab_b_correlation.png")

    # Summary table of top correlated positions
    top_corr = []
    for i in range(aln_len):
        if gap_all[i] > 0.5:
            continue
        j = int(np.argmax(np.abs(corr_mat[i, :])))
        r = corr_mat[i, j]
        if abs(r) >= 0.2:
            top_corr.append(dict(aln_pos=i+1, aa=AAS[j], spearman_r=round(r, 3),
                                 freq_high=round(freq_h[i, j], 3),
                                 freq_low=round(freq_l[i, j], 3)))
    top_df = pd.DataFrame(top_corr).sort_values("spearman_r", ascending=False) if top_corr else pd.DataFrame()
    top_df.to_csv(os.path.join(OUT_DIR, "krab_b_top_correlated.tsv"), sep="\t", index=False)
    print(f"\n  Top correlated positions (|r|>=0.2): {len(top_df)}")
    if len(top_df):
        print(top_df.head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

SECTIONS = {
    "summary":   run_summary,
    "phylogeny": run_phylogeny,
    "features":  run_features,
    "ml":        run_ml,
    "krab_b":    run_krab_b,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(SECTIONS), default=None,
                        help="Run only this section (default: run all)")
    args = parser.parse_args()

    to_run = [args.only] if args.only else list(SECTIONS)
    for name in to_run:
        SECTIONS[name]()

    print("\nDone. Outputs in:", OUT_DIR)
