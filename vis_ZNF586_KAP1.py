#!/usr/bin/env python3
"""
Visualize Vpr–ZNF586 vs KAP1–ZNF586 predicted complexes to illustrate
Vpr competition with KAP1 for the ZNF586 KRAB domain.

Three complexes are compared:
  (A) Vpr + ZNF586-KRAB              (existing prediction)
  (B) KAP1-RBCC monomer + ZNF586-KRAB (new prediction)
  (C) KAP1-RBCC dimer  + ZNF586-KRAB  (new prediction, 3-chain)

Outputs → analysis/vis_ZNF586-KAP1/
  PAE_*.png          — PAE heatmaps
  pLDDT_*.png        — per-residue confidence
  view_*.pml         — PyMOL scripts
  viewer_*.html      — interactive 3Dmol viewers
  comparison_figure.png   — combined panel (runs if all 3 dirs exist)
  znf586_contact_overlap.png — ZNF586 residue contact comparison
"""

import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
PRED_DIR = os.path.join(BASE_DIR, "predictions")
OUT_DIR  = os.path.join(BASE_DIR, "analysis", "vis_ZNF586-KAP1")
os.makedirs(OUT_DIR, exist_ok=True)

PAE_CONTACT_THR = 8.0   # Å threshold for interface Cα contacts

AA1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prediction(pred_dir):
    """Load rank_001 score JSON and PDB; fall back to best-ipTM model."""
    score_files = glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json"))
    if not score_files:
        score_files = glob.glob(os.path.join(pred_dir, "*scores*.json"))
    if not score_files:
        raise FileNotFoundError(f"No score files in {pred_dir}")
    best_sf  = max(score_files, key=lambda f: json.load(open(f)).get("iptm", 0))
    best_pdb = best_sf.replace("_scores_", "_unrelaxed_").replace(".json", ".pdb")
    data     = json.load(open(best_sf))
    return data, best_pdb, os.path.basename(best_sf)


def read_pdb_ca(pdb_path):
    """Return {chain_id: [(resid, resname, [x,y,z])]} from Cα atoms."""
    chains = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            ch  = line[21]
            rid = int(line[22:26].strip())
            rn  = line[17:20].strip()
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            chains.setdefault(ch, []).append((rid, rn, xyz))
    return chains


def interface_residues_2chain(chain_ca, ch_a, ch_b, thresh=PAE_CONTACT_THR):
    """Interface residues between two chains. Returns (list_a, list_b, dist_matrix)."""
    ca_a  = np.array([r[2] for r in chain_ca[ch_a]])
    ca_b  = np.array([r[2] for r in chain_ca[ch_b]])
    rid_a = [r[0] for r in chain_ca[ch_a]]
    rid_b = [r[0] for r in chain_ca[ch_b]]
    rn_a  = [r[1] for r in chain_ca[ch_a]]
    rn_b  = [r[1] for r in chain_ca[ch_b]]
    dists = np.linalg.norm(ca_a[:, None] - ca_b[None], axis=-1)
    mask_a = np.any(dists < thresh, axis=1)
    mask_b = np.any(dists < thresh, axis=0)
    fmt = lambda ids, names, mask: [
        f"{AA1.get(names[i],'?')}{ids[i]}" for i in range(len(ids)) if mask[i]]
    return fmt(rid_a, rn_a, mask_a), fmt(rid_b, rn_b, mask_b), dists


def interface_residues_dimer(chain_ca, kap1_chains, znf_chain, thresh=PAE_CONTACT_THR):
    """
    For KAP1-dimer + ZNF586 (3-chain complex).
    Returns (kap1_iface_residues, znf_iface_residues, kap1_kap1_residues).
    kap1_iface includes residues from both KAP1 chains contacting ZNF586.
    """
    ca_znf  = np.array([r[2] for r in chain_ca[znf_chain]])
    rid_znf = [r[0] for r in chain_ca[znf_chain]]
    rn_znf  = [r[1] for r in chain_ca[znf_chain]]
    mask_znf = np.zeros(len(ca_znf), dtype=bool)
    kap1_iface = []

    for kch in kap1_chains:
        ca_k  = np.array([r[2] for r in chain_ca[kch]])
        rid_k = [r[0] for r in chain_ca[kch]]
        rn_k  = [r[1] for r in chain_ca[kch]]
        dists = np.linalg.norm(ca_k[:, None] - ca_znf[None], axis=-1)
        mask_k   = np.any(dists < thresh, axis=1)
        mask_znf |= np.any(dists < thresh, axis=0)
        kap1_iface += [
            (kch, f"{AA1.get(rn_k[i],'?')}{rid_k[i]}")
            for i in range(len(rid_k)) if mask_k[i]
        ]

    znf_iface = [
        f"{AA1.get(rn_znf[i],'?')}{rid_znf[i]}"
        for i in range(len(rid_znf)) if mask_znf[i]
    ]

    # KAP1–KAP1 dimer interface
    kap1_kap1 = []
    if len(kap1_chains) == 2:
        ca_k1  = np.array([r[2] for r in chain_ca[kap1_chains[0]]])
        ca_k2  = np.array([r[2] for r in chain_ca[kap1_chains[1]]])
        rid_k1 = [r[0] for r in chain_ca[kap1_chains[0]]]
        rn_k1  = [r[1] for r in chain_ca[kap1_chains[0]]]
        dimer_d = np.linalg.norm(ca_k1[:, None] - ca_k2[None], axis=-1)
        mask_d  = np.any(dimer_d < thresh, axis=1)
        kap1_kap1 = [f"{AA1.get(rn_k1[i],'?')}{rid_k1[i]}"
                     for i in range(len(rid_k1)) if mask_d[i]]

    return kap1_iface, znf_iface, kap1_kap1


# ── PAE heatmap ───────────────────────────────────────────────────────────────

def plot_pae(pae, chain_lens, chain_labels, title, out_path):
    """PAE heatmap supporting arbitrary number of chains."""
    L = sum(chain_lens)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(pae[:L, :L], cmap="bwr_r", vmin=0, vmax=30, aspect="auto")
    plt.colorbar(im, ax=ax, label="PAE (Å)", fraction=0.04)

    boundaries = np.cumsum([0] + chain_lens)
    chain_colors = ["#FF7D45", "#4575b4", "#2ca02c"]

    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axhline(e - 0.5, color="white", lw=1.5)
        ax.axvline(e - 0.5, color="white", lw=1.5)
        ax.text((s + e) / 2, -4, chain_labels[i],
                ha="center", fontsize=8, fontweight="bold",
                color=chain_colors[i % len(chain_colors)])

    # Highlight inter-chain PAE blocks
    for i, (si, ei) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        for j, (sj, ej) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if i != j:
                ax.add_patch(Rectangle(
                    (sj - 0.5, si - 0.5), ej - sj, ei - si,
                    lw=1.2, edgecolor="gold", facecolor="none"))

    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    ax.set_title(title, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ── pLDDT bar plot ────────────────────────────────────────────────────────────

def plot_plddt(plddt, chain_lens, chain_labels, title, out_path):
    L = sum(chain_lens)
    boundaries = np.cumsum([0] + chain_lens)
    chain_colors = ["#FF7D45", "#4575b4", "#2ca02c"]

    colors = []
    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        colors += [chain_colors[i % len(chain_colors)]] * (e - s)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(L), plddt[:L], color=colors, width=1.0, linewidth=0)
    for e in boundaries[1:-1]:
        ax.axvline(e - 0.5, color="black", lw=1.2, ls="--")
    ax.axhline(70, color="gray", lw=0.8, ls=":", label="pLDDT=70")
    ax.axhline(90, color="gray", lw=0.8, ls="-.", label="pLDDT=90")
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("pLDDT")
    ax.set_title(title, fontsize=10)
    handles = [Patch(color=chain_colors[i % len(chain_colors)], label=chain_labels[i])
               for i in range(len(chain_labels))]
    handles += [plt.Line2D([0],[0], color="gray", ls=":", label="pLDDT=70"),
                plt.Line2D([0],[0], color="gray", ls="-.", label="pLDDT=90")]
    ax.legend(handles=handles, fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ── PyMOL script ──────────────────────────────────────────────────────────────

def write_pymol(pdb_path, chain_iface_dict, chain_labels, out_path):
    """
    chain_iface_dict: {chain_letter: [residue_strings like 'R62']}
    chain_labels: {chain_letter: label}
    """
    chain_colors_pml = {"A": "tv_orange", "B": "tv_blue", "C": "tv_green"}
    lines = [f"# PyMOL script — {', '.join(chain_labels.values())}",
             f"# Run: pymol {out_path}", "",
             f"load {os.path.abspath(pdb_path)}, complex", "",
             "hide everything", "show cartoon, complex"]
    for ch, lbl in chain_labels.items():
        col = chain_colors_pml.get(ch, "gray")
        lines.append(f"color {col}, chain {ch}   # {lbl}")
    lines.append("")
    for ch, residues in chain_iface_dict.items():
        if residues:
            resi_sel = "+".join(r[1:] for r in residues)
            lines += [
                f"select iface_{ch}, chain {ch} and resi {resi_sel}",
                f"show sticks, iface_{ch}",
                f"color yellow, iface_{ch}",
            ]
    lines += ["", "distance hbonds, iface_A, iface_B, 3.5, mode=2",
              "", "orient complex", "zoom complex, 5",
              "set ray_shadows, 0", "set depth_cue, 0",
              "", f"ray 1200, 900",
              f"png {os.path.splitext(out_path)[0]}_pymol.png, dpi=150",
              f"save {os.path.splitext(out_path)[0]}.pse"]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ── HTML viewer ───────────────────────────────────────────────────────────────

def write_html(pdb_path, iptm, ptm, chain_iface_dict, chain_labels, out_path):
    pdb_str = open(pdb_path).read()
    chain_colors_hex = {"A": "#FF7D45", "B": "#4fc3f7", "C": "#69d84f"}
    iface_colors_hex = {"A": "#ffcc00", "B": "#00ffcc", "C": "#ff69b4"}

    legend_html = " &nbsp;|&nbsp; ".join(
        f'<span style="color:{chain_colors_hex.get(ch,"#fff")}"><b>'
        f'{lbl}</b></span>' for ch, lbl in chain_labels.items())
    iface_html  = "<br>".join(
        f'<span style="color:{iface_colors_hex.get(ch,"#fff")}">■ {chain_labels[ch]}:</span> '
        + ", ".join(r for r in res[:20])
        + ("..." if len(res) > 20 else "")
        for ch, res in chain_iface_dict.items() if res)

    js_styles = "\n".join(
        f'viewer.setStyle({{chain:"{ch}"}}, {{cartoon:{{color:"{chain_colors_hex.get(ch,"gray")}", thickness:0.4}}}});'
        for ch in chain_labels)
    js_iface = "\n".join(
        f'viewer.addStyle({{chain:"{ch}", resi:[{", ".join(r[1:] for r in res)}]}}, '
        f'{{stick:{{color:"{iface_colors_hex.get(ch,"#fff")}", radius:0.2}}}});'
        for ch, res in chain_iface_dict.items() if res)
    js_surfaces = "\n".join(
        f'viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.12, color:"{chain_colors_hex.get(ch,"gray")}"}}, {{chain:"{ch}"}});'
        for ch in chain_labels)

    title = " – ".join(chain_labels.values())
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ font-family: Arial; margin:20px; background:#1a1a2e; color:#eee; }}
  h2 {{ color:#00d4ff; }}
  .info {{ background:#162447; padding:10px; border-radius:8px; margin-bottom:10px; }}
  #viewport {{ width:800px; height:600px; position:relative;
               border:2px solid #00d4ff; border-radius:8px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div class="info">
  ipTM = <b>{iptm:.3f}</b> &nbsp;|&nbsp; pTM = <b>{ptm:.3f}</b><br>
  {legend_html}
</div>
<div id="viewport"></div>
<script>
let viewer = $3Dmol.createViewer("viewport", {{backgroundColor:"#1a1a2e"}});
let pdbdata = `{pdb_str.replace("`","'")}`;
viewer.addModel(pdbdata, "pdb");
{js_styles}
{js_iface}
{js_surfaces}
viewer.zoomTo(); viewer.render();
</script>
<p style="color:#aaa; font-size:12px">
  Interface residues (Cα &lt; 8 Å):<br>
  {iface_html}
</p>
</body></html>"""
    with open(out_path, "w") as fh:
        fh.write(html)


# ── Per-complex processing ────────────────────────────────────────────────────

def process(pred_dir, chain_labels, tag, is_dimer=False):
    """
    chain_labels: dict {chain_letter: label}, e.g. {"A":"Vpr","B":"ZNF586-KRAB"}
                  For dimer: {"A":"KAP1-1","B":"KAP1-2","C":"ZNF586-KRAB"}
    Returns result dict with all data needed for figures.
    """
    print(f"\n{'='*55}")
    print(f"Processing: {tag}  ({' + '.join(chain_labels.values())})")
    data, pdb_path, model_name = load_prediction(pred_dir)
    pae   = np.array(data["pae"])
    plddt = np.array(data["plddt"])
    iptm  = data["iptm"]
    ptm   = data["ptm"]
    print(f"  Model: {model_name}")
    print(f"  ipTM={iptm:.3f}  pTM={ptm:.3f}")

    chain_ca  = read_pdb_ca(pdb_path)
    sorted_ch = sorted(chain_ca.keys())
    chain_lens = [len(chain_ca[ch]) for ch in sorted_ch]
    for ch, L in zip(sorted_ch, chain_lens):
        print(f"  Chain {ch} ({chain_labels.get(ch, '?')}): {L} residues")

    # Interface detection
    if is_dimer:
        kap1_chains = sorted_ch[:2]   # A, B = KAP1 copies
        znf_chain   = sorted_ch[2]    # C = ZNF586
        kap1_iface, znf_iface, kap1_kap1 = interface_residues_dimer(
            chain_ca, kap1_chains, znf_chain)
        chain_iface = {ch: [] for ch in sorted_ch}
        for ch, res in kap1_iface:
            chain_iface[ch].append(res)
        chain_iface[znf_chain] = znf_iface
        znf_iface_flat = znf_iface
        print(f"  KAP1 dimer–dimer interface: {len(kap1_kap1)} residues (chain A)")
        print(f"  KAP1 dimer–ZNF586 interface: KAP1 chain A: {sum(1 for ch,_ in kap1_iface if ch==kap1_chains[0])}, "
              f"chain B: {sum(1 for ch,_ in kap1_iface if ch==kap1_chains[1])}, "
              f"ZNF586: {len(znf_iface)}")
    else:
        ch_a, ch_b = sorted_ch[0], sorted_ch[1]
        iface_a, iface_b, _ = interface_residues_2chain(chain_ca, ch_a, ch_b)
        chain_iface = {ch_a: iface_a, ch_b: iface_b}
        znf_iface_flat = iface_b   # ZNF586 is always chain B in 2-chain complexes
        kap1_kap1 = []
        print(f"  Interface: {len(iface_a)} ({chain_labels[ch_a]}) + "
              f"{len(iface_b)} ({chain_labels[ch_b]})")
        print(f"  {chain_labels[ch_a]}: {', '.join(iface_a)}")
        print(f"  {chain_labels[ch_b]}: {', '.join(iface_b)}")

    # PAE heatmap
    plot_pae(pae, chain_lens,
             [chain_labels.get(ch, ch) for ch in sorted_ch],
             f"{' – '.join(chain_labels.values())}  |  ipTM={iptm:.3f}",
             os.path.join(OUT_DIR, f"PAE_{tag}.png"))
    print(f"  Saved: PAE_{tag}.png")

    # pLDDT
    plot_plddt(plddt, chain_lens,
               [chain_labels.get(ch, ch) for ch in sorted_ch],
               f"pLDDT  |  {' – '.join(chain_labels.values())}  (ipTM={iptm:.3f})",
               os.path.join(OUT_DIR, f"pLDDT_{tag}.png"))
    print(f"  Saved: pLDDT_{tag}.png")

    # PyMOL
    write_pymol(pdb_path, chain_iface, chain_labels,
                os.path.join(OUT_DIR, f"view_{tag}.pml"))
    print(f"  Saved: view_{tag}.pml")

    # HTML
    write_html(pdb_path, iptm, ptm, chain_iface, chain_labels,
               os.path.join(OUT_DIR, f"viewer_{tag}.html"))
    print(f"  Saved: viewer_{tag}.html")

    return dict(pae=pae, plddt=plddt, iptm=iptm, ptm=ptm,
                chain_lens=chain_lens, chain_labels=chain_labels,
                chain_iface=chain_iface, znf_iface=znf_iface_flat,
                kap1_kap1=kap1_kap1, tag=tag, pdb_path=pdb_path,
                is_dimer=is_dimer)


# ── ZNF586 contact overlap figure ─────────────────────────────────────────────

def plot_contact_overlap(results, znf_len=73, out_path=None):
    """
    Bar chart showing which ZNF586 residue positions are contacted by
    Vpr vs KAP1 (monomer) vs KAP1 (dimer). Overlapping positions
    highlight the competition surface.
    """
    def parse_positions(iface_list):
        pos = set()
        for r in iface_list:
            try:
                pos.add(int(r[1:]))
            except ValueError:
                pass
        return pos

    positions = {tag: parse_positions(res["znf_iface"])
                 for tag, res in results.items()}

    # Determine all residue positions covered
    all_pos = sorted(set().union(*positions.values()))
    if not all_pos:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(all_pos) * 0.35 + 2), 4))

    tags      = list(positions.keys())
    tag_colors = {"Vpr-ZNF586": "#e74c3c",
                  "KAP1mono-ZNF586": "#3498db",
                  "KAP1dimer-ZNF586": "#2980b9"}
    bar_h = 0.25

    for i, tag in enumerate(tags):
        y_offset = i * bar_h
        pos_set  = positions[tag]
        label    = (tag.replace("-ZNF586","").replace("KAP1mono","KAP1 monomer")
                      .replace("KAP1dimer","KAP1 dimer"))
        for j, p in enumerate(all_pos):
            ax.bar(j, bar_h * 0.9 if p in pos_set else 0,
                   bottom=y_offset, color=tag_colors.get(tag, "gray"),
                   width=0.8, linewidth=0, alpha=0.85)
        ax.text(-0.8, y_offset + bar_h/2, label,
                ha="right", va="center", fontsize=8,
                color=tag_colors.get(tag, "gray"), fontweight="bold")

    # Highlight positions contacted by ALL groups (competition core)
    overlap_all = set.intersection(*positions.values()) if len(positions) > 1 else set()
    for j, p in enumerate(all_pos):
        if p in overlap_all:
            ax.axvspan(j - 0.45, j + 0.45,
                       ymin=0, ymax=1, color="gold", alpha=0.25, zorder=0)

    ax.set_xticks(range(len(all_pos)))
    ax.set_xticklabels([str(p) for p in all_pos], rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("ZNF586-KRAB residue position (relative to KRAB aa15)")
    ax.set_title("ZNF586-KRAB interface residues: Vpr vs KAP1 competition surface\n"
                 "(gold shading = residues contacted by all compared complexes)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(-1.5, len(all_pos) - 0.5)
    ax.set_ylim(-0.05, len(tags) * bar_h + 0.1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(OUT_DIR, "znf586_contact_overlap.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: znf586_contact_overlap.png")
    if overlap_all:
        print(f"  Competition core residues (contacted by all): "
              + ", ".join(str(p) for p in sorted(overlap_all)))


# ── Combined comparison figure ─────────────────────────────────────────────────

def plot_comparison(result_list):
    """
    result_list: list of result dicts from process(), up to 3 complexes.
    Layout: rows = [PAE, pLDDT], cols = each complex + final ipTM bar.
    """
    n = len(result_list)
    fig = plt.figure(figsize=(5 * (n + 1), 10))
    gs  = gridspec.GridSpec(2, n + 1, figure=fig, hspace=0.5, wspace=0.4,
                            width_ratios=[1] * n + [0.7])

    chain_colors_hex = ["#FF7D45", "#4575b4", "#2ca02c"]
    bar_colors = ["#e74c3c", "#3498db", "#2980b9", "#8e44ad"]

    for col, res in enumerate(result_list):
        pae        = res["pae"]
        plddt      = res["plddt"]
        chain_lens = res["chain_lens"]
        iptm       = res["iptm"]
        labels     = [res["chain_labels"].get(ch, ch)
                      for ch in sorted(res["chain_labels"])]
        L          = sum(chain_lens)
        boundaries = np.cumsum([0] + chain_lens)
        title      = " – ".join(labels)

        # PAE heatmap
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(pae[:L, :L], cmap="bwr_r", vmin=0, vmax=30, aspect="auto")
        plt.colorbar(im, ax=ax, label="PAE (Å)", fraction=0.046)
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            ax.axhline(e - 0.5, color="white", lw=1.5)
            ax.axvline(e - 0.5, color="white", lw=1.5)
        for i, (si, ei) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            for j, (sj, ej) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                if i != j:
                    ax.add_patch(Rectangle((sj-0.5, si-0.5), ej-sj, ei-si,
                                           lw=1.0, edgecolor="gold", facecolor="none"))
        ax.set_title(f"{title}\nipTM = {iptm:.3f}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Residue"); ax.set_ylabel("Residue")

        # pLDDT
        ax = fig.add_subplot(gs[1, col])
        colors = []
        for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            colors += [chain_colors_hex[i % len(chain_colors_hex)]] * (e - s)
        ax.bar(range(L), plddt[:L], color=colors, width=1.0, linewidth=0)
        for e in boundaries[1:-1]:
            ax.axvline(e - 0.5, color="black", lw=1.0, ls="--")
        ax.axhline(70, color="gray", lw=0.7, ls=":")
        ax.axhline(90, color="gray", lw=0.7, ls="-.")
        ax.set_xlim(-0.5, L - 0.5); ax.set_ylim(0, 100)
        ax.set_xlabel("Residue"); ax.set_ylabel("pLDDT")
        ax.set_title(f"pLDDT  |  {title}", fontsize=8, fontweight="bold")
        ax.legend(handles=[Patch(color=chain_colors_hex[i % len(chain_colors_hex)],
                                 label=labels[i]) for i in range(len(labels))],
                  fontsize=6, ncol=2)

    # ipTM bar chart (rightmost column, spanning both rows)
    ax = fig.add_subplot(gs[:, n])
    bar_labels = ["\n".join(res["chain_labels"].values()) for res in result_list]
    bar_vals   = [res["iptm"] for res in result_list]
    bars = ax.bar(range(len(result_list)), bar_vals,
                  color=bar_colors[:len(result_list)], alpha=0.85, width=0.55)
    for bar, v in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.8, color="black", ls="--", lw=0.9, label="0.8 (confident)")
    ax.axhline(0.6, color="gray",  ls="--", lw=0.9, label="0.6 (grey zone)")
    ax.axhline(0.5, color="gray",  ls=":",  lw=0.9, label="0.5 (threshold)")
    ax.set_xticks(range(len(result_list)))
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_ylim(0, 1.0); ax.set_ylabel("ipTM")
    ax.set_title("ipTM summary", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("Vpr competes with KAP1 for ZNF586-KRAB binding",
                 fontsize=13, fontweight="bold", y=1.01)
    out = os.path.join(OUT_DIR, "comparison_figure.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: comparison_figure.png")


# ── Main ──────────────────────────────────────────────────────────────────────

results = {}

# (A) Vpr + ZNF586-KRAB  (existing)
res_vpr = process(
    pred_dir     = os.path.join(PRED_DIR, "Vpr_ZNF586-KRAB"),
    chain_labels = {"A": "Vpr", "B": "ZNF586-KRAB"},
    tag          = "Vpr-ZNF586",
)
results["Vpr-ZNF586"] = res_vpr

# (B) KAP1-RBCC monomer + ZNF586-KRAB
kap1_mono_dir = os.path.join(PRED_DIR, "KAP1-RBCC_ZNF586-KRAB")
if os.path.isdir(kap1_mono_dir) and glob.glob(os.path.join(kap1_mono_dir, "*.done.txt")):
    res_kap1_mono = process(
        pred_dir     = kap1_mono_dir,
        chain_labels = {"A": "KAP1-RBCC", "B": "ZNF586-KRAB"},
        tag          = "KAP1mono-ZNF586",
    )
    results["KAP1mono-ZNF586"] = res_kap1_mono
else:
    print(f"\nSkipping KAP1-RBCC monomer+ZNF586: prediction not yet complete.")

# (C) KAP1-RBCC dimer + ZNF586-KRAB
kap1_dimer_dir = os.path.join(PRED_DIR, "KAP1-RBCC_dimer_ZNF586-KRAB")
if os.path.isdir(kap1_dimer_dir) and glob.glob(os.path.join(kap1_dimer_dir, "*.done.txt")):
    res_kap1_dimer = process(
        pred_dir     = kap1_dimer_dir,
        chain_labels = {"A": "KAP1-RBCC-1", "B": "KAP1-RBCC-2", "C": "ZNF586-KRAB"},
        tag          = "KAP1dimer-ZNF586",
        is_dimer     = True,
    )
    results["KAP1dimer-ZNF586"] = res_kap1_dimer
else:
    print(f"\nSkipping KAP1-RBCC dimer+ZNF586: prediction not yet complete.")

# Combined figures (run when ≥2 results available)
if len(results) >= 2:
    plot_comparison(list(results.values()))
    plot_contact_overlap(results)

print(f"\n{'='*55}")
print(f"All outputs in: {OUT_DIR}/")
for tag in results:
    print(f"  PAE_{tag}.png  pLDDT_{tag}.png  view_{tag}.pml  viewer_{tag}.html")
if len(results) >= 2:
    print(f"  comparison_figure.png")
    print(f"  znf586_contact_overlap.png")
print(f"\nTo view in PyMOL:")
for tag in results:
    print(f"  pymol {OUT_DIR}/view_{tag}.pml")
