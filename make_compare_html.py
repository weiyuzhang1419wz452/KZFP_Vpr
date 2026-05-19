#!/usr/bin/env python3
"""
Generate a side-by-side HTML viewer:
  Left  — Vpr + ZNF586-KRAB          (ipTM 0.620)
  Right — KAP1-RBCC dimer + ZNF586-KRAB (ipTM 0.580)

Both complexes are superimposed onto the ZNF586-KRAB Cα atoms so that
ZNF586 occupies the same position in both panels, making it easy to
compare how Vpr vs KAP1-dimer approach the same surface.
Competition-core residues (contacted in both complexes) are shown in magenta.
"""

import glob, json, os
import numpy as np

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
PRED_DIR = os.path.join(BASE_DIR, "predictions")
OUT_DIR  = os.path.join(BASE_DIR, "analysis", "vis_ZNF586-KAP1")
OUT_HTML = os.path.join(OUT_DIR, "compare_Vpr_vs_KAP1dimer_ZNF586.html")

PAE_THR  = 8.0
AA1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


# ── helpers ───────────────────────────────────────────────────────────────────

def best_pdb_and_scores(pred_dir):
    jsons = glob.glob(os.path.join(pred_dir, "*scores*.json"))
    best  = max(jsons, key=lambda f: json.load(open(f)).get("iptm", 0))
    data  = json.load(open(best))
    pdb   = best.replace("_scores_", "_unrelaxed_").replace(".json", ".pdb")
    return pdb, data["iptm"], data["ptm"]


def parse_pdb(pdb_path):
    """Return list of ATOM lines."""
    lines = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                lines.append(line)
    return lines


def get_ca_coords(atom_lines, chain):
    coords, idx = [], []
    for i, line in enumerate(atom_lines):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            idx.append(i)
    return np.array(coords), idx


def kabsch(P, Q):
    """Return rotation R and translation t such that P@R.T + t ≈ Q (minimise RMSD)."""
    Pc = P.mean(0); Qc = Q.mean(0)
    P0 = P - Pc;    Q0 = Q - Qc
    H  = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    R  = Vt.T @ np.diag([1, 1, d]) @ U.T
    t  = Qc - Pc @ R.T
    return R, t


def apply_transform(atom_lines, R, t):
    """Return new PDB string with all ATOM coordinates rotated/translated."""
    out = []
    for line in atom_lines:
        if line.startswith("ATOM"):
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            xyz = np.array([x, y, z]) @ R.T + t
            line = (line[:30]
                    + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    + line[54:])
        out.append(line)
    return "".join(out)


def iface_sets(atom_lines, ch_a, ch_b, thr=PAE_THR):
    """Cα contact residue sets between two chains."""
    def ca(chain):
        coords, rids = [], []
        for line in atom_lines:
            if line.startswith("ATOM") and line[12:16].strip()=="CA" and line[21]==chain:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                rids.append(int(line[22:26].strip()))
        return np.array(coords), rids
    ca_a, ra = ca(ch_a)
    ca_b, rb = ca(ch_b)
    d = np.linalg.norm(ca_a[:, None] - ca_b[None], axis=-1)
    set_a = {ra[i] for i in range(len(ra)) if np.any(d[i] < thr)}
    set_b = {rb[j] for j in range(len(rb)) if np.any(d[:, j] < thr)}
    return set_a, set_b


def resi_js(pos_set):
    return ", ".join(str(p) for p in sorted(pos_set))


# ── load raw PDBs ─────────────────────────────────────────────────────────────

vpr_pdb_path, vpr_iptm, vpr_ptm = best_pdb_and_scores(
    os.path.join(PRED_DIR, "Vpr_ZNF586-KRAB"))
dim_pdb_path, dim_iptm, dim_ptm = best_pdb_and_scores(
    os.path.join(PRED_DIR, "KAP1-RBCC_dimer_ZNF586-KRAB"))

vpr_lines = parse_pdb(vpr_pdb_path)   # A=Vpr, B=ZNF586
dim_lines = parse_pdb(dim_pdb_path)   # A=KAP1-1, B=KAP1-2, C=ZNF586


# ── superimpose BOTH complexes onto ZNF586-KRAB ───────────────────────────────
# Reference frame: ZNF586-KRAB from the Vpr complex (chain B, kept fixed)
# The dimer complex is rotated to match.

znf_vpr_ca, _ = get_ca_coords(vpr_lines, "B")   # reference
znf_dim_ca, _ = get_ca_coords(dim_lines, "C")   # to align

n = min(len(znf_vpr_ca), len(znf_dim_ca))
R, t = kabsch(znf_dim_ca[:n], znf_vpr_ca[:n])
rmsd = np.sqrt((((znf_dim_ca[:n] @ R.T + t) - znf_vpr_ca[:n])**2).sum(1).mean())
print(f"ZNF586-KRAB Cα RMSD before superposition: "
      f"{np.sqrt(((znf_vpr_ca[:n] - znf_dim_ca[:n])**2).sum(1).mean()):.2f} Å")
print(f"ZNF586-KRAB Cα RMSD after  superposition: {rmsd:.2f} Å  (n={n})")

# Apply transform to dimer complex
dim_pdb_aligned_str = apply_transform(dim_lines, R, t).replace("`", "'")
# Vpr complex stays in original frame
vpr_pdb_str = "".join(vpr_lines).replace("`", "'")


# ── interface residues ────────────────────────────────────────────────────────

vpr_iface_vpr, vpr_iface_znf = iface_sets(vpr_lines, "A", "B")
dim_iface_kap1A, dim_iface_znf_A = iface_sets(dim_lines, "A", "C")
dim_iface_kap1B, dim_iface_znf_B = iface_sets(dim_lines, "B", "C")
dim_iface_znf = dim_iface_znf_A | dim_iface_znf_B

competition_core = vpr_iface_znf & dim_iface_znf

print(f"\nVpr    ZNF586 interface ({len(vpr_iface_znf)}): {sorted(vpr_iface_znf)}")
print(f"KAP1d  ZNF586 interface ({len(dim_iface_znf)}): {sorted(dim_iface_znf)}")
print(f"Competition core        ({len(competition_core)}): {sorted(competition_core)}")


# ── build HTML ────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vpr vs KAP1-dimer competing for ZNF586-KRAB binding</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f1724;
    color: #e0e0e0;
    padding: 18px;
  }}
  h1 {{ text-align:center; color:#00d4ff; font-size:1.3em; margin-bottom:5px; }}
  .subtitle {{ text-align:center; color:#aaa; font-size:0.82em; margin-bottom:14px; }}
  .panels {{ display:flex; gap:14px; justify-content:center; }}
  .panel {{
    flex:1; max-width:680px;
    background:#162033; border-radius:10px; padding:10px;
    border:1px solid #2a3a55;
  }}
  .panel h2 {{ font-size:0.95em; margin-bottom:6px; text-align:center; }}
  .viewer {{
    width:100%; height:520px;
    border-radius:6px; border:1.5px solid #2a3a55; position:relative;
  }}
  .legend {{ margin-top:8px; font-size:0.75em; line-height:1.8; color:#bbb; }}
  .dot {{
    display:inline-block; width:10px; height:10px;
    border-radius:50%; margin-right:5px; vertical-align:middle;
  }}
  .tag {{
    display:inline-block; background:#1e2d45; border-radius:4px;
    padding:1px 7px; font-size:0.78em; margin-left:5px; color:#ddd;
  }}
  .core-box {{
    margin-top:12px; background:#1a1030;
    border:1px solid #7b2fbe; border-radius:6px;
    padding:8px 14px; font-size:0.8em; text-align:center; color:#e0aaff;
  }}
  .core-box b {{ color:#ff79f7; }}
  .align-note {{
    text-align:center; color:#8ecfff; font-size:0.78em; margin-bottom:10px;
    font-style:italic;
  }}
</style>
</head>
<body>

<h1>Vpr competes with KAP1 for ZNF586-KRAB binding</h1>
<div class="subtitle">
  AlphaFold2-Multimer (ColabFold v1.5.5) &nbsp;|&nbsp;
  <span style="color:#69d84f"><b>Green</b></span> = ZNF586-KRAB (same position in both panels) &nbsp;|&nbsp;
  <span style="color:#ff79f7"><b>Magenta</b></span> = competition core
</div>
<div class="align-note">
  Both complexes are superimposed on ZNF586-KRAB Cα atoms (post-alignment RMSD = {rmsd:.2f} Å),
  so ZNF586 occupies the same position in both panels for direct comparison.
</div>

<div class="panels">

  <!-- LEFT: Vpr + ZNF586 -->
  <div class="panel">
    <h2 style="color:#FF7D45;">
      Vpr &nbsp;+&nbsp; ZNF586-KRAB
      <span class="tag">ipTM = {vpr_iptm:.3f}</span>
      <span class="tag">pTM = {vpr_ptm:.3f}</span>
    </h2>
    <div id="vpr_view" class="viewer"></div>
    <div class="legend">
      <span class="dot" style="background:#FF7D45"></span><b>Orange</b> — Vpr (96 aa)<br>
      <span class="dot" style="background:#69d84f"></span><b>Green</b> — ZNF586-KRAB (73 aa)<br>
      <span class="dot" style="background:#ffcc00"></span><b>Yellow sticks</b> — Vpr interface residues<br>
      <span class="dot" style="background:#00e5cc"></span><b>Cyan sticks</b> — ZNF586-specific interface<br>
      <span class="dot" style="background:#ff79f7"></span><b>Magenta</b> — competition core on ZNF586
    </div>
  </div>

  <!-- RIGHT: KAP1-dimer + ZNF586 -->
  <div class="panel">
    <h2 style="color:#4fc3f7;">
      KAP1-RBCC dimer &nbsp;+&nbsp; ZNF586-KRAB
      <span class="tag">ipTM = {dim_iptm:.3f}</span>
      <span class="tag">pTM = {dim_ptm:.3f}</span>
    </h2>
    <div id="dim_view" class="viewer"></div>
    <div class="legend">
      <span class="dot" style="background:#4575b4"></span><b>Blue</b> — KAP1-RBCC chain 1 (480 aa)<br>
      <span class="dot" style="background:#74b9d4"></span><b>Light blue</b> — KAP1-RBCC chain 2 (480 aa)<br>
      <span class="dot" style="background:#69d84f"></span><b>Green</b> — ZNF586-KRAB (73 aa, same orientation as left)<br>
      <span class="dot" style="background:#00e5cc"></span><b>Cyan sticks</b> — ZNF586-specific interface<br>
      <span class="dot" style="background:#ff79f7"></span><b>Magenta</b> — competition core on ZNF586
    </div>
  </div>

</div>

<div class="core-box">
  <b>Competition core — ZNF586-KRAB residues contacted by BOTH Vpr and KAP1-RBCC dimer
  ({len(competition_core)} residues):</b><br>
  positions {', '.join(str(p) for p in sorted(competition_core))}
</div>

<script>
// ── Left viewer: Vpr + ZNF586 (reference frame) ───────────────────────────────
(function() {{
  let v = $3Dmol.createViewer("vpr_view", {{backgroundColor:"#0f1724"}});
  v.addModel(`{vpr_pdb_str}`, "pdb");
  v.setStyle({{chain:"A"}}, {{cartoon:{{color:"#FF7D45", thickness:0.45}}}});
  v.setStyle({{chain:"B"}}, {{cartoon:{{color:"#69d84f", thickness:0.45}}}});
  v.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.10,color:"#FF7D45"}}, {{chain:"A"}});
  v.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.12,color:"#69d84f"}}, {{chain:"B"}});
  // Vpr interface (yellow)
  v.addStyle({{chain:"A", resi:[{resi_js(vpr_iface_vpr)}]}},
    {{stick:{{color:"#ffcc00",radius:0.22}}}});
  // ZNF586 non-core interface (cyan)
  let znf_nc_v = [{resi_js(vpr_iface_znf - competition_core)}];
  if (znf_nc_v.length) v.addStyle({{chain:"B", resi:znf_nc_v}},
    {{stick:{{color:"#00e5cc",radius:0.22}}}});
  // ZNF586 competition core (magenta + spheres)
  let core = [{resi_js(competition_core)}];
  v.addStyle({{chain:"B", resi:core}}, {{stick:{{color:"#ff79f7",radius:0.30}}}});
  v.addStyle({{chain:"B", resi:core}}, {{sphere:{{color:"#ff79f7",radius:0.50}}}});
  v.zoomTo(); v.render();
}})();

// ── Right viewer: KAP1-dimer + ZNF586 (aligned to Vpr frame) ──────────────────
(function() {{
  let v = $3Dmol.createViewer("dim_view", {{backgroundColor:"#0f1724"}});
  v.addModel(`{dim_pdb_aligned_str}`, "pdb");
  v.setStyle({{chain:"A"}}, {{cartoon:{{color:"#4575b4",thickness:0.45}}}});
  v.setStyle({{chain:"B"}}, {{cartoon:{{color:"#74b9d4",thickness:0.45}}}});
  v.setStyle({{chain:"C"}}, {{cartoon:{{color:"#69d84f",thickness:0.45}}}});
  v.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.08,color:"#4575b4"}}, {{chain:"A"}});
  v.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.08,color:"#74b9d4"}}, {{chain:"B"}});
  v.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.12,color:"#69d84f"}}, {{chain:"C"}});
  // KAP1 chain A interface (yellow)
  v.addStyle({{chain:"A", resi:[{resi_js(dim_iface_kap1A)}]}},
    {{stick:{{color:"#ffcc00",radius:0.22}}}});
  // KAP1 chain B interface (orange)
  v.addStyle({{chain:"B", resi:[{resi_js(dim_iface_kap1B)}]}},
    {{stick:{{color:"#ffa500",radius:0.22}}}});
  // ZNF586 non-core interface (cyan)
  let znf_nc_d = [{resi_js(dim_iface_znf - competition_core)}];
  if (znf_nc_d.length) v.addStyle({{chain:"C", resi:znf_nc_d}},
    {{stick:{{color:"#00e5cc",radius:0.22}}}});
  // ZNF586 competition core (magenta + spheres)
  let core = [{resi_js(competition_core)}];
  v.addStyle({{chain:"C", resi:core}}, {{stick:{{color:"#ff79f7",radius:0.30}}}});
  v.addStyle({{chain:"C", resi:core}}, {{sphere:{{color:"#ff79f7",radius:0.50}}}});
  v.zoomTo(); v.render();
}})();
</script>
</body>
</html>
"""

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_HTML, "w") as fh:
    fh.write(html)
print(f"\nSaved: {OUT_HTML}")
