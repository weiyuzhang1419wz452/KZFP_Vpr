#!/usr/bin/env python3
"""
setup_f69a_e29a_panel.py
========================
Set up all predictions needed to characterise F69A and E29A:

  1. Vpr monomer (WT / F69A / E29A) — structural integrity
  2. E29A vs known Vpr targets: DCAF1-WD40full, HLTF-HIRANv2
  3. F69A + UNG2, E29A + UNG2
  4. WT / F69A / E29A + MUS81 (aa480-786, ERCC4 domain)

Already done (skipped):
  - F69A + DCAF1 (ipTM=0.71)
  - F69A + HLTF  (ipTM=0.46)
  - WT  + UNG2   (ipTM=0.86)
  - WT  + DCAF1  (ipTM=0.77)
  - WT  + HLTF   (ipTM=0.82)
"""

import glob, os, subprocess

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
SEQ_DIR  = os.path.join(BASE_DIR, "sequences")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
LOG_DIR  = os.path.join(BASE_DIR, "logs")

VPR_WT   = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
VPR_F69A = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLAIHFRIGCRHSRIGVTRQRRARNGASRS"
VPR_E29A = "MEQAPEDQGPQREPYNEWTLELLEELKSAAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"

DCAF1    = "QRRQAPINFTSRLNRRASFPKYGGVDGGCFDRHLIFSRFRPISVFREANEDESGFTCCAFSARERFLMLGTCTGQLKLYNVFSGQEEASYNCHNSAITHLEPSRDGSLLLTSATWSQPLSALWGMKSVFDMKHSFTEDHYVEFSKHSQDRVIGTKGDIAHIYDIQTGNKLLTLFNPDLANNYKRNCATFNPTDDLVLNDGVLWDVRSAQAIHKFDKFNMNISGVFHPNGLEVIINTEIWDLRTFHLLHTVPALDQCRVVFNHTGTVMYGAMLQADDEDDLMEERMKSPFGSSFRTFNATDYKPIATIDVKRNIFDLCTDTKDCYLAVIENQGSMDALNMDTVCRLYEVGRQRLA"
HLTF     = "VDSVLFGSLRGHVVGLRYYTGVVNNNEMVALQRDPNNPYDKNAIKVNNVNGNQVGHLKKELAGALAYIMDNKLAQIEGVVPFGANNAFTMPLHMTFWGKEENRKAVSDQLKKHGFKLGPAPKTLGF"
UNG2     = open(os.path.join(SEQ_DIR, "Vpr_UNG2.fasta")).read().strip().split("\n")[1].split(":")[1]
MUS81    = "EGEVTTMNHEDLSLLKEILKRPVDPIRAAGLHPTAEQIEMFAYHLPDATLSNLIDIFVDFSQVDGQYFVCNMDDFKFSAELIQHIPLSLRVRYVFCTAPINKKQPFVCSSLLQFARQYSRNEPLTFAWLRRYIKWPLLPPKNIKDLMDLEAVHDVLDLYLWLSYRFMDMFPDASLIRDLQKELDGIIQDGVHNITKLIKMSETHKLLNLEGFPSGSQSRLSGTLKSQARRTRGTKALGSKATEPPSPDAGELSLASRLVQQGLLTPDMLKQLEKEWMTQQTEHNKEKTESGTHPKGTRRKKKEPDSD"

# (header_name, fasta_name, vpr_label, vpr_seq, partner_label, partner_seq, model_type)
JOBS = [
    # ── Monomers ────────────────────────────────────────────────────────────────
    ("HIV1_Vpr",      "Vpr_monomer",      "WT",   VPR_WT,   None, None, "alphafold2_ptm"),
    ("HIV1_Vpr_F69A", "VprF69A_monomer",  "F69A", VPR_F69A, None, None, "alphafold2_ptm"),
    ("HIV1_Vpr_E29A", "VprE29A_monomer",  "E29A", VPR_E29A, None, None, "alphafold2_ptm"),
    # ── E29A vs known targets ────────────────────────────────────────────────────
    ("HIV1_Vpr_E29A:DCAF1_WD40full_aa1042-1395", "VprE29A_DCAF1-WD40full", "E29A", VPR_E29A, "DCAF1_WD40full", DCAF1, "alphafold2_multimer_v3"),
    ("HIV1_Vpr_E29A:HLTF_HIRAN_aa55-180",        "VprE29A_HLTF-HIRANv2",   "E29A", VPR_E29A, "HLTF_HIRAN",    HLTF,  "alphafold2_multimer_v3"),
    # ── UNG2 ────────────────────────────────────────────────────────────────────
    ("HIV1_Vpr_F69A:UNG2_aa84-313", "VprF69A_UNG2", "F69A", VPR_F69A, "UNG2_aa84-313", UNG2, "alphafold2_multimer_v3"),
    ("HIV1_Vpr_E29A:UNG2_aa84-313", "VprE29A_UNG2", "E29A", VPR_E29A, "UNG2_aa84-313", UNG2, "alphafold2_multimer_v3"),
    # ── MUS81 ───────────────────────────────────────────────────────────────────
    ("HIV1_Vpr:MUS81_aa480-786",      "Vpr_MUS81",      "WT",   VPR_WT,   "MUS81_aa480-786", MUS81, "alphafold2_multimer_v3"),
    ("HIV1_Vpr_F69A:MUS81_aa480-786", "VprF69A_MUS81",  "F69A", VPR_F69A, "MUS81_aa480-786", MUS81, "alphafold2_multimer_v3"),
    ("HIV1_Vpr_E29A:MUS81_aa480-786", "VprE29A_MUS81",  "E29A", VPR_E29A, "MUS81_aa480-786", MUS81, "alphafold2_multimer_v3"),
]


def is_complete(pred_dir):
    return bool(glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")) or
                glob.glob(os.path.join(pred_dir, "*scores*model_1*.json")))


def write_fasta(path, header, vpr_seq, partner_seq):
    with open(path, "w") as fh:
        if partner_seq:
            fh.write(f">{header}\n{vpr_seq}:{partner_seq}\n")
        else:
            fh.write(f">{header}\n{vpr_seq}\n")


pending = []

for header, name, vpr_label, vpr_seq, partner_label, partner_seq, model_type in JOBS:
    fasta   = os.path.join(SEQ_DIR, f"{name}.fasta")
    out_dir = os.path.join(PRED_DIR, name)

    if is_complete(out_dir):
        print(f"SKIP (done): {name}")
        continue

    os.makedirs(out_dir, exist_ok=True)
    write_fasta(fasta, header, vpr_seq, partner_seq)
    pending.append((name, fasta, out_dir, model_type))
    print(f"QUEUED: {name}")

print(f"\n{len(pending)} jobs to run")

if not pending:
    print("All done.")
else:
    # Write pending list for array
    pending_file = os.path.join(BASE_DIR, "pending_panel.txt")
    meta_file    = os.path.join(BASE_DIR, "pending_panel_meta.tsv")
    with open(pending_file, "w") as fh:
        fh.write("\n".join(f[1] for f in pending) + "\n")
    with open(meta_file, "w") as fh:
        for name, fasta, out_dir, model_type in pending:
            fh.write(f"{fasta}\t{out_dir}\t{model_type}\n")

    n = len(pending)
    script = f"""#!/bin/bash
#SBATCH --job-name=VprPanel
#SBATCH --array=1-{n}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output={LOG_DIR}/panel_%a.out
#SBATCH --error={LOG_DIR}/panel_%a.err

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {meta_file})
FASTA=$(echo "$LINE" | cut -f1)
OUTDIR=$(echo "$LINE" | cut -f2)
MODEL=$(echo "$LINE" | cut -f3)
NAME=$(basename "$FASTA" .fasta)

echo "[${{SLURM_ARRAY_TASK_ID}}/{n}] $NAME  model=$MODEL  on $(hostname)"
date

source /workdir/wz452/miniconda/etc/profile.d/conda.sh
conda activate colabfold
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

colabfold_batch \\
    --num-recycle 3 --num-models 5 \\
    --model-type "$MODEL" \\
    "$FASTA" "$OUTDIR"

echo "Done: $NAME"
date
"""
    script_path = os.path.join(BASE_DIR, "run_panel_array.sh")
    with open(script_path, "w") as fh:
        fh.write(script)

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("sbatch error:", result.stderr.strip())
