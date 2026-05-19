#!/usr/bin/env python3
"""
setup_slx4_panel.py
===================
Test Vpr (WT / F69A / E29A) against two SLX4 C-terminal domains:
  - SLX4 aa1430-1834 (full C-term incl. MUS81- and SLX1-binding regions)
  - SLX4 aa1633-1834 (minimal SLX1-binding domain / core Vpr-interaction region)

SLX4 (Q8IY92) is the scaffold that bridges Vpr to the SLX4-MUS81-EME1 complex
(Laguette et al., Cell 2014; DOI: 10.1016/j.cell.2013.12.010).
Vpr directly contacts the C-terminus of SLX4; interaction with MUS81 is indirect.
"""

import glob, os, subprocess

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
SEQ_DIR  = os.path.join(BASE_DIR, "sequences")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
LOG_DIR  = os.path.join(BASE_DIR, "logs")

VPR_WT   = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
VPR_F69A = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLAIHFRIGCRHSRIGVTRQRRARNGASRS"
VPR_E29A = "MEQAPEDQGPQREPYNEWTLELLEELKSAAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"

# SLX4 (Q8IY92) C-terminal domains
SLX4_1430 = "MEPLSPIPIDHWNLERTGPLSTSSPSRRMNEAADSRDCRSPGLLDTTPIRGSCTTQRKLQEKSSGAGSLGNSRPSFLNSALWDVWDGEEQRPPETPPPAQMPSAGGAQKPEGLETPKGANRKKNLPPKVPITPMPQYSIMETPVLKKELDRFGVRPLPKRQMVLKLKEIFQYTHQTLDSDSEDESQSSQPLLQAPHCQTLASQTYKPSRAGVHAQQEATTGPGAHRPKGPAKTKGPRHQRKHHESITPPSRSPTKEAPPGLNDDAQIPASQESVATSVDGSDSSLSSQSSSSCEFGAAFESAGEEEGEGEVSASQAAVQAADTDEALRCYIRSKPALYQKVLLYQPFELRELQAELRQNGLRVSSRRLLDFLDTHCITFTTAATRREKLQGRRRQPRGKKKVERN"
SLX4_1633 = "TYKPSRAGVHAQQEATTGPGAHRPKGPAKTKGPRHQRKHHESITPPSRSPTKEAPPGLNDDAQIPASQESVATSVDGSDSSLSSQSSSSCEFGAAFESAGEEEGEGEVSASQAAVQAADTDEALRCYIRSKPALYQKVLLYQPFELRELQAELRQNGLRVSSRRLLDFLDTHCITFTTAATRREKLQGRRRQPRGKKKVERN"

# (fasta_header, fasta_name, vpr_seq, partner_label, partner_seq)
JOBS = [
    # Broader C-terminal region aa1430-1834 (405 aa)
    ("HIV1_Vpr:SLX4_aa1430-1834",      "Vpr_SLX4-1430",      VPR_WT,   "SLX4_aa1430-1834", SLX4_1430),
    ("HIV1_Vpr_F69A:SLX4_aa1430-1834", "VprF69A_SLX4-1430",  VPR_F69A, "SLX4_aa1430-1834", SLX4_1430),
    ("HIV1_Vpr_E29A:SLX4_aa1430-1834", "VprE29A_SLX4-1430",  VPR_E29A, "SLX4_aa1430-1834", SLX4_1430),
    # Minimal SLX1-binding domain aa1633-1834 (202 aa)
    ("HIV1_Vpr:SLX4_aa1633-1834",      "Vpr_SLX4-1633",      VPR_WT,   "SLX4_aa1633-1834", SLX4_1633),
    ("HIV1_Vpr_F69A:SLX4_aa1633-1834", "VprF69A_SLX4-1633",  VPR_F69A, "SLX4_aa1633-1834", SLX4_1633),
    ("HIV1_Vpr_E29A:SLX4_aa1633-1834", "VprE29A_SLX4-1633",  VPR_E29A, "SLX4_aa1633-1834", SLX4_1633),
]


def is_complete(pred_dir):
    return bool(glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")))


os.makedirs(SEQ_DIR,  exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

pending = []

for header, name, vpr_seq, partner_label, partner_seq in JOBS:
    fasta   = os.path.join(SEQ_DIR, f"{name}.fasta")
    out_dir = os.path.join(PRED_DIR, name)

    if is_complete(out_dir):
        print(f"SKIP (done): {name}")
        continue

    os.makedirs(out_dir, exist_ok=True)
    with open(fasta, "w") as fh:
        fh.write(f">{header}\n{vpr_seq}:{partner_seq}\n")
    pending.append((name, fasta, out_dir))
    print(f"QUEUED: {name}")

print(f"\n{len(pending)} jobs to run")

if not pending:
    print("All done.")
else:
    meta_file = os.path.join(BASE_DIR, "pending_slx4_meta.tsv")
    with open(meta_file, "w") as fh:
        for name, fasta, out_dir in pending:
            fh.write(f"{fasta}\t{out_dir}\talphafold2_multimer_v3\n")

    n = len(pending)
    script = f"""#!/bin/bash
#SBATCH --job-name=VprSLX4
#SBATCH --array=1-{n}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output={LOG_DIR}/slx4_%a.out
#SBATCH --error={LOG_DIR}/slx4_%a.err

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
    script_path = os.path.join(BASE_DIR, "run_slx4_array.sh")
    with open(script_path, "w") as fh:
        fh.write(script)

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("sbatch error:", result.stderr.strip())
