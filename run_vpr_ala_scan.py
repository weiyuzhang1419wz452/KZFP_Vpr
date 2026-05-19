#!/usr/bin/env python3
"""
run_vpr_ala_scan.py
===================
Alanine scan of all 96 Vpr residues paired with ZNF430 KRAB (aa35-107).
Each residue is mutated to Ala; existing Ala residues are mutated to Gly.

Creates:
  sequences/ala_scan/Vpr_{WT}{pos}{MUT}_ZNF430-KRAB.fasta   (96 files)
  predictions/ala_scan/Vpr_{WT}{pos}{MUT}_ZNF430-KRAB/      (output dirs)
  run_ala_scan_array.sh                                       (SLURM array)

Usage:
  python run_vpr_ala_scan.py              # prep + submit
  python run_vpr_ala_scan.py --dry        # prep only, no sbatch
"""

import argparse, glob, os, subprocess

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
SEQ_DIR  = os.path.join(BASE_DIR, "sequences", "ala_scan")
PRED_DIR = os.path.join(BASE_DIR, "predictions", "ala_scan")
LOG_DIR  = os.path.join(BASE_DIR, "logs", "ala_scan")

for d in (SEQ_DIR, PRED_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

VPR_WT = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
ZNF430_KRAB = "LTFRDVAIEFSLEEWQCLDTAQQDLYRKVMLENYRNLVFLAGIAVSKPDLITCLEQGKEPWNMKRHAMVDQPP"
ZNF430_COORDS = "aa35-107"

SLURM_CPUS = 8
SLURM_MEM  = "32G"
SLURM_TIME = "12:00:00"
SLURM_CONC = 30   # max concurrent jobs


def is_complete(pred_dir):
    return bool(glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")))


def make_mutant(seq, pos0, mut_aa):
    """Return sequence with 0-indexed pos0 substituted to mut_aa."""
    return seq[:pos0] + mut_aa + seq[pos0+1:]


def main(dry_run=False):
    pending = []   # list of fasta paths to submit

    for pos0, wt_aa in enumerate(VPR_WT):
        pos1   = pos0 + 1
        mut_aa = "G" if wt_aa == "A" else "A"
        label  = f"Vpr_{wt_aa}{pos1}{mut_aa}_ZNF430-KRAB"

        fasta_path = os.path.join(SEQ_DIR, f"{label}.fasta")
        out_dir    = os.path.join(PRED_DIR, label)

        if is_complete(out_dir):
            print(f"  DONE  {label}")
            continue

        # Write FASTA
        mut_seq = make_mutant(VPR_WT, pos0, mut_aa)
        os.makedirs(out_dir, exist_ok=True)
        with open(fasta_path, "w") as fh:
            fh.write(f">HIV1_Vpr_{wt_aa}{pos1}{mut_aa}:ZNF430_KRAB_{ZNF430_COORDS}\n"
                     f"{mut_seq}:{ZNF430_KRAB}\n")

        pending.append(fasta_path)

    print(f"\nPending: {len(pending)} jobs  |  Already done: {96 - len(pending)}")

    if not pending:
        print("All predictions already complete.")
        return

    # Write pending list
    pending_file = os.path.join(BASE_DIR, "pending_ala_scan.txt")
    with open(pending_file, "w") as fh:
        fh.write("\n".join(pending) + "\n")

    n = len(pending)
    script = f"""#!/bin/bash
#SBATCH --job-name=VprAlaScan
#SBATCH --array=1-{n}%{SLURM_CONC}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={LOG_DIR}/ala_%a.out
#SBATCH --error={LOG_DIR}/ala_%a.err

FASTA=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {pending_file})
[ -z "$FASTA" ] && exit 0

NAME=$(basename "$FASTA" .fasta)
OUTDIR="{PRED_DIR}/$NAME"
mkdir -p "$OUTDIR"

echo "[${{SLURM_ARRAY_TASK_ID}}/{n}] $NAME on $(hostname)"
date

source /workdir/wz452/miniconda/etc/profile.d/conda.sh
conda activate colabfold
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

colabfold_batch \\
    --num-recycle 3 --num-models 5 \\
    --model-type alphafold2_multimer_v3 \\
    "$FASTA" "$OUTDIR"

echo "Done: $NAME"
date
"""
    script_path = os.path.join(BASE_DIR, "run_ala_scan_array.sh")
    with open(script_path, "w") as fh:
        fh.write(script)

    print(f"Array script: {script_path}")
    print(f"Pending list: {pending_file}")

    if dry_run:
        print(f"Dry run — to submit: sbatch {script_path}")
    else:
        result = subprocess.run(["sbatch", script_path],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("sbatch error:", result.stderr.strip())
        print("\nAfter completion, run:")
        print("  python analyze_vpr_ala_scan.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry)
