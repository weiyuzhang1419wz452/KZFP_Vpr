#!/usr/bin/env python3
"""
run_mutant_kzfp_scan.py
=======================
Cross top-20 Vpr alanine-scan hotspot mutants against all KZFPs whose
KRAB domain shows significant WT interaction (ipTM >= 0.7, 39 KZFPs).

Creates:
  sequences/mutant_scan/Vpr_{MUT}_{GENE}-KRAB.fasta   (up to 780 files)
  predictions/mutant_scan/Vpr_{MUT}_{GENE}-KRAB/       (output dirs)
  run_mutant_scan_array.sh                              (SLURM array)

Usage:
  python run_mutant_kzfp_scan.py          # prep + submit
  python run_mutant_kzfp_scan.py --dry    # prep only
"""

import argparse, glob, json, os, re, subprocess
import numpy as np
import pandas as pd

BASE_DIR = "/local/workdir/wz452/Claude/KZFP_Vpr"
SEQ_DIR  = os.path.join(BASE_DIR, "sequences", "mutant_scan")
PRED_DIR = os.path.join(BASE_DIR, "predictions", "mutant_scan")
LOG_DIR  = os.path.join(BASE_DIR, "logs", "mutant_scan")

for d in (SEQ_DIR, PRED_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

VPR_WT = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"

SLURM_CPUS = 8
SLURM_MEM  = "32G"
SLURM_TIME = "12:00:00"
SLURM_CONC = 30

IPTM_THRESHOLD = 0.7   # only test KZFPs with strong WT interaction


def is_complete(d):
    return bool(glob.glob(os.path.join(d, "*scores*rank_001*.json")))


def make_mutant(seq, pos0, mut_aa):
    return seq[:pos0] + mut_aa + seq[pos0+1:]


def get_top_mutants(n=20):
    tsv = os.path.join(BASE_DIR, "analysis", "ala_scan", "ala_scan_scores.tsv")
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["complete"]].sort_values("delta_iptm").head(n)
    mutants = []
    for _, row in df.iterrows():
        pos0   = int(row["pos"]) - 1
        mut_aa = row["mut_aa"]
        label  = row["mutation"]   # e.g. "Y47A"
        mut_seq = make_mutant(VPR_WT, pos0, mut_aa)
        mutants.append({"label": label, "seq": mut_seq,
                        "delta_iptm": row["delta_iptm"]})
    return mutants


def get_significant_kzfps():
    """Return list of dicts with gene, wt_iptm, krab_seq, coords."""
    kzfps = []
    pred_base = os.path.join(BASE_DIR, "predictions")
    seq_base  = os.path.join(BASE_DIR, "sequences")

    for d in sorted(glob.glob(os.path.join(pred_base, "Vpr_*-KRAB"))):
        gene = os.path.basename(d).replace("Vpr_", "").replace("-KRAB", "")
        rank1 = sorted(glob.glob(os.path.join(d, "*scores*rank_001*.json")))
        if not rank1:
            continue
        try:
            data  = json.load(open(rank1[0]))
            iptm  = data.get("iptm", np.nan)
        except Exception:
            continue
        if np.isnan(iptm) or iptm < IPTM_THRESHOLD:
            continue

        fasta = os.path.join(seq_base, f"Vpr_{gene}-KRAB.fasta")
        if not os.path.exists(fasta):
            continue
        with open(fasta) as fh:
            lines = fh.read().strip().split("\n")
        header   = lines[0]
        seq_line = "".join(l for l in lines if not l.startswith(">"))
        krab_seq = seq_line.split(":")[1] if ":" in seq_line else seq_line
        m        = re.search(r"aa\d+-\d+", header)
        coords   = m.group(0) if m else "unk"

        kzfps.append({"gene": gene, "wt_iptm": iptm,
                      "krab_seq": krab_seq, "coords": coords})

    kzfps.sort(key=lambda x: -x["wt_iptm"])
    return kzfps


def main(dry_run=False):
    mutants = get_top_mutants(20)
    kzfps   = get_significant_kzfps()

    print(f"Top-20 Vpr mutants:")
    for m in mutants:
        print(f"  {m['label']:8s}  ΔipTM={m['delta_iptm']:+.2f}")
    print(f"\nKZFPs with WT ipTM >= {IPTM_THRESHOLD}: {len(kzfps)}")
    print(f"Total combinations: {len(mutants)} × {len(kzfps)} = {len(mutants)*len(kzfps)}\n")

    pending = []
    n_done  = 0

    for mut in mutants:
        for kzfp in kzfps:
            name    = f"Vpr_{mut['label']}_{kzfp['gene']}-KRAB"
            fasta   = os.path.join(SEQ_DIR, f"{name}.fasta")
            out_dir = os.path.join(PRED_DIR, name)

            if is_complete(out_dir):
                n_done += 1
                continue

            os.makedirs(out_dir, exist_ok=True)
            with open(fasta, "w") as fh:
                fh.write(f">HIV1_Vpr_{mut['label']}:{kzfp['gene']}_KRAB_{kzfp['coords']}\n"
                         f"{mut['seq']}:{kzfp['krab_seq']}\n")
            pending.append(fasta)

    print(f"Already complete: {n_done}  |  Pending: {len(pending)}")

    if not pending:
        print("All predictions already complete — run analyze_mutant_kzfp_scan.py")
        return

    pending_file = os.path.join(BASE_DIR, "pending_mutant_scan.txt")
    with open(pending_file, "w") as fh:
        fh.write("\n".join(pending) + "\n")

    n = len(pending)
    script = f"""#!/bin/bash
#SBATCH --job-name=VprMutScan
#SBATCH --array=1-{n}%{SLURM_CONC}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={LOG_DIR}/mut_%a.out
#SBATCH --error={LOG_DIR}/mut_%a.err

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
    script_path = os.path.join(BASE_DIR, "run_mutant_scan_array.sh")
    with open(script_path, "w") as fh:
        fh.write(script)

    print(f"Array script : {script_path}")
    print(f"Pending list : {pending_file}")

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
        print("  python analyze_mutant_kzfp_scan.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry)
