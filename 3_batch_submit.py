#!/usr/bin/env python3
"""
3_batch_submit.py
=================
Prepare FASTA files and submit ColabFold SLURM array jobs for large-scale
Vpr-KZFP domain analysis.

Domains:
  krab       -- Vpr + KRAB domain (one job per KZFP)
  znf        -- Vpr + ZNF array   (one job per KZFP, capped at 200 aa)
  krab_boxes -- Vpr + KRAB-A and Vpr + KRAB-B (two jobs per KZFP)
               Uses Pfam PF01352 to find the KRAB-A/B boundary.
               Also runs MAFFT to produce krab_b_aligned.fasta.

Usage:
  python 3_batch_submit.py --domain krab
  python 3_batch_submit.py --domain znf
  python 3_batch_submit.py --domain krab_boxes
  python 3_batch_submit.py --domain krab --no-submit   # prep only, no sbatch
"""

import argparse, glob, json, os, re, subprocess, time
import urllib.request, urllib.parse

BASE_DIR  = "/local/workdir/wz452/script/project/KZFP_Vpr"
SEQ_DIR   = os.path.join(BASE_DIR, "sequences")
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUT_DIR   = os.path.join(BASE_DIR, "analysis")

for d in (SEQ_DIR, PRED_DIR, LOG_DIR, CACHE_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

VPR_SEQ         = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
MAX_ZNF_LEN     = 200   # cap ZNF domain at 200 aa (~8 zinc fingers)
KRAB_A_FALLBACK = 40    # fallback KRAB-A length if Pfam unavailable

SLURM_CPUS        = 8
SLURM_MEM         = "32G"
SLURM_TIME        = "12:00:00"
SLURM_CONCURRENCY = 30

# ── KZFP gene list (from MS/GSEA proteomics) ──────────────────────────────────
KZFP_GENES = [
    "KRBOX4","LINC02156","PRDM7","PRDM9","RBAK","ZFP1","ZFP14","ZFP2","ZFP28",
    "ZFP30","ZFP37","ZFP57","ZFP69","ZFP69B","ZFP82","ZFP90","ZFP92","ZIK1",
    "ZIM2","ZIM3","ZKSCAN1","ZKSCAN2","ZKSCAN3","ZKSCAN4","ZKSCAN5","ZKSCAN7",
    "ZKSCAN8","ZNF10","ZNF100","ZNF101","ZNF107","ZNF112","ZNF114","ZNF117",
    "ZNF12","ZNF124","ZNF132","ZNF133","ZNF134","ZNF135","ZNF136","ZNF138",
    "ZNF14","ZNF140","ZNF141","ZNF154","ZNF155","ZNF157","ZNF160","ZNF169",
    "ZNF17","ZNF175","ZNF177","ZNF18","ZNF180","ZNF181","ZNF182","ZNF184",
    "ZNF189","ZNF19","ZNF195","ZNF197","ZNF2","ZNF20","ZNF202","ZNF205",
    "ZNF208","ZNF211","ZNF212","ZNF213","ZNF214","ZNF215","ZNF221","ZNF222",
    "ZNF223","ZNF224","ZNF225","ZNF226","ZNF227","ZNF229","ZNF23","ZNF230",
    "ZNF233","ZNF234","ZNF235","ZNF239","ZNF248","ZNF25","ZNF250","ZNF251",
    "ZNF253","ZNF254","ZNF256","ZNF257","ZNF26","ZNF263","ZNF264","ZNF266",
    "ZNF267","ZNF268","ZNF273","ZNF274","ZNF275","ZNF28","ZNF282","ZNF283",
    "ZNF284","ZNF285","ZNF286A","ZNF287","ZNF3","ZNF30","ZNF300","ZNF302",
    "ZNF304","ZNF311","ZNF316","ZNF317","ZNF320","ZNF324","ZNF324B","ZNF331",
    "ZNF333","ZNF334","ZNF337","ZNF33A","ZNF33B","ZNF34","ZNF343","ZNF345",
    "ZNF347","ZNF350","ZNF354A","ZNF354B","ZNF354C","ZNF382","ZNF383","ZNF394",
    "ZNF398","ZNF404","ZNF41","ZNF415","ZNF416","ZNF417","ZNF418","ZNF419",
    "ZNF420","ZNF425","ZNF426","ZNF429","ZNF43","ZNF430","ZNF431","ZNF432",
    "ZNF433","ZNF436","ZNF439","ZNF44","ZNF440","ZNF441","ZNF442","ZNF443",
    "ZNF445","ZNF446","ZNF45","ZNF454","ZNF460","ZNF461","ZNF468","ZNF470",
    "ZNF471","ZNF473","ZNF479","ZNF480","ZNF483","ZNF484","ZNF485","ZNF486",
    "ZNF487","ZNF490","ZNF491","ZNF492","ZNF493","ZNF500","ZNF506","ZNF510",
    "ZNF514","ZNF517","ZNF519","ZNF525","ZNF527","ZNF528","ZNF529","ZNF530",
    "ZNF534","ZNF540","ZNF543","ZNF544","ZNF546","ZNF547","ZNF548","ZNF549",
    "ZNF550","ZNF551","ZNF552","ZNF554","ZNF555","ZNF556","ZNF557","ZNF558",
    "ZNF559","ZNF560","ZNF561","ZNF562","ZNF563","ZNF564","ZNF565","ZNF566",
    "ZNF567","ZNF568","ZNF569","ZNF57","ZNF570","ZNF571","ZNF573","ZNF577",
    "ZNF578","ZNF582","ZNF583","ZNF584","ZNF585A","ZNF585B","ZNF586","ZNF587",
    "ZNF587B","ZNF589","ZNF595","ZNF596","ZNF597","ZNF599","ZNF600","ZNF605",
    "ZNF606","ZNF607","ZNF610","ZNF611","ZNF613","ZNF614","ZNF615","ZNF616",
    "ZNF619","ZNF620","ZNF621","ZNF624","ZNF625","ZNF626","ZNF627","ZNF630",
    "ZNF641","ZNF649","ZNF655","ZNF658","ZNF66","ZNF662","ZNF665","ZNF667",
    "ZNF669","ZNF670","ZNF671","ZNF674","ZNF675","ZNF676","ZNF677","ZNF678",
    "ZNF679","ZNF680","ZNF681","ZNF682","ZNF684","ZNF688","ZNF689","ZNF69",
    "ZNF695","ZNF699","ZNF7","ZNF700","ZNF701","ZNF705A","ZNF705B","ZNF705C",
    "ZNF705D","ZNF705E","ZNF705G","ZNF707","ZNF708","ZNF709","ZNF71","ZNF713",
    "ZNF714","ZNF716","ZNF717","ZNF718","ZNF720","ZNF721","ZNF723","ZNF724",
    "ZNF726","ZNF727","ZNF728","ZNF729","ZNF730","ZNF732","ZNF735","ZNF736",
    "ZNF737","ZNF738","ZNF74","ZNF746","ZNF747","ZNF749","ZNF75A","ZNF75D",
    "ZNF761","ZNF763","ZNF764","ZNF765","ZNF766","ZNF77","ZNF772","ZNF773",
    "ZNF776","ZNF777","ZNF778","ZNF780A","ZNF780B","ZNF781","ZNF782","ZNF783",
    "ZNF785","ZNF786","ZNF789","ZNF790","ZNF791","ZNF792","ZNF793","ZNF799",
    "ZNF8","ZNF805","ZNF806","ZNF807","ZNF808","ZNF81","ZNF813","ZNF814",
    "ZNF816","ZNF823","ZNF829","ZNF83","ZNF836","ZNF84","ZNF841","ZNF844",
    "ZNF845","ZNF846","ZNF85","ZNF850","ZNF852","ZNF860","ZNF875","ZNF878",
    "ZNF879","ZNF880","ZNF888","ZNF891","ZNF90","ZNF91","ZNF92","ZNF93",
    "ZNF98","ZNF99","ZSCAN18",
]


# ── UniProt helpers ────────────────────────────────────────────────────────────

def uniprot_search(gene, organism=9606):
    """Return best UniProt entry dict for a human gene, or None."""
    for query in [
        f"gene_exact:{gene} AND organism_id:{organism} AND reviewed:true",
        f"gene_exact:{gene} AND organism_id:{organism}",
    ]:
        url = (f"https://rest.uniprot.org/uniprotkb/search?"
               f"query={urllib.parse.quote(query)}"
               f"&format=json&fields=accession,sequence,ft_domain,ft_zn_fing&size=1")
        for attempt in range(3):
            try:
                data = json.loads(urllib.request.urlopen(url, timeout=20).read())
                results = data.get("results", [])
                if results:
                    return results[0]
            except Exception:
                if attempt < 2:
                    time.sleep(3)
    return None


def get_krab(entry):
    """Return (krab_seq, 'aa{s}-{e}') or (None, reason)."""
    seq = entry["sequence"]["value"]
    for f in entry.get("features", []):
        if f.get("type") == "Domain" and "KRAB" in f.get("description", "").upper():
            s = f["location"]["start"]["value"]
            e = f["location"]["end"]["value"]
            return seq[s-1:e], f"aa{s}-{e}"
    has_c2h2 = any(f.get("type") == "Zinc finger" and "C2H2" in f.get("description", "")
                   for f in entry.get("features", []))
    if has_c2h2:
        return seq[:90], "aa1-90_fallback"
    return None, "no_KRAB_no_C2H2"


def get_znf(entry):
    """Return (znf_seq, 'aa{s}-{e}') capped at MAX_ZNF_LEN aa, or (None, reason)."""
    seq = entry["sequence"]["value"]
    starts = [f["location"]["start"]["value"]
              for f in entry.get("features", []) if f.get("type") == "Zinc finger"]
    if not starts:
        return None, "no_znf_annotation"
    zs = min(starts) - 1
    ze = min(zs + MAX_ZNF_LEN, len(seq))
    return seq[zs:ze], f"aa{zs+1}-{ze}"


def pfam_krab_a_end(uniprot_acc):
    """Return end position (1-based, full protein) of Pfam PF01352. None if not found."""
    url = (f"https://www.ebi.ac.uk/interpro/api/entry/pfam/PF01352"
           f"/protein/UniProt/{uniprot_acc}/?format=json")
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        for prot in data.get("proteins", []):
            for loc in prot.get("entry_protein_locations", []):
                for frag in loc.get("fragments", []):
                    return int(frag["end"])
    except Exception:
        pass
    return None


def write_fasta(path, gene, domain_label, domain_seq, coord_desc):
    with open(path, "w") as fh:
        fh.write(f">HIV1_Vpr:{gene}_{domain_label}_{coord_desc}\n"
                 f"{VPR_SEQ}:{domain_seq}\n")


def is_complete(pred_dir):
    return bool(glob.glob(os.path.join(pred_dir, "*scores*rank_001*.json")))


# ── Domain preparation functions ──────────────────────────────────────────────

def prepare_krab():
    cache_path = os.path.join(CACHE_DIR, "uniprot_krab_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    pending = []
    n_done = n_no_seq = n_no_krab = 0

    for i, gene in enumerate(KZFP_GENES):
        fasta_path = os.path.join(SEQ_DIR, f"Vpr_{gene}-KRAB.fasta")
        pred_dir   = os.path.join(PRED_DIR, f"Vpr_{gene}-KRAB")

        if is_complete(pred_dir):
            n_done += 1
            continue

        if not os.path.exists(fasta_path):
            if gene not in cache:
                entry = uniprot_search(gene)
                time.sleep(0.3)
                if entry is None:
                    print(f"  [{i+1}/{len(KZFP_GENES)}] NOT FOUND    {gene}")
                    n_no_seq += 1
                    cache[gene] = None
                    json.dump(cache, open(cache_path, "w"), indent=2)
                    continue
                krab_seq, krab_desc = get_krab(entry)
                cache[gene] = {"acc": entry["primaryAccession"],
                               "seq": krab_seq, "desc": krab_desc}
                json.dump(cache, open(cache_path, "w"), indent=2)

            cached = cache.get(gene)
            if not cached or not cached.get("seq"):
                n_no_krab += 1
                continue

            os.makedirs(pred_dir, exist_ok=True)
            write_fasta(fasta_path, gene, "KRAB", cached["seq"], cached["desc"])
            print(f"  [{i+1}/{len(KZFP_GENES)}] FASTA written {gene}  {cached['desc']}")

        pending.append(fasta_path)

    print(f"\n  Already complete: {n_done}  |  No UniProt: {n_no_seq}  |  No KRAB: {n_no_krab}")
    return pending


def prepare_znf():
    cache_path = os.path.join(CACHE_DIR, "uniprot_znf_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    krab_fastas = sorted(f for f in glob.glob(os.path.join(SEQ_DIR, "Vpr_*-KRAB.fasta"))
                         if "KRAB_A" not in f and "KRAB_B" not in f)
    genes = [os.path.basename(f).replace("Vpr_", "").replace("-KRAB.fasta", "")
             for f in krab_fastas]
    print(f"  Source genes (from existing KRAB FASTAs): {len(genes)}")

    pending = []
    n_done = n_no_znf = 0

    for i, gene in enumerate(genes):
        fasta_path = os.path.join(SEQ_DIR, f"Vpr_{gene}-ZNF.fasta")
        pred_dir   = os.path.join(PRED_DIR, f"Vpr_{gene}-ZNF")

        if is_complete(pred_dir):
            n_done += 1
            continue

        if not os.path.exists(fasta_path):
            if gene not in cache:
                entry = uniprot_search(gene)
                time.sleep(0.2)
                if entry is None:
                    n_no_znf += 1
                    cache[gene] = None
                    continue
                znf_seq, znf_desc = get_znf(entry)
                cache[gene] = {"seq": znf_seq, "desc": znf_desc} if znf_seq else None
                if i % 20 == 0:
                    json.dump(cache, open(cache_path, "w"), indent=2)

            cached = cache.get(gene)
            if not cached or not cached.get("seq"):
                n_no_znf += 1
                continue
            write_fasta(fasta_path, gene, "ZNF", cached["seq"], cached["desc"])

        pending.append(fasta_path)

    json.dump(cache, open(cache_path, "w"), indent=2)
    print(f"\n  Already complete: {n_done}  |  No ZNF annotation: {n_no_znf}")
    return pending


def prepare_krab_boxes():
    cache_path = os.path.join(CACHE_DIR, "krab_box_boundaries.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    krab_fastas = sorted(f for f in glob.glob(os.path.join(SEQ_DIR, "Vpr_*-KRAB.fasta"))
                         if "KRAB_A" not in f and "KRAB_B" not in f)
    print(f"  KRAB FASTAs to split: {len(krab_fastas)}")

    # Parse sequences and coordinates from FASTAs
    genes_info = {}
    for fasta in krab_fastas:
        gene = os.path.basename(fasta).replace("Vpr_", "").replace("-KRAB.fasta", "")
        with open(fasta) as fh:
            content = fh.read().strip()
        seq_line = "".join(l for l in content.split("\n") if not l.startswith(">"))
        krab_seq = seq_line.split(":")[1] if ":" in seq_line else seq_line
        header   = next(l for l in content.split("\n") if l.startswith(">"))
        m = re.search(r"aa(\d+)-(\d+)", header)
        genes_info[gene] = dict(
            seq        = krab_seq,
            krab_start = int(m.group(1)) if m else None,
            krab_end   = int(m.group(2)) if m else None,
        )

    # Fetch UniProt IDs and Pfam boundaries for uncached genes
    missing = [g for g in genes_info
               if g not in cache
               or "uniprot" not in cache[g]
               or "kraba_end_prot" not in cache[g]]
    print(f"  Fetching UniProt/Pfam for {len(missing)} genes...")
    for i, gene in enumerate(missing):
        if gene not in cache:
            cache[gene] = {}
        if "uniprot" not in cache[gene]:
            entry = uniprot_search(gene)
            time.sleep(0.2)
            cache[gene]["uniprot"] = entry["primaryAccession"] if entry else None
        acc = cache[gene].get("uniprot")
        if acc and "kraba_end_prot" not in cache[gene]:
            cache[gene]["kraba_end_prot"] = pfam_krab_a_end(acc)
            time.sleep(0.15)
        if (i + 1) % 20 == 0:
            json.dump(cache, open(cache_path, "w"), indent=2)
            print(f"    {i+1}/{len(missing)}", flush=True)
    json.dump(cache, open(cache_path, "w"), indent=2)

    pfam_hits = sum(1 for g in genes_info if cache.get(g, {}).get("kraba_end_prot"))
    print(f"  Pfam-based splits: {pfam_hits}  |  Fallback (40 aa): {len(genes_info)-pfam_hits}")

    # Write KRAB-A / KRAB-B FASTAs
    pending = []
    krab_b_records = []
    n_skip = 0

    for gene, info in genes_info.items():
        krab_seq   = info["seq"]
        krab_start = info["krab_start"]
        krab_end   = info["krab_end"]
        n_krab     = len(krab_seq)

        kraba_end_prot = cache.get(gene, {}).get("kraba_end_prot")
        if kraba_end_prot and krab_start:
            split_idx = max(10, min(kraba_end_prot - krab_start + 1, n_krab - 5))
        else:
            split_idx = min(KRAB_A_FALLBACK, max(10, n_krab - 10))

        krab_a = krab_seq[:split_idx]
        krab_b = krab_seq[split_idx:]
        if len(krab_a) < 5 or len(krab_b) < 5:
            n_skip += 1
            continue

        for box, seq in [("A", krab_a), ("B", krab_b)]:
            name       = f"Vpr_{gene}-KRAB_{box}"
            fasta_path = os.path.join(SEQ_DIR, f"{name}.fasta")
            if box == "A":
                aa_range = f"aa{krab_start}-{krab_start + split_idx - 1}"
            else:
                aa_range = f"aa{krab_start + split_idx}-{krab_end or krab_start + n_krab - 1}"
            with open(fasta_path, "w") as fh:
                fh.write(f">HIV1_Vpr:{gene}_KRAB_{box}_{aa_range}\n"
                         f"{VPR_SEQ}:{seq}\n")
            pred_dir = os.path.join(PRED_DIR, name)
            if not is_complete(pred_dir):
                pending.append(fasta_path)

        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", gene)
        krab_b_records.append((safe_name, krab_b))

    print(f"  Skipped (too short): {n_skip}")

    # Run MAFFT on KRAB-B sequences
    raw_b_fa = os.path.join(OUT_DIR, "krab_b_sequences.fasta")
    aln_b_fa = os.path.join(OUT_DIR, "krab_b_aligned.fasta")
    with open(raw_b_fa, "w") as fh:
        for name, seq in krab_b_records:
            fh.write(f">{name}\n{seq}\n")
    print(f"\n  Running MAFFT on {len(krab_b_records)} KRAB-B sequences...")
    with open(aln_b_fa, "w") as fh:
        result = subprocess.run(["mafft", "--auto", "--quiet", raw_b_fa],
                                stdout=fh, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(f"  KRAB-B alignment: {aln_b_fa}")
    else:
        print("  MAFFT error:", result.stderr.decode())

    return pending


# ── SLURM array writer / submitter ────────────────────────────────────────────

def submit_array(pending, domain, do_submit):
    if not pending:
        print("Nothing to submit — all predictions already complete.")
        return

    pending_file = os.path.join(BASE_DIR, f"pending_{domain}.txt")
    with open(pending_file, "w") as fh:
        fh.write("\n".join(pending) + "\n")
    print(f"\nPending list ({len(pending)} jobs): {pending_file}")

    n = len(pending)
    script_content = f"""#!/bin/bash
#SBATCH --job-name=Vpr_{domain}
#SBATCH --array=1-{n}%{SLURM_CONCURRENCY}
#SBATCH --cpus-per-task={SLURM_CPUS}
#SBATCH --mem={SLURM_MEM}
#SBATCH --time={SLURM_TIME}
#SBATCH --output={LOG_DIR}/{domain}_%a.out
#SBATCH --error={LOG_DIR}/{domain}_%a.err

FASTA=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {pending_file})
[ -z "$FASTA" ] && exit 0

NAME=$(basename "$FASTA" .fasta)
OUT_DIR="{PRED_DIR}/$NAME"
mkdir -p "$OUT_DIR"

echo "Running $NAME on $(hostname)"
date

source /workdir/wz452/miniconda/etc/profile.d/conda.sh
conda activate colabfold
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

colabfold_batch \\
    --num-recycle 3 --num-models 5 \\
    --model-type alphafold2_multimer_v3 \\
    "$FASTA" "$OUT_DIR"

echo "Done: $NAME"
date
"""

    script_path = os.path.join(BASE_DIR, f"run_{domain}_array.sh")
    with open(script_path, "w") as fh:
        fh.write(script_content)
    print(f"Array script: {script_path}")

    if do_submit:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        print(result.stdout.strip() if result.returncode == 0
              else f"sbatch error: {result.stderr.strip()}")
    else:
        print(f"Dry run -- to submit: sbatch {script_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

DOMAIN_FUNCS = {
    "krab":       prepare_krab,
    "znf":        prepare_znf,
    "krab_boxes": prepare_krab_boxes,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=list(DOMAIN_FUNCS), required=True)
    parser.add_argument("--no-submit", action="store_true",
                        help="Prepare FASTAs only, do not sbatch")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Domain: {args.domain}")
    print("=" * 60)

    pending = DOMAIN_FUNCS[args.domain]()
    submit_array(pending, args.domain, do_submit=not args.no_submit)


if __name__ == "__main__":
    main()
