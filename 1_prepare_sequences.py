#!/usr/bin/env python3
"""
Prepare FASTA sequences for AlphaFold Multimer analysis of Vpr-KZFP interactions.

Fetches sequences from UniProt by gene name (reviewed, human),
extracts domain boundaries (KRAB, ZNF), and creates FASTA files for all
Vpr + KZFP domain combinations.

Usage: python 1_prepare_sequences.py
"""

import requests
import os
import json
import time

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "sequences"
os.makedirs(OUT_DIR, exist_ok=True)

# ── HIV-1 Vpr (HXB2, stop codon stripped) ────────────────────────────────────
VPR_SEQ  = "MEQAPEDQGPQREPYNEWTLELLEELKSEAVRHFPRIWLHNLGQHIYETYGDTWAGVEAIIRILQQLLFIHFRIGCRHSRIGVTRQRRARNGASRS"
VPR_NAME = "HIV1_Vpr"

# ── Proteins to analyze ───────────────────────────────────────────────────────
# Gene name → role
KZFP_GENES = {
    "ZNF93":  "KZFP restricting endogenous retroviruses",
    "ZNF91":  "KZFP controlling retroelement expression",
    "ZNF10":  "Prototypical KZFP",
    "ZNF274": "Well-studied KZFP (H3K9me3 at ZNF loci)",
}
CONTROL_GENES = {
    "TRIM28": "KRAB co-repressor KAP1 (KRAB interactor, positive control)",
    "DCAF1":  "VprBP — Vpr substrate receptor in CRL4 complex (indirect control)",
}

# ── UniProt REST API helpers ──────────────────────────────────────────────────
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FETCH  = "https://rest.uniprot.org/uniprotkb"

def search_uniprot(gene: str, taxon: int = 9606) -> dict | None:
    """
    Search for reviewed (Swiss-Prot) human entry by gene name.
    Returns the best match entry dict or None.
    """
    params = {
        "query": f'gene_exact:{gene} AND organism_id:{taxon} AND reviewed:true',
        "format": "json",
        "size": 5,
        "fields": "accession,gene_names,sequence,ft_domain,ft_region,ft_zn_fing,protein_name"
    }
    for attempt in range(3):
        try:
            r = requests.get(UNIPROT_SEARCH, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if results:
                return results[0]   # top reviewed hit
            return None
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)
    return None

def fetch_uniprot_by_id(uid: str) -> dict:
    url = f"{UNIPROT_FETCH}/{uid}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Retry {attempt+1} for {uid}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {uid}")

def get_sequence(entry: dict) -> str:
    return entry["sequence"]["value"]

def get_domain_regions(entry: dict) -> dict:
    """
    Extract domain boundaries from UniProt feature annotations.
    Returns {description: (start, end)} in 1-based coords.
    """
    domains = {}
    for feat in entry.get("features", []):
        ftype = feat.get("type", "")
        if ftype in ("Domain", "Zinc finger", "Region", "Repeat"):
            desc  = feat.get("description", "")
            loc   = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end   = loc.get("end",   {}).get("value")
            if start and end:
                key = f"{ftype}:{desc}"
                domains[key] = (int(start), int(end))
    return domains

def extract_krab(seq: str, domains: dict) -> tuple[str, str]:
    """Return (krab_seq, description). Falls back to first 90 aa."""
    for key, (s, e) in domains.items():
        if "KRAB" in key.upper():
            return seq[s-1:e], f"aa{s}-{e}"
    print("    KRAB not annotated; using aa1-90 as fallback")
    return seq[:90], "aa1-90_fallback"

def extract_znf_array(seq: str, domains: dict) -> tuple[str, str]:
    """Return (znf_seq, description) spanning all C2H2 zinc finger repeats."""
    zf_pos = [(s, e) for key, (s, e) in domains.items()
              if "Zinc finger" in key or "C2H2" in key.upper()]
    if zf_pos:
        s_min = min(s for s, e in zf_pos)
        e_max = max(e for s, e in zf_pos)
        return seq[s_min-1:e_max], f"aa{s_min}-{e_max}"
    print("    ZNF array not annotated; using aa91-end as fallback")
    return seq[90:], "aa91-end_fallback"

def extract_wd40(seq: str, domains: dict, gene_len: int) -> tuple[str, str]:
    """For DCAF1: extract WD40 / beta-propeller region.
    DCAF1/VprBP is ~1507 aa; the C-terminal WD40 domain that binds Vpr
    spans approximately residues 1042-1393.
    """
    wd_pos = [(s, e) for key, (s, e) in domains.items() if "WD" in key.upper()]
    if wd_pos:
        s_min = min(s for s, e in wd_pos)
        e_max = max(e for s, e in wd_pos)
        return seq[s_min-1:e_max], f"aa{s_min}-{e_max}"
    # Literature-based fallback for DCAF1
    if gene_len > 1000:
        s, e = 1042, min(1393, gene_len)
        print(f"    WD40 not annotated; using literature-based aa{s}-{e}")
        return seq[s-1:e], f"aa{s}-{e}_fallback"
    return "", "empty"

def write_fasta(filepath: str, records: list[tuple[str, str]]):
    """ColabFold multi-chain FASTA (chains separated by ':')."""
    header   = ":".join(name for name, seq in records)
    seq_line = ":".join(seq  for name, seq in records)
    with open(filepath, "w") as f:
        f.write(f">{header}\n{seq_line}\n")
    chain_lens = " + ".join(f"{len(s)}aa" for _, s in records)
    print(f"    Written: {os.path.basename(filepath)}  ({chain_lens})")

# ── Main ──────────────────────────────────────────────────────────────────────

def process_kzfp(gene: str) -> dict | None:
    """Fetch KZFP entry, extract domains, write FASTA files. Returns info dict."""
    print(f"\n[{gene}]")
    entry = search_uniprot(gene)
    if entry is None:
        print("  Not found in UniProt — skipping")
        return None

    uid  = entry["primaryAccession"]
    seq  = get_sequence(entry)
    doms = get_domain_regions(entry)
    print(f"  UniProt: {uid}  length: {len(seq)} aa")
    for key, (s, e) in sorted(doms.items()):
        print(f"    {key}: {s}-{e}")

    krab_seq, krab_desc = extract_krab(seq, doms)
    znf_seq,  znf_desc  = extract_znf_array(seq, doms)
    print(f"  KRAB domain: {krab_desc} ({len(krab_seq)} aa)")
    print(f"  ZNF array:   {znf_desc}  ({len(znf_seq)} aa)")

    write_fasta(f"{OUT_DIR}/Vpr_{gene}-full.fasta",
                [(VPR_NAME, VPR_SEQ), (f"{gene}_full", seq)])
    write_fasta(f"{OUT_DIR}/Vpr_{gene}-KRAB.fasta",
                [(VPR_NAME, VPR_SEQ), (f"{gene}_KRAB_{krab_desc}", krab_seq)])
    if len(znf_seq) > 0:
        write_fasta(f"{OUT_DIR}/Vpr_{gene}-ZNF.fasta",
                    [(VPR_NAME, VPR_SEQ), (f"{gene}_ZNF_{znf_desc}", znf_seq)])
    else:
        print(f"  ZNF sequence empty — skipping Vpr_{gene}-ZNF.fasta")

    return {
        "gene": gene, "uniprot": uid, "length": len(seq),
        "krab": krab_desc, "znf": znf_desc,
        "krab_seq": krab_seq,
    }

def process_trim28() -> dict | None:
    print("\n[TRIM28]  (KAP1 — KRAB co-repressor; positive control)")
    entry = search_uniprot("TRIM28")
    if entry is None:
        return None
    uid = entry["primaryAccession"]
    seq = get_sequence(entry)
    doms = get_domain_regions(entry)
    print(f"  UniProt: {uid}  length: {len(seq)} aa")
    # TRIM28 interacts with KRAB through its PHD-Bromo module (C-terminal) and
    # NuRD domain. Full protein test.
    write_fasta(f"{OUT_DIR}/Vpr_TRIM28-full.fasta",
                [(VPR_NAME, VPR_SEQ), ("TRIM28_full", seq)])
    return {"gene": "TRIM28", "uniprot": uid, "length": len(seq)}

def process_dcaf1() -> dict | None:
    print("\n[DCAF1]  (VprBP — substrate receptor for Vpr)")
    entry = search_uniprot("DCAF1")
    if entry is None:
        # Fallback: try VPRBP gene synonym
        print("  Trying VPRBP synonym...")
        entry = search_uniprot("VPRBP")
    if entry is None:
        print("  Not found — skipping DCAF1")
        return None
    uid = entry["primaryAccession"]
    seq = get_sequence(entry)
    doms = get_domain_regions(entry)
    print(f"  UniProt: {uid}  length: {len(seq)} aa")
    for key, (s, e) in sorted(doms.items()):
        print(f"    {key}: {s}-{e}")

    wd_seq, wd_desc = extract_wd40(seq, doms, len(seq))
    if not wd_seq:
        print("  Cannot extract WD40 region — skipping DCAF1 FASTA files")
        return None
    print(f"  WD40 region: {wd_desc} ({len(wd_seq)} aa)")

    write_fasta(f"{OUT_DIR}/Vpr_DCAF1-WD40.fasta",
                [(VPR_NAME, VPR_SEQ), (f"DCAF1_WD40_{wd_desc}", wd_seq)])

    return {"gene": "DCAF1", "uniprot": uid, "length": len(seq),
            "wd40": wd_desc, "wd40_seq": wd_seq}

def main():
    print("=" * 60)
    print("Preparing sequences for Vpr-KZFP AlphaFold analysis")
    print("=" * 60)

    kzfp_info = {}
    dcaf1_info = None

    # ── KZFPs ─────────────────────────────────────────────────────────────────
    for gene in KZFP_GENES:
        info = process_kzfp(gene)
        if info:
            kzfp_info[gene] = info

    # ── Controls ──────────────────────────────────────────────────────────────
    process_trim28()
    dcaf1_info = process_dcaf1()

    # ── DCAF1_WD40 + KZFP_KRAB pairs (indirect mechanism) ────────────────────
    if dcaf1_info and dcaf1_info.get("wd40_seq"):
        print("\n[DCAF1_WD40 + KZFP_KRAB pairs  (indirect recruitment test)]")
        for gene, info in kzfp_info.items():
            if info.get("krab_seq"):
                krab_desc = info["krab"]
                write_fasta(f"{OUT_DIR}/DCAF1-WD40_{gene}-KRAB.fasta",
                            [(f"DCAF1_WD40_{dcaf1_info['wd40']}",
                              dcaf1_info["wd40_seq"]),
                             (f"{gene}_KRAB_{krab_desc}",
                              info["krab_seq"])])

    # ── Summary ───────────────────────────────────────────────────────────────
    all_fastas = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".fasta"))
    print("\n" + "=" * 60)
    print(f"Total FASTA files ready: {len(all_fastas)}")
    for fn in all_fastas:
        path = f"{OUT_DIR}/{fn}"
        with open(path) as f:
            lines = f.readlines()
        seq_line = [l.strip() for l in lines if not l.startswith(">")][0]
        chains   = seq_line.split(":")
        lens_str = " + ".join(f"{len(c)}aa" for c in chains)
        print(f"  {fn:<45} [{lens_str}]")

    # Save summary JSON (without sequences to keep it small)
    summary = {g: {k: v for k, v in d.items() if k not in ("krab_seq", "wd40_seq")}
               for g, d in {**kzfp_info, **({"DCAF1": dcaf1_info} if dcaf1_info else {})}.items()}
    with open(f"{OUT_DIR}/sequences_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {OUT_DIR}/sequences_summary.json")
    print("\nNext step: bash 2_setup_colabfold.sh   (install ColabFold)")
    print("Then:      bash 3_run_predictions.sh   (submit jobs)")

if __name__ == "__main__":
    main()
