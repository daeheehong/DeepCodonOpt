#!/usr/bin/env python3
"""Build a CodonTransformer fine-tuning dataset from an NCBI genome.

Pipeline: parse native CDS -> strict QC -> (protein, DNA) pairs -> codon-token JSONL,
then split into train/val/test and (optionally) make nested size subsets of the train
set for the Stage 1 data-scaling experiment.

Inputs (place one of these under ``--input-dir``; ``.gz`` is fine, auto-detected)
    *_genomic.gbff(.gz)                                  (recommended: one file)
    *_cds_from_genomic.fna(.gz) + *_protein.faa(.gz)

Outputs (under ``--out-dir``)
    <prefix>_cds.csv     human-readable protein-DNA pair table
    codons.jsonl         full set, {"idx", "codons", "organism"}
    train/val/test.jsonl split (val/test held out before subsetting)
    sizes/train_<n>.jsonl  nested subsets (+ train_all.jsonl symlink) if --sizes given

Reference genome used in this project: E. coli BL21(DE3), RefSeq GCF_000022665.1.
``organism`` defaults to 51 = "Escherichia coli general" so the data is compatible with
the public CodonTransformer organism vocabulary (the real strain name is kept in the CSV).
"""

import argparse
import csv
import glob
import gzip
import json
import os
import random
import re
import sys
from collections import Counter

# Constants shared with CodonTransformer.CodonUtils
START_CODONS = {"ATG", "TTG", "CTG", "GTG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
STOP_SYMBOL = "_"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_sense(dna):
    """Translate a sense region (stop codon excluded) with the standard table; no start re-coding."""
    return "".join(_CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, len(dna), 3))


def get_merged_seq(protein, dna, sep="_"):
    """Reproduce CodonTransformer.CodonData.get_merged_seq.
    ``protein`` ends with the stop symbol '_' and ``dna`` includes the stop codon.
    e.g. ('MAV_', 'ATGGCTGTGTAA') -> 'M_ATG A_GCT V_GTG __TAA'."""
    return " ".join(f"{protein[i]}{sep}{dna[i * 3:i * 3 + 3]}" for i in range(len(protein)))


# ---- input parsing ---------------------------------------------------------
def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_fasta(path):
    header, seq = None, []
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line.strip())
    if header is not None:
        yield header, "".join(seq)


_TAG_RE = re.compile(r"\[(\w+)=([^\]]*)\]")


def _translate_with_met_start(dna):
    """Fallback when no protein FASTA: translate DNA, drop stop, force a Met start codon to M."""
    dna = dna.upper()
    aa = translate_sense(dna[:-3]) if dna[-3:] in STOP_CODONS else translate_sense(dna)
    if aa and dna[:3] in START_CODONS:
        aa = "M" + aa[1:]
    return aa


def load_from_fna_faa(fna_path, faa_path=None):
    """cds_from_genomic.fna (+ protein.faa) -> record dicts. Official proteins come from the
    .faa matched by protein_id; without it, DNA translation is used as a fallback."""
    faa = {}
    if faa_path:
        for h, s in read_fasta(faa_path):
            faa[h.split()[0]] = s.upper().replace("*", "")
    recs = []
    for header, dna in read_fasta(fna_path):
        tags = dict(_TAG_RE.findall(header))
        if "protein_id" not in tags:                 # pseudogene / non-CDS
            continue
        if tags.get("pseudo", "").lower() == "true":
            continue
        pid = tags["protein_id"]
        prot = faa.get(pid) or _translate_with_met_start(dna)
        recs.append({
            "locus_tag": tags.get("locus_tag", ""), "gene": tags.get("gene", ""),
            "protein_id": pid, "product": tags.get("protein", ""),
            "dna": dna.upper(), "protein": prot,
        })
    return recs


def load_from_gbff(gbff_path):
    """genomic.gbff -> record dicts using each CDS feature's /translation and extracted DNA.
    Requires biopython."""
    try:
        from Bio import SeqIO
    except ImportError:
        sys.exit("[ERROR] parsing .gbff needs biopython: pip install biopython")
    recs = []
    with _open(gbff_path) as fh:
        for record in SeqIO.parse(fh, "genbank"):
            for feat in record.features:
                if feat.type != "CDS":
                    continue
                q = feat.qualifiers
                if "translation" not in q or "pseudo" in q or "pseudogene" in q:
                    continue
                recs.append({
                    "locus_tag": q.get("locus_tag", [""])[0], "gene": q.get("gene", [""])[0],
                    "protein_id": q.get("protein_id", [""])[0], "product": q.get("product", [""])[0],
                    "dna": str(feat.extract(record.seq)).upper(),
                    "protein": q["translation"][0].upper().replace("*", ""),
                })
    return recs


# ---- QC --------------------------------------------------------------------
def validate(rec, drop_nonstandard_aa=True):
    """Strict QC. Returns (ok, reason, codons_str_or_None)."""
    dna = rec["dna"].upper()
    prot = rec["protein"].upper().rstrip("*")
    if set(dna) - set("ACGT"):
        return False, "ambiguous_nt", None
    if len(dna) % 3 != 0:
        return False, "len_not_mult3", None
    if dna[:3] not in START_CODONS:
        return False, "bad_start", None
    if dna[-3:] not in STOP_CODONS:
        return False, "no_stop_end", None
    aa_body = translate_sense(dna[:-3])
    if "*" in aa_body:
        return False, "internal_stop", None
    if len(prot) != len(dna) // 3 - 1:
        return False, "len_mismatch", None
    # start residue (M vs V/L/I) may differ; every other position must match the official protein
    if any(a != b for a, b in zip(prot[1:], aa_body[1:])):
        return False, "seq_mismatch", None
    if drop_nonstandard_aa and (set(prot) - STANDARD_AA):
        return False, "nonstandard_aa", None
    return True, "ok", get_merged_seq(prot + STOP_SYMBOL, dna)


def gc_percent(dna):
    return 100.0 * (dna.count("G") + dna.count("C")) / len(dna) if dna else 0.0


def autodetect_inputs(input_dir):
    def first(pat):
        hits = sorted(glob.glob(os.path.join(input_dir, pat)))
        return hits[0] if hits else None
    gbff = first("*_genomic.gbff*") or first("*.gbff*")
    fna = first("*_cds_from_genomic.fna*") or first("*cds*.fna*")
    faa = first("*_protein.faa*") or first("*protein*.faa*")
    return gbff, fna, faa


# ---- split + nested subsets ------------------------------------------------
def split_train_val_test(records, val_ratio, test_ratio, seed):
    """Shuffle once, then carve test / val / train. The same seed keeps splits reproducible."""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]
    pick = lambda idx: [records[i] for i in idx]
    return pick(train_idx), pick(val_idx), pick(test_idx)


def write_jsonl(records, path, organism_id):
    with open(path, "w") as f:
        for i, rec in enumerate(records):
            f.write(json.dumps({"idx": i, "codons": rec["codons"], "organism": organism_id}) + "\n")


def make_nested_subsets(train_records, out_dir, sizes, seed):
    """Nested train subsets: shuffle once and take prefixes, so a smaller set is a subset of a
    larger one (keeps the data-scaling curve from picking up per-size sampling noise)."""
    os.makedirs(out_dir, exist_ok=True)
    n_total = len(train_records)
    order = list(range(n_total))
    random.Random(seed).shuffle(order)
    for n in sorted(set(sizes)):
        if n >= n_total:
            print(f"  skip n={n}: >= train size ({n_total}); use train_all")
            continue
        idx = sorted(order[:n])
        with open(os.path.join(out_dir, f"train_{n}.jsonl"), "w") as f:
            for i in idx:
                f.write(json.dumps({"idx": i, "codons": train_records[i]["codons"],
                                    "organism": train_records[i]["_org"]}) + "\n")
        print(f"  sizes/train_{n}.jsonl ({n})")
    all_path = os.path.join(out_dir, "train_all.jsonl")
    if os.path.islink(all_path) or os.path.exists(all_path):
        os.remove(all_path)
    os.symlink(os.path.abspath(os.path.join(out_dir, "..", "train.jsonl")), all_path)
    print(f"  sizes/train_all.jsonl -> ../train.jsonl ({n_total})")


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NCBI genome -> CodonTransformer dataset (+ split, subsets)")
    ap.add_argument("--input-dir", default="data/ncbi", help="folder holding the NCBI FASTA/GBFF")
    ap.add_argument("--gbff"), ap.add_argument("--fna"), ap.add_argument("--faa")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--prefix", default="bl21de3", help="prefix for the CSV pair table")
    ap.add_argument("--organism-id", type=int, default=51, help="JSONL organism id (51 = E. coli general)")
    ap.add_argument("--organism-name", default="Escherichia coli BL21(DE3)", help="strain name kept in CSV")
    ap.add_argument("--max-codons", type=int, default=0, help="cap protein length (codons incl. stop); 0 = no cap")
    ap.add_argument("--keep-duplicates", action="store_true", help="do not drop identical DNA")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--sizes", type=int, nargs="*", default=[],
                    help="nested train subset sizes for the data-scaling experiment (e.g. 500 1000 2000)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    gbff, fna, faa = args.gbff, args.fna, args.faa
    if not (gbff or fna):
        gbff, fna, faa = autodetect_inputs(args.input_dir)
    if gbff:
        print(f"[input] GBFF: {gbff}")
        recs, source = load_from_gbff(gbff), os.path.basename(gbff)
    elif fna:
        print(f"[input] CDS FASTA: {fna}\n[input] protein FASTA: {faa or '(none -> DNA translation)'}")
        recs, source = load_from_fna_faa(fna, faa), os.path.basename(fna)
    else:
        sys.exit(f"[ERROR] no input found in {args.input_dir} "
                 "(need *_genomic.gbff or *_cds_from_genomic.fna [+ *_protein.faa])")
    print(f"[parse] raw CDS records: {len(recs)}")

    # QC + dedup
    reasons, kept, seen = Counter(), [], set()
    for rec in recs:
        ok, reason, codons = validate(rec)
        if not ok:
            reasons[reason] += 1
            continue
        if args.max_codons and (len(rec["protein"]) + 1) > args.max_codons:
            reasons["too_long"] += 1
            continue
        if not args.keep_duplicates and rec["dna"] in seen:
            reasons["duplicate_dna"] += 1
            continue
        seen.add(rec["dna"])
        rec["codons"] = codons
        rec["_org"] = args.organism_id
        kept.append(rec)
    print(f"[QC] kept {len(kept)} / dropped {sum(reasons.values())}")
    for r, c in reasons.most_common():
        print(f"      - {r:16s}: {c}")
    if not kept:
        sys.exit("[ERROR] no records passed QC; check the input files.")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) human-readable pair table
    csv_path = os.path.join(args.out_dir, f"{args.prefix}_cds.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "locus_tag", "gene", "protein_id", "product", "organism",
                    "organism_id", "n_aa", "n_codons", "dna_len", "gc", "protein", "dna"])
        for i, rec in enumerate(kept):
            n_aa = len(rec["protein"])
            w.writerow([i, rec["locus_tag"], rec["gene"], rec["protein_id"], rec["product"],
                        args.organism_name, args.organism_id, n_aa, n_aa + 1, len(rec["dna"]),
                        f"{gc_percent(rec['dna']):.2f}", rec["protein"], rec["dna"]])

    # 2) full JSONL + train/val/test split
    write_jsonl(kept, os.path.join(args.out_dir, "codons.jsonl"), args.organism_id)
    train, val, test = split_train_val_test(kept, args.val_ratio, args.test_ratio, args.seed)
    write_jsonl(train, os.path.join(args.out_dir, "train.jsonl"), args.organism_id)
    write_jsonl(val, os.path.join(args.out_dir, "val.jsonl"), args.organism_id)
    write_jsonl(test, os.path.join(args.out_dir, "test.jsonl"), args.organism_id)

    # 3) optional nested size subsets (train only)
    if args.sizes:
        make_nested_subsets(train, os.path.join(args.out_dir, "sizes"), args.sizes, args.seed)

    lens = sorted(len(r["protein"]) + 1 for r in kept)
    n = len(lens)
    print("\n===== done =====")
    print(f"  source     : {source}")
    print(f"  organism   : {args.organism_name} (id={args.organism_id})")
    print(f"  records    : {n}  (train {len(train)} / val {len(val)} / test {len(test)})")
    print(f"  codon len  : min {lens[0]}  median {lens[n // 2]}  p95 {lens[int(n * 0.95)]}  max {lens[-1]}")
    print(f"  -> {csv_path}")
    print(f"  -> {os.path.join(args.out_dir, 'train.jsonl')} / val.jsonl / test.jsonl")


if __name__ == "__main__":
    main()
