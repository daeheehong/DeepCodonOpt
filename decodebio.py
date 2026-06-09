#!/usr/bin/env python3
"""Stage 3 (Method C) -- constraint-aware decoding, no retraining.

Take the Stage 1 baseline model and change only *how it decodes*: during generation we mask
out any synonymous codon that would create a constraint violation in the actual sequence,
and pick the highest-probability codon among the rest. Because the masking is applied to the
real discrete sequence (not a soft mean-field surrogate as in Method B / lossbio.py), it
cannot be "gamed" -- a codon is forbidden only if it genuinely introduces a violation.

Algorithm (model-guided iterative decoding, cf. Mask-Predict)
    1. Encode the protein as amino-acid query tokens ("M_UNK K_UNK ...") and forward once to
       get, per position, the model's distribution over that residue's synonymous codons.
    2. Greedily assign each position the highest-probability synonymous codon that does not
       violate a local constraint given its already-chosen neighbours.
    3. (Optional) re-encode the now-decoded sequence, forward again, and re-decode -- repeat
       until the sequence stops changing (bidirectional refinement).

Locally maskable constraints (enforced here)
    rare codons, homopolymer runs, restriction-enzyme sites, internal Shine-Dalgarno (AGGAGG).
    These were the dominant failures in the Stage 2 analysis (internal SD, enzymes,
    homopolymers). Long-range repeats / hairpins are NOT directly maskable in a single
    left-to-right pass (they need look-ahead / beam search) and are left to evaluation only.

Output
    sequences.csv with columns: seq_id, real_dna, unconstrained_dna, constrained_dna
    (unconstrained = plain argmax decode from the same model; constrained = Method C).
    Score it with:  python analyze.py --sequences <out_dir>/sequences.csv ...
"""

import argparse
import csv
import os

import codon_utils as cu

# --- restriction-enzyme recognition sites + internal Shine-Dalgarno (both strands) ----------
ENZYMES = {
    "BsaI": "GGTCTC", "BsmBI": "CGTCTC", "BbsI": "GAAGAC", "SapI": "GCTCTTC", "AarI": "CACCTGC",
    "EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT", "XhoI": "CTCGAG", "NdeI": "CATATG",
    "NotI": "GCGGCCGC", "XbaI": "TCTAGA",
}
SHINE_DALGARNO = "AGGAGG"
_COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}


def revcomp(s):
    return "".join(_COMP[b] for b in reversed(s))


def build_motifs(enzymes, include_sd):
    motifs = set()
    for e in enzymes:
        m = ENZYMES.get(e)
        if m:
            motifs.add(m)
            motifs.add(revcomp(m))          # restriction sites are double-stranded
    if include_sd:
        motifs.add(SHINE_DALGARNO)
    return motifs


# --- local constraint checks (pure string ops; unit-tested without torch) -------------------
def _homopolymer_bad(seg, off, H):
    """Does any of the 3 candidate nt at [off, off+3) sit in a run of >= H identical bases?"""
    for p in range(off, off + 3):
        ch = seg[p]
        run, l, r = 1, p - 1, p + 1
        while l >= 0 and seg[l] == ch:
            run += 1
            l -= 1
        while r < len(seg) and seg[r] == ch:
            run += 1
            r += 1
        if run >= H:
            return True
    return False


def _motif_overlaps(seg, motif, lo, hi):
    """Does any occurrence of motif in seg overlap the candidate region [lo, hi)?"""
    start = seg.find(motif)
    while start != -1:
        if start < hi and start + len(motif) > lo:
            return True
        start = seg.find(motif, start + 1)
    return False


def codon_ok(codons, i, cand, cfg):
    """Is placing codon `cand` at position i acceptable given current neighbours?"""
    usage = cfg["usage"]
    if usage is not None and usage.get(cand, 1.0) < cfg["rare_freq"]:
        return False

    pad = cfg["pad"]                                   # codons of context on each side
    a = max(0, i - pad)
    b = min(len(codons), i + pad + 1)
    seg = "".join(codons[a:i]) + cand + "".join(codons[i + 1:b])
    off = 3 * (i - a)                                  # offset of cand within seg

    if _homopolymer_bad(seg, off, cfg["H"]):
        return False
    for m in cfg["motifs"]:
        if _motif_overlaps(seg, m, off, off + 3):
            return False
    if cfg["gc_hi"] < 1.0:                              # optional local GC window
        w = cfg["gc_win"]
        g = seg[max(0, off - w):off + 3 + w]
        gc = (g.count("G") + g.count("C")) / max(len(g), 1)
        if gc > cfg["gc_hi"] or gc < cfg["gc_lo"]:
            return False
    return True


def count_violations(dna, cfg):
    """Quick local-violation flags for one finished sequence (for the built-in summary)."""
    v = {"rare": 0, "homopolymer": 0, "site": 0}
    if cfg["usage"] is not None:
        codons = [dna[i:i + 3] for i in range(0, len(dna) - 2, 3)]
        if any(cfg["usage"].get(c, 1.0) < cfg["rare_freq"] for c in codons):
            v["rare"] = 1
    H = cfg["H"]
    if any(dna[k] == dna[k - 1] for k in range(1, len(dna))):     # cheap pre-check
        run = 1
        for k in range(1, len(dna)):
            run = run + 1 if dna[k] == dna[k - 1] else 1
            if run >= H:
                v["homopolymer"] = 1
                break
    if any(m in dna for m in cfg["motifs"]):
        v["site"] = 1
    return v


# --- codon vocabulary tables ----------------------------------------------------------------
def build_aa_tables(tokenizer, device):
    """amino acid -> (LongTensor of its synonymous codon vocab indices, list of codon strings).
    The stop symbol is keyed as '_' (tokens '__taa' etc.)."""
    import torch
    vocab = tokenizer.get_vocab()
    cols, cods = {}, {}
    for tok, idx in vocab.items():
        parts = tok.split("_")
        codon = parts[-1].upper().replace("U", "T")
        if len(parts) <= 3 and len(codon) == 3 and all(c in "ACGT" for c in codon):
            aa = parts[0].upper() if parts[0] else "_"
            cols.setdefault(aa, []).append(idx)
            cods.setdefault(aa, []).append(codon)
    cols = {aa: torch.tensor(v, dtype=torch.long, device=device) for aa, v in cols.items()}
    return cols, cods


# --- model forward --------------------------------------------------------------------------
def forward_logits(tokens, organism_id, model, tokenizer, device):
    """Forward a list of token strings ('M_UNK' or 'M_ATG'); return (logits[L,V], residue_positions)."""
    import torch
    from CodonTransformer.CodonUtils import MAX_LEN
    tok = tokenizer(" ".join(tokens), return_attention_mask=True, return_token_type_ids=True,
                    truncation=True, max_length=MAX_LEN, return_tensors="pt")
    tok["token_type_ids"] = torch.full_like(tok["input_ids"], organism_id)
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        logits = model(**tok).logits[0]
    res_pos = (tok["input_ids"][0] >= 5).nonzero(as_tuple=True)[0]   # residue (non-special) tokens
    return logits, res_pos


def _pick(lp, allowed, temperature, top_p, gen):
    """Choose one candidate index from `allowed` by argmax (T<=0) or temperature/top-p sampling."""
    if not allowed:
        return None
    if not temperature or temperature <= 0:
        return max(allowed, key=lambda j: lp[j])
    import torch
    probs = torch.softmax(torch.tensor([lp[j] for j in allowed], dtype=torch.float) / temperature, dim=-1)
    if top_p and top_p < 1.0:                            # nucleus (top-p) filter
        sp, si = torch.sort(probs, descending=True)
        sp = torch.where((torch.cumsum(sp, dim=-1) - sp) < top_p, sp, torch.zeros_like(sp))
        if float(sp.sum()) <= 0:
            sp[0] = 1.0
        return allowed[int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen).item()])]
    return allowed[int(torch.multinomial(probs, 1, generator=gen).item())]


def decode_one(protein, organism_id, model, tokenizer, device, aa_cols, aa_cods, cfg,
               refine_iters, temperature=0.0, top_p=1.0, gen=None):
    """Return (unconstrained_dna, constrained_dna) for one protein, or (None, None) on mismatch.

    temperature > 0 samples codons (argmax otherwise) to add diversity / break long-range
    repeats. Constraint masking is still applied to the `constrained` output, so it stays
    diverse *and* free of local violations.
    """
    import torch.nn.functional as F

    logits, res_pos = forward_logits([f"{aa}_UNK" for aa in protein],
                                     organism_id, model, tokenizer, device)
    if len(res_pos) != len(protein):
        return None, None

    def cand_logp(logits, res_pos, p):
        aa = protein[p]
        lp = F.log_softmax(logits[res_pos[p], aa_cols[aa]], dim=-1)
        return aa_cods[aa], lp.detach().cpu().tolist()

    # unconstrained: one pass over the all-query forward
    uncon = []
    for p in range(len(protein)):
        cods, lp = cand_logp(logits, res_pos, p)
        uncon.append(cods[_pick(lp, list(range(len(cods))), temperature, top_p, gen)])

    # constrained: iterative; pick (sample/argmax) only among codons that pass the constraints
    codons = list(uncon)
    for it in range(max(1, refine_iters)):
        if it > 0:                                       # bidirectional refinement: re-forward on decoded seq
            logits, res_pos = forward_logits(
                [f"{protein[p]}_{codons[p]}" for p in range(len(protein))],
                organism_id, model, tokenizer, device)
            if len(res_pos) != len(protein):
                break
        changed = False
        for p in range(len(protein)):
            cods, lp = cand_logp(logits, res_pos, p)
            allowed = [j for j in range(len(cods)) if codon_ok(codons, p, cods[j], cfg)]
            j = _pick(lp, allowed, temperature, top_p, gen)
            if j is None:                                # no feasible codon -> keep model's top
                j = _pick(lp, list(range(len(cods))), 0.0, 1.0, gen)
            if cods[j] != codons[p]:
                codons[p] = cods[j]
                changed = True
        if not changed:
            break
    return "".join(uncon), "".join(codons)


# --- main -----------------------------------------------------------------------------------
def load_usage(path):
    usage = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = (r.get("codon") or "").strip().upper().replace("U", "T")
            fr = r.get("frequency")
            if c and fr not in (None, ""):
                usage[c] = float(fr)
    return usage


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 / Method C: constraint-aware decoding")
    p.add_argument("--checkpoint", required=True, help="Stage 1 baseline raw state_dict")
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--n", type=int, default=500, help="number of proteins (0 = all)")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default=None)
    p.add_argument("--organism", default=None, help="force organism (default: per-record id)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--refine_iters", type=int, default=3, help="max bidirectional decode passes")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy argmax; >0 samples codons (e.g. 0.8) to add diversity / break repeats")
    p.add_argument("--top_p", type=float, default=1.0, help="nucleus cutoff used with --temperature")
    p.add_argument("--codon_usage", default=None, help="codon usage CSV for the rare-codon rule")
    p.add_argument("--rare_freq", type=float, default=0.1)
    p.add_argument("--homopolymer_len", type=int, default=6)
    p.add_argument("--enzymes", nargs="+", default=list(ENZYMES))
    p.add_argument("--no_sd", dest="sd", action="store_false", help="do not forbid internal AGGAGG")
    p.add_argument("--gc_window", type=int, default=50)
    p.add_argument("--gc_lo", type=float, default=0.0, help="local GC lower bound (0 disables)")
    p.add_argument("--gc_hi", type=float, default=1.0, help="local GC upper bound (1 disables)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[setup] device={device} checkpoint={args.checkpoint}", flush=True)

    cfg = dict(
        usage=(load_usage(args.codon_usage) if args.codon_usage else None),
        rare_freq=args.rare_freq, H=args.homopolymer_len,
        motifs=build_motifs(args.enzymes, args.sd),
        pad=3, gc_win=args.gc_window, gc_lo=args.gc_lo, gc_hi=args.gc_hi,
    )
    print(f"[constraints] rare<{args.rare_freq} homopolymer>={args.homopolymer_len} "
          f"motifs={len(cfg['motifs'])} (SD={args.sd})  refine_iters={args.refine_iters}  "
          f"temperature={args.temperature} top_p={args.top_p}", flush=True)
    gen = torch.Generator().manual_seed(args.seed) if args.temperature and args.temperature > 0 else None

    data = cu.load_eval_dataset(args.data, args.n, args.seed)
    print(f"[data] {len(data)} sequences from {args.data}", flush=True)

    tokenizer = cu.get_tokenizer()
    model = cu.load_finetuned_model(args.checkpoint, device)
    aa_cols, aa_cods = build_aa_tables(tokenizer, device)
    from CodonTransformer.CodonUtils import MAX_LEN

    rows, skipped = [], 0
    for idx, row in enumerate(data):
        protein = row["protein"]
        if len(protein) > MAX_LEN - 2:
            skipped += 1
            continue
        organism = args.organism or row["organism"]
        uncon, con = decode_one(protein, organism, model, tokenizer, device,
                                aa_cols, aa_cods, cfg, args.refine_iters,
                                temperature=args.temperature, top_p=args.top_p, gen=gen)
        if uncon is None or len(uncon) != len(row["real_dna"]) or len(con) != len(row["real_dna"]):
            skipped += 1
            continue
        rows.append({"seq_id": len(rows), "real_dna": row["real_dna"],
                     "unconstrained_dna": uncon, "constrained_dna": con})
        if (idx + 1) % 50 == 0:
            print(f"  decoded {len(rows)} (seen {idx + 1}/{len(data)})", flush=True)

    cols = ["real_dna", "unconstrained_dna", "constrained_dna"]
    seq_path = os.path.join(args.out_dir, "sequences.csv")
    with open(seq_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id"] + cols)
        for r in rows:
            w.writerow([r["seq_id"]] + [r[c] for c in cols])
    print(f"[dump] {seq_path}  ({len(rows)} rows, skipped {skipped})", flush=True)

    # built-in local-violation summary (the constraints Method C enforces)
    print("\n[local violations: % of sequences hitting each]  (lower = better)")
    print(f'{"column":>20}{"rare":>10}{"homopolymer":>14}{"enzyme/SD":>12}')
    for c in cols:
        n = len(rows) or 1
        agg = {"rare": 0, "homopolymer": 0, "site": 0}
        for r in rows:
            v = count_violations(r[c], cfg)
            for k in agg:
                agg[k] += v[k]
        name = c[:-4] if c.endswith("_dna") else c
        print(f"{name:>20}{100*agg['rare']/n:>9.1f}%{100*agg['homopolymer']/n:>13.1f}%"
              f"{100*agg['site']/n:>11.1f}%")
    print(f"\nnext: python analyze.py --sequences {seq_path} "
          f"--codon_usage {args.codon_usage or 'ecoli_codon_usage_table.csv'} "
          f"[--idt_credentials creds.json] --out_dir {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
