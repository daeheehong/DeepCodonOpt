#!/usr/bin/env python3
"""Stage 2 -- biological-rule violation analysis.

For the same set of proteins we generate DNA with the baseline (E. coli fine-tuned) model
and, for comparison, the stock pretrained CodonTransformer, then measure how much each
violates manufacturing / biological rules against the natural ("real") sequence:

  * IDT SciTools Plus complexity   -- the industry synthesis-difficulty score (needs API
                                      credentials); the headline manufacturability metric.
  * DnaChisel constraint set       -- per-constraint pass/fail (GC, homopolymers, repeats,
                                      k-mer uniqueness, hairpins, internal Shine-Dalgarno,
                                      restriction sites, rare codons); local, no API.
  * Continuous metrics             -- GC%, CAI, rare-codon CFD%, 5' MFE.

Outputs (under --out_dir)
    sequences.csv          seq_id, real_dna, baseline_dna, pretrained_dna
    violations_summary.csv per-constraint violation rate for each column
    violations.png         grouped bar chart of the above
    metrics_summary.csv    mean GC/CAI/CFD/MFE per column
    idt_complexity.csv     per-sequence IDT score per column        (if --idt_credentials)
    idt_issue_breakdown.csv % of sequences hitting each IDT issue   (if --idt_credentials)
    idt_complexity.png     complexity histogram per column          (if --idt_credentials)

Example
    python analyze.py --baseline_checkpoint out/size_all/best.pt \
        --data data/test.jsonl --n 500 --codon_usage ecoli_codon_usage_table.csv \
        --idt_credentials credentials.json --out_dir results
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import codon_utils as cu

COLUMNS = ["real", "baseline", "pretrained"]      # comparison columns (pretrained optional)
COLORS = {"real": "#4C72B0", "baseline": "#C44E52", "pretrained": "#55A868",
          "unconstrained": "#C44E52", "constrained": "#8172B3"}


def _ref_target(cols):
    """Reference column (natural sequence) and the column of interest, for sorting tables."""
    ref = "real" if "real" in cols else cols[0]
    rest = [c for c in cols if c != ref]
    if not rest:
        return ref, ref
    for pref in ("constrained", "baseline"):           # the model column we most want to see
        if pref in rest:
            return ref, pref
    return ref, rest[-1]


# ============================================================================
#  DnaChisel constraints
# ============================================================================
def build_constraints(args, codon_usage=None):
    """List of (name, factory). Each factory builds a fresh spec instance so evaluation
    never reuses mutated state. A codon-usage table, if given, is the AvoidRareCodons reference."""
    from dnachisel import (EnforceGCContent, AvoidPattern, AvoidRareCodons,
                           AvoidHairpins, UniquifyAllKmers, EnforceTerminalGCContent)
    from dnachisel.SequencePattern import HomopolymerPattern, RepeatedKmerPattern
    h = args.homopolymer_len
    cons = [
        ("GC_global_40_65",     lambda: EnforceGCContent(mini=0.40, maxi=0.65)),
        ("GC_window50_25_80",   lambda: EnforceGCContent(mini=0.25, maxi=0.80, window=50)),
        ("GC_terminal30_30_75", lambda: EnforceTerminalGCContent(window_size=30, mini=0.30, maxi=0.75)),
        # strand=1 keeps A-runs and T-runs separate (both-strand would alias A==T)
        (f"homopolymer_A>={h}", lambda: AvoidPattern(HomopolymerPattern("A", h), strand=1)),
        (f"homopolymer_T>={h}", lambda: AvoidPattern(HomopolymerPattern("T", h), strand=1)),
        (f"homopolymer_G>={h}", lambda: AvoidPattern(HomopolymerPattern("G", h), strand=1)),
        (f"homopolymer_C>={h}", lambda: AvoidPattern(HomopolymerPattern("C", h), strand=1)),
        ("repeat_9mer_x2", lambda: AvoidPattern(RepeatedKmerPattern(2, 9))),
        ("unique_15mer",   lambda: UniquifyAllKmers(15)),
        ("hairpin_stem20", lambda: AvoidHairpins(stem_size=20, hairpin_window=200)),
        ("internal_SD_AGGAGG", lambda: AvoidPattern("AGGAGG")),
        ("rare_codons_ecoli", lambda: AvoidRareCodons(
            args.rare_freq,
            **({"codon_usage_table": codon_usage} if codon_usage else {"species": "e_coli"}))),
    ]
    cons += [(f"enzyme_{e}", (lambda e=e: AvoidPattern(f"{e}_site"))) for e in args.enzymes]
    ok = []
    for name, fac in cons:
        try:
            fac()
            ok.append((name, fac))
        except Exception as e:
            print(f"[constraint] '{name}' unavailable -> skipped ({e})", flush=True)
    return ok


def eval_constraints(dna, constraints):
    """{name: (passed: bool|None, n_sites: int|None)} for one sequence (None = eval failed)."""
    from dnachisel import DnaOptimizationProblem
    out = {}
    for name, fac in constraints:
        try:
            prob = DnaOptimizationProblem(sequence=dna, constraints=[fac()], logger=None)
            ev = prob.constraints_evaluations().evaluations[0]
            out[name] = (bool(ev.passes), len(ev.locations) if getattr(ev, "locations", None) else 0)
        except Exception:
            out[name] = (None, None)
    return out


# ============================================================================
#  continuous metrics
# ============================================================================
def load_codon_frequencies(codon_usage_csv):
    """CSV (codon,amino_acid,frequency) -> {aa: ([codons], [freqs])} for CFD (get_cfd)."""
    usage = {}
    with open(codon_usage_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = (r.get("codon") or "").strip().upper().replace("U", "T")
            aa = (r.get("amino_acid") or "").strip()
            fr = r.get("frequency")
            if c and aa and fr not in (None, ""):
                usage.setdefault(aa, {})[c] = float(fr)
    return {aa: (list(d.keys()), list(d.values())) for aa, d in usage.items()}


def build_metrics_context(real_dnas, codon_usage_csv, mfe_window):
    ctx = {"gc": None, "cai_w": None, "cfreq": None, "mfe_window": mfe_window}
    try:
        from CodonTransformer.CodonEvaluation import get_GC_content
        ctx["gc"] = get_GC_content
    except Exception as e:
        print(f"[metric] GC helper unavailable: {e}", flush=True)
    ctx["cai_w"] = cu.load_cai_weights(real_dnas, codon_usage_csv)
    if codon_usage_csv:
        try:
            ctx["cfreq"] = load_codon_frequencies(codon_usage_csv)
        except Exception as e:
            print(f"[metric] codon-frequency table failed: {e}", flush=True)
    if ctx["cfreq"] is None:
        try:
            from CodonTransformer.CodonData import get_codon_frequencies
            ctx["cfreq"] = get_codon_frequencies(real_dnas)
        except Exception as e:
            print(f"[metric] codon frequencies unavailable: {e}", flush=True)
    return ctx


def mfe_5prime(dna, n):
    """Minimum free energy (kcal/mol) of the first n nt; needs ViennaRNA. More negative =
    stronger 5' structure = harder ribosome entry = the dominant factor for E. coli expression."""
    try:
        import RNA
        return float(RNA.fold(dna[:n].replace("T", "U"))[1])
    except Exception:
        return float("nan")


def seq_metrics(dna, ctx):
    m = {"gc": math.nan, "cai": math.nan, "cfd": math.nan, "mfe5": math.nan}
    if ctx["gc"]:
        try:
            m["gc"] = ctx["gc"](dna)
        except Exception:
            pass
    if ctx["cai_w"] is not None:
        m["cai"] = cu.cai_of(dna, ctx["cai_w"])
    if ctx["cfreq"] is not None:
        try:
            from CodonTransformer.CodonEvaluation import get_cfd
            m["cfd"] = get_cfd(dna, ctx["cfreq"])
        except Exception:
            pass
    m["mfe5"] = mfe_5prime(dna, ctx["mfe_window"])
    return m


# ============================================================================
#  generation
# ============================================================================
def generate_columns(args, data, device):
    """Generate baseline + pretrained DNA for each protein; keep rows where both match the
    real length. Returns {col: [dna]} aligned by index, plus the kept data rows."""
    tokenizer = cu.get_tokenizer()
    baseline = cu.load_finetuned_model(args.baseline_checkpoint, device)
    pretrained = cu.load_pretrained_model(device) if args.include_pretrained else None
    from CodonTransformer.CodonUtils import MAX_LEN

    seqs = {c: [] for c in COLUMNS if c == "real" or c == "baseline"
            or (c == "pretrained" and pretrained is not None)}
    kept, skip_len, skip_gen = [], 0, 0
    for i, row in enumerate(data):
        protein = row["protein"]
        if len(protein) > MAX_LEN - 2:
            skip_len += 1
            continue
        organism = args.organism or row["organism"]
        b = cu.generate_dna(baseline, tokenizer, protein, organism, device, deterministic=True)
        p = (cu.generate_dna(pretrained, tokenizer, protein, organism, device, deterministic=True)
             if pretrained is not None else None)
        if len(b) != len(row["real_dna"]) or (pretrained is not None and len(p) != len(row["real_dna"])):
            skip_gen += 1
            continue
        seqs["real"].append(row["real_dna"])
        seqs["baseline"].append(b)
        if pretrained is not None:
            seqs["pretrained"].append(p)
        kept.append(row)
        if (i + 1) % 50 == 0:
            print(f"  generated {len(kept)} (seen {i+1}/{len(data)})", flush=True)
    print(f"[gen] kept={len(kept)} skip(len={skip_len}, mismatch={skip_gen})", flush=True)
    return seqs, kept


# ============================================================================
#  analyses
# ============================================================================
def run_dnachisel(args, seqs, cols, out_dir, codon_usage):
    constraints = build_constraints(args, codon_usage)
    names = [c[0] for c in constraints]
    print(f"[dnachisel] {len(names)} constraints", flush=True)
    agg = {n: {c: {"viol": 0, "n": 0} for c in cols} for n in names}
    n_seq = len(seqs[cols[0]])
    for sid in range(n_seq):
        for c in cols:
            for name, (passed, _) in eval_constraints(seqs[c][sid], constraints).items():
                if passed is not None:
                    agg[name][c]["n"] += 1
                    agg[name][c]["viol"] += int(not passed)
        if (sid + 1) % 100 == 0:
            print(f"  evaluated {sid+1}/{n_seq}", flush=True)

    rate = lambda d: (d["viol"] / d["n"]) if d["n"] else float("nan")
    summary = [{"constraint": n, **{f"{c}_viol_rate": rate(agg[n][c]) for c in cols}} for n in names]
    ref, tgt = _ref_target(cols)
    # sort by how much the model column of interest exceeds the natural reference
    summary.sort(key=lambda s: (s.get(f"{tgt}_viol_rate", 0) - s.get(f"{ref}_viol_rate", 0)), reverse=True)
    path = os.path.join(out_dir, "violations_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    labels = [s["constraint"] for s in summary]
    x = np.arange(len(labels))
    width = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 5))
    for j, c in enumerate(cols):
        ax.bar(x + (j - (len(cols) - 1) / 2) * width,
               [s[f"{c}_viol_rate"] * 100 for s in summary], width, label=c, color=COLORS.get(c))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("Stage 2 - constraint violations by model", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "violations.png"), dpi=150, bbox_inches="tight")

    print("\n" + "=" * 70)
    print(f"DnaChisel violation rate (%)  [sorted by {tgt} - {ref}]")
    print(f'{"constraint":>22}' + "".join(f"{c:>12}" for c in cols))
    for s in summary:
        print(f'{s["constraint"]:>22}' + "".join(
            f'{s[f"{c}_viol_rate"]*100:>11.1f}' for c in cols))
    print(f"saved: {path}")
    return constraints


def run_metrics(seqs, cols, out_dir, ctx):
    means = {c: {} for c in cols}
    for c in cols:
        per = [seq_metrics(d, ctx) for d in seqs[c]]
        for k in ("gc", "cai", "cfd", "mfe5"):
            vals = [m[k] for m in per if not math.isnan(m[k])]
            means[c][k] = float(np.mean(vals)) if vals else float("nan")
    path = os.path.join(out_dir, "metrics_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + cols)
        for k, lab in [("gc", "GC%"), ("cai", "CAI"), ("cfd", "rare_codon_cfd%"), ("mfe5", "5'MFE")]:
            w.writerow([lab] + [f"{means[c][k]:.4f}" for c in cols])
    print("\n[continuous metrics: mean per model]")
    print(f'{"metric":>16}' + "".join(f"{c:>12}" for c in cols))
    for k, lab in [("gc", "GC%"), ("cai", "CAI"), ("cfd", "cfd%"), ("mfe5", "5'MFE")]:
        print(f"{lab:>16}" + "".join(f"{means[c][k]:>12.3f}" for c in cols))
    print(f"saved: {path}")


def run_idt(args, seqs, cols, out_dir):
    from idt_client import IdtClient
    client = IdtClient(args.idt_credentials, host=args.idt_host, kind=args.idt_kind, rate=args.idt_rate)
    print(f"[idt] host={args.idt_host} kind={args.idt_kind}", flush=True)
    col_scores = {}
    for c in cols:
        print(f"[idt] scoring '{c}' ...", flush=True)
        named = [(f"{c}_{i}", s) for i, s in enumerate(seqs[c]) if s]
        if args.idt_limit > 0:
            named = named[:args.idt_limit]
        col_scores[c] = client.score(named, batch=args.idt_batch, workers=args.idt_workers,
                                     debug_path=os.path.join(out_dir, f"idt_raw_{c}.json"))

    n = len(seqs[cols[0]])
    path = os.path.join(out_dir, "idt_complexity.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id"] + [f"{c}_complexity" for c in cols])
        for i in range(n):
            row = [i]
            for c in cols:
                sc = col_scores[c].get(f"{c}_{i}", (float("nan"), []))[0]
                row.append("" if sc != sc else f"{sc:.2f}")
            w.writerow(row)

    tc, tr = args.t_complex, args.t_reject
    print("\n" + "=" * 70)
    print(f"IDT complexity (higher = harder; >{tc} complex, >{tr} reject; kind={args.idt_kind})")
    arrs = {}
    for c in cols:
        v = [col_scores[c][f"{c}_{i}"][0] for i in range(n)
             if f"{c}_{i}" in col_scores[c] and col_scores[c][f"{c}_{i}"][0] == col_scores[c][f"{c}_{i}"][0]]
        arrs[c] = v
        if v:
            a = np.array(v)
            print(f"  {c:>11}: n={len(a)}  mean={a.mean():.2f}  median={np.median(a):.2f}  "
                  f">{tc}={100*(a > tc).mean():.1f}%  >{tr}={100*(a > tr).mean():.1f}%")
    _idt_issue_breakdown(cols, col_scores, n, out_dir)

    plt.figure(figsize=(8, 5))
    for c in cols:
        if arrs[c]:
            plt.hist(arrs[c], bins=40, alpha=0.5, label=c, color=COLORS.get(c))
    plt.axvline(tc, ls="--", c="orange", lw=1, label=f"complex >{tc}")
    plt.axvline(tr, ls="--", c="red", lw=1, label=f"reject >{tr}")
    plt.xlabel("IDT complexity score")
    plt.ylabel("count")
    plt.legend()
    plt.title("IDT synthesis complexity", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "idt_complexity.png"), dpi=150)
    print(f"saved: {path}\n       {os.path.join(out_dir, 'idt_complexity.png')}")


def _idt_issue_breakdown(cols, col_scores, n, out_dir):
    """Percent of sequences hitting each IDT issue, per column (IDT's own 'what to fix')."""
    pct = {c: {} for c in cols}
    for c in cols:
        for i in range(n):
            v = col_scores[c].get(f"{c}_{i}")
            if not v:
                continue
            for nm in {tup[0] for tup in v[1]}:
                pct[c][nm] = pct[c].get(nm, 0) + 1
    names = sorted(set().union(*[set(pct[c]) for c in cols])) if cols else []
    path = os.path.join(out_dir, "idt_issue_breakdown.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idt_issue"] + [f"{c}_pct" for c in cols])
        for nm in names:
            w.writerow([nm] + [f"{100.0 * pct[c].get(nm, 0) / max(n, 1):.1f}" for c in cols])
    _, tgt = _ref_target(cols)
    print("\n[IDT issues: % of sequences hitting each]")
    print(f'{"idt_issue":>26}' + "".join(f"{c:>12}" for c in cols))
    for nm in sorted(names, key=lambda nm: pct.get(tgt, {}).get(nm, 0), reverse=True)[:12]:
        print(f"{nm:>26}" + "".join(f'{100.0*pct[c].get(nm,0)/max(n,1):>11.1f}' for c in cols))


# ============================================================================
#  main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: violation analysis (baseline vs pretrained)")
    p.add_argument("--baseline_checkpoint", default=None, help="raw state_dict from baseline.py")
    p.add_argument("--sequences", default=None,
                   help="score a pre-generated sequences.csv (e.g. decodebio.py output) and skip generation")
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--n", type=int, default=500, help="number of proteins to evaluate (0 = all)")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default=None)
    p.add_argument("--organism", default=None, help="force a generation organism (default: per-record id)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--codon_usage", default=None, help="codon usage CSV (codon,amino_acid,frequency)")
    p.add_argument("--no_pretrained", dest="include_pretrained", action="store_false",
                   help="skip the stock CodonTransformer comparison column")
    # DnaChisel
    p.add_argument("--no_dnachisel", dest="run_dnachisel", action="store_false")
    p.add_argument("--homopolymer_len", type=int, default=6)
    p.add_argument("--rare_freq", type=float, default=0.1)
    p.add_argument("--enzymes", nargs="+",
                   default=["BsaI", "BsmBI", "BbsI", "SapI", "AarI", "EcoRI", "BamHI",
                            "HindIII", "XhoI", "NdeI", "NotI", "XbaI"])
    p.add_argument("--mfe_window", type=int, default=45, help="5' MFE window (needs ViennaRNA)")
    # IDT (optional; only runs if credentials are given)
    p.add_argument("--idt_credentials", default=None, help="IDT credentials JSON; enables IDT scoring")
    p.add_argument("--idt_host", default="sg.idtdna.com")
    p.add_argument("--idt_kind", default="gene", choices=["gene", "gblock", "gblock_hifi", "eblock", "old"])
    p.add_argument("--idt_batch", type=int, default=90)
    p.add_argument("--idt_workers", type=int, default=8)
    p.add_argument("--idt_rate", type=int, default=450)
    p.add_argument("--idt_limit", type=int, default=0)
    p.add_argument("--t_complex", type=float, default=7.0)
    p.add_argument("--t_reject", type=float, default=20.0)
    return p.parse_args()


def load_sequences_csv(path):
    """Read a sequences.csv (seq_id + one *_dna column per model) -> ({col: [dna]}, cols)."""
    rows = list(csv.DictReader(open(path)))
    headers = [h for h in rows[0].keys() if h != "seq_id"] if rows else []
    cols = [h[:-4] if h.endswith("_dna") else h for h in headers]
    seqs = {c: [] for c in cols}
    for r in rows:
        for h, c in zip(headers, cols):
            seqs[c].append((r.get(h) or "").strip().upper().replace("U", "T"))
    return seqs, cols


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.sequences:                                   # score a pre-generated CSV (Method C etc.)
        seqs, cols = load_sequences_csv(args.sequences)
        if not seqs or not seqs[cols[0]]:
            sys.exit(f"no sequences in {args.sequences}")
        print(f"[sequences] {len(seqs[cols[0]])} rows from {args.sequences}, cols={cols}", flush=True)
    else:
        if not args.baseline_checkpoint:
            sys.exit("provide --baseline_checkpoint (to generate) or --sequences (to score a CSV)")
        import torch
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[setup] device={device} baseline={args.baseline_checkpoint}", flush=True)
        data = cu.load_eval_dataset(args.data, args.n, args.seed)
        print(f"[data] {len(data)} sequences from {args.data}", flush=True)
        if not data:
            sys.exit("no data / bad codons format")
        seqs, _ = generate_columns(args, data, device)
        cols = [c for c in COLUMNS if c in seqs]
        if not seqs["baseline"]:
            sys.exit("no sequences generated")
        with open(os.path.join(args.out_dir, "sequences.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seq_id"] + [f"{c}_dna" for c in cols])
            for i in range(len(seqs["baseline"])):
                w.writerow([i] + [seqs[c][i] for c in cols])
        print(f"[dump] sequences.csv ({len(seqs['baseline'])} rows, cols={cols})", flush=True)

    # codon usage table for DnaChisel rare codons (aa -> {codon: freq})
    usage = None
    if args.codon_usage:
        try:
            usage = {}
            with open(args.codon_usage, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    c = (r.get("codon") or "").strip().upper().replace("U", "T")
                    aa = (r.get("amino_acid") or "").strip()
                    fr = r.get("frequency")
                    if c and aa and fr not in (None, ""):
                        usage.setdefault(aa, {})[c] = float(fr)
        except Exception as e:
            print(f"[codon_usage] read failed: {e}", flush=True)
            usage = None

    if args.run_dnachisel:
        try:
            run_dnachisel(args, seqs, cols, args.out_dir, usage)
        except Exception as e:
            print(f"[dnachisel] skipped: {e}", flush=True)

    ref_real = seqs["real"] if "real" in seqs else seqs[cols[0]]
    ctx = build_metrics_context(ref_real, args.codon_usage, args.mfe_window)
    run_metrics(seqs, cols, args.out_dir, ctx)

    if args.idt_credentials:
        run_idt(args, seqs, cols, args.out_dir)
    else:
        print("\n[idt] no --idt_credentials -> skipping IDT complexity scoring", flush=True)


if __name__ == "__main__":
    main()
