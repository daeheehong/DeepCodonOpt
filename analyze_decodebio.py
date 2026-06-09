#!/usr/bin/env python3
"""Aggregate several decodebio.py temperature runs into one comparison + Pareto.

Each run directory holds a ``sequences.csv`` (columns real / unconstrained / constrained)
produced by decodebio.py at a given temperature. This script reads them all, computes the
headline quality and manufacturability numbers per temperature per column, writes a summary
table, and draws the temperature trade-off curves (the Method C Pareto).

Manufacturability is measured with the same DnaChisel constraint set as analyze.py (no API),
and -- if IDT credentials are supplied -- the IDT complexity score as well.

Usage
    python analyze_decodebio.py --glob './output/T*' \
        --codon_usage ecoli_codon_usage_table.csv --n 500 \
        [--idt_credentials credentials.json] --out_dir ./output/sweep
"""

import argparse
import csv
import glob
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze

# DnaChisel constraint name -> headline group
GROUPS = {
    "enzyme": lambda n: n.startswith("enzyme_"),
    "homopolymer": lambda n: n.startswith("homopolymer_"),
    "internal_SD": lambda n: n == "internal_SD_AGGAGG",
    "rare": lambda n: n == "rare_codons_ecoli",
    "repeat": lambda n: n in ("repeat_9mer_x2", "unique_15mer"),
    "hairpin": lambda n: n == "hairpin_stem20",
    "GC": lambda n: n.startswith("GC_"),
}
MODEL_COLS = ["unconstrained", "constrained"]


def parse_temp(label):
    """Pull a temperature out of a directory name. 'T08' -> 0.8, 'T10' -> 1.0, 'T0.8' -> 0.8."""
    m = re.search(r"T[_-]?(\d+(?:\.\d+)?)", label, re.I) or re.search(r"(\d+(?:\.\d+)?)", label)
    if not m:
        return None
    s = m.group(1)
    if "." in s:
        return float(s)
    v = int(s)
    return v / 10.0 if v >= 1 else 0.0          # the T01..T10 (tenths) naming used by the sweep


def load_codon_usage(path):
    """codon,amino_acid,frequency CSV -> {aa: {codon: freq}} (DnaChisel AvoidRareCodons table)."""
    usage = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = (r.get("codon") or "").strip().upper().replace("U", "T")
            aa = (r.get("amino_acid") or "").strip()
            fr = r.get("frequency")
            if c and aa and fr not in (None, ""):
                usage.setdefault(aa, {})[c] = float(fr)
    return usage


def group_rates(dnas, constraints):
    """% of sequences that fail >=1 constraint in each headline group."""
    grp_for = {}
    for name, _ in constraints:
        for g, f in GROUPS.items():
            if f(name):
                grp_for[name] = g
    counts = {g: 0 for g in GROUPS}
    for dna in dnas:
        hit = set()
        for name, (passed, _) in analyze.eval_constraints(dna, constraints).items():
            if passed is False and name in grp_for:
                hit.add(grp_for[name])
        for g in hit:
            counts[g] += 1
    n = max(len(dnas), 1)
    return {g: 100.0 * counts[g] / n for g in GROUPS}


def metric_means(dnas, ctx):
    acc = {"gc": [], "cai": [], "cfd": [], "mfe5": []}
    for dna in dnas:
        m = analyze.seq_metrics(dna, ctx)
        for k in acc:
            if not math.isnan(m[k]):
                acc[k].append(m[k])
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def idt_stats(dnas, label, client, batch, workers, tc):
    named = [(f"{label}_{i}", s) for i, s in enumerate(dnas) if s]
    res = client.score(named, batch=batch, workers=workers)
    totals, n_repeat = [], 0
    for i in range(len(dnas)):
        v = res.get(f"{label}_{i}")
        if not v:
            continue
        if v[0] == v[0]:
            totals.append(v[0])
        if any(t[0] == "Overall Repeat" for t in v[1]):
            n_repeat += 1
    a = np.array(totals) if totals else np.array([np.nan])
    return dict(idt_mean=float(np.nanmean(a)), idt_median=float(np.nanmedian(a)),
                idt_complex_pct=(100.0 * np.mean(a > tc)) if totals else float("nan"),
                idt_repeat_pct=100.0 * n_repeat / max(len(dnas), 1))


def evaluate(dnas, constraints, ctx, client, args):
    row = {}
    row.update({f"{g}_viol%": v for g, v in group_rates(dnas, constraints).items()})
    m = metric_means(dnas, ctx)
    row.update({"GC%": m["gc"], "CAI": m["cai"], "cfd%": m["cfd"], "MFE5": m["mfe5"]})
    if client is not None:
        row.update(idt_stats(dnas, "x", client, args.idt_batch, args.idt_workers, args.t_complex))
    return row


# --------------------------------------------------------------------------- plots
def _xy(rows, col, xkey):
    pts = sorted([r for r in rows if r["column"] == col and r.get(xkey) == r.get(xkey)
                  and r["CAI"] == r["CAI"]], key=lambda r: r["temperature"])
    return [r[xkey] for r in pts], [r["CAI"] for r in pts], [r["temperature"] for r in pts]


def make_plots(rows, real_row, xkey, xlabel, out_dir):
    # 1) Pareto: CAI vs manufacturability, one line per column, annotated by temperature
    plt.figure(figsize=(7, 5.5))
    for col, color in [("unconstrained", "#C44E52"), ("constrained", "#8172B3")]:
        x, y, t = _xy(rows, col, xkey)
        if not x:
            continue
        plt.plot(x, y, "-o", color=color, label=col, alpha=0.85)
        for xi, yi, ti in zip(x, y, t):
            plt.annotate(f"{ti:g}", (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if real_row and real_row.get(xkey) == real_row.get(xkey):
        plt.scatter([real_row[xkey]], [real_row["CAI"]], marker="*", s=180, c="#4C72B0",
                    zorder=5, label="real (natural)")
    plt.xlabel(xlabel + "  (lower = easier to manufacture)")
    plt.ylabel("CAI  (higher = better codon usage)")
    plt.title("Method C temperature trade-off", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "pareto_temperature.png")
    plt.savefig(p1, dpi=150)

    # 2) metric-vs-temperature panels
    panels = [("CAI", "CAI"), (xkey, xlabel), ("enzyme_viol%", "enzyme site %"),
              ("homopolymer_viol%", "homopolymer %")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, lab) in zip(axes.ravel(), panels):
        for col, color in [("unconstrained", "#C44E52"), ("constrained", "#8172B3")]:
            pts = sorted([r for r in rows if r["column"] == col and r.get(key) == r.get(key)],
                         key=lambda r: r["temperature"])
            if pts:
                ax.plot([r["temperature"] for r in pts], [r[key] for r in pts], "-o",
                        color=color, label=col)
        ax.set_xlabel("temperature")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    fig.suptitle("decodebio: metrics vs temperature", fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "metrics_vs_temperature.png")
    plt.savefig(p2, dpi=150)
    return p1, p2


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="aggregate decodebio temperature runs")
    ap.add_argument("--glob", default="./output/T*", help="glob of run dirs (each with sequences.csv)")
    ap.add_argument("--temps", type=float, nargs="*", help="override parsed temperatures (aligned to sorted dirs)")
    ap.add_argument("--n", type=int, default=500, help="evaluate the first N sequences per run (0 = all)")
    ap.add_argument("--codon_usage", default=None)
    ap.add_argument("--homopolymer_len", type=int, default=6)
    ap.add_argument("--rare_freq", type=float, default=0.1)
    ap.add_argument("--enzymes", nargs="+",
                    default=["BsaI", "BsmBI", "BbsI", "SapI", "AarI", "EcoRI", "BamHI",
                             "HindIII", "XhoI", "NdeI", "NotI", "XbaI"])
    ap.add_argument("--mfe_window", type=int, default=45)
    ap.add_argument("--idt_credentials", default=None, help="enables IDT complexity scoring")
    ap.add_argument("--idt_host", default="sg.idtdna.com")
    ap.add_argument("--idt_kind", default="gene")
    ap.add_argument("--idt_batch", type=int, default=90)
    ap.add_argument("--idt_workers", type=int, default=8)
    ap.add_argument("--idt_rate", type=int, default=450)
    ap.add_argument("--t_complex", type=float, default=7.0)
    ap.add_argument("--out_dir", default="./output/sweep")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dirs = sorted(d for d in glob.glob(args.glob)
                  if os.path.isfile(os.path.join(d, "sequences.csv")))
    if not dirs:
        raise SystemExit(f"no run dirs with sequences.csv matched {args.glob}")
    temps = args.temps if args.temps else [parse_temp(os.path.basename(d.rstrip("/"))) for d in dirs]
    print("[runs] " + ", ".join(f"{os.path.basename(d.rstrip('/'))}->T={t}" for d, t in zip(dirs, temps)))

    usage = load_codon_usage(args.codon_usage) if args.codon_usage else None
    cons_args = SimpleNamespace(homopolymer_len=args.homopolymer_len,
                                rare_freq=args.rare_freq, enzymes=args.enzymes)
    constraints = analyze.build_constraints(cons_args, usage)
    print(f"[dnachisel] {len(constraints)} constraints", flush=True)

    client = None
    if args.idt_credentials:
        from idt_client import IdtClient
        client = IdtClient(args.idt_credentials, host=args.idt_host, kind=args.idt_kind, rate=args.idt_rate)

    rows, real_row, ctx = [], None, None
    for d, t in zip(dirs, temps):
        seqs, cols = analyze.load_sequences_csv(os.path.join(d, "sequences.csv"))
        cut = (lambda lst: lst[:args.n] if args.n else lst)
        seqs = {c: cut(seqs[c]) for c in cols}
        if ctx is None:                                   # build CAI/cfd reference + score real once
            ctx = analyze.build_metrics_context(seqs.get("real", seqs[cols[0]]), args.codon_usage, args.mfe_window)
            if "real" in seqs:
                real_row = {"temperature": float("nan"), "column": "real",
                            **evaluate(seqs["real"], constraints, ctx, client, args)}
        print(f"[T={t}] evaluating {os.path.basename(d.rstrip('/'))} "
              f"({len(seqs[cols[0]])} seqs/col) ...", flush=True)
        for col in MODEL_COLS:
            if col in seqs:
                rows.append({"temperature": t, "column": col,
                             **evaluate(seqs[col], constraints, ctx, client, args)})

    # ---- summary CSV ----
    keys = ["temperature", "column"] + [k for k in rows[0] if k not in ("temperature", "column")]
    summary = os.path.join(args.out_dir, "decodebio_sweep_summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        if real_row:
            w.writerow(real_row)
        w.writerows(sorted(rows, key=lambda r: (r["column"], r["temperature"])))

    # ---- console table ----
    xkey = "idt_mean" if client is not None else "repeat_viol%"
    xlabel = "IDT complexity (mean)" if client is not None else "repeat violation %"
    print("\n" + "=" * 88)
    print(f'{"T":>5} {"column":>14} {"CAI":>7} {xlabel:>22} {"enzyme%":>8} {"homopol%":>9} {"SD%":>6}')
    if real_row:
        print(f'{"real":>5} {"real":>14} {real_row["CAI"]:>7.3f} {real_row.get(xkey, float("nan")):>22.2f} '
              f'{real_row["enzyme_viol%"]:>8.1f} {real_row["homopolymer_viol%"]:>9.1f} {real_row["internal_SD_viol%"]:>6.1f}')
    for r in sorted(rows, key=lambda r: (r["column"], r["temperature"])):
        print(f'{r["temperature"]:>5g} {r["column"]:>14} {r["CAI"]:>7.3f} {r.get(xkey, float("nan")):>22.2f} '
              f'{r["enzyme_viol%"]:>8.1f} {r["homopolymer_viol%"]:>9.1f} {r["internal_SD_viol%"]:>6.1f}')

    p1, p2 = make_plots(rows, real_row, xkey, xlabel, args.out_dir)
    print(f"\nsaved: {summary}\n       {p1}\n       {p2}", flush=True)


if __name__ == "__main__":
    main()
