#!/usr/bin/env python3
"""Regenerate the report figures from the _result CSVs (IDT-only, DnaChisel excluded).

Writes to report_figs/:
    fig3_idt_breakdown.png        IDT issue breakdown: real vs baseline vs pretrained   (Section 3)
    fig4_methodA_breakdown.png    IDT issue breakdown: real vs baseline vs Method A      (Section 4)
    fig5_pareto.png               CAI vs IDT complexity over temperature (Method B)        (Section 5)
    fig5_issues.png               the four IDT issues vs temperature (Method B)            (Section 5)

Method A = constraint loss (Section 4); Method B = temperature sampling + masking (Section 5).
Pure csv + matplotlib (no torch / CodonTransformer). Run from the DeepCodonOpt folder:
    python make_report_figs.py
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_REAL = "#4C72B0"
C_BASE = "#C44E52"
C_PRE = "#55A868"
C_MODEL = "#8172B3"          # Method A / Method B accent

# the four IDT issues shown in the report (exact API names)
ISSUES = ["Overall Repeat", "Repeat Length (Fragment)",
          "Windowed Repeat Percentage", "Windowed High GC (100)"]
ISSUE_SHORT = {"Overall Repeat": "Overall Repeat", "Repeat Length (Fragment)": "Repeat Length (frag)",
               "Windowed Repeat Percentage": "Windowed Repeat %", "Windowed High GC (100)": "Windowed High GC"}


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def grouped_bar(series, labels, colors, issues, title, path):
    import numpy as np
    x = np.arange(len(issues))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(issues)), 4.8))
    for j, lab in enumerate(labels):
        vals = [series.get(iss, {}).get(lab, 0.0) for iss in issues]
        ax.bar(x + (j - (len(labels) - 1) / 2) * w, vals, w, label=lab, color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([ISSUE_SHORT.get(i, i).replace(" (frag)", "\n(frag)").replace(" GC", " GC")
                        for i in issues], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("% of sequences hitting issue")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


# --------------------------------------------------------------------------- Section 3
def fig_step2(path_csv, out):
    rows = read_rows(path_csv)
    series = {r["idt_issue"]: {"real": _f(r["real_pct"]), "baseline": _f(r["baseline_pct"]),
                               "pretrained": _f(r["pretrained_pct"])} for r in rows}
    grouped_bar(series, ["real", "baseline", "pretrained"], [C_REAL, C_BASE, C_PRE],
                ISSUES, "IDT issues: natural vs baseline vs pretrained", out)


# --------------------------------------------------------------------------- Section 4
def fig_methodA(step2_csv, lossbio_csv, out):
    s2 = {r["idt_issue"]: r for r in read_rows(step2_csv)}          # has real + baseline (Stage-1 FT)
    lb = {r["idt_issue"]: r for r in read_rows(lossbio_csv)}        # 'baseline' column here = Method A
    series = {}
    for iss in set(s2) | set(lb):
        series[iss] = {
            "real": _f(s2.get(iss, {}).get("real_pct"), _f(lb.get(iss, {}).get("real_pct"))),
            "baseline": _f(s2.get(iss, {}).get("baseline_pct")),
            "Method A": _f(lb.get(iss, {}).get("baseline_pct")),
        }
    grouped_bar(series, ["real", "baseline", "Method A"], [C_REAL, C_BASE, C_MODEL],
                ISSUES, "Method A (constraint loss): GC down, repeats up", out)


# --------------------------------------------------------------------------- Section 5
def fig_pareto(sweep_csv, out):
    rows = read_rows(sweep_csv)
    cts = sorted([r for r in rows if r["column"] == "constrained"], key=lambda r: _f(r["temperature"]))
    real = next((r for r in rows if r["column"] == "real"), None)
    xs = [_f(r["idt_mean"]) for r in cts]
    ys = [_f(r["CAI"]) for r in cts]
    ts = [_f(r["temperature"]) for r in cts]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(xs, ys, "-o", color=C_MODEL, label="Method B", alpha=0.9)
    for x, y, t in zip(xs, ys, ts):
        ax.annotate(f"T={t:g}", (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    if real:
        ax.scatter([_f(real["idt_mean"])], [_f(real["CAI"])], marker="*", s=230, c=C_REAL,
                   zorder=5, label="natural (real)")
    ax.set_xlabel("IDT complexity (mean)   — lower = easier to synthesize")
    ax.set_ylabel("CAI   — higher = better codon usage")
    ax.set_title("Method B: codon adaptation vs synthesizability (temperature)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def fig_issues_vs_temp(decode_out_dir, out):
    """Read the per-temperature folders T01..T10 and plot the four issues (Method B) vs temperature."""
    dirs = sorted(glob.glob(os.path.join(decode_out_dir, "T[0-9][0-9]")))
    temps, data, real = [], {iss: [] for iss in ISSUES}, {}
    for d in dirs:
        m = re.search(r"T(\d+)$", os.path.basename(d))
        bd = os.path.join(d, "idt_issue_breakdown.csv")
        if not m or not os.path.isfile(bd):
            continue
        rows = {r["idt_issue"]: r for r in read_rows(bd)}
        temps.append(int(m.group(1)) / 10.0)
        for iss in ISSUES:
            data[iss].append(_f(rows.get(iss, {}).get("constrained_pct")))
            real.setdefault(iss, _f(rows.get(iss, {}).get("real_pct")))
    if not temps:
        print("warn: no per-temperature folders found under", decode_out_dir)
        return
    colors = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for iss, c in zip(ISSUES, colors):
        ax.plot(temps, data[iss], "-o", color=c, label=ISSUE_SHORT[iss])
        ax.axhline(real[iss], ls="--", color=c, lw=0.8, alpha=0.55)
    ax.set_xlabel("temperature")
    ax.set_ylabel("% of sequences hitting issue")
    ax.set_title("Method B: IDT issues vs temperature  (dashed = natural)", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description="regenerate report figures from _result CSVs")
    ap.add_argument("--result_dir", default="_result")
    ap.add_argument("--out_dir", default="report_figs")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    R = a.result_dir

    fig_step2(os.path.join(R, "2_analyze", "step2_idt_breakdown.csv"),
              os.path.join(a.out_dir, "fig3_idt_breakdown.png"))
    fig_methodA(os.path.join(R, "2_analyze", "step2_idt_breakdown.csv"),
                os.path.join(R, "3_lossbio", "lossbio3_output", "idt_issue_breakdown.csv"),
                os.path.join(a.out_dir, "fig4_methodA_breakdown.png"))
    decode_out = os.path.join(R, "4_decodebio", "output")
    fig_pareto(os.path.join(decode_out, "decodebio_sweep_summary.csv"),
               os.path.join(a.out_dir, "fig5_pareto.png"))
    fig_issues_vs_temp(decode_out, os.path.join(a.out_dir, "fig5_issues.png"))
    print("\nall figures ->", a.out_dir)


if __name__ == "__main__":
    main()
