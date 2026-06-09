# DeepCodonOpt

**Where should biological constraints enter a neural codon optimizer?**
Code for our study of [CodonTransformer](https://github.com/Adibvafa/CodonTransformer) under the
IDT synthesis-complexity score.

## Finding

A CodonTransformer fine-tuned on *E. coli* has **better-than-natural codon usage (CAI 0.89 vs 0.74)**
but **far-worse synthesizability (IDT complexity 4.9 vs 1.9)** because deterministic decoding is
*repetitive*. Adding a **differentiable constraint loss (Method A)** backfires — it is minimized by
flattening the model's distribution rather than fixing the sequence, so real repeats explode and
complexity rises to 9.3. **Constrained temperature decoding (Method B)** — masking violating codons
and sampling at temperature *T* — instead removes the repeats and, at *T* ≈ 0.7–0.8, **beats natural
genes on both synthesizability and codon adaptation**. Discrete constraints belong on the discrete
output at decode time, not in a training surrogate.

## Repository layout

```
codon_utils.py     Shared: data pipeline, training driver, model loading, generation, CAI
bio_losses.py      Differentiable constraint losses (Method A)
idt_client.py      IDT SciTools complexity API client
build_dataset.py   NCBI genome → JSONL + split + size subsets   (data prep, optional)
baseline.py        §2  Baseline: fine-tune CodonTransformer on E. coli
analyze.py         §3  IDT + metric evaluation (also scores a sequences.csv via --sequences)
lossbio.py         §4  Method A — constraint-loss fine-tuning
decodebio.py       §5  Method B — temperature sampling + constraint masking
analyze_decodebio.py   §5  aggregate a temperature sweep into a summary + Pareto
make_report_figs.py    regenerate the paper figures from results
ecoli_codon_usage_table.csv   E. coli codon usage (CAI / rare-codon reference)
paper/             main.tex, reference.bib, slides_content.md
```

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/Adibvafa/CodonTransformer && pip install -e ./CodonTransformer
```

## IDT credentials
**IDT credentials** are read from a JSON file:
`{"ID":"…","secret":"…","username":"…","password":"…","token_file_path":"~/.idt_token.json"}`.

## Artifacts

The fine-tuned checkpoint is archived on Zenodo — see `10.5281/zenodo.20602294`.

## Acknowledgements

Built on [CodonTransformer](https://github.com/Adibvafa/CodonTransformer) (Fallahpour et al., 2025).
Synthesis complexity from IDT SciTools Plus.
