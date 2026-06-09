# DeepCodonOpt

**Where should biological constraints enter a neural codon optimizer?**
Code for our study of [CodonTransformer](https://github.com/Adibvafa/CodonTransformer) under the
IDT synthesis-complexity score.

> 📄 Paper: `paper/main.tex` (NeurIPS-2025 format) &nbsp;·&nbsp; 📦 Checkpoint + sequences: Zenodo `10.5281/zenodo.20602294` (see `DATA_UPLOAD.md`)

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

## Reproduce

```bash
# (data) build E. coli train/val/test  — or use the CodonTransformer corpus directly
python build_dataset.py --input-dir data/ncbi --out-dir data --val-ratio 0.1 --test-ratio 0.1

# §2 baseline fine-tuning
python baseline.py --train_data_path data/train.jsonl --val_data_path data/val.jsonl \
    --test_data_path data/test.jsonl --learning_rate 5e-5 --checkpoint_dir out/baseline

# §3 diagnosis: real vs baseline vs pretrained (needs IDT credentials for complexity)
python analyze.py --baseline_checkpoint out/baseline/best.pt --data data/test.jsonl --n 3000 \
    --codon_usage ecoli_codon_usage_table.csv --idt_credentials credentials.json --out_dir results/analyze

# §4 Method A (constraint loss) — then re-score with analyze.py
python lossbio.py --train_data_path data/train.jsonl --val_data_path data/val.jsonl \
    --test_data_path data/test.jsonl --init_checkpoint out/baseline/best.pt \
    --checkpoint_dir out/methodA --lambda_bio 1.0 --repeat_thr 0.4 --hairpin_thr 0.4

# §5 Method B (temperature + masking) — sweep T, then aggregate
for T in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  python decodebio.py --checkpoint out/baseline/best.pt --data data/test.jsonl --n 3000 \
      --codon_usage ecoli_codon_usage_table.csv --temperature $T --top_p 0.95 \
      --out_dir results/decode/T$T
done
python analyze_decodebio.py --glob 'results/decode/T*' \
    --codon_usage ecoli_codon_usage_table.csv --idt_credentials credentials.json --out_dir results/sweep

# figures for the paper
python make_report_figs.py --result_dir _result --out_dir paper/figs
```

**IDT credentials** are read from a JSON file and never committed (see `.gitignore`):
`{"ID":"…","secret":"…","username":"…","password":"…","token_file_path":"~/.idt_token.json"}`.

## Artifacts

The fine-tuned checkpoint (`ep50_lr5e-05.ckpt`) and the Method B best sequences are archived on
Zenodo — see `DATA_UPLOAD.md`.

## Acknowledgements

Built on [CodonTransformer](https://github.com/Adibvafa/CodonTransformer) (Fallahpour et al., 2025).
Synthesis complexity from IDT SciTools Plus.
