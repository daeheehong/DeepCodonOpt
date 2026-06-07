# DeepCodonOpt

Biologically-constrained codon optimization for *E. coli*. The project asks a simple
question: **a neural codon optimizer ([CodonTransformer](https://github.com/Adibvafa/CodonTransformer))
produces high-quality coding sequences, but how badly do those sequences violate the
manufacturing and biological rules that real gene synthesis must satisfy — and can we fix
that by adding differentiable constraint losses during fine-tuning?**

The pipeline has three stages:

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 | `baseline.py` | Fine-tune CodonTransformer on *E. coli* (masked-LM); sweep learning rate / epochs / data size to pick the best baseline checkpoint. |
| 2 | `analyze.py` | Generate DNA with the **baseline** and the **stock pretrained** model for the same proteins, then measure rule violations with the **IDT SciTools Plus** complexity API and the **DnaChisel** constraint set. |
| 3 | `lossbio.py` | Continue fine-tuning with differentiable manufacturability losses (GC / repeat / hairpin, each mapped to an IDT issue) added to the MLM loss, and track how well each term is driven down. |

```
build_dataset.py ──▶ data/{train,val,test}.jsonl
                          │
                          ▼
   Stage 1  baseline.py ──▶ baseline checkpoint
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   Stage 2  analyze.py        Stage 3  lossbio.py ──▶ constraint-aware checkpoint
   (baseline vs pretrained)            │
            ▲                          │
            └──────── validate ────────┘   (re-run analyze.py on the Stage 3 checkpoint)
```

## Repository layout

```
codon_utils.py     Shared: data pipeline, training driver, model loading, generation, CAI
bio_losses.py      Differentiable GC / repeat / hairpin losses (codon → soft-nucleotide)
idt_client.py      IDT SciTools Plus complexity API client (auth, rate limit, batching)
build_dataset.py   NCBI genome → CodonTransformer JSONL + train/val/test split + size subsets
baseline.py        Stage 1 entry point
analyze.py         Stage 2 entry point
lossbio.py         Stage 3 entry point
ecoli_codon_usage_table.csv   E. coli codon usage (CAI / rare-codon reference)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# CodonTransformer is a required dependency, installed from source:
git clone https://github.com/Adibvafa/CodonTransformer
pip install -e ./CodonTransformer
```

## 1. Data preparation

Download an *E. coli* genome from NCBI and place it under `data/ncbi/`. This project uses
**E. coli BL21(DE3)**, RefSeq `GCF_000022665.1`; either the single GBFF file or the
CDS + protein FASTA pair works (`.gz` is fine, auto-detected):

```
https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/022/665/GCF_000022665.1_ASM2266v1/GCF_000022665.1_ASM2266v1_genomic.gbff.gz
```

Then build the dataset (strict QC, split, optional size subsets):

```bash
python build_dataset.py --input-dir data/ncbi --out-dir data \
    --max-codons 2046 --val-ratio 0.1 --test-ratio 0.1 \
    --sizes 500 1000 2000 5000 10000
```

Produces `data/train.jsonl`, `val.jsonl`, `test.jsonl`, a human-readable pair table, and
`data/sizes/train_<n>.jsonl` for the data-scaling experiment.

## 2. Stage 1 — baseline fine-tuning

```bash
python baseline.py \
    --train_data_path data/train.jsonl \
    --val_data_path data/val.jsonl \
    --test_data_path data/test.jsonl \
    --learning_rate 5e-5 --max_epochs 15 \
    --checkpoint_dir out/baseline
```

Multi-GPU is automatic with `torchrun --nproc_per_node=N baseline.py ...`. Sweep
`--learning_rate` / `--max_epochs`, or point `--train_data_path` at the size subsets, to
run the two Stage 1 experiments. The best (lowest val-loss) checkpoint is saved to
`<checkpoint_dir>/best.pt`.

## 3. Stage 2 — violation analysis

```bash
python analyze.py \
    --baseline_checkpoint out/baseline/best.pt \
    --data data/test.jsonl --n 500 \
    --codon_usage ecoli_codon_usage_table.csv \
    --idt_credentials credentials.json \
    --out_dir results
```

Generates DNA from the baseline and the stock pretrained model, then writes per-constraint
violation rates (DnaChisel), continuous metrics (GC / CAI / CFD / 5'MFE), and — if
`--idt_credentials` is supplied — IDT complexity scores and an issue breakdown. Omit
`--idt_credentials` to run the local DnaChisel analysis only; add `--no_pretrained` to skip
the pretrained comparison column.

**IDT credentials** are read from a JSON file and are never committed (see `.gitignore`):

```json
{"ID": "<client_id>", "secret": "<client_secret>",
 "username": "<idt_login>", "password": "<pw>",
 "token_file_path": "~/.idt_token.json"}
```

## 4. Stage 3 — constraint-aware fine-tuning

```bash
python lossbio.py \
    --train_data_path data/train.jsonl \
    --val_data_path data/val.jsonl \
    --test_data_path data/test.jsonl \
    --init_checkpoint out/baseline/best.pt \
    --checkpoint_dir out/lossbio_full \
    --lambda_bio 1.0 --lambda_gc 1.0 --lambda_repeat 1.0 --lambda_hairpin 0.5
```

Each loss component is logged every epoch. Ablate a single term by zeroing the others
(e.g. `--lambda_gc 0 --lambda_repeat 1 --lambda_hairpin 0`). The constraint losses are
**surrogates** for the IDT score, so always re-run `analyze.py` on the Stage 3 checkpoint
to confirm the true complexity and DnaChisel violations actually drop.

## Notes

- The differentiable losses operate on the model's soft codon distribution via a
  codon → nucleotide decomposition; the mean-field approximation can under-count repeats
  when the model is uncertain (see the docstring in `bio_losses.py`).
- Checkpoints, datasets, and result files are git-ignored; only code is tracked.

## Acknowledgements

Built on [CodonTransformer](https://github.com/Adibvafa/CodonTransformer) (Fallahpour et al.)
and [DnaChisel](https://github.com/Edinburgh-Genome-Foundry/DnaChisel) (Zulko).
