#!/usr/bin/env python3
"""Stage 3 -- constraint-aware fine-tuning with differentiable biological losses.

Continue fine-tuning the Stage 1 baseline while adding a manufacturability penalty to the
MLM loss:  L_total = L_MLM + lambda * (w_gc L_gc + w_repeat L_repeat + w_hairpin L_hairpin),
where each term is a differentiable surrogate of an IDT complexity issue (see bio_losses.py).

How the constraint loss is applied each step
    The MLM forward runs as usual on the 15%-masked batch. For the constraint term we
    rebuild the *unmasked* codons from the labels, map them to the amino-acid query tokens
    ("{aa}_unk"), and run a second forward -- this yields the model's full generation
    distribution P(codon | amino acid, context), from which the soft-nucleotide losses are
    computed. So the penalty acts on the sequence the model would actually generate.

Per-term analysis
    The mean of each component (gc / repeat / hairpin) is logged every epoch, so you can see
    how well each term is being driven down. After training, run analyze.py on the resulting
    checkpoint to confirm the *true* IDT score and DnaChisel violations actually drop -- the
    losses are surrogates and must be validated.

Usage (start from the baseline checkpoint)
    python lossbio.py \
        --train_data_path data/train.jsonl --val_data_path data/val.jsonl \
        --test_data_path data/test.jsonl \
        --init_checkpoint out/size_all/best.pt \
        --checkpoint_dir out/lossbio_full \
        --lambda_bio 1.0 --lambda_gc 1.0 --lambda_repeat 1.0 --lambda_hairpin 0.5

Ablate a single term by setting its weight (and the others) accordingly, e.g. repeat only:
    ... --lambda_gc 0 --lambda_repeat 1 --lambda_hairpin 0
"""

import argparse
import copy

import torch

from bio_losses import (DEFAULT_CFG, build_codon_tables, manufacturability_loss,
                        query_token_range)
from codon_utils import add_training_args, run_training


def make_cfg(args):
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg["weights"] = dict(gc=args.lambda_gc, repeat=args.lambda_repeat, hairpin=args.lambda_hairpin)
    cfg["gc_window"] = args.gc_window
    cfg["gc_thr"] = args.gc_thr
    cfg["repeat"]["thr"] = args.repeat_thr
    cfg["hairpin"]["thr"] = args.hairpin_thr
    return cfg


def make_aux_factory(args):
    """Build the aux-loss factory consumed by codon_utils.run_training."""
    cfg = make_cfg(args)
    lam = args.lambda_bio

    def factory(core, tokenizer, device):
        codon_cols, codon_nuc, codon_to_query = build_codon_tables(tokenizer, device)
        lo, hi = query_token_range(tokenizer)

        def aux_loss_fn(model, batch):
            # Reconstruct the true codons (label where masked, else the input token),
            # then map each codon position to its "{aa}_unk" query token.
            orig = torch.where(batch["labels"] != -100, batch["labels"], batch["input_ids"])
            query_input = codon_to_query[orig]
            out = model(input_ids=query_input,
                        token_type_ids=batch["token_type_ids"],
                        attention_mask=batch["attention_mask"])
            total, comps = manufacturability_loss(
                out.logits, query_input, batch["attention_mask"],
                codon_cols, codon_nuc, lo, hi, cfg)
            return lam * total, comps

        return aux_loss_fn

    return factory


def main():
    p = argparse.ArgumentParser(description="Stage 3: constraint-aware fine-tuning (Method B)")
    add_training_args(p)
    p.add_argument("--lambda_bio", type=float, default=1.0, help="global weight on the constraint loss")
    p.add_argument("--lambda_gc", type=float, default=1.0)
    p.add_argument("--lambda_repeat", type=float, default=1.0)
    p.add_argument("--lambda_hairpin", type=float, default=0.5)
    p.add_argument("--gc_thr", type=float, default=0.60,
                   help="penalise GC fraction above this (windowed + overall)")
    p.add_argument("--gc_window", type=int, default=100, help="window (bp) for the windowed-GC penalty")
    p.add_argument("--repeat_thr", type=float, default=0.4, help="per-window agreement threshold")
    p.add_argument("--hairpin_thr", type=float, default=0.4)
    args = p.parse_args()
    run_training(args, aux_loss_factory=make_aux_factory(args))


if __name__ == "__main__":
    main()
