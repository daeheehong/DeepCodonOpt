#!/usr/bin/env python3
"""Stage 1 -- E. coli baseline fine-tuning.

Vanilla masked-LM fine-tuning of CodonTransformer on the host organism (E. coli general),
producing the checkpoint that Stages 2 and 3 build on. This is the standard MLM objective
with no constraint terms.

The two Stage 1 experiments are run by sweeping the CLI args, not by editing code:

  * learning rate / epochs : run with different --learning_rate and --max_epochs
        for lr in 3e-5 5e-5; do
          python baseline.py --train_data_path data/train.jsonl \
              --val_data_path data/val.jsonl --test_data_path data/test.jsonl \
              --learning_rate $lr --max_epochs 15 --checkpoint_dir out/lr${lr}
        done

  * data-size scaling : point --train_data_path at the nested subsets from build_dataset.py
        for n in 500 1000 2000 5000 train_all; do
          python baseline.py --train_data_path data/sizes/train_${n}.jsonl \
              --val_data_path data/val.jsonl --test_data_path data/test.jsonl \
              --checkpoint_dir out/size_${n}
        done

Single GPU:  python baseline.py ...
Multi GPU :  torchrun --nproc_per_node=2 baseline.py ...   (DDP is automatic)

The best (lowest val-loss) checkpoint is saved to <checkpoint_dir>/<checkpoint_filename>.
"""

import argparse

from codon_utils import add_training_args, run_training


def main():
    p = argparse.ArgumentParser(description="Stage 1: E. coli baseline MLM fine-tuning")
    add_training_args(p)
    run_training(p.parse_args())


if __name__ == "__main__":
    main()
