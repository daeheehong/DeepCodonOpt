# Archiving the model & sequences (Zenodo)

GitHub is for code; large artifacts (the checkpoint, generated sequences) go to **Zenodo**,
which mints a permanent **DOI** you cite from the paper and README. Zenodo accepts up to
50 GB per record, so the ~358 MB checkpoint is fine.

## What to upload

1. **Baseline checkpoint** — `ep50_lr5e-05.ckpt` (the E. coli fine-tuned model used everywhere).
2. **Method B "best" sequences** — the constrained outputs at the operating temperature.
   Recommended: package T=0.7 and T=0.8 together (the knee). From each run's `sequences.csv`
   take the `constrained` column. Provide both CSV and FASTA so reviewers can use them directly.
3. A short **`DATA_README.txt`** describing each file (below).

Suggested bundle:

```
deepcodonopt-artifacts/
├── ep50_lr5e-05.ckpt              # baseline fine-tuned checkpoint (raw state_dict)
├── methodB_T0.7_sequences.csv     # real / unconstrained / constrained DNA, 3000 genes
├── methodB_T0.8_sequences.csv
├── methodB_best.fasta             # constrained sequences at T=0.7 (or your chosen T)
└── DATA_README.txt
```

Make the FASTA from a `sequences.csv` (no extra deps):

```bash
python - <<'PY'
import csv
rows = list(csv.DictReader(open("methodB_T0.7_sequences.csv")))
with open("methodB_best.fasta", "w") as f:
    for r in rows:
        f.write(f">seq{r['seq_id']}_T0.7_constrained\n{r['constrained_dna']}\n")
print("wrote", len(rows), "sequences")
PY
```

Then zip the folder: `zip -r deepcodonopt-artifacts.zip deepcodonopt-artifacts/`
(or upload the files individually — Zenodo keeps them separate either way).

## Uploading (web, ~5 minutes)

1. Sign in at **https://zenodo.org** (log in with GitHub or ORCID).
2. **New upload** → drag the files (or the zip).
3. Fill metadata:
   - **Upload type:** Dataset (or "Model" if offered).
   - **Title:** *DeepCodonOpt: E. coli fine-tuned CodonTransformer checkpoint and constrained-decoding sequences*.
   - **Authors:** all team members (add ORCIDs if available).
   - **Description:** 2–3 lines + link to the GitHub repo and the paper.
   - **License:** e.g. CC-BY-4.0 (data) / MIT (if you prefer).
   - **Related identifiers:** add the GitHub repo URL as *"is supplemented by"*.
4. **Reserve DOI** (button in the metadata panel) — this gives you the DOI string *before*
   publishing, so you can paste it into the paper/README first.
5. **Publish.** The DOI (e.g. `10.5281/zenodo.XXXXXXX`) is now permanent.

## Link it from the paper & repo

- **Paper (`paper/main.tex`)** — replace the placeholder in the Conclusion:
  `\url{https://doi.org/<zenodo-DOI>}`.
- **README** — add an "Artifacts" line with the DOI badge:
  `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)`

## Alternative: auto-DOI for the GitHub repo itself

To also get a DOI for the *code*: in Zenodo go to **Settings → GitHub**, flip the switch
on your `DeepCodonOpt` repo, then cut a **Release** on GitHub. Zenodo snapshots the release
and mints a DOI automatically. (Use this for code; use the manual upload above for the large
checkpoint, which is too big to live in the git release.)

---

### DATA_README.txt (template)

```
DeepCodonOpt artifacts
======================
Paper: Where Should Biological Constraints Enter a Neural Codon Optimizer?
Code:  https://github.com/<user>/DeepCodonOpt

ep50_lr5e-05.ckpt
  CodonTransformer fine-tuned on E. coli (organism id 51), raw PyTorch state_dict.
  Load: codon_utils.load_finetuned_model("ep50_lr5e-05.ckpt", device).

methodB_T0.7_sequences.csv, methodB_T0.8_sequences.csv
  Columns: seq_id, real_dna, unconstrained_dna, constrained_dna.
  3000 held-out E. coli proteins. "constrained" = Method B (temperature T + masking).

methodB_best.fasta
  Method B constrained DNA at T=0.7 (recommended operating point).
```
