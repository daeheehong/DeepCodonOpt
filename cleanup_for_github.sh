#!/usr/bin/env bash
# Prepare a clean DeepCodonOpt tree for GitHub. Only code + paper are tracked; large
# outputs/checkpoints/legacy stay local (gitignored) and go to Zenodo instead.
set -e
cd "$(dirname "$0")"

echo "== removing caches =="
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '.DS_Store' -delete 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

# the 13 files that should be tracked (everything else is gitignored)
KEEP="codon_utils.py bio_losses.py idt_client.py build_dataset.py baseline.py analyze.py \
lossbio.py decodebio.py analyze_decodebio.py make_report_figs.py ecoli_codon_usage_table.csv \
README.md requirements.txt .gitignore DATA_UPLOAD.md paper/main.tex paper/reference.bib \
paper/slides_content.md"

echo "== essential files present? =="
for f in $KEEP; do [ -e "$f" ] && echo "  ok  $f" || echo "  MISSING  $f"; done

git init -q 2>/dev/null || true
git add -A
echo ""
echo "== what git will commit (should be only code + paper) =="
git status --short
echo ""
echo "If _result/, _legacy/, data/, *.ckpt, *.png, output/ appear above, check .gitignore."
echo "Then:  git commit -m 'DeepCodonOpt: code + paper'  &&  git push"
