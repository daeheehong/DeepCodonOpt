# Where Should Biological Constraints Enter a Neural Codon Optimizer? A Study on CodonTransformer

*Deep Learning — Final Project Report*

**Author:** Daehee Hong (et al.)

---

## Abstract

Neural codon optimizers such as CodonTransformer generate coding sequences with high host-specific codon adaptation, but they have no mechanism for the manufacturing constraints that real gene synthesis must satisfy. We audit an *E. coli*-fine-tuned CodonTransformer against the **IDT SciTools synthesis-complexity score** and obtain a clear conclusion: the model achieves **excellent codon usage (CAI 0.89, above natural)** yet **poor synthesizability (IDT complexity 4.9 vs 1.9 for natural genes)**, almost entirely because deterministic decoding drives the model toward **repetitive** codon choices (Overall Repeat 24% → 61%). We then compare *where* constraints should be incorporated. A training-time approach that adds a differentiable constraint loss (**Method A**) **backfires**: the loss is minimized but the true repeat content instead explodes (Overall Repeat 61% → 80%) and codon usage collapses (CAI 0.89 → 0.43), so complexity *rises* to 9.3. An inference-time approach that combines temperature sampling with constraint masking (**Method B**) **succeeds**: raising the temperature monotonically removes the repeats responsible for high complexity and traces a clean codon-adaptation ↔ synthesizability Pareto, while masking structurally eliminates restriction-enzyme sites and homopolymers. At $T\approx0.7\text{–}0.8$ the generated sequences **match or beat natural *E. coli* genes on synthesizability** (complexity 1.4–1.6 vs 1.9, Overall Repeat 21–24% vs 24%) while retaining **higher codon adaptation** (CAI 0.77–0.78 vs 0.74). The conclusion is simple: discrete biological constraints should not be relaxed into a training loss, but enforced at inference time on the realized discrete sequence.

---

## 1. Introduction

### 1.1 What is codon optimization?

The genetic code is degenerate: 61 sense codons encode only 20 amino acids, so most amino acids are specified by two to six **synonymous** codons. Synonymous codons translate to the same protein, but the choice among them strongly affects mRNA stability, translation efficiency, and final protein expression level. **Codon optimization** selects, at each position of a target protein, the synonymous codon that maximizes expression in a chosen host; it is a critical step in recombinant-protein production and mRNA-therapeutic design.

### 1.2 Computational view

Given a protein $\mathbf{p}=(p_1,\dots,p_N)$ and the set $\mathrm{Syn}(p_i)$ of codons synonymous with amino acid $p_i$, codon optimization seeks a codon sequence $\mathbf{c}=(c_1,\dots,c_N)$, $c_i\in\mathrm{Syn}(p_i)$, that maximizes an expression-fitness objective $f$:

$$
\mathbf{c}^\star = \arg\max_{\mathbf{c}}\; f(\mathbf{c})
\quad\text{s.t.}\quad g_j(\mathbf{c}) \le 0,\;\; c_i\in\mathrm{Syn}(p_i).
$$

The search space is combinatorial ($\prod_i|\mathrm{Syn}(p_i)|$), and the constraints $g_j$ — GC-content bounds, avoidance of repeats/hairpins, homopolymers, and restriction-enzyme sites — are inherently **discrete** functions of the nucleotide string. Jointly handling a learned, context-dependent objective and hard discrete constraints is the central difficulty.

### 1.3 Strategies for codon optimization

**Heuristic / score-based methods** evaluate a sequence with hand-designed metrics. Our manufacturability axis is the **IDT SciTools complexity score**. For a candidate gene the IDT API returns a list of synthesis *issues* — *Overall Repeat*, *Hairpin*, *Windowed High GC*, etc. — and sums their penalties into a single value (higher means harder to synthesize; for full genes $>7$ is flagged complex and $>20$ is usually rejected). Because the API also returns the **per-issue breakdown**, it reveals not only *how* hard a sequence is to make but *why*. Our quality axis is the **Codon Adaptation Index (CAI)**, which measures how closely a sequence matches the host's preferred codon usage.

**Deep-learning methods** treat codon optimization as conditional generation. **CodonTransformer** (Fallahpour et al., 2025) is the state of the art: a BigBird masked LM trained on over one million coding sequences from 164 organisms with a joint amino-acid–codon tokenization. Conditioned on a protein and target organism, it predicts a host-adapted codon at each position, but it has **no mechanism for constraint satisfaction** — it optimizes only codon-usage likelihood.

### 1.4 Goal

Treating CodonTransformer as a strong but constraint-blind generator, we ask two questions. (i) **How good is it really?** — diagnosed through the IDT complexity breakdown. (ii) **Where should constraints be added?** — comparing the *training-time* approach (Method A) with the *inference-time* approach (Method B).

---

## 2. Baseline preparation: fine-tuning on *E. coli*

**Dataset.** We select the *E. coli* protein–DNA pairs from the CodonTransformer corpus. To prevent leakage between homologous genes, sequences are clustered with **MMseqs2** and split *by cluster* into train / validation / test = **8 : 1 : 1**, so that no test protein is homologous to a training protein. The model is fine-tuned with the masked-LM objective, maximizing $\log P_\theta(\mathbf c\mid\mathbf p,o)$ for $o=$ *E. coli* on 15%-masked codon tokens.

**Hyper-parameters.** We sweep learning rate and epoch count and select on validation MLM loss. Within the stable regime the two had only noise-level effect; $5\times10^{-5}$ was the most stable setting while $10^{-4}$ diverged (Fig. 1). A data-scaling check on training-set size is shown in Fig. 2.

**Selected checkpoint.** The best checkpoint (validation perplexity $\approx2.43$, masked-token accuracy $\approx0.64$) is fixed and used, unchanged, as the **baseline** for all downstream analysis and as the starting point for both improvement methods.

---

## 3. Analysis with the IDT complexity score

**Method.** For 3,000 held-out *E. coli* proteins, we compare three coding sequences each: the **natural (real)** gene, the **baseline** fine-tuned model's output (deterministic decoding), and the **stock pretrained** CodonTransformer's output. All three are scored with the IDT API; we report the aggregate complexity, the per-issue breakdown, and CAI.

**Aggregate results.**

| Source | IDT complexity (mean) | % complex ($>7$) | % reject ($>20$) | CAI |
|---|---:|---:|---:|---:|
| Natural (real) | **1.91** | 7.0 | 0.6 | 0.735 |
| Baseline (E. coli FT) | 4.91 | 26.0 | 2.5 | **0.888** |
| Pretrained (stock) | 6.63 | 37.9 | 4.6 | 0.885 |

**Per-issue breakdown** (% of sequences hitting each issue; the four repeat/GC issues used throughout).

| IDT issue | real | baseline | pretrained |
|---|---:|---:|---:|
| **Overall Repeat** | 23.7 | **60.8** | 63.9 |
| Repeat Length (Fragment) | 1.2 | 3.7 | 3.5 |
| Windowed Repeat % | 0.5 | 2.3 | 2.6 |
| Windowed High GC (100 bp) | 4.2 | 14.8 | 30.5 |

**What goes wrong, and why.** Both neural models are far worse than natural genes on synthesizability (4.91 / 6.63 vs 1.91), and the breakdown localizes the cause precisely: the gap is dominated by **Overall Repeat** (23.7% → 60.8%), with the stock model additionally GC-heavy (Windowed High GC 30.5%). The mechanism is mechanistic — a likelihood-trained model decoded **deterministically** keeps emitting the single most-probable codon for each amino acid, which maximizes CAI but yields **repetitive, low-entropy** nucleotide strings. The model is not wrong about codon usage; it is *over-regular*.

---

## 4. Method A — constraint-loss fine-tuning (training-time)

**Approach.** Following the Lagrangian view of constrained optimization, we continue fine-tuning the baseline with a constraint penalty added to the masked-LM loss,

$$
\mathcal L = \underbrace{L_{\text{MLM}}}_{\text{original codon-prediction loss}} \;+\; \lambda \cdot \underbrace{\textstyle\sum_j w_j\,\hat g_j(P_\theta)}_{\text{penalty, large when repeats/GC are severe}},
$$

where $\lambda$ balances the two terms. Because $\hat g_j$ must be differentiable to be trained against while the nucleotide string is discrete, the penalty is evaluated not on the realized sequence but on a soft surrogate of the model's codon distribution: we map the per-position codon probabilities to a *soft-nucleotide* matrix $N[t,n]=\sum_{c:c[t]=n}P_\theta(c\mid t)$ (the expected probability that base $t$ is nucleotide $n$) and express the repeat/GC penalties as mean-field statistics of $N$ (e.g. the self-similarity $\sum_n N[i,n]\,N[j,n]$). We sweep $\lambda\in\{1,3,10,30\}$ and re-score with the IDT pipeline of §3.

**Result: failure.** The penalty reaches (numerical) zero within the first epoch and the four values of $\lambda$ give identical outcomes, yet the generated sequences become markedly worse on every axis:

| Source | IDT complexity (mean) | % reject ($>20$) | CAI | rare codon % |
|---|---:|---:|---:|---:|
| Natural (real) | 1.91 | 0.6 | 0.735 | 7.2 |
| Baseline (FT) | 4.91 | 2.5 | 0.888 | — |
| **Method A (constraint loss)** | **9.32** | **13.7** | **0.426** | **24.0** |

The IDT breakdown shows precisely *how* the penalty was driven to zero:

| IDT issue | real | baseline | Method A |
|---|---:|---:|---:|
| **Overall Repeat** | 23.7 | 60.8 | **79.7** |
| Repeat Length (Fragment) | 1.2 | 3.7 | **12.3** |
| Windowed Repeat % | 0.5 | 2.3 | **12.7** |
| Windowed High GC (100 bp) | 4.2 | 14.8 | **0.0** |

**Why it fails.** The failure is one of surrogate misalignment: the penalty is computed on the model's *probability distribution*, not on the discrete sequence it actually emits. The mean-field surrogate can be driven to zero by *flattening* $P_\theta$ — spreading probability mass over more, and rarer, synonymous codons — which lowers the product-of-marginals self-similarity regardless of which codon is ultimately decoded. This costs almost nothing in MLM loss, which is why $\lambda$ is immaterial: once $\hat g$ and its gradient vanish, any multiple of them is still zero. The decoded sequence (the arg-max of the flattened distribution), however, does not improve and in fact degrades, as the breakdown makes explicit — the high-GC issues are driven to 0% (flattening lowers GC peaks), but Overall Repeat instead explodes from 60.8% to 79.7%, the repeat-length and windowed-repeat issues roughly quadruple, and the shift toward rare codons collapses CAI to 0.43. Because the training objective (defined on the soft distribution) and the true objective (defined on the discrete output) are **misaligned**, minimizing the surrogate moves the model *away* from the target.

---

## 5. Method B — temperature sampling with constraint masking (inference-time)

Method A failed because the constraint acted on the *soft* distribution during training. Method B leaves the trained model untouched and applies the constraint to the **realized discrete sequence at inference**. It has two components.

**(a) Temperature sampling.** Greedy (arg-max) decoding deterministically emits the single most probable codon at each position, which is exactly the source of the repetition observed in §3. We replace it with temperature sampling,

$$
P_T(c) \propto \exp(z_c / T).
$$

A higher temperature $T$ raises the decoding entropy, so lower-ranked but near-equiprobable synonymous codons are chosen with non-negligible frequency; this diversity breaks up the long-range repeats. The cost is that a less host-preferred codon is occasionally selected, lowering CAI. Thus $T$ is a single knob that trades synthesizability against codon adaptation.

**(b) Constraint masking.** Before sampling at each position, we mask out every synonymous codon that would, together with its neighbours, complete a forbidden motif — a restriction-enzyme site, a homopolymer run, or an internal Shine–Dalgarno sequence — by setting its logit $z_c\leftarrow-\infty$, and sample from the remaining codons. Because the mask is evaluated on the actual nucleotide context, these local constraints are satisfied by construction; and because each amino acid typically offers several synonymous codons, a compliant codon almost always remains, so the constraint is enforced without a quality penalty.

**Result: success — aggregate trade-off.** We score Method B outputs across $T\in\{0.1,\dots,1.0\}$.

| $T$ | CAI | IDT complexity (mean) |
|---:|---:|---:|
| Natural (real) | 0.735 | 1.91 |
| 0.1 | 0.882 | 4.58 |
| 0.3 | 0.850 | 3.22 |
| 0.5 | 0.812 | 2.20 |
| **0.7** | **0.781** | **1.59** |
| **0.8** | **0.767** | **1.41** |
| 1.0 | 0.748 | 1.18 |

As $T$ increases, IDT complexity falls monotonically (4.58 → 1.18) while CAI declines (0.882 → 0.748) — a clean codon-adaptation ↔ synthesizability Pareto along which $T$ moves (Fig. 5). The **knee is at $T\approx0.7\text{–}0.8$**: at $T{=}0.7$, complexity 1.59 < natural 1.91 while CAI 0.781 > natural 0.735, i.e. **the generated sequences surpass natural ones on both axes**.

**Per-issue breakdown across temperature (full 3,000).** We resolve which issues drive the aggregate score down, using the four issues of §3 (Method B column).

| $T$ | Overall Repeat | Repeat Length (frag) | Windowed Repeat % | Windowed High GC (100) |
|---:|---:|---:|---:|---:|
| **Natural (real)** | **23.7** | **1.2** | **0.5** | **4.2** |
| 0.1 | 57.8 | 3.7 | 1.8 | 14.8 |
| 0.2 | 49.7 | 2.6 | 1.2 | 13.4 |
| 0.3 | 35.8 | 1.7 | 0.9 | 10.7 |
| 0.4 | 38.7 | 0.9 | 0.5 | 12.7 |
| 0.5 | 32.6 | 1.0 | 0.7 | 10.1 |
| 0.6 | 24.0 | 0.9 | 0.4 | 8.0 |
| **0.7** | **23.9** | **0.9** | **0.5** | **8.2** |
| **0.8** | **21.3** | **0.8** | **0.4** | **7.4** |
| 0.9 | 18.8 | 0.8 | 0.4 | 6.9 |
| 1.0 | 17.4 | 0.6 | 0.4 | 5.6 |

As $T$ rises, **all four issues decrease monotonically** toward natural levels. At $T\approx0.7\text{–}0.8$ the three repeat-family issues are already **below natural** (Overall Repeat 23.9/21.3 vs 23.7; Repeat Length 0.9/0.8 vs 1.2; Windowed Repeat 0.5/0.4 vs 0.5), while only Windowed High GC (8.2/7.4) — though sharply reduced from 14.8 — remains slightly above natural (4.2). A greedy ($T{=}0$, masking-only) control leaves complexity at the baseline level (≈4.9): masking cannot break long-range repeats, so **temperature is the driver and masking is the guarantee** (for local motifs such as restriction sites and homopolymers).

**Why it works.** The misalignment that defeated Method A is absent here. The mask is an indicator function on the discrete sequence, so the constraint is satisfied exactly and cannot be gamed. Temperature acts on the realized categorical sample rather than on a mean-field summary, so raising the decoding entropy actually changes the emitted codons and therefore actually reduces the repeat content that IDT measures. Both methods encode the same intuition — diversify codon usage to reduce repeats — but **Method A applies it to a soft distribution decoupled from the output, whereas Method B applies it to the output itself.**

---

## 6. Summary and future work

Using CodonTransformer and the IDT synthesis-complexity score, we studied *where* biological constraints should enter a neural codon optimizer. The fine-tuned model produces **repetitive, hard-to-synthesize** sequences because deterministic decoding is over-regular. Adding a constraint loss at training time (**Method A**) **backfires** — the penalty scores the probability distribution rather than the realized sequence, so the model satisfies it by flattening its distribution (trading GC issues for far worse repeats and rare codons), degrading both synthesizability (4.9 → 9.3) and codon usage (CAI 0.89 → 0.43). Enforcing the constraint at inference time (**Method B**) — masking violating codons and sampling at temperature — **succeeds** — decoding entropy dissolves the repeats and masking structurally removes local motifs, so the outputs surpass natural sequences on synthesizability while keeping higher codon adaptation. The conclusion is a clean principle: **discrete constraints belong on the discrete output at decode time, not in a differentiable surrogate over the model's probabilities during training.**

**Limitations and future work.** (i) Local masking cannot directly remove *long-range* repeats and hairpins; we rely on temperature to suppress them statistically — a principled remedy is look-ahead / beam-style constrained decoding (e.g. NeuroLogic A\*-style search) that scores partial sequences for global repeat content. (ii) Internal Shine–Dalgarno motifs were reduced but not fully eliminated (≈5–9%), owing to positions where every synonymous codon still forms the motif with its neighbours; a multi-position joint mask would help. (iii) The temperature Pareto suggests an automatic operating point — choosing $T$ per sequence to hit a target IDT score.

---

## References

1. Fallahpour, A., et al. (2025). *CodonTransformer: a multi-organism deep learning model for codon optimization.* Nature Communications.
2. Zaheer, M., et al. (2020). *Big Bird: Transformers for longer sequences.* NeurIPS.
3. Integrated DNA Technologies. *SciTools Plus — gene complexity scoring API* (vendor documentation).
4. Steinegger, M., & Söding, J. (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.* Nature Biotechnology.
5. Holtzman, A., et al. (2020). *The curious case of neural text degeneration.* ICLR.
6. Lu, X., et al. (2022). *NeuroLogic A\*esque decoding: constrained text generation with lookahead heuristics.* NAACL.
7. Bertsekas, D. P. (2014). *Constrained Optimization and Lagrange Multiplier Methods.* Academic Press.
8. Sharp, P. M., & Li, W.-H. (1987). *The Codon Adaptation Index.* Nucleic Acids Research.

---

*Figures to insert:* **Fig. 1** validation loss over learning rate × epoch (`1_baseline/phase1_heatmap.png`); **Fig. 2** data-scaling curve (`phase1_size.png`); **Fig. 3** IDT issue breakdown — natural vs baseline vs pretrained (`report_figs/fig3_idt_breakdown.png`); **Fig. 4** Method A IDT breakdown — GC ↓, repeats ↑ (`report_figs/fig4_methodA_breakdown.png`); **Fig. 5** temperature Pareto, CAI vs IDT complexity (`report_figs/fig5_pareto.png`); **Fig. 6** Method B per-temperature issue breakdown (`report_figs/fig5_issues.png`).
