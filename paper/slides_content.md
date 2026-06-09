# 3-minute talk — slide content (6 slides)

Paste into Claude Design. Each slide ≈ 30 s. Keep one visual + one message per slide.
Notation: **headline** / bullets (what's on the slide) / *visual* / 🎤 speaker note.

---

## Slide 1 — Title & the question (≈25 s)

**Where should biological constraints enter a neural codon optimizer?**

- CodonTransformer: SOTA deep model that picks codons for high expression
- But real DNA synthesis has hard rules (no repeats, GC bounds, no restriction sites)
- *Question:* add the rules at **training time** or **inference time**?

*visual:* one-line pipeline — Protein → CodonTransformer → DNA → ❓constraints

🎤 "Neural codon optimizers write great DNA for expression, but ignore whether it can actually be manufactured. We ask where to put the manufacturing rules."

---

## Slide 2 — Diagnosis: great codons, terrible to synthesize (≈35 s)

**The model is over-regular**

- Scored 3,000 *E. coli* genes with the **IDT complexity score** (industry synthesis-difficulty index)
- Baseline model: **CAI 0.89 > natural 0.74** (codon usage is excellent) …
- … but **IDT complexity 4.9 vs 1.9** (much harder to synthesize)
- Breakdown → one culprit: **Overall Repeat 24% → 61%**

*visual:* Table 1 (real / baseline / pretrained: complexity, CAI, Overall Repeat) — highlight the 4.9 vs 1.9 and 61%

🎤 "Greedy decoding always picks each amino acid's favorite codon — maximizing CAI but producing repetitive DNA that IDT flags as hard to synthesize."

---

## Slide 3 — Method A: add a constraint loss → it backfires (≈35 s)

**Training-time differentiable constraint loss**

- $\mathcal L = L_\text{MLM} + \lambda\cdot(\text{repeat/GC penalty on the model's soft distribution})$
- Result: complexity **4.9 → 9.3**, CAI **0.89 → 0.43**, Overall Repeat **61% → 80%**
- Breakdown: GC issues → 0%, but **repeats explode** ⇒ it traded one issue for a worse one

*visual:* Table 2 / bar chart (real vs baseline vs Method A) — repeats up, CAI down

🎤 "Adding a differentiable constraint loss made everything worse. Why? The penalty scores the model's probabilities, not the sequence it writes."

---

## Slide 4 — Why A fails: surrogate misalignment (≈25 s)

**The loss measures the wrong thing**

- Penalty acts on the **probability distribution**, not the **emitted sequence**
- Model satisfies it by **flattening** its distribution (→ rare codons) at ~zero MLM cost
- So $\lambda$ doesn't matter, and the decoded DNA gets *worse*

*visual:* cartoon — "soft distribution" (penalty = 0) vs "argmax sequence" (still repetitive)

🎤 "It's like lowering your test average by leaving answers blank instead of getting them right. The soft objective and the real objective are misaligned."

---

## Slide 5 — Method B: constrained temperature decoding → it works (≈40 s)

**Inference-time: temperature sampling + constraint masking**

- **Temperature** $P_T(c)\propto e^{z_c/T}$: diversifies codons → breaks repeats
- **Masking**: delete codons that form restriction sites / homopolymers before sampling
- Sweeping $T$ → clean **Pareto**; knee at **T ≈ 0.7–0.8 beats natural on BOTH axes**
  (complexity 1.6 < 1.9, CAI 0.78 > 0.74; all four repeat/GC issues ≤ natural)

*visual:* Fig — CAI vs IDT-complexity Pareto over temperature, ★ = natural (curve passes above/left of ★)

🎤 "Change only the decoding: sample at temperature to break repeats, mask out illegal codons. This operates on the real sequence — so it actually works, and beats natural genes."

---

## Slide 6 — Takeaway (≈20 s)

**Discrete constraints belong on the discrete output, at decode time**

- Training-time soft loss → gameable, backfires
- Inference-time masking + temperature → exact + effective, surpasses natural
- Future: look-ahead decoding for long-range repeats

*visual:* one-line summary box; GitHub + Zenodo links

🎤 "One principle: don't relax discrete biological constraints into a training loss — enforce them on the actual sequence at decode time."

---

### Design notes
- Palette: natural = blue, baseline = red, Method B = purple (matches the paper figures).
- Slides 3–4 can merge into one if time is tight (→ 5 slides).
- Reuse `report_figs/fig5_pareto.png` for Slide 5 and the Table-1 / Table-2 numbers for Slides 2–3.
