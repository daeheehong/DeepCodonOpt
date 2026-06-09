"""Differentiable manufacturability losses (Stage 3 / "Method B").

Idea (physics-informed style)
    From the model's per-codon softmax distribution we build a *soft nucleotide*
    matrix ``N[T, 4]`` where ``N[t, n]`` is the expected probability that base ``t`` is
    nucleotide ``n`` (A/C/G/T); each row sums to 1. ``N`` is differentiable w.r.t. the
    model parameters, so any constraint score written as a function of ``N`` backprops.

    The IDT "complexity" issues we target are all window statistics of ``N``:
        - high GC      -> windowed mean GC over threshold            -> gc_window_loss
        - direct repeat-> self-similarity of t and t+delta (forward) -> repeat_loss
        - hairpin      -> self-similarity of i and j (reverse comp.) -> hairpin_loss
    Repeats and hairpins are the forward / reverse-complement versions of the same part.

Honest limitation (verify after training!)
    ``N`` scores matches from the product of per-position marginals (a mean-field
    approximation). It is accurate when the model is confident (usually true) but can
    under-count real repeats when the model is uncertain. These losses are therefore a
    *differentiable surrogate* for IDT -- always confirm that the true IDT score drops
    after training (see analyze.py).

CodonTransformer token convention (vocab = 90)
    0-4   special tokens        [UNK][CLS][SEP][PAD][MASK]
    5-25  amino-acid query tokens "{aa}_unk"   (codon undecided = generation input)
    26-89 codon tokens            "{aa}_{codon}"  (e.g. "k_aaa")
    The mapping is read from ``tokenizer.get_vocab()`` at runtime (no hard-coded indices).
"""

import torch
import torch.nn.functional as F

NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP_PERM = [3, 2, 1, 0]   # channel complement: A(0)<->T(3), C(1)<->G(2)


# ============================================================================
#  0. token tables (build once from the tokenizer)
# ============================================================================
def build_codon_tables(tokenizer, device="cpu"):
    """Returns:
        codon_cols     LongTensor[64]      vocab indices of codon tokens
        codon_nuc      FloatTensor[64,3,4] one-hot of each codon's 3 nucleotides
        codon_to_query LongTensor[V]       codon token -> matching "{aa}_unk" query token
                                           (identity for non-codon indices)
    ``codon_to_query`` lets a fine-tuning batch (which holds codon tokens) be turned into
    the amino-acid query input used for the constraint forward pass.
    """
    vocab = tokenizer.get_vocab()                # {token_str: idx}
    V = len(vocab)
    codon_to_query = torch.arange(V, dtype=torch.long)
    cols, mats = [], []
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        parts = tok.split("_")
        codon = parts[-1].upper().replace("U", "T")
        if len(parts) == 2 and len(codon) == 3 and all(c in NUC for c in codon):
            m = torch.zeros(3, 4)
            for o, c in enumerate(codon):
                m[o, NUC[c]] = 1.0
            cols.append(idx)
            mats.append(m)
            q = f"{parts[0]}_unk"                 # the same amino acid's query token
            if q in vocab:
                codon_to_query[idx] = vocab[q]
    codon_cols = torch.tensor(cols, dtype=torch.long, device=device)
    codon_nuc = torch.stack(mats).to(device)     # [64, 3, 4]
    return codon_cols, codon_nuc, codon_to_query.to(device)


def query_token_range(tokenizer):
    """Index range of the "{aa}_unk" query tokens (normally 5..25), used to locate codon positions."""
    vocab = tokenizer.get_vocab()
    qidx = [i for t, i in vocab.items() if t.endswith("_unk")]
    return (min(qidx), max(qidx)) if qidx else (5, 25)


# ============================================================================
#  1. logits -> soft nucleotides (single sequence)
# ============================================================================
def soft_nuc_one(logits_1d, is_codon_pos, codon_cols, codon_nuc):
    """``logits_1d`` [L, V] (from an amino-acid query forward), ``is_codon_pos`` [L] bool.
    Returns N [3*Tcod, 4] (or [0, 4] if there are no codon positions)."""
    pos = is_codon_pos.nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return logits_1d.new_zeros((0, 4))
    lg = logits_1d[pos][:, codon_cols]           # [Tcod, 64] codon columns only
    p = F.softmax(lg, dim=-1)                     # distribution over synonymous codons
    nuc = torch.einsum("tc,cof->tof", p, codon_nuc)   # [Tcod, 3, 4]
    return nuc.reshape(-1, 4)                     # [3*Tcod, 4]


# ============================================================================
#  2. individual constraint losses (input N: [T, 4], rows sum to 1)
# ============================================================================
def gc_window_loss(nuc, win, hi):
    """Squared penalty on the amount by which the mean GC of a length-``win`` window exceeds ``hi``."""
    T = nuc.shape[0]
    if T < win:
        return nuc.sum() * 0.0
    gc = (nuc[:, 1] + nuc[:, 2]).view(1, 1, -1)          # G+C probability, [1,1,T]
    m = F.avg_pool1d(gc, win, stride=1).view(-1)         # windowed mean GC (avg_pool avoids cuDNN)
    return F.relu(m - hi).pow(2).mean()


def repeat_loss(nuc, kmer=8, min_lag=3, max_lag=80, thr=0.6):
    """Forward repeat: average base agreement of positions t and t+delta over a kmer window;
    penalise windows whose agreement exceeds ``thr`` (a near-identical k-mer pair).
    Differentiable surrogate of IDT 'Overall Repeat'."""
    T = nuc.shape[0]
    if T < kmer + min_lag:
        return nuc.sum() * 0.0
    total = nuc.sum() * 0.0
    for d in range(min_lag, min(max_lag, T - 1) + 1):
        m = (nuc[:-d] * nuc[d:]).sum(-1)                 # per-position agreement [T-d] in [0,1]
        if m.numel() < kmer:
            continue
        win_id = F.avg_pool1d(m.view(1, 1, -1), kmer, stride=1).view(-1)  # k-mer mean (no cuDNN)
        total = total + F.relu(win_id - thr).pow(2).sum()
    return total / T                                     # per-base normalisation


def hairpin_loss(nuc, stem=11, min_loop=3, max_span=70, thr=0.6):
    """Hairpin = inverted repeat: outer pair (i, i+d) inward to (i+o, i+d-o) is reverse-complementary.
    Penalise when the mean complementary agreement over ``stem`` pairs exceeds ``thr``
    (antiparallel geometry)."""
    T = nuc.shape[0]
    comp = nuc[:, COMP_PERM]                              # complement distribution [T, 4]
    d_min = 2 * stem - 1 + min_loop                       # guarantees loop >= min_loop
    if T <= d_min:
        return nuc.sum() * 0.0
    total = nuc.sum() * 0.0
    for d in range(d_min, min(max_span, T - 1) + 1):
        npos = T - d                                      # outer pair i = 0 .. npos-1
        if npos <= 0:
            break
        acc = nuc.new_zeros(npos)
        for o in range(stem):                             # average over inner stem pairs
            a = nuc[o:o + npos]                           # nuc[i+o]
            b = comp[d - o:d - o + npos]                  # comp[i+d-o]
            acc = acc + (a * b).sum(-1)
        stem_id = acc / stem                              # [npos] mean complementary agreement
        total = total + F.relu(stem_id - thr).pow(2).sum()
    return total / T                                      # per-base normalisation


# ============================================================================
#  3. combined loss (batch)
# ============================================================================
# IDT issue -> loss mapping. Thresholds (thr/hi) sit near natural levels so the loss only
# fires on genuinely problematic windows. ``weights`` are the lambda knobs (Pareto handle).
DEFAULT_CFG = dict(
    gc_window=100,                             # window (bp) for the windowed high-GC penalty
    gc_thr=0.60,                               # penalise GC fraction above this (windowed + overall)
    repeat=dict(kmer=8, min_lag=3, max_lag=80, thr=0.4),
    hairpin=dict(stem=11, min_loop=3, max_span=70, thr=0.4),
    weights=dict(gc=1.0, repeat=1.0, hairpin=0.5),
)


def manufacturability_loss(logits, query_ids, attn_mask, codon_cols, codon_nuc,
                           query_token_lo, query_token_hi, cfg=DEFAULT_CFG):
    """Mean over a batch of the weighted GC + repeat + hairpin surrogate losses.

    ``logits`` [B, L, V] from an amino-acid query forward, ``query_ids`` [B, L] the query
    input (codon positions hold "{aa}_unk" tokens in ``[query_token_lo, query_token_hi]``).
    Returns ``(total_loss_tensor, components_dict_of_floats)``.
    """
    comp = dict(gc=0.0, repeat=0.0, hairpin=0.0)
    nseq = 0
    for b in range(logits.shape[0]):
        is_codon_pos = ((attn_mask[b] > 0)
                        & (query_ids[b] >= query_token_lo)
                        & (query_ids[b] <= query_token_hi))
        nuc = soft_nuc_one(logits[b], is_codon_pos, codon_cols, codon_nuc)
        if nuc.shape[0] < 12:
            continue
        gc = gc_window_loss(nuc, cfg["gc_window"], cfg["gc_thr"])          # windowed high GC
        gc = gc + gc_window_loss(nuc, nuc.shape[0], cfg["gc_thr"])         # overall high GC (whole CDS)
        comp["gc"] = comp["gc"] + gc
        comp["repeat"] = comp["repeat"] + repeat_loss(nuc, **cfg["repeat"])
        comp["hairpin"] = comp["hairpin"] + hairpin_loss(nuc, **cfg["hairpin"])
        nseq += 1

    if nseq == 0:
        z = logits.sum() * 0.0
        return z, {k: 0.0 for k in comp}

    w = cfg["weights"]
    for k in comp:
        comp[k] = comp[k] / nseq
    total = w["gc"] * comp["gc"] + w["repeat"] * comp["repeat"] + w["hairpin"] * comp["hairpin"]
    comp = {k: (float(v) if not torch.is_tensor(v) else float(v.detach())) for k, v in comp.items()}
    return total, comp
