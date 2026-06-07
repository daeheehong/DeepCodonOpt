"""Shared utilities for the DeepCodonOpt pipeline.

Sections
    1. Reproducibility & distributed (DDP) helpers
    2. MLM training data pipeline (dataset, masked collator, loaders)
    3. Train / evaluate loops (with an optional auxiliary-loss hook for Stage 3)
    4. Optimizer + constant-LR-with-warmup scheduler
    5. Model loading (fine-tuned checkpoint / pretrained CodonTransformer) + tokenizer
    6. Training driver shared by baseline.py and lossbio.py
    7. Sequence parsing, DNA generation, and CAI scoring (used by analyze.py)

Requires the CodonTransformer package (https://github.com/Adibvafa/CodonTransformer)
installed in the same environment; see README.
"""

import json
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    BigBirdForMaskedLM,
    get_constant_schedule_with_warmup,
)

from CodonTransformer.CodonUtils import MAX_LEN, TOKEN2MASK

BASE_MODEL = "adibvafa/CodonTransformer-base"   # bare weights, used as the fine-tuning start
PRETRAINED_MODEL = "adibvafa/CodonTransformer"  # fully trained multi-organism model
TOKENIZER = "adibvafa/CodonTransformer"


# ============================================================================
#  1. reproducibility & distributed
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_distributed():
    """Initialise the process group when launched with torchrun (WORLD_SIZE > 1).

    Returns (is_ddp, rank, world_size, local_rank). Running ``python script.py``
    directly keeps the single-GPU path.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            # Pin rank<->GPU so NCCL does not warn about an unknown device / potential hang.
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local}"))
        else:
            dist.init_process_group(backend="gloo")
        return True, dist.get_rank(), dist.get_world_size(), local
    return False, 0, 1, 0


def enable_tf32():
    """Allow TF32 matmul on Ampere+ GPUs (negligible accuracy impact, faster)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def wrap_ddp(model, local_rank):
    # static_graph=True: handles unused params (pooler), is compatible with gradient
    # checkpointing, and is faster than find_unused_parameters for a fixed graph.
    return DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=True)


# ============================================================================
#  2. data pipeline
# ============================================================================
class JsonlDataset(Dataset):
    """One JSON object per line: ``{"codons": "M_ATG ... __TAA", "organism": <int>}``."""

    def __init__(self, path):
        self.data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MaskedTokenizerCollator:
    """BERT-style 15% MLM masking over codon tokens.

    ``token_type_ids`` carries the organism id broadcast across the sequence, as
    expected by CodonTransformer's BigBird model.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        seq_len = tokenized["input_ids"].shape[-1]
        organism = torch.tensor([[ex["organism"]] for ex in examples])
        tokenized["token_type_ids"] = organism.repeat(1, seq_len)

        inputs = tokenized["input_ids"]
        targets = inputs.clone()

        prob = torch.full(inputs.shape, 0.15)
        prob[inputs < 5] = 0.0                       # never mask special tokens (0-4)
        selected = torch.bernoulli(prob).bool()

        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(
            list(map(TOKEN2MASK.__getitem__, inputs[replaced].numpy()))
        )
        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool() & selected & ~replaced
        )
        rand = torch.randint(26, 90, inputs.shape, dtype=torch.long)
        inputs[randomized] = rand[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, torch.tensor(-100))
        return tokenized


def make_loader(dataset, collator, batch_size, shuffle, num_workers,
                sampler=None, persistent=False):
    kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),     # a sampler owns shuffling
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    if num_workers > 0:
        kw["persistent_workers"] = persistent
        kw["prefetch_factor"] = 4
    return DataLoader(**kw)


# ============================================================================
#  3. train / evaluate
# ============================================================================
def _masked_correct_total(logits, labels):
    mask = labels != -100
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).sum().item(), mask.sum().item()


def train_one_epoch(model, core, loader, optimizer, scheduler, scaler, device, attn,
                    accum, use_amp, log_every, epoch, is_main, aux_loss_fn=None):
    """Run one MLM epoch.

    If ``aux_loss_fn`` is given (Stage 3) it is called as
    ``aux_loss_fn(model, batch) -> (aux_loss_tensor, components_dict)`` and its
    value is added to the MLM loss before backprop. Returns
    ``(mean_mlm_loss, mean_aux_components)``.
    """
    model.train()
    optimizer.zero_grad()
    running, aux_running, n_batches = 0.0, {}, 0
    did_step = False
    for i, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        core.bert.set_attention_type(attn)           # re-assert each step (avoids BigBird OOM drift)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(**batch)
            loss = out.loss
            if aux_loss_fn is not None:
                aux, comps = aux_loss_fn(model, batch)
                loss = loss + aux
                for k, v in comps.items():
                    aux_running[k] = aux_running.get(k, 0.0) + v
            loss = loss / accum

        scaler.scale(loss).backward()
        did_step = False
        if (i + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            did_step = True

        running += out.loss.item()
        n_batches += 1
        if is_main and (i + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
            extra = ("  " + "  ".join(f"{k} {v / n_batches:.3f}" for k, v in aux_running.items())
                     if aux_running else "")
            print(f"  [epoch {epoch}] step {i+1}  loss {out.loss.item():.4f}  lr {lr:.2e}  "
                  f"mem {mem:.1f}GB{extra}", flush=True)

    if not did_step:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    mean_aux = {k: v / max(n_batches, 1) for k, v in aux_running.items()}
    return running / max(n_batches, 1), mean_aux


@torch.no_grad()
def evaluate(core, loader, device, attn, use_amp, eval_seed):
    """MLM loss / perplexity / accuracy. A fixed seed makes the masking (and thus the
    metric) deterministic and identical across DDP ranks, so early-stopping agrees."""
    core.eval()
    rng_state = torch.get_rng_state()
    torch.manual_seed(eval_seed)

    tot_loss, tot_masked, correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        core.bert.set_attention_type(attn)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = core(**batch)
        n = (batch["labels"] != -100).sum().item()
        tot_loss += out.loss.item() * n
        tot_masked += n
        c, _ = _masked_correct_total(out.logits, batch["labels"])
        correct += c

    torch.set_rng_state(rng_state)
    avg = tot_loss / max(tot_masked, 1)
    return {"loss": avg, "ppl": math.exp(avg), "acc": correct / max(tot_masked, 1)}


# ============================================================================
#  4. optimizer / scheduler
# ============================================================================
def build_optimizer_scheduler(model, train_size, args, world):
    """AdamW + constant LR with warmup.

    LR is decoupled from ``max_epochs``: warmup ramps 0 -> lr then stays flat, so
    ``max_epochs`` is purely an early-stop ceiling rather than a decay horizon.
    Warmup is expressed in epochs (``warmup_steps`` > 0 overrides) because a fixed
    step count turns into tens of epochs on small data subsets.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    n_per_rank = math.ceil(train_size / world)
    steps_per_epoch = math.ceil(n_per_rank / args.batch_size)
    opt_steps = math.ceil(steps_per_epoch / args.accumulate_grad_batches)
    warmup = (args.warmup_steps if args.warmup_steps > 0
              else max(1, round(args.warmup_epochs * opt_steps)))
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    return optimizer, scheduler, opt_steps, warmup


# ============================================================================
#  5. models & tokenizer
# ============================================================================
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _strip_prefixes(state_dict):
    """Accept a raw state_dict or a {'state_dict': ...} wrapper and drop module./model. prefixes."""
    if (isinstance(state_dict, dict) and "state_dict" in state_dict
            and not any(k.startswith(("bert.", "cls.")) for k in state_dict)):
        state_dict = state_dict["state_dict"]
    clean = {}
    for k, v in state_dict.items():
        for pre in ("module.", "model."):
            if k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v
    return clean


def load_finetuned_model(checkpoint, device, attn="original_full"):
    """Load a raw state_dict saved by baseline.py / lossbio.py (not a Lightning ckpt)."""
    model = BigBirdForMaskedLM.from_pretrained(BASE_MODEL)
    missing, unexpected = model.load_state_dict(
        _strip_prefixes(torch.load(checkpoint, map_location="cpu")), strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} (expected 0)", flush=True)
    model.bert.set_attention_type(attn)
    return model.eval().to(device)


def load_pretrained_model(device, attn="original_full"):
    """Stock multi-organism CodonTransformer (no E. coli fine-tuning) for comparison."""
    model = BigBirdForMaskedLM.from_pretrained(PRETRAINED_MODEL)
    model.bert.set_attention_type(attn)
    return model.eval().to(device)


def new_base_model(device, attn, grad_checkpoint=False):
    """Fresh base model to fine-tune (used by baseline.py / lossbio.py)."""
    model = BigBirdForMaskedLM.from_pretrained(BASE_MODEL).to(device)
    model.bert.set_attention_type(attn)
    if grad_checkpoint:
        # Trade ~20% speed for lower memory on long sequences. use_reentrant=False is DDP-safe.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model


# ============================================================================
#  6. training driver (shared by baseline.py and lossbio.py)
# ============================================================================
def add_training_args(p):
    """Register the training arguments shared by Stage 1 and Stage 3."""
    p.add_argument("--train_data_path", required=True, help="train.jsonl")
    p.add_argument("--val_data_path", required=True, help="val.jsonl")
    p.add_argument("--test_data_path", required=True, help="test.jsonl")
    p.add_argument("--init_checkpoint", default=None,
                   help="raw state_dict to start from (e.g. the baseline checkpoint for Stage 3)")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--checkpoint_filename", default="best.pt")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--max_epochs", type=int, default=15)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_epochs", type=float, default=1.0,
                   help="warmup length in epochs (used when warmup_steps == 0)")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="explicit warmup steps; 0 -> warmup_epochs x (opt_steps/epoch)")
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="gradient checkpointing (lower memory, ~20%% slower)")
    p.add_argument("--attention_type", default="block_sparse",
                   choices=["auto", "original_full", "block_sparse"])
    p.add_argument("--patience", type=int, default=3, help="epochs without val improvement before stop")
    p.add_argument("--min_delta", type=float, default=0.0, help="minimum val improvement to count")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eval_seed", type=int, default=42, help="fixes val/test masking")
    p.add_argument("--log_every", type=int, default=50)
    return p


def run_training(args, aux_loss_factory=None):
    """End-to-end fine-tuning: data -> model -> train/val (early stop) -> test.

    ``aux_loss_factory(core, tokenizer, device) -> aux_loss_fn`` lets Stage 3 inject a
    differentiable constraint loss without duplicating this driver. Returns best val loss.
    """
    is_ddp, rank, world, local = maybe_init_distributed()
    is_main = (rank == 0)

    def log(*a, **k):
        if is_main:
            print(*a, **k, flush=True)

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local}")
        enable_tf32()
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"

    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()
    best_path = os.path.join(args.checkpoint_dir, args.checkpoint_filename)
    attn = "original_full" if args.attention_type == "auto" else args.attention_type
    log(f"[setup] device={device} amp={use_amp} attn={attn} ddp={is_ddp} world={world}")

    # --- data ---
    tokenizer = get_tokenizer()
    collator = MaskedTokenizerCollator(tokenizer)
    train_data = JsonlDataset(args.train_data_path)
    val_data = JsonlDataset(args.val_data_path)
    test_data = JsonlDataset(args.test_data_path)
    train_sampler = (DistributedSampler(train_data, num_replicas=world, rank=rank,
                                        shuffle=True, drop_last=False) if is_ddp else None)
    train_loader = make_loader(train_data, collator, args.batch_size, True,
                               args.num_workers, sampler=train_sampler, persistent=True)
    eval_workers = min(2, args.num_workers)
    val_loader = make_loader(val_data, collator, args.batch_size, False, eval_workers)
    test_loader = make_loader(test_data, collator, args.batch_size, False, eval_workers)

    # --- model ---
    model = new_base_model(device, attn, args.grad_checkpoint)
    if getattr(args, "init_checkpoint", None):
        missing, unexpected = model.load_state_dict(
            _strip_prefixes(torch.load(args.init_checkpoint, map_location=device)), strict=False)
        log(f"[init] loaded {args.init_checkpoint} (missing={len(missing)} unexpected={len(unexpected)})")
    if args.grad_checkpoint:
        log("[mem] gradient checkpointing ON")
    if is_ddp:
        model = wrap_ddp(model, local)
    core = model.module if is_ddp else model         # unwrapped, for set_attention_type / state_dict

    optimizer, scheduler, opt_steps, warmup = build_optimizer_scheduler(
        model, len(train_data), args, world)
    log(f"[sched] opt_steps/epoch={opt_steps}  constant-lr  warmup_steps={warmup}")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    aux_loss_fn = aux_loss_factory(core, tokenizer, device) if aux_loss_factory else None

    # --- train / val loop with early stopping ---
    best_val, patience_ctr = float("inf"), 0
    for epoch in range(args.max_epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        train_loss, aux = train_one_epoch(
            model, core, train_loader, optimizer, scheduler, scaler, device, attn,
            args.accumulate_grad_batches, use_amp, args.log_every, epoch, is_main, aux_loss_fn)
        val = evaluate(core, val_loader, device, attn, use_amp, args.eval_seed)
        aux_str = ("  | " + "  ".join(f"{k} {v:.3f}" for k, v in aux.items())) if aux else ""
        log(f"[epoch {epoch}] train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
            f"val_ppl {val['ppl']:.2f} val_acc {val['acc']:.4f}{aux_str}")
        if device.type == "cuda":
            log(f"  [mem] peak {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)

        if val["loss"] < best_val - args.min_delta:
            best_val = val["loss"]
            patience_ctr = 0
            if is_main:
                torch.save(core.state_dict(), best_path)
                log(f"  -> new best, saved {best_path}")
        else:
            patience_ctr += 1
            log(f"  -> no improvement (patience {patience_ctr}/{args.patience})")
            if patience_ctr >= args.patience:
                log("  -> early stopping")
                break

    # --- test on best checkpoint ---
    if is_ddp:
        dist.barrier()
    log("\n[test] loading best checkpoint")
    core.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(core, test_loader, device, attn, use_amp, args.eval_seed)
    log(f"[test] loss {test['loss']:.4f} ppl {test['ppl']:.2f} acc {test['acc']:.4f}")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return best_val


# ============================================================================
#  7. parsing / generation / CAI  (used by analyze.py)
# ============================================================================
def parse_codons(codons_str):
    """``'M_ATG K_AAA ... __TAA'`` -> (protein, dna). U->T normalised. None if malformed."""
    toks = codons_str.strip().split()
    if not toks:
        return None
    aa, dna = [], []
    for t in toks:
        if len(t) < 5 or t[1] != "_":
            return None
        aa.append(t[0])
        dna.append(t[-3:].upper().replace("U", "T"))
    protein, real = "".join(aa), "".join(dna)
    if set(real) - set("ACGT") or len(real) != 3 * len(protein):
        return None
    return protein, real


def load_eval_dataset(path, n=0, seed=123, default_organism="Escherichia coli general"):
    """JSONL with a ``codons`` field -> ``[{protein, real_dna, organism}]`` (shuffled, first n if n>0)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parsed = parse_codons(obj["codons"])
            if parsed:
                rows.append({"protein": parsed[0], "real_dna": parsed[1],
                             "organism": obj.get("organism", default_organism)})
    random.Random(seed).shuffle(rows)
    return rows[:n] if n and n > 0 else rows


def generate_dna(model, tokenizer, protein, organism, device, **kw):
    """Wrap CodonTransformer.predict_dna_sequence; return a DNA string ('' on failure)."""
    from CodonTransformer.CodonPrediction import predict_dna_sequence
    try:
        out = predict_dna_sequence(
            protein=protein, organism=organism, device=device,
            tokenizer=tokenizer, model=model,
            attention_type="original_full", match_protein=True, **kw)
        if isinstance(out, list):
            out = out[0]
        return out.predicted_dna.upper().replace("U", "T")
    except Exception:
        return ""


def load_cai_weights(real_dnas, codon_usage_csv=None):
    """CAI weight per codon = freq / max synonymous freq.

    Uses a codon-usage CSV (columns ``codon,amino_acid,frequency``) if given, otherwise
    estimates the reference distribution from ``real_dnas``.
    """
    if codon_usage_csv:
        import csv
        usage = {}
        try:
            with open(codon_usage_csv, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    c = (r.get("codon") or "").strip().upper().replace("U", "T")
                    aa = (r.get("amino_acid") or "").strip()
                    fr = r.get("frequency")
                    if c and aa and fr not in (None, ""):
                        usage.setdefault(aa, {})[c] = float(fr)
            if usage:
                w = {}
                for aa, d in usage.items():
                    mx = max(d.values()) or 1.0
                    for c, fr in d.items():
                        w[c] = fr / mx
                return w
            print("[cai] codon usage table empty -> falling back to real sequences", flush=True)
        except Exception as e:
            print(f"[cai] could not read {codon_usage_csv}: {e}; using real sequences", flush=True)
    try:
        from CodonTransformer.CodonEvaluation import get_CSI_weights
        return get_CSI_weights(real_dnas)
    except Exception as e:
        print(f"[cai] could not build weights: {e}", flush=True)
        return None


def cai_of(dna, weights):
    """CAI (a.k.a. CSI) of a CDS given codon weights. Strips a trailing stop codon."""
    try:
        from CodonTransformer.CodonEvaluation import get_CSI_value
        cds = dna[:-3] if dna[-3:] in ("TAA", "TAG", "TGA") else dna
        return get_CSI_value(cds, weights)
    except Exception:
        return float("nan")
