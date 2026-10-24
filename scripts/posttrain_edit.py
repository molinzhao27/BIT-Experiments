#!/usr/bin/env python
"""Step 1 of the two-step revision design (docs/REVISION_RUN.md): post-train Fast-dLLM v2 1.5B on general English text
with its own block-diffusion objective plus the token-to-token editing term, and save the result as a new base for
Stage 2. Prepared, not run: the user decides. The GPU is shared with the queue, so take the GPU lock.

    <bit env>/python scripts/posttrain_edit.py --out checkpoints/hf/Fast_dLLM_v2_1.5B-edit --exclude-bit
        [--llm fast-dllm-v2-1.5b | --base DIR] [--mask-id N] [--tokens 100M] [--save-at 10M 30M] [--edit-rate 0.1]
        [--seq-len 2048] [--batch-rows 64] [--micro-rows 1] [--lr 2e-5] [--warmup 50] [--lora 0] [--grad-ckpt]
        [--exclude FILE] [--text-file FILE] [--heldout-rows 64] [--eval-every 50] [--seed 42] [--device cuda]

Data: FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT, streamed in its own order), or --text-file (one document
per line, or JSON lines with "text"). Each document is tokenised and followed by <|endoftext|>, and the stream is cut
into rows of --seq-len tokens, every token a target (as Fast-dLLM v2's own training). Documents that contain a BIT
sentence of 4 or more words (--exclude-bit: train/val/test; --exclude FILE: one per line), word for word after
stage2's normalisation, are dropped, so that unseen sentences stay unseen. The first --heldout-rows rows are held out.

Step: text_block_diffusion_ce (bit_experiments.diffusion_llm) with --edit-rate. Each micro-batch's CE is averaged over
its supervised tokens, then over the micro-batches of a --batch-rows step (Fast-dLLM v2 averages; there is no
contrastive term here). Training uses:
- AdamW (no weight decay), linear warmup then a constant LR (as SCOPE: the intermediate checkpoints stay comparable);
- grad-norm clip 1.0;
- fp32 weights + bf16 autocast on CUDA;
- seeded.
--lora R: LoRA (alpha 2R, no dropout, BIT's seven projections), merged into every saved snapshot; 0 = full
fine-tune. --grad-ckpt: per-layer activation checkpointing (the vendored decoder has none under transformers 4.48).

Held-out check, at step 0, every --eval-every steps and at each save: with fixed masks, M2T CE / accuracy without
corruption (vs the base: forgetting), and T2T CE / accuracy on the corrupted tokens at --eval-edit-rate (the editor).

Outputs:
- <out>/log.jsonl;
- <out>/tokens_<N>/ for each --save-at and the final --tokens: fp32 safetensors + config, with the base snapshot's
  tokenizer and code files, loadable as the base is;
- <out>/posttrain.json: args, base, data counts, git commit, minutes, peak memory.
Register a snapshot in configs/llms.yaml by its local_dir to use it in Stage 2 (docs/REVISION_RUN.md).

The base loads as Stage 2 loads it (BCI_DiffusionLLM.from_pretrained): a Fast-dLLM v2 snapshot as before, or a Qwen2
causal LM or Aero-1-Audio's language model converted to the block-diffusion architecture (docs/AERO_DIFFUSION.md,
path 2a). Their tokenizers have no |<MASK>|: the mask is --mask-id, else the entry's mask_id, else |<MASK>| in the
base's added_tokens.json; a snapshot saved from them needs mask_id in its registry entry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

from bit_experiments.diffusion_llm import MASK_TOKEN, BCI_DiffusionLLM, text_block_diffusion_ce  # noqa: E402
from bit_experiments.textmetrics import normalize  # noqa: E402

DATASET = "brandman_2024_text"
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # BIT's
MIN_WORDS = 4  # shorter sentences ("thank you") are in every corpus and say nothing about leakage
EVAL_SEED = 1234


def count(text: str) -> int:
    """'100M' -> 100_000_000 (K, M, B suffixes)."""
    scale = {"K": 10**3, "M": 10**6, "B": 10**9}.get(text[-1].upper(), 1)
    return int(float(text[:-1] if scale > 1 else text) * scale)


class Excluder:
    """Does a document contain one of these sentences, word for word after stage2's normalisation? Sentences shorter
    than MIN_WORDS are skipped (`skipped`)."""

    def __init__(self, sentences):
        self.by_start, self.skipped = {}, 0
        for s in sentences:
            words = tuple(normalize(s).split())
            if len(words) < MIN_WORDS:
                self.skipped += 1
                continue
            self.by_start.setdefault(words[:3], set()).add(words)
        self.n = sum(len(v) for v in self.by_start.values())

    def __call__(self, text: str) -> bool:
        words = normalize(text).split()
        for i in range(len(words) - MIN_WORDS + 1):
            for sentence in self.by_start.get(tuple(words[i:i + 3]), ()):
                if tuple(words[i:i + len(sentence)]) == sentence:
                    return True
        return False


def pack(docs, tokenize, sep_id: int, seq_len: int, exclude=None, stats: dict | None = None):
    """Rows of seq_len token ids: the documents (minus the excluded ones), each followed by sep_id, back to back."""
    stats = {} if stats is None else stats
    stats.setdefault("docs", 0)
    stats.setdefault("dropped", 0)
    buf: list[int] = []
    for text in docs:
        stats["docs"] += 1
        if exclude is not None and exclude(text):
            stats["dropped"] += 1
            continue
        buf += list(tokenize(text)) + [sep_id]
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def documents(args):
    """--text-file's documents, else FineWeb-Edu streamed."""
    if args.text_file:
        with open(args.text_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"] if line.startswith("{") else line
        return
    from datasets import load_dataset

    for doc in load_dataset(args.dataset, args.dataset_config, split="train", streaming=True):
        yield doc["text"]


def bit_sentences() -> list[str]:
    """BIT's train/val/test sentences, loaded as train_stage2.py does (on the GPU host: needs BIT's data)."""
    from bit_experiments import stage1
    from registry import dataset_registry

    cwd = os.getcwd()
    os.chdir(BIT)  # BIT opens configs/ by relative path
    try:
        config = stage1.bit_sentence_config()
        data_config = config.data
        data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{DATASET}"
        ds = __import__(dataset_registry[DATASET], fromlist=["load_data"]).load_data(**data_config.dataset)
    finally:
        os.chdir(cwd)
    return [r["sentence"] for split in ("train", "val", "test") for r in ds[split]]


def base_of(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def checkpoint_layers(model):
    """Per-layer activation checkpointing (non-reentrant; the RNG state is replayed, so LoRA dropout matches)."""
    for layer in base_of(model).model.layers:
        layer.forward = partial(torch.utils.checkpoint.checkpoint, layer.forward, use_reentrant=False)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def heldout_metrics(model, rows, mask_id, micro_rows, eval_edit_rate) -> dict:
    """M2T (no corruption) and T2T (corrupted tokens at eval_edit_rate) CE and accuracy on the held-out rows, with
    fixed masks and corruption (EVAL_SEED). Leaves the RNG and the train / eval mode as it found them."""
    dev = next(model.parameters()).device
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
    was_training = model.training
    model.eval()
    res = {}
    for kind, rate in (("m2t", 0.0), ("t2t", eval_edit_rate)):
        torch.manual_seed(EVAL_SEED)
        ce = hit = n = 0
        for i in range(0, len(rows), micro_rows):
            ids = torch.tensor(rows[i:i + micro_rows], device=dev)
            with autocast(dev):
                _, _, logits, labels, corrupted = text_block_diffusion_ce(model, ids, mask_id, edit_rate=rate,
                                                                          with_corrupted=True)
            keep = corrupted if kind == "t2t" else ~corrupted
            ce += float(F.cross_entropy(logits[keep].float(), labels[keep], reduction="sum"))
            hit += int((logits[keep].argmax(-1) == labels[keep]).sum())
            n += int(keep.sum())
        res |= {f"{kind}_ce": round(ce / max(n, 1), 5), f"{kind}_acc": round(hit / max(n, 1), 5), f"{kind}_n": n}
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    model.train(was_training)
    return res


def merged_state_dict(model) -> dict:
    """The model's weights with any LoRA merged in (W + delta, out of place: training continues unchanged), under the
    plain model's names."""
    base = base_of(model)
    if base is model:
        return model.state_dict()
    delta = {name: m.get_delta_weight("default") for name, m in base.named_modules()
             if hasattr(m, "get_delta_weight") and hasattr(m, "base_layer")}
    merged = {}
    for key, value in base.state_dict().items():
        if "lora_" in key:
            continue
        if key.endswith(".base_layer.weight"):
            value = value + delta[key[: -len(".base_layer.weight")]]
        merged[key.replace(".base_layer.", ".")] = value
    return merged


def save_snapshot(model, base_dir: Path, path: Path):
    """fp32 safetensors (LoRA merged) + config, with every other file of the base snapshot (tokenizer, added tokens,
    code), so that it loads as the base does."""
    path.mkdir(parents=True, exist_ok=True)
    base_of(model).save_pretrained(path, state_dict=merged_state_dict(model), safe_serialization=True)
    for f in base_dir.iterdir():
        weights = f.suffix in (".safetensors", ".bin", ".pt") or f.name.endswith(".index.json")
        if f.is_file() and not weights and not (path / f.name).exists():
            shutil.copy2(f, path / f.name)


def train(model, rows, heldout, mask_id, args, out: Path, base_dir: Path) -> dict:
    """The post-training loop (module docstring). rows: an iterator of seq_len-token lists. Returns the last held-out
    check and the milestones saved."""
    dev = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    accum, tokens_per_step = args.batch_rows // args.micro_rows, args.batch_rows * args.seq_len
    saves = sorted(set(args.save_at) | {args.tokens})
    log = open(out / "log.jsonl", "a")
    t0, step, tokens, saved = time.time(), 0, 0, []

    def record(entry):
        entry |= {"step": step, "tokens": tokens, "minutes": round((time.time() - t0) / 60, 2)}
        if dev.type == "cuda":
            entry["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(f"[posttrain] {json.dumps(entry)}", flush=True)
        return entry

    last = record({"heldout": heldout_metrics(model, heldout, mask_id, args.micro_rows, args.eval_edit_rate)})
    model.train()
    while tokens < args.tokens:
        lr = args.lr * min(1.0, (step + 1) / args.warmup) if args.warmup else args.lr
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            try:
                ids = torch.tensor([next(rows) for _ in range(args.micro_rows)], device=dev)
            except StopIteration:
                raise RuntimeError(f"the corpus ran out after {tokens} tokens") from None
            with autocast(dev):
                ce, n, _, _ = text_block_diffusion_ce(model, ids, mask_id, edit_rate=args.edit_rate)
            loss = ce / n / accum
            loss.backward()
            loss_sum += float(loss)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        step, tokens = step + 1, tokens + tokens_per_step
        due = [s for s in saves if s <= tokens and s not in saved]
        if step % args.eval_every == 0 or due:
            last = record({"loss": round(loss_sum, 5), "grad_norm": round(grad_norm, 4), "lr": lr,
                           "heldout": heldout_metrics(model, heldout, mask_id, args.micro_rows, args.eval_edit_rate)})
        for s in due:
            save_snapshot(model, base_dir, out / f"tokens_{s}")
            saved.append(s)
    log.close()
    return {"last": last, "saved": saved}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (new)")
    ap.add_argument("--llm", default="fast-dllm-v2-1.5b", help="configs/llms.yaml entry of the base")
    ap.add_argument("--base", help="base snapshot directory instead of --llm")
    ap.add_argument("--mask-id", type=int, help="the mask's embedding row (default: the entry's mask_id, else |<MASK>| "
                                                "in the base's added_tokens.json)")
    ap.add_argument("--tokens", type=count, default=count("100M"), help="training tokens (K/M/B suffixes)")
    ap.add_argument("--save-at", type=count, nargs="*", default=[count("10M"), count("30M")])
    ap.add_argument("--edit-rate", type=float, default=0.1, help="T2T corruption rate (0 = M2T-only control)")
    ap.add_argument("--eval-edit-rate", type=float, default=0.1)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-rows", type=int, default=64)
    ap.add_argument("--micro-rows", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50, help="warmup steps")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 = full fine-tune")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--exclude-bit", action="store_true", help="drop documents with a BIT sentence (needs BIT data)")
    ap.add_argument("--exclude", help="file of sentences (one per line) to drop documents for")
    ap.add_argument("--text-file", help="local documents instead of FineWeb-Edu")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset-config", default="sample-10BT")
    ap.add_argument("--heldout-rows", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=50, help="steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main() -> int:
    ap = parser()
    args = ap.parse_args()
    if args.batch_rows % args.micro_rows or args.seq_len % 32:
        ap.error("--batch-rows must be a multiple of --micro-rows, --seq-len of the block size (32)")
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} exists and is not empty")
    out.mkdir(parents=True)
    t0 = time.time()
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    if args.base:
        base_dir, entry = Path(args.base), None
    else:
        from bit_experiments.text_llm import llm_entry

        entry = llm_entry(args.llm)
        base_dir = Path(entry["path"])
    if args.mask_id is not None:
        mask_id = args.mask_id
    elif entry is not None and "mask_id" in entry:  # an AR snapshot without |<MASK>| (Instruct, Aero-1-Audio)
        mask_id = entry["mask_id"]
    else:
        mask_id = json.loads((base_dir / "added_tokens.json").read_text())[MASK_TOKEN]
    model = BCI_DiffusionLLM.from_pretrained(base_dir)  # a Fast-dLLM v2 snapshot: the same call as before
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    sentences = (bit_sentences() if args.exclude_bit else []) + \
        (Path(args.exclude).read_text().splitlines() if args.exclude else [])
    exclude = Excluder(sentences) if sentences else None
    stats: dict = {}
    rows = pack(documents(args), lambda t: tokenizer(t, add_special_tokens=False)["input_ids"],
                model.config.bos_token_id, args.seq_len, exclude, stats)
    heldout = [next(rows) for _ in range(args.heldout_rows)]
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS))
    if args.grad_ckpt:
        checkpoint_layers(model)
    model.to(args.device)
    run = {"args": vars(args), "base": str(base_dir), "entry": entry, "mask_id": mask_id,
           "exclude": {"sentences": exclude.n, "too_short": exclude.skipped} if exclude else None,
           "git": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(), "torch": torch.__version__}
    (out / "posttrain.json").write_text(json.dumps(run, indent=1, default=str))
    result = train(model, rows, heldout, mask_id, args, out, base_dir)
    run |= result | {"data": stats, "minutes": round((time.time() - t0) / 60, 2)}
    if torch.cuda.is_available():
        run["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "posttrain.json").write_text(json.dumps(run, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
