"""Phase 4 Stage 2: sentence-level LLM training and evaluation, identical for every arm (A-E).

Model and training are BIT's (BCI_AudioLLM via BIT's Trainer, its sentence config as patched for 4a: NDT ->
MLP projector -> Aero-1-Audio + LoRA, CE + BIT's contrastive term, batch 8, 150 epochs). Four things differ from
BIT's released pipeline, all identical across arms:

1. Initialization is explicit and verified (`load_init`). BIT's `--ft_ckpt` never reaches the encoder in the
   sentence stage (results/SUMMARY.md, 4a). Here the model is built without any checkpoint, then the encoder
   is loaded by name and shape from the arm's intermediate checkpoint: an SSL core (arms B, B-long), a CTC phoneme
   model (A), a Stage 1 alignment checkpoint (C, E; its projector too). Every loaded tensor is checked afterwards.
2. Checkpoint selection uses free-generation WER on BIT-val (greedy, <= 25 new tokens), not teacher-forced WER.
3. Evaluation is free generation only (`generate`): greedy and nucleus (p 0.9, T 0.7, <= 25 tokens, seeded per
   trial), one trial at a time as BIT's eval_llm.py, and null controls: the neural input of another trial
   (`null_shuffled`, a seeded derangement), Gaussian noise of the same shape (`null_noise`; the features are
   z-scored), or none at all (`null_prompt_only`: the prompt alone).
4. Training sees each trial as evaluation does (`fix_padding`, approved 2026-10-20). In BIT a trial's training
   output depends on its batch: the encoder smooths the padded batch (padding = -100 leaks into the last ~6 bins of
   shorter trials), and the merged LLM input keeps the neural padding between a trial's neural tokens and its text
   (shifted positions; with no prompt after the neural tokens the first word is predicted from a padding slot).
   Evaluation runs one trial at a time, without padding. Here smoothing ignores padding and each trial's real
   tokens are contiguous, so a trial's logits in a training batch are those it gets alone (DESIGN.md §2).

Runs in BIT's environment with cwd = external/bit (BIT opens its configs by relative path).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import torch

from bit_experiments.textmetrics import normalize, word_errors

GREEDY = {"max_new_tokens": 25, "do_sample": False}
NUCLEUS = {"max_new_tokens": 25, "do_sample": True, "top_p": 0.9, "temperature": 0.7}
DECODINGS = {"greedy": GREEDY, "nucleus": NUCLEUS}
CONDITIONS = ("real", "null_shuffled", "null_noise", "null_prompt_only")
SPIKE_KEYS = ("spikes", "spikes_mask", "spikes_timestamp", "spikes_spacestamp", "spikes_lengths")


# ---------------------------------------------------------------- initialization

def _shape_load(module: torch.nn.Module, state: dict) -> dict:
    own = module.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    module.load_state_dict(ok, strict=False)
    after = module.state_dict()
    bad = [k for k, v in ok.items() if not torch.equal(after[k].cpu(), v.to(after[k].dtype).cpu())]
    if bad:
        raise RuntimeError(f"loaded tensors differ after loading: {bad[:5]}")
    return {"loaded": len(ok), "skipped_shape": sorted(k for k in state if k in own and own[k].shape != state[k].shape),
            "not_in_checkpoint": sorted(k for k in own if k not in ok),
            "unexpected": sorted(k for k in state if k not in own)}


def load_init(model, init_dir: Path | None, load_projector: bool = True) -> dict:
    """Load an arm's intermediate checkpoint into a freshly built BCI_AudioLLM (unwrapped), in place.

    Encoder: `encoder.bin` (NDT.save_checkpoint) into `model.neural_encoder.encoder`, by name and shape. A read-in
    whose width differs (TX-only SSL core vs TX+SBP here) is not in the file or is skipped, and keeps its fresh
    initialization. `projector.bin` too if present (Stage 1 checkpoints). Raises if no core tensor was loaded."""
    if init_dir is None:
        return {"init": None}
    init_dir = Path(init_dir)
    enc = model.neural_encoder.encoder
    rep = {"init": str(init_dir),
           "encoder": _shape_load(enc, torch.load(init_dir / "encoder.bin", map_location="cpu", weights_only=True))}
    core = [k for k in enc.state_dict() if not k.startswith("embedder.read_in")]
    missing_core = [k for k in core if k in rep["encoder"]["not_in_checkpoint"]]
    if missing_core:
        raise RuntimeError(f"{init_dir}: encoder core tensors not loaded: {missing_core[:5]}")
    rep["read_in_loaded"] = not any(k.startswith("embedder.read_in") for k in rep["encoder"]["not_in_checkpoint"])
    proj = init_dir / "projector.bin"
    if load_projector and proj.exists():
        rep["projector"] = _shape_load(model.projector, torch.load(proj, map_location="cpu", weights_only=True))
        if rep["projector"]["not_in_checkpoint"]:
            raise RuntimeError(f"{proj}: projector only partly loaded")
    return rep


# ---------------------------------------------------------------- padding (difference 4)

PAD_VALUE = -100.0  # BIT's pad value for spikes (pad_collate_fn, pad_for_patching)


class PadMaskedSmoothing(torch.nn.Module):
    """BIT's SmoothAndNoise (`inner`) without the padding: batch-padding bins (every channel = -100) are zeros while
    smoothing, which is the zero edge a lone trial gets from conv1d(padding="same"), and -100 again afterwards,
    which is what pad_for_patching appends to a lone trial. BIT's own kernel and settings are used unchanged."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        pad = (spikes == PAD_VALUE).all(-1, keepdim=True)
        return self.inner(spikes.masked_fill(pad, 0)).masked_fill(pad, PAD_VALUE)


def packed_merge(self, text_embeds, spikes_embeds, spikes_mask, attention_mask, input_split, targets=None):
    """BCI_AudioLLM.merge_embeds with each trial's real tokens contiguous: [prompt_a, its valid neural tokens,
    prompt_b + sentence (+ its text padding)], then its neural-token padding. BIT inserts the batch-padded neural
    tokens whole, which shifts the positions of a shorter trial's text and, with no prompt after the neural tokens,
    predicts its first word from a padding slot. A trial without padding (batch 1, as at evaluation) is unchanged."""
    embeds, masks, labels = [], [], []
    for i, (t, s, m, a, d) in enumerate(zip(text_embeds, spikes_embeds, spikes_mask, attention_mask, input_split)):
        n, d = int(m.sum()), int(d)
        if not bool(m[:n].all()):
            raise ValueError("expected right-padded neural tokens (valid positions first)")
        embeds.append(torch.cat((t[:d], s[:n], t[d:], s[n:]), dim=0))
        masks.append(torch.cat((a[:d], m[:n], a[d:], m[n:]), dim=0))
        if targets is not None:
            y = targets[i]
            labels.append(torch.cat((y[:d], y.new_full((n,), -100), y[d:], y.new_full((m.shape[0] - n,), -100))))
    return torch.stack(embeds), torch.stack(masks), (torch.stack(labels) if targets is not None else None)


def fix_padding(model) -> dict:
    """Apply difference 4 to a freshly built Stage 2 model (BCI_AudioLLM, BCI_TextLLM or BCI_DiffusionLLM), in place:
    pad-masked smoothing in its NDT encoder, and the packed merge (the diffusion model builds its own packed input
    and never merges)."""
    enc = model.neural_encoder.encoder
    if not isinstance(enc.smooth_and_noise, PadMaskedSmoothing):
        enc.smooth_and_noise = PadMaskedSmoothing(enc.smooth_and_noise)
    model.merge_embeds = types.MethodType(packed_merge, model)
    inner = enc.smooth_and_noise.inner
    return {"smoothing": "pad-masked", "smoothing_taps": int(inner.kernel.numel()) if inner.smooth else 0,
            "noise": bool(inner.noise), "merge": "packed"}


SLIM = "slim.safetensors"
ACCELERATE_FILES = ("model.safetensors", "optimizer.bin", "scheduler.bin")


def slim_state(model) -> dict:
    """The model without its frozen base LLM: encoder, projector, contrastive heads and LoRA weights. Raises if a
    trainable parameter would be dropped."""
    keep = {k: v for k, v in model.state_dict().items() if not k.startswith("llm.") or "lora_" in k}
    dropped = sorted(n for n, p in model.named_parameters() if p.requires_grad and n not in keep)
    if dropped:
        raise RuntimeError(f"trainable parameters outside the slim state: {dropped[:5]}")
    return keep


def load_model_state(model, ckpt_dir: Path) -> dict:
    """A trained Stage 2 / BIT sentence checkpoint: slim.safetensors (base LLM from its snapshot) if present, else
    accelerate's full model.safetensors, as BIT's eval_llm.py. Only base-LLM tensors may be missing."""
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)
    f = ckpt_dir / SLIM if (ckpt_dir / SLIM).exists() else ckpt_dir / "model.safetensors"
    r = model.load_state_dict(load_file(str(f)), strict=False)
    bad = [k for k in r.missing_keys if not k.startswith("llm.") or "lora_" in k]
    if bad or r.unexpected_keys:
        raise RuntimeError(f"{f}: missing {bad[:5]}, unexpected {r.unexpected_keys[:5]}")
    return {"file": f.name, "missing_base_llm": len(r.missing_keys), "unexpected": 0}


def slim_checkpoint(model, ckpt_dir: Path) -> dict:
    """After evaluation: save the slim state of `model` (loaded from ckpt_dir) and delete accelerate's full model and
    optimizer copies there (8.5 GB, almost all of it the frozen base LLM, which stays in checkpoints/hf/)."""
    from safetensors.torch import save_file

    ckpt_dir = Path(ckpt_dir)
    state = {k: v.detach().contiguous().cpu() for k, v in slim_state(model).items()}
    save_file(state, str(ckpt_dir / SLIM))
    removed = []
    for f in [ckpt_dir / n for n in ACCELERATE_FILES] + sorted(ckpt_dir.glob("random_states_*.pkl")):
        if f.exists():
            f.unlink()
            removed.append(f.name)
    return {"slim_tensors": len(state), "slim_mb": round((ckpt_dir / SLIM).stat().st_size / 2**20, 1),
            "removed": removed}


# ---------------------------------------------------------------- data

def subsample_train(dataset: dict, fraction: float, seed: int) -> dict:
    """Keep the first round(fraction * N) trials of a seeded permutation (nested across fractions for one seed)."""
    if fraction >= 1:
        return dataset
    n = len(dataset["train"])
    keep = np.sort(np.random.default_rng(seed).permutation(n)[: max(1, round(fraction * n))])
    return dataset | {"train": [dataset["train"][i] for i in keep]}


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point (a cyclic shift of a random order)."""
    order = rng.permutation(n)
    perm = np.empty(n, dtype=int)
    perm[order] = np.roll(order, -1)
    return perm


# ---------------------------------------------------------------- generation

@torch.no_grad()
def generate(model, ds, collate, tokenizer, decoding: str, condition: str = "real", seed: int = 0,
             device: torch.device | str = "cuda", dtype=torch.bfloat16, limit: int | None = None) -> list[str]:
    """Free generation for every trial of `ds` (a SpikingDatasetForDecoding), one trial at a time.

    The prompt is the trial's input_ids where targets == -100 (BIT's prompt; the reference never enters).
    Nucleus sampling reseeds torch with seed * 100003 + trial before each trial."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    gen = dict(DECODINGS[decoding], pad_token_id=tokenizer.vocab_size, eos_token_id=tokenizer.eos_token_id)
    n = len(ds) if limit is None else min(limit, len(ds))
    rng = np.random.default_rng(seed)
    perm = derangement(n, rng) if condition == "null_shuffled" else None
    device = torch.device(device)
    model.eval()
    out = []
    for i in range(n):
        item = ds[i]
        if perm is not None:
            other = ds[int(perm[i])]
            item.update({k: other[k] for k in SPIKE_KEYS if k in other})
        elif condition == "null_noise":
            item["spikes"] = rng.standard_normal(item["spikes"].shape).astype(np.float32)
        inputs, _ = collate([item])
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        prompt = inputs["input_ids"][inputs["targets"] == -100].unsqueeze(0)
        inputs.update(input_ids=prompt, attention_mask=torch.ones_like(prompt))
        inputs.pop("targets", None)
        if condition == "null_prompt_only":
            inputs["inputs_embeds"] = model.llm.get_input_embeddings()(prompt)
        if gen["do_sample"]:
            torch.manual_seed(seed * 100003 + i)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            ids = model.generate(**inputs, **gen)
        out.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())
    return out


def score(refs: list[str], hyps: list[str], seen: np.ndarray | None = None) -> dict:
    """Corpus WER (BIT's normalization), also for trials whose sentence is / is not in the training set."""
    errs = np.array([word_errors(r, h) for r, h in zip(refs, hyps, strict=True)])
    res = {"wer": float(errs[:, 0].sum() / max(1, errs[:, 1].sum())), "n": len(refs),
           "exact": float((errs[:, 0] == 0).mean()) if len(refs) else float("nan")}
    if seen is not None:
        for name, m in (("seen", seen), ("unseen", ~seen)):
            res[f"wer_{name}"] = float(errs[m, 0].sum() / max(1, errs[m, 1].sum())) if m.any() else float("nan")
            res[f"n_{name}"] = int(m.sum())
    return res


def write_decodes(path: Path, refs: list[str], hyps: list[str], seen: np.ndarray, meta: dict) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trial": range(len(hyps)), "reference": refs[: len(hyps)], "decoded": hyps,
                  "seen_in_train": seen[: len(hyps)]}).to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))


def seen_mask(refs: list[str], train_sentences: list[str]) -> np.ndarray:
    train = {normalize(s) for s in train_sentences}
    return np.array([normalize(r) in train for r in refs])
