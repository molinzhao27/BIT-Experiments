#!/usr/bin/env python
"""What does Aero block diffusion actually output, and what context does it use? CPU only (2026-10-23).

    CUDA_VISIBLE_DEVICES="" nice -n 19 <bit env>/python scripts/aero_decode_inspect.py \
        --ckpt <trained Stage 2 checkpoint dir> [--llm aero-1-audio-diffusion] [--init checkpoints/ssl_human/BEST_core]
        [--parts decode,teacher,context,zeroshot] [--n 12] [--seen 4] [--trials 64] [--ratios 0.1,0.3,0.5,0.7,0.9,1.0]
        [--threads 6] [--dtype bf16] [--zero-shot 100] [--passes 8]

Time on 6 threads: decode + teacher ~5 min, context ~5-10 min for 64 trials (scales with --trials), zeroshot ~4 min,
plus ~2 min to load the data and model. For other models (the control, run 3) pass their --llm and --ckpt and
usually --parts context.

Question: the fast Aero run's val decodes are never exact, not even on sentences seen in training (WER_seen 95-112%),
while the control memorizes those quickly. What gets generated?

It builds the model exactly as train_stage2.py does (config, model, --init, seed 42, data, BIT trainer; on the CPU when
CUDA is hidden) and loads --ckpt. Trials: the first --n BIT-val trials, of which up to --seen are seen-in-train ones.
1. Vocabulary: the tokenizer's eos and size, the model's eos, mask and filler, the embedding and head rows, the rows
   beyond 151936 (Aero's extra rows), and the rows bitwise equal to the mask row.
2. Greedy decodes with the model's own sampler and stage2.generate's settings (max 25 new tokens, the tokenizer's eos,
   bf16 autocast as in the GPU eval; --dtype fp32 to compare). Per trial: reference, decode, WER, generated ids and
   tokens, flags (ids >= 151936, ids in the mask-identical set, the mask id, other special ids, the first eos
   position, stopped early or late or at the 25-token cap), the number of denoising passes and blocks, and the
   commits of the first --passes passes.
3. Teacher-forced view of the same trials: the Stage 2 objective (the packed step, which equals the plain one) with
   one seed per trial. At each masked position: argmax token and probability against the target, split into sentence
   tokens, the first eos (end of sentence), and the eos padding to the end of the block.
4. Zero-shot split (--zero-shot N, 0 to skip): scripts/text_blank_fill_check.py's test (the same sentences, prompt
   and masks; untrained Instruct and Aero) with the CE split into sentence tokens, the first eos, and the eos padding.
   It shows whether Aero's zero-shot deficit is in the sentence or in the padding.
5. Context use (--parts context), teacher-forced on the first --trials val trials. It uses one noisy copy per ratio,
   in the training layout's attention: each target block sees the clean blocks before it and itself. This runs
   through the decoder's own KV-cache path, and the first trial checks it against the training layout. Each target
   token is masked with probability r for r in --ratios (r = 1: all), with the masks keyed by target offset, so
   they are the same for every model and condition.
   - CE by ratio, split into sentence tokens, end-of-sentence eos and eos padding, with the real neural input and
     with the next trial's (S = shuffled - real). If CE barely falls as more of the sentence is visible, the model
     is not using the text context.
   - Attention mass by ratio, per layer, averaged over heads: from masked sentence tokens' predictor positions
     (j - 1, which decode them under the token shift) and from the masked positions themselves, to the neural span,
     the prompt text, visible sentence tokens and eos/padding in earlier blocks, visible sentence tokens and
     eos/padding in the same block, and masks.
Writes results/aero_inspect/<name>.json. Nothing is saved to checkpoints. Its results/stage2/<name> is removed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
QWEN_ROWS = 151936  # Qwen2.5's embedding rows; Aero has 64 more
DATA = {}


class InspectDone(Exception):
    def __init__(self, results):
        super().__init__("inspect done")
        self.results = results


def supervised_positions(labels: torch.Tensor, block_size: int, seed: int):
    """block_diffusion_ce's supervised outputs, in its order, as (copy row, target position j). Re-draws its first
    two random calls under `seed` (edit_rate 0), as block_diffusion_ce makes them right after the seed is set."""
    from bit_experiments.diffusion_llm import EPS

    torch.manual_seed(seed)
    b, length = labels.shape
    n_blocks = length // block_size
    ratio = ((1 - EPS) * torch.rand(b * n_blocks, device=labels.device) + EPS)[:, None].repeat(1, block_size)
    masked = (torch.rand(b * n_blocks, block_size, device=labels.device) < ratio).reshape(b, length)
    is_target = labels != -100
    sup = torch.cat([masked & is_target, ~masked & is_target])
    sup[:, 0] = False  # token shift: position 0 is never decoded
    return sup.nonzero(as_tuple=True)


def classes(labels_row: torch.Tensor, eos_ids: set[int]) -> tuple[int, int]:
    """(first target position P, position of the first eos) in a [L] labels row: sentence tokens are [P, eos), the
    end-of-sentence eos is at eos, the padding is after it. BIT appends the tokenizer's eos to the sentence; forward
    pads with the model's, so either counts."""
    tgt = (labels_row != -100).nonzero().flatten()
    p = int(tgt[0])
    eos = torch.isin(labels_row[p:], torch.tensor(sorted(eos_ids), device=labels_row.device)).nonzero().flatten()
    return p, p + int(eos[0]) if eos.numel() else int(tgt[-1]) + 1


def split_ce(logits, targets, rows, js, first_eos):
    """Per class (sentence / eos / pad): count, CE sum, argmax accuracy, mean argmax probability, mean target probability."""
    logp = F.log_softmax(logits.float(), -1)
    ce = -logp.gather(1, targets[:, None]).squeeze(1)
    p_max, arg = logp.exp().max(-1)
    cls = np.where(js.numpy() < first_eos, "sentence", np.where(js.numpy() == first_eos, "eos", "pad"))
    out = {}
    for c in ("sentence", "eos", "pad"):
        m = torch.from_numpy(cls == c)
        if m.any():
            out[c] = {"n": int(m.sum()), "ce_sum": float(ce[m].sum()), "acc": float((arg[m] == targets[m]).float().mean()),
                      "p_argmax": float(p_max[m].mean()), "p_target": float((-ce[m]).exp().mean())}
    return out, cls, arg, p_max


def merge(stats: list[dict]) -> dict:
    out = {}
    for c in ("sentence", "eos", "pad"):
        ss = [s[c] for s in stats if c in s]
        n = sum(s["n"] for s in ss)
        if n:
            out[c] = {"n": n, "ce": round(sum(s["ce_sum"] for s in ss) / n, 4),
                      "acc": round(sum(s["acc"] * s["n"] for s in ss) / n, 4),
                      "p_argmax": round(sum(s["p_argmax"] * s["n"] for s in ss) / n, 4),
                      "p_target": round(sum(s["p_target"] * s["n"] for s in ss) / n, 4)}
    return out


def step_inputs(model, inputs, device, dtype):
    """The (clean [1, L, d], labels [1, L]) that BCI_DiffusionLLM.forward hands to denoising_ce for these inputs."""
    seen = {}
    model.denoising_ce = lambda clean, labels: (seen.update(clean=clean, labels=labels)
                                                or (clean.sum() * 0, labels.new_ones(()), clean, labels))
    try:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            model(**inputs)
    finally:
        del model.denoising_ce
    return seen["clean"], seen["labels"]


CATS = ("neural", "prompt", "earlier_sentence", "earlier_eos_pad", "same_sentence", "same_eos_pad", "masks")


@torch.no_grad()
def context_use(model, val_ds, collate, trials, ratios, device, dtype, eos_ids) -> dict:
    """Teacher-forced, one noisy copy per ratio, in the training layout's attention (module docstring, part 5).

    Each target block b is denoised as in training: it sees the clean blocks before it and itself. This is computed
    with the vendored decoder's own KV-cache path (as the sampler uses it): the clean prefix and clean earlier target
    blocks are cached once per trial and condition, and only the noisy target blocks run per ratio. A self-check on the
    first trial compares it with the training layout ([noisy ‖ clean], block_diff_mask). Masks: each target token is
    masked with probability r (r = 1: all). They are drawn per target offset from a seed per trial and ratio, so they
    are the same under both conditions (real, neural input of the next trial) and for every model. A masked target j
    is decoded from position j - 1 (token shift). Attention mass, real condition: from the predictor positions j - 1
    of masked sentence tokens ("pred") and from the masked sentence positions themselves ("mask"), per layer,
    averaged over heads, summed over key categories (CATS)."""
    import copy

    from transformers.cache_utils import DynamicCache

    from bit_experiments import stage2
    from bit_experiments.diffusion_llm import diffusion_layout
    from bit_experiments.fast_dllm_v2 import modeling

    decoder, lm_head, embed = model._parts()
    bs = model.block_size
    mask_embed = embed(torch.tensor([model.mask_id], device=device))
    n_layers = decoder.config.num_hidden_layers
    rec = {"rows": None, "out": {}}
    sdpa = modeling.ALL_ATTENTION_FUNCTIONS["sdpa"]

    def recording_sdpa(module, query, key, value, attention_mask, **kw):
        out = sdpa(module, query, key, value, attention_mask, **kw)
        if rec["rows"]:
            with torch.autocast(query.device.type, enabled=False):  # fp32 attention weights
                rows = torch.tensor(rec["rows"], device=query.device)
                q = query[0, :, rows].float()
                k = modeling.repeat_kv(key, query.shape[1] // key.shape[1])[0].float()
                scores = (q @ k.transpose(-1, -2)) * (kw.get("scaling") or module.scaling)
                scores = scores.masked_fill(~attention_mask[0, 0, rows].bool(), float("-inf"))
                rec["out"][module.layer_idx] = scores.softmax(-1).mean(0).float().cpu()
        return out

    def amp():
        return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)

    def fwd(x, cache, commit):
        with model._layout(training=False), amp():
            return decoder(inputs_embeds=x[None], past_key_values=cache, use_cache=True,
                           update_past_key_values=commit, block_size=bs).last_hidden_state[0]

    def item(i, condition):
        it = val_ds[i]
        if condition == "shuffled":
            other = val_ds[trials[(trials.index(i) + 1) % len(trials)]]
            it.update({k: other[k] for k in stage2.SPIKE_KEYS if k in other})
        inputs, _ = collate([it])
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    acc = {c: {r: {k: [0.0, 0, 0] for k in ("sentence", "eos", "pad")} for r in ratios} for c in ("real", "shuffled")}
    mass = {r: {q: torch.zeros(n_layers, len(CATS)) for q in ("pred", "mask")} for r in ratios}
    n_rows = {r: {"pred": 0, "mask": 0} for r in ratios}
    selfcheck = None
    modeling.ALL_ATTENTION_FUNCTIONS["sdpa"] = recording_sdpa
    t_start = time.time()
    try:
        for ti, i in enumerate(trials):
            for condition in ("real", "shuffled"):
                inputs = item(i, condition)
                clean, labels = step_inputs(model, inputs, device, dtype)
                clean, labels = clean[0], labels[0]
                p, first_eos = classes(labels, eos_ids)
                tgt = (labels != -100).nonzero().flatten()
                end = int(tgt[-1]) + 1
                k0, k_last = p // bs, (end - 1) // bs
                n_text = int(inputs["attention_mask"][0].sum())
                n_prompt = int((inputs["targets"][0, :n_text] == -100).sum())
                split = int(inputs["input_split"][0])
                neural = (split, split + p - n_prompt)  # [prompt_a, neural tokens, prompt_b] (no fillers)
                cache, prefix_last = DynamicCache(), None
                if k0:
                    prefix_last = fwd(clean[: k0 * bs], cache, True)[-1]
                caches = {k0: copy.deepcopy(cache)}
                for b in range(k0, k_last):
                    fwd(clean[b * bs: (b + 1) * bs], cache, True)
                    caches[b + 1] = copy.deepcopy(cache)
                for ri, r in enumerate(ratios):
                    g = torch.Generator().manual_seed(10007 * i + 101 * ri)
                    m = torch.ones(len(tgt), dtype=torch.bool) if r >= 1 else torch.rand(len(tgt), generator=g) < r
                    masked = tgt[m.to(tgt.device)]
                    if masked.numel() == 0:  # nothing masked at this ratio (short sentence, small r): nothing to score
                        continue
                    is_masked = torch.zeros_like(labels, dtype=torch.bool)
                    is_masked[masked] = True
                    x = torch.where(is_masked[:, None], mask_embed.to(clean.dtype), clean)
                    hidden = {}
                    for b in range(k0, k_last + 1):
                        lo = b * bs
                        msent = [int(j) for j in masked if int(j) < first_eos and lo <= int(j) < lo + bs]
                        pred = [j - 1 - lo for j in msent if j - 1 >= lo]
                        own = [j - lo for j in msent]
                        record = condition == "real" and (pred or own)
                        rec["rows"], rec["out"] = (sorted(set(pred) | set(own)) if record else None), {}
                        hidden[b] = fwd(x[lo: lo + bs], caches[b], False)
                        if record:
                            pos = torch.arange(lo + bs)
                            cat = torch.full((lo + bs,), CATS.index("prompt"))
                            cat[(pos >= neural[0]) & (pos < neural[1])] = CATS.index("neural")
                            earlier = (pos >= p) & (pos < lo)
                            cat[earlier & (pos < first_eos)] = CATS.index("earlier_sentence")
                            cat[earlier & (pos >= first_eos)] = CATS.index("earlier_eos_pad")
                            same = (pos >= max(p, lo))
                            cat[same & (pos < first_eos)] = CATS.index("same_sentence")
                            cat[same & (pos >= first_eos)] = CATS.index("same_eos_pad")
                            cat[same & is_masked[: lo + bs].cpu()] = CATS.index("masks")
                            onehot = F.one_hot(cat, len(CATS)).float()  # [S, n_cats]
                            index = {row: n for n, row in enumerate(rec["rows"])}
                            for layer, probs in rec["out"].items():
                                per_cat = probs.float() @ onehot  # [rows, n_cats]
                                for qtype, rows in (("pred", pred), ("mask", own)):
                                    if rows:
                                        mass[r][qtype][layer] += per_cat[[index[x_] for x_ in rows]].sum(0)
                            n_rows[r]["pred"] += len(pred)
                            n_rows[r]["mask"] += len(own)
                    rec["rows"] = None
                    preds = torch.stack([hidden[(int(j) - 1) // bs][(int(j) - 1) % bs] if int(j) - 1 >= k0 * bs
                                         else prefix_last for j in masked])
                    with amp():
                        logits = lm_head(preds).float()
                    target = labels[masked]
                    ce = F.cross_entropy(logits, target, reduction="none")
                    right = logits.argmax(-1) == target
                    for j, c, ok in zip(masked.tolist(), ce.tolist(), right.tolist()):
                        k = "sentence" if j < first_eos else "eos" if j == first_eos else "pad"
                        a = acc[condition][r][k]
                        a[0] += c
                        a[1] += 1
                        a[2] += int(ok)
                    if selfcheck is None and condition == "real" and abs(r - 0.5) < 1e-9:
                        lab = torch.where(is_masked, labels, -100)
                        with diffusion_layout(decoder, training=True), amp():
                            h = decoder(inputs_embeds=torch.cat([x, clean])[None], labels=lab[None]).last_hidden_state
                        with amp():
                            ref = F.cross_entropy(lm_head(h[0, masked - 1]).float(), target, reduction="none")
                        selfcheck = {"trial": i, "ratio": r, "max_abs_ce_diff": float((ref - ce).abs().max()),
                                     "mean_ce_cache_path": float(ce.mean()), "mean_ce_training_layout": float(ref.mean())}
                        print(f"[context] self-check (cache path vs training layout): {json.dumps(selfcheck)}",
                              flush=True)
            if ti % 8 == 7:
                print(f"[context] {ti + 1}/{len(trials)} trials, {time.time() - t_start:.0f} s", flush=True)
    finally:
        modeling.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa

    ce_table = {c: {str(r): {k: {"n": v[1], "ce": round(v[0] / v[1], 4) if v[1] else None,
                                 "acc": round(v[2] / v[1], 4) if v[1] else None} for k, v in acc[c][r].items()}
                    for r in ratios} for c in acc}
    attention = {str(r): {q: {"n_queries": n_rows[r][q],
                              "per_layer": [dict(zip(CATS, (mass[r][q][layer] / max(n_rows[r][q], 1)).tolist()))
                                            for layer in range(n_layers)],
                              "mean_over_layers": dict(zip(CATS, (mass[r][q].mean(0) / max(n_rows[r][q], 1)).tolist()))}
                          for q in ("pred", "mask")} for r in ratios}
    print("\n[context] CE (nats/token) by mask ratio: real | shuffled neural input | S = shuffled - real", flush=True)
    print(f"{'ratio':>6} | {'sent real':>9} {'shuf':>7} {'S':>7} {'acc':>5} {'n':>5} | {'eos real':>8} {'shuf':>7} | "
          f"{'pad real':>8} {'shuf':>7} {'acc':>5} {'n':>5}", flush=True)
    for r in ratios:
        a, b = ce_table["real"][str(r)], ce_table["shuffled"][str(r)]

        def f(v):
            return f"{v:7.3f}" if v is not None else "      -"
        s_sent = (b["sentence"]["ce"] - a["sentence"]["ce"]) if a["sentence"]["ce"] is not None else None
        print(f"{r:>6} | {f(a['sentence']['ce']):>9} {f(b['sentence']['ce'])} {f(s_sent)} "
              f"{a['sentence']['acc'] or 0:5.2f} {a['sentence']['n']:5d} | {f(a['eos']['ce']):>8} {f(b['eos']['ce'])} | "
              f"{f(a['pad']['ce']):>8} {f(b['pad']['ce'])} {a['pad']['acc'] or 0:5.2f} {a['pad']['n']:5d}", flush=True)
    for q in ("pred", "mask"):
        print(f"\n[context] attention mass from masked sentence tokens ({q}: "
              f"{'their predictor positions j - 1' if q == 'pred' else 'the masked positions'}), mean over layers:",
              flush=True)
        print(f"{'ratio':>6} " + " ".join(f"{c[:14]:>14}" for c in CATS) + "  queries", flush=True)
        for r in ratios:
            mo = attention[str(r)][q]["mean_over_layers"]
            print(f"{r:>6} " + " ".join(f"{mo[c]:14.3f}" for c in CATS) + f"  {attention[str(r)][q]['n_queries']}",
                  flush=True)
    r_mid = min(ratios, key=lambda r: abs(r - 0.5))
    print(f"\n[context] attention mass per layer at ratio {r_mid} (pred):", flush=True)
    print(f"{'layer':>6} " + " ".join(f"{c[:14]:>14}" for c in CATS), flush=True)
    for layer, row in enumerate(attention[str(r_mid)]["pred"]["per_layer"]):
        print(f"{layer:>6} " + " ".join(f"{row[c]:14.3f}" for c in CATS), flush=True)
    return {"trials": trials, "ratios": list(ratios), "ce": ce_table, "attention": attention, "selfcheck": selfcheck,
            "seconds": round(time.time() - t_start, 1)}


def inspect(self, args):
    """Replaces BIT's Trainer.train (module docstring)."""
    from bit_experiments import fast_block_diffusion as fast, stage2
    from bit_experiments.textmetrics import word_errors
    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn
    from utils.eval_phoneme import get_model_inputs

    torch.set_num_threads(args.threads)
    parts = set(args.parts.split(","))
    model = self.accelerator.unwrap_model(self.model)
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    report = {"llm": args.llm, "ckpt": None, "device": str(device), "dtype": args.dtype}
    if args.ckpt:
        report["ckpt"] = {"dir": args.ckpt, **stage2.load_model_state(model, Path(args.ckpt))}
    model.eval()
    tok, dataset, config = DATA["tokenizer"], DATA["dataset"], self.config
    decoder, lm_head, embed = model._parts()

    # 1. vocabulary
    w = embed.weight.detach()
    same_as_mask = (w == w[model.mask_id]).all(1)
    mask_like = set(same_as_mask.nonzero().flatten().tolist())
    extra_norm = w[QWEN_ROWS:].norm(dim=1) if w.shape[0] > QWEN_ROWS else torch.zeros(0)
    vocab = {"tokenizer_vocab_size": tok.vocab_size, "len_tokenizer": len(tok), "tokenizer_eos": tok.eos_token_id,
             "tokenizer_pad": tok.pad_token_id, "model_eos": model.eos_id, "model_mask": model.mask_id,
             "model_filler": model.filler_id, "embedding_rows": int(w.shape[0]), "head_rows": int(lm_head.weight.shape[0]),
             "head_tied": lm_head.weight.data_ptr() == embed.weight.data_ptr(),
             "rows_bitwise_equal_to_mask": sorted(mask_like),
             "extra_rows_norm_mean": round(float(extra_norm.mean()), 4) if extra_norm.numel() else None,
             "ordinary_rows_norm_mean": round(float(w[:151643].norm(dim=1).mean()), 4)}
    report["vocab"] = vocab
    print(f"[inspect] vocab {json.dumps(vocab)}", flush=True)
    if tok.eos_token_id != model.eos_id:
        print(f"[inspect] WARNING tokenizer eos {tok.eos_token_id} != model eos {model.eos_id}: the sampler stops at the "
              "tokenizer's (stage2.generate passes it), forward pads targets with the model's", flush=True)

    def name(i):
        return tok.convert_ids_to_tokens(int(i)) if int(i) < len(tok) else f"<row {int(i)}>"

    # trials: the first --n val trials, with up to --seen seen-in-train ones
    collate = partial(pad_collate_fn, model_inputs=get_model_inputs(model), **config.method.dataloader_kwargs)
    val_ds = SpikingDatasetForDecoding(dataset, "val", **config.method.dataset_kwargs)
    refs = [r["sentence"] for r in dataset["val"]]
    seen = stage2.seen_mask(refs, [r["sentence"] for r in dataset["train"]])
    seen_idx = [int(i) for i in np.flatnonzero(seen)[: args.seen]]
    trials = seen_idx + [i for i in range(len(refs)) if i not in seen_idx][: args.n - len(seen_idx)]
    gen = dict(stage2.GREEDY, pad_token_id=tok.vocab_size, eos_token_id=tok.eos_token_id)

    # 2. decodes
    if "decode" in parts:
        report["decodes"] = []
        errs = np.zeros(2)
        for i in trials:
            inputs, _ = collate([val_ds[i]])
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            prompt = inputs["input_ids"][inputs["targets"] == -100].unsqueeze(0)
            gin = dict(inputs, input_ids=prompt, attention_mask=torch.ones_like(prompt))
            gin.pop("targets", None)
            model.trace = []
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                ids = model.generate(**gin, **gen)[0].tolist()
            rec = model.trace[-1] if model.trace else {"events": []}
            model.trace = None
            try:
                hyp = tok.decode(ids, skip_special_tokens=True).strip()
            except Exception:  # ids the tokenizer does not know (rows >= len(tokenizer))
                hyp = tok.decode([t for t in ids if t < len(tok)], skip_special_tokens=True).strip() + " <undecodable ids>"
            e, nw = word_errors(refs[i], hyp)
            errs += (e, nw)
            eos = gen["eos_token_id"]
            first_eos = ids.index(eos) if eos in ids else None
            passes = [ev for ev in rec["events"] if "m" in ev or "e" in ev]
            ref_len = len(tok(refs[i])["input_ids"])  # as BIT tokenizes it (sentence + eos, no leading space)
            d = {"trial": i, "seen": bool(seen[i]), "ref": refs[i], "hyp": hyp, "wer_errors": int(e), "ref_words": int(nw),
                 "ids": ids, "tokens": [name(t) for t in ids], "first_eos": first_eos, "ref_tokens": ref_len,
                 "stop": ("cap" if first_eos is None else "early" if first_eos < ref_len - 1
                          else "late" if first_eos > ref_len + 1 else "ok"),
                 "ids_ge_151936": [t for t in ids if t >= QWEN_ROWS], "mask_like_ids": [t for t in ids if t in mask_like],
                 "mask_id": model.mask_id in ids,
                 "special_ids": sorted({t for t in ids if 151643 <= t < QWEN_ROWS and t != eos}),
                 "n_passes": len(passes), "n_blocks": 1 + sum("first" in ev for ev in rec["events"] if ev.get("b", 0) > 0),
                 "prefix_len": rec.get("prefix_len"), "tail": rec.get("tail"), "seconds": round(time.time() - t0, 1)}
            report["decodes"].append(d)
            flags = [k for k in ("ids_ge_151936", "mask_like_ids", "special_ids") if d[k]] + (["mask_id"] if d["mask_id"] else [])
            print(f"\n[decode] trial {i}{' (seen)' if d['seen'] else ''}: WER {e}/{nw}, stop {d['stop']} "
                  f"(eos at {first_eos}, ref {ref_len} tokens + eos), {d['n_passes']} passes, {d['n_blocks']} block(s), "
                  f"prefix {d['prefix_len']} tail {d['tail']}{', FLAGS ' + str(flags) if flags else ''}", flush=True)
            print(f"  ref: {refs[i]}\n  hyp: {hyp}\n  ids: {ids}\n  tok: {d['tokens']}", flush=True)
            for n, ev in enumerate(passes[: args.passes]):
                commits = [f"{pos}:{name(t)!r}@{p:.2f}" for pos, t, p, c in ev.get("m", []) if c]
                print(f"  pass {n + 1} (block {ev.get('b')}): {' '.join(commits) or '-'}", flush=True)
        report["decode_wer"] = float(errs[0] / max(errs[1], 1))
        report["decode_exact"] = float(np.mean([d["wer_errors"] == 0 for d in report["decodes"]]))
        print(f"\n[decode] WER over these {len(trials)} trials {report['decode_wer']:.3f}, exact {report['decode_exact']:.3f}",
              flush=True)

    # 3. teacher-forced view, the same trials
    if "teacher" in parts:
        report["teacher_forced"] = {"trials": [], "summary": None}
        stats = []
        for i in trials:
            inputs, _ = collate([val_ds[i]])
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            clean, labels = step_inputs(model, inputs, device, dtype)
            seed = 5000 + i
            torch.manual_seed(seed)
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                ce, n, logits, targets = fast.fast_block_diffusion_ce(decoder, lm_head, embed, clean, labels,
                                                                      model.block_size, model.mask_id, attention="dense")
            rows, js = supervised_positions(labels, model.block_size, seed)
            assert rows.numel() == int(n) and torch.equal(labels.reshape(-1)[js + rows % labels.shape[0] * labels.shape[1]],
                                                          targets), "position re-draw does not match the step"
            p, first_eos = classes(labels[0], {model.eos_id, tok.eos_token_id})
            st, cls, arg, p_max = split_ce(logits, targets, rows, js, first_eos)
            stats.append(st)
            sent = [f"{int(j) - p}:{name(t)!r}->{name(a)!r}@{float(q):.2f}"
                    for j, t, a, q, c in zip(js, targets, arg, p_max, cls) if c != "pad"]
            pad = [(name(a), round(float(q), 2)) for a, q, c in zip(arg, p_max, cls) if c == "pad"][:6]
            report["teacher_forced"]["trials"].append({"trial": i, "classes": st})
            print(f"\n[teacher] trial {i}: " + ", ".join(f"{c} n {s['n']} ce {s['ce_sum'] / s['n']:.3f} acc {s['acc']:.2f}"
                                                       for c, s in st.items()), flush=True)
            print(f"  masked sentence/eos positions (offset:target->argmax@p): {' '.join(sent)}", flush=True)
            print(f"  first padding argmaxes: {pad}", flush=True)
        report["teacher_forced"]["summary"] = merge(stats)
        print(f"\n[teacher] summary {json.dumps(report['teacher_forced']['summary'])}", flush=True)

    # 5. context use: CE against the mask ratio, with the neural input shuffled, and attention mass
    if "context" in parts:
        ctx_trials = list(range(min(args.trials, len(refs))))
        report["context"] = context_use(model, val_ds, collate, ctx_trials, args.ratios, device, dtype,
                                        {model.eos_id, tok.eos_token_id})

    # 4. zero-shot split (untrained Instruct and Aero, text only)
    if "zeroshot" in parts and args.zero_shot:
        report["zero_shot"] = zero_shot_split(args.zero_shot)
    raise InspectDone(report)


@torch.no_grad()
def zero_shot_split(n: int, tbf=None, tok=None) -> dict:
    """text_blank_fill_check.py's diffusion CE (same sentences, prompt, masks) split by class, for Instruct and Aero.
    tbf, tok: that module and its tokenizer (loaded here by default)."""
    if tbf is None:
        spec = importlib.util.spec_from_file_location("text_blank_fill_check",
                                                      ROOT / "scripts" / "text_blank_fill_check.py")
        tbf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tbf)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tbf.HF / "Qwen2.5-1.5B-Instruct")
    refs, data = tbf.rows(tok, n)
    per = {}
    for model_name, path in (("instruct", "Qwen2.5-1.5B-Instruct"), ("aero", "Aero-1-Audio")):
        llm = tbf.BCI_DiffusionLLM.from_pretrained(tbf.HF / path).eval()
        decoder, lm_head, embed = llm.model, llm.lm_head, llm.get_input_embeddings()
        per[model_name] = []
        for i, (prompt, target, n_real) in enumerate(data):
            ids = torch.tensor([prompt + target])
            labels = torch.tensor([[-100] * len(prompt) + target])
            torch.manual_seed(1000 + i)  # text_blank_fill_check's seed: the same masks
            ce, k, logits, targets = tbf.block_diffusion_ce(decoder, lm_head, embed, embed(ids), labels, tbf.BLOCK,
                                                            tbf.MASK_ID)
            rows, js = supervised_positions(labels, tbf.BLOCK, 1000 + i)
            assert rows.numel() == int(k)
            st, *_ = split_ce(logits, targets, rows, js, len(prompt) + n_real - 1)
            per[model_name].append(st)
        print(f"[zero-shot] {model_name}: total {sum(s[c]['ce_sum'] for s in per[model_name] for c in s) / sum(s[c]['n'] for s in per[model_name] for c in s):.4f} "
              f"nats/token | {json.dumps(merge(per[model_name]))}", flush=True)
        del llm
    out = {m: merge(v) for m, v in per.items()}
    rng = np.random.default_rng(0)
    for c in ("sentence", "eos", "pad"):
        a = np.array([s[c]["ce_sum"] if c in s else 0.0 for s in per["aero"]])
        b = np.array([s[c]["ce_sum"] if c in s else 0.0 for s in per["instruct"]])
        k = np.array([s[c]["n"] if c in s else 0 for s in per["aero"]])
        if k.sum() == 0:
            continue
        idx = rng.integers(0, len(k), (5000, len(k)))
        boot = (a[idx].sum(1) - b[idx].sum(1)) / np.maximum(k[idx].sum(1), 1)
        d = (a.sum() - b.sum()) / k.sum()
        out[f"aero_minus_instruct_{c}"] = [round(float(d), 4), round(float(np.percentile(boot, 2.5)), 4),
                                          round(float(np.percentile(boot, 97.5)), 4)]
        print(f"[zero-shot] aero - instruct, {c}: {d:+.4f} [{np.percentile(boot, 2.5):+.4f}, "
              f"{np.percentile(boot, 97.5):+.4f}] nats/token over {k.sum()} tokens", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="aero-1-audio-diffusion")
    ap.add_argument("--init", default="checkpoints/ssl_human/BEST_core")
    ap.add_argument("--ckpt", help="trained Stage 2 checkpoint dir (e.g. the fast run's BEST)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=12, help="val trials")
    ap.add_argument("--seen", type=int, default=4, help="of which seen-in-train (at most)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--zero-shot", type=int, default=100, help="sentences for the zero-shot split (0: skip)")
    ap.add_argument("--passes", type=int, default=8, help="passes to print per decode")
    ap.add_argument("--parts", default="decode,teacher,context,zeroshot",
                    help="comma list of decode, teacher, context, zeroshot (the vocabulary is always printed)")
    ap.add_argument("--trials", type=int, default=64, help="val trials for the context part")
    ap.add_argument("--ratios", type=lambda v: [float(x) for x in v.split(",")], default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    help="mask ratios for the context part")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if any(r <= 0 for r in args.ratios):
        print(f"[inspect] dropping mask ratios <= 0 (nothing is masked): {[r for r in args.ratios if r <= 0]}")
        args.ratios = [r for r in args.ratios if r > 0]

    spec = importlib.util.spec_from_file_location("train_stage2", ROOT / "scripts" / "train_stage2.py")
    train_stage2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_stage2)  # puts src/ and BIT on sys.path
    from models.trainer import Trainer

    load = train_stage2.load_dataset

    def capture(*a, **kw):  # keep the dataset and tokenizer train_stage2 loads
        DATA["dataset"], DATA["tokenizer"] = out = load(*a, **kw)
        return out

    train_stage2.load_dataset = capture
    name = f"aero_decode_inspect_{time.strftime('%Y%m%d_%H%M%S')}"
    Trainer.train = lambda self: inspect(self, args)
    sys.argv = ["train_stage2.py", "--runtime", "reference", "--name", name, "--arm", "B", "--init", args.init, "--seed", str(args.seed),
                "--llm", args.llm, "--epochs", "1", "--eval-every", "1"]
    try:
        train_stage2.main()
        raise SystemExit("train_stage2.main returned without running the inspection")
    except InspectDone as done:
        report = done.results
    finally:
        for d in (ROOT / "results" / "stage2" / name, ROOT / "checkpoints" / "stage2" / name):
            if d.name.startswith("aero_decode_inspect_"):
                shutil.rmtree(d, ignore_errors=True)
    out = ROOT / "results" / "aero_inspect" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n[inspect] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
