#!/usr/bin/env python
"""Decoding-order sweep for a trained block-diffusion Stage 2 run (eval only; DESIGN.md §8, run 3).

    <bit env>/python scripts/decoding_order_sweep.py --run B_ssl_fast-dllm-v2-1.5b_s42 [--sbs 8 1 4 32]
        [--splits val test] [--limit N] [--out DIR] [--device cuda]

Same weights, same greedy proposals and threshold; only the sub-block ("chunk") size of Fast-dLLM v2's sampler
changes. The decoder fills each 32-token block chunk by chunk, left to right, and inside a chunk commits every masked
position whose confidence clears the threshold (at least the most confident one) per forward pass:
  1 = strictly left to right, one token per pass; 4 / 8 = chunked (8 = the model card's and run 3's setting);
  32 = the whole block at once, in confidence order.
Training masked tokens at random ratios within each block, so every order here is one the model was trained for.

The model is built and loaded exactly as train_stage2.py --eval-only does (BIT's sentence config, the run's LLM entry,
stage2.fix_padding, BEST via stage2.load_model_state); --run must be a finished diffusion run. The chunk size the run
itself used is decoded first and must reproduce its summary.json greedy WER (reported as `matches_run`).
Every decode also records its passes (traced_denoise): which masked positions held which proposal at what probability,
and which were committed, for visualising the denoising.
Then, at the run's own chunk size, the revision-headroom check (denoise_and_reconsider, revision_report): each decoded
token is hidden and re-predicted with the rest of its block visible; how often would the model change its mind, and
would that fix or break the sentence? (--no-revision-check skips it.)
Last, the re-masking sampler (remask_denoise; --remask 0.9 3 = threshold, rounds; rounds 0 skips it): revision with
no training, at the run's own chunk size, compared sentence by sentence with that chunk size's plain decode.
Outputs: results/stage2/<run>/decoding_order/{summary.json, decodes/<split>_greedy_real_sbs<k>[_remask<t>x<r>].csv
+ .meta.json, traces/<same>.json.gz, revision_check/<split>.json.gz}.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import types
from functools import partial
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

try:
    from bit_experiments import stage1, stage2, text_llm  # noqa: E402
except ImportError:  # the GPU host's pre-restructure layout
    from kunzdecoder import stage1, stage2, text_llm  # noqa: E402

DATASET = "brandman_2024_text"
RESULTS = ROOT / "results" / ("phase4" if (ROOT / "results" / "phase4").is_dir() else "") / "stage2"


def load_dataset(config):
    """train_stage2.load_dataset without the label fraction (evaluation only)."""
    from registry import dataset_registry
    from transformers import AutoTokenizer

    data_config = config.data
    data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{DATASET}"
    module = __import__(dataset_registry[DATASET], fromlist=["load_data"])
    ds = module.load_data(**data_config.dataset)
    ds.pop("holdout", None)  # no labels; never used
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_path, add_bos_token=False, add_eos_token=False)
    return text_llm.create_llm_labels(ds, tokenizer, data_config.prompt), tokenizer


TRACES: list[dict] = []


def traced_denoise(self, prefix, max_new_tokens, do_sample, top_p, temperature, eos) -> list[int]:
    """BCI_DiffusionLLM._denoise, line for line, plus a record of every pass in TRACES. The sweep decodes with this
    copy throughout, so `matches_run` (the run's own chunk size vs its summary.json) also checks that it is unchanged.
    Record: prefix_len, tail (prefix tokens sharing the first block), out (the returned ids), and events, in order:
    {"b": block, "first": [pos, token]} = a block's first token, the AR next token of the committed block before it;
    {"b": block, "m": [[pos, proposal, probability, committed 0/1], ...]} = one pass, every masked position of the
    sub-block. pos counts output tokens from 0."""
    decoder, lm_head, embed = self._parts()
    bs, sbs, threshold = self.block_size, self.sampler["small_block_size"], self.sampler["threshold"]
    cache = DynamicCache()
    rec = {"prefix_len": int(prefix.shape[0]), "events": []}
    TRACES.append(rec)

    def run(x, commit):
        h = decoder(inputs_embeds=x[None], past_key_values=cache, use_cache=True, update_past_key_values=commit,
                    block_size=bs).last_hidden_state
        logits = lm_head(h)[0].float()
        logits[:, self.mask_id] = -torch.inf  # an accepted mask proposal would leave its slot masked forever
        return logits

    n_full = prefix.shape[0] // bs * bs
    tail = prefix[n_full:]
    rec["tail"] = int(tail.shape[0])
    slots = torch.full((bs - tail.shape[0],), self.mask_id, device=prefix.device)
    if n_full:
        logits = run(prefix[:n_full], commit=True)
        if tail.shape[0] == 0:
            slots[0] = logits[-1].argmax()  # the block's first token is the AR next token, as Fast-dLLM v2
            rec["events"].append({"b": 0, "first": [0, int(slots[0])]})

    out: list[int] = []
    block = 0
    while True:
        for start in range(0, bs, sbs):
            lo, hi = max(start - tail.shape[0], 0), max(start + sbs - tail.shape[0], 0)
            while bool((slots[lo:hi] == self.mask_id).any()) and not self._finished(slots, eos):
                logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=False)
                logits = torch.cat([logits[:1], logits[:-1]])[tail.shape[0]:][lo:hi]  # token shift
                proposal, confidence = self._propose(logits, do_sample, top_p, temperature)
                is_masked = slots[lo:hi] == self.mask_id
                probability = confidence
                confidence = torch.where(is_masked, confidence, -torch.inf)
                accept = confidence > threshold
                accept[confidence.argmax()] = True
                accept &= is_masked
                slots[lo:hi] = torch.where(accept, proposal, slots[lo:hi])
                rec["events"].append({"b": block, "m": [
                    [len(out) + lo + i, int(proposal[i]), round(float(probability[i]), 4), int(accept[i])]
                    for i in is_masked.nonzero().flatten().tolist()]})
            if self._finished(slots, eos):
                break
        out += slots.tolist()
        if eos in out or len(out) >= max_new_tokens:
            break
        logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=True)
        tail = prefix[:0]
        slots = torch.full((bs,), self.mask_id, device=prefix.device)
        slots[0] = logits[-1].argmax()
        block += 1
        rec["events"].append({"b": block, "first": [len(out), int(slots[0])]})
    if eos in out:
        out = out[: out.index(eos) + 1]
    rec["out"] = out[:max_new_tokens]
    return out[:max_new_tokens]


def remask_denoise(self, prefix, max_new_tokens, do_sample, top_p, temperature, eos, remask: float,
                   rounds: int) -> list[int]:
    """traced_denoise plus re-masking (Mask-Predict / Mask-CTC style; no training; user, 2026-10-21): when a block is
    done, every committed token up to the first EOS whose probability when committed was below `remask` is masked
    again and the same sampler re-fills the block, now with the rest of it visible; up to `rounds` times, stopping
    early when a round re-fills exactly what it masked. The rule of BCI_DiffusionLLM._denoise's remask_threshold /
    remask_rounds (src/bit_experiments/diffusion_llm.py, editing off). rounds 0 = traced_denoise. A re-masking round
    adds the trace event {"b": block, "r": [[pos, token, probability when committed], ...]}."""
    decoder, lm_head, embed = self._parts()
    bs, sbs, threshold = self.block_size, self.sampler["small_block_size"], self.sampler["threshold"]
    cache = DynamicCache()
    rec = {"prefix_len": int(prefix.shape[0]), "events": []}
    TRACES.append(rec)

    def run(x, commit):
        h = decoder(inputs_embeds=x[None], past_key_values=cache, use_cache=True, update_past_key_values=commit,
                    block_size=bs).last_hidden_state
        logits = lm_head(h)[0].float()
        logits[:, self.mask_id] = -torch.inf
        return logits

    n_full = prefix.shape[0] // bs * bs
    tail = prefix[n_full:]
    rec["tail"] = int(tail.shape[0])
    slots = torch.full((bs - tail.shape[0],), self.mask_id, device=prefix.device)
    if n_full:
        logits = run(prefix[:n_full], commit=True)
        if tail.shape[0] == 0:
            slots[0] = logits[-1].argmax()
            rec["events"].append({"b": 0, "first": [0, int(slots[0])]})

    out: list[int] = []
    block = 0
    while True:
        chunks = [(max(s - tail.shape[0], 0), max(s + sbs - tail.shape[0], 0)) for s in range(0, bs, sbs)]
        conf = torch.ones(slots.shape, device=slots.device)  # probability when committed (AR first token: 1)
        before = None
        for round_ in range(rounds + 1):  # the block's denoising, then up to `rounds` re-mask + re-fill rounds
            while True:  # one pass each: fill the first sub-block that has a mask, unless finished
                finished = self._finished(slots, eos)
                lo, hi = next(((lo, hi) for lo, hi in chunks
                               if not finished and bool((slots[lo:hi] == self.mask_id).any())), (0, 0))
                if hi == lo:
                    break
                logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=False)
                logits = torch.cat([logits[:1], logits[:-1]])[tail.shape[0]:][lo:hi]  # token shift
                proposal, probability = self._propose(logits, do_sample, top_p, temperature)
                is_masked = slots[lo:hi] == self.mask_id
                confidence = torch.where(is_masked, probability, -torch.inf)
                accept = confidence > threshold
                accept[confidence.argmax()] = True
                accept &= is_masked
                rec["events"].append({"b": block, "m": [
                    [len(out) + lo + i, int(proposal[i]), round(float(probability[i]), 4), int(accept[i])]
                    for i in is_masked.nonzero().flatten().tolist()]})
                slots[lo:hi] = torch.where(accept, proposal, slots[lo:hi])
                conf[lo:hi] = torch.where(accept, probability, conf[lo:hi])
            if round_ == rounds or (before is not None and torch.equal(slots, before)):
                break  # out of rounds, or the last re-fill changed nothing
            low = (slots != self.mask_id) & (conf < remask)
            eos_at = (slots == eos).nonzero()
            if eos_at.numel():
                low[int(eos_at[0]) + 1:] = False  # after the first EOS: padding
            if not bool(low.any()):
                break
            rec["events"].append({"b": block, "r": [[len(out) + i, int(slots[i]), round(float(conf[i]), 4)]
                                                    for i in low.nonzero().flatten().tolist()]})
            before = slots.clone()
            slots[low] = self.mask_id
        out += slots.tolist()
        if eos in out or len(out) >= max_new_tokens:
            break
        logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=True)
        tail = prefix[:0]
        slots = torch.full((bs,), self.mask_id, device=prefix.device)
        slots[0] = logits[-1].argmax()
        block += 1
        rec["events"].append({"b": block, "first": [len(out), int(slots[0])]})
    if eos in out:
        out = out[: out.index(eos) + 1]
    rec["out"] = out[:max_new_tokens]
    return out[:max_new_tokens]


REVISIONS: list[dict] = []
EDIT_THRESHOLDS = (0.5, 0.7, 0.9)


def denoise_and_reconsider(self, prefix, max_new_tokens, do_sample, top_p, temperature, eos) -> list[int]:
    """traced_denoise, then the revision-headroom check on its output (same prefix; user, 2026-10-21): every output
    token up to the first EOS is hidden on its own and re-predicted with the rest of its block as decoded (after the
    EOS: EOS, as training pads) and the earlier blocks committed, which is what a within-block editing sampler would
    see. A block's first token when no prompt shares the block is the AR next token, not denoised, and is skipped.
    REVISIONS gets [pos, decoded token, its probability, top prediction, top probability] per position; the returned
    ids are traced_denoise's."""
    out = traced_denoise(self, prefix, max_new_tokens, do_sample, top_p, temperature, eos)
    decoder, lm_head, embed = self._parts()
    bs = self.block_size
    cache = DynamicCache()

    def run(x, commit):
        h = decoder(inputs_embeds=x[None], past_key_values=cache, use_cache=True, update_past_key_values=commit,
                    block_size=bs).last_hidden_state
        logits = lm_head(h)[0].float()
        logits[:, self.mask_id] = -torch.inf
        return logits

    n_full = prefix.shape[0] // bs * bs
    tail = prefix[n_full:]
    k0 = bs - tail.shape[0]
    n_eval = out.index(eos) + 1 if eos in out else len(out)
    total = k0 + max(0, -(-(len(out) - k0) // bs)) * bs
    seq = torch.tensor(out + [eos] * (total - len(out)), device=prefix.device)
    if n_full:
        run(prefix[:n_full], commit=True)
    rows, start, ctx = [], 0, tail
    while start < n_eval:
        size = k0 if start == 0 else bs
        block = seq[start:start + size]
        for j in range(min(size, n_eval - start)):
            if j == 0 and ctx.shape[0] == 0:
                continue  # the AR first token of a block
            ids = block.clone()
            ids[j] = self.mask_id
            logits = run(torch.cat([ctx, embed(ids).to(ctx.dtype)]), commit=False)
            probs = torch.cat([logits[:1], logits[:-1]])[ctx.shape[0] + j].softmax(-1)  # token shift
            top_prob, top = probs.max(-1)
            rows.append([start + j, int(block[j]), round(float(probs[block[j]]), 4), int(top), round(float(top_prob), 4)])
        run(torch.cat([ctx, embed(block).to(ctx.dtype)]), commit=True)
        start, ctx = start + size, prefix[:0]
    REVISIONS.append({"out": out, "n_eval": n_eval, "pos": rows})
    return out


def revision_report(tokenizer, refs, hyps, seen, eos) -> tuple[dict, list[dict]]:
    """Single edits (one re-predicted token swapped in): how many would fix a sentence (fewer word errors) or break it.
    One-round edits (every position whose top prediction differs and clears a threshold, all at once): corpus WER as if
    an editing sampler made one pass at the end of the block. Oracle: only the individually fixing edits, all at once."""
    if len(REVISIONS) != len(hyps):
        raise RuntimeError(f"{len(REVISIONS)} revision records for {len(hyps)} decodes")

    def text(ids):
        ids = ids[: ids.index(eos)] if eos in ids else ids
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    counts = {"positions": 0, "changed": 0, "fix": 0, "break": 0, "neutral": 0}
    one_round = {t: [] for t in EDIT_THRESHOLDS}
    oracle, trials = [], []
    for r, ref, hyp in zip(REVISIONS, refs, hyps, strict=True):
        if text(list(r["out"])) != hyp:
            raise RuntimeError(f"revision record does not reproduce the decode: {hyp!r}")
        base = stage2.word_errors(ref, hyp)[0]
        edits = []
        for pos, tok, p_tok, top, p_top in r["pos"]:
            counts["positions"] += 1
            if top == tok:
                continue
            ids = list(r["out"])
            ids[pos] = top
            delta = stage2.word_errors(ref, text(ids))[0] - base
            kind = "fix" if delta < 0 else "break" if delta > 0 else "neutral"
            counts["changed"] += 1
            counts[kind] += 1
            edits.append({"pos": pos, "from": tok, "to": top, "p_from": p_tok, "p_to": p_top, "delta": int(delta)})
        for t in EDIT_THRESHOLDS:
            ids = list(r["out"])
            for e in edits:
                if e["p_to"] > t:
                    ids[e["pos"]] = e["to"]
            one_round[t].append(text(ids))
        ids = list(r["out"])
        for e in edits:
            if e["delta"] < 0:
                ids[e["pos"]] = e["to"]
        oracle.append(text(ids))
        trials.append({"ref": ref, "hyp": hyp, "edits": edits})
    report = counts | {"wer_decoded": stage2.score(refs, hyps, seen)["wer"],
                       "wer_one_round": {str(t): stage2.score(refs, h, seen)["wer"] for t, h in one_round.items()},
                       "wer_oracle_fixes_only": stage2.score(refs, oracle, seen)["wer"]}
    return report, trials


def write_traces(path: Path, tokenizer, refs, hyps, seen, meta):
    """TRACES (one per trial, in decode order) with the reference, the hypothesis and the text of every token used."""
    if len(TRACES) != len(hyps):
        raise RuntimeError(f"{len(TRACES)} traces for {len(hyps)} decodes")
    ids = {t for r in TRACES for e in r["events"] for t in ([e["first"][1]] if "first" in e else
                                                             [m[1] for m in e.get("m", e.get("r", []))])}
    ids |= {t for r in TRACES for t in r["out"]}
    trials = [r | {"i": i, "ref": ref, "hyp": hyp, "seen": bool(s)}
              for i, (r, ref, hyp, s) in enumerate(zip(TRACES, refs, hyps, seen))]
    vocab = {str(t): tokenizer.decode([t]) for t in sorted(ids)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"meta": meta, "vocab": vocab, "trials": trials}, f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a finished Stage 2 run with a diffusion LLM")
    ap.add_argument("--sbs", type=int, nargs="+", default=[8, 1, 4, 32], help="sub-block sizes (divisors of 32)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    ap.add_argument("--out", type=Path, help="default: results/stage2/<run>/decoding_order")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-revision-check", action="store_true", help="skip the revision-headroom check")
    ap.add_argument("--remask", type=float, nargs=2, default=[0.9, 3], metavar=("THRESHOLD", "ROUNDS"),
                    help="re-masking sampler at the run's chunk size (remask_denoise); ROUNDS 0 = off")
    args = ap.parse_args()
    t0 = time.time()
    run_dir = RESULTS / args.run
    run = json.loads((run_dir / "run.json").read_text())
    if run["config"]["model"]["model_class"] != "BCI_DiffusionLLM":
        ap.error(f"{args.run} is not a block-diffusion run")
    ckpt = ROOT / "checkpoints" / "stage2" / args.run / "BEST"
    out = args.out or run_dir / "decoding_order"
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(BIT)  # BIT opens configs/ by relative path

    from models.trainer import Trainer
    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn
    from utils.eval_phoneme import get_model_inputs

    rargs = run["args"]
    config = stage1.bit_sentence_config()
    config["seed"] = rargs["seed"]
    config["savestring"] = args.run
    config["dirs"]["checkpoint_dir"] = str(ROOT / "checkpoints" / "stage2")
    config["log_to_wandb"] = False
    text_llm.use_llm(config, rargs["llm"])
    if rargs.get("prompt") is not None:
        config["data"]["prompt"] = rargs["prompt"]
    dataset, tokenizer = load_dataset(config)
    if args.limit:
        dataset = {k: v[: args.limit] for k, v in dataset.items()}
    train_sentences = [r["sentence"] for r in dataset["train"]]

    class EvalTrainer(Trainer):
        def set_model(self, model, extra_model_kwargs=None):
            super().set_model(model, extra_model_kwargs)
            self.padding_report = stage2.fix_padding(self.model)  # as every Stage 2 run

    trainer = EvalTrainer(config, dataset=dataset, metric_fns={}, extra_model_kwargs={"features": "all"},
                          eval_mode=True)
    model = trainer.accelerator.unwrap_model(trainer.model)
    load = stage2.load_model_state(model, ckpt)
    model.to(args.device)
    collate = partial(pad_collate_fn, model_inputs=get_model_inputs(model), **config.method.dataloader_kwargs)
    splits = {s: SpikingDatasetForDecoding(dataset, s, **config.method.dataset_kwargs) for s in args.splits}
    refs = {s: [r["sentence"] for r in dataset[s]] for s in splits}
    seen = {s: stage2.seen_mask(refs[s], train_sentences) for s in splits}

    model._denoise = types.MethodType(traced_denoise, model)
    sampler0 = dict(model.sampler)
    run_summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    passes = [0]
    hook = model._parts()[0].register_forward_hook(lambda *_: passes.__setitem__(0, passes[0] + 1))
    summary = {"run": args.run, "checkpoint": str(ckpt), "load": load, "sampler": sampler0,
               "padding": trainer.padding_report, "limit": args.limit, "results": {}}
    decoded = {}
    for sbs in args.sbs:
        if model.block_size % sbs:
            ap.error(f"--sbs {sbs} does not divide the block size {model.block_size}")
        model.sampler["small_block_size"] = sbs
        for split in args.splits:
            passes[0], t = 0, time.time()
            TRACES.clear()
            hyps = stage2.generate(model, splits[split], collate, tokenizer, "greedy", device=args.device)
            res = stage2.score(refs[split], hyps, seen[split])
            res |= {"sec_per_trial": round((time.time() - t) / len(hyps), 4),
                    "decoder_passes_per_trial": round(passes[0] / len(hyps), 2)}
            ref_run = run_summary.get("results", {}).get(f"{split}_greedy_real")
            if sbs == sampler0["small_block_size"] and ref_run and not args.limit:
                res["matches_run"] = abs(res["wer"] - ref_run["wer"]) < 1e-9
                res["run_wer"] = ref_run["wer"]
            summary["results"][f"{split}_greedy_real_sbs{sbs}"] = res
            decoded[f"{split}_greedy_real_sbs{sbs}"] = hyps
            name = args.run[: -len(f"_s{rargs['seed']}")] if args.run.endswith(f"_s{rargs['seed']}") else args.run
            meta = {"model": f"stage2 {name}-sbs{sbs}_s{rargs['seed']}", "arm": rargs["arm"], "seed": rargs["seed"],
                    "split": f"{DATASET}:{split}", "decoding": "greedy", "condition": "real",
                    "init": rargs.get("init"), "checkpoint": str(ckpt), "label_fraction": rargs["label_fraction"],
                    "llm": rargs["llm"], "prompt": config.data.prompt,
                    "sampler": dict(model.sampler), "source_run": args.run}
            stage2.write_decodes(out / "decodes" / f"{split}_greedy_real_sbs{sbs}.csv", refs[split], hyps, seen[split],
                                 meta)
            write_traces(out / "traces" / f"{split}_greedy_real_sbs{sbs}.json.gz", tokenizer, refs[split], hyps,
                         seen[split], meta)
            print(f"[sweep] sbs {sbs} {split}: {json.dumps(res)}", flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
    hook.remove()
    model.sampler.update(sampler0)

    if not args.no_revision_check:  # at the run's own chunk size (denoise_and_reconsider)
        model._denoise = types.MethodType(denoise_and_reconsider, model)
        summary["revision_check"] = {}
        for split in args.splits:
            REVISIONS.clear()
            TRACES.clear()
            t = time.time()
            hyps = stage2.generate(model, splits[split], collate, tokenizer, "greedy", device=args.device)
            report, trials = revision_report(tokenizer, refs[split], hyps, seen[split], tokenizer.eos_token_id)
            key = f"{split}_greedy_real_sbs{sampler0['small_block_size']}"
            report["same_decodes_as_sweep"] = (hyps == decoded[key]) if key in decoded else None
            report["minutes"] = round((time.time() - t) / 60, 2)
            summary["revision_check"][split] = report
            ids = {e[k] for tr in trials for e in tr["edits"] for k in ("from", "to")}
            path = out / "revision_check" / f"{split}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt") as f:
                json.dump({"meta": {"source_run": args.run, "checkpoint": str(ckpt), "split": f"{DATASET}:{split}",
                                    "sampler": sampler0, "thresholds": EDIT_THRESHOLDS},
                           "vocab": {str(i): tokenizer.decode([i]) for i in sorted(ids)}, "trials": trials}, f)
            print(f"[revision] {split}: {json.dumps(report)}", flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))

    remask, rounds = args.remask[0], int(args.remask[1])
    if rounds:  # the re-masking sampler, at the run's own chunk size, vs that chunk size's plain decode
        sbs = sampler0["small_block_size"]
        model._denoise = types.MethodType(partial(remask_denoise, remask=remask, rounds=rounds), model)
        hook = model._parts()[0].register_forward_hook(lambda *_: passes.__setitem__(0, passes[0] + 1))
        tag = f"sbs{sbs}_remask{remask:g}x{rounds}"
        summary["remask"] = {"threshold": remask, "rounds": rounds, "results": {}}
        for split in args.splits:
            TRACES.clear()
            passes[0], t = 0, time.time()
            hyps = stage2.generate(model, splits[split], collate, tokenizer, "greedy", device=args.device)
            res = stage2.score(refs[split], hyps, seen[split])
            res |= {"sec_per_trial": round((time.time() - t) / len(hyps), 4),
                    "decoder_passes_per_trial": round(passes[0] / len(hyps), 2),
                    "trials_remasked": sum(any("r" in e for e in r["events"]) for r in TRACES),
                    "tokens_remasked": sum(len(e["r"]) for r in TRACES for e in r["events"] if "r" in e)}
            plain = decoded.get(f"{split}_greedy_real_sbs{sbs}")
            if plain is not None:
                diff = [(stage2.word_errors(r, h)[0], stage2.word_errors(r, p)[0])
                        for r, h, p in zip(refs[split], hyps, plain, strict=True) if h != p]
                res |= {"wer_plain": stage2.score(refs[split], plain, seen[split])["wer"],
                        "sentences_changed": len(diff), "sentences_better": sum(a < b for a, b in diff),
                        "sentences_worse": sum(a > b for a, b in diff)}
            summary["remask"]["results"][split] = res
            meta = {"model": f"stage2 {args.run} {tag}", "split": f"{DATASET}:{split}", "decoding": "greedy",
                    "condition": "real", "checkpoint": str(ckpt), "sampler": dict(model.sampler),
                    "remask_threshold": remask, "remask_rounds": rounds, "source_run": args.run}
            stage2.write_decodes(out / "decodes" / f"{split}_greedy_real_{tag}.csv", refs[split], hyps, seen[split],
                                 meta)
            write_traces(out / "traces" / f"{split}_greedy_real_{tag}.json.gz", tokenizer, refs[split], hyps,
                         seen[split], meta)
            print(f"[remask] {split}: {json.dumps(res)}", flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
        hook.remove()
    summary["minutes"] = round((time.time() - t0) / 60, 2)
    if torch.cuda.is_available():
        summary["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
