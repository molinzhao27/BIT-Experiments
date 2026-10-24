#!/usr/bin/env python
"""Decisive GPU test for the fast step's training failure (2026-10-24): plain Aero diffusion reached val WER 42% at
epoch 5, the fast step 108.6%, although the one-checkpoint eval-mode check (gpu_fast_step_check.py) passed. About
15-20 minutes under the GPU lock, with the GPU free (the plain step needs ~80 GB):

    /workspace/envs/bit/bin/python scripts/gpu_fast_train_check.py --llm aero-1-audio-diffusion \
        [--init checkpoints/ssl_human/BEST_core] [--shapes 6] [--static-shapes 3] [--steps 250] [--probe-every 25] \
        [--variants plain,fast,fast_static] [--sched-epochs 150]

Leading hypothesis (from reading the code; nothing else differs in train mode): the packed step's flex mask_mod
captures tensors whose length is the packed length T, which changes from batch to batch. After the second T, Dynamo
recompiles packed_flex_attention with dynamic shapes, and every later T runs on that graph (the failed run logged
2 recompile lines). The plain step's mask_mod captures only an int (the batch length), so its shapes stay static.
The test can also falsify this and locate the bug elsewhere.

A. The kernel across packed lengths, as the first flex work in a fresh process, so its compile history is a training
   run's: static at the first T, dynamic from the second. It picks real training batches until --shapes distinct T
   are covered, and runs each in eval mode (no dropout), grads on, one seed per batch. It compares the packed step's
   compiled kernel against the same step with dense SDPA (no flex, no compile): loss, logits cosine, per-group
   gradient cosine and norm ratio. The kernels:
   - "dynamic": torch.compile's default, as in the failed run, with grads and without (the val-loss graph);
   - "static": dynamic=False, one compile per T, on the first --static-shapes batches.
B. Training curves. From the same init and seed, with the same batches in the same order, each variant runs --steps
   steps of BIT's training step (autocast forward, backward, AdamW, OneCycleLR as for --sched-epochs epochs,
   zero_grad). The variants:
   - plain: block_diffusion_ce;
   - fast: the packed step with the dynamic kernel, as the failed run;
   - fast_static: the packed step with the static kernel.
   Logged per step: loss, supervised tokens (identical across variants for the same batch), packed T, gradient norm
   per group. Every --probe-every steps: the loss on 2 fixed val batches, using the dense packed step in eval mode
   (the same trusted function for every variant). For the fast variants it also checks kernel drift: on the current
   batch with dropout off, the compiled kernel against dense.
Verdicts:
- A: per batch and kernel;
- B: identical supervised counts; per-token loss over the last 100 steps, paired by step, fast - plain (FAIL if
  z > 3 and the mean > 0.02 nats); the final probe loss, fast - plain (FAIL if > 0.05 nats); kernel drift (FAIL like
  A's tolerances).
Reading: A dynamic FAIL with A static PASS means the dynamic-shape kernel is the bug (fix: static compile, which
fast_block_diffusion.PACKED_FLEX_DYNAMIC = False does). All of A passing while B fails points elsewhere; B's drift
checks and per-group gradient norms then narrow it down. Nothing is saved. Its results/stage2/<name> is removed, and
it writes results/gpu_check/<name>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import statistics as st
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("encoder", "projector", "lora", "contrastive", "other")
TOL = {"loss_rel": 0.01, "logit_cos": 0.999, "grad_cos": 0.98, "grad_ratio": 0.10, "z": 3.0, "loss_gap": 0.02,
       "probe_gap": 0.05}


class TestDone(Exception):
    def __init__(self, results):
        super().__init__("test done")
        self.results = results


def group_of(name: str) -> str:
    if name.startswith("neural_encoder."):
        return "encoder"
    if name.startswith("projector."):
        return "projector"
    if "lora_" in name:
        return "lora"
    if name.startswith(("modality_aligner.", "contrastive_loss_fn.")):
        return "contrastive"
    return "other"


def _flex_dynamic(q, k, v, mask=None):  # two functions: two separate compile caches
    from torch.nn.attention.flex_attention import flex_attention
    from bit_experiments.fast_block_diffusion import PACKED_FLEX_KERNEL_OPTIONS

    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True, kernel_options=PACKED_FLEX_KERNEL_OPTIONS)


def _flex_static(q, k, v, mask=None):
    from torch.nn.attention.flex_attention import flex_attention
    from bit_experiments.fast_block_diffusion import PACKED_FLEX_KERNEL_OPTIONS

    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True, kernel_options=PACKED_FLEX_KERNEL_OPTIONS)


def run_test(self, args):
    """Replaces BIT's Trainer.train (module docstring)."""
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader
    from utils.datasets import GroupBatchSampler, pad_collate_fn

    from bit_experiments import fast_block_diffusion as fast

    cfg = self.config
    model = self.accelerator.unwrap_model(self.model)
    device = next(model.parameters()).device
    decoder, lm_head, embed = model._parts()
    bs = model.block_size
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    limit = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    setattr(torch._dynamo.config, limit, max(getattr(torch._dynamo.config, limit), 64))
    kernels = {"dynamic": torch.compile(_flex_dynamic, fullgraph=True),
               "static": torch.compile(_flex_static, fullgraph=True, dynamic=False)}
    report = {"llm": args.llm, "A": [], "B": {}, "fails": []}
    collate = partial(pad_collate_fn, model_inputs=self.model_inputs, **cfg.method.dataloader_kwargs)

    def loader(dataset, seed=cfg.seed):
        sampler = GroupBatchSampler(dataset, batch_size=cfg.training.train_batch_size, shuffle=True,
                                    drop_last=cfg.training.drop_last_train_dataloader, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, pin_memory=True)

    def to_dev(inputs):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def verdict(name, ok, detail):
        print(f"[test] {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        if not ok:
            report["fails"].append(name)
        return ok

    @contextmanager
    def kernel(kind):
        before = fast.packed_flex_attention
        if kind is not None:
            fast.packed_flex_attention = kernels[kind]
        try:
            yield
        finally:
            fast.packed_flex_attention = before

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16)

    def packed_T(labels):
        return len(fast.pack_plan(labels, bs))

    def labels_of(inputs):
        seen = {}
        model.denoising_ce = lambda clean, labels: (seen.update(labels=labels)
                                                    or (clean.sum() * 0, labels.new_ones(()), clean, labels))
        try:
            with torch.no_grad(), amp():
                model(**inputs)
        finally:
            del model.denoising_ce
        return seen["labels"]

    def packed_run(inputs, seed, attention, kind, grad):
        """The packed step through the model's forward (eval or train mode as set), one seed."""
        seen = {}

        def ce(clean, labels):
            out = fast.fast_block_diffusion_ce(decoder, lm_head, embed, clean, labels, bs, model.mask_id,
                                               model.revision["edit_rate"], model.filler_id, attention=attention)
            seen["n"] = int(out[1])
            return out

        model.denoising_ce = ce
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        try:
            with kernel(kind), torch.set_grad_enabled(grad), amp():
                out = model(**inputs)
            res = {"loss": float(out.loss), "n": seen["n"], "logits": out.preds.detach().float(),
                   "targets": out.targets.detach()}
            if grad:
                out.loss.backward()
                res["grads"] = group_grads()
        finally:
            del model.denoising_ce
            model.zero_grad(set_to_none=True)
        return res

    def group_grads():
        return {g: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten()
                              for n, p in params if group_of(n) == g] or [torch.zeros(1, device=device)])
                for g in GROUPS}

    def compare(tag, ref, got):
        out = {"loss_rel": abs(got["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-8)}
        ok = got["targets"].shape == ref["targets"].shape and torch.equal(got["targets"], ref["targets"])
        if ok:
            out["logit_cos"] = float(F.cosine_similarity(got["logits"].flatten(), ref["logits"].flatten(), 0))
            ok = out["logit_cos"] >= TOL["logit_cos"]
        ok = ok and out["loss_rel"] <= TOL["loss_rel"]
        if "grads" in got and "grads" in ref:
            out["grads"] = {}
            for g in GROUPS:
                a, b = got["grads"][g], ref["grads"][g]
                na, nb = float(a.norm()), float(b.norm())
                if na == 0 and nb == 0:
                    continue
                cos = float(F.cosine_similarity(a, b, 0)) if na and nb else 0.0
                out["grads"][g] = {"cos": round(cos, 5), "ratio": round(na / nb, 4) if nb else None}
                ok = ok and cos >= TOL["grad_cos"] and nb > 0 and abs(na / nb - 1) <= TOL["grad_ratio"]
        verdict(tag, ok, f"loss rel {out['loss_rel']:.2e}, logits cos {out.get('logit_cos', 0):.6f}, grads "
                         + ", ".join(f"{g} {v['cos']:.4f}/{v['ratio']}" for g, v in out.get("grads", {}).items()))
        out["ok"] = ok
        return out

    # ------------------------------------------------------------ A. kernel across packed lengths
    model.eval()
    picked, seen_T = [], set()
    for k, (inputs, _) in enumerate(loader(self.train_dataset)):
        inputs = to_dev(inputs)
        t = packed_T(labels_of(inputs))
        if t not in seen_T:
            seen_T.add(t)
            picked.append((t, inputs))
        if len(picked) == args.shapes or k >= 400:
            break
    print(f"[test] A: packed lengths {[t for t, _ in picked]}", flush=True)
    for n, (t, inputs) in enumerate(picked):
        seed = cfg.seed + n
        ref = packed_run(inputs, seed, "dense", None, grad=True)
        rec = {"T": t, "dynamic": compare(f"A T={t} dynamic (grad)", ref, packed_run(inputs, seed, "flex", "dynamic",
                                                                                   grad=True))}
        rec["dynamic_nograd"] = compare(f"A T={t} dynamic (no grad)", ref,
                                        packed_run(inputs, seed, "flex", "dynamic", grad=False))
        if n < args.static_shapes:
            rec["static"] = compare(f"A T={t} static (grad)", ref, packed_run(inputs, seed, "flex", "static",
                                                                            grad=True))
        report["A"].append(rec)
        del ref
        torch.cuda.empty_cache()
    report["A_dynamo"] = dict(torch._dynamo.utils.counters["stats"])

    # ------------------------------------------------------------ B. training curves
    init = {n: p.detach().clone() for n, p in params}
    probes = [to_dev(b) for b, _ in zip((b for b, _ in loader(self.val_dataset)), range(2))]
    steps_per_epoch = len(loader(self.train_dataset))
    curves = {}
    for variant in args.variants.split(","):
        with torch.no_grad():
            for n, p in params:
                p.copy_(init[n])
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.wd,
                                eps=cfg.optimizer.eps)
        sched = OneCycleLR(opt, total_steps=args.sched_epochs * steps_per_epoch, max_lr=cfg.optimizer.lr,
                           pct_start=cfg.optimizer.warmup_pct, div_factor=cfg.optimizer.div_factor)
        model.sampler["fast_step"] = variant != "plain"
        kind = {"plain": None, "fast": "dynamic", "fast_static": "static"}[variant]
        log = {"loss": [], "n": [], "T": [], "grad_norm": {g: [] for g in GROUPS}, "probe": [], "drift": []}
        model.eval()  # probe before training: identical for every variant if the reset to init worked
        probe = [packed_run(b, 777 + j, "dense", None, grad=False) for j, b in enumerate(probes)]
        log["probe"].append((-1, sum(p["loss"] for p in probe) / sum(p["n"] for p in probe)))
        t0 = time.time()
        real_ce = model.denoising_ce
        for step, (inputs, _) in enumerate(loader(self.train_dataset)):
            if step == args.steps:
                break
            inputs = to_dev(inputs)
            model.train()
            ts = {}

            def ce(clean, labels):
                ts["T"] = packed_T(labels)
                return real_ce(clean, labels)

            model.denoising_ce = ce
            try:
                with kernel(kind), amp():
                    out = model(**inputs)
                    loss = out.loss
                loss.backward()
            finally:
                del model.denoising_ce
            for g, v in group_grads().items():
                log["grad_norm"][g].append(float(v.norm()))
            opt.step()
            sched.step()
            opt.zero_grad()
            log["loss"].append(float(loss))
            log["n"].append(int(out.n_examples))
            log["T"].append(ts["T"])
            del out, loss
            if step % args.probe_every == 0 or step == args.steps - 1:
                model.eval()
                probe = [packed_run(b, 777 + j, "dense", None, grad=False) for j, b in enumerate(probes)]
                log["probe"].append((step, sum(p["loss"] for p in probe) / sum(p["n"] for p in probe)))
                if kind is not None:
                    ref = packed_run(inputs, 999 + step, "dense", None, grad=True)
                    got = packed_run(inputs, 999 + step, "flex", kind, grad=True)
                    d = compare(f"B {variant} drift step {step} (T={ts['T']})", ref, got)
                    log["drift"].append((step, ts["T"], d["ok"]))
                    del ref, got
                print(f"[test] B {variant} step {step}: loss/token {log['loss'][-1] / log['n'][-1]:.4f}, probe "
                      f"{log['probe'][-1][1]:.4f}, T {ts['T']}, {time.time() - t0:.0f} s", flush=True)
                torch.cuda.empty_cache()
        log["seconds"] = round(time.time() - t0, 1)
        curves[variant] = log
    model.eval()

    # ------------------------------------------------------------ B verdicts
    names = list(curves)
    if len(names) > 1:
        start = [curves[v]["probe"][0][1] for v in names]
        verdict("B identical start (probe before training)", max(start) - min(start) < 1e-3,
                ", ".join(f"{v} {x:.5f}" for v, x in zip(names, start)))
        verdict("B supervised counts identical across variants",
                all(curves[v]["n"] == curves[names[0]]["n"] for v in names), f"{names} over {args.steps} steps")
    if "plain" in curves:
        p = curves["plain"]
        for v in names:
            if v == "plain":
                continue
            c = curves[v]
            w = min(100, len(c["loss"]))
            d = [c["loss"][s] / c["n"][s] - p["loss"][s] / p["n"][s] for s in range(len(c["loss"]) - w, len(c["loss"]))]
            mean, sd = st.mean(d), (st.stdev(d) if len(d) > 1 else 0.0)
            z = mean / (sd / len(d) ** 0.5) if sd > 0 else 0.0
            verdict(f"B {v} - plain, per-token train loss, last {w} steps",
                    not (z > TOL["z"] and mean > TOL["loss_gap"]), f"mean {mean:+.4f}, sd {sd:.4f}, z {z:+.2f}")
            gap = c["probe"][-1][1] - p["probe"][-1][1]
            verdict(f"B {v} - plain, final probe loss", gap <= TOL["probe_gap"],
                    f"{c['probe'][-1][1]:.4f} vs {p['probe'][-1][1]:.4f} ({gap:+.4f}); plain went "
                    f"{p['probe'][0][1]:.4f} -> {p['probe'][-1][1]:.4f}")
    print("\n[test] probe loss (dense packed step, eval) by step:", flush=True)
    steps_logged = [s for s, _ in curves[names[0]]["probe"]]
    print(f"{'step':>6} " + " ".join(f"{v:>12}" for v in names), flush=True)
    for k, s in enumerate(steps_logged):
        print(f"{s:>6} " + " ".join(f"{curves[v]['probe'][k][1]:12.4f}" for v in names), flush=True)
    for v in names:
        gn = {g: round(st.mean(x), 3) for g, x in curves[v]["grad_norm"].items() if x and max(x) > 0}
        print(f"[test] {v}: mean grad norm per group {gn}, distinct T {len(set(curves[v]['T']))}, "
              f"{curves[v]['seconds']} s", flush=True)
    report["B"] = curves
    report["dynamo"] = dict(torch._dynamo.utils.counters["stats"])
    report["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None
    raise TestDone(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="aero-1-audio-diffusion")
    ap.add_argument("--init", default="checkpoints/ssl_human/BEST_core")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shapes", type=int, default=6, help="A: distinct packed lengths")
    ap.add_argument("--static-shapes", type=int, default=3, help="A: of which also with the static kernel")
    ap.add_argument("--steps", type=int, default=250, help="B: training steps per variant")
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--variants", default="plain,fast,fast_static")
    ap.add_argument("--sched-epochs", type=int, default=150, help="B: OneCycleLR length, as the real run")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("needs a GPU")

    spec = importlib.util.spec_from_file_location("train_stage2", ROOT / "scripts" / "train_stage2.py")
    train_stage2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_stage2)  # puts src/ and BIT on sys.path
    from models.trainer import Trainer

    name = f"gpu_fast_train_check_{time.strftime('%Y%m%d_%H%M%S')}"
    Trainer.train = lambda self: run_test(self, args)
    sys.argv = ["train_stage2.py", "--name", name, "--arm", "B", "--init", args.init, "--seed", str(args.seed),
                "--llm", args.llm, "--epochs", "1", "--eval-every", "1"]
    try:
        train_stage2.main()
        raise SystemExit("train_stage2.main returned without running the test")
    except TestDone as done:
        report = done.results
    finally:
        for d in (ROOT / "results" / "stage2" / name, ROOT / "checkpoints" / "stage2" / name):
            if d.name.startswith("gpu_fast_train_check_"):
                shutil.rmtree(d, ignore_errors=True)
    out = ROOT / "results" / "gpu_check" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[test] {'ALL PASS' if not report['fails'] else 'FAILED: ' + ', '.join(report['fails'])} -> {out}")
    return 0 if not report["fails"] else 1


if __name__ == "__main__":
    sys.exit(main())
