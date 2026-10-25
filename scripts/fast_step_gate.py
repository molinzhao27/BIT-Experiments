#!/usr/bin/env python
"""Launch-time GPU gate for the packed diffusion step (fast_step), 2026-10-24. Decides whether a run may use the fixed
fast step instead of the plain one. About 20 minutes on the RTX PRO 6000, peak ~80 GB (the plain step's).

    <bit env>/python scripts/fast_step_gate.py --init checkpoints/stage1/C_l14_b1024_s42/best --arm C \
        [--llm aero-1-audio-diffusion] [--seed 42] [--shapes 5] [--steps 160] [--probe-every 20]

Exit code: 0 = the fixed fast step is validated (use it); 3 = not validated (use plain); anything else = error (use
plain). Writes results/fast_gate/<name>.json, with fixed_module_md5 = the md5 of the fixed module it validated (train
with that exact file), and a last line "[gate] VALIDATED" or "[gate] NOT VALIDATED: <why>".

Background: on 10-23, the fast step (host module md5 79bbfef978d42d265b30bacf6154eafc, kept byte-identical in
scripts/fast_gate/) trained a broken Aero model: val WER 108.6% at epoch 5 against plain's 42.0%. An eval-mode check
on 4 batches had passed. Those batches covered only 2 packed lengths. Leading cause: that module compiles its flex
attention with torch's default automatic dynamic shapes. The mask's captured tables have the packed length T, so the
second T recompiles a dynamic-shape graph, and every later T runs on it. The fixed module (src/bit_experiments/
fast_block_diffusion.py) compiles statically, one graph per T (PACKED_FLEX_DYNAMIC = False). Nothing else differs
in the step.

The gate builds the run as train_stage2.py does (its --llm, --init, --arm, --seed; data, model and BIT trainer). It
runs these phases in a fresh process, so the compiles happen in the same order as in a training run:
0. Sanity: on one batch, in eval mode, the packed step with dense SDPA (no flex, no compile) equals the plain step.
   This makes it the trusted reference for the later phases. Fail -> rc 3.
1. Kernels: real training batches, picked until --shapes distinct packed lengths are covered, in eval mode (no
   dropout), grads on, one seed per batch. On each, the old module's step and the fixed module's step against the
   dense one: loss, logits, gradients per group (encoder, projector, LoRA, contrastive head).
   - The old module is compiled static at the first T, dynamic from the second. If the cause is right, it fails at
     untraced lengths.
   - The fixed module compiles each T statically and must pass at every T.
2. Train mode: from the same init and seed, with the same batches in the same order, --steps steps of BIT's training
   step for each of plain, old and fixed: autocast, backward, AdamW and OneCycleLR as for 150 epochs, zero_grad;
   LoRA dropout and encoder noise on. Logged: per-step loss and supervised-token counts, which must be identical
   across variants. Every --probe-every steps, the loss on 2 fixed val batches with the dense step in eval mode.
Verdict:
- reproduced = the old module fails phase 1, or in phase 2 the old step is worse than plain. Worse means the paired
  per-token train loss over the last 100 steps has z > 3 and mean > 0.02 nats, or the mean of the last 3 probes is
  > 0.05 above plain's;
- fixed_ok = the fixed module passes every phase 1 length, and in phase 2 it is not worse than plain by the same
  rule;
- rc 0 iff the sanity check passed, reproduced, and fixed_ok. Otherwise rc 3: if the old step's bug does not
  show, the gate cannot vouch for the fix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import statistics as st
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = ROOT / "scripts" / "fast_gate" / "fast_block_diffusion_79bbfef.py"
OLD_MD5 = "79bbfef978d42d265b30bacf6154eafc"
GROUPS = ("encoder", "projector", "lora", "contrastive", "other")
TOL = {"loss_rel": 0.01, "logit_cos": 0.999, "grad_cos": 0.98, "grad_ratio": 0.10,  # phase 0 and 1 (bf16)
       "z": 3.0, "loss_gap": 0.02, "probe_gap": 0.05, "window": 100, "probe_last": 3}  # phase 2
RC_VALIDATED, RC_NOT_VALIDATED, RC_ERROR = 0, 3, 2
FAST_ATTENTION = None  # the old and fixed steps' attention: None = the training path (flex on CUDA); tests set "flex"


class GateDone(Exception):
    def __init__(self, report):
        super().__init__("gate done")
        self.report = report


def load_old_module():
    """The 10-23 host module, byte-identical (md5-checked), as its own module with its own compile cache."""
    if hashlib.md5(OLD_PATH.read_bytes()).hexdigest() != OLD_MD5:
        raise RuntimeError(f"{OLD_PATH} is not the 10-23 host module (md5 {OLD_MD5})")
    name = "fast_block_diffusion_79bbfef"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, OLD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclass (PackPlan) resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


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


def worse(variant: list[float], plain: list[float], probes_variant: list[float], probes_plain: list[float]) -> dict:
    """Phase 2's rule: is `variant` worse than plain? Paired per-token train losses (same steps, same batches) over
    the last TOL['window'] steps, and the probe loss averaged over the last TOL['probe_last'] probes (less noise than
    one point)."""
    w = min(TOL["window"], len(variant))
    d = [a - b for a, b in zip(variant[-w:], plain[-w:])]
    mean = st.mean(d) if d else 0.0
    sd = st.stdev(d) if len(d) > 1 else 0.0
    z = mean / (sd / len(d) ** 0.5) if sd > 0 else (float("inf") if mean > 0 else 0.0)
    k = TOL["probe_last"]
    gap = st.mean(probes_variant[-k:]) - st.mean(probes_plain[-k:])
    return {"mean": mean, "sd": sd, "z": z, "probe_gap": gap,
            "worse": bool((z > TOL["z"] and mean > TOL["loss_gap"]) or gap > TOL["probe_gap"])}


def decide(report: dict) -> tuple[int, list[str]]:
    """The rc contract from the phases' results (module docstring)."""
    why = []
    if not report.get("sanity_ok"):
        why.append("dense packed step != plain (no trusted reference)")
    kernel = report.get("kernels", [])
    old_kernel_fails = any(not k["old"]["ok"] for k in kernel)
    fixed_kernel_ok = bool(kernel) and all(k["fixed"]["ok"] for k in kernel)
    train = report.get("train_verdicts", {})
    old_train_worse = train.get("old", {}).get("worse", False)
    fixed_train_ok = "fixed" in train and not train["fixed"]["worse"] and report.get("counts_identical", False)
    reproduced = old_kernel_fails or old_train_worse
    if not reproduced:
        why.append("the old step's bug did not show (old kernel passed every length and old training matched plain)")
    if not fixed_kernel_ok:
        why.append("the fixed kernel failed at some packed length")
    if not fixed_train_ok:
        why.append("the fixed step trained worse than plain (or supervised counts differed)")
    report["verdict"] = {"reproduced": reproduced, "old_kernel_fails": old_kernel_fails,
                         "old_train_worse": old_train_worse, "fixed_kernel_ok": fixed_kernel_ok,
                         "fixed_train_ok": fixed_train_ok}
    return (RC_VALIDATED if not why else RC_NOT_VALIDATED), why


def run_gate(self, args, old, new):
    """Replaces BIT's Trainer.train: the phases (module docstring). Raises GateDone(report)."""
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader
    from utils.datasets import GroupBatchSampler, pad_collate_fn

    cfg = self.config
    model = self.accelerator.unwrap_model(self.model)
    device = next(model.parameters()).device
    decoder, lm_head, embed = model._parts()
    bs = model.block_size
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    report = {"llm": args.llm, "init": args.init, "arm": args.arm, "old_module_md5": OLD_MD5,
              "fixed_module_md5": hashlib.md5(Path(new.__file__).read_bytes()).hexdigest(),
              "fixed_dynamic_setting": getattr(new, "PACKED_FLEX_DYNAMIC", "missing"), "kernels": []}
    if getattr(new, "PACKED_FLEX_DYNAMIC", None) is not False:
        raise RuntimeError("the fixed module does not compile statically (PACKED_FLEX_DYNAMIC must be False)")
    collate = partial(pad_collate_fn, model_inputs=self.model_inputs, **cfg.method.dataloader_kwargs)
    t_start = time.time()

    def log(msg):
        print(f"[gate] {msg} ({time.time() - t_start:.0f} s)", flush=True)

    def loader(dataset):
        sampler = GroupBatchSampler(dataset, batch_size=cfg.training.train_batch_size, shuffle=True,
                                    drop_last=cfg.training.drop_last_train_dataloader, seed=cfg.seed)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, pin_memory=device.type == "cuda")

    def to_dev(inputs):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16)

    def empty_cache():
        if device.type == "cuda":
            torch.cuda.empty_cache()

    a = lambda clean, labels: (decoder, lm_head, embed, clean, labels, bs, model.mask_id,  # noqa: E731
                               model.revision["edit_rate"], model.filler_id)
    steps = {"plain": None,  # the model's own denoising_ce (the entry has no fast_step)
             "dense": lambda clean, labels: new.fast_block_diffusion_ce(*a(clean, labels), attention="dense"),
             "old": lambda clean, labels: old.fast_block_diffusion_ce(*a(clean, labels), attention=FAST_ATTENTION),
             "fixed": lambda clean, labels: new.fast_block_diffusion_ce(*a(clean, labels), attention=FAST_ATTENTION)}

    def forward(inputs, variant, record=None):
        if steps[variant] is not None:
            def ce(clean, labels):
                if record is not None:
                    record["T"] = len(new.pack_plan(labels, bs))
                return steps[variant](clean, labels)
            model.denoising_ce = ce
        elif record is not None:
            real = model.denoising_ce

            def ce(clean, labels):
                record["T"] = len(new.pack_plan(labels, bs))
                return real(clean, labels)
            model.denoising_ce = ce
        try:
            with amp():
                return model(**inputs)
        finally:
            if "denoising_ce" in model.__dict__:
                del model.denoising_ce

    def group_grads():
        return {g: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten()
                              for n, p in params if group_of(n) == g] or [torch.zeros(1, device=device)])
                for g in GROUPS}

    def evaluate(inputs, variant, seed, grad=True):
        """One step in eval mode under a seed: loss, logits, targets, and the gradient per group."""
        model.eval()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        with torch.set_grad_enabled(grad):
            out = forward(inputs, variant)
        res = {"loss": float(out.loss), "logits": out.preds.detach().float(), "targets": out.targets.detach()}
        if grad:
            out.loss.backward()
            res["grads"] = group_grads()
        model.zero_grad(set_to_none=True)
        return res

    def compare(tag, ref, got):
        out = {"loss_rel": abs(got["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-8)}
        ok = got["targets"].shape == ref["targets"].shape and torch.equal(got["targets"], ref["targets"])
        if ok:
            out["logit_cos"] = float(F.cosine_similarity(got["logits"].flatten(), ref["logits"].flatten(), 0))
            ok = out["logit_cos"] >= TOL["logit_cos"]
        ok = ok and out["loss_rel"] <= TOL["loss_rel"]
        out["grads"] = {}
        for g in GROUPS:
            x, y = got["grads"][g], ref["grads"][g]
            nx, ny = float(x.norm()), float(y.norm())
            if nx == 0 and ny == 0:
                continue
            cos = float(F.cosine_similarity(x, y, 0)) if nx and ny else 0.0
            ratio = nx / ny if ny else float("inf")
            out["grads"][g] = {"cos": round(cos, 5), "ratio": round(ratio, 4)}
            ok = ok and cos >= TOL["grad_cos"] and abs(ratio - 1) <= TOL["grad_ratio"]
        out["ok"] = bool(ok)
        log(f"{'PASS' if ok else 'FAIL'}  {tag}: loss rel {out['loss_rel']:.2e}, logits cos {out.get('logit_cos', 0):.6f}, "
            + ", ".join(f"{g} cos {v['cos']:.4f} ratio {v['ratio']}" for g, v in out["grads"].items()))
        return out

    def labels_T(inputs):
        rec = {}
        model.eval()
        model.denoising_ce = lambda clean, labels: (rec.update(T=len(new.pack_plan(labels, bs)))
                                                    or (clean.sum() * 0, labels.new_ones(()), clean, labels))
        try:
            with torch.no_grad(), amp():
                model(**inputs)
        finally:
            del model.denoising_ce
        return rec["T"]

    # ---------------------------------------------------------------- phase 0 and 1: sanity, kernels
    picked, seen = [], set()
    for k, (inputs, _) in enumerate(loader(self.train_dataset)):
        inputs = to_dev(inputs)
        t = labels_T(inputs)
        if t not in seen:
            seen.add(t)
            picked.append((t, inputs))
        if len(picked) == args.shapes or k >= 400:
            break
    log(f"packed lengths for phase 1, in order: {[t for t, _ in picked]}")
    ref = evaluate(picked[0][1], "dense", args.seed)
    report["sanity"] = compare(f"phase 0: dense packed step vs plain (T={picked[0][0]})", evaluate(picked[0][1], "plain", args.seed), ref)
    report["sanity_ok"] = report["sanity"]["ok"]
    for n, (t, inputs) in enumerate(picked):
        seed = args.seed + n
        ref = evaluate(inputs, "dense", seed) if n else ref
        rec = {"T": t, "old": compare(f"phase 1: old step, T={t}", ref, evaluate(inputs, "old", seed)),
               "fixed": compare(f"phase 1: fixed step, T={t}", ref, evaluate(inputs, "fixed", seed))}
        report["kernels"].append(rec)
        empty_cache()
    del ref

    # ---------------------------------------------------------------- phase 2: train mode
    init = {n: p.detach().clone() for n, p in params}
    probes = [to_dev(b) for b, _ in zip((b for b, _ in loader(self.val_dataset)), range(2))]
    steps_per_epoch = len(loader(self.train_dataset))

    def probe():
        with torch.no_grad():
            outs = [evaluate(b, "dense", 777 + j, grad=False) for j, b in enumerate(probes)]
        return sum(o["loss"] for o in outs) / sum(int(o["targets"].numel()) for o in outs)

    curves = {}
    for variant in ("plain", "old", "fixed"):
        with torch.no_grad():
            for n, p in params:
                p.copy_(init[n])
        torch.manual_seed(cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.wd,
                                eps=cfg.optimizer.eps)
        sched = OneCycleLR(opt, total_steps=args.sched_epochs * steps_per_epoch, max_lr=cfg.optimizer.lr,
                           pct_start=cfg.optimizer.warmup_pct, div_factor=cfg.optimizer.div_factor)
        c = {"loss": [], "n": [], "T": [], "probe": [(-1, probe())]}
        t0 = time.time()
        for step, (inputs, _) in enumerate(loader(self.train_dataset)):
            if step == args.steps:
                break
            model.train()
            rec = {}
            out = forward(to_dev(inputs), variant, rec)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            c["loss"].append(float(out.loss) / int(out.n_examples))
            c["n"].append(int(out.n_examples))
            c["T"].append(rec["T"])
            del out
            if (step + 1) % args.probe_every == 0 or step == args.steps - 1:
                c["probe"].append((step, probe()))
                log(f"phase 2 {variant} step {step}: train loss/token {c['loss'][-1]:.4f}, probe {c['probe'][-1][1]:.4f}")
                empty_cache()
        c["seconds"] = round(time.time() - t0, 1)
        curves[variant] = c
    report["train"] = curves
    report["counts_identical"] = curves["old"]["n"] == curves["plain"]["n"] == curves["fixed"]["n"]
    start = [curves[v]["probe"][0][1] for v in curves]
    report["identical_start"] = max(start) - min(start) < 1e-3
    report["train_verdicts"] = {v: worse(curves[v]["loss"], curves["plain"]["loss"],
                                         [x for _, x in curves[v]["probe"][1:]],
                                         [x for _, x in curves["plain"]["probe"][1:]]) for v in ("old", "fixed")}
    for v, w in report["train_verdicts"].items():
        log(f"phase 2 {v} vs plain: paired train loss/token mean {w['mean']:+.4f} (z {w['z']:+.2f}), probe gap "
            f"{w['probe_gap']:+.4f} -> {'WORSE' if w['worse'] else 'not worse'}")
    log(f"phase 2 supervised counts identical: {report['counts_identical']}; identical start: {report['identical_start']}")
    if not report["identical_start"]:
        raise RuntimeError(f"phase 2 variants did not start from the same model (probes {start})")
    report["minutes"] = round((time.time() - t_start) / 60, 1)
    report["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None
    raise GateDone(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="aero-1-audio-diffusion", help="the run's diffusion entry, without fast_step")
    ap.add_argument("--init", default="checkpoints/ssl_human/BEST_core", help="the run's --init")
    ap.add_argument("--arm", default="B", help="the run's --arm (metadata)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shapes", type=int, default=5, help="phase 1: distinct packed lengths")
    ap.add_argument("--steps", type=int, default=160, help="phase 2: training steps per variant")
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--sched-epochs", type=int, default=150)
    args = ap.parse_args()
    name = f"fast_step_gate_{time.strftime('%Y%m%d_%H%M%S')}"
    out = ROOT / "results" / "fast_gate" / f"{name}.json"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("needs a GPU")
        spec = importlib.util.spec_from_file_location("train_stage2", ROOT / "scripts" / "train_stage2.py")
        train_stage2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_stage2)  # puts src/ and BIT on sys.path
        from models.trainer import Trainer

        from bit_experiments import fast_block_diffusion as new

        old = load_old_module()
        Trainer.train = lambda self: run_gate(self, args, old, new)
        sys.argv = ["train_stage2.py", "--runtime", "reference", "--name", name, "--arm", args.arm, "--init", args.init, "--seed", str(args.seed),
                    "--llm", args.llm, "--epochs", "1", "--eval-every", "1"]
        try:
            train_stage2.main()
            raise RuntimeError("train_stage2.main returned without running the gate")
        except GateDone as done:
            report = done.report
        finally:
            for d in (ROOT / "results" / "stage2" / name, ROOT / "checkpoints" / "stage2" / name):
                if d.name.startswith("fast_step_gate_"):
                    shutil.rmtree(d, ignore_errors=True)
        rc, why = decide(report)
        report["rc"], report["why"] = rc, why
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str))
        print(f"[gate] verdict {json.dumps(report['verdict'])}; {report.get('minutes')} min, "
              f"max mem {report.get('max_mem_gb')} GB -> {out}", flush=True)
        print("[gate] VALIDATED" if rc == RC_VALIDATED else f"[gate] NOT VALIDATED: {'; '.join(why)}", flush=True)
        return rc
    except Exception:  # anything unexpected: an error, so the run uses plain
        traceback.print_exc()
        print(f"[gate] ERROR (rc {RC_ERROR}): use the plain step", flush=True)
        return RC_ERROR


if __name__ == "__main__":
    sys.exit(main())
