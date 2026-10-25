#!/usr/bin/env python
"""Is the packed diffusion step (fast_step) the plain step on the GPU, and does it see the neural prefix? A standalone
check for the Aero block-diffusion run (docs/DIFFUSION_SPEEDUP.md). Takes about 5-8 minutes. Run it on the GPU through
the lock wrapper, from the code copy:

    /workspace/envs/bit/bin/python scripts/gpu_fast_step_check.py --llm aero-1-audio-diffusion \
        [--init checkpoints/ssl_human/BEST_core] [--ckpt checkpoints/stage2/<fast run>/BEST] [--batches 4] [--trials 8]
        [--train-mode 8]

It builds the model exactly as train_stage2.py does (config, model, --init, seed, data, BIT trainer), optionally loads
a trained Stage 2 checkpoint (--ckpt), and takes the first --batches training batches. The model is in eval mode (no
LoRA dropout, no encoder noise), so each step is deterministic apart from the masks, and every variant draws the same
masks from the same seed. Grad stays on. bf16 autocast, as in training. Variants of the step, run through the model's
forward so that the encoder, projector, contrastive head and LoRA all get gradients:
- plain: block_diffusion_ce, the vendored compiled flex attention (run 3's and the control's path; the reference);
- fast:  fast_block_diffusion_ce, the packed step's compiled flex (packed_flex_attention). This is the training path;
- fast_dense: the packed step with SDPA and the dense boolean mask on CUDA (no flex at all);
- fast_eager: the packed step with eager (uncompiled) flex and the same CUDA block mask, forward only. Torch's eager
  flex computes q @ k^T in bf16 before upcasting (torch/_higher_order_ops/flex_attention.py, _math_attention_inner),
  so under bf16 it is ~2-4% off the fused kernels in loss. Its checks are information only ("info"), with loose
  tolerances (loss 5%, logits cosine 0.99).
Checks, each printed PASS / FAIL with bf16 tolerances:
1. per batch and variant, against plain: the same supervised targets; loss within 1%; logits cosine ≥ 0.999 (and
   argmax agreement ≥ 90%: near-tied logits flip under bf16, so this one is loose); per parameter group (encoder, projector, LoRA, contrastive head) gradient cosine ≥ 0.98 and
   norm ratio within 10%;
2. the CUDA block mask (create_block_mask with the plan's table-reading mask_mod) equals the CPU-built one;
3. stale tables: after all the other batches, batch 0's fast logits, recomputed in the same grad mode, have cosine
   ≥ 0.99999 to its first fast logits. Fast against plain on every later batch covers it too;
4. prefix sensitivity: the neural input is rolled by one trial across the batch, so each sentence gets another
   trial's neural tokens. Per-token CE change S = CE/n(shuffled) - CE/n(real), for plain and fast with the same masks.
   PASS if fast's S matches plain's (|ΔS| ≤ 10% of |S_plain| + 0.005 nats). S_plain itself says whether these weights
   use the neural input at all (at --init they barely can; at a trained checkpoint they should);
6. with --train-mode K: train mode (LoRA dropout and encoder noise on), K paired seeds per batch on the first two
   batches. Same seed means the same encoder noise and masks; only the dropout draws differ. Per-token CE: the paired
   difference plain - fast must have |z| < 3.5. Gradients: the cosine between the mean over half the seeds of plain
   and the mean over the other half of fast must be at least the plain-plain cosine of the same halves minus 0.02,
   and the full means' norm ratio within 15%;
5. diagnostics, not pass/fail: the mask row against the EOS row, how many embedding rows are bitwise the mask row, and
   any prompt token whose row is bitwise the mask row (it would look like a mask to the model).
Reading: all PASS, with S small at the trained fast checkpoint, means the fast step is right and the model learned to
ignore the brain input (Aero- or training-specific). fast FAIL with fast_eager and fast_dense PASS points at the
compiled kernel or tiles; fast and fast_eager FAIL with fast_dense PASS at the CUDA block mask; all three FAIL at the
packed layout on CUDA. Nothing is saved. Its results/stage2/<name> is removed at the end. Writes
results/gpu_check/<name>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NEURAL_KEYS = ("spikes", "spikes_mask", "spikes_timestamp", "spikes_lengths", "day_idx", "dataset_name")
GROUPS = ("encoder", "projector", "lora", "contrastive", "other")
TOL = {"loss_rel": 0.01, "logit_cos": 0.999, "argmax": 0.90, "grad_cos": 0.98, "grad_ratio": 0.10,
       "stale_cos": 0.99999, "sens_rel": 0.10, "sens_abs": 0.005, "eager_loss_rel": 0.05, "eager_logit_cos": 0.99,
       "train_z": 3.5, "train_cos_margin": 0.02, "train_ratio": 0.15}


class CheckDone(Exception):
    def __init__(self, results):
        super().__init__("check done")
        self.results = results


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def run_checks(self, args):
    """Replaces BIT's Trainer.train (module docstring)."""
    from bit_experiments import diffusion_llm, fast_block_diffusion as fast, stage2

    model = self.accelerator.unwrap_model(self.model)
    report = {"llm": args.llm, "init": args.init, "ckpt": None, "batches": [], "fails": []}
    if args.ckpt:
        report["ckpt"] = {"dir": args.ckpt, **stage2.load_model_state(model, Path(args.ckpt))}
    model.eval()  # no LoRA dropout, no encoder noise: deterministic apart from the masks
    decoder, lm_head, embed = model._parts()
    device = embed.weight.device
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[check] trainable tensors per group: "
          f"{ {g: sum(group_of(n) == g for n, _ in params) for g in GROUPS} }", flush=True)

    # 5. diagnostics: the mask row
    w = embed.weight.detach()
    mask_row, eos_row = w[model.mask_id], w[model.eos_id]
    same_as_mask = (w == mask_row).all(1)
    report["mask_row"] = {"mask_id": model.mask_id, "eos_id": model.eos_id,
                          "cos_mask_eos": round(float(F.cosine_similarity(mask_row, eos_row, 0)), 4),
                          "rows_bitwise_equal_to_mask": int(same_as_mask.sum()),
                          "first_equal_rows": same_as_mask.nonzero().flatten()[:5].tolist()}

    batches = []
    for model_inputs, _ in self.train_dataloader:
        n = model_inputs["input_ids"].shape[0]
        if args.trials and args.trials < n:  # the first --trials trials of each batch (less memory)
            model_inputs = {k: v[: args.trials] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v
                            for k, v in model_inputs.items()}
        batches.append(model_inputs)
        if len(batches) == args.batches:
            break
    b0 = batches[0]
    prompt_ids = b0["input_ids"][0][(b0["targets"][0] == -100) & (b0["attention_mask"][0] == 1)]
    report["mask_row"]["prompt_ids_like_mask"] = sorted(set(prompt_ids[same_as_mask[prompt_ids]].tolist()))
    print(f"[check] mask row: {json.dumps(report['mask_row'])}", flush=True)

    def make_ce(variant, seen):
        def ce(clean, labels):
            a = (decoder, lm_head, embed, clean, labels, model.block_size, model.mask_id, model.revision["edit_rate"],
                 model.filler_id)
            if variant == "plain":
                out = diffusion_llm.block_diffusion_ce(*a)
            elif variant == "fast":
                out = fast.fast_block_diffusion_ce(*a, attention="flex")
            elif variant == "fast_dense":
                out = fast.fast_block_diffusion_ce(*a, attention="dense")
            else:  # fast_eager
                with mock.patch.object(fast, "_compiled_flex", lambda device: False):
                    out = fast.fast_block_diffusion_ce(*a, attention="flex")
            seen.update(ce=float(out[0]), n=int(out[1]), labels=labels)
            return out
        return ce

    def run(inputs, variant, seed, grad):
        seen = {}
        model.denoising_ce = make_ce(variant, seen)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        try:
            with torch.set_grad_enabled(grad), torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(**inputs)
            res = {"loss": float(out.loss), "ce": seen["ce"], "n": seen["n"], "logits": out.preds.detach().float(),
                   "targets": out.targets.detach(), "labels": seen["labels"]}
            if grad:
                out.loss.backward()
                res["grads"] = {g: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten()
                                              for n, p in params if group_of(n) == g] or [torch.zeros(1, device=device)])
                                for g in GROUPS}
        finally:
            del model.denoising_ce
        model.zero_grad(set_to_none=True)
        return res

    def verdict(name, ok, detail, info=False):
        print(f"[check] {'PASS' if ok else ('info-off' if info else 'FAIL')}  {name}: {detail}", flush=True)
        if not ok and not info:
            report["fails"].append(name)
        return ok

    def compare(tag, ref, got, info=False):
        loss_tol, cos_tol = (TOL["eager_loss_rel"], TOL["eager_logit_cos"]) if info else (TOL["loss_rel"], TOL["logit_cos"])
        out = {"loss_rel": abs(got["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-8)}
        same_targets = got["targets"].shape == ref["targets"].shape and torch.equal(got["targets"], ref["targets"])
        verdict(f"{tag} targets", same_targets, f"n {got['n']} vs {ref['n']}")
        verdict(f"{tag} loss", out["loss_rel"] <= loss_tol,
                f"{got['loss']:.4f} vs {ref['loss']:.4f} (rel {out['loss_rel']:.2e})", info)
        if same_targets:
            a, b = got["logits"], ref["logits"]
            out["logit_cos"] = float(F.cosine_similarity(a.flatten(), b.flatten(), 0))
            out["argmax"] = float((a.argmax(-1) == b.argmax(-1)).float().mean())
            out["logit_maxdiff_rel"] = float((a - b).abs().max() / b.abs().max())
            verdict(f"{tag} logits", out["logit_cos"] >= cos_tol and out["argmax"] >= TOL["argmax"],
                    f"cos {out['logit_cos']:.5f}, argmax agree {out['argmax']:.3f}, "
                    f"max diff {out['logit_maxdiff_rel']:.2e} of max", info)
        if "grads" in got and "grads" in ref:
            out["grads"] = {}
            for g in GROUPS:
                ga, gb = got["grads"][g], ref["grads"][g]
                na, nb = float(ga.norm()), float(gb.norm())
                if nb == 0 and na == 0:
                    continue
                cos = float(F.cosine_similarity(ga, gb, 0)) if na > 0 and nb > 0 else 0.0
                ratio = na / nb if nb > 0 else float("inf")
                out["grads"][g] = {"cos": round(cos, 5), "norm_ratio": round(ratio, 4), "norm_ref": nb}
                verdict(f"{tag} grad {g}", cos >= TOL["grad_cos"] and abs(ratio - 1) <= TOL["grad_ratio"],
                        f"cos {cos:.5f}, norm ratio {ratio:.4f} (ref norm {nb:.3e})")
        return out

    graphs0 = dict(torch._dynamo.utils.counters["stats"])
    t0 = time.time()
    first_fast = None
    for k, inputs in enumerate(batches):
        seed = args.seed + k
        rec = {"batch": k}
        print(f"[check] --- batch {k} (seed {seed})", flush=True)
        ref = run(inputs, "plain", seed, grad=True)
        labels = ref["labels"]
        plan = fast.pack_plan(labels, model.block_size)
        rec["layout"] = {"L": labels.shape[1], "packed_T": len(plan), "packed_real": plan.n_real,
                         "prefix_lengths": (labels != -100).int().argmax(1).tolist()}
        print(f"[check] layout {rec['layout']}", flush=True)

        # 2. the CUDA block mask against the CPU one
        cuda_bm = fast.create_block_mask(plan.to(device).mask_mod(), None, None, len(plan), len(plan), device=device)
        cpu_bm = fast.create_block_mask(plan.mask_mod(), None, None, len(plan), len(plan), device="cpu")
        fields = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices")
        verdict(f"b{k} CUDA block mask", all(torch.equal(getattr(cuda_bm, f).cpu(), getattr(cpu_bm, f)) for f in fields),
                f"T {len(plan)}")

        for variant in ("fast", "fast_dense"):
            got = run(inputs, variant, seed, grad=True)
            rec[variant] = compare(f"b{k} {variant} vs plain", ref, got)
            if variant == "fast":
                fast_res = got
                if k == 0:
                    first_fast = got["logits"]
            del got
            empty_cache()
        eager = run(inputs, "fast_eager", seed, grad=False)
        rec["fast_eager"] = compare(f"b{k} fast_eager vs plain (info)", ref, eager, info=True)
        rec["fast_vs_eager"] = compare(f"b{k} fast vs fast_eager (info)", eager,
                                       {k_: v for k_, v in fast_res.items() if k_ != "grads"}, info=True)
        del eager, fast_res
        empty_cache()

        # 4. prefix sensitivity: roll the neural input by one trial
        perm = torch.roll(torch.arange(inputs["input_ids"].shape[0]), 1)
        shuffled = {key: (v[perm.to(v.device)] if key in NEURAL_KEYS and torch.is_tensor(v) and v.dim() > 0 else v)
                    for key, v in inputs.items()}
        sens = {}
        for variant in ("plain", "fast"):
            real = ref if variant == "plain" else run(inputs, "fast", seed, grad=False)
            shuf = run(shuffled, variant, seed, grad=False)
            sens[variant] = shuf["ce"] / shuf["n"] - real["ce"] / real["n"]
        diff = abs(sens["fast"] - sens["plain"])
        rec["prefix_sensitivity"] = {"S_plain": round(sens["plain"], 5), "S_fast": round(sens["fast"], 5)}
        verdict(f"b{k} prefix sensitivity", diff <= TOL["sens_rel"] * abs(sens["plain"]) + TOL["sens_abs"],
                f"S_plain {sens['plain']:+.4f}, S_fast {sens['fast']:+.4f} nats/token (shuffled - real)")
        del ref
        empty_cache()
        report["batches"].append(rec)

    # 3. stale tables: batch 0 again, after the others, in the same grad mode as its first fast run
    again = run(batches[0], "fast", args.seed, grad=True)["logits"]
    same_shape = again.shape == first_fast.shape
    stale_cos = float(F.cosine_similarity(again.flatten(), first_fast.flatten(), 0)) if same_shape else 0.0
    stale_rel = float((again - first_fast).abs().max() / first_fast.abs().max()) if same_shape else 1.0
    verdict("batch 0 fast logits after other batches", stale_cos >= TOL["stale_cos"],
            f"cos {stale_cos:.7f}, max diff {stale_rel:.2e} of max")
    report["stale"] = {"cos": stale_cos, "max_diff_rel": stale_rel}

    # 6. train mode: K paired seeds per batch, dropout and encoder noise on
    if args.train_mode:
        model.train()
        report["train_mode"] = []
        for k, inputs in enumerate(batches[:2]):
            ce = {"plain": [], "fast": []}
            halves = {v: [dict.fromkeys(GROUPS, 0.0), dict.fromkeys(GROUPS, 0.0)] for v in ce}
            for j in range(args.train_mode):
                for variant in ce:
                    r = run(inputs, variant, args.seed + 1000 * (k + 1) + j, grad=True)
                    ce[variant].append(r["ce"] / r["n"])
                    for g in GROUPS:
                        halves[variant][j % 2][g] = halves[variant][j % 2][g] + r["grads"][g]
                    del r
            diffs = [a - b for a, b in zip(ce["plain"], ce["fast"])]
            mean = sum(diffs) / len(diffs)
            sd = (sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1)) ** 0.5
            z = mean / (sd / len(diffs) ** 0.5) if sd > 0 else (0.0 if mean == 0 else float("inf"))
            rec = {"batch": k, "ce_plain": ce["plain"], "ce_fast": ce["fast"], "z": z, "grads": {}}
            verdict(f"train b{k} per-token CE (plain - fast)", abs(z) < TOL["train_z"],
                    f"mean {mean:+.5f}, sd {sd:.5f}, z {z:+.2f} over {len(diffs)} seeds")
            for g in GROUPS:
                (pa, pb), (fa, fb) = halves["plain"], halves["fast"]
                if not torch.is_tensor(pa[g]) or float((pa[g] + pb[g]).norm()) == 0:
                    continue
                cos = lambda u, v: float(F.cosine_similarity(u, v, 0))
                c_pp = cos(pa[g], pb[g])
                c_pf = (cos(pa[g], fb[g]) + cos(pb[g], fa[g])) / 2
                ratio = float((fa[g] + fb[g]).norm() / (pa[g] + pb[g]).norm())
                rec["grads"][g] = {"cos_plain_plain": round(c_pp, 4), "cos_plain_fast": round(c_pf, 4),
                                   "mean_norm_ratio": round(ratio, 4)}
                verdict(f"train b{k} grad {g}",
                        c_pf >= c_pp - TOL["train_cos_margin"] and abs(ratio - 1) <= TOL["train_ratio"],
                        f"cos plain-fast {c_pf:.4f} vs plain-plain {c_pp:.4f}, mean norm ratio {ratio:.4f}")
            report["train_mode"].append(rec)
            del halves
            empty_cache()
        model.eval()
    report["dynamo_stats"] = {k: v - graphs0.get(k, 0) for k, v in torch._dynamo.utils.counters["stats"].items()}
    report["minutes"] = round((time.time() - t0) / 60, 2)
    report["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None
    raise CheckDone(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="aero-1-audio-diffusion", help="a configs/llms.yaml diffusion entry")
    ap.add_argument("--init", default="checkpoints/ssl_human/BEST_core")
    ap.add_argument("--ckpt", help="a trained Stage 2 checkpoint dir to load after --init (e.g. the fast run's BEST)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--train-mode", type=int, default=0, metavar="K",
                    help="also compare in train mode (dropout, encoder noise) over K paired seeds per batch (first 2)")
    ap.add_argument("--trials", type=int, help="use only the first N trials of each batch (default: all 8). The plain "
                    "step needs about run 3's training peak (~80 GB) for 8; use 4 to share the GPU")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("needs a GPU")

    spec = importlib.util.spec_from_file_location("train_stage2", ROOT / "scripts" / "train_stage2.py")
    train_stage2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_stage2)  # puts src/ and BIT on sys.path
    from models.trainer import Trainer

    name = f"gpu_fast_step_check_{time.strftime('%Y%m%d_%H%M%S')}"
    Trainer.train = lambda self: run_checks(self, args)
    sys.argv = ["train_stage2.py", "--runtime", "reference", "--name", name, "--arm", "B", "--init", args.init, "--seed", str(args.seed),
                "--llm", args.llm, "--epochs", "1", "--eval-every", "1"]
    try:
        train_stage2.main()
        raise SystemExit("train_stage2.main returned without running the checks")
    except CheckDone as done:
        report = done.results
    finally:
        for d in (ROOT / "results" / "stage2" / name, ROOT / "checkpoints" / "stage2" / name):
            if d.name.startswith("gpu_fast_step_check_"):
                shutil.rmtree(d, ignore_errors=True)
    out = ROOT / "results" / "gpu_check" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[check] {'ALL PASS' if not report['fails'] else 'FAILED: ' + ', '.join(report['fails'])}")
    print(f"[check] {report['minutes']} min, max mem {report['max_mem_gb']} GB, dynamo {report['dynamo_stats']} -> {out}")
    return 0 if not report["fails"] else 1


if __name__ == "__main__":
    sys.exit(main())
