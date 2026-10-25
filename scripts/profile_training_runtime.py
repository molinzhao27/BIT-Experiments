#!/usr/bin/env python
"""Bounded CUDA comparison of Stage 2 reference and optimized runtime overhead.

Run through the project's GPU lock, with its usual BIT environment. This loads
the real Stage 2 model/init/data but never trains or overwrites a named run.
Each alternating window restores parameters, buffers, optimizer, scheduler and
all RNGs, then replays identical real batches. Warmup covers every replay batch
before timing. CPU collation is measured separately; CUDA timing excludes fetch
and host-to-device transfer and is NOT a full-job speedup estimate.

Exit 0: bounded numerical/RNG checks pass; 3: comparison fails. This is a runtime
regression gate, not evidence of long-run learning equivalence for a new engine.
Packed fast_step variants are refused. No checkpoints are saved.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external" / "bit")]


class ProfileDone(Exception):
    def __init__(self, report):
        self.report = report


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(v, fn) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return {"torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def capture_state(trainer):
    cpu = lambda t: t.detach().cpu().clone()
    model = trainer.accelerator.unwrap_model(trainer.model)
    scaler = getattr(trainer.accelerator, "scaler", None)
    return {"parameters": {n: cpu(p) for n, p in model.named_parameters() if p.requires_grad},
            "gradients": {n: None if p.grad is None else cpu(p.grad)
                          for n, p in model.named_parameters() if p.requires_grad},
            "buffers": {n: cpu(b) for n, b in model.named_buffers()},
            "optimizer": map_tensors(trainer.optimizer.state_dict(), cpu),
            "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
            "scaler": None if scaler is None else copy.deepcopy(scaler.state_dict()),
            "training": {n: m.training for n, m in model.named_modules()}, "rng": rng_state()}


def restore_state(trainer, state):
    model = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state["parameters"]:
                parameter.copy_(state["parameters"][name])
                grad = state["gradients"][name]
                parameter.grad = None if grad is None else grad.to(parameter.device).clone()
        for name, buffer in model.named_buffers():
            buffer.copy_(state["buffers"][name])
    trainer.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    trainer.lr_scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    if state["scaler"] is not None:
        trainer.accelerator.scaler.load_state_dict(copy.deepcopy(state["scaler"]))
    for name, module in model.named_modules():
        module.training = state["training"][name]
    restore_rng(state["rng"])


def compare_values(reference, candidate, rtol=1e-4, atol=1e-7):
    """Compare nested scientific state, reporting exactness and finite closeness."""
    result = {"exact": True, "close": True, "max_abs": 0.0, "failures": []}

    def fail(path):
        result["close"] = False
        if len(result["failures"]) < 12:
            result["failures"].append(path or "root")

    def visit(a, b, path):
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
                result["exact"] = False
                return fail(path)
            if a.dtype.kind not in "biufc":
                equal = bool(np.array_equal(a, b))
                result["exact"] &= equal
                if not equal:
                    fail(path)
                return
            a, b = torch.from_numpy(a), torch.from_numpy(b)
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                result["exact"] = False
                return fail(path)
            a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
            exact = torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
            result["exact"] &= exact
            if a.is_floating_point() or a.is_complex():
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                if finite and a.numel():
                    dtype = torch.complex128 if a.is_complex() else torch.float64
                    result["max_abs"] = max(result["max_abs"], float((a.to(dtype) - b.to(dtype)).abs().max()))
                if not finite or not torch.allclose(a, b, rtol=rtol, atol=atol):
                    fail(path)
            elif not exact:
                fail(path)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                result["exact"] = False
                return fail(path)
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                result["exact"] = False
                return fail(path)
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{i}]")
        elif isinstance(a, (float, np.floating)):
            if not isinstance(b, (float, np.floating)):
                result["exact"] = False
                return fail(path)
            result["exact"] &= float(a).hex() == float(b).hex()
            if math.isfinite(a) and math.isfinite(b):
                result["max_abs"] = max(result["max_abs"], abs(float(a) - float(b)))
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                fail(path)
        elif type(a) is not type(b) or a != b:
            result["exact"] = False
            fail(path)

    visit(reference, candidate, "")
    return result


def compare_runs(reference, candidate, initial, rtol, atol):
    updates = lambda run: {n: p - initial["parameters"][n] for n, p in run["state"]["parameters"].items()}
    pairs = {"losses": (reference["losses"], candidate["losses"]),
             "parameter_updates": (updates(reference), updates(candidate))}
    for key in ("optimizer", "scheduler", "buffers", "scaler"):
        pairs[key] = (reference["state"][key], candidate["state"][key])
    if "gradients" in reference:
        pairs["gradients"] = (reference["gradients"], candidate["gradients"])
    checks = {k: compare_values(a, b, rtol, atol) for k, (a, b) in pairs.items()}
    for key, values in {"counts": (reference["counts"], candidate["counts"]),
                        "rng": (reference["state"]["rng"], candidate["state"]["rng"])}.items():
        checks[key] = compare_values(*values, rtol=0, atol=0)
    passed = all(v["close"] for v in checks.values()) and checks["rng"]["exact"] and checks["counts"]["exact"]
    return {"passed": passed, "exact": all(v["exact"] for v in checks.values()), "checks": checks}


def profile_train(trainer, args):
    from bit_experiments.performance import configure_collators, configure_runtime
    from bit_experiments.training_runtime import DeferredScalars

    if trainer.accelerator.num_processes != 1:
        raise ValueError("this benchmark supports one process/GPU only")
    if trainer.config.optimizer.gradient_accumulation_steps != 1:
        raise ValueError("this benchmark requires gradient_accumulation_steps=1")
    model = trainer.accelerator.unwrap_model(trainer.model)
    if getattr(model, "sampler", {}).get("fast_step", False):
        raise ValueError("packed fast_step variants are excluded; use a plain decoder entry")
    if trainer.accelerator.device.type != "cuda":
        raise ValueError("this benchmark requires CUDA")
    trainer.model.train()
    initial = capture_state(trainer)

    def set_mode(mode):
        configure_runtime(model, mode)
        configure_collators(trainer, mode)
        if args.selective_ar_head:
            from bit_experiments.selective_ar import install_selective_ar_head, uninstall_selective_ar_head

            (install_selective_ar_head if mode == "optimized" else uninstall_selective_ar_head)(model)

    # Use the real sampler and dataset, retaining fixed indices for both modes.
    indices = []
    while len(indices) < args.steps:
        new = list(trainer.train_dataloader.batch_sampler)
        if not new:
            raise ValueError("empty training sampler")
        indices.extend(new[:args.steps - len(indices)])
    inputs, fetch, collator_checks = {}, {"reference": [], "optimized": []}, []
    for mode in ("reference", "optimized"):
        configure_collators(trainer, mode)
        batches = []
        for ids in indices:
            started = time.perf_counter()
            batch = trainer.train_dataloader.collate_fn([trainer.train_dataset[i] for i in ids])
            fetch[mode].append(time.perf_counter() - started)
            batches.append(batch)
        inputs[mode] = batches
    for a, b in zip(inputs["reference"], inputs["optimized"]):
        collator_checks.append(compare_values(a, b, rtol=0, atol=0))
    collator_exact = all(check["exact"] for check in collator_checks)
    # Exclude device transfer from timing, and replay exactly the reference input.
    batches = [map_tensors(x[0], lambda t: t.to(trainer.accelerator.device)) for x in inputs["reference"]]
    del inputs

    def run(mode, sequence, gradients=False):
        # A fresh real loader batch cannot retain a previous call's in-place
        # changes. Preserve that property in replay; cloning is outside timing.
        sequence = [map_tensors(batch, lambda t: t.clone()) for batch in sequence]
        losses = [] if mode == "reference" else DeferredScalars(64)
        counts = [] if mode == "reference" else DeferredScalars(64)
        saved_gradients = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        for i, batch in enumerate(sequence):
            with torch.autocast(device_type=trainer.device_type, dtype=trainer.dtype):
                outputs = trainer.model(**batch)
                loss = outputs.loss / trainer.config.optimizer.gradient_accumulation_steps
            trainer.accelerator.backward(loss)
            if gradients and i == len(sequence) - 1:
                saved_gradients = {n: None if p.grad is None else p.grad.detach().cpu().clone()
                                   for n, p in model.named_parameters() if p.requires_grad}
            trainer.optimizer.step()
            if trainer.config.optimizer.scheduler in ("linear", "cosine"):
                trainer.lr_scheduler.step()
            trainer.optimizer.zero_grad()
            reduced_loss = trainer.accelerator.gather(loss).sum().detach()
            losses.append(reduced_loss.item() if mode == "reference" else reduced_loss)
            reduced_n = trainer.accelerator.gather(outputs.n_examples).sum().detach()
            counts.append(reduced_n.item() if mode == "reference" else reduced_n)
        loss_values = losses if mode == "reference" else losses.values()
        count_values = counts if mode == "reference" else counts.values()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {"seconds": elapsed, "losses": loss_values, "counts": count_values}
        if saved_gradients is not None:
            result["gradients"] = saved_gradients
        return result

    windows, comparisons, warmup_s = [], [], 0.0
    for round_idx in range(args.rounds):
        modes = ("reference", "optimized") if round_idx % 2 == 0 else ("optimized", "reference")
        pair = {}
        for mode in modes:
            set_mode(mode)
            restore_state(trainer, initial)
            # Every shape is warmed, even when --warmup is smaller than --steps.
            warm = batches + [batches[i % len(batches)] for i in range(args.warmup)]
            warmup_s += run(mode, warm)["seconds"]
            restore_state(trainer, initial)
            measured = run(mode, batches)
            measured["state"] = capture_state(trainer)
            pair[mode] = measured
            windows.append({"round": round_idx + 1, "mode": mode, "seconds": measured["seconds"],
                            "s_per_step": measured["seconds"] / args.steps,
                            "losses": measured["losses"], "counts": measured["counts"]})
        comparisons.append(compare_runs(pair["reference"], pair["optimized"], initial, args.rtol, args.atol))
        del pair

    # An untimed replay also compares final-step gradients before optimizer.step.
    probes = {}
    for mode in ("reference", "optimized"):
        set_mode(mode)
        restore_state(trainer, initial)
        probes[mode] = run(mode, batches, gradients=True)
        probes[mode]["state"] = capture_state(trainer)
    gradient_probe = compare_runs(probes["reference"], probes["optimized"], initial, args.rtol, args.atol)
    restore_state(trainer, initial)
    set_mode("reference")

    modes = {mode: {"median_s_per_step": statistics.median(w["s_per_step"] for w in windows if w["mode"] == mode),
                    "cpu_collate_mean_ms": 1000 * statistics.mean(fetch[mode])}
             for mode in ("reference", "optimized")}
    report = {"passed": collator_exact and gradient_probe["passed"] and all(c["passed"] for c in comparisons),
              "measurement": "training replay only; excludes fetch and H2D; CPU collation reported separately",
              "modes": modes, "speedup": modes["reference"]["median_s_per_step"] / modes["optimized"]["median_s_per_step"],
              "warmup_seconds": warmup_s, "windows": windows, "comparisons": comparisons,
              "gradient_probe": gradient_probe, "collator_exact": collator_exact,
              "collator_checks": collator_checks, "batch_indices": indices,
              "tolerances": {"rtol": args.rtol, "atol": args.atol},
              "max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
              "config": json.loads(json.dumps(trainer.config, default=str)),
              "candidate": {"selective_ar_head": args.selective_ar_head},
              "provenance": provenance(args.selective_ar_head)}
    raise ProfileDone(report)


def provenance(selective_ar_head=False):
    import accelerate
    import transformers

    files = ["scripts/profile_training_runtime.py", "scripts/train_stage2.py", "src/bit_experiments/performance.py",
             "src/bit_experiments/training_runtime.py", "src/bit_experiments/exact_data.py",
             "src/bit_experiments/diffusion_llm.py", "src/bit_experiments/fast_dllm_v2/modeling.py",
             "external/bit/models/trainer.py"]
    if selective_ar_head:
        files.append("src/bit_experiments/selective_ar.py")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__, "accelerate": accelerate.__version__,
            "python": sys.version, "gpu": props.name, "gpu_capability": [props.major, props.minor],
            "gpu_memory_bytes": props.total_memory,
            "module_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}


@contextmanager
def intercepted_training(train_module, trainer_class, argv, callback, root=ROOT):
    """Reserve only our unique run directories; always restore process hooks."""
    name = argv[argv.index("--name") + 1]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("profile run name must be a simple directory name")
    owned = []
    old_train, old_argv, old_cwd = trainer_class.train, sys.argv, Path.cwd()
    try:
        for directory in (root / "results/stage2" / name, root / "checkpoints/stage2" / name):
            directory.mkdir(parents=True, exist_ok=False)
            owned.append(directory)
        trainer_class.train = callback
        sys.argv = argv
        yield train_module
    finally:
        trainer_class.train, sys.argv = old_train, old_argv
        os.chdir(old_cwd)
        for directory in owned:
            shutil.rmtree(directory)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="aero-1-audio-diffusion", help="registry entry, or default for BIT's Aero AR")
    ap.add_argument("--selective-ar-head", action="store_true", help="also test the opt-in selective AR head")
    ap.add_argument("--init", default="checkpoints/ssl_human/BEST_core")
    ap.add_argument("--arm", default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2, help="extra warmup steps, in addition to every replay batch")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    if (args.steps < 1 or args.rounds < 2 or args.warmup < 0
            or not math.isfinite(args.rtol) or not math.isfinite(args.atol) or args.rtol < 0 or args.atol < 0):
        ap.error("need steps>=1, rounds>=2, warmup>=0 and nonnegative tolerances")
    if not torch.cuda.is_available():
        ap.error("requires CUDA; run through the GPU lock on an idle training GPU")
    spec = importlib.util.spec_from_file_location("profiled_stage2", ROOT / "scripts/train_stage2.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    from models.trainer import Trainer

    name = f"runtime_profile_{uuid.uuid4().hex}"
    command = ["train_stage2.py", "--runtime", "reference", "--name", name, "--arm", args.arm,
               "--init", args.init, "--seed", str(args.seed)]
    if args.llm != "default":
        command += ["--llm", args.llm]
    try:
        with intercepted_training(train_module, Trainer, command, lambda trainer: profile_train(trainer, args), root=ROOT):
            train_module.main()
            raise RuntimeError("Stage 2 returned without entering the benchmark")
    except ProfileDone as done:
        report = done.report
    output = args.output or ROOT / "results/profile" / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "speedup": report["speedup"],
                      "modes": report["modes"], "report": str(output)}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
