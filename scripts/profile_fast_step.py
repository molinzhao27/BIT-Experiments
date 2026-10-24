#!/usr/bin/env python
"""GPU-busy % and step time of the packed diffusion step with its host waits (fast_step) and without them (sync_free),
in one process on one model (docs/DIFFUSION_SPEEDUP.md "Host waits"). Takes about 4-5 minutes. Run it on the GPU
through the lock wrapper, from the code copy that holds this commit's files:

    <bit env>/python scripts/profile_fast_step.py --llm aero-1-audio-diffusion-fast [--init checkpoints/ssl_human/BEST_core]

It builds the run exactly as train_stage2.py does (same config, model, init, data and BIT trainer). In place of the
training loop it runs BIT's training step (autocast forward, backward, optimizer and scheduler step, zero_grad, and
the two .item() calls of the trainer's loss log), switching the model's sync_free flag between windows:
- warm-up: compiles, recompiles for dynamic shapes;
- --rounds x (--steps with the host waits, --steps without), alternating so that drift affects both alike. Per step
  it records wall time and its phases: fetch (data loader), forward and backward (the time the CPU spends queuing
  them), optimizer, and item (the time the CPU waits at the loss log for the GPU to finish). If nvidia-smi is there,
  it samples GPU utilization and power every 100 ms;
- per mode, one step under torch's CUDA sync debug mode, which lists every host wait by file:line;
- per mode, --profile-steps steps under torch.profiler: GPU-busy % from the union of device-event intervals over
  wall time. The profiler adds CPU overhead, so this busy % is a lower bound; nvidia-smi's figure is unprofiled.
Training runs for real (optimizer steps), but nothing is saved. Its results/stage2/<name> and checkpoints/stage2/<name>
are removed at the end. Writes results/profile/<name>.json and prints a summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics as st
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class ProfileDone(Exception):
    """Raised by the replacement training loop to leave train_stage2.main before its evaluation."""

    def __init__(self, results):
        super().__init__("profile done")
        self.results = results


class SmiSampler:
    """nvidia-smi utilization (%) and power (W) every 100 ms while running; empty if nvidia-smi is missing."""

    def __enter__(self):
        self.rows = []
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                                          "--format=csv,noheader,nounits", "-lms", "100"],
                                         stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            self.proc = None
            return self
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                util, power = (float(v) for v in line.split(",")[:2])
                self.rows.append((util, power))
            except ValueError:
                pass

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()


def busy_fraction(prof, wall_s: float) -> float:
    """Union of the device events' time intervals over the wall time."""
    from torch.autograd import DeviceType

    spans = sorted((e.time_range.start, e.time_range.end) for e in prof.events() if e.device_type == DeviceType.CUDA)
    busy, end = 0.0, float("-inf")
    for s, e in spans:
        if s > end:
            busy, end = busy + (e - s), e
        elif e > end:
            busy, end = busy + (e - end), e
    return busy / 1e6 / wall_s


def profile_train(self, args):
    """Replaces BIT's Trainer.train (module docstring)."""
    from bit_experiments.diffusion_llm import FAST_STEP_DEFAULTS

    cfg = self.config
    model = self.accelerator.unwrap_model(self.model)
    if not (FAST_STEP_DEFAULTS | model.sampler)["fast_step"]:
        raise SystemExit(f"{args.llm} has no fast_step: nothing to compare")
    if cfg.optimizer.gradient_accumulation_steps != 1:
        raise SystemExit("expected gradient_accumulation_steps 1 (BIT's sentence stage)")
    self.model.train()

    def batches():
        while True:
            yield from self.train_dataloader

    it = batches()

    def step(sync_free: bool) -> dict:
        model.sampler["sync_free"] = sync_free
        t0 = time.perf_counter()
        model_inputs, _ = next(it)
        t1 = time.perf_counter()
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):  # BIT's train(), one step
            outputs = self.model(**model_inputs)
            n_examples = outputs.n_examples
            loss = outputs.loss / cfg.optimizer.gradient_accumulation_steps
        t2 = time.perf_counter()
        self.accelerator.backward(loss)
        t3 = time.perf_counter()
        self.optimizer.step()
        if cfg.optimizer.scheduler in ["linear", "cosine"]:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        t4 = time.perf_counter()
        loss_v = self.accelerator.gather(loss).sum().detach().item()
        n_v = self.accelerator.gather(n_examples).sum().detach().item()
        t5 = time.perf_counter()
        return {"fetch": t1 - t0, "forward": t2 - t1, "backward": t3 - t2, "optimizer": t4 - t3, "item": t5 - t4,
                "total": t5 - t0, "loss": loss_v, "n": n_v}

    modes = {"fast_step": False, "sync_free": True}
    t_start = time.time()
    for name, flag in (("fast_step", False), ("sync_free", True), ("fast_step", False)):
        n = args.warmup if name == "fast_step" else args.warmup // 2
        for _ in range(n):
            step(flag)
    torch.cuda.synchronize()
    warmup_s = time.time() - t_start

    steps = {m: [] for m in modes}
    smi = {m: [] for m in modes}
    for _ in range(args.rounds):
        for m, flag in modes.items():
            with SmiSampler() as s:
                steps[m] += [step(flag) for _ in range(args.steps)]
            smi[m] += s.rows

    syncs = {}
    for m, flag in modes.items():
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step(flag)
        torch.cuda.set_sync_debug_mode(0)
        hits = [w for w in caught if "synchroniz" in str(w.message)]
        where = Counter(f"{Path(w.filename).name}:{w.lineno}" for w in hits)
        syncs[m] = {"count": len(hits), "where": dict(where.most_common())}

    busy = {}
    for m, flag in modes.items():
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(args.profile_steps):
                step(flag)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        waits = Counter(e.name for e in prof.events()
                        if e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"))
        busy[m] = {"busy_fraction_profiled": round(busy_fraction(prof, wall), 3),
                   "s_per_step_profiled": round(wall / args.profile_steps, 4),
                   "sync_calls_per_step": {k: round(v / args.profile_steps, 1) for k, v in waits.items()}}

    def summary(m):
        rows = steps[m]
        out = {"steps": len(rows), "s_per_step": round(st.mean(r["total"] for r in rows), 4),
               "it_per_s": round(len(rows) / sum(r["total"] for r in rows), 2),
               "phases_ms": {k: round(1e3 * st.mean(r[k] for r in rows), 1)
                             for k in ("fetch", "forward", "backward", "optimizer", "item")},
               "train_loss_mean": round(sum(r["loss"] for r in rows) / sum(r["n"] for r in rows), 4),
               "host_waits_per_step": syncs[m], **busy[m]}
        if smi[m]:
            util = [u for u, _ in smi[m]]
            out["smi"] = {"util_mean": round(st.mean(util), 1),
                          "frac_samples_under_50": round(sum(u < 50 for u in util) / len(util), 3),
                          "power_mean_w": round(st.mean(p for _, p in smi[m]), 1), "samples": len(util)}
        return out

    results = {"llm": args.llm, "warmup_s": round(warmup_s, 1), "modes": {m: summary(m) for m in modes},
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "torch": torch.__version__}
    a, b = results["modes"]["fast_step"], results["modes"]["sync_free"]
    results["speedup"] = round(a["s_per_step"] / b["s_per_step"], 3)
    raise ProfileDone(results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="aero-1-audio-diffusion-fast", help="a configs/llms.yaml entry with fast_step")
    ap.add_argument("--init", default="checkpoints/ssl_human/BEST_core")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=30, help="warm-up steps (compiles), per mode")
    ap.add_argument("--steps", type=int, default=60, help="steps per window")
    ap.add_argument("--rounds", type=int, default=2, help="alternating windows per mode")
    ap.add_argument("--profile-steps", type=int, default=12, help="steps under torch.profiler, per mode")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("needs a GPU")

    spec = importlib.util.spec_from_file_location("train_stage2", ROOT / "scripts" / "train_stage2.py")
    train_stage2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_stage2)  # puts src/ and BIT on sys.path
    from models.trainer import Trainer

    name = f"profile_fast_step_{time.strftime('%Y%m%d_%H%M%S')}"
    Trainer.train = lambda self: profile_train(self, args)
    sys.argv = ["train_stage2.py", "--name", name, "--arm", "B", "--init", args.init, "--seed", str(args.seed),
                "--llm", args.llm, "--epochs", "1", "--eval-every", "1"]
    try:
        train_stage2.main()
        raise SystemExit("train_stage2.main returned without running the profile")
    except ProfileDone as done:
        results = done.results
    finally:
        for d in (ROOT / "results" / "stage2" / name, ROOT / "checkpoints" / "stage2" / name):
            if d.name.startswith("profile_fast_step_"):
                shutil.rmtree(d, ignore_errors=True)
    out = ROOT / "results" / "profile" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    for m, r in results["modes"].items():
        print(f"[profile] {m:9s} {r['s_per_step']:.3f} s/step ({r['it_per_s']} it/s) | phases ms {r['phases_ms']} | "
              f"host waits/step {r['host_waits_per_step']['count']} | busy (profiled) {r['busy_fraction_profiled']}"
              + (f" | smi util {r['smi']['util_mean']}%, <50%: {r['smi']['frac_samples_under_50']}" if "smi" in r else ""))
    print(f"[profile] speedup {results['speedup']}x -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
