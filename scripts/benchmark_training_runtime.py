#!/usr/bin/env python
"""Paired CPU component benchmarks; these are NOT production GPU throughput estimates.

Run in BIT's environment: python scripts/benchmark_training_runtime.py --output results/profile/runtime_cpu.json
Uses synthetic data, no downloads/checkpoints, one CPU thread by default. The real-GPU gate is
scripts/profile_training_runtime.py. Reference/optimized order alternates to reduce timing drift.
"""

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external" / "bit")]

from utils.datasets import pad_collate_fn
from bit_experiments.exact_data import exact_pad_collate_fn
from bit_experiments.diffusion_llm import BCI_DiffusionLLM
from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig
from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM


def paired(reference, optimized, rounds, iterations):
    samples = {"reference": [], "optimized": []}
    functions = {"reference": reference, "optimized": optimized}
    for fn in functions.values():
        fn()
    for round_id in range(rounds):
        order = ("reference", "optimized") if round_id % 2 == 0 else ("optimized", "reference")
        for mode in order:
            start = time.perf_counter()
            for _ in range(iterations):
                functions[mode]()
            samples[mode].append((time.perf_counter() - start) * 1000 / iterations)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    return {"median_ms": medians, "speedup": medians["reference"] / medians["optimized"],
            "samples_ms": samples, "iterations_per_round": iterations}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.rounds < 3 or args.threads < 1:
        ap.error("use at least 3 rounds and 1 thread")
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(42)
    rows = [{"spikes": rng.normal(size=(1200 - 43 * i, 512)).astype(np.float32),
             "input_ids": np.arange(12 + i, dtype=np.int64)} for i in range(8)]
    pads = {"spikes": {"dim": 0, "side": "right", "value": -100},
            "input_ids": {"dim": 0, "side": "right", "value": 0}}
    keys = list(pads)
    ref = lambda: pad_collate_fn(rows, keys, pads)
    opt = lambda: exact_pad_collate_fn(rows, keys, pads)
    for key in keys:
        assert torch.equal(ref()[0][key].view(torch.uint8), opt()[0][key].view(torch.uint8))
    report = {"scope": "synthetic CPU components, not whole-job speedup", "torch": torch.__version__,
              "python": platform.python_version(), "machine": platform.machine(), "threads": args.threads,
              "collation": paired(ref, opt, args.rounds, 30)}

    torch.manual_seed(31)
    cfg = Fast_dLLM_QwenConfig(vocab_size=8192, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               max_position_embeddings=1024, bos_token_id=8188, eos_token_id=8189, bd_size=32)
    llm = Fast_dLLM_QwenForCausalLM(cfg).eval()
    decoder = llm.model
    raw = decoder.gen_mask(384, 32, 16, 12, "cpu")
    decoder.configure_mask_cache(32)
    cached = decoder.gen_mask(384, 32, 16, 12, "cpu")
    assert torch.equal(raw, cached)
    report["dense_mask_L384"] = paired(
        lambda: decoder._build_train_mask(384, 32, 16, 12, torch.device("cpu")),
        lambda: decoder.gen_mask(384, 32, 16, 12, "cpu"), args.rounds, 30)

    model = BCI_DiffusionLLM.__new__(BCI_DiffusionLLM)
    torch.nn.Module.__init__(model)
    model.llm = llm
    model._configure({"block_size": 32, "small_block_size": 8, "threshold": 0.9, "mask_id": 8191})
    prefix = torch.randn(193, 64)

    @torch.no_grad()
    def decode(skip):
        model._skip_unused_prefill_logits = skip
        with model._layout(training=False):
            return model._denoise(prefix, 8189, max_new_tokens=8, do_sample=False,
                                  top_p=1.0, temperature=1.0)

    # Isolate prefill-head elimination: both modes use the same mask cache.
    assert decode(False) == decode(True)
    report["decode_D64_V8192_prefix193"] = paired(lambda: decode(False), lambda: decode(True), args.rounds, 3)
    report["parity"] = {"collation_bytes": True, "mask_values": True, "greedy_tokens": True}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
