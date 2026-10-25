#!/usr/bin/env python
"""Paired selective AR head CPU forward/backward benchmark; no downloads and no test imports.

This measures a synthetic B2,L128,D64,V8192 Qwen2 + LoRA workload, NOT production GPU throughput.
Seven alternating timing windows follow warmup. Both variants start each window from the same RNG
state; neither updates weights. A separate loss/logit/gradient/RNG check gates the reported speedup.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

import peft
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external" / "bit")]
from bit_experiments.selective_ar import install_selective_ar_head, uninstall_selective_ar_head
from bit_experiments.stage2 import packed_merge
from bit_experiments.text_llm import BCI_TextLLM

D, V, B, L = 64, 8192, 2, 128


def _neural(self, spikes, spikes_mask, spikes_timestamp, dataset_name=None, day_idx=None):
    return self.neural_input(spikes), spikes_mask


def make_workload():
    torch.manual_seed(19)
    cfg = Qwen2Config(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      bos_token_id=V - 3, eos_token_id=V - 2, pad_token_id=V - 1,
                      attention_dropout=0.0, use_sliding_window=False, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    model = BCI_TextLLM.__new__(BCI_TextLLM)
    nn.Module.__init__(model)
    model.llm = peft.get_peft_model(Qwen2ForCausalLM(cfg), peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for name, parameter in model.llm.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.05)
    model.project_to_audio = False
    model.neural_input, model.projector = nn.Linear(7, D), nn.Linear(D, D)
    model.prepare_neural_embeds = MethodType(_neural, model)
    model.merge_embeds = MethodType(packed_merge, model)
    model.modality_aligner = model.contrastive_loss_fn = None
    model.loss_fn = nn.CrossEntropyLoss(reduction="sum")
    rng = torch.Generator().manual_seed(31)
    ids = torch.randint(0, V - 3, (B, 32), generator=rng)
    targets = torch.full_like(ids, -100)
    targets[0, -4:], targets[1, -2:] = ids[0, -4:], ids[1, -2:]
    data = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "input_split": torch.tensor([2, 3]),
            "spikes": torch.randn(B, 96, 7, generator=rng).requires_grad_(),
            "spikes_mask": torch.ones(B, 96, dtype=torch.long),
            "spikes_timestamp": torch.arange(96)[None].expand(B, -1),
            "spikes_lengths": torch.full((B,), 96), "targets": targets}
    return model.train(), data


def _toggle(model, mode):
    return install_selective_ar_head(model) if mode == "selective" else uninstall_selective_ar_head(model)


def run_benchmark(rounds=7, iterations=5, warmup=2):
    torch.set_num_threads(1)
    model, data = make_workload()
    records = {}
    for mode in ("reference", "selective"):
        _toggle(model, mode)
        model.zero_grad(set_to_none=True)
        data["spikes"].grad = None
        torch.manual_seed(101)
        out = model(**data)
        out.loss.backward()
        if mode == "reference":
            selected = out.targets[:, 1:] != -100
            logits, labels = out.preds[:, :-1][selected], out.targets[:, 1:][selected]
        else:
            logits, labels = out.preds, out.targets
        records[mode] = {"loss": out.loss.detach(), "n": out.n_examples.detach(), "logits": logits.detach(),
                         "labels": labels.detach(), "rng": torch.get_rng_state().clone(),
                         "input_grad": data["spikes"].grad.clone(),
                         "grads": {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}}
    reference, candidate = records["reference"], records["selective"]
    assert all(torch.equal(reference[k], candidate[k]) for k in ("n", "labels", "rng"))
    assert reference["grads"].keys() == candidate["grads"].keys()
    errors = {}
    pairs = {key: (reference[key], candidate[key]) for key in ("loss", "logits", "input_grad")}
    pairs.update({name: (value, candidate["grads"][name]) for name, value in reference["grads"].items()})
    for name, (ref, actual) in pairs.items():
        scale = float(ref.abs().max())
        difference = float((ref - actual).abs().max())
        errors[name] = difference / max(scale, 1e-30)
        assert difference <= 2e-7 + 2e-5 * scale, (name, difference, scale)
    del out, records, reference, candidate, pairs

    def step():
        model.zero_grad(set_to_none=True)
        data["spikes"].grad = None
        model(**data).loss.backward()

    for mode in ("reference", "selective"):
        _toggle(model, mode)
        torch.manual_seed(101)
        for _ in range(warmup):
            step()
    samples = {"reference": [], "selective": []}
    for round_id in range(rounds):
        for mode in (("reference", "selective") if round_id % 2 == 0 else ("selective", "reference")):
            _toggle(model, mode)
            torch.manual_seed(211 + round_id)
            start = time.perf_counter()
            for _ in range(iterations):
                step()
            samples[mode].append((time.perf_counter() - start) * 1000 / iterations)

    projected = {}
    rows = []
    hook = model.llm.get_base_model().lm_head.register_forward_pre_hook(
        lambda _, args: rows.append(args[0].numel() // D))
    try:
        for mode in ("reference", "selective"):
            _toggle(model, mode)
            step()
            projected[mode] = rows[-1]
    finally:
        hook.remove()
        uninstall_selective_ar_head(model)
    median = {key: statistics.median(value) for key, value in samples.items()}
    return {"scope": "synthetic CPU forward+backward, not GPU throughput", "torch": torch.__version__,
            "shape": {"B": B, "L": L, "D": D, "V": V}, "threads": 1, "lora_dropout": 0.2,
            "samples_ms": samples, "median_ms": median,
            "speedup": median["reference"] / median["selective"], "head_rows": projected,
            "logit_elements": {key: rows * V for key, rows in projected.items()},
            "logit_storage_bytes_fp32": {key: rows * V * 4 for key, rows in projected.items()},
            "logit_element_reduction": 1 - projected["selective"] / projected["reference"],
            "parity": {"targets_counts_rng_identical": True, "max_gradient_relative_error":
                       max(value for key, value in errors.items() if key not in ("loss", "logits")),
                       "loss_relative_error": errors["loss"], "logit_relative_error": errors["logits"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 3 or args.iterations < 1 or args.warmup < 1:
        parser.error("use at least 3 rounds, 1 iteration, and 1 warmup")
    result = run_benchmark(args.rounds, args.iterations, args.warmup)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
