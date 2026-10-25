"""Selective AR head: real Qwen2/text and Qwen2Audio + LoRA, no downloads or GPU.

The decoder/dropout RNG must match exactly. FP32 logits/loss/gradients use 2e-5 relative-to-maximum
error (plus 2e-7 absolute); BF16 uses 0.5% (plus 2e-6). AdamW parameter/state comparisons use the same
bound per tensor. These are rounding gates, not evidence of production CUDA speed or accuracy.
scripts/benchmark_selective_ar.py measures a separate synthetic CPU forward+backward workload.
"""

import copy
import inspect
from pathlib import Path
import sys
from types import MethodType

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external" / "bit")]
peft = pytest.importorskip("peft")
from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration, Qwen2Config, Qwen2ForCausalLM

from models.bci import BCI_AudioLLM, ContrastiveLoss, ModalityAligner
from bit_experiments.stage2 import packed_merge
from bit_experiments.text_llm import BCI_TextLLM
from bit_experiments.selective_ar import install_selective_ar_head, uninstall_selective_ar_head

D, V = 64, 96


def neural(self, spikes, spikes_mask, spikes_timestamp, dataset_name=None, day_idx=None):
    return self.neural_input(self.neural_dropout(spikes)), spikes_mask


def make_model(route="text", contrastive=False, vocab=V):
    torch.manual_seed(19)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      bos_token_id=vocab - 3, eos_token_id=vocab - 2, pad_token_id=vocab - 1,
                      attention_dropout=0.0, use_sliding_window=False, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    if route == "text":
        llm = Qwen2ForCausalLM(cfg)
        model = BCI_TextLLM.__new__(BCI_TextLLM)
    else:
        audio = {"num_mel_bins": 8, "encoder_layers": 1, "encoder_attention_heads": 2,
                 "encoder_ffn_dim": 32, "d_model": 16, "max_source_positions": 16}
        llm = Qwen2AudioForConditionalGeneration(Qwen2AudioConfig(text_config=cfg.to_dict(), audio_config=audio))
        model = BCI_AudioLLM.__new__(BCI_AudioLLM)
    nn.Module.__init__(model)
    model.llm = peft.get_peft_model(llm, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for name, param in model.llm.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    model.llm_config = model.llm.config
    model.project_to_audio = False
    model.neural_input, model.neural_dropout = nn.Linear(7, D), nn.Dropout(0.1)
    model.projector = nn.Linear(D, D)
    model.prepare_neural_embeds = MethodType(neural, model)
    model.merge_embeds = MethodType(packed_merge, model)
    model.modality_aligner = ModalityAligner(D, proj_dim=8) if contrastive else None
    model.contrastive_loss_fn = ContrastiveLoss(reduction="sum") if contrastive else None
    model.loss_fn = nn.CrossEntropyLoss(reduction="sum")
    return model.train()


def batch(seed=7, text_length=11, neural_length=9, vocab=V, ignored=False):
    rng = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab - 3, (2, text_length), generator=rng)
    target = torch.full_like(ids, -100)
    if not ignored:
        target[0, -4:] = ids[0, -4:]
        target[1, -2:] = ids[1, -2:]
    mask = torch.ones(2, neural_length, dtype=torch.long)
    mask[1, -3:] = 0
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "input_split": torch.tensor([2, 3]),
            "spikes": torch.randn(2, neural_length, 7, generator=rng), "spikes_mask": mask,
            "spikes_timestamp": torch.arange(neural_length)[None].expand(2, -1),
            "spikes_lengths": mask.sum(-1), "targets": target,
            "dataset_name": torch.zeros(2, dtype=torch.long), "day_idx": torch.zeros(2, dtype=torch.long)}


def close(a, b, bf16=False):
    assert a.dtype == b.dtype and a.shape == b.shape
    a, b = a.float(), b.float()
    if not a.numel():
        return
    relative, absolute = (5e-3, 2e-6) if bf16 else (2e-5, 2e-7)
    assert (a - b).abs().max() <= absolute + relative * b.abs().max()


def run(model, inputs, bf16=False, seed=31, optimizer=None):
    model.zero_grad(set_to_none=True)
    inputs = {key: value.clone() for key, value in inputs.items()}
    inputs["spikes"].requires_grad_()
    if seed is not None:
        torch.manual_seed(seed)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out = model(**inputs)
    out.loss.backward()
    supervised = out.targets[..., 1:] != -100 if out.preds.ndim == 3 else None
    logits = out.preds[..., :-1, :][supervised] if supervised is not None else out.preds
    labels = out.targets[..., 1:][supervised] if supervised is not None else out.targets
    record = {"loss": out.loss.detach(), "n": out.n_examples.detach(), "logits": logits.detach(),
              "labels": labels.detach(), "input_grad": inputs["spikes"].grad.clone(),
              "grads": {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None},
              "rng": torch.get_rng_state().clone()}
    if optimizer:
        optimizer.step()
        record["weights"] = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        record["optimizer"] = copy.deepcopy(optimizer.state_dict())
    return record


def compare(a, b, bf16=False):
    for name in ("n", "labels", "rng"):
        assert torch.equal(a[name], b[name]), name
    for name in ("loss", "logits", "input_grad"):
        close(a[name], b[name], bf16)
    assert a["grads"].keys() == b["grads"].keys()
    for name in a["grads"]:
        close(a["grads"][name], b["grads"][name], bf16)


@pytest.mark.parametrize("route", ["text", "audio"])
@pytest.mark.parametrize("contrastive", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_real_qwen2_lora_loss_gradients_rng(route, contrastive, bf16):
    reference = make_model(route, contrastive)
    candidate = copy.deepcopy(reference)
    install_selective_ar_head(candidate)
    before_ids = {n: id(p) for n, p in candidate.named_parameters()}
    a, b = run(reference, batch(), bf16), run(candidate, batch(), bf16)
    compare(a, b, bf16)
    assert before_ids == {n: id(p) for n, p in candidate.named_parameters()}
    assert all(grad.abs().max() > 0 for name, grad in b["grads"].items() if "lora_" in name)


@pytest.mark.parametrize("route", ["text", "audio"])
@pytest.mark.parametrize("bf16", [False, True])
def test_multistep_adamw_with_unreset_rng(route, bf16):
    initial = make_model(route, contrastive=True)
    all_records = []
    for selective in (False, True):
        model = copy.deepcopy(initial)
        if selective:
            install_selective_ar_head(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.manual_seed(123)
        records = [run(model, batch(seed=8 + i, text_length=11 + i), bf16, seed=None, optimizer=optimizer)
                   for i in range(3)]
        all_records.append(records)
    for a, b in zip(*all_records):
        compare(a, b, bf16)
        for name in a["weights"]:
            close(a["weights"][name], b["weights"][name], bf16)
        assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
        for param, states in a["optimizer"]["state"].items():
            for name, value in states.items():
                close(value, b["optimizer"]["state"][param][name], bf16)


@pytest.mark.parametrize("contrastive", [False, True])
def test_all_ignored_targets_keep_zero_gradient_paths(contrastive):
    reference = make_model(contrastive=contrastive)
    candidate = copy.deepcopy(reference)
    install_selective_ar_head(candidate)
    a, b = run(reference, batch(ignored=True)), run(candidate, batch(ignored=True))
    compare(a, b)
    assert b["n"].item() == 0 and b["logits"].shape == (0, V)
    if not contrastive:
        assert b["loss"].item() == 0
        assert all(torch.count_nonzero(g) == 0 for g in b["grads"].values())


def test_causal_decoder_never_uses_future_target_token():
    model = make_model().eval()
    install_selective_ar_head(model)
    inputs = batch()
    with torch.no_grad():
        before = model(**inputs).preds
        inputs["input_ids"][:, -1] = (inputs["input_ids"][:, -1] + 1) % (V - 3)
        after = model(**inputs).preds
    assert torch.equal(before, after)


@pytest.mark.parametrize("route", ["text", "audio"])
def test_install_uninstall_signature_state_and_generation(route):
    model = make_model(route).eval()
    signature, keys = inspect.signature(model.forward), set(model.state_dict())
    original_generate = model.generate.__func__
    original_forward = model.forward.__func__
    inputs = batch()
    generation = {k: v[:1] for k, v in inputs.items() if k != "targets"}
    options = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": V - 1, "eos_token_id": V - 2}
    with torch.no_grad():
        original = model.generate(**generation, **options)
    report = install_selective_ar_head(model)
    assert report == install_selective_ar_head(model)
    actual_signature = inspect.signature(model.forward)
    assert list(signature.parameters) == list(actual_signature.parameters)
    assert [(p.kind, p.default) for p in signature.parameters.values()] == [
        (p.kind, p.default) for p in actual_signature.parameters.values()]
    assert set(model.state_dict()) == keys and model.generate.__func__ is original_generate
    with torch.no_grad():
        actual = model.generate(**generation, **options)
    assert torch.equal(original, actual)
    assert uninstall_selective_ar_head(model)
    assert not uninstall_selective_ar_head(model)
    assert model.forward.__func__ is original_forward and "forward" not in model.__dict__


@pytest.mark.parametrize("problem", ["audio_projection", "head_trainable", "head_stochastic", "loss_mean",
                                     "loss_weight", "loss_smoothing", "decoder", "forward", "quantized"])
def test_unsupported_configuration_fails_closed(problem):
    model = make_model()
    lm = model.llm.get_base_model()
    if problem == "audio_projection":
        model.project_to_audio = True
    elif problem == "head_trainable":
        lm.lm_head.weight.requires_grad_(True)
    elif problem == "head_stochastic":
        lm.lm_head = nn.Sequential(nn.Dropout(0.2), lm.lm_head)
    elif problem == "loss_mean":
        model.loss_fn = nn.CrossEntropyLoss()
    elif problem == "loss_weight":
        model.loss_fn = nn.CrossEntropyLoss(reduction="sum", weight=torch.ones(V))
    elif problem == "loss_smoothing":
        model.loss_fn = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.1)
    elif problem == "decoder":
        model.llm = nn.Linear(3, 3)
    elif problem == "forward":
        model.forward = MethodType(lambda self, **kwargs: None, model)
    else:
        lm.is_quantized = True
    with pytest.raises(ValueError):
        install_selective_ar_head(model)
    assert not hasattr(model, "_selective_ar_original_forward")


def test_missing_targets_fails_before_any_random_draw():
    model = make_model()
    install_selective_ar_head(model)
    inputs = batch()
    del inputs["targets"]
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="requires targets"):
        model(**inputs)
    assert torch.equal(before, torch.get_rng_state())


