"""Opt-in mask caches preserve the plain decoder's masks, RNG and training trajectory.

No GPU is needed. A CPU BlockMask constructor stands in for CUDA in metadata/key tests; actual CUDA
kernel parity and throughput are deliberately left to the hardware gate.
"""

import copy
from functools import partial
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("peft")

from bit_experiments import diffusion_llm  # noqa: E402
from bit_experiments.fast_dllm_v2 import modeling  # noqa: E402
from test_fast_block_diffusion import BS, MASK, layout, parts, tiny_llm  # noqa: E402


def assert_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
    assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().contiguous().reshape(-1).view(torch.uint8))


def decoder():
    return parts(tiny_llm())[0]


def test_disabled_by_default_and_configuration_validation():
    model = decoder()
    assert model._mask_cache_max_entries == 0
    a = model.gen_mask(64, BS, 2, 4, "cpu")
    b = model.gen_mask(64, BS, 2, 4, "cpu")
    assert a is not b and not model._mask_cache
    assert_bytes(a, b)
    for value in (-1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            model.configure_mask_cache(value)


def test_training_cache_retains_all_shape_keys_and_rng():
    model = decoder()
    calls = [(64, 32, 2, 4), (64, 16, 2, 4), (128, 32, 2, 4),
             (64, 32, 4, 4), (64, 32, 2, 8)]
    refs = [model.gen_mask(*args, device="cpu") for args in calls]
    state = torch.get_rng_state().clone()
    model.configure_mask_cache(8)
    for args, ref in zip(calls, refs):
        got = model.gen_mask(*args, device="cpu")
        assert_bytes(ref, got)
        assert got is model.gen_mask(*args, device=torch.device("cpu:0"))
    assert len(model._mask_cache) == len(calls)
    assert_bytes(state, torch.get_rng_state())


def test_one_lru_bound_covers_both_modes():
    model = decoder().configure_mask_cache(2)
    train = model.gen_mask(64, BS, 2, 4, "cpu")
    first_eval = model.eval_mask(32, BS, 64)
    assert train is model.gen_mask(64, BS, 2, 4, "cpu")  # refresh training entry
    model.eval_mask(32, BS, 96)
    assert len(model._mask_cache) == 2
    assert train is model.gen_mask(64, BS, 2, 4, "cpu")
    assert first_eval is not model.eval_mask(32, BS, 64)  # oldest was evicted
    assert len(model._mask_cache) == 2
    model.configure_mask_cache(0)
    assert not model._mask_cache


@pytest.mark.parametrize("seqlen,block_size,cache_len", [(1, 32, 0), (32, 32, 64), (17, 16, 32)])
def test_eval_device_cache_matches_original(seqlen, block_size, cache_len):
    model = decoder()
    ref = model.eval_mask(seqlen, block_size, cache_len).to("cpu")[None, None]
    state = torch.get_rng_state().clone()
    model.configure_mask_cache(3)
    got = model.eval_mask(seqlen, block_size, cache_len, device="cpu")
    assert_bytes(ref, got[None, None])
    assert got is model.eval_mask(seqlen, block_size, cache_len, device="cpu")
    assert_bytes(state, torch.get_rng_state())


def test_cuda_mask_preserves_batch_heads_metadata_and_concrete_device(monkeypatch):
    model = decoder().configure_mask_cache(8)
    original = modeling.create_block_mask
    calls = []
    current = [1]

    def cpu_constructor(mask_mod, **kwargs):
        calls.append(kwargs.copy())
        return original(mask_mod, **(kwargs | {"device": "cpu"}))

    monkeypatch.setattr(modeling, "create_block_mask", cpu_constructor)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    got = model.gen_mask(64, BS, 2, 4, "cuda")
    assert got is model.gen_mask(64, BS, 2, 4, "cuda:1")
    assert len(calls) == 1
    assert calls[0] == {"B": 2, "H": 4, "Q_LEN": 128, "KV_LEN": 128, "device": torch.device("cuda:1")}
    current[0] = 2
    assert got is not model.gen_mask(64, BS, 2, 4, "cuda")
    assert len(calls) == 2 and calls[1]["device"] == torch.device("cuda:2")
    ref = original(partial(modeling.block_diff_mask, block_size=BS, n=64), 2, 4, 128, 128, device="cpu")
    assert got.seq_lengths == ref.seq_lengths and got.BLOCK_SIZE == ref.BLOCK_SIZE
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                 "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):
        assert_bytes(getattr(got, name), getattr(ref, name))


def test_apply_clears_cache_and_state_dict_has_no_cache():
    model = decoder().configure_mask_cache(2)
    keys = set(model.state_dict())
    model.gen_mask(64, BS, 2, 4, "cpu")
    model.eval_mask(32, BS, 64)
    assert set(model.state_dict()) == keys
    model.to(dtype=torch.float64)
    assert not model._mask_cache and model._mask_cache_max_entries == 2
    model.eval_mask(32, BS, 64)
    model.clear_mask_cache()
    assert not model._mask_cache


def test_inference_populated_masks_can_be_saved_for_backward():
    model = decoder().configure_mask_cache(2)
    with torch.inference_mode():
        train = model.gen_mask(64, BS, 2, 4, "cpu")
        evaluation = model.eval_mask(32, BS, 64)
    assert not torch.is_inference(train) and not torch.is_inference(evaluation)
    for mask in (train, evaluation):
        x = torch.ones(mask.shape, requires_grad=True)
        x.masked_fill(~mask, 0).sum().backward()
        assert_bytes(x.grad, mask.float())


@pytest.mark.parametrize("autocast", [False, True])
def test_cache_preserves_multistep_lora_training_and_rng(autocast):
    initial = tiny_llm(17, dropout=0.2)
    batches = [layout(rows, seed=40 + i) for i, rows in enumerate(
        [[(20, 5), (41, 8)], [(24, 6), (40, 4)], [(140, 5)], [(20, 5), (41, 8)]])]

    def train(cache_size):
        llm = copy.deepcopy(initial)
        parts(llm)[0].configure_mask_cache(cache_size)
        opt = torch.optim.AdamW(llm.parameters(), lr=1e-4)
        records = []
        torch.manual_seed(91)  # once for the whole trajectory, never reset between steps
        for clean, labels in batches:
            x = clean.clone().requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out = diffusion_llm.block_diffusion_ce(*parts(llm), x, labels, BS, MASK)
            out[0].backward()
            record = {"out": [t.detach().clone() for t in out], "xgrad": x.grad.clone(),
                      "grads": {n: p.grad.clone() for n, p in llm.named_parameters() if p.grad is not None}}
            opt.step()
            opt.zero_grad(set_to_none=True)
            record["weights"] = {n: p.detach().clone() for n, p in llm.named_parameters() if p.requires_grad}
            record["optimizer"] = copy.deepcopy(opt.state_dict())
            record["rng"] = torch.get_rng_state().clone()
            records.append(record)
        return records

    plain, cached = train(0), train(1)  # hits, eviction, then a shape returning after eviction
    for a, b in zip(plain, cached):
        for x, y in zip(a["out"], b["out"]):
            assert_bytes(x, y)
        for key in ("xgrad", "rng"):
            assert_bytes(a[key], b[key])
        for key in ("grads", "weights"):
            assert a[key].keys() == b[key].keys()
            for name in a[key]:
                assert_bytes(a[key][name], b[key][name])
        assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
        for param, states in a["optimizer"]["state"].items():
            for key, value in states.items():
                assert_bytes(value, b["optimizer"]["state"][param][key])
