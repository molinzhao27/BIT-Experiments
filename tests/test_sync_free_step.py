"""The packed step without host waits (bit_experiments.sync_free_step, sync_free: true) against the packed step with
them (fast_step), on the CPU. They should be bit for bit equal: the same random draws in the same order, the same
loss, logits, targets and gradients, and the same generator state afterwards. These tests need BIT's environment
(transformers 4.48.2, peft 0.11.1) and are skipped without peft.

    <bit env>/python -m pytest tests/test_sync_free_step.py

- The CPU layout gives forward's clean sequences and labels exactly. The CPU-built flex block mask has exactly the
  fields create_block_mask gives. pack / unpack round-trip.
- Step and forward are bit-identical to fast_step with real LoRA dropout on. This holds over seeds and layouts, for
  dense and flex attention, share_clean, pack_quantum, the aligned block, and T2T (which keeps the boolean
  selection).
- The option checks, and the registry: aero-1-audio-diffusion-fast stays as the running job uses it.
The host waits themselves can only be counted on a GPU (scripts/profile_fast_step.py, torch's sync debug mode).
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("peft")

from bit_experiments import diffusion_llm, fast_block_diffusion as fast, sync_free_step as sf  # noqa: E402
from test_fast_block_diffusion import (BS, EOS, LAYOUTS, MASK, REGISTRY, TAILS, BOS, batch, bci, layout,  # noqa: E402
                                       parts, tiny_llm)

ROWS = [(24, 5), (25, 12), (29, 25), (48, 3), (55, 18), (120, 9)]  # (neural tokens, sentence tokens): prefix 32..128
ROWS_T15 = [(240, 9), (140, 3), (200, 7), (85, 17)]                 # T15-like prefixes (run 3's: 85-502)


def same(a, b):
    """Bit for bit: same dtype, shape and values (NaN-free here)."""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


# ------------------------------------------------------------------ layout, block mask, transfer


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("rows", [ROWS, ROWS_T15])
def test_layout_is_forwards(rows, align):
    """forward's clean sequences and labels (captured at denoising_ce) are the CPU layout's gather and labels."""
    llm = tiny_llm(1)
    model = bci(llm, fast_step=True, align_sentence_block=align)
    b = batch(rows, 2)
    seen = {}
    model.denoising_ce = lambda clean, labels: seen.update(clean=clean, labels=labels) or (
        clean.sum(), labels.new_ones(()), clean, labels)
    model(**b)
    lay = sf.layout(b["attention_mask"], b["targets"], b["input_ids"], b["input_split"], b["spikes_mask"], BS, EOS,
                    align)
    embed = parts(llm)[2]
    text = embed(b["input_ids"])
    table = torch.cat([text.reshape(-1, text.shape[-1]), b["spikes"].reshape(-1, text.shape[-1]),
                       embed(torch.tensor([EOS, BOS]))])
    assert same(lay.labels, seen["labels"])
    assert same(table[lay.gather].view(*lay.labels.shape, -1), seen["clean"])


@pytest.mark.parametrize("quantum", [128, 512])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("name", [*LAYOUTS, "holes"])
def test_block_mask_is_create_block_masks(name, share, quantum):
    _, labels = layout(LAYOUTS.get(name, TAILS), holes=name == "holes")
    plan = fast.pack_plan(labels, BS, share, quantum)
    ref = fast.create_block_mask(plan.mask_mod(), None, None, len(plan), len(plan), device="cpu")
    got = sf.plan_block_mask(plan, plan, BS)
    assert got.seq_lengths == ref.seq_lengths and got.BLOCK_SIZE == ref.BLOCK_SIZE
    for f in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices", "q_num_blocks", "q_indices",
              "full_q_num_blocks", "full_q_indices"):
        assert same(getattr(got, f), getattr(ref, f)), f
    assert got.mask_mod.keywords["noisy"] is plan.noisy  # reads the plan's (device) tables


def test_pack_unpack_round_trip():
    ts = [torch.tensor([[1, -2], [3, 4]]), torch.tensor([True, False, True]), torch.arange(5, dtype=torch.int32),
          torch.zeros(0, dtype=torch.long)]
    back = sf.unpack(sf.pack(ts), ts)
    assert all(same(a, b) for a, b in zip(ts, back))
    assert sf.to_device(ts, "cpu") is ts


# ------------------------------------------------------------------ the step, bit for bit


def run_step(llm, clean, labels, seed, sync_free, **kw):
    llm.zero_grad()
    x = clean.clone().requires_grad_()
    torch.manual_seed(seed)
    args = (*parts(llm), x)
    if sync_free:
        out = sf.sync_free_ce(*args, labels, labels, BS, MASK, n_ordinary=BOS, **kw)
    else:
        out = fast.fast_block_diffusion_ce(*args, labels, BS, MASK, n_ordinary=BOS, **kw)
    out[0].backward()
    grads = {k: p.grad.clone() for k, p in llm.named_parameters() if p.requires_grad}
    return out, grads, x.grad, torch.get_rng_state()


def assert_bitwise(a, b):
    (out_a, g_a, x_a, rng_a), (out_b, g_b, x_b, rng_b) = a, b
    assert all(same(u, v) for u, v in zip(out_a, out_b))  # CE, n, logits, targets
    assert all(same(g_a[k], g_b[k]) for k in g_a) and same(x_a, x_b)
    assert torch.equal(rng_a, rng_b)  # the same draws, in the same order


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", [*LAYOUTS, "holes"])
@pytest.mark.parametrize("share, attention", [(False, "dense"), (False, "flex"), (True, "dense"), (True, "flex")])
def test_step_is_fast_steps_bit_for_bit(seed, name, share, attention):
    """Real LoRA dropout (p 0.2) on: the draws are the same, so everything is."""
    llm = tiny_llm(seed, dropout=0.2)
    clean, labels = layout(LAYOUTS.get(name, TAILS), seed, holes=name == "holes")
    kw = {"share_clean": share, "quantum": 128, "attention": attention}
    assert_bitwise(run_step(llm, clean, labels, 30 + seed, False, **kw),
                   run_step(llm, clean, labels, 30 + seed, True, **kw))


@pytest.mark.parametrize("attention", ["dense", "flex"])
def test_step_with_t2t_keeps_the_boolean_selection(attention):
    llm = tiny_llm(4, dropout=0.2)
    clean, labels = layout(TAILS, 4)
    kw = {"edit_rate": 0.3, "quantum": 128, "attention": attention}
    assert_bitwise(run_step(llm, clean, labels, 5, False, **kw), run_step(llm, clean, labels, 5, True, **kw))


@pytest.mark.parametrize("options", [{}, {"share_clean": True}, {"pack_quantum": 128}, {"align_sentence_block": True},
                                     {"edit_rate": 0.2}, {"edit_rate": 0.2, "align_sentence_block": True}])
@pytest.mark.parametrize("rows", [ROWS, ROWS_T15])
def test_forward_is_fast_steps_bit_for_bit(rows, options):
    """BCI_DiffusionLLM.forward with sync_free against without: loss, examples, logits, targets, every gradient
    (LoRA and the neural tokens, which stand in for the encoder's output) and the generator state, with real LoRA
    dropout on."""
    llm = tiny_llm(6, dropout=0.2)
    outs = []
    for sync_free in (False, True):
        model = bci(llm, fast_step=True, sync_free=sync_free, **options)
        b = batch(rows, 7)
        b["spikes"].requires_grad_()
        llm.zero_grad()
        torch.manual_seed(8)
        out = model(**b)
        out.loss.backward()
        outs.append((out, {k: p.grad.clone() for k, p in llm.named_parameters() if p.requires_grad},
                     b["spikes"].grad, torch.get_rng_state()))
    (a, ga, sa, ra), (f, gf, sf_, rf) = outs
    for k in ("loss", "n_examples", "preds", "targets"):
        assert same(getattr(a, k), getattr(f, k)), k
    assert all(same(ga[k], gf[k]) for k in ga) and same(sa, sf_) and torch.equal(ra, rf)


# ------------------------------------------------------------------ options and registry


def test_sync_free_options_are_checked():
    llm = tiny_llm()
    bci(llm, fast_step=True, sync_free=True)
    for bad in ({"sync_free": True}, {"fast_step": True, "sync_free": 1}):
        with pytest.raises(ValueError):
            bci(llm, **bad)


def test_registry_keeps_the_running_entry():
    """aero-1-audio-diffusion-fast (the running job) is unchanged; the new entry is it plus sync_free."""
    running = REGISTRY["aero-1-audio-diffusion-fast"]
    assert running["sampler"] == {"block_size": 32, "small_block_size": 8, "threshold": 0.9, "fast_step": True}
    new = REGISTRY["aero-1-audio-diffusion-fast-syncfree"]
    assert {k: v for k, v in new.items() if k != "sampler"} == {k: v for k, v in running.items() if k != "sampler"}
    assert new["sampler"] == running["sampler"] | {"sync_free": True}
    for name, entry in REGISTRY.items():
        if name != "aero-1-audio-diffusion-fast-syncfree":
            assert "sync_free" not in entry.get("sampler", {}), name


def test_profile_script_busy_fraction():
    """scripts/profile_fast_step.py's GPU-busy measure: the union of device intervals (overlaps counted once)."""
    import importlib.util
    from types import SimpleNamespace

    from torch.autograd import DeviceType

    spec = importlib.util.spec_from_file_location("profile_fast_step", ROOT / "scripts" / "profile_fast_step.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ev = lambda s, e, dev=DeviceType.CUDA: SimpleNamespace(time_range=SimpleNamespace(start=s, end=e), device_type=dev)
    prof = SimpleNamespace(events=lambda: [ev(0, 100), ev(50, 150), ev(300, 400), ev(120, 130), ev(0, 1000, DeviceType.CPU)])
    assert mod.busy_fraction(prof, 1e-3) == pytest.approx(0.25)  # (150 + 100) us of 1000 us
