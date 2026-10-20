"""Stage 2 helpers: verified initialization, nested label fractions, null-control shuffling, seen/unseen WER."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bit_experiments import stage2  # noqa: E402


def test_derangement_has_no_fixed_point():
    for n in (2, 3, 10, 713):
        p = stage2.derangement(n, np.random.default_rng(n))
        assert sorted(p) == list(range(n)) and not (p == np.arange(n)).any()


def test_label_fractions_are_nested_for_one_seed():
    ds = {"train": list(range(100)), "val": [0]}
    small, big = stage2.subsample_train(ds, 0.1, 7)["train"], stage2.subsample_train(ds, 0.5, 7)["train"]
    assert len(small) == 10 and len(big) == 50 and set(small) <= set(big)
    assert stage2.subsample_train(ds, 1.0, 7) is ds


def test_score_splits_seen_and_unseen():
    refs, hyps = ["a b c d", "x y"], ["a b c d", "x z"]
    r = stage2.score(refs, hyps, stage2.seen_mask(refs, ["A b, c d!"]))
    assert r["wer"] == pytest.approx(1 / 6) and r["wer_seen"] == 0 and r["wer_unseen"] == 0.5
    assert r["n_seen"] == 1 and r["exact"] == 0.5


class _Enc(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedder = nn.Module()
        self.embedder.read_in = nn.Linear(width, 4)
        self.core = nn.Linear(4, 4)


class _Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.neural_encoder = nn.Module()
        self.neural_encoder.encoder = _Enc(width)
        self.projector = nn.Linear(4, 3)


def test_load_init_core_only_when_read_in_width_differs(tmp_path):
    src = _Model(width=6)  # e.g. TX-only checkpoint; the target model has TX+SBP (width 12)
    torch.save(src.neural_encoder.encoder.state_dict(), tmp_path / "encoder.bin")
    torch.save(src.projector.state_dict(), tmp_path / "projector.bin")
    dst = _Model(width=12)
    rep = stage2.load_init(dst, tmp_path)
    assert torch.equal(dst.neural_encoder.encoder.core.weight, src.neural_encoder.encoder.core.weight)
    assert not rep["read_in_loaded"] and rep["encoder"]["skipped_shape"] == ["embedder.read_in.weight"]
    assert torch.equal(dst.projector.weight, src.projector.weight) and rep["projector"]["loaded"] == 2
    (tmp_path / "projector.bin").unlink()
    torch.save({"embedder.read_in.weight": torch.zeros(4, 12)}, tmp_path / "encoder.bin")
    with pytest.raises(RuntimeError, match="core tensors not loaded"):
        stage2.load_init(_Model(width=12), tmp_path)


class _WithLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = nn.Module()
        self.llm.base = nn.Linear(3, 3)
        self.llm.lora_A = nn.Linear(3, 1, bias=False)
        self.projector = nn.Linear(3, 3)
        self.llm.base.requires_grad_(False)


def test_slim_checkpoint_round_trip(tmp_path):
    from safetensors.torch import save_file

    m = _WithLLM()
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(tmp_path / "model.safetensors"))
    (tmp_path / "optimizer.bin").write_bytes(b"x")
    rep = stage2.slim_checkpoint(m, tmp_path)
    assert rep["removed"] == ["model.safetensors", "optimizer.bin"] and rep["slim_tensors"] == 3
    fresh = _WithLLM()
    fresh.llm.base.load_state_dict(m.llm.base.state_dict())  # the frozen base comes from its own snapshot
    assert stage2.load_model_state(fresh, tmp_path)["missing_base_llm"] == 2
    assert torch.equal(fresh.llm.lora_A.weight, m.llm.lora_A.weight) and torch.equal(fresh.projector.bias, m.projector.bias)
    m.llm.base.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable parameters outside"):
        stage2.slim_state(m)


# ---------------------------------------------------------------- padding (stage2 difference 4)

def _bit_merge(text_embeds, spikes_embeds, spikes_mask, attention_mask, input_split, targets=None):
    """BIT's BCI_AudioLLM.merge_embeds (external/bit/models/bci.py), copied for the comparison."""
    e = torch.stack([torch.cat((t[:d], s, t[d:])) for t, s, d in zip(text_embeds, spikes_embeds, input_split)])
    a = torch.stack([torch.cat((x[:d], s, x[d:])) for x, s, d in zip(attention_mask, spikes_mask, input_split)])
    if targets is not None:
        targets = torch.stack([torch.cat((t[:d], torch.ones_like(s) * (-100), t[d:]))
                               for t, s, d in zip(targets, spikes_mask, input_split)])
    return e, a, targets


def _merge_inputs(n_valid, n_text, split=2, width=3, max_s=None, max_t=None):
    """A right-padded batch as BIT's collate + encoder give it: trial i has n_valid[i] neural tokens and n_text[i]
    text tokens (the first `split` are prompt_a, the next one prompt_b, the rest targets)."""
    g = torch.Generator().manual_seed(0)
    B, S, T = len(n_valid), max_s or max(n_valid), max_t or max(n_text)
    spikes = torch.randn(B, S, width, generator=g)
    smask = torch.zeros(B, S, dtype=torch.long)
    text = torch.randn(B, T, width, generator=g)
    amask = torch.zeros(B, T, dtype=torch.long)
    targets = torch.full((B, T), -100)
    for i, (n, m) in enumerate(zip(n_valid, n_text)):
        smask[i, :n] = 1
        amask[i, :m] = 1
        targets[i, split + 1:m] = torch.arange(split + 1, m)
    return text, spikes, smask, amask, torch.full((B,), split), targets


def test_packed_merge_equals_bit_without_padding_and_lone_trial_with():
    one = _merge_inputs([5], [7])
    for x, y in zip(stage2.packed_merge(None, *one), _bit_merge(*one)):
        assert torch.equal(x, y)
    for x, y in zip(stage2.packed_merge(None, *one[:5]), _bit_merge(*one[:5])):  # generation: no targets
        assert (x is None and y is None) or torch.equal(x, y)
    batch = _merge_inputs([2, 5, 4], [6, 4, 7])
    e, a, t = stage2.packed_merge(None, *batch)
    for i, (n, m) in enumerate(zip([2, 5, 4], [6, 4, 7])):
        alone = [x[i:i + 1] for x in batch]
        alone[1], alone[2] = alone[1][:, :n], alone[2][:, :n]  # no neural padding
        alone[0], alone[3], alone[5] = alone[0][:, :m], alone[3][:, :m], alone[5][:, :m]  # no text padding
        ea, aa, ta = _bit_merge(*alone)
        k = 2 + n + (m - 2)
        assert torch.equal(e[i, :k], ea[0]) and torch.equal(a[i, :k], aa[0]) and torch.equal(t[i, :k], ta[0])
        assert not a[i, k:].any() and (t[i, k:] == -100).all()  # all padding after the trial's real tokens


def test_pad_masked_smoothing_gives_a_trial_its_lone_output():
    class Smooth(nn.Module):  # BIT's SmoothAndNoise without noise: Gaussian conv1d over time, zero edges
        def __init__(self):
            super().__init__()
            self.smooth, self.noise = True, False
            k = torch.exp(-0.5 * (torch.arange(-6, 7) / 2.0) ** 2)
            self.register_buffer("kernel", k / k.sum(), persistent=False)

        def forward(self, x):
            n = x.shape[-1]
            w = self.kernel[None, None].expand(n, 1, -1)
            return torch.nn.functional.conv1d(x.transpose(1, 2), w, padding="same", groups=n).transpose(1, 2)

    g = torch.Generator().manual_seed(1)
    long, short = torch.randn(1, 40, 4, generator=g), torch.randn(1, 15, 4, generator=g)
    batch = torch.cat([long, torch.cat([short, torch.full((1, 25, 4), stage2.PAD_VALUE)], 1)])
    bit, fixed = Smooth(), stage2.PadMaskedSmoothing(Smooth())
    assert (bit(batch)[1, 14] - bit(short)[0, 14]).abs().max() > 10  # BIT: -100 leaks into the last bins
    out = fixed(batch)
    assert torch.allclose(out[1, :15], bit(short)[0], atol=1e-6) and torch.allclose(out[0], bit(long)[0], atol=1e-6)
    assert (out[1, 15:] == stage2.PAD_VALUE).all()


def test_fix_padding_wraps_once_and_packs():
    model = nn.Module()
    model.neural_encoder = nn.Module()
    model.neural_encoder.encoder = nn.Module()
    inner = nn.Module()
    inner.smooth, inner.noise = 2.0, False
    inner.register_buffer("kernel", torch.ones(13), persistent=False)
    model.neural_encoder.encoder.smooth_and_noise = inner
    for _ in range(2):
        rep = stage2.fix_padding(model)
    sn = model.neural_encoder.encoder.smooth_and_noise
    assert isinstance(sn, stage2.PadMaskedSmoothing) and sn.inner is inner
    assert rep == {"smoothing": "pad-masked", "smoothing_taps": 13, "noise": False, "merge": "packed"}
    assert model.merge_embeds.__func__ is stage2.packed_merge and not model.state_dict()
