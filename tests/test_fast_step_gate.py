"""scripts/fast_step_gate.py (the launch-time fast-step gate), on the CPU with a tiny model. Needs BIT's environment
(transformers 4.48.2, peft 0.11.1); skipped without peft.

    <bit env>/python -m pytest tests/test_fast_step_gate.py

- decide(): the rc contract. 0 only if the old step's bug showed and the fixed step passed every check, else 3.
- The old module is the 10-23 host file byte for byte, and it computes the fixed module's maths (same loss, logits
  and gradients). Only the compile mode differs: dynamic by default against static.
- run_gate end to end: the phases with fake data and kernels. A broken old kernel with a correct fixed one gives
  rc 0; correct kernels give rc 3 (not reproduced); a broken fixed kernel gives rc 3.
"""

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external" / "bit"), str(ROOT / "tests")]
pytest.importorskip("peft")

from bit_experiments import fast_block_diffusion as fast  # noqa: E402
from test_fast_block_diffusion import BS, BOS, MASK, TAILS, batch, bci, layout, parts, run, tiny_llm  # noqa: E402

spec = importlib.util.spec_from_file_location("fast_step_gate", ROOT / "scripts" / "fast_step_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


# ------------------------------------------------------------------ the decision


def report(old_kernel_ok=True, fixed_kernel_ok=True, old_worse=False, fixed_worse=False, sanity=True, counts=True):
    return {"sanity_ok": sanity, "counts_identical": counts,
            "kernels": [{"T": 512, "old": {"ok": True}, "fixed": {"ok": True}},
                        {"T": 1024, "old": {"ok": old_kernel_ok}, "fixed": {"ok": fixed_kernel_ok}}],
            "train_verdicts": {"old": {"worse": old_worse}, "fixed": {"worse": fixed_worse}}}


@pytest.mark.parametrize("kw, rc", [
    ({"old_kernel_ok": False}, 0),                      # reproduced by the kernel, fixed fine
    ({"old_worse": True}, 0),                           # reproduced by training, fixed fine
    ({"old_kernel_ok": False, "old_worse": True}, 0),
    ({}, 3),                                            # the bug did not show: cannot vouch
    ({"old_kernel_ok": False, "fixed_kernel_ok": False}, 3),
    ({"old_kernel_ok": False, "fixed_worse": True}, 3),
    ({"old_kernel_ok": False, "sanity": False}, 3),
    ({"old_kernel_ok": False, "counts": False}, 3),
])
def test_decide(kw, rc):
    r = report(**kw)
    got, why = gate.decide(r)
    assert got == rc and (rc == 0) == (not why)
    assert r["verdict"]["reproduced"] == (not kw.get("old_kernel_ok", True) or kw.get("old_worse", False))


def test_worse():
    plain = [1.0 + 0.01 * np.sin(i) for i in range(120)]
    probes = [2.0, 1.5, 1.2, 1.1]
    assert not gate.worse(plain, plain, probes, probes)["worse"]
    assert gate.worse([x + 0.1 for x in plain], plain, probes, probes)["worse"]          # train loss consistently higher
    assert gate.worse(plain, plain, [p + 0.2 for p in probes], probes)["worse"]          # probes higher
    assert not gate.worse(plain, plain, probes[:-1] + [1.22], probes)["worse"]           # one noisy last probe (+0.12)
    noisy = [x + (0.3 if i % 2 else -0.3) for i, x in enumerate(plain)]
    assert not gate.worse(noisy, plain, probes, probes)["worse"]                         # noise, no shift


# ------------------------------------------------------------------ the old module


def test_old_module_is_the_host_file_and_the_same_maths():
    assert hashlib.md5(gate.OLD_PATH.read_bytes()).hexdigest() == "79bbfef978d42d265b30bacf6154eafc"
    old = gate.load_old_module()
    assert old is not fast and not hasattr(old, "PACKED_FLEX_DYNAMIC") and fast.PACKED_FLEX_DYNAMIC is False
    assert old.PACKED_FLEX_KERNEL_OPTIONS == fast.PACKED_FLEX_KERNEL_OPTIONS
    llm = tiny_llm(3)
    clean, labels = layout(TAILS, 3)
    for attention in ("dense", "flex"):
        outs = []
        for mod in (old, fast):
            llm.zero_grad()
            x = clean.clone().requires_grad_()
            torch.manual_seed(5)
            ce, n, logits, lab = mod.fast_block_diffusion_ce(*parts(llm), x, labels, BS, MASK, 0.0, BOS,
                                                             quantum=128, attention=attention)
            ce.backward()
            outs.append((ce, n, logits, lab, x.grad, [p.grad.clone() for p in llm.parameters() if p.requires_grad]))
        (a, b) = outs
        assert all(torch.equal(u, v) for u, v in zip(a[:5], b[:5])) and all(torch.equal(u, v) for u, v in zip(a[5], b[5]))


# ------------------------------------------------------------------ run_gate end to end (fakes)


def broken_flex(q, k, v, mask=None):
    """A kernel that ignores the block mask (attention across rows): the kind of error the gate must catch."""
    return fast.flex_attention(q, k, v, enable_gqa=True)


def correct_flex(q, k, v, mask=None):
    return fast.flex_attention(q, k, v, block_mask=mask, enable_gqa=True)


class Sampler:
    def __init__(self, ds, batch_size, shuffle=True, drop_last=False, seed=None):
        self.ds, self.bs, self.rng = ds, batch_size, np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.ds))
        return iter([list(idx[i:i + self.bs]) for i in range(0, len(idx), self.bs)])

    def __len__(self):
        return -(-len(self.ds) // self.bs)


@pytest.mark.parametrize("old_kernel, fixed_kernel, rc", [
    (broken_flex, correct_flex, 0),
    (correct_flex, correct_flex, 3),
    (broken_flex, broken_flex, 3),
])
def test_run_gate_end_to_end(monkeypatch, old_kernel, fixed_kernel, rc):
    import utils.datasets

    old = gate.load_old_module()
    monkeypatch.setattr(gate, "FAST_ATTENTION", "flex")  # on CUDA the default (None) is flex already
    for mod, kern in ((old, old_kernel), (fast, fixed_kernel)):  # route both through their (here eager) kernels
        monkeypatch.setattr(mod, "_compiled_flex", lambda device: True)
        monkeypatch.setattr(mod, "packed_flex_attention", kern)
    rng = np.random.default_rng(0)
    train = [(int(rng.integers(20, 130)), int(rng.integers(2, 20))) for _ in range(48)]
    val = [(int(rng.integers(20, 130)), int(rng.integers(2, 20))) for _ in range(8)]
    monkeypatch.setattr(utils.datasets, "GroupBatchSampler", Sampler)
    monkeypatch.setattr(utils.datasets, "pad_collate_fn",
                        lambda items, **kw: (batch(list(items), seed=sum(n for n, _ in items)), None))
    model = bci(tiny_llm(4, dropout=0.2))
    cfg = types.SimpleNamespace(
        seed=42, training=types.SimpleNamespace(train_batch_size=4, drop_last_train_dataloader=False),
        method=types.SimpleNamespace(dataloader_kwargs={}),
        optimizer=types.SimpleNamespace(lr=5e-4, wd=1e-5, eps=1e-8, warmup_pct=0.0, div_factor=25))
    trainer = types.SimpleNamespace(accelerator=types.SimpleNamespace(unwrap_model=lambda m: m), model=model,
                                    config=cfg, model_inputs=[], train_dataset=train, val_dataset=val)
    args = types.SimpleNamespace(llm="tiny", init="none", arm="B", seed=42, shapes=3, steps=8, probe_every=4,
                                 sched_epochs=150)
    with pytest.raises(gate.GateDone) as done:
        gate.run_gate(trainer, args, old, fast)
    r = done.value.report
    got, why = gate.decide(r)
    assert got == rc, why
    assert r["sanity_ok"] and r["identical_start"] and r["counts_identical"]
    assert len(r["kernels"]) >= 2
