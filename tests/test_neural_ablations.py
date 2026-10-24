"""Retraining ablations (docs/RETRAINING_ABLATIONS.md), CPU only. The alignment, label-cache and LEACE tests need
only torch and numpy. The model tests need BIT's environment (transformers 4.48.2, peft 0.11.1) and are skipped
without peft.

    <bit env>/python -m pytest tests/test_neural_ablations.py

- CTC forced alignment: the Viterbi path is the best of all valid paths (brute force) and collapses to the labels;
  segment labels follow the midpoint / margin rule; BIT's PER collapse differs from standard CTC's as documented.
- LEACE:
  - zero cross-covariance with the concept and a singular feature covariance handled;
  - an idempotent eraser;
  - no larger a change than the orthogonal-projection eraser;
  - held-out linear probe at the majority class, while an MLP still finds a planted non-linear code;
  - streaming moments equal batch moments.
- The phoneme read-out: BIT's CTC read-out from decoder.bin, floor, standardization, vocabulary check.
- Stage 2 (a tiny Qwen2 in BCI_TextLLM, and in BCI_DiffusionLLM), each kind end to end, with both projector shapes:
  BIT's released Linear and the host's MLP (configs/bit_baseline.yaml inter_size 1024):
  - the artifacts from scripts/prepare_ablations.py's own functions on synthetic trials;
  - apply after load_init: frozen encoder (no gradient, eval mode even in train mode), the projector's first Linear
    re-sized to 41 inputs (an MLP projector keeps its ReLU and second Linear as built), global RNG untouched;
  - the LLM's neural tokens are exactly the bottleneck / erased features, batch = lone trial;
  - slim checkpoint round trip, evaluated with the ablation re-applied (--eval-only path), verify();
  - the hash checks refuse another encoder or a wrong --init.
- train_stage2.py: every neural_ablations call sits under `if args.ablation` (default off).
"""

import ast
import importlib.util
import itertools
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))

from bit_experiments import neural_ablations as na  # noqa: E402
from bit_experiments import phoneme_frames as pf  # noqa: E402
from bit_experiments import probes  # noqa: E402


# ------------------------------------------------------------------ CTC forced alignment

def _collapse(path, blank=0):
    out, last = [], None
    for p in path:
        if p != last and p != blank:
            out.append(p)
        last = p
    return out


@pytest.mark.parametrize("labels,t", [([1, 2], 5), ([1, 1], 4), ([2, 1, 2], 6), ([3], 3)])
def test_viterbi_is_the_best_valid_path(labels, t):
    rng = np.random.default_rng(len(labels) * 10 + t)
    v = 4
    logp = np.log(rng.dirichlet(np.ones(v), size=t))
    states = pf.ctc_viterbi(logp, labels)
    ext = np.zeros(2 * len(labels) + 1, int)
    ext[1::2] = labels
    path = ext[states]
    assert _collapse(path.tolist()) == labels
    best = max(logp[np.arange(t), list(p)].sum() for p in itertools.product(range(v), repeat=t)
               if _collapse(list(p)) == labels)
    assert pf.path_score(logp, states, labels) == pytest.approx(best)


def test_viterbi_refuses_a_trial_too_short_for_its_labels():
    logp = np.log(np.full((4, 3), 1 / 3))
    assert pf.ctc_viterbi(logp, [1, 1, 1]) is None  # needs 5 frames (a blank between repeats)
    lab, tok, ok = pf.align(logp, [1, 1, 1])
    assert not ok and (lab == -1).all() and (tok == -1).all()


def test_segment_labels_split_blank_gaps_at_the_midpoint_with_margins():
    labels = [7, 9, 40]
    # frames: b b b 7 7 b b b b 9 b b b 40 b b b
    ext_state = {"b0": 0, 7: 1, "b1": 2, 9: 3, "b2": 4, 40: 5, "b3": 6}
    seq = ["b0"] * 3 + [7, 7] + ["b1"] * 4 + [9] + ["b2"] * 3 + [40] + ["b3"] * 3
    states = np.array([ext_state[s] for s in seq])
    lab, tok = pf.segment_labels(states, labels, margin=2)
    assert lab.tolist() == [-1, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 40, 40, 40, 40, -1]
    assert tok.tolist() == [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, -1]
    c = pf.concept(lab, tok, np.array(labels), (-1, 0, 1))
    assert c[3].tolist() == [-1, 6, 8] and c[8].tolist() == [6, 8, 39] and c[14].tolist() == [8, 39, -1]
    assert (c[0] == -1).all()


def test_greedy_ctc_bit_collapse_merges_repeats_across_blanks():
    logp = np.log(np.eye(4)[[1, 0, 1, 1, 2, 0, 0, 2]] * 0.97 + 0.01)
    assert pf.greedy_ctc(logp) == [1, 1, 2, 2]
    assert pf.greedy_ctc(logp, bit=True) == [1, 2]  # BIT's format_ctc (utils/eval_llm.py)
    assert pf.per([[1, 2]], [[1, 1, 2, 2]]) == 0.5


def test_label_cache_round_trip_and_tamper_check(tmp_path):
    per_split = {s: {"labels": [np.array([-1, 3, 3]), np.array([5])], "tokens": [np.array([-1, 0, 0]), np.array([0])],
                     "phonemes": [np.array([3, 40]), np.array([5, 40])], "sentences": ["a b", "c"]}
                 for s in ("train", "val")}
    path = pf.save_labels(tmp_path, per_split, {"margin": 2})
    back, meta = pf.load_labels(tmp_path)
    assert meta["margin"] == 2 and back["val"]["sentences"] == ["a b", "c"]
    assert [x.tolist() for x in back["train"]["labels"]] == [[-1, 3, 3], [5]]
    z = dict(np.load(path, allow_pickle=True))
    z["train/labels"] = z["train/labels"].copy()
    z["train/labels"][1] = 4
    np.savez(path, **z)
    with pytest.raises(ValueError, match="hash mismatch"):
        pf.load_labels(path)


# ------------------------------------------------------------------ LEACE

def _planted(n=6000, d=12, k=5, seed=0):
    """Features with class-dependent means, a non-linear (XOR) code of a binary label, and a singular covariance
    (every x lies in a hyperplane, as after LayerNorm)."""
    g = np.random.default_rng(seed)
    y = g.integers(0, k, n)
    means = g.normal(0, 1.5, (k, d))
    x = g.normal(0, 1, (n, d)) + means[y]
    xor = g.integers(0, 2, n)
    x[:, 0] = g.choice([-1, 1], n) * 2
    x[:, 1] = x[:, 0] * np.where(xor, 1, -1)
    x -= x.mean(1, keepdims=True)  # sum-zero hyperplane
    return x, y, xor


def test_leace_erases_linear_information_minimally():
    x, y, _ = _planted()
    tr, te = slice(0, 4000), slice(4000, None)
    acc = na.CovAccumulator(x.shape[1], 5)
    for i in range(0, 4000, 700):  # streaming = batch moments
        acc.update(torch.from_numpy(x[i:min(i + 700, 4000)]), na.one_hot_concept(torch.from_numpy(y[i:min(i + 700, 4000)]), 5))
    mean, cxx, cxz = acc.moments()
    xt = torch.from_numpy(x[tr])
    assert torch.allclose(cxx, torch.cov(xt.T, correction=0), atol=1e-9)
    fit = na.leace(mean, cxx, cxz)
    assert fit["cov_rank"] == x.shape[1] - 1 and fit["rank"] == 4  # one-hot of 5 classes: rank 4
    A, b = fit["A"], fit["b"]
    erased = (xt @ A.T + b).numpy()
    z = na.one_hot_concept(torch.from_numpy(y[tr]), 5).numpy()
    cov_after = (erased - erased.mean(0)).T @ (z - z.mean(0)) / len(z)
    assert np.abs(cov_after).max() < 1e-10
    assert torch.allclose(A @ A, A, atol=1e-10)  # an (oblique) projection
    # least change: no larger than the orthogonal projection out of span(Cov_xz)
    u = torch.linalg.svd(cxz, full_matrices=False)[0][:, :4]
    ortho = (xt - mean) - (xt - mean) @ u @ u.T + mean
    leace_change = ((xt @ A.T + b) - xt).pow(2).sum(1).mean()
    assert leace_change <= (ortho - xt).pow(2).sum(1).mean() + 1e-9
    # held out: the linear probe falls to the majority class; the MLP still reads the non-linear (XOR) code
    xe = x @ A.numpy().T + b.numpy()
    before = probes.score_probe(probes.fit_linear_probe(x[tr], y[tr], x[te], y[te], 5), x[te], y[te])
    after = probes.score_probe(probes.fit_linear_probe(xe[tr], y[tr], xe[te], y[te], 5), xe[te], y[te])
    assert before["accuracy"] > 0.6 and after["accuracy"] < after["majority"] + 0.04 and after["info_bits"] < 0.02


def test_one_hot_concept_with_offsets_and_missing():
    z = na.one_hot_concept(torch.tensor([[0, -1], [2, 1]]), 3)
    assert z.tolist() == [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0]]


# ------------------------------------------------------------------ phoneme read-out

def test_phoneme_readout_from_checkpoint(tmp_path):
    w, b = torch.randn(41, 8), torch.randn(41)
    torch.save({"read_out.readout_dict.0.weight": w, "read_out.readout_dict.0.bias": b,
                "read_out.readout_dict.1.weight": torch.zeros(41, 8), "read_out.readout_dict.1.bias": torch.zeros(41)},
               tmp_path / "decoder.bin")
    h = torch.randn(2, 5, 8) * 30
    mean, sd = torch.randn(41), torch.rand(41) + 0.5
    r = na.PhonemeReadout.from_checkpoint(tmp_path, floor=1e-8, mean=mean, sd=sd)
    lp = torch.log_softmax(h @ w.T + b, -1)
    assert torch.allclose(r.log_posteriors(h), lp, atol=1e-5)
    assert lp.min() < np.log(1e-8)  # the floor binds somewhere
    assert torch.allclose(r(h, None), (lp.clamp_min(np.log(1e-8)) - mean) / sd, atol=1e-4)
    assert r.out_features == 41
    assert torch.equal(na.PhonemeReadout.from_checkpoint(tmp_path, feature="one_hot").raw(h).argmax(-1), lp.argmax(-1))
    torch.save({"read_out.readout_dict.0.weight": torch.zeros(40, 8), "read_out.readout_dict.0.bias": torch.zeros(40)},
               tmp_path / "decoder.bin")
    with pytest.raises(ValueError, match="41"):
        na.PhonemeReadout.from_checkpoint(tmp_path)


def test_registry_entries():
    reg = yaml.safe_load(na.REGISTRY.read_text())
    assert {e["kind"] for e in reg.values()} <= set(na.KINDS)
    assert reg["phoneme-bottleneck"]["phoneme_model"] == reg["frozen-A"]["encoder"]
    assert reg["frozen-B"]["encoder"] == reg["phoneme-erasure"]["encoder"]
    assert reg["phoneme-bottleneck"]["feature"] in na.FEATURES and reg["phoneme-erasure"]["offsets"] == [0]


def test_train_stage2_calls_ablations_only_under_the_flag():
    tree = ast.parse((ROOT / "scripts" / "train_stage2.py").read_text())
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
             and n.value.id == "neural_ablations"]
    assert len(calls) >= 2
    for node in calls:
        p = node
        while p in parents and not (isinstance(p, ast.If) and "args.ablation" in ast.unparse(p.test)):
            p = parents[p]
        assert isinstance(p, ast.If), f"neural_ablations.{node.attr} outside `if args.ablation`"


# ------------------------------------------------------------------ Stage 2 model (BIT's environment)

needs_bit = pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="needs BIT's environment (peft)")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bit_trainer(monkeypatch):
    try:
        from models import trainer
    except ImportError:
        import models

        trainer = types.ModuleType("models.trainer")
        trainer.NAME2MODEL = {}
        monkeypatch.setitem(sys.modules, "models.trainer", trainer)
        monkeypatch.setattr(models, "trainer", trainer, raising=False)
    return trainer


V, BOS, EOS, D, TINY_MASK = 96, 80, 81, 64, 95


def tiny_qwen2(path, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                       rope_theta=1e6, bos_token_id=BOS, eos_token_id=EOS, tie_word_embeddings=True,
                                       use_sliding_window=False))
    with torch.no_grad():
        for n, p in llm.named_parameters():
            p.uniform_(0.5, 1.5) if "norm" in n else p.normal_(0, 0.1)
    llm.save_pretrained(path)


@pytest.fixture
def world(tmp_path, monkeypatch, bit_trainer):
    """A tiny repository root: LLM snapshots, an ablation registry, and a model builder (train_stage2.py's steps)."""
    from bit_experiments import stage1, stage2, text_llm

    reg = yaml.safe_load(text_llm.REGISTRY.read_text())
    tiny_qwen2(tmp_path / reg["qwen2.5-1.5b-instruct"]["local_dir"])
    monkeypatch.setattr(text_llm, "ROOT", tmp_path)
    monkeypatch.setattr(na, "ROOT", tmp_path)
    real_entry = text_llm.llm_entry
    monkeypatch.setattr(text_llm, "llm_entry", lambda name: real_entry(name) | (
        {"mask_id": TINY_MASK} if "diffusion" in name else {}))
    registry = tmp_path / "configs" / "ablations.yaml"
    registry.parent.mkdir(parents=True)
    monkeypatch.setattr(na, "REGISTRY", registry)

    def build(seed=0, llm="qwen2.5-1.5b-instruct", inter_size=None):
        """inter_size None: the repo's released BIT projector (one Linear); 1024: the host's patched one
        (configs/bit_baseline.yaml), an MLP Sequential(Linear(384, 1024), ReLU, Linear(1024, hidden))."""
        config = stage1.bit_sentence_config()
        text_llm.use_llm(config, llm)
        config["model"]["projector"]["embed_size"] = D // 2
        config["model"]["projector"]["inter_size"] = inter_size
        with stage1.in_bit_dir():
            torch.manual_seed(seed)
            model = bit_trainer.NAME2MODEL[config.model.model_class](
                config.model, **config.method.model_kwargs, **config.method.dataset_kwargs, features="all")
        stage2.fix_padding(model)
        return model, config

    def save_ckpt(model, d, readout=None):
        d.mkdir(parents=True, exist_ok=True)
        model.neural_encoder.save_checkpoint(str(d))
        if readout is not None:
            torch.save({"read_out.readout_dict.0.weight": readout[0], "read_out.readout_dict.0.bias": readout[1]},
                       d / "decoder.bin")
        return d

    return types.SimpleNamespace(root=tmp_path, build=build, save_ckpt=save_ckpt, registry=registry,
                                 write_registry=lambda e: registry.write_text(yaml.safe_dump(e)))


def _records(n, seed, lengths=(137, 123, 150, 131, 144)):
    g = np.random.default_rng(seed)
    words = ["the cat sat", "a dog ran far", "it is good", "we like this", "no way"]
    return [{"spikes": g.normal(0, 1, (lengths[i % len(lengths)], 512)).astype(np.float32),
             "sentence": words[i % len(words)]} for i in range(n)]


def _seqs(sentences):
    return [[(sum(map(ord, w)) % 39) + 1 for w in s.split()] + [40] for s in sentences]


def _batch(n_bins=(40, 25), seed=0):
    """A right-padded batch of two trials, BIT's collate layout (prompt ids 3,4 | 5; sentence + EOS)."""
    g = torch.Generator().manual_seed(seed)
    b, t = len(n_bins), max(n_bins)
    spikes = torch.randn(b, t, 512, generator=g)
    mask = torch.zeros(b, t, dtype=torch.long)
    for i, n in enumerate(n_bins):
        mask[i, :n] = 1
        spikes[i, n:] = -100.0
    sentences = [[10, 11, 12], [20, 21]]
    length = 3 + max(map(len, sentences)) + 1
    ids, att, tgt = torch.full((b, length), 0), torch.zeros(b, length, dtype=torch.long), torch.full((b, length), -100)
    for i, s in enumerate(sentences):
        row = [3, 4, 5] + s + [EOS]
        ids[i, :len(row)], att[i, :len(row)] = torch.tensor(row), 1
        tgt[i, 3:len(row)] = torch.tensor(s + [EOS])
    return {"input_ids": ids, "attention_mask": att, "input_split": torch.full((b,), 2), "spikes": spikes,
            "spikes_mask": mask, "spikes_timestamp": torch.arange(t).expand(b, t), "spikes_lengths": torch.tensor(n_bins),
            "dataset_name": torch.zeros(b, dtype=torch.long), "targets": tgt}


def _neural_tokens(model, batch):
    sp, _, m, _ = model.prepare_embeds(batch["input_ids"], batch["spikes"], batch["spikes_mask"],
                                       batch["spikes_timestamp"], batch["dataset_name"])
    return sp, m


PROJECTORS = pytest.mark.parametrize("inter_size", [None, 1024], ids=["linear-projector", "mlp-projector"])


@needs_bit
@PROJECTORS
def test_bottleneck_end_to_end(world, inter_size):
    from bit_experiments import stage1, stage2

    prep = _script("prepare_ablations")
    arm_a, _ = world.build(seed=1)  # a "trained" phoneme model: different weights from the Stage 2 build
    readout = (torch.randn(41, 384) * 0.3, torch.randn(41))
    a_dir = world.save_ckpt(arm_a, world.root / "ckpt" / "armA", readout)
    entry = {"phoneme-bottleneck": {"kind": "phoneme_bottleneck", "phoneme_model": "ckpt/armA",
                                    "feature": "log_posterior", "floor": 1e-8, "stats": "abl/stats.pt"},
             "frozen-A": {"kind": "frozen", "encoder": "ckpt/armA"}}
    world.write_registry(entry)
    config = stage1.bit_sentence_config()
    ds = {"train": _records(6, 0), "val": _records(3, 1), "test": _records(3, 2)}
    stats = prep.make_bottleneck_stats(config, ds, na.entry("phoneme-bottleneck"), "cpu", batch_size=4)
    assert stats["n_frames"] == sum(pf.n_patches(len(r["spikes"])) for r in ds["train"])

    model, _ = world.build(seed=0, inter_size=inter_size)
    assert na.projector_in_features(model.projector) == 384
    mlp = isinstance(model.projector, torch.nn.Sequential)
    assert mlp == (inter_size is not None)
    kept = list(model.projector)[1:] if mlp else []
    kept_state = [{k: v.clone() for k, v in m.state_dict().items()} for m in kept]
    with pytest.raises(ValueError, match="expects --init"):
        na.apply(model, "phoneme-bottleneck", world.root / "elsewhere", 0)
    stage2.load_init(model, a_dir)
    torch.manual_seed(5)
    rng = torch.get_rng_state()
    rep = na.apply(model, "phoneme-bottleneck", a_dir, seed=0)
    assert torch.equal(torch.get_rng_state(), rng)  # new modules come from a forked RNG
    assert rep["projector_resized"]["in_features"] == 41 and na.projector_in_features(model.projector) == 41
    first = model.projector[0] if mlp else model.projector
    assert first.out_features == (inter_size or 64)  # the first Linear's outputs are unchanged
    if mlp:  # only the first Linear is new; the activation and the second Linear are the ones the model built
        assert all(a is b for a, b in zip(list(model.projector)[1:], kept)) and len(model.projector) == 3
        assert all(torch.equal(v, st[k]) for m, st in zip(kept, kept_state) for k, v in m.state_dict().items())
        assert rep["projector_resized"]["kept"] == ["ReLU", "Linear"]
    n_proj = sum(p.numel() for p in model.projector.parameters())
    assert n_proj == 41 * first.out_features + first.out_features + sum(p.numel() for m in kept for p in m.parameters())
    assert not any(p.requires_grad for p in model.neural_encoder.encoder.parameters())

    batch = _batch()
    model.train()
    tokens, mask = _neural_tokens(model, batch)
    again, _ = _neural_tokens(model, batch)
    assert torch.equal(tokens, again)  # the frozen part runs in eval mode (no dropout) even in train mode
    enc = model.neural_encoder.encoder.eval()
    h, *_ = enc(batch["spikes"], batch["spikes_mask"], batch["spikes_timestamp"], batch["dataset_name"],
                force_mask=False)
    lp = torch.log_softmax(h @ readout[0].T + readout[1], -1).clamp_min(np.log(1e-8))
    s = torch.load(world.root / "abl" / "stats.pt", weights_only=True)
    assert torch.allclose(tokens, model.projector((lp - s["mean"]) / s["sd"]), atol=1e-5)
    lone, _ = _neural_tokens(model, {k: v[1:] if torch.is_tensor(v) else v for k, v in batch.items()} |
                             {"spikes": batch["spikes"][1:, :25], "spikes_mask": batch["spikes_mask"][1:, :25],
                              "spikes_timestamp": batch["spikes_timestamp"][1:, :25]})
    n = int(mask[1].sum())
    assert n == lone.shape[1] and torch.allclose(tokens[1, :n], lone[0], atol=1e-5)  # batch = lone trial

    model.train()
    out = model(**batch)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(p.grad is not None for p in model.projector.parameters())
    assert all(p.grad is None for p in model.neural_encoder.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.llm.named_parameters() if "lora_B" in n)

    # slim checkpoint round trip, evaluated with the ablation re-applied (train_stage2.py --eval-only path)
    ckpt = world.root / "BEST"
    ckpt.mkdir()
    from safetensors.torch import save_file

    save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(model).items()}, str(ckpt / stage2.SLIM))
    assert not any("neural_ablation" in k for k in stage2.slim_state(model))
    fresh, _ = world.build(seed=3, inter_size=inter_size)
    na.apply(fresh, "phoneme-bottleneck", None, seed=3)
    stage2.load_model_state(fresh, ckpt)
    assert na.verify(fresh)["expect_encoder_matches"]
    model.eval(), fresh.eval()
    assert torch.allclose(_neural_tokens(fresh, batch)[0], _neural_tokens(model, batch)[0], atol=1e-6)
    assert na.verify(model)["encoder_at_start_matches"]

    # stats computed on another encoder are refused
    other, _ = world.build(seed=0, inter_size=inter_size)
    stage2.load_init(other, world.save_ckpt(world.build(seed=2)[0], world.root / "ckpt" / "armA2", readout))
    world.write_registry(entry | {"phoneme-bottleneck": entry["phoneme-bottleneck"] | {"phoneme_model": "ckpt/armA2"}})
    with pytest.raises(ValueError, match="different encoder"):
        na.apply(other, "phoneme-bottleneck", world.root / "ckpt" / "armA2", 0)

    # frozen-A: features unchanged, projector kept
    world.write_registry(entry)
    plain, _ = world.build(seed=0, inter_size=inter_size)
    stage2.load_init(plain, a_dir)
    ref = plain.projector(enc(batch["spikes"], batch["spikes_mask"], batch["spikes_timestamp"], batch["dataset_name"],
                              force_mask=False)[0])
    rep = na.apply(plain, "frozen-A", a_dir, seed=0)
    assert "projector_resized" not in rep and torch.allclose(_neural_tokens(plain, batch)[0], ref, atol=1e-5)


@needs_bit
@PROJECTORS
def test_erasure_end_to_end_and_eval_only(world, inter_size):
    from bit_experiments import stage1, stage2

    prep = _script("prepare_ablations")
    config = stage1.bit_sentence_config()
    arm_a, _ = world.build(seed=1)
    a_dir = world.save_ckpt(arm_a, world.root / "ckpt" / "armA", (torch.randn(41, 384), torch.randn(41)))
    b_trained, _ = world.build(seed=4, inter_size=inter_size)  # stands in for B_ssl_s42's trained Stage 2 model
    run_best = world.root / "checkpoints" / "stage2" / "B_fake" / "BEST"
    world.save_ckpt(b_trained, run_best)
    from safetensors.torch import save_file

    save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(b_trained).items()}, str(run_best / stage2.SLIM))
    monkey_root = prep.ROOT
    prep.ROOT = world.root
    try:
        info = prep.export_encoder(config, "B_fake", world.root / "abl" / "B_fake-encoder")
    finally:
        prep.ROOT = monkey_root
    world.write_registry({
        "frozen-B": {"kind": "frozen", "encoder": "abl/B_fake-encoder"},
        "phoneme-erasure": {"kind": "phoneme_erasure", "encoder": "abl/B_fake-encoder", "eraser": "abl/er/eraser.pt",
                            "offsets": [0]}})
    ds = {"train": _records(10, 0), "val": _records(5, 1), "test": _records(4, 2)}
    path, meta = prep.make_labels(config, ds, a_dir, world.root / "labels", "cpu", batch_size=4, sequences=_seqs)
    labels, lmeta = pf.load_labels(path)
    assert lmeta["report"]["train"]["labelled"] > 0.5 and lmeta["report"]["train"]["n_unaligned"] == 0
    check, _ = prep.make_eraser(config, ds, labels, lmeta, na.entry("phoneme-erasure"), "cpu", batch_size=4,
                                max_train_frames=10_000)
    assert check["max_abs_cross_cov"]["after"] < 1e-5 * max(1.0, check["max_abs_cross_cov"]["before"])
    eraser = torch.load(world.root / "abl" / "er" / "eraser.pt", weights_only=True)
    assert eraser["encoder_digest"] == info["encoder_digest"] and eraser["n_classes"] == 40

    model, _ = world.build(seed=0, inter_size=inter_size)
    stage2.load_init(model, world.root / "abl" / "B_fake-encoder")
    rep = na.apply(model, "phoneme-erasure", world.root / "abl" / "B_fake-encoder", seed=0)
    assert rep["eraser"]["rank"] == eraser["rank"] and "projector_resized" not in rep
    batch = _batch()
    model.eval()
    enc = model.neural_encoder.encoder
    h, *_ = enc(batch["spikes"], batch["spikes_mask"], batch["spikes_timestamp"], batch["dataset_name"],
                force_mask=False)
    assert torch.allclose(_neural_tokens(model, batch)[0], model.projector(h @ eraser["A"].T + eraser["b"]), atol=1e-5)

    # eval-only: B's own trained checkpoint with the eraser inserted, no retraining
    ev, _ = world.build(seed=9, inter_size=inter_size)
    na.apply(ev, "phoneme-erasure", None, seed=9)
    stage2.load_model_state(ev, run_best)
    assert na.verify(ev)["expect_encoder_matches"]
    # the wrong checkpoint is refused
    wrong, _ = world.build(seed=9, inter_size=inter_size)
    na.apply(wrong, "phoneme-erasure", None, seed=9)
    with pytest.raises(RuntimeError, match="digest"):
        na.verify(wrong)


@needs_bit
@PROJECTORS
def test_bottleneck_in_the_diffusion_model(world, inter_size):
    from bit_experiments import stage1, stage2, text_llm

    from bit_experiments.diffusion_llm import BCI_DiffusionLLM

    reg = yaml.safe_load(text_llm.REGISTRY.read_text())
    assert reg["qwen2.5-1.5b-instruct-diffusion"]["local_dir"] == reg["qwen2.5-1.5b-instruct"]["local_dir"]
    prep = _script("prepare_ablations")
    arm_a, _ = world.build(seed=1)
    a_dir = world.save_ckpt(arm_a, world.root / "ckpt" / "armA", (torch.randn(41, 384), torch.randn(41)))
    world.write_registry({"phoneme-bottleneck": {"kind": "phoneme_bottleneck", "phoneme_model": "ckpt/armA",
                                                 "feature": "log_posterior", "floor": 1e-8, "stats": "abl/stats.pt"}})
    prep.make_bottleneck_stats(stage1.bit_sentence_config(), {"train": _records(4, 0)}, na.entry("phoneme-bottleneck"),
                               "cpu", batch_size=4)
    model, _ = world.build(seed=0, llm="qwen2.5-1.5b-instruct-diffusion", inter_size=inter_size)
    assert isinstance(model, BCI_DiffusionLLM)
    stage2.load_init(model, a_dir)
    rep = na.apply(model, "phoneme-bottleneck", a_dir, seed=0)
    assert rep["projector_resized"]["in_features"] == 41 and na.projector_in_features(model.projector) == 41
    model.train()
    torch.manual_seed(0)
    out = model(**_batch())
    out.loss.backward()
    assert torch.isfinite(out.loss) and all(p.grad is not None for p in model.projector.parameters())
    model.eval()
    one = {k: v[:1] for k, v in _batch().items() if k != "targets"} | {
        "input_ids": torch.tensor([[3, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    ids = model.generate(**one, max_new_tokens=6, do_sample=False)
    assert ids.shape[0] == 1 and TINY_MASK not in ids.tolist()[0]


def _aero_snapshot(root):
    """A tiny snapshot in Aero-1-Audio's own layout at the aero-1-audio-diffusion entry's local_dir
    (tests/test_aero_diffusion.py's writer: model_type "aero", language_model.* tensors, an audio tower)."""
    from bit_experiments import text_llm

    spec = importlib.util.spec_from_file_location("_aero_tests", ROOT / "tests" / "test_aero_diffusion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reg = yaml.safe_load(text_llm.REGISTRY.read_text())
    path = root / reg["aero-1-audio-diffusion"]["local_dir"]
    mod.write_aero_snapshot(path)
    return path


@needs_bit
@PROJECTORS
def test_eraser_pair_with_aero_block_diffusion(world, inter_size):
    """The approved design (2026-10-24): B_ssl_s42's frozen encoder (trained with Aero AR) under the Aero
    block-diffusion decoder, with and without the eraser; and the decode-time checks on a diffusion run's own BEST."""
    from safetensors.torch import save_file

    from bit_experiments import stage1, stage2
    from bit_experiments.diffusion_llm import BCI_DiffusionLLM

    _aero_snapshot(world.root)
    prep = _script("prepare_ablations")
    config = stage1.bit_sentence_config()
    a_dir = world.save_ckpt(world.build(seed=1)[0], world.root / "ckpt" / "armA", (torch.randn(41, 384), torch.randn(41)))
    b_ar, _ = world.build(seed=4, inter_size=inter_size)  # stands in for B_ssl_s42 (Aero AR, arm B)
    b_best = world.save_ckpt(b_ar, world.root / "checkpoints" / "stage2" / "B_fake" / "BEST")
    save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(b_ar).items()}, str(b_best / stage2.SLIM))
    d_run, _ = world.build(seed=7, llm="aero-1-audio-diffusion", inter_size=inter_size)  # a trained Aero diffusion run
    d_best = world.save_ckpt(d_run, world.root / "other-copy" / "checkpoints" / "stage2" / "D_fake" / "BEST")
    save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(d_run).items()}, str(d_best / stage2.SLIM))
    world.write_registry({
        "frozen-B": {"kind": "frozen", "encoder": "abl/B_fake-encoder"},
        "phoneme-erasure": {"kind": "phoneme_erasure", "encoder": "abl/B_fake-encoder", "eraser": "abl/er/eraser.pt",
                            "offsets": [0]},
        "phoneme-erasure-aero-diffusion": {"kind": "phoneme_erasure", "encoder": "abl/D_fake-encoder",
                                           "eraser": "abl/er-d/eraser.pt", "offsets": [0]},
        "frozen-evalonly": {"kind": "frozen"}})
    prep.ROOT, root0 = world.root, prep.ROOT
    try:
        prep.export_encoder(config, "B_fake", world.root / "abl" / "B_fake-encoder")
        prep.export_encoder(config, "D_fake", world.root / "abl" / "D_fake-encoder", best=d_best)  # another copy's BEST
    finally:
        prep.ROOT = root0
    ds = {"train": _records(10, 0), "val": _records(5, 1), "test": _records(4, 2)}
    path, _ = prep.make_labels(config, ds, a_dir, world.root / "labels", "cpu", batch_size=4, sequences=_seqs)
    labels, lmeta = pf.load_labels(path)
    for name in ("phoneme-erasure", "phoneme-erasure-aero-diffusion"):
        prep.make_eraser(config, ds, labels, lmeta, na.entry(name), "cpu", batch_size=4, max_train_frames=10_000)
    eraser = torch.load(world.root / "abl" / "er" / "eraser.pt", weights_only=True)
    batch = _batch()

    # the retraining pair: B's frozen encoder under the Aero diffusion decoder, frozen-B and phoneme-erasure
    for kind in ("frozen-B", "phoneme-erasure"):
        model, _ = world.build(seed=0, llm="aero-1-audio-diffusion", inter_size=inter_size)
        assert isinstance(model, BCI_DiffusionLLM) and model.llm.get_base_model().config.vocab_size == 96
        rep = stage2.load_init(model, world.root / "abl" / "B_fake-encoder")
        assert rep["read_in_loaded"]
        rep = na.apply(model, kind, world.root / "abl" / "B_fake-encoder", seed=0)
        assert "projector_resized" not in rep and not any(p.requires_grad for p in model.neural_encoder.encoder.parameters())
        model.eval()
        enc = model.neural_encoder.encoder
        h = enc(batch["spikes"], batch["spikes_mask"], batch["spikes_timestamp"], batch["dataset_name"], force_mask=False)[0]
        want = model.projector(h @ eraser["A"].T + eraser["b"] if kind == "phoneme-erasure" else h)
        assert torch.allclose(_neural_tokens(model, batch)[0], want, atol=1e-5)
        mask_row = model.llm.get_base_model().model.embed_tokens.weight[TINY_MASK].clone()
        model.train()
        torch.manual_seed(0)
        out = model(**batch)
        out.loss.backward()
        assert torch.isfinite(out.loss) and all(p.grad is not None for p in model.projector.parameters())
        assert all(p.grad is None for p in enc.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.llm.named_parameters() if "lora_B" in n)
        assert torch.equal(model.llm.get_base_model().model.embed_tokens.weight[TINY_MASK], mask_row)
        model.eval()
        one = {k: v[:1] for k, v in batch.items() if k != "targets"} | {
            "input_ids": torch.tensor([[3, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        for decoding in (stage2.DECODINGS["greedy"], stage2.DECODINGS["nucleus"]):
            torch.manual_seed(0)
            ids = model.generate(**one, **decoding)
            assert ids.shape[0] == 1 and TINY_MASK not in ids.tolist()[0]
        best = world.root / f"BEST-{kind}"
        best.mkdir()
        save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(model).items()}, str(best / stage2.SLIM))
        again, _ = world.build(seed=5, llm="aero-1-audio-diffusion", inter_size=inter_size)
        na.apply(again, kind, None, seed=5)
        stage2.load_model_state(again, best)
        assert na.verify(again)["frozen"]
        assert torch.allclose(_neural_tokens(again.eval(), batch)[0], _neural_tokens(model, batch)[0], atol=1e-6)

    # decode-time checks on the diffusion run's own BEST: frozen-evalonly reproduces its decodes exactly; the eraser
    # fit to its own encoder loads and verifies; frozen-evalonly refuses an --init
    one = {k: v[:1] for k, v in batch.items() if k != "targets"} | {
        "input_ids": torch.tensor([[3, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    ref, _ = world.build(seed=8, llm="aero-1-audio-diffusion", inter_size=inter_size)
    stage2.load_model_state(ref, d_best)
    hooked, _ = world.build(seed=9, llm="aero-1-audio-diffusion", inter_size=inter_size)
    na.apply(hooked, "frozen-evalonly", None, seed=9)
    stage2.load_model_state(hooked, d_best)
    for decoding in (stage2.DECODINGS["greedy"], stage2.DECODINGS["nucleus"]):
        torch.manual_seed(3)
        a = ref.eval().generate(**one, **decoding)
        torch.manual_seed(3)
        assert torch.equal(hooked.eval().generate(**one, **decoding), a)
    with pytest.raises(ValueError, match="eval-only"):
        na.apply(world.build(seed=0)[0], "frozen-evalonly", world.root / "abl" / "B_fake-encoder", 0)
    erased, _ = world.build(seed=9, llm="aero-1-audio-diffusion", inter_size=inter_size)
    na.apply(erased, "phoneme-erasure-aero-diffusion", None, seed=9)
    stage2.load_model_state(erased, d_best)
    assert na.verify(erased)["expect_encoder_matches"]
    with pytest.raises(RuntimeError, match="digest"):  # B's eraser on the diffusion run's encoder is refused
        wrong, _ = world.build(seed=9, llm="aero-1-audio-diffusion", inter_size=inter_size)
        na.apply(wrong, "phoneme-erasure", None, seed=9)
        stage2.load_model_state(wrong, d_best)
        na.verify(wrong)
