"""Revision options of the block-diffusion decoder (docs/REVISION_RUN.md), CPU only: a tiny random Fast-dLLM v2 with
BIT's LoRA, and a scripted stand-in decoder for the sampler's edge cases. Needs BIT's environment (transformers 4.48.2,
peft 0.11.1); skipped without peft.

    <bit env>/python -m pytest tests/test_diffusion_revision.py

- Options off: forward (loss, logits, labels, gradients) and decodes (greedy and nucleus, sub-blocks 1/4/8/32,
  prefixes ending anywhere in a block, one or more blocks) are identical to run 3's diffusion_llm.py (RUN3), and the
  native decode trace equals scripts/decoding_order_sweep.py's traced_denoise.
- Options on: the aligned layout, the T2T corruption and its supervision, backward, the text-only loss (two-step
  design), the editing sampler (termination, caps, EOS, the AR first token, earlier blocks, trace replay) and the
  re-masking sampler (re-fill with context, early stop, EOS, the AR first token).
"""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
peft = pytest.importorskip("peft")

from bit_experiments import diffusion_llm  # noqa: E402
from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig  # noqa: E402
from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM  # noqa: E402

RUN3 = "0b98fa9"  # the commit holding run 3's diffusion_llm.py (B_ssl_fast-dllm-v2-1.5b_s42)
V, BOS, EOS, MASK, BS, D = 96, 80, 81, 95, 32, 64  # tiny vocabulary: ordinary tokens are the ids below BOS (filler)
SAMPLER = {"block_size": BS, "small_block_size": 8, "threshold": 0.9, "mask_id": MASK}
GREEDY = {"max_new_tokens": 25, "do_sample": False, "top_p": 1.0, "temperature": 1.0}  # as stage2.DECODINGS
NUCLEUS = {"max_new_tokens": 25, "do_sample": True, "top_p": 0.9, "temperature": 0.7}
PROMPT_A, PROMPT_B = [3, 4, 5], [6, 7, 8, 9, 10]
ROWS = [(24, 5), (25, 12), (29, 25), (48, 3), (55, 18), (120, 9)]  # (neural tokens, sentence tokens): prefix 32..128
REGISTRY = yaml.safe_load((ROOT / "configs" / "llms.yaml").read_text())
EDIT_ENTRY = REGISTRY["fast-dllm-v2-1.5b-edit"]["sampler"]
REMASK_ENTRY = REGISTRY["fast-dllm-v2-1.5b-remask"]["sampler"]


def tiny_llm(seed=0, head_scale=60.0):
    """Random 2-layer, 64-d Fast-dLLM v2 in BIT's LoRA (r 8, alpha 32, dropout 0.2, 7 projections), with nonzero LoRA
    weights. The head is scaled up so that about a third of the proposals clear the 0.9 threshold, and EOS's row a bit
    more, so that some decodes end at an EOS and others at the 25-token cap."""
    torch.manual_seed(seed)
    cfg = Fast_dLLM_QwenConfig(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                               bos_token_id=BOS, eos_token_id=EOS, bd_size=BS)
    targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    llm = peft.get_peft_model(Fast_dLLM_QwenForCausalLM(cfg),
                              peft.LoraConfig(r=8, lora_alpha=32, lora_dropout=0.2, target_modules=targets))
    with torch.no_grad():
        for name, p in llm.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
        llm.get_base_model().lm_head.weight.mul_(head_scale)
        llm.get_base_model().lm_head.weight[EOS] *= 1.5
    return llm


def bci(llm, module=diffusion_llm, **options):
    """BCI_DiffusionLLM around `llm` without BIT's encoder: prepare_embeds gives the ids' embeddings as the text and
    the batch's `spikes` as the neural tokens."""
    m = module.BCI_DiffusionLLM.__new__(module.BCI_DiffusionLLM)
    nn.Module.__init__(m)
    m.llm = llm
    if hasattr(m, "_configure"):
        m._configure(SAMPLER | options)
    else:  # run 3's __init__
        assert not options.keys() - SAMPLER.keys()
        m.sampler, m.block_size, m.eos_id, m.mask_id = SAMPLER | options, BS, EOS, MASK
    m.prepare_embeds = lambda ids, spikes, spikes_mask, *_: (spikes, m._parts()[2](ids), spikes_mask, None)
    return m


def batch(rows=ROWS, seed=0):
    """forward's kwargs: per trial prompt_a + prompt_b, then the sentence + EOS (right-padded), split after prompt_a,
    and that many neural tokens."""
    g = torch.Generator().manual_seed(seed)
    n_prompt = len(PROMPT_A) + len(PROMPT_B)
    length, n_neural = n_prompt + max(s for _, s in rows) + 1, max(n for n, _ in rows)
    input_ids, targets = torch.full((len(rows), length), EOS), torch.full((len(rows), length), -100)
    attention = torch.zeros(len(rows), length, dtype=torch.long)
    spikes_mask = torch.zeros(len(rows), n_neural, dtype=torch.long)
    for i, (n, s) in enumerate(rows):
        sentence = torch.randint(0, BOS, (s,), generator=g).tolist() + [EOS]
        ids = PROMPT_A + PROMPT_B + sentence
        input_ids[i, : len(ids)], targets[i, n_prompt: len(ids)] = torch.tensor(ids), torch.tensor(sentence)
        attention[i, : len(ids)], spikes_mask[i, :n] = 1, 1
    return {"input_ids": input_ids, "attention_mask": attention, "input_split": torch.full((len(rows),), len(PROMPT_A)),
            "spikes": torch.randn(len(rows), n_neural, D, generator=g), "spikes_mask": spikes_mask,
            "spikes_timestamp": None, "spikes_lengths": None, "targets": targets}


def prompt(n_neural, seed):
    """generate's kwargs for one trial (prompt only)."""
    g = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.tensor([PROMPT_A + PROMPT_B]), "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "input_split": torch.tensor([len(PROMPT_A)]), "spikes": torch.randn(1, n_neural, D, generator=g),
            "spikes_mask": torch.ones(1, n_neural, dtype=torch.long), "spikes_timestamp": None, "spikes_lengths": None}


def captured(model, b, seed=0):
    """forward under a seed, with the decoder's input and labels (x: both copies, [noisy ‖ clean]; lab) and denoising_ce's
    arguments (clean, labels)."""
    seen = {}
    real = model.denoising_ce

    def spy(clean, labels):
        seen["clean"], seen["labels"] = clean, labels
        return real(clean, labels)

    hook = model._parts()[0].register_forward_pre_hook(
        lambda _, args, kw: seen.update(x=kw["inputs_embeds"], lab=kw["labels"]), with_kwargs=True)
    model.denoising_ce = spy
    try:
        torch.manual_seed(seed)
        out = model(**b)
    finally:
        del model.denoising_ce
        hook.remove()
    return seen, out


def replay(rec):
    """A decode trace's commits and edits, applied in order: the output ids (cut at EOS / the length cap, as `out`)."""
    seq = {}
    for e in rec["events"]:
        if "first" in e:
            seq[e["first"][0]] = e["first"][1]
            continue
        for pos, tok, _, committed in e.get("m", []):
            assert pos not in seq
            if committed:
                seq[pos] = tok
        for pos, tok, new, _, edited in e.get("e", []):
            assert seq[pos] == tok and new != tok
            if edited:
                seq[pos] = new
        for pos, tok, _ in e.get("r", []):
            assert seq.pop(pos) == tok
    out, pos = [], 0
    while pos in seq and len(out) < len(rec["out"]):
        out.append(seq[pos])
        pos += 1
    return out


@pytest.fixture(scope="module")
def run3(tmp_path_factory):
    """Run 3's diffusion_llm.py (RUN3), imported as its own module."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{RUN3}:src/bit_experiments/diffusion_llm.py"],
                             check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"run 3's code ({RUN3}) is not in this checkout")
    path = tmp_path_factory.mktemp("run3") / "diffusion_llm_run3.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("diffusion_llm_run3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ options off = run 3


OFF = {"align_sentence_block": False, "edit_rate": 0.0, "editing_threshold": None}


@pytest.mark.parametrize("options", [{}, OFF], ids=["absent", "explicitly-off"])
def test_options_off_forward_is_run3s(run3, options):
    llm = tiny_llm()
    new, old = bci(llm, **options), bci(llm, run3)
    llm.train()  # LoRA dropout on: its draws must line up too
    for seed in range(3):
        b = batch(seed=seed)
        torch.manual_seed(seed)
        a = new(**b)
        torch.manual_seed(seed)
        r = old(**b)
        assert torch.equal(a.loss, r.loss) and torch.equal(a.preds, r.preds) and torch.equal(a.targets, r.targets)
        assert int(a.n_examples) == int(r.n_examples)
    grads = []
    for out in (a, r):
        llm.zero_grad()
        out.loss.backward()
        grads.append({n: p.grad.clone() for n, p in llm.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0])


@pytest.mark.parametrize("sbs", [1, 4, 8, 32])
def test_options_off_decodes_are_run3s(run3, sbs):
    llm = tiny_llm()
    llm.eval()
    new, old = bci(llm, small_block_size=sbs, **OFF), bci(llm, run3, small_block_size=sbs)
    g = torch.Generator().manual_seed(0)
    lengths = []
    for n in (5, 31, 32, 33, 37, 63, 64, 100, 40, 90):  # prefix tokens: tail 5, 31, 0, 1, 5, 31, 0, 4, 8, 26
        prefix = torch.randn(n, D, generator=g)
        for kw in (GREEDY, NUCLEUS, GREEDY | {"max_new_tokens": 70}):
            torch.manual_seed(n)
            a = new._denoise(prefix, eos=EOS, **kw)
            torch.manual_seed(n)
            assert a == old._denoise(prefix, eos=EOS, **kw)
            lengths.append(len(a))
    for n, seed in ((24, 0), (57, 1)):  # through generate (stubbed encoder)
        assert new.generate(**prompt(n, seed)).tolist() == old.generate(**prompt(n, seed)).tolist()
    assert max(lengths) > 25 and min(lengths) < 25  # multi-block and EOS-terminated decodes both occurred


def test_options_off_trace_is_the_sweeps():
    path = ROOT / "scripts" / "decoding_order_sweep.py"
    if not path.exists():
        pytest.skip("no scripts/decoding_order_sweep.py")
    spec = importlib.util.spec_from_file_location("decoding_order_sweep", path)
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    llm = tiny_llm()
    llm.eval()
    model = bci(llm)
    model.trace = []
    g = torch.Generator().manual_seed(1)
    for n in (5, 32, 33, 63, 100):
        prefix = torch.randn(n, D, generator=g)
        sweep.TRACES.clear()
        assert model._denoise(prefix, eos=EOS, **GREEDY) == sweep.traced_denoise(model, prefix, eos=EOS, **GREEDY)
        assert model.trace[-1] == sweep.TRACES[-1] and replay(model.trace[-1]) == model.trace[-1]["out"]


# ------------------------------------------------------------------ the registry entry and the option checks


@pytest.mark.parametrize("name, added", [
    ("fast-dllm-v2-1.5b-edit", {"align_sentence_block", "edit_rate", "editing_threshold", "max_post_steps", "max_edits"}),
    ("fast-dllm-v2-1.5b-remask", {"remask_threshold", "remask_rounds"})])
def test_entries_are_run3_plus_revision_options(name, added):
    run3_entry, entry = REGISTRY["fast-dllm-v2-1.5b"], REGISTRY[name]
    assert {k: v for k, v in entry.items() if k != "sampler"} == {k: v for k, v in run3_entry.items() if k != "sampler"}
    assert {k: entry["sampler"][k] for k in run3_entry["sampler"]} == run3_entry["sampler"]
    assert set(entry["sampler"]) - set(run3_entry["sampler"]) == added <= set(diffusion_llm.REVISION_DEFAULTS)
    assert all(entry["sampler"][k] not in (None, False, 0) for k in added)  # each option on


def test_option_checks():
    llm = tiny_llm()
    assert bci(llm).revision == diffusion_llm.REVISION_DEFAULTS | SAMPLER
    for entry in (EDIT_ENTRY, REMASK_ENTRY):
        assert bci(llm, **entry).revision == diffusion_llm.REVISION_DEFAULTS | SAMPLER | entry
    for bad in ({"edit_rte": 0.1}, {"edit_rate": 1.0}, {"editing_threshold": 0.0}, {"editing_threshold": 1.5},
                {"remask_threshold": 0.0}, {"remask_threshold": 1.2}, {"remask_rounds": 0}, {"remask_rounds": 1.5}):
        with pytest.raises(ValueError):
            bci(llm, **bad)


# ------------------------------------------------------------------ training: aligned block, T2T


def test_aligned_sentence_block_layout():
    llm = tiny_llm()
    llm.train()
    b = batch()
    (plain, _), (aligned, _) = captured(bci(llm), b), captured(bci(llm, align_sentence_block=True), b)
    filler = llm.get_input_embeddings()(torch.tensor(BOS))
    for i, (n, s) in enumerate(ROWS):
        p = len(PROMPT_A) + n + len(PROMPT_B)
        start = int((aligned["labels"][i] != -100).nonzero()[0])
        n_fill = (1 - p) % BS
        assert start % BS == 1 and start == n_fill + p  # the prefix's last token opens the sentence's block
        assert (start + s) // BS == start // BS  # sentence + EOS in that one block
        assert torch.equal(aligned["clean"][i, :n_fill], filler.expand(n_fill, -1))
        assert torch.equal(aligned["clean"][i, n_fill:start + BS - 1], plain["clean"][i, :p + BS - 1])
        assert aligned["labels"][i, start: start + BS - 1].tolist() == b["targets"][i, 8: 9 + s].tolist() + \
            [EOS] * (BS - 2 - s)  # sentence, EOS, then the block's EOS padding
        assert int((plain["labels"][i] != -100).nonzero()[0]) == p  # off: right after the prefix


def test_t2t_corrupts_and_supervises_visible_targets():
    llm = tiny_llm()
    llm.train()
    embed = llm.get_input_embeddings().weight
    rate, n_extra, n_target = 0.3, 0, 0
    for seed in range(4):
        b = batch(seed=seed)
        (off, o), (on, a) = captured(bci(llm), b, seed), captured(bci(llm, edit_rate=rate), b, seed)
        length = on["labels"].shape[1]
        is_target = (on["labels"] != -100).repeat(2, 1)  # both copies
        noisy_off, noisy_on = off["x"][:, :length], on["x"][:, :length]
        is_masked = (noisy_on == embed[MASK]).all(-1)
        assert torch.equal(is_masked, (noisy_off == embed[MASK]).all(-1))  # the same masks as with T2T off
        assert torch.equal(on["x"][:, length:], off["x"][:, length:])  # clean halves untouched
        sup_off, sup_on = off["lab"] != -100, on["lab"] != -100
        assert torch.equal(sup_off, is_masked & is_target)  # M2T: the masked target tokens
        extra = sup_on & ~sup_off
        assert torch.equal(sup_on & sup_off, sup_off) and not (extra & (is_masked | ~is_target)).any()
        assert torch.equal(on["lab"][extra], on["labels"].repeat(2, 1)[extra])  # supervised with the original
        changed = (noisy_on != noisy_off).any(-1)
        assert not (changed & ~extra).any()  # every other input as without T2T
        assert all((embed[:BOS] == r).all(-1).any() for r in noisy_on[extra])  # an ordinary token's embedding
        assert int(a.n_examples) == int(o.n_examples) + int(extra.sum())
        assert torch.isfinite(a.loss) and a.loss > o.loss
        n_extra, n_target = n_extra + int(extra.sum()), n_target + int(is_target[: len(ROWS)].sum())
    assert abs(n_extra / n_target - rate) < 0.05, (n_extra, n_target)  # each target is visible in exactly one copy
    llm.zero_grad()
    a.loss.backward()
    lora = [p.grad for n, p in llm.named_parameters() if "lora_" in n]
    assert all(g is not None and torch.isfinite(g).all() for g in lora) and any(g.abs().sum() > 0 for g in lora)
    assert all(p.grad is None for n, p in llm.named_parameters() if "lora_" not in n)  # base frozen (BIT's LoRA)


def test_text_path_is_the_released_objective():
    """text_block_diffusion_ce (text-only post-training, two-step design) on token ids: with edit_rate 0 it is Fast-dLLM
    v2's own training forward (same seed: same masks, same LoRA dropout); with edit_rate it trains."""
    llm = tiny_llm()
    llm.train()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, BOS, (3, 4 * BS), generator=g)
    labels = ids.clone()
    labels[0, :40] = -100  # a context span, never masked
    for seed in range(3):
        torch.manual_seed(seed)
        released = llm(input_ids=ids, labels=labels.clone(), mask_id=MASK).loss  # mean over the masked tokens
        torch.manual_seed(seed)
        ce, n, _, _ = diffusion_llm.text_block_diffusion_ce(llm, ids, MASK, labels)
        torch.testing.assert_close(ce / n, released, rtol=1e-5, atol=1e-6)
    torch.manual_seed(0)
    ce, n_edit, _, _ = diffusion_llm.text_block_diffusion_ce(llm, ids, MASK, edit_rate=0.3)
    llm.zero_grad()
    ce.backward()
    assert n_edit > ids.numel() and all(torch.isfinite(p.grad).all() for p in llm.parameters() if p.grad is not None)


# ------------------------------------------------------------------ the editing sampler


class Scripted:
    """Stand-in (decoder, lm_head, embed): one-hot embeddings carry the ids through an identity decoder, and
    policy(k, n) gives the logits of the k-th forward over n tokens. propose() writes slot proposals into such logits
    (slot i of a block is read from row tail + i - 1: token shift)."""

    def __init__(self, policy):
        self.policy, self.calls = policy, 0
        self.embed = nn.Embedding(V, V)
        self.embed.weight.data = torch.eye(V)

    def parts(self):
        return self.decoder, self.lm_head, self.embed

    def decoder(self, inputs_embeds, **_):
        return SimpleNamespace(last_hidden_state=inputs_embeds)

    def lm_head(self, h):
        logits = self.policy(self.calls, h.shape[1])
        self.calls += 1
        return logits[None]

    @staticmethod
    def propose(n, tail, proposals):
        """{slot: (token, probability)} -> [n, V] logits (other tokens equally likely; the mask is excluded later)."""
        logits = torch.zeros(n, V)
        for slot, (tok, p) in proposals.items():
            logits[max(tail + slot - 1, 0), tok] = math.log(p * (V - 2) / (1 - p))
        return logits


def scripted(policy, **options):
    s = Scripted(policy)
    model = bci(tiny_llm(), **options)
    model._parts = s.parts
    model.trace = []
    return model, s


EDITING = {"editing_threshold": 0.5, "max_post_steps": 16, "max_edits": 2}


@pytest.mark.parametrize("max_edits", [2, None])
def test_editing_terminates_when_the_model_never_agrees(max_edits):
    """Every pass proposes the other of two tokens everywhere at p 0.99: each committed token is overwritten on every
    pass until its edit cap, or until max_post_steps passes that only edit have run."""
    model, s = scripted(lambda k, n: Scripted.propose(n, 1, {i: (1 + k % 2, 0.99) for i in range(BS - 1)}),
                        **EDITING | {"max_edits": max_edits})
    out = model._denoise(torch.randn(BS + 1, V), eos=EOS, **GREEDY)  # tail 1: one prefix pass, then 31 slots
    rec = model.trace[-1]
    passes = [e for e in rec["events"] if "m" in e]
    fill = [e for e in passes if e["m"]]
    assert len(fill) == 4 and s.calls == 1 + len(passes)  # 4 sub-blocks, all proposals above 0.9
    edits = {}
    for e in passes:
        for pos, *_, edited in e["e"]:
            edits[pos] = edits.get(pos, 0) + edited
    if max_edits is None:
        assert len(passes) == 4 + 16 and max(edits.values()) > 2  # stopped by max_post_steps
    else:
        assert max(edits.values()) == 2 and not any(x[-1] for x in passes[-1]["e"])  # capped, then a pass edits nothing
        assert len(passes) < 4 + 16
    assert len(out) == 25 and EOS not in out and replay(rec) == out == rec["out"]


def test_editing_eos_handling():
    """An edit that turns the first EOS into a word reopens the block: its masks are filled again. Positions after the
    first EOS are never edited. The output is cut at the first EOS."""
    A, B, C, X, W = 11, 12, 13, 14, 15

    def policy(k, n):
        return Scripted.propose(n, 1, {
            1: {0: (A, .99), 1: (B, .99), 2: (EOS, .99), **{i: (W, .3) for i in range(3, 31)}},  # A B EOS: finished
            2: {0: (A, .99), 1: (B, .99), 2: (C, .95)},  # post pass: EOS -> C, so masks precede any EOS again
            3: {0: (A, .99), 1: (B, .99), 2: (C, .99), **{i: (EOS, .99) for i in range(3, 31)}},  # fills EOS
            4: {0: (A, .99), 1: (B, .99), 2: (C, .99), 3: (EOS, .99), **{i: (X, .99) for i in range(4, 31)}},
        }.get(k, {}))

    model, s = scripted(policy, **EDITING)
    out = model._denoise(torch.randn(BS + 1, V), eos=EOS, **GREEDY)
    passes = [e for e in model.trace[-1]["events"] if "m" in e]
    assert out == [A, B, C, EOS] and s.calls == 5
    assert [len(e["m"]) for e in passes] == [7, 0, 4, 0]  # fill, edit only, fill (reopened), edit only (none: stop)
    assert passes[1]["e"] == [[2, EOS, C, 0.95, 1]]
    assert passes[3]["e"] == []  # X proposed after the first EOS only
    assert replay(model.trace[-1]) == out


def test_editing_spares_the_ar_first_token_and_earlier_blocks():
    """No prompt in the first block (prefix of whole blocks): slot 0 is the AR next token, predicted from the cached
    block, so it is never edited. Edits stay inside the block being denoised."""
    Z, A, B = 20, 21, 22

    def policy(k, n):
        if k == 0:
            return Scripted.propose(n, 0, {n: (Z, .99)})  # prefix pass: the AR first token of block 0 is Z
        return Scripted.propose(n, 0, {i: ((A, B)[k % 2], .99) for i in range(1, BS)})

    model, _ = scripted(policy, **EDITING)
    out = model._denoise(torch.randn(BS, V), eos=EOS, **GREEDY | {"max_new_tokens": 70})
    rec = model.trace[-1]
    blocks = [e for e in rec["events"] if "first" in e]
    assert rec["tail"] == 0 and [e["b"] for e in blocks] == [0, 1, 2] and out[0] == Z
    assert all(out[e["first"][0]] == e["first"][1] for e in blocks)  # each block's AR first token kept
    for e in rec["events"]:
        if "m" in e:
            assert all(BS * e["b"] < pos < BS * (e["b"] + 1) for pos, *_ in e["e"])  # not slot 0, not earlier blocks
    assert any(e.get("e") for e in rec["events"]) and replay(rec) == out and len(out) == 70


def test_editing_off_has_no_post_passes_or_edit_events():
    model, s = scripted(lambda k, n: Scripted.propose(n, 1, {i: (1 + k % 2, 0.99) for i in range(BS - 1)}))
    model._denoise(torch.randn(BS + 1, V), eos=EOS, **GREEDY)
    assert s.calls == 1 + 4 and all("e" not in e for e in model.trace[-1]["events"])


REMASK = {"remask_threshold": 0.9, "remask_rounds": 2}


def test_remasking_refills_low_confidence_tokens_with_context():
    """A token committed as the sub-block's best guess (p 0.6 < 0.9) is masked again once the block is done and
    re-filled with the rest of the sentence visible."""
    A, B, C, W = 11, 12, 13, 15

    def policy(k, n):
        return Scripted.propose(n, 1, {
            1: {0: (A, .99), 1: (B, .6), 2: (EOS, .99), **{i: (W, .3) for i in range(3, 31)}},  # A _ EOS
            2: {1: (B, .6), **{i: (W, .3) for i in range(3, 31)}},  # B as the forced best guess: A B EOS, finished
            3: {1: (C, .95)},  # re-fill round: C, now with A and EOS visible
        }.get(k, {}))

    model, s = scripted(policy, **REMASK)
    out = model._denoise(torch.randn(BS + 1, V), eos=EOS, **GREEDY)
    events = model.trace[-1]["events"]
    assert out == [A, C, EOS] and s.calls == 4  # the second round finds nothing below 0.9 and needs no pass
    assert [e["r"] for e in events if "r" in e] == [[[1, B, 0.6]]]
    assert replay(model.trace[-1]) == out


def test_remasking_stops_when_a_round_changes_nothing():
    A, B, W = 11, 12, 15

    def policy(k, n):
        if k == 1:
            return Scripted.propose(n, 1, {0: (A, .99), 1: (B, .6), 2: (EOS, .99), **{i: (W, .3) for i in range(3, 31)}})
        return Scripted.propose(n, 1, {1: (B, .6), **{i: (W, .3) for i in range(3, 31)}})  # always B again

    model, s = scripted(policy, remask_threshold=0.9, remask_rounds=5)
    assert model._denoise(torch.randn(BS + 1, V), eos=EOS, **GREEDY) == [A, B, EOS]
    assert s.calls == 4 and sum("r" in e for e in model.trace[-1]["events"]) == 1  # one round, unchanged: stop


def test_remasking_the_eos_reopens_the_block():
    A, B, W = 11, 12, 15

    def policy(k, n):
        return Scripted.propose(n, 1, {
            1: {0: (A, .99), **{i: (W, .3) for i in range(1, 31)}},
            2: {1: (EOS, .5), **{i: (W, .3) for i in range(2, 31)}},  # EOS as the best guess: A EOS, finished
            3: {1: (B, .99), **{i: (EOS, .99) for i in range(2, 31)}},  # re-filled: the sentence goes on
        }.get(k, {}))

    model, s = scripted(policy, **REMASK)
    assert model._denoise(torch.randn(BS + 1, V), eos=EOS, **GREEDY) == [A, B, EOS]
    assert [e["r"] for e in model.trace[-1]["events"] if "r" in e] == [[[1, EOS, 0.5]]]
    assert replay(model.trace[-1]) == [A, B, EOS]


def test_remasking_spares_the_ar_first_token():
    Z, A = 20, 21

    def policy(k, n):
        if k == 0:
            return Scripted.propose(n, 0, {n: (Z, .99)})
        return Scripted.propose(n, 0, {i: (A, .6) for i in range(1, BS)})

    model, _ = scripted(policy, remask_threshold=1.0, remask_rounds=3)
    out = model._denoise(torch.randn(BS, V), eos=EOS, **GREEDY)
    remasked = [pos for e in model.trace[-1]["events"] for pos, *_ in e.get("r", [])]
    assert out[0] == Z and remasked and 0 not in remasked and replay(model.trace[-1]) == out


# ------------------------------------------------------------------ all options on, real (tiny) model


def test_all_options_on_end_to_end():
    llm = tiny_llm()
    model = bci(llm, **EDIT_ENTRY)
    llm.train()
    torch.manual_seed(0)
    loss = model(**batch()).loss
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in llm.parameters() if p.grad is not None)
    llm.eval()
    model.trace = []
    passes = []
    for i, n in enumerate((24, 25, 30, 41, 57, 88, 120, 150)):
        for kw in (GREEDY, NUCLEUS):
            torch.manual_seed(i)
            out = model.generate(**prompt(n, i), **kw)[0].tolist()
            rec = model.trace[-1]
            assert rec["tail"] == 1 and rec["prefix_len"] % BS == 1 and rec["out"] == out and replay(rec) == out
            assert len(out) <= 25 and all(e["b"] == 0 for e in rec["events"])  # one block
            passes.append(sum("m" in e for e in rec["events"]))
    edits = [x for r in model.trace for e in r["events"] for x in e.get("e", [])]
    assert max(passes) <= BS - 1 + EDIT_ENTRY["max_post_steps"]
    assert any(x[-1] for x in edits) and not all(x[-1] for x in edits)  # some edits made, some proposals declined


def test_remask_entry_end_to_end():
    """Run 3's layout and sampler plus re-masking (decode-only entry): decodes terminate, replay, and re-mask."""
    llm = tiny_llm()
    llm.eval()
    model = bci(llm, **REMASK_ENTRY)
    model.trace = []
    for i, n in enumerate((24, 25, 30, 41, 57, 88, 120, 150)):
        for kw in (GREEDY, NUCLEUS):
            torch.manual_seed(i)
            out = model.generate(**prompt(n, i), **kw)[0].tolist()
            assert replay(model.trace[-1]) == out == model.trace[-1]["out"]
    rounds = [sum("r" in e for e in r["events"] if e.get("b") == 0) for r in model.trace]
    assert any(rounds) and max(rounds) <= REMASK_ENTRY["remask_rounds"]
