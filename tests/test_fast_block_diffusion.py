"""The packed block-diffusion training step (bit_experiments.fast_block_diffusion, docs/DIFFUSION_SPEEDUP.md) against
the plain one (diffusion_llm.block_diffusion_ce, run 3's), CPU only. The model is a tiny random Fast-dLLM v2 in BIT's
LoRA. These tests need BIT's environment (transformers 4.48.2, peft 0.11.1) and are skipped without peft.

    <bit env>/python -m pytest tests/test_fast_block_diffusion.py

- Plan: the packed tokens are exactly the dependency closure of the decoded outputs under block_diff_mask, and the
  packed mask is block_diff_mask restricted to them (flex block mask included).
- Step: the same masks, labels, summed CE, logits and gradients (every LoRA parameter, and the input embeddings,
  which carry the encoder's and projector's gradients). This holds over seeds and layouts: tails 0 to 31, prefixes
  shorter than a block, sentences across blocks, T15-like prefixes, label holes, T2T on, dense and flex attention,
  share_clean on and off.
- LoRA dropout: with a dropout keyed on the token copy, the steps agree with dropout on. So they differ only in which
  random draw each copy gets. With real dropout, the paired loss difference has mean 0. With dropout off, the
  generator's state after the step is the same.
- bf16 autocast, the BCI_DiffusionLLM hook-up (fast_step flag), and the registry entry.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
peft = pytest.importorskip("peft")

from bit_experiments import diffusion_llm, fast_block_diffusion as fast  # noqa: E402
from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig  # noqa: E402
from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM, block_diff_mask  # noqa: E402

V, BOS, EOS, MASK, BS, D = 96, 80, 81, 95, 32, 64
REGISTRY = yaml.safe_load((ROOT / "configs" / "llms.yaml").read_text())
# (prefix length, target tokens incl. EOS) per trial
TAILS = [(32, 5), (33, 31), (37, 12), (63, 2), (81, 40), (20, 9)]  # tails 0, 1, 5, 31, 17; prefix inside block 0
T15 = [(252, 10), (150, 4), (209, 8), (96, 18)]                     # T15-like (run 3's test prefixes: 85-502, mean 210)
EDGES = [(64, 1), (31, 1), (1, 3), (95, 33)]                        # one-token target, tail 31 / 1, 2 target blocks
LAYOUTS = {"tails": TAILS, "t15": T15, "edges": EDGES}


def tiny_llm(seed=0, dropout=0.0):
    """Random 2-layer, 64-d Fast-dLLM v2 in BIT's LoRA (r 8, alpha 32, 7 projections), with nonzero LoRA weights."""
    torch.manual_seed(seed)
    cfg = Fast_dLLM_QwenConfig(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                               bos_token_id=BOS, eos_token_id=EOS, bd_size=BS)
    targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    llm = peft.get_peft_model(Fast_dLLM_QwenForCausalLM(cfg),
                              peft.LoraConfig(r=8, lora_alpha=32, lora_dropout=dropout, target_modules=targets))
    with torch.no_grad():
        for name, p in llm.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
        llm.get_base_model().lm_head.weight.mul_(10)
    return llm.train()


def layout(rows, seed=0, bucket=128, holes=False):
    """block_diffusion_ce's (clean, labels) as BCI_DiffusionLLM.forward builds them: per trial a prefix (-100), the
    target padded to the end of its block, then batch padding up to a multiple of `bucket`. With holes, part of each
    target is relabelled -100 (a context span inside the targets, as the text path allows)."""
    g = torch.Generator().manual_seed(seed)
    ends = [p + t + (-(p + t)) % BS for p, t in rows]
    length = -(-max(ends) // bucket) * bucket
    labels = torch.full((len(rows), length), -100)
    for i, ((p, _), e) in enumerate(zip(rows, ends)):
        labels[i, p:e] = torch.randint(0, BOS, (e - p,), generator=g)
        if holes and e - p > 4:
            labels[i, p + 2: p + 2 + (e - p) // 2] = -100
    clean = torch.randn(len(rows), length, D, generator=g)
    return clean, labels


def parts(llm):
    base = llm.get_base_model()
    return base.model, base.lm_head, base.get_input_embeddings()


def run(llm, clean, labels, seed, step="plain", edit_rate=0.0, **kw):
    """One step under a seed: (CE, n, logits, labels, corrupted, LoRA gradients, input-embedding gradient)."""
    llm.zero_grad()
    x = clean.clone().requires_grad_()
    torch.manual_seed(seed)
    fn = diffusion_llm.block_diffusion_ce if step == "plain" else fast.fast_block_diffusion_ce
    ce, n, logits, lab, corrupted = fn(*parts(llm), x, labels, BS, MASK, edit_rate, BOS, True, **kw)
    ce.backward()
    grads = {k: p.grad.clone() for k, p in llm.named_parameters() if p.requires_grad}
    return ce.detach(), n, logits.detach(), lab, corrupted, grads, x.grad


def assert_same_step(a, b, rtol=1e-5, grad_rtol=1e-4):
    ce_a, n_a, logits_a, lab_a, cor_a, g_a, x_a = a
    ce_b, n_b, logits_b, lab_b, cor_b, g_b, x_b = b
    assert torch.equal(n_a, n_b) and torch.equal(lab_a, lab_b) and torch.equal(cor_a, cor_b)  # same masks, same order
    assert torch.allclose(ce_a, ce_b, rtol=rtol, atol=0)
    assert torch.allclose(logits_a, logits_b, rtol=rtol, atol=1e-5)
    for k in g_a:
        scale = g_a[k].abs().max()
        assert scale > 0, k  # every LoRA parameter is reached
        assert (g_a[k] - g_b[k]).abs().max() <= grad_rtol * scale, k
    assert (x_a - x_b).abs().max() <= grad_rtol * x_a.abs().max()


# ------------------------------------------------------------------ plan and mask


def closure(labels):
    """The plain step's needed tokens, found independently of the plan: start from the decoded outputs (noisy p with
    p + 1 a target) and add every token a needed token attends to, until nothing changes (one round per layer at
    most, but iterate to the fixpoint). Returns [B, 2L] booleans over [noisy ‖ clean]."""
    b, length = labels.shape
    idx = torch.arange(2 * length)
    m = block_diff_mask(None, None, idx[:, None], idx[None, :], block_size=BS, n=length)
    need = torch.zeros(b, 2 * length, dtype=torch.bool)
    need[:, : length - 1] = labels[:, 1:] != -100
    while True:
        grown = need | (need[:, :, None] & m[None]).any(1)
        if torch.equal(grown, need):
            return need
        need = grown


@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("name", [*LAYOUTS, "holes"])
def test_plan_is_the_dependency_closure(name, share):
    rows = LAYOUTS.get(name, TAILS)
    _, labels = layout(rows, holes=name == "holes")
    b, length = labels.shape
    plan = fast.pack_plan(labels, BS, share, quantum=128)
    real = plan.src[: plan.n_real]
    row, t = real // (2 * length), real % (2 * length)
    got = torch.zeros(2 * b, 2 * length, dtype=torch.bool)
    got[row, t] = True
    assert real.unique().numel() == real.numel()  # no token twice
    need = closure(labels)
    for c in range(2):
        if share and c == 1:  # copy 2 reads copy 1's clean tokens
            assert torch.equal(got[b:, :length], need[:, :length]) and not got[b:, length:].any()
        else:
            assert torch.equal(got[c * b:(c + 1) * b], need)
    assert len(plan) % 128 == 0 and len(plan) - plan.n_real < 128
    assert torch.equal(plan.pos[: plan.n_real], t % length)
    assert torch.equal(plan.noisy[: plan.n_real], t < length)
    assert torch.equal(plan.out_src, real[t < length])  # every computed noisy output goes back to its slot


@pytest.mark.parametrize("share", [False, True])
def test_packed_mask_is_block_diff_mask(share):
    _, labels = layout(TAILS)
    b, length = labels.shape
    plan = fast.pack_plan(labels, BS, share, quantum=128)
    dense = plan.dense_mask()
    idx = torch.arange(2 * length)
    plain = block_diff_mask(None, None, idx[:, None], idx[None, :], block_size=BS, n=length)
    n = plan.n_real
    row, t = plan.src[:n] // (2 * length), plan.src[:n] % (2 * length)
    same = row[:, None] == row[None, :]
    if share:  # the clean tokens of trial i serve both of its copies
        same |= (row[:, None] % b == row[None, :] % b) & (t[None, :] >= length)
    assert torch.equal(dense[:n, :n], same & plain[t[:, None], t[None, :]])
    assert not dense[:n, n:].any() and not dense[n:, :n].any() and dense[n:, n:].all()  # pads: on their own
    assert dense.any(1).all()  # every query sees at least one key (no NaN rows under SDPA)
    # the flex block mask comes from the same mask_mod: tile by tile, it marks exactly the tiles with an allowed pair
    block = fast.create_block_mask(plan.mask_mod(), None, None, len(plan), len(plan), device="cpu")
    tiles = dense.view(len(plan) // 128, 128, len(plan) // 128, 128).any(3).any(1)
    assert torch.equal(block.to_dense()[0, 0].bool(), tiles)


def test_plan_depends_on_targets_only():
    """Masked or not, every target's predecessor block is computed: the shapes do not depend on the draws."""
    _, labels = layout(TAILS)
    other = torch.where(labels != -100, torch.randint_like(labels, 0, BOS), labels)
    a, b = fast.pack_plan(labels, BS), fast.pack_plan(other, BS)
    assert all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("src", "pos", "noisy", "blk", "ctx", "grp"))


# ------------------------------------------------------------------ the step


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", [*LAYOUTS, "holes"])
@pytest.mark.parametrize("share, attention", [(False, "dense"), (False, "flex"), (True, "dense")])
def test_step_matches_plain(seed, name, share, attention):
    llm = tiny_llm(seed)
    clean, labels = layout(LAYOUTS.get(name, TAILS), seed, holes=name == "holes")
    plain = run(llm, clean, labels, 10 + seed)
    packed = run(llm, clean, labels, 10 + seed, "fast", share_clean=share, quantum=128, attention=attention)
    assert_same_step(plain, packed)


@pytest.mark.parametrize("share", [False, True])
def test_step_matches_plain_with_t2t(share):
    """edit_rate on (the revision run's T2T term): the same corruption, supervision and gradients."""
    llm = tiny_llm(3)
    clean, labels = layout(TAILS, 3)
    plain = run(llm, clean, labels, 7, edit_rate=0.3)
    packed = run(llm, clean, labels, 7, "fast", edit_rate=0.3, share_clean=share, quantum=128)
    assert plain[4].any()  # some tokens corrupted
    assert_same_step(plain, packed)


def test_default_quantum_and_quantum_check():
    llm = tiny_llm(4)
    clean, labels = layout(T15, 4)
    assert len(fast.pack_plan(labels, BS)) % fast.PACK_QUANTUM == 0
    assert_same_step(run(llm, clean, labels, 5), run(llm, clean, labels, 5, "fast"))
    with pytest.raises(ValueError):
        fast.pack_plan(labels, BS, quantum=100)


def test_bf16_autocast():
    """Under bf16 autocast (the training precision), the two steps agree to bf16 accuracy."""
    llm = tiny_llm(5)
    clean, labels = layout(TAILS, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        plain = run(llm, clean, labels, 6)
        packed = run(llm, clean, labels, 6, "fast", quantum=128)
    assert torch.equal(plain[3], packed[3])
    assert torch.allclose(plain[0], packed[0], rtol=2e-2)
    for k in plain[5]:
        a, b = plain[5][k].flatten().float(), packed[5][k].flatten().float()
        assert torch.cosine_similarity(a, b, 0) > 0.99, k


# ------------------------------------------------------------------ LoRA dropout and the random stream


class KeyedDropout(nn.Module):
    """Dropout whose mask is a fixed function of (module, token copy, feature). The same token copy then gets the same
    mask in the plain and the packed step. `keys` ([rows, T], the LoRA input's leading dims) names each row's token
    copies by their flat index in block_diffusion_ce's [2B, 2L] input."""
    keys = None

    def __init__(self, p, n_keys, width, seed):
        super().__init__()
        self.p = p
        self.register_buffer("u", torch.rand(n_keys, width, generator=torch.Generator().manual_seed(seed)))

    def forward(self, x):
        if not self.training:
            return x
        return x * (self.u[KeyedDropout.keys] >= self.p) / (1 - self.p)


def keyed(llm, n_keys, p=0.2):
    """Replace every LoRA dropout of `llm` by a KeyedDropout."""
    lora = [m for m in llm.modules() if hasattr(m, "lora_dropout") and "default" in m.lora_dropout]
    for i, m in enumerate(lora):
        m.lora_dropout["default"] = KeyedDropout(p, n_keys, m.in_features, seed=i)
    return llm


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("share", [False, True])
def test_keyed_dropout_matches(seed, share):
    """Dropout on, keyed by token copy: the packed step equals the plain one. The only dropout difference between the
    two is therefore which random draw each computed copy gets, and that the skipped copies draw nothing. With
    share_clean, the plain step must give a trial's two clean copies the same mask, the one difference it makes."""
    clean, labels = layout(TAILS, seed)
    b, length = labels.shape
    llm = keyed(tiny_llm(seed), 2 * b * 2 * length)
    grid = torch.arange(2 * b * 2 * length).view(2 * b, 2 * length)
    if share:  # copy 2's clean half takes copy 1's keys
        grid[b:, length:] = grid[:b, length:]
    KeyedDropout.keys = grid
    plain = run(llm, clean, labels, 20 + seed)
    KeyedDropout.keys = fast.pack_plan(labels, BS, share, quantum=128).src[None]
    packed = run(llm, clean, labels, 20 + seed, "fast", share_clean=share, quantum=128)
    assert_same_step(plain, packed)
    llm.eval()  # and the keys matter: without them the losses differ
    KeyedDropout.keys = None
    assert not torch.allclose(run(llm, clean, labels, 20 + seed)[0], plain[0])


def test_real_dropout_same_distribution():
    """Real LoRA dropout (p 0.2). On each seed the masks are the same and only the dropout draws differ. Over 200
    seeds, the paired loss difference has mean 0 (|z| < 4)."""
    llm = tiny_llm(6, dropout=0.2)
    clean, labels = layout(TAILS, 6)
    diffs = []
    with torch.no_grad():
        for s in range(200):
            torch.manual_seed(s)
            a = diffusion_llm.block_diffusion_ce(*parts(llm), clean, labels, BS, MASK)[0]
            torch.manual_seed(s)
            b = fast.fast_block_diffusion_ce(*parts(llm), clean, labels, BS, MASK, quantum=128)[0]
            diffs.append(float(a - b))
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    assert sd > 0  # dropout is on
    assert abs(mean) < 4 * sd / math.sqrt(len(diffs))


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_random_stream_after_the_step(dropout):
    """With dropout off, the generator's state after the packed step is the plain step's: the masks are the only
    draws. With dropout on, the packed step draws fewer dropout masks, so the state (and every later draw) differs."""
    llm = tiny_llm(7, dropout=dropout)
    clean, labels = layout(TAILS, 7)
    states = []
    for step, kw in (("plain", {}), ("fast", {"quantum": 128})):
        run(llm, clean, labels, 8, step, **kw)
        states.append(torch.get_rng_state())
    assert torch.equal(*states) == (dropout == 0)


# ------------------------------------------------------------------ hook-up: BCI_DiffusionLLM and the registry

SAMPLER = {"block_size": BS, "small_block_size": 8, "threshold": 0.9, "mask_id": MASK}
PROMPT_A, PROMPT_B = [3, 4, 5], [6, 7, 8, 9, 10]


def bci(llm, **options):
    """BCI_DiffusionLLM around `llm` without BIT's encoder: prepare_embeds gives the ids' embeddings as the text and
    the batch's `spikes` as the neural tokens (as in tests/test_diffusion_revision.py)."""
    m = diffusion_llm.BCI_DiffusionLLM.__new__(diffusion_llm.BCI_DiffusionLLM)
    nn.Module.__init__(m)
    m.llm = llm
    m._configure(SAMPLER | options)
    m.prepare_embeds = lambda ids, spikes, spikes_mask, *_: (spikes, m._parts()[2](ids), spikes_mask, None)
    return m


def batch(rows, seed=0):
    """forward's kwargs: per trial prompt_a + prompt_b, then the sentence + EOS, and that many neural tokens."""
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


@pytest.mark.parametrize("options", [{}, {"share_clean": True}, {"pack_quantum": 128},
                                     {"edit_rate": 0.2, "align_sentence_block": True}])
def test_forward_with_fast_step(options):
    """BCI_DiffusionLLM.forward with fast_step: the same loss, examples, logits, targets and gradients (LoRA and the
    neural tokens, which stand in for the encoder's output) as without it. Revision options combine with it."""
    rows = [(24, 5), (25, 12), (29, 25), (48, 3), (55, 18), (120, 9)]  # (neural tokens, sentence tokens)
    llm = tiny_llm(8)
    outs = []
    plain_options = {k: v for k, v in options.items() if k not in diffusion_llm.FAST_STEP_DEFAULTS}
    for opts in (plain_options, options | {"fast_step": True}):
        model = bci(llm, **opts)
        b = batch(rows, 9)
        b["spikes"].requires_grad_()
        llm.zero_grad()
        torch.manual_seed(11)
        out = model(**b)
        out.loss.backward()
        outs.append((out, {k: p.grad.clone() for k, p in llm.named_parameters() if p.requires_grad},
                     b["spikes"].grad))
    (a, ga, sa), (f, gf, sf) = outs
    assert torch.equal(a.n_examples, f.n_examples) and torch.equal(a.targets, f.targets)
    assert torch.allclose(a.loss, f.loss, rtol=1e-5) and torch.allclose(a.preds, f.preds, rtol=1e-5, atol=1e-5)
    for k in ga:
        assert (ga[k] - gf[k]).abs().max() <= 1e-4 * ga[k].abs().max(), k
    assert (sa - sf).abs().max() <= 1e-4 * sa.abs().max()


def test_fast_step_options_are_checked():
    llm = tiny_llm()
    assert bci(llm).revision == diffusion_llm.REVISION_DEFAULTS | SAMPLER  # fast-step options are not revision options
    bci(llm, fast_step=True, share_clean=True, pack_quantum=1024)
    for bad in ({"fast_stp": True}, {"share_clean": True}, {"pack_quantum": 512}, {"fast_step": True, "pack_quantum": 100},
                {"fast_step": "yes"}):
        with pytest.raises(ValueError):
            bci(llm, **bad)


def test_fast_step_off_by_default_in_the_registry():
    """Only the entries meant for it turn it on; run 3's entry and the Instruct control keep the plain step."""
    for name in ("fast-dllm-v2-1.5b", "qwen2.5-1.5b-instruct-diffusion"):
        assert not set(REGISTRY[name].get("sampler", {})) & set(diffusion_llm.FAST_STEP_DEFAULTS)
    run3_entry, entry = REGISTRY["fast-dllm-v2-1.5b"], REGISTRY["fast-dllm-v2-1.5b-fast"]
    assert {k: v for k, v in entry.items() if k != "sampler"} == {k: v for k, v in run3_entry.items() if k != "sampler"}
    assert entry["sampler"] == run3_entry["sampler"] | {"fast_step": True}


def test_step_cost():
    """Token counts on a hand-checked layout: trial 0 (prefix 32, 10 targets: blocks 0-1) needs clean block 0 and noisy
    blocks 0-1; trial 1 (prefix 45, 30 targets: blocks 1-2) clean 0-1, noisy 1-2; trial 2 (prefix 70, 5 targets:
    block 2) clean 0-1, noisy 2. Twice (two copies): 2 * (3 + 4 + 3) blocks = 640 tokens, against 2 * 3 * 256."""
    labels = torch.full((3, 128), -100)
    for i, (p, e) in enumerate([(32, 64), (45, 96), (70, 96)]):
        labels[i, p:e] = 1
    cost = fast.step_cost(labels, BS, quantum=128)
    assert cost["plain"]["tokens"] == 1536 and cost["packed"]["real_tokens"] == cost["packed"]["tokens"] == 640
    shared = fast.step_cost(labels, BS, share_clean=True, quantum=128)["packed"]
    assert shared["real_tokens"] == 640 - 32 * (1 + 2 + 2)  # copy 2's clean blocks


# ------------------------------------------------------------------ flex kernel tiles (first GPU smoke, 2026-10-23)

SMEM_LIMIT = 101_376  # bytes of shared memory per block on the RTX PRO 6000 Blackwell (sm_120), as Triton reported


def fwd_smem(block_m, block_n, stages, head_dim=128, resident_q=False, fp32_scratch=False):
    """Shared memory of flex attention's forward in bf16: `stages` buffers of a K and a V tile (block_n x head_dim),
    plus the scratch for converting the packed mask's block_m x block_n output (one byte per element, rows padded by
    8; or 4 bytes, rows padded by 4, as a worst case), plus a resident Q tile as a worst case."""
    kv = stages * 2 * block_n * head_dim * 2
    scratch = block_m * (block_n + 4) * 4 if fp32_scratch else block_m * (block_n + 8)
    return kv + scratch + (block_m * head_dim * 2 if resident_q else 0)


def test_packed_flex_tiles_fit_and_divide():
    """This model gives exactly the failed smoke's 107,520 bytes for torch 2.7's forward default on this GPU (128, 64,
    3 stages), and 98,304 (fits) without the mask scratch (the plain step, which ran). The packed tiles fit even in
    the worst case, divide the block mask's tile, and the backward ones are the vendored ones (which fit for run 3)."""
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
    from bit_experiments.fast_dllm_v2 import modeling

    assert fwd_smem(128, 64, 3) == 107_520 and fwd_smem(128, 64, 3) - 128 * 72 == 98_304 < SMEM_LIMIT
    opt = fast.PACKED_FLEX_KERNEL_OPTIONS
    m, n, stages = opt["fwd_BLOCK_M"], opt["fwd_BLOCK_N"], opt["fwd_num_stages"]
    assert fwd_smem(m, n, stages) == 70_144
    assert fwd_smem(m, n, stages, resident_q=True, fp32_scratch=True) == 99_328 < SMEM_LIMIT
    assert {k: v for k, v in opt.items() if k.startswith("bwd_")} == modeling.FLEX_KERNEL_OPTIONS
    assert modeling.FLEX_KERNEL_OPTIONS == {"bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 64, "bwd_BLOCK_M2": 64,
                                            "bwd_BLOCK_N2": 32, "bwd_num_warps": 4, "bwd_num_stages": 2}  # run 3's
    assert set(opt) == {f"{d}_{k}" for d, ks in (("fwd", ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages")),
                                                 ("bwd", ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2",
                                                          "num_warps", "num_stages"))) for k in ks}
    for k, v in opt.items():
        if "BLOCK" in k:
            assert fast.FLEX_TILE % v == 0 and _DEFAULT_SPARSE_BLOCK_SIZE % v == 0, k


def test_packed_flex_attention_passes_its_options(monkeypatch):
    """The compiled packed kernel calls flex attention with PACKED_FLEX_KERNEL_OPTIONS (checked on its uncompiled
    function; Inductor cannot build flex on this CPU)."""
    seen = {}

    def recorder(q, k, v, **kw):
        seen.update(kw)
        return v

    monkeypatch.setattr(fast, "flex_attention", recorder)
    fn = getattr(fast.packed_flex_attention, "_torchdynamo_orig_callable", None) or fast.packed_flex_attention.__wrapped__
    mask = object()
    fn(torch.zeros(1), torch.zeros(1), torch.zeros(1), mask=mask)
    assert seen == {"block_mask": mask, "enable_gqa": True, "kernel_options": fast.PACKED_FLEX_KERNEL_OPTIONS}


def test_compiled_branch_uses_the_packed_kernel(monkeypatch):
    """The CUDA branch of packed_forward (forced on CPU) calls packed_flex_attention once per layer and never the
    vendored fused_flex_attention, and the step still matches the plain one (the kernel replaced by eager flex)."""
    from torch.nn.attention.flex_attention import BlockMask
    from bit_experiments.fast_dllm_v2 import modeling

    calls = []

    def packed(q, k, v, mask=None):
        calls.append(mask)
        assert q.dtype == k.dtype == v.dtype
        return fast.flex_attention(q, k, v, block_mask=mask, enable_gqa=True)

    def vendored(*a, **kw):
        raise AssertionError("the packed step called the plain step's kernel")

    llm = tiny_llm(9)
    clean, labels = layout(TAILS, 9)
    plain = run(llm, clean, labels, 12)
    monkeypatch.setattr(fast, "_compiled_flex", lambda device: True)
    monkeypatch.setattr(fast, "packed_flex_attention", packed)
    monkeypatch.setattr(modeling, "fused_flex_attention", vendored)
    packed_run = run(llm, clean, labels, 12, "fast", quantum=128, attention="flex")
    assert len(calls) == 2 and all(isinstance(m, BlockMask) for m in calls)  # 2 layers, forward
    assert_same_step(plain, packed_run)


def test_packed_kernel_compiles_statically_per_packed_length():
    """The candidate fix for the training failure: packed_flex_attention is compiled with dynamic=False, so every
    packed length gets its own static graph (Dynamo level, aot_eager on CPU), with correct results for new tables of
    the same length; and the recompile limit leaves room for the lengths x grad modes of a run."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    assert fast.PACKED_FLEX_DYNAMIC is False
    limit = getattr(torch._dynamo.config, "recompile_limit", None) or torch._dynamo.config.cache_size_limit
    assert limit >= 64
    torch._dynamo.reset()
    counters.clear()
    compiled = fast.compile_packed_flex(backend="aot_eager")
    g = torch.Generator().manual_seed(0)
    for rows in (TAILS, TAILS[:3], T15, TAILS):  # packed lengths 1024, 512, ..., 1024 again (new tables)
        _, labels = layout(rows)
        plan = fast.pack_plan(labels, BS, quantum=128)
        t = len(plan)
        q, k, v = (torch.randn(1, h, t, 16, generator=g) for h in (4, 2, 2))
        bm = fast.create_block_mask(plan.mask_mod(), None, None, t, t, device="cpu")
        out = compiled(q, k, v, mask=bm)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=plan.dense_mask(), enable_gqa=True)
        assert torch.allclose(out, ref, atol=1e-5)
    lengths = {len(fast.pack_plan(layout(r)[1], BS, quantum=128)) for r in (TAILS, TAILS[:3], T15)}
    assert counters["stats"]["unique_graphs"] == len(lengths)  # one static graph per length, none dynamic
