"""Probes, RSA and foil contrasts (docs/PROBES_FOILS.md), CPU only. The probe, RSA, foil-construction and analysis
tests need only torch / numpy / pandas. The scorer and launcher tests need BIT's environment and are skipped without
peft.

    <bit env>/python -m pytest tests/test_probes_foils.py

- Probes: a linear probe reads planted linear information and finds none in noise (information ~0 bits, accuracy at
  the majority class); an MLP probe reads an XOR code a linear probe cannot.
- RSA: BIT's 10-segment pooling; equal RDMs give r = 1 with a small Mantel p; a model RDM related to the neural one
  only through a shared phonetic factor gives a positive plain r and a partial r near 0.
- Foil rules, on the real configs/foils.yaml tables with a small fake pronunciation lexicon:
  - every type and variant;
  - determinism;
  - the exclusions: homophones and inflections as sound-alikes, compound numbers, adverbial "first", "new <name>",
    "they'll not", "you've really";
  - YAML booleans refused.
- Scorers, on a tiny Qwen2 in BCI_TextLLM / BCI_DiffusionLLM with a word-level tokenizer, with both projector
  shapes (BIT's released Linear and the host's MLP, configs/bit_baseline.yaml):
  - `loglik` equals explicit teacher forcing on the merged [prompt, neural tokens, prompt, sentence] embeddings, the
    batch equals lone rows, and prompt-only equals the LLM on the text alone;
  - `aligner` equals BIT's ModalityAligner on the lone trial;
  - `denoise`: the per-row CE sums to block_diffusion_ce's with the same masks, is deterministic per seed, and is
    the same for the same sentence in any row (common random numbers).
- Forced choice and type effects recover planted accuracies and effects.
- scripts/run_probes_foils.py end to end on a synthetic world:
  - every source kind (input, random, ssl, A_ctc, C_stage1, an AR, a diffusion and a bottleneck Stage 2 run);
  - every part, the summary;
  - a foil set built by scripts/build_foils.py.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))

from bit_experiments import foils as fo  # noqa: E402
from bit_experiments import probes  # noqa: E402

needs_bit = pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="needs BIT's environment (peft)")
TABLES = fo.check_tables(yaml.safe_load((ROOT / "configs" / "foils.yaml").read_text()))


# ------------------------------------------------------------------ probes

def test_linear_probe_reads_linear_information_and_none_in_noise():
    g = np.random.default_rng(0)
    y = g.integers(0, 4, 3000)
    x = g.normal(0, 1, (3000, 10))
    x[:, 0] += 2.5 * (y == 1) - 2.5 * (y == 2)
    x[:, 1] += 2.5 * (y == 3)
    trials = np.arange(3000) // 30
    p = probes.fit_linear_probe(x[:2000], y[:2000], x[2000:2500], y[2000:2500], 4)
    r = probes.score_probe(p, x[2500:], y[2500:], trials[2500:], n_boot=200)
    assert r["accuracy"] > 0.75 and r["info_bits"] > 0.5 and r["accuracy_ci"][0] < r["accuracy"] < r["accuracy_ci"][1]
    noise = g.normal(0, 1, (3000, 10))
    p0 = probes.fit_linear_probe(noise[:2000], y[:2000], noise[2000:2500], y[2000:2500], 4)
    r0 = probes.score_probe(p0, noise[2500:], y[2500:])
    assert abs(r0["accuracy"] - r0["majority"]) < 0.06 and r0["info_bits"] < 0.02
    assert {g["l2"] for g in p["grid"]} == {1e-5, 1e-4, 1e-3, 1e-2}


def test_mlp_probe_reads_an_xor_code_a_linear_probe_cannot():
    g = np.random.default_rng(1)
    s = g.choice([-1, 1], (4000, 2))
    x = np.c_[s + g.normal(0, 0.2, s.shape), g.normal(0, 1, (4000, 3))]
    y = (s[:, 0] * s[:, 1] > 0).astype(int)
    lin = probes.score_probe(probes.fit_linear_probe(x[:3000], y[:3000], x[3000:3500], y[3000:3500], 2), x[3500:],
                             y[3500:])
    mlp = probes.score_probe(probes.fit_mlp_probe(x[:3000], y[:3000], x[3000:3500], y[3000:3500], 2, seed=0),
                             x[3500:], y[3500:])
    assert lin["accuracy"] < 0.6 and mlp["accuracy"] > 0.95


# ------------------------------------------------------------------ RSA

def test_segment_pool_is_bits_concatenated_segment_means():
    x = np.arange(23 * 2, dtype=float).reshape(23, 2)
    v = probes.segment_pool(x, 10)
    parts = np.array_split(x, 10)
    assert v.shape == (20,) and np.allclose(v[:2], parts[0].mean(0)) and np.allclose(v[-2:], parts[-1].mean(0))
    with pytest.raises(ValueError):
        probes.segment_pool(x[:9], 10)


def test_rsa_plain_partial_and_mantel():
    g = np.random.default_rng(0)
    n = 60
    ph = probes.cosine_rdm(g.normal(0, 1, (n, 5)))
    a, b = g.normal(0, 0.15, (n, n)), g.normal(0, 0.15, (n, n))
    neural, model = ph + (a + a.T) / 2, ph + (b + b.T) / 2
    np.fill_diagonal(neural, 0), np.fill_diagonal(model, 0)
    same = probes.rsa(neural, neural, n_perm=200, n_boot=50)
    assert same["r"] == pytest.approx(1) and same["p_mantel"] < 0.01
    plain = probes.rsa(neural, model, n_perm=200, n_boot=50)
    partial = probes.rsa(neural, model, [ph], n_perm=200, n_boot=50)
    assert plain["r"] > 0.5 and plain["p_mantel"] < 0.01 and abs(partial["r"]) < 0.1 and partial["p_mantel"] > 0.01
    assert plain["ci"][0] < plain["r"] < plain["ci"][1]
    unrelated = probes.rsa(neural, probes.cosine_rdm(g.normal(0, 1, (n, 5))), n_perm=200, n_boot=0)
    assert unrelated["p_mantel"] > 0.01 and "ci" not in unrelated


def test_sequence_and_difference_rdms():
    m = probes.sequence_rdm([[1, 2, 3], [1, 2, 4], [5]])
    assert m[0, 1] == pytest.approx(1 / 3) and m[0, 2] == 1 and m[1, 0] == m[0, 1] and m[0, 0] == 0
    assert probes.abs_diff_rdm([2, 4])[0, 1] == 0.5


# ------------------------------------------------------------------ foil construction

CMU = {"cat": [["K", "AE1", "T"]], "cut": [["K", "AH1", "T"]], "cats": [["K", "AE1", "T", "S"]],
       "kat": [["K", "AE1", "T"]], "bat": [["B", "AE1", "T"]], "sat": [["S", "AE1", "T"]], "sit": [["S", "IH1", "T"]],
       "the": [["DH", "AH0"]], "big": [["B", "IH1", "G"]], "large": [["L", "AA1", "R", "JH"]],
       "dog": [["D", "AO1", "G"]], "dig": [["D", "IH1", "G"]], "is": [["IH1", "Z"]], "not": [["N", "AA1", "T"]],
       "here": [["HH", "IY1", "R"]], "two": [["T", "UW1"]], "three": [["TH", "R", "IY1"]], "four": [["F", "AO1", "R"]],
       "thousand": [["TH", "AW1", "Z", "AH0", "N", "D"]], "first": [["F", "ER1", "S", "T"]],
       "third": [["TH", "ER1", "D"]], "england": [["IH1", "NG", "G", "L", "AH0", "N", "D"]],
       "texas": [["T", "EH1", "K", "S", "AH0", "S"]], "ohio": [["OW0", "HH", "AY1", "OW0"]],
       "kansas": [["K", "AE1", "N", "Z", "AH0", "S"]], "new": [["N", "UW1"]], "visit": [["V", "IH1", "Z", "AH0", "T"]],
       "do": [["D", "UW1"]], "they": [["DH", "EY1"]], "will": [["W", "IH1", "L"]], "go": [["G", "OW1"]]}
ZIPF = {"cut": 5.0, "bat": 4.0, "sit": 5.2, "dig": 4.1, "cats": 4.5, "kat": 3.5}


RealLexicon = fo.Lexicon


def fake_lexicon():
    return RealLexicon(CMU, lambda w: ZIPF.get(w, 4.0), min_zipf=3.0, blocklist=TABLES["blocklist"],
                      stopwords=TABLES["stopwords"])


def _rule(fn, sentence, seed=0, lex=None):
    out = fn(sentence.split(), TABLES, np.random.default_rng(seed), lex)
    return None if out is None else (" ".join(out[0]), out[1])


def test_foil_rules():
    lex = fake_lexicon()
    # sound-alike: distance-1 neighbours, most frequent first; not the homophone "kat", not the inflection "cats"
    assert lex.neighbours("cat") == ["cut", "bat", "sat"]
    assert _rule(fo.foil_sound_alike, "the cat", lex=lex)[0] == "the cut"
    assert _rule(fo.foil_synonym, "the big dog")[0] == "the large dog"
    assert _rule(fo.foil_synonym, "the dog") is None
    assert _rule(fo.foil_contraction, "they will go")[0] == "they'll go"
    assert _rule(fo.foil_contraction, "they will not go")[0] == "they won't go"  # never "they'll not"
    assert _rule(fo.foil_contraction, "you have really strong views") is None  # never "you've really"
    assert _rule(fo.foil_contraction, "i have got it")[0] == "i've got it"
    assert _rule(fo.foil_contraction, "it's been fun")[0] == "it has been fun"
    assert _rule(fo.foil_contraction, "it's fun")[0] == "it is fun"
    assert _rule(fo.foil_negation, "i don't know") == ("i do know", {"variant": "remove", "position": 1,
                                                                   "from": "don't", "to": "do"})
    assert _rule(fo.foil_negation, "the dog is here")[0] == "the dog is not here"
    assert _rule(fo.foil_negation, "dogs bark") is None
    assert _rule(fo.foil_order_swap, "the cat")[0] == "cat the"
    assert _rule(fo.foil_order_swap, "go go") is None
    assert _rule(fo.foil_number_name, "two thousand dogs", lex=lex) is None  # compound number
    assert _rule(fo.foil_number_name, "do it first", lex=lex) is None  # adverbial "first"
    assert _rule(fo.foil_number_name, "the first visit", lex=lex)[0] == "the third visit"
    assert _rule(fo.foil_number_name, "visit new england", lex=lex) is None  # "new england" is not "new japan"
    state = _rule(fo.foil_number_name, "visit texas", lex=lex)
    assert state == ("visit kansas", {"variant": "name:state", "position": 1, "from": "texas", "to": "kansas"})
    assert _rule(fo.foil_number_name, "two dogs", lex=lex)[0] in ("three dogs", "four dogs")  # same syllable count


def test_build_is_deterministic_and_tables_are_checked():
    lex = fake_lexicon()
    sents = ["the big cat is here", "they will not go", "i don't know", "two thousand dogs", "two dogs"]
    a, b = fo.build(sents, "test", TABLES, lex), fo.build(sents, "test", TABLES, lex)
    assert a == b and {r["type"] for r in a} == set(fo.TYPES)
    assert not [r for r in a if r["trial"] == 3 and r["type"] == "number_name"]  # compound number left alone
    assert all(r["foil"] != r["reference"] for r in a) and all(set(r) == set(fo.COLUMNS) for r in a)
    first = [r for r in a if r["trial"] == 0]
    assert fo.build(sents[:1], "test", TABLES, lex) == first  # adding trials changes no other trial's foils
    with pytest.raises(ValueError, match="non-string"):
        fo.check_tables(TABLES | {"synonyms": {"yeah": True}})


# ------------------------------------------------------------------ forced choice and type effects

def _synthetic_scores(seed=0):
    g = np.random.default_rng(seed)
    rows = []
    effect = {"sound_alike": 0.0, "paraphrase": -1.0, "negation": 2.0}
    for trial in range(300):
        for kind, eff in effect.items():
            d = g.uniform(0, 0.5)
            prior = g.normal(0, 1)
            for cond, margin in (("real", prior + 4 * d + eff + 0.5), ("null_shuffled", prior)):
                rows.append({"split": "test", "trial": trial, "type": kind, "variant": "v", "foil": f"{kind}{trial}",
                             "phon_dist": d, "condition": cond, "score_true": margin, "score_foil": 0.0})
    return pd.DataFrame(rows)


def test_forced_choice_and_type_effects_recover_planted_values():
    s = _synthetic_scores()
    fc = fo.forced_choice(s, n_boot=200)
    assert set(fc) == {"negation", "paraphrase", "sound_alike", "negation/v", "paraphrase/v", "sound_alike/v"}
    neg = fc["negation"]
    assert neg["acc_real"] > neg["acc_null_shuffled"] and neg["n"] == 300
    lo, hi = neg["margin_real_minus_null_shuffled_ci"]
    assert lo < neg["margin_real_minus_null_shuffled"] < hi
    te = fo.type_effects(s, n_boot=200)["coef"]
    assert te["phon_dist"]["est"] == pytest.approx(4, abs=1e-6)
    assert te["negation_vs_sound_alike"]["est"] == pytest.approx(2, abs=1e-6)
    assert te["paraphrase_vs_sound_alike"]["est"] == pytest.approx(-1, abs=1e-6)
    ties = s.assign(score_true=0.0)
    assert fo.forced_choice(ties, n_boot=10)["negation"]["acc_real"] == 0.5


# ------------------------------------------------------------------ scorers on a tiny model (BIT's environment)

V, BOS, EOS, D, TINY_MASK = 96, 80, 81, 64, 95


def tiny_tokenizer(words, path=None):
    """Word-level fast tokenizer: words 0..79, <|endoftext|> = 80 (bos), <|im_end|> = 81 (eos), [UNK] = 79."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(sorted(set(words))[:78])}
    vocab["[UNK]"] = 79
    vocab["<|endoftext|>"], vocab["<|im_end|>"] = BOS, EOS
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="<|im_end|>",
                                   bos_token="<|endoftext|>", pad_token="<|endoftext|>")
    fast.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|endoftext|>"]})
    if path is not None:
        fast.save_pretrained(path)
    return fast


def tiny_qwen2(path=None, seed=0, layers=2, hidden=D):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=hidden, intermediate_size=2 * hidden,
                                       num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=1024, rope_theta=1e6, bos_token_id=BOS,
                                       eos_token_id=EOS, tie_word_embeddings=True, use_sliding_window=False))
    with torch.no_grad():
        for n, p in llm.named_parameters():
            p.uniform_(0.5, 1.5) if "norm" in n else p.normal_(0, 0.1)
    if path is not None:
        llm.save_pretrained(path)
    return llm


WORDS = "the a cat sat dog ran far it is good we like this no way big large cut bat not here do don't know".split()


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


@pytest.fixture(params=[None, 1024], ids=["linear-projector", "mlp-projector"])
def world(request, tmp_path, monkeypatch, bit_trainer):
    """A tiny repository root. Projector: BIT's released one Linear (inter_size None), or the host's MLP
    (configs/bit_baseline.yaml inter_size 1024)."""
    from bit_experiments import neural_ablations as na
    from bit_experiments import stage1, stage2, text_llm

    reg = yaml.safe_load(text_llm.REGISTRY.read_text())
    snap = tmp_path / reg["qwen2.5-1.5b-instruct"]["local_dir"]
    tiny_qwen2(snap)
    tok = tiny_tokenizer(WORDS, snap)
    monkeypatch.setattr(text_llm, "ROOT", tmp_path)
    monkeypatch.setattr(na, "ROOT", tmp_path)
    real_entry = text_llm.llm_entry
    monkeypatch.setattr(text_llm, "llm_entry", lambda name: real_entry(name) | (
        {"mask_id": TINY_MASK} if "diffusion" in name else {}))
    real_config = stage1.bit_sentence_config

    def config_fn():
        c = real_config()
        c["model"]["projector"]["embed_size"] = D // 2  # BIT's contrastive head is sized hidden / 2
        c["model"]["projector"]["inter_size"] = request.param
        return c

    monkeypatch.setattr(stage1, "bit_sentence_config", config_fn)

    def build(seed=0, llm="qwen2.5-1.5b-instruct"):
        config = stage1.bit_sentence_config()
        text_llm.use_llm(config, llm)
        with stage1.in_bit_dir():
            torch.manual_seed(seed)
            model = bit_trainer.NAME2MODEL[config.model.model_class](
                config.model, **config.method.model_kwargs, **config.method.dataset_kwargs, features="all")
        stage2.fix_padding(model)
        return model.eval(), config

    return types.SimpleNamespace(root=tmp_path, build=build, tok=tok, monkeypatch=monkeypatch, inter_size=request.param)


PROMPT = "a#this is"


def _explicit_loglik(model, spikes, text, tok, prompt):
    """Teacher forcing by hand: LLM over [prompt_a, neural tokens, prompt_b, sentence + EOS] embeddings."""
    a, b = tok(prompt.split("#")[0])["input_ids"], tok(prompt.split("#")[1])["input_ids"]
    s = tok(text + tok.eos_token)["input_ids"]
    x = torch.as_tensor(spikes)[None]
    h, m = model.prepare_neural_embeds(x, torch.ones(1, x.shape[1], dtype=torch.long), torch.arange(x.shape[1])[None],
                                       torch.zeros(1, dtype=torch.long))
    emb = model.llm.get_input_embeddings()
    seq = torch.cat([emb(torch.tensor(a)), model.projector(h)[0], emb(torch.tensor(b + s))])
    logits = model.llm(inputs_embeds=seq[None]).logits[0].float()
    logp = torch.log_softmax(logits[:-1], -1)
    start = seq.shape[0] - len(s)
    return float(sum(logp[start - 1 + i, t] for i, t in enumerate(s)))


@needs_bit
def test_ar_loglik_is_teacher_forcing_and_prompt_only_is_the_text_prior(world):
    model, _ = world.build()
    spikes = np.random.default_rng(0).normal(0, 1, (63, 512)).astype(np.float32)
    texts = ["the cat sat", "a big dog ran far", "no"]
    batch = fo.ar_loglik(model, spikes, texts, world.tok, PROMPT)
    lone = [fo.ar_loglik(model, spikes, [t], world.tok, PROMPT)[0] for t in texts]
    explicit = [_explicit_loglik(model, spikes, t, world.tok, PROMPT) for t in texts]
    assert np.allclose(batch, lone, atol=1e-4) and np.allclose(batch, explicit, atol=1e-4)
    po = fo.ar_loglik(model, spikes, texts, world.tok, PROMPT, prompt_only=True)
    ids = world.tok("a this is the cat sat" + world.tok.eos_token)["input_ids"]
    logp = torch.log_softmax(model.llm(input_ids=torch.tensor([ids])).logits[0, :-1].float(), -1)
    n_prompt = len(world.tok("a")["input_ids"]) + len(world.tok("this is")["input_ids"])
    assert po[0] == pytest.approx(float(sum(logp[i - 1, ids[i]] for i in range(n_prompt, len(ids)))), abs=1e-4)
    assert not hasattr(model.__dict__, "prepare_embeds") and "prepare_embeds" not in model.__dict__


@needs_bit
def test_aligner_cosine_is_bits_modality_aligner(world):
    model, _ = world.build()
    spikes = np.random.default_rng(1).normal(0, 1, (41, 512)).astype(np.float32)
    texts = ["the cat sat", "we like this"]
    cos = fo.aligner_cosine(model, spikes, texts, world.tok, PROMPT)
    for i, t in enumerate(texts):
        enc = fo.encode_texts([t], world.tok, PROMPT)
        nin = fo.neural_inputs(spikes, 1)
        sp, te, _, _ = model.prepare_embeds(enc["input_ids"], nin["spikes"], nin["spikes_mask"], nin["spikes_timestamp"],
                                            nin["dataset_name"])
        z_s, z_t = model.modality_aligner(sp, te)
        assert cos[i] == pytest.approx(float(z_s[0] @ z_t[0]), abs=1e-5)


@needs_bit
def test_denoise_matches_block_diffusion_ce_and_uses_common_random_numbers(world):
    from bit_experiments import diffusion_llm

    model, _ = world.build(llm="qwen2.5-1.5b-instruct-diffusion")
    spikes = np.random.default_rng(2).normal(0, 1, (77, 512)).astype(np.float32)
    texts = ["the cat sat", "a big dog ran far it is good", "the cat sat"]
    # the captured prefix / targets, scored with block_diffusion_ce's own draws from the global RNG
    captured = {}

    def capture(clean, labels):
        captured.update(clean=clean, labels=labels)
        raise fo._Captured

    enc = fo.encode_texts(texts, world.tok, PROMPT)
    inputs = {k: v for k, v in enc.items() if k != "n_prompt"} | fo.neural_inputs(spikes, len(texts))
    model.denoising_ce = capture
    with pytest.raises(fo._Captured):
        model(**inputs)
    del model.denoising_ce
    clean, labels = captured["clean"], captured["labels"]
    b, length = labels.shape
    torch.manual_seed(7)
    ref = diffusion_llm.block_diffusion_ce(*model._parts(), clean, labels, model.block_size, model.mask_id)[0]
    torch.manual_seed(7)
    n_blocks = length // model.block_size
    ratio = ((1 - diffusion_llm.EPS) * torch.rand(b * n_blocks) + diffusion_llm.EPS)[:, None].repeat(1, model.block_size)
    masked = (torch.rand(b * n_blocks, model.block_size) < ratio).reshape(b, length)
    rows = fo.rowwise_block_ce(*model._parts(), clean, labels, model.mask_id, masked)
    assert float(rows.sum()) == pytest.approx(float(ref), rel=1e-5)
    s1, se1 = fo.diffusion_denoise(model, spikes, texts, world.tok, PROMPT, n_draws=4, seed=3)
    s2, _ = fo.diffusion_denoise(model, spikes, texts, world.tok, PROMPT, n_draws=4, seed=3)
    s3, _ = fo.diffusion_denoise(model, spikes, texts, world.tok, PROMPT, n_draws=4, seed=4)
    assert np.array_equal(s1, s2) and not np.array_equal(s1, s3) and (se1 > 0).all()
    assert s1[0] == pytest.approx(s1[2], abs=1e-5)  # same sentence, same masks, any row
    lone = fo.diffusion_denoise(model, spikes, texts[1:2], world.tok, PROMPT, n_draws=4, seed=3)[0]
    assert lone[0] == pytest.approx(s1[1], abs=1e-4)  # batch row = lone trial
    assert "denoising_ce" not in model.__dict__


# ------------------------------------------------------------------ the launcher end to end

def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_bit
def test_run_probes_foils_end_to_end(world, monkeypatch):
    from safetensors.torch import save_file

    from bit_experiments import neural_ablations as na
    from bit_experiments import phoneme_frames as pf
    from bit_experiments import stage1, stage2, text_targets

    root = world.root
    run_pf, prep, build_foils = _script("run_probes_foils"), _script("prepare_ablations"), _script("build_foils")
    monkeypatch.setattr(run_pf, "ROOT", root)
    config = stage1.bit_sentence_config()
    g = np.random.default_rng(0)
    sents = ["the cat sat", "a dog ran far", "it is good", "we like this", "no way", "the big dog is here",
             "i don't know", "two cats"]
    ds = {s: [{"spikes": g.normal(0, 1, (int(g.integers(90, 160)), 512)).astype(np.float32),
               "sentence": sents[(i + k) % len(sents)]} for i in range(n)]
          for k, (s, n) in enumerate((("train", 16), ("val", 8), ("test", 8)))}
    monkeypatch.setattr(pf, "load_t15", lambda config, limit=None: ds)

    # sources
    def save_enc(enc, d, readout=None):
        d.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), d / "encoder.bin")
        if readout is not None:
            torch.save({"read_out.readout_dict.0.weight": readout[0], "read_out.readout_dict.0.bias": readout[1]},
                       d / "decoder.bin")
        return d

    torch.manual_seed(11)
    save_enc(pf.build_ndt_encoder(config, "tx1"), root / "checkpoints" / "ssl_human" / "BEST")
    a_dir = save_enc(pf.build_ndt_encoder(config), root / "ckpt" / "armA", (torch.randn(41, 384), torch.randn(41)))
    lm_t = tiny_qwen2(layers=28, hidden=32, seed=5)  # stands in for Aero's LM (fixed targets: layers up to 28)
    tok_t = tiny_tokenizer(WORDS)
    tok_t.padding_side = "right"
    monkeypatch.setattr(run_pf, "aero_lm", lambda config, device: (lm_t, tok_t))
    c_dir = save_enc(pf.build_ndt_encoder(config), root / "ckpt" / "C")
    proj = stage1.build_projector(config, 384, 32)
    assert isinstance(proj, torch.nn.Sequential) == (world.inter_size is not None)
    torch.save(proj.state_dict(), c_dir / "projector.bin")
    head = stage1.AlignHead(32, 32, 8, "infonce")
    torch.save({"state": head.state_dict(), "epoch": 3, "config": {"head": {"proj_dim": 8, "init_temp": 0.1,
                                                                            "max_scale": 100.0}, "loss": "infonce",
                                                                   "targets": {"name": "layer14"}}},
               c_dir / "align_head.pt")

    # artifacts: labels (prepare_ablations), bottleneck stats, foils (build_foils), text targets
    labels_path, _ = prep.make_labels(config, ds, a_dir, root / "data" / "processed" / "phoneme_frames", "cpu", 8,
                                      sequences=lambda ss: [[(sum(map(ord, w)) % 39) + 1 for w in s.split()] + [40]
                                                            for s in ss])
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "ablations.yaml").write_text(yaml.safe_dump({"phoneme-bottleneck": {
        "kind": "phoneme_bottleneck", "phoneme_model": "ckpt/armA", "feature": "log_posterior", "floor": 1e-8,
        "stats": "abl/stats.pt"}}))
    monkeypatch.setattr(na, "REGISTRY", root / "configs" / "ablations.yaml")
    prep.make_bottleneck_stats(config, ds, na.entry("phoneme-bottleneck"), "cpu", 8)
    refs = root / "refs"
    refs.mkdir()
    for s in ("val", "test"):
        pd.DataFrame({"trial": range(len(ds[s])), "reference": [r["sentence"] for r in ds[s]]}).to_csv(
            refs / f"{s}_greedy_real.csv", index=False)
    monkeypatch.setattr(fo, "Lexicon", lambda **kw: fake_lexicon())
    monkeypatch.setattr(sys, "argv", ["build_foils.py", "--refs", str(refs), "--out",
                                      str(root / "data" / "processed" / "foils")])
    assert build_foils.main() == 0
    every = sorted({r["sentence"] for s in ds.values() for r in s})
    with torch.no_grad():
        arrays = text_targets.compute_targets(lm_t, tok_t, every)
    text_targets.save(root / "data" / "processed" / "text_targets", every, arrays, {"teacher": "tiny"})

    # Stage 2 runs: AR, diffusion, bottleneck
    def stage2_run(name, llm, ablation=None):
        model, _ = world.build(seed=1, llm=llm)
        if ablation:
            stage2.load_init(model, a_dir)
            na.apply(model, ablation, a_dir, seed=1)
        best = root / "checkpoints" / "stage2" / name / "BEST"
        best.mkdir(parents=True)
        model.neural_encoder.save_checkpoint(str(best))
        torch.save(model.projector.state_dict(), best / "projector.bin")
        save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(model).items()}, str(best / stage2.SLIM))
        res = root / "results" / "stage2" / name
        res.mkdir(parents=True)
        (res / "run.json").write_text(json.dumps({"args": {"llm": llm, "prompt": PROMPT, "ablation": ablation,
                                                           "seed": 42, "arm": "B"}}))

    stage2_run("B_ar", "qwen2.5-1.5b-instruct")
    other = json.loads((root / "results" / "stage2" / "B_ar" / "run.json").read_text())  # an eval-only run elsewhere
    other["args"]["eval_only"] = "/elsewhere/copy/checkpoints/stage2/B_ar/BEST"
    assert run_pf.stage2_checkpoint("B_x", other) == root / "checkpoints" / "stage2" / "B_ar" / "BEST"
    stage2_run("B_diff", "qwen2.5-1.5b-instruct-diffusion")
    stage2_run("A_bottleneck", "qwen2.5-1.5b-instruct", "phoneme-bottleneck")

    cfg = yaml.safe_load((ROOT / "configs" / "probes.yaml").read_text())
    cfg["sources"]["A_ctc"]["dir"], cfg["sources"]["C_stage1"]["dir"] = "ckpt/armA", "ckpt/C"
    cfg["probe"] |= {"max_iter": 20, "n_boot": 20}
    cfg["rsa"] |= {"patch_range": [20, 26], "n_perm": 20, "n_boot": 5, "targets": ["emb", "layer14"]}
    cfg["foils"] |= {"diffusion_draws": 2, "n_boot": 20}
    (root / "probes.yaml").write_text(yaml.safe_dump(cfg))
    runs = ["input", "random", "ssl", "A_ctc", "C_stage1", "B_ar", "B_diff", "A_bottleneck"]
    out = root / "results" / "probes"
    monkeypatch.setattr(sys, "argv", ["run_probes_foils.py", "--runs", *runs, "--config", str(root / "probes.yaml"),
                                      "--device", "cpu", "--out", str(out)])
    assert run_pf.main() == 0
    for r in runs:
        p = json.loads((out / r / "probe.json").read_text())
        reps = [k for k in p if k != "source"]
        assert reps and all(0 <= p[k]["test"]["accuracy"] <= 1 for k in reps)
        assert json.loads((out / r / "rsa.json").read_text())
    assert set(json.loads((out / "A_ctc" / "probe.json").read_text())) == {"source", "encoder", "posterior"}
    assert set(json.loads((out / "A_bottleneck" / "probe.json").read_text())) == {"source", "encoder", "ablated",
                                                                                  "projector"}
    assert json.loads((out / "input" / "probe.json").read_text())["input"]["dim"] == 2560
    assert json.loads((out / "C_stage1" / "probe.json").read_text())["projector"]["dim"] == 32
    assert json.loads((out / "A_bottleneck" / "probe.json").read_text())["ablated"]["dim"] == 41
    for r, scorers in (("B_ar", {"loglik", "aligner"}), ("B_diff", {"denoise", "aligner"}),
                       ("A_bottleneck", {"loglik", "aligner"}), ("C_stage1", {"stage1"})):
        sc = pd.read_csv(out / r / "foil_scores.csv")
        assert set(sc.scorer) == scorers
        conds = set(sc[sc.scorer == next(iter(scorers - {"aligner"}))].condition)
        assert conds == ({"real", "null_shuffled", "null_prompt_only"} if r != "C_stage1" else {"real", "null_shuffled"})
        fj = json.loads((out / r / "foils.json").read_text())
        assert any(k.endswith("/test") for k in fj["results"]) and fj["foil_set"]
    assert not (out / "input" / "foils.json").exists()
    summary = (out / "SUMMARY.md").read_text()
    assert "| B_diff | denoise |" in summary and "| A_ctc | posterior |" in summary
    timing = json.loads((out / "B_diff" / "timing.json").read_text())
    assert {"extract_min", "probe_min", "rsa_min", "foils_min"} <= set(timing)
