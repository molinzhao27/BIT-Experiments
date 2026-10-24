"""Aero block diffusion, path 1 (docs/AERO_DIFFUSION.md): Aero-1-Audio's language model in run 3's block-diffusion
model, objective and sampler, CPU only. Needs BIT's environment (transformers 4.48.2, peft 0.11.1); skipped without
peft.

    <bit env>/python -m pytest tests/test_aero_diffusion.py

- Registry: the two new entries are the control's with Aero's pinned snapshot (BIT's), plain and fast_step; the
  earlier entries are untouched.
- Extraction: a tiny snapshot in Aero's own layout (model_type "aero", language_model.* + a separate equal lm_head,
  audio_tower.*, audio_modal_projector.*, bf16) loads into the vendored model with exactly the LM that BIT's own call
  (Qwen2AudioForConditionalGeneration.from_pretrained) gives, tied, and with one-token blocks the same causal forward.
  A snapshot whose lm_head differs from its embeddings is refused.
- Config: Aero's pinned text_config converts to the control's converted Instruct config except vocab_size.
- Full size (LLM on the meta device): trainable parameters are run 3's; only the frozen embedding has 64 more rows.
- The new entries end to end: registry, use_llm, BIT's model build on the tiny Aero snapshot, loss, backward (mask row
  frozen), greedy and nucleus decodes; fast_step gives the plain step's loss and every gradient (float64, dropout
  off). At one seed, the trainable weights start exactly as the Instruct control's.
- Path 2 tools: scripts/posttrain_edit.py post-trains an Aero base (option a); scripts/merge_aero_delta.py adds Aero's
  change from Instruct to a block-diffusion base (option d).
"""

import copy
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
peft = pytest.importorskip("peft")

from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration, Qwen2ForCausalLM  # noqa: E402

from bit_experiments import diffusion_llm, stage1, stage2, text_llm  # noqa: E402
from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM  # noqa: E402

NAME, FAST, CONTROL, RUN3 = ("aero-1-audio-diffusion", "aero-1-audio-diffusion-fast",
                             "qwen2.5-1.5b-instruct-diffusion", "fast-dllm-v2-1.5b")
# text_config of lmms-lab/Aero-1-Audio @ 2efba6e9 (config.json), and Qwen/Qwen2.5-1.5B-Instruct @ 989aa798's config.json
AERO_TEXT_CONFIG = {
    "_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0,
    "bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu", "hidden_size": 1536,
    "initializer_range": 0.02, "intermediate_size": 8960, "max_position_embeddings": 32768, "max_window_layers": 21,
    "model_type": "qwen2", "num_attention_heads": 12, "num_hidden_layers": 28, "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 1000000.0, "sliding_window": 32768,
    "tie_word_embeddings": True, "torch_dtype": "bfloat16", "use_cache": True, "use_sliding_window": False,
    "vocab_size": 152000}
INSTRUCT_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
    "hidden_act": "silu", "hidden_size": 1536, "initializer_range": 0.02, "intermediate_size": 8960,
    "max_position_embeddings": 32768, "max_window_layers": 21, "model_type": "qwen2", "num_attention_heads": 12,
    "num_hidden_layers": 28, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
    "sliding_window": 32768, "tie_word_embeddings": True, "torch_dtype": "bfloat16", "transformers_version": "4.43.1",
    "use_cache": True, "use_sliding_window": False, "vocab_size": 151936}
MASK_ID = 151665
V, BOS, EOS, D = 96, 80, 81, 64  # tiny vocabulary: 82..95 unused, like Aero's padded rows
TINY_MASK = 95
REGISTRY = yaml.safe_load((ROOT / "configs" / "llms.yaml").read_text())


@pytest.fixture
def bit_trainer(monkeypatch):
    """BIT's models.trainer for use_llm; a stand-in with its NAME2MODEL where its imports (wandb) are missing."""
    try:
        from models import trainer
    except ImportError:
        import models

        trainer = types.ModuleType("models.trainer")
        trainer.NAME2MODEL = {}
        monkeypatch.setitem(sys.modules, "models.trainer", trainer)
        monkeypatch.setattr(models, "trainer", trainer, raising=False)
    return trainer


def tiny_text_config():
    return {"_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
            "vocab_size": V, "hidden_size": D, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 1024, "rope_theta": 1e6,
            "bos_token_id": BOS, "eos_token_id": EOS, "tie_word_embeddings": True, "use_sliding_window": False,
            "sliding_window": 1024, "max_window_layers": 2, "rms_norm_eps": 1e-6, "hidden_act": "silu",
            "torch_dtype": "bfloat16"}


def tiny_audio_config():
    return {"model_type": "qwen2_audio_encoder", "num_mel_bins": 8, "encoder_layers": 1, "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32, "d_model": 16, "max_source_positions": 16, "activation_function": "gelu",
            "scale_embedding": False}


def write_aero_snapshot(path: Path, seed=0, head_differs=False, vocab=V):
    """A random tiny snapshot in Aero-1-Audio's own layout: config.json with model_type "aero" (a qwen2 text_config,
    a qwen2_audio_encoder audio_config), one bf16 model.safetensors holding language_model.* with a separate lm_head
    equal to the embeddings, audio_tower.* and audio_modal_projector.linear.*. Every weight random (norms and biases
    too). Returns the LM's state dict (bf16 values, Qwen2ForCausalLM names, lm_head included)."""
    torch.manual_seed(seed)
    text, audio = tiny_text_config() | {"vocab_size": vocab}, tiny_audio_config()
    model = Qwen2AudioForConditionalGeneration(Qwen2AudioConfig(text_config=copy.deepcopy(text),
                                                                audio_config=copy.deepcopy(audio)))
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.uniform_(0.5, 1.5) if "norm" in name else p.normal_(0, 0.1)
    state = {}
    for key, value in model.state_dict().items():
        key = key.replace("multi_modal_projector.", "audio_modal_projector.")  # Aero's name for its projector
        state[key] = value.to(torch.bfloat16).clone()  # separate tensors, as Aero ships lm_head and embed_tokens
    if head_differs:
        state["language_model.lm_head.weight"][0, 0] += 1
    path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps({
        "architectures": ["AeroForConditionalGeneration"], "audio_config": audio, "audio_token_index": V - 1,
        "auto_map": {"AutoConfig": "configuration_aero.AeroConfig",
                     "AutoModelForCausalLM": "modeling_aero.AeroForConditionalGeneration"},
        "model_type": "aero", "text_config": text, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "transformers_version": "4.52.0.dev0"}))
    prefix = diffusion_llm.AERO_LM_PREFIX
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


@pytest.fixture
def snapshot(tmp_path):
    """The tiny Aero snapshot at the entry's local_dir under tmp_path (= the repository root for use_llm)."""
    path = tmp_path / REGISTRY[NAME]["local_dir"]
    return write_aero_snapshot(path), path


# ------------------------------------------------------------------ registry


def test_registry_entries_are_the_controls_with_aeros_snapshot():
    bit = yaml.safe_load((ROOT / "configs" / "bit_baseline.yaml").read_text())["llm"]
    plain, fast, control = REGISTRY[NAME], REGISTRY[FAST], REGISTRY[CONTROL]
    assert {k: plain[k] for k in ("repo_id", "revision", "local_dir")} == bit  # BIT's pinned Aero snapshot
    assert plain["loader"] == control["loader"] == "diffusion" and plain["mask_id"] == control["mask_id"] == MASK_ID
    assert plain["sampler"] == control["sampler"] == REGISTRY[RUN3]["sampler"]
    assert fast == plain | {"sampler": plain["sampler"] | {"fast_step": True}}
    assert REGISTRY["fast-dllm-v2-1.5b-fast"]["sampler"] == REGISTRY[RUN3]["sampler"] | {"fast_step": True}


# ------------------------------------------------------------------ extraction


def test_aero_lm_loads_as_bits_own_aero_load(snapshot):
    lm_state, path = snapshot
    llm = diffusion_llm.BCI_DiffusionLLM.from_pretrained(path)
    assert isinstance(llm, Fast_dLLM_QwenForCausalLM) and llm.config.bd_size == 32
    assert llm.lm_head.weight is llm.model.embed_tokens.weight and llm.dtype == torch.float32
    bit = Qwen2AudioForConditionalGeneration.from_pretrained(path).language_model  # BIT's BCI_AudioLLM.load_llm call
    assert bit.lm_head.weight is bit.model.embed_tokens.weight  # BIT's Aero is tied too
    ours, theirs = llm.state_dict(), bit.state_dict()
    assert ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)
    assert all(torch.equal(ours[k], v.float()) for k, v in lm_state.items())  # the file's values, bf16 -> fp32 exact
    llm.eval(), bit.eval()
    ids = torch.randint(0, BOS, (2, 45), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        hidden = llm.model(inputs_embeds=llm.model.embed_tokens(ids), block_size=1).last_hidden_state  # causal
        ref = bit.model(input_ids=ids).last_hidden_state
        assert torch.allclose(hidden, ref, atol=1e-5) and torch.allclose(llm.lm_head(hidden), bit(input_ids=ids).logits,
                                                                         atol=1e-5)
        blocks = llm.model(inputs_embeds=llm.model.embed_tokens(ids), block_size=32).last_hidden_state
    assert (blocks - ref).abs().max() > 1e-2  # 32-token blocks are bidirectional: the check above is not vacuous


def test_aero_extraction_keeps_only_the_lm_and_refuses_an_untied_head(tmp_path):
    lm_state = write_aero_snapshot(tmp_path / "ok")
    text_config, state = diffusion_llm.aero_language_model(tmp_path / "ok")
    assert "_name_or_path" not in text_config and text_config["vocab_size"] == V
    assert set(state) == set(lm_state) - {"lm_head.weight"} and all(k.startswith("model.") for k in state)
    assert Qwen2ForCausalLM(diffusion_llm.qwen2_as_fast_dllm_config(text_config)).state_dict().keys() == \
        set(state) | {"lm_head.weight"}  # every LM tensor, no audio tower or projector
    write_aero_snapshot(tmp_path / "bad", head_differs=True)
    with pytest.raises(ValueError, match="lm_head differs"):
        diffusion_llm.aero_language_model(tmp_path / "bad")
    with pytest.raises(ValueError, match="expected an Aero-1-Audio snapshot"):
        Qwen2ForCausalLM(Qwen2AudioConfig(text_config=tiny_text_config()).text_config).save_pretrained(tmp_path / "q")
        diffusion_llm.aero_language_model(tmp_path / "q")


def test_aero_config_converts_to_the_controls_but_for_vocab_size():
    aero = diffusion_llm.qwen2_as_fast_dllm_config(
        {k: v for k, v in AERO_TEXT_CONFIG.items() if k != "_name_or_path"}).to_dict()
    control = diffusion_llm.qwen2_as_fast_dllm_config(INSTRUCT_CONFIG).to_dict()
    assert {k for k in aero.keys() | control.keys() if aero.get(k) != control.get(k)} == {"vocab_size"}
    assert aero["vocab_size"] == 152000 > MASK_ID and aero["bd_size"] == 32 and aero["tie_word_embeddings"]


# ------------------------------------------------------------------ full size


def test_full_size_trainable_parameters_are_run3s(monkeypatch, bit_trainer):
    def build(cfg):
        def meta_llm(path):
            with torch.device("meta"):
                return Fast_dLLM_QwenForCausalLM(cfg)

        monkeypatch.setattr(diffusion_llm.BCI_DiffusionLLM, "from_pretrained", staticmethod(meta_llm))
        config = stage1.bit_sentence_config()
        config["method"]["model_kwargs"]["llm_path"] = "unused"
        config["method"]["model_kwargs"]["diffusion_sampler"] = REGISTRY[NAME]["sampler"] | {"mask_id": MASK_ID}
        with stage1.in_bit_dir():
            return diffusion_llm.BCI_DiffusionLLM(config.model, **config.method.model_kwargs,
                                                  **config.method.dataset_kwargs, features="all")

    def trainable(model):
        return {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}

    text = {k: v for k, v in AERO_TEXT_CONFIG.items() if k != "_name_or_path"}
    aero = build(diffusion_llm.qwen2_as_fast_dllm_config(text))
    control = build(diffusion_llm.qwen2_as_fast_dllm_config(INSTRUCT_CONFIG))
    # run 3's Fast-dLLM v2 config differs from the converted Instruct one only in metadata and pad_token_id
    run3 = build(diffusion_llm.qwen2_as_fast_dllm_config(INSTRUCT_CONFIG | {"pad_token_id": 151645}))
    assert trainable(aero) == trainable(control) == trainable(run3)
    # The total depends on BIT's data config (the neural read-in), which the GPU host patches: 21,280,001 here,
    # 22,657,281 on the GPU host. The LoRA part follows from the LLM's and BIT's LoRA config alone.
    lora = stage1.bit_sentence_config()["method"]["model_kwargs"]["lora"]
    h, i, kv = text["hidden_size"], text["intermediate_size"], \
        text["num_key_value_heads"] * text["hidden_size"] // text["num_attention_heads"]
    dims = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h), "gate_proj": (h, i),
            "up_proj": (h, i), "down_proj": (i, h)}
    want = text["num_hidden_layers"] * lora["r"] * sum(sum(dims[m]) for m in lora["target_modules"])
    assert sum(n for k, n in trainable(aero).items() if k.startswith("llm.")) == want  # 9,232,384 at r 8
    frozen = sum(p.numel() for p in aero.llm.parameters() if not p.requires_grad)
    assert frozen == 1_543_714_304 + 64 * 1536  # Aero's embedding has 152,000 rows instead of 151,936
    base = aero.llm.get_base_model()
    assert not base.model.embed_tokens.weight.requires_grad and base.lm_head.weight is base.model.embed_tokens.weight


# ------------------------------------------------------------------ the new entries end to end


def batch():
    """Three trials: prompt halves around the neural tokens, sentences of 3, 20 and 1 tokens (+ EOS)."""
    b, t = 3, 60
    g = torch.Generator().manual_seed(0)
    prompt_a, prompt_b, sentences = [3, 4], [5, 6, 7], [[10, 11, 12], list(range(20, 40)), [50]]
    length = len(prompt_a) + len(prompt_b) + max(map(len, sentences)) + 1
    input_ids, targets = torch.full((b, length), EOS), torch.full((b, length), -100)
    attention = torch.zeros(b, length, dtype=torch.long)
    for i, s in enumerate(sentences):
        ids = prompt_a + prompt_b + s + [EOS]
        input_ids[i, : len(ids)], attention[i, : len(ids)] = torch.tensor(ids), 1
        targets[i, len(prompt_a) + len(prompt_b): len(ids)] = torch.tensor(s + [EOS])
    spikes, spikes_mask = torch.randn(b, t, 512, generator=g), torch.ones(b, t, dtype=torch.long)
    spikes[2, 40:], spikes_mask[2, 40:] = stage2.PAD_VALUE, 0
    inputs = {"input_ids": input_ids, "attention_mask": attention, "input_split": torch.full((b,), len(prompt_a)),
              "spikes": spikes, "spikes_mask": spikes_mask, "spikes_timestamp": torch.arange(t).expand(b, t),
              "spikes_lengths": torch.tensor([t, t, 40]), "dataset_name": torch.zeros(b, dtype=torch.long)}
    one = {k: v[:1] for k, v in inputs.items()} | {"input_ids": torch.tensor([prompt_a + prompt_b]),
                                                   "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    return inputs, targets, one, sentences


def build_entry(name, path, monkeypatch, bit_trainer, mask_id=TINY_MASK):
    monkeypatch.setattr(text_llm, "ROOT", path.parents[2])
    config = stage1.bit_sentence_config()
    entry = text_llm.use_llm(config, name)
    assert entry["path"] == str(path) and config["data"]["tokenizer_path"] == str(path)  # Aero's tokenizer, as BIT
    assert config["model"]["model_class"] == "BCI_DiffusionLLM"
    kwargs = config.method.model_kwargs
    assert kwargs["llm_path"] == str(path) and kwargs["diffusion_sampler"]["mask_id"] == MASK_ID
    config["model"]["projector"]["embed_size"] = D // 2  # the contrastive head expects hidden / 2
    kwargs["diffusion_sampler"]["mask_id"] = mask_id  # the registry's 151665 has no row in the tiny vocabulary
    with stage1.in_bit_dir():
        torch.manual_seed(0)
        model = bit_trainer.NAME2MODEL[config.model.model_class](config.model, **kwargs,
                                                                  **config.method.dataset_kwargs, features="all")
    stage2.fix_padding(model)
    return model


def test_aero_entry_builds_trains_and_decodes(snapshot, monkeypatch, bit_trainer):
    _, path = snapshot
    with pytest.raises(ValueError, match="no embedding row"):
        build_entry(NAME, path, monkeypatch, bit_trainer, mask_id=V)
    model = build_entry(NAME, path, monkeypatch, bit_trainer)
    base = model.llm.get_base_model()
    assert isinstance(base, Fast_dLLM_QwenForCausalLM) and model.block_size == 32 and model.mask_id == TINY_MASK
    assert not (diffusion_llm.FAST_STEP_DEFAULTS | model.sampler)["fast_step"]  # the matched entry: run 3's step
    mask_row = base.model.embed_tokens.weight[TINY_MASK].clone()
    inputs, targets, one, sentences = batch()

    model.train()
    torch.manual_seed(0)
    out = model(**inputs, targets=targets)
    assert torch.isfinite(out.loss) and out.preds.shape == (int(out.n_examples), V)
    assert int(out.n_examples) >= sum(len(s) + 1 for s in sentences)  # sentence + EOS, and the block's EOS padding
    out.loss.backward()
    lora_b = {n: p.grad for n, p in model.llm.named_parameters() if "lora_B" in n}
    assert len(lora_b) == 2 * 7 and all(g is not None and g.abs().sum() > 0 for g in lora_b.values())
    assert base.model.embed_tokens.weight.grad is None
    assert torch.equal(base.model.embed_tokens.weight[TINY_MASK], mask_row)
    assert all(p.grad is not None for n, p in model.named_parameters() if n.startswith("projector."))

    model.eval()
    for decoding in (stage2.DECODINGS["greedy"], stage2.DECODINGS["nucleus"]):
        torch.manual_seed(0)
        ids = model.generate(**one, **decoding)
        assert ids.shape[0] == 1 and 1 <= ids.shape[1] <= decoding["max_new_tokens"]
        assert TINY_MASK not in ids.tolist()[0]


def test_fast_entry_gives_the_plain_steps_loss_and_gradients(snapshot, monkeypatch, bit_trainer):
    _, path = snapshot
    inputs, targets, _, _ = batch()
    res = {}
    for name in (NAME, FAST):
        model = build_entry(name, path, monkeypatch, bit_trainer)
        with torch.no_grad():
            for n, p in model.llm.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)  # nonzero LoRA, the same in both builds (same seed, same draws)
        # float64: in fp32 the NDT encoder's query/key gradients (small, mostly cancelling) differ by up to ~3e-3 of
        # their max from summation order alone; in float64 every gradient agrees to ~1e-15
        model.double()
        model.eval()  # dropout off; denoising_ce runs the training layout regardless (diffusion_layout)
        torch.manual_seed(0)
        out = model(**{k: v.double() if v.is_floating_point() else v for k, v in inputs.items()}, targets=targets)
        out.loss.backward()
        res[name] = ((diffusion_llm.FAST_STEP_DEFAULTS | model.sampler)["fast_step"], out.loss.detach(),
                     int(out.n_examples), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    (fast_p, loss_p, n_p, g_p), (fast_f, loss_f, n_f, g_f) = res[NAME], res[FAST]
    assert not fast_p and fast_f
    assert n_p == n_f and torch.allclose(loss_p, loss_f, rtol=1e-12)
    assert g_p.keys() == g_f.keys() and any(n.startswith("neural_encoder.") for n in g_p)
    worst = max(float((g_p[n] - g_f[n]).abs().max() / g_p[n].abs().max().clamp_min(1e-30)) for n in g_p)
    assert worst < 1e-10, worst


def test_same_seed_gives_the_controls_initial_weights(tmp_path, monkeypatch, bit_trainer):
    """At one seed, path 1 and the Instruct control start from the same encoder, projector, contrastive head and LoRA:
    only the frozen LM differs (BIT's own Aero load draws extra random numbers; this one does not)."""
    aero_dir = tmp_path / REGISTRY[NAME]["local_dir"]
    write_aero_snapshot(aero_dir, seed=0)
    qwen = Qwen2ForCausalLM(Qwen2AudioConfig(text_config=tiny_text_config()).text_config)
    qwen.save_pretrained(tmp_path / REGISTRY[CONTROL]["local_dir"])
    aero = build_entry(NAME, aero_dir, monkeypatch, bit_trainer)
    control = build_entry(CONTROL, tmp_path / REGISTRY[CONTROL]["local_dir"], monkeypatch, bit_trainer)
    a = {n: p for n, p in aero.named_parameters() if p.requires_grad}
    c = {n: p for n, p in control.named_parameters() if p.requires_grad}
    assert a.keys() == c.keys() and any("lora_A" in n for n in a)
    assert all(torch.equal(a[n], c[n]) for n in a)
    assert not torch.equal(aero.llm.get_base_model().model.embed_tokens.weight,
                           control.llm.get_base_model().model.embed_tokens.weight)


# ------------------------------------------------------------------ path 2 tools


def load_script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_posttrain_edit_converts_an_aero_base(tmp_path, monkeypatch):
    """Option (a): scripts/posttrain_edit.py on Aero's LM (--mask-id, as the entry's mask_id) saves a block-diffusion
    snapshot that loads as Fast-dLLM v2 snapshots do and holds Aero's LM at step 0's weights plus training."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    posttrain = load_script("posttrain_edit")
    base = tmp_path / "aero"
    write_aero_snapshot(base)
    words = [a + b for a in "abcd" for b in "abcdefghijklmnopqrst"]  # 80 ordinary tokens, ids 0..79 (< BOS)
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token=words[0]))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(base)
    g = torch.Generator().manual_seed(0)
    (tmp_path / "docs.txt").write_text("\n".join(" ".join(words[int(i)] for i in torch.randint(0, 80, (8,), generator=g))
                                                 for _ in range(200)))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "posttrain_edit.py", "--base", str(base), "--mask-id", str(TINY_MASK), "--edit-rate", "0", "--text-file",
        str(tmp_path / "docs.txt"), "--out", str(out), "--tokens", "1024", "--save-at", "--seq-len", "64",
        "--batch-rows", "4", "--micro-rows", "2", "--heldout-rows", "2", "--eval-every", "2", "--device", "cpu"])
    assert posttrain.main() == 0
    run = json.loads((out / "posttrain.json").read_text())
    assert run["mask_id"] == TINY_MASK and run["saved"] == [1024]
    snap = diffusion_llm.BCI_DiffusionLLM.from_pretrained(out / "tokens_1024")
    assert isinstance(snap, Fast_dLLM_QwenForCausalLM) and snap.config.bd_size == 32
    assert snap.lm_head.weight is snap.model.embed_tokens.weight
    start = diffusion_llm.BCI_DiffusionLLM.from_pretrained(base).state_dict()
    trained = snap.state_dict()
    assert trained.keys() == start.keys() and not all(torch.equal(trained[k], start[k]) for k in start)


def test_merge_adds_aeros_change_to_the_base(tmp_path):
    """Option (d): scripts/merge_aero_delta.py gives W_base + alpha (W_aero - W_instruct) on every LM tensor, keeps the
    base's mask row, drops Aero's extra embedding rows, and saves a snapshot that loads as the base does."""
    from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig

    merge = load_script("merge_aero_delta")
    text = tiny_text_config()
    torch.manual_seed(1)
    base = Fast_dLLM_QwenForCausalLM(Fast_dLLM_QwenConfig(**{k: v for k, v in text.items() if k not in (
        "_name_or_path", "architectures", "model_type")}, bd_size=32))
    base.save_pretrained(tmp_path / "base")
    (tmp_path / "base" / "added_tokens.json").write_text(json.dumps({"|<MASK>|": TINY_MASK}))
    torch.manual_seed(2)
    Qwen2ForCausalLM(Qwen2AudioConfig(text_config=text).text_config).save_pretrained(tmp_path / "instruct")
    donor = write_aero_snapshot(tmp_path / "aero", seed=3, vocab=V + 4)  # Aero's embedding has extra rows
    args = ["--base", str(tmp_path / "base"), "--donor", str(tmp_path / "aero"), "--reference", str(tmp_path / "instruct")]
    for alpha in (1.0, 0.0):
        out = tmp_path / f"merged_{alpha}"
        assert merge.main(args + ["--out", str(out), "--alpha", str(alpha)]) == 0
        got = diffusion_llm.BCI_DiffusionLLM.from_pretrained(out).state_dict()
        b = diffusion_llm.BCI_DiffusionLLM.from_pretrained(tmp_path / "base").state_dict()
        r = Qwen2ForCausalLM.from_pretrained(tmp_path / "instruct").state_dict()
        assert got.keys() == b.keys()
        for k in b:
            if k == "lm_head.weight":
                continue
            d = donor[k].float()[: b[k].shape[0]] if k == merge.EMBED else donor[k].float()
            want = b[k] + alpha * (d - r[k])
            if k == merge.EMBED:
                want[TINY_MASK] = b[k][TINY_MASK]
            assert torch.allclose(got[k], want, atol=1e-6), k
        assert torch.equal(got["lm_head.weight"], got[merge.EMBED])
        meta = json.loads((out / "merge.json").read_text())
        assert meta["kept_rows"] == [TINY_MASK] and meta["summary"]["tensors"] == len(b) - 1
        assert (out / "added_tokens.json").exists()
    assert all(torch.equal(got[k], b[k]) for k in b)  # alpha 0: the base itself
