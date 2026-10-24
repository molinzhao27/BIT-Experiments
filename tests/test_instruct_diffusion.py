"""Instruct control (docs/INSTRUCT_DIFFUSION_CONTROL.md): Qwen2.5-1.5B-Instruct's own weights in run 3's
block-diffusion model, objective and sampler, CPU only. Needs BIT's environment (transformers 4.48.2, peft 0.11.1);
skipped without peft.

    <bit env>/python -m pytest tests/test_instruct_diffusion.py

- Run 3 unchanged: its registry entry is RUN3's, use_llm gives it what run 3's text_llm.py gave, and a Fast-dLLM v2
  snapshot still loads through the same call.
- Conversion: Instruct's pinned config.json becomes Fast-dLLM v2 1.5B's except `architectures` and `pad_token_id`;
  a converted Qwen2 snapshot keeps every weight (tied head), and with one-token blocks the vendored model gives
  Qwen2's causal forward exactly.
- Full size (LLM on the meta device): the control's trainable parameters are run 3's, embeddings frozen and tied.
- The new entry end to end: registry, use_llm, BIT's model build (NDT encoder, projector, contrastive head, LoRA) on a
  tiny converted Qwen2, then loss, backward, and greedy and nucleus decodes.
"""

import importlib.util
import json
import subprocess
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

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from bit_experiments import diffusion_llm, stage1, stage2, text_llm  # noqa: E402
from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig  # noqa: E402
from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM  # noqa: E402

RUN3 = "0b98fa9"  # the commit holding run 3's code (B_ssl_fast-dllm-v2-1.5b_s42)
NAME, RUN3_NAME, RUN1_NAME = "qwen2.5-1.5b-instruct-diffusion", "fast-dllm-v2-1.5b", "qwen2.5-1.5b-instruct"
# config.json of Qwen/Qwen2.5-1.5B-Instruct @ 989aa798 and of Efficient-Large-Model/Fast_dLLM_v2_1.5B @ 25093b6f
INSTRUCT_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
    "hidden_act": "silu", "hidden_size": 1536, "initializer_range": 0.02, "intermediate_size": 8960,
    "max_position_embeddings": 32768, "max_window_layers": 21, "model_type": "qwen2", "num_attention_heads": 12,
    "num_hidden_layers": 28, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
    "sliding_window": 32768, "tie_word_embeddings": True, "torch_dtype": "bfloat16", "transformers_version": "4.43.1",
    "use_cache": True, "use_sliding_window": False, "vocab_size": 151936}
FAST_DLLM_CONFIG = INSTRUCT_CONFIG | {
    "architectures": ["Fast_dLLM_QwenForCausalLM"], "model_type": "Fast_dLLM_Qwen", "bd_size": 32,
    "layer_types": ["full_attention"] * 28, "pad_token_id": 151645, "rope_scaling": None, "sliding_window": None,
    "transformers_version": "4.53.1", "auto_map": {
        "AutoConfig": "configuration.Fast_dLLM_QwenConfig", "AutoModel": "modeling.Fast_dLLM_QwenModel",
        "AutoModelForCausalLM": "modeling.Fast_dLLM_QwenForCausalLM"}}
MASK_ID = 151665  # |<MASK>| in Fast-dLLM v2's added_tokens.json
V, BOS, EOS, D = 96, 80, 81, 64  # tiny vocabulary: 82..95 unused, like Qwen's padded rows
TINY_MASK = 95
REGISTRY = yaml.safe_load((ROOT / "configs" / "llms.yaml").read_text())


def git_show(path):
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{RUN3}:{path}"], check=True, capture_output=True,
                              text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"run 3's code ({RUN3}) is not in this checkout")


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


def tiny_qwen2(seed=0):
    """A random 2-layer, 64-d Qwen2 causal LM with tied embeddings, every weight random (biases and norms too)."""
    torch.manual_seed(seed)
    llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                       rope_theta=1e6, bos_token_id=BOS, eos_token_id=EOS, tie_word_embeddings=True,
                                       use_sliding_window=False))
    with torch.no_grad():
        for name, p in llm.named_parameters():
            p.uniform_(0.5, 1.5) if "norm" in name else p.normal_(0, 0.1)
    return llm


@pytest.fixture
def snapshot(tmp_path):
    """A tiny Qwen2 snapshot at the control's local_dir under tmp_path (= the repository root for use_llm)."""
    path = tmp_path / REGISTRY[NAME]["local_dir"]
    llm = tiny_qwen2()
    llm.save_pretrained(path)
    return llm, path


# ------------------------------------------------------------------ run 3 unchanged


def test_registry_keeps_runs_1_and_3():
    old = yaml.safe_load(git_show("configs/llms.yaml"))
    assert REGISTRY[RUN3_NAME] == old[RUN3_NAME] and REGISTRY[RUN1_NAME] == old[RUN1_NAME]
    new = REGISTRY[NAME]
    assert {k: new[k] for k in ("repo_id", "revision", "local_dir")} == \
        {k: old[RUN1_NAME][k] for k in ("repo_id", "revision", "local_dir")}  # run 1's pinned snapshot
    assert new["loader"] == old[RUN3_NAME]["loader"] and new["sampler"] == old[RUN3_NAME]["sampler"]
    assert new["mask_id"] == MASK_ID < INSTRUCT_CONFIG["vocab_size"]


def test_use_llm_gives_run3_what_run3s_code_gave_and_the_control_the_same_sampler(tmp_path, bit_trainer):
    path = tmp_path / "text_llm_run3.py"
    path.write_text(git_show("src/bit_experiments/text_llm.py"))
    spec = importlib.util.spec_from_file_location("text_llm_run3", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    snap = tmp_path / REGISTRY[RUN3_NAME]["local_dir"]
    snap.mkdir(parents=True)
    (snap / "added_tokens.json").write_text(json.dumps({"<|im_end|>": 151645, "|<MASK>|": MASK_ID}))

    def configure(module, name):
        module.ROOT, module.REGISTRY = tmp_path, text_llm.REGISTRY
        config = {"method": {"model_kwargs": {}}, "model": {}, "data": {}}
        entry = module.use_llm(config, name)
        return config, entry

    new3, old3 = configure(text_llm, RUN3_NAME), configure(old, RUN3_NAME)
    assert new3 == old3
    control, entry = configure(text_llm, NAME)
    run1, _ = configure(text_llm, RUN1_NAME)
    assert control["method"]["model_kwargs"]["diffusion_sampler"] == \
        new3[0]["method"]["model_kwargs"]["diffusion_sampler"] == \
        {"block_size": 32, "small_block_size": 8, "threshold": 0.9, "mask_id": MASK_ID}
    assert control["model"]["model_class"] == new3[0]["model"]["model_class"] == "BCI_DiffusionLLM"
    assert control["data"] == run1["data"] and control["method"]["model_kwargs"]["llm_path"] == \
        run1["method"]["model_kwargs"]["llm_path"] == str(tmp_path / "checkpoints/hf/Qwen2.5-1.5B-Instruct")
    assert bit_trainer.NAME2MODEL["BCI_DiffusionLLM"] is diffusion_llm.BCI_DiffusionLLM


def test_fast_dllm_snapshot_loads_as_before(tmp_path):
    cfg = Fast_dLLM_QwenConfig(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                               bos_token_id=BOS, eos_token_id=EOS, bd_size=32, tie_word_embeddings=True)
    torch.manual_seed(0)
    Fast_dLLM_QwenForCausalLM(cfg).save_pretrained(tmp_path)
    got = diffusion_llm.BCI_DiffusionLLM.from_pretrained(tmp_path)
    ref = Fast_dLLM_QwenForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float32)  # run 3's call
    assert got.config.to_dict() == ref.config.to_dict()
    a, b = got.state_dict(), ref.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ the conversion


def test_instruct_config_converts_to_fast_dllm_v2s():
    conv = diffusion_llm.qwen2_as_fast_dllm_config(INSTRUCT_CONFIG).to_dict()
    ref = Fast_dLLM_QwenConfig(**{k: v for k, v in FAST_DLLM_CONFIG.items() if k != "model_type"}).to_dict()
    differ = {k for k in conv.keys() | ref.keys() if conv.get(k) != ref.get(k)}
    # architectures: metadata; auto_map: the snapshot's own code, never run; pad_token_id: marks EOS's embedding row
    # as gradient-free, and the embeddings are frozen under LoRA anyway
    assert differ == {"architectures", "auto_map", "pad_token_id"}
    for bad in ({"use_sliding_window": True}, {"rope_scaling": {"type": "yarn", "factor": 4.0}},
                {"model_type": "llama"}, {"hidden_act": "gelu"}):
        with pytest.raises(ValueError):
            diffusion_llm.qwen2_as_fast_dllm_config(INSTRUCT_CONFIG | bad)


def test_converted_snapshot_keeps_every_weight_and_qwen2s_causal_forward(snapshot):
    qwen, path = snapshot
    llm = diffusion_llm.BCI_DiffusionLLM.from_pretrained(path)
    assert isinstance(llm, Fast_dLLM_QwenForCausalLM) and llm.config.bd_size == diffusion_llm.BD_SIZE == 32
    assert llm.lm_head.weight is llm.model.embed_tokens.weight  # tied, as Instruct
    a, b = llm.state_dict(), qwen.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    qwen.eval(), llm.eval()
    ids = torch.randint(0, BOS, (2, 45), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = qwen.model(input_ids=ids).last_hidden_state
        hidden = llm.model(inputs_embeds=llm.model.embed_tokens(ids), block_size=1).last_hidden_state  # causal
        logits = llm.lm_head(hidden)
    assert torch.allclose(hidden, ref, atol=1e-5) and torch.allclose(logits, qwen(input_ids=ids).logits, atol=1e-5)


# ------------------------------------------------------------------ full size


def full_size_model(cfg, monkeypatch, bit_trainer):
    """The Stage 2 model as train_stage2.py builds it (BIT's sentence config), with a meta-device LLM of config cfg."""
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


def test_full_size_trainable_parameters_are_run3s(monkeypatch, bit_trainer):
    run3 = full_size_model(Fast_dLLM_QwenConfig(**{k: v for k, v in FAST_DLLM_CONFIG.items() if k != "model_type"}),
                           monkeypatch, bit_trainer)
    control = full_size_model(diffusion_llm.qwen2_as_fast_dllm_config(INSTRUCT_CONFIG), monkeypatch, bit_trainer)
    assert trainable(control) == trainable(run3)
    lora = sum(n for k, n in trainable(control).items() if k.startswith("llm."))
    assert lora == 28 * 8 * (2 * (1536 + 1536) + 2 * (1536 + 256) + 3 * (1536 + 8960)) == 9_232_384
    assert sum(trainable(control).values()) == 21_280_001  # + NDT encoder, projector, contrastive head and scale
    base = control.llm.get_base_model()
    assert not base.model.embed_tokens.weight.requires_grad and base.lm_head.weight is base.model.embed_tokens.weight


# ------------------------------------------------------------------ the new entry end to end


def test_control_entry_builds_trains_and_decodes(snapshot, monkeypatch, bit_trainer):
    _, path = snapshot
    monkeypatch.setattr(text_llm, "ROOT", path.parents[2])
    config = stage1.bit_sentence_config()
    entry = text_llm.use_llm(config, NAME)
    assert entry["path"] == str(path) and config["model"]["model_class"] == "BCI_DiffusionLLM"
    kwargs = config.method.model_kwargs
    assert kwargs["diffusion_sampler"]["mask_id"] == MASK_ID
    config["model"]["projector"]["embed_size"] = D // 2  # the contrastive head expects hidden / 2

    def build(mask_id):
        kwargs["diffusion_sampler"]["mask_id"] = mask_id
        with stage1.in_bit_dir():
            torch.manual_seed(0)
            return bit_trainer.NAME2MODEL[config.model.model_class](
                config.model, **kwargs, **config.method.dataset_kwargs, features="all")

    with pytest.raises(ValueError, match="no embedding row"):
        build(V)  # the registry's 151665 has no row in the tiny vocabulary
    model = build(TINY_MASK)
    stage2.fix_padding(model)
    base = model.llm.get_base_model()
    assert isinstance(base, Fast_dLLM_QwenForCausalLM) and model.block_size == 32 and model.mask_id == TINY_MASK
    mask_row = base.model.embed_tokens.weight[TINY_MASK].clone()

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

    model.train()
    torch.manual_seed(0)
    out = model(**inputs, targets=targets)
    assert torch.isfinite(out.loss) and out.preds.shape == (int(out.n_examples), V)
    assert int(out.n_examples) >= sum(len(s) + 1 for s in sentences)  # sentence + EOS, and the block's EOS padding
    out.loss.backward()
    lora_b = {n: p.grad for n, p in model.llm.named_parameters() if "lora_B" in n}
    assert len(lora_b) == 2 * 7 and all(g is not None and g.abs().sum() > 0 for g in lora_b.values())
    assert base.model.embed_tokens.weight.grad is None and torch.equal(base.model.embed_tokens.weight[TINY_MASK],
                                                                       mask_row)
    assert all(p.grad is not None for n, p in model.named_parameters() if n.startswith("projector."))

    model.eval()
    one = {k: v[:1] for k, v in inputs.items()} | {"input_ids": torch.tensor([prompt_a + prompt_b]),
                                                   "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    for decoding in (stage2.DECODINGS["greedy"], stage2.DECODINGS["nucleus"]):
        torch.manual_seed(0)
        ids = model.generate(**one, **decoding)
        assert ids.shape[0] == 1 and 1 <= ids.shape[1] <= decoding["max_new_tokens"]
        assert TINY_MASK not in ids.tolist()[0]
