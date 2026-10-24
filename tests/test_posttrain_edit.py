"""scripts/posttrain_edit.py (two-step revision design, step 1; docs/REVISION_RUN.md) on CPU with a tiny random
Fast-dLLM v2: the leakage filter, corpus packing, layer checkpointing, the training loop with its held-out check, and
the snapshots it saves, for a full fine-tune and for LoRA. The script's HF glue (FineWeb-Edu streaming, the tokenizer,
BIT's sentences) needs the GPU host and is not exercised here. Needs BIT's environment (skipped without peft):

    <bit env>/python -m pytest tests/test_posttrain_edit.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
peft = pytest.importorskip("peft")

from bit_experiments.diffusion_llm import text_block_diffusion_ce  # noqa: E402
from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig  # noqa: E402
from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM  # noqa: E402

spec = importlib.util.spec_from_file_location("posttrain_edit", ROOT / "scripts" / "posttrain_edit.py")
posttrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(posttrain)

V, BOS, EOS, MASK, BS = 96, 80, 81, 95, 32  # as tests/test_diffusion_revision.py


def tiny_base(seed=0):
    """Random 2-layer, 64-d Fast-dLLM v2. Its head is scaled up: with the random head (std 0.02) and the final norm
    frozen, LoRA could not express a confident prediction (logits within about ±1.3) and would learn nothing."""
    torch.manual_seed(seed)
    model = Fast_dLLM_QwenForCausalLM(Fast_dLLM_QwenConfig(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=1024, bos_token_id=BOS, eos_token_id=EOS, bd_size=BS))
    with torch.no_grad():
        model.lm_head.weight.mul_(20.0)
    return model


def lora(model, r=8):
    return peft.get_peft_model(model, peft.LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                                      target_modules=posttrain.LORA_TARGETS))


def corpus(seq_len, seed=0):
    """Endless rows from 8 fixed 'sentences' of ordinary tokens in random order: learnable, so the editor improves."""
    g = torch.Generator().manual_seed(seed)
    sentences = [torch.randint(0, BOS, (int(n),), generator=g).tolist() for n in torch.randint(5, 12, (8,), generator=g)]

    def docs():
        while True:
            yield sentences[int(torch.randint(0, 8, (1,), generator=g))]

    return posttrain.pack(docs(), lambda d: d, BOS, seq_len)


def test_count():
    assert posttrain.count("100M") == 10**8 and posttrain.count("2.5k") == 2500 and posttrain.count("2048") == 2048


def test_excluder_matches_whole_sentences_after_normalisation():
    ex = posttrain.Excluder(["I would like a cup of tea.", "Thank you", "How are you doing today?"])
    assert ex.n == 2 and ex.skipped == 1  # "thank you" is too short to mean anything
    assert ex("Well -- I would LIKE a cup of tea, please!") and ex("so, how are you doing today")
    assert not ex("I would like a cup of coffee") and not ex("thank you very much") and not ex("a cup of tea")


def test_pack_drops_excluded_documents():
    tokenize = lambda t: [ord(c) % 50 for c in t if c != " "]  # noqa: E731
    stats = {}
    docs = ["a b c", "and I would like a cup of tea", "d e"]
    rows = list(posttrain.pack(docs, tokenize, BOS, 4, posttrain.Excluder(["I would like a cup of tea"]), stats))
    assert stats == {"docs": 3, "dropped": 1} and all(len(r) == 4 for r in rows)
    assert sum(rows, []) == (tokenize("a b c") + [BOS] + tokenize("d e") + [BOS])[: 4 * len(rows)]


@pytest.mark.parametrize("use_lora", [False, True])
def test_layer_checkpointing_is_exact(use_lora):
    ids = torch.randint(0, BOS, (2, 2 * BS), generator=torch.Generator().manual_seed(0))
    results = []
    for ckpt in (False, True):
        model = lora(tiny_base()) if use_lora else tiny_base()
        if ckpt:
            posttrain.checkpoint_layers(model)
        model.train()
        torch.manual_seed(0)
        ce, n, _, _ = text_block_diffusion_ce(model, ids, MASK, edit_rate=0.2)
        ce.backward()
        results.append((ce, {k: p.grad for k, p in model.named_parameters() if p.grad is not None}))
    (ce0, g0), (ce1, g1) = results
    torch.testing.assert_close(ce1, ce0)
    assert g0.keys() == g1.keys() and all(torch.allclose(g0[k], g1[k], atol=1e-6) for k in g0)


@pytest.mark.parametrize("use_lora", [False, True])
def test_training_loop_and_snapshots(tmp_path, use_lora):
    base_dir, out = tmp_path / "base", tmp_path / "out"
    tiny_base().save_pretrained(base_dir)
    (base_dir / "added_tokens.json").write_text(json.dumps({"|<MASK>|": MASK}))
    (base_dir / "tokenizer_config.json").write_text("{}")
    args = posttrain.parser().parse_args([
        "--out", str(out), "--tokens", "8192", "--save-at", "4096", "--seq-len", "64", "--batch-rows", "4",
        "--micro-rows", "2", "--lr", "3e-2" if use_lora else "3e-3", "--warmup", "2", "--eval-every", "8", "--heldout-rows", "4",
        "--edit-rate", "0.3", "--eval-edit-rate", "0.3", "--grad-ckpt"] + (["--lora", "8"] if use_lora else []))
    torch.manual_seed(args.seed)
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    if use_lora:
        model = lora(model)
    posttrain.checkpoint_layers(model)
    rows = corpus(args.seq_len)
    heldout = [next(rows) for _ in range(args.heldout_rows)]
    out.mkdir()
    result = posttrain.train(model, rows, heldout, MASK, args, out, base_dir)

    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log] == [0, 8, 16, 24, 32] and log[-1]["tokens"] == 8192  # 256 tokens per step
    first, last = log[0]["heldout"], log[-1]["heldout"]
    assert last["t2t_ce"] < first["t2t_ce"] - 2 and last["m2t_ce"] < first["m2t_ce"] - 2  # held-out loss falls
    if not use_lora:  # and, full fine-tune, recovers corrupted tokens (LoRA is slower on this random tiny model)
        assert last["t2t_acc"] > first["t2t_acc"] + 0.2
    assert first["t2t_n"] > 0 and first["m2t_n"] > 0 and result["saved"] == [4096, 8192]
    assert (out / "tokens_4096" / "tokenizer_config.json").exists()
    assert (out / "tokens_4096" / "added_tokens.json").exists()

    snapshot = Fast_dLLM_QwenForCausalLM.from_pretrained(out / "tokens_8192", torch_dtype=torch.float32)
    assert not any("lora" in k for k in snapshot.state_dict())
    ids = torch.tensor(heldout[:2])
    losses = []
    for m in (model, snapshot):  # the saved (merged) weights behave as the trained model
        m.eval()
        torch.manual_seed(1)
        with torch.no_grad():
            losses.append(text_block_diffusion_ce(m, ids, MASK, edit_rate=0.3)[0])
    torch.testing.assert_close(losses[1], losses[0], rtol=1e-5, atol=1e-4)


def test_main_end_to_end(tmp_path, monkeypatch):
    """main() on a local text file and an exclusion file, with a word-level tokenizer saved in the base snapshot."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [a + b for a in "abcd" for b in "abcdefghijklmnopqrst"]  # 80 ordinary tokens, ids 0..79 (< BOS)
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token=words[0]))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    base_dir = tmp_path / "base"
    tiny_base().save_pretrained(base_dir)
    PreTrainedTokenizerFast(tokenizer_object=tok).save_pretrained(base_dir)
    (base_dir / "added_tokens.json").write_text(json.dumps({"|<MASK>|": MASK}))
    g = torch.Generator().manual_seed(0)
    excluded = "ab ac ad ae af"
    lines = [" ".join(words[int(i)] for i in torch.randint(0, 80, (8,), generator=g)) for _ in range(400)]
    lines[3] = lines[50] = f"ba {excluded} bb"  # contain a "BIT sentence": dropped
    (tmp_path / "docs.txt").write_text("\n".join(lines))
    (tmp_path / "exclude.txt").write_text(excluded + "\nab\n")  # "ab": too short to use
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "posttrain_edit.py", "--base", str(base_dir), "--text-file", str(tmp_path / "docs.txt"), "--exclude",
        str(tmp_path / "exclude.txt"), "--out", str(out), "--tokens", "2048", "--save-at", "--seq-len", "64",
        "--batch-rows", "4", "--micro-rows", "2", "--heldout-rows", "2", "--eval-every", "4", "--device", "cpu"])
    assert posttrain.main() == 0
    run = json.loads((out / "posttrain.json").read_text())
    assert run["exclude"] == {"sentences": 1, "too_short": 1} and run["data"]["dropped"] == 2
    assert run["saved"] == [2048] and run["mask_id"] == MASK
    assert (out / "tokens_2048" / "tokenizer.json").exists() and (out / "tokens_2048" / "model.safetensors").exists()
