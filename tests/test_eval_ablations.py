"""scripts/eval_ablations.py (docs/EVAL_ABLATIONS.md), CPU only.

    .venv/bin/python -m pytest tests/test_eval_ablations.py        # transforms, statistics, main() with a stub model
    <bit env>/python -m pytest tests/test_eval_ablations.py        # + BIT's real model build on tiny LLMs

- Config: the channel map covers the 512 model columns exactly once; each array = its 64 threshold-crossing and its
  64 spike-band-power columns (nejm-brain-to-text README order).
- Transforms on small synthetic trials: shapes, dtypes, which bins / columns are filled, the input never modified,
  seeded determinism, the patch scramble a permutation of whole patches, time reversal an involution.
- Statistics: the train-mean fill, the paired bootstrap (zero for identical decodes, exact point difference).
- main() end to end with a stub model whose words depend on which arrays and which half of the trial carry
  activity: every ablation's decodes, WERs and deltas, the `[ablation]` lines, a failing run logged and skipped,
  exit 0, the budget skip, the summaries, and the phonetic measures (or the note that they are pending).
- BIT's environment only: load_run's real path (BIT's sentence config, use_llm, BIT's Trainer, fix_padding, a slim
  BEST, the NDT encoder) for a block-diffusion and an autoregressive tiny Qwen2, 3 trials each: the rebuilt model
  reproduces the saved decodes, and the ablated features are what reaches the encoder.
"""

import gc
import importlib.machinery
import importlib.util
import json
import sys
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_ablations", ROOT / "scripts" / "eval_ablations.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_ablations"] = mod
    spec.loader.exec_module(mod)
    return mod


ea = _load_script()
CFG = ea.load_config()
ARRAYS = list(CFG["arrays"])


def trial(t=37, c=512, seed=0):
    return np.random.default_rng(seed).standard_normal((t, c)).astype(np.float32)


def fill_vec(c=512):
    return np.linspace(-0.1, 0.1, c).astype(np.float32)


def transform(name, fill=None, patch=5, cfg=CFG):
    return ea.make_transform(cfg["ablations"][name], cfg, fill_vec() if fill is None else fill, patch)


# ------------------------------------------------------------------ config


def test_channel_map_is_four_arrays_of_tx_and_sbp():
    cols = {a: ea.array_columns(CFG, [a]) for a in ARRAYS}
    assert ARRAYS == ["v6v", "a4", "55b", "d6v"]
    for k, a in enumerate(ARRAYS):
        assert cols[a].tolist() == list(range(64 * k, 64 * k + 64)) + list(range(256 + 64 * k, 256 + 64 * k + 64))
        assert ea.array_columns(CFG, [a], ["tx"]).tolist() == list(range(64 * k, 64 * k + 64))
    assert sorted(np.concatenate(list(cols.values())).tolist()) == list(range(512))
    assert ea.array_columns(CFG, ARRAYS, ["sbp"]).tolist() == list(range(256, 512))


def test_config_rejects_overlapping_arrays(tmp_path):
    bad = dict(CFG, arrays=dict(CFG["arrays"], a4={"area": "x", "tx": [60, 128], "sbp": [320, 384]}))
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly once"):
        ea.load_config(path)


def test_every_default_ablation_builds():
    for name in CFG["default"] + ["drop_tx", "drop_sbp"]:
        fn = transform(name)
        assert (fn is None) == (name == "none")


# ------------------------------------------------------------------ transforms


@pytest.mark.parametrize("array", ARRAYS)
def test_drop_and_only_fill_exactly_the_right_columns(array):
    x, fill = trial(), fill_vec()
    x0 = x.copy()
    cols = ea.array_columns(CFG, [array])
    rest = np.setdiff1d(np.arange(512), cols)
    for name, filled, kept in ((f"drop_{array}", cols, rest), (f"only_{array}", rest, cols)):
        y = transform(name, fill)(x, 3)
        assert y.shape == x.shape and y.dtype == np.float32
        assert np.array_equal(y[:, filled], np.broadcast_to(fill[filled], (len(x), len(filled))))
        assert np.array_equal(y[:, kept], x[:, kept])
        assert np.array_equal(x, x0)  # the input is never modified


def test_drop_feature_type_fills_it_on_every_array():
    x, fill = trial(), fill_vec()
    y = transform("drop_tx", fill)(x, 0)
    assert np.array_equal(y[:, :256], np.broadcast_to(fill[:256], (len(x), 256))) and np.array_equal(y[:, 256:],
                                                                                                    x[:, 256:])
    y = transform("drop_sbp", fill)(x, 0)
    assert np.array_equal(y[:, :256], x[:, :256]) and np.array_equal(y[:, 256:], np.broadcast_to(fill[256:],
                                                                                                (len(x), 256)))


@pytest.mark.parametrize("t", [1, 2, 37, 40, 101])
def test_time_windows(t):
    x, fill = trial(t), fill_vec()
    kept = {}
    for name in ("time_first_half", "time_second_half", "time_middle_half"):
        y = transform(name, fill)(x, 0)
        assert y.shape == x.shape and y.dtype == x.dtype
        lo, hi = CFG["ablations"][name]["window"]
        a, b = ea.window_bins(t, lo, hi)
        mask = np.zeros(t, bool)
        mask[a:b] = True
        assert np.array_equal(y[mask], x[mask])
        assert np.array_equal(y[~mask], np.broadcast_to(fill, ((~mask).sum(), 512)))
        kept[name] = mask
    assert not (kept["time_first_half"] & kept["time_second_half"]).any()
    assert (kept["time_first_half"] | kept["time_second_half"]).all()  # the two halves partition the trial
    assert kept["time_middle_half"].sum() == int(0.75 * t) - int(0.25 * t)
    assert abs(kept["time_first_half"].sum() - t / 2) <= 0.5


def test_scramble_permutes_whole_patches_seeded_per_trial():
    x = trial(43)  # 8 full patches of 5 bins + 3 bins
    fn = transform("scramble_patches", patch=5)
    y = fn(x, 7)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert np.array_equal(y, fn(x, 7))  # deterministic for (seed, trial)
    assert not np.array_equal(y, fn(x, 8))  # trials get their own order
    assert np.array_equal(y[40:], x[40:])  # the trailing partial patch stays last
    patches_x, patches_y = x[:40].reshape(8, 5, 512), y[:40].reshape(8, 5, 512)
    order = [next(j for j in range(8) if np.array_equal(p, patches_x[j])) for p in patches_y]
    assert sorted(order) == list(range(8)) and order != list(range(8))  # whole patches, bins in order inside
    assert order == np.random.default_rng([CFG["seed"], 7]).permutation(8).tolist()
    other = ea.make_transform(CFG["ablations"]["scramble_patches"] | {"seed": 1}, CFG, fill_vec(), 5)
    assert not np.array_equal(other(x, 7), y)
    short = trial(4)
    assert np.array_equal(fn(short, 0), short)  # shorter than one patch: nothing to permute


def test_reverse_is_an_involution():
    x = trial(23)
    fn = transform("reverse_time")
    y = fn(x, 0)
    assert np.array_equal(y, x[::-1]) and y.flags.c_contiguous and np.array_equal(fn(y, 0), x)


def test_ablated_dataset_changes_only_spikes_and_checks_shapes():
    rows = [{"spikes": trial(10, seed=i), "other": np.arange(3)} for i in range(2)]
    ds = ea.Ablated(rows, transform("drop_v6v", np.zeros(512, np.float32)))
    item = ds[1]
    assert len(ds) == 2 and np.array_equal(item["other"], np.arange(3))
    assert (item["spikes"][:, :64] == 0).all() and np.array_equal(rows[1]["spikes"], trial(10, seed=1))
    assert ea.Ablated(rows, None)[0]["spikes"] is rows[0]["spikes"]
    with pytest.raises(RuntimeError, match="changed the input"):
        ea.Ablated(rows, lambda x, i: x[:5])[0]


# ------------------------------------------------------------------ fill and statistics


def test_train_mean_fill():
    rows = [{"spikes": trial(t, seed=t)} for t in (3, 8, 20)]
    fill, info = ea.fill_vector(CFG, rows)
    assert fill.dtype == np.float32 and np.allclose(fill, np.concatenate([r["spikes"] for r in rows]).mean(0),
                                                    atol=1e-6)
    assert info["n_train_bins"] == 31 and info["n_train_trials"] == 3 and len(info["vector"]) == 512
    assert set(info) >= {"tx", "sbp"} and info["tx"]["max_abs"] >= info["tx"]["mean_abs"]
    zero, zinfo = ea.fill_vector(CFG | {"fill": "zero"}, rows)
    assert not zero.any() and zinfo["kind"] == "zero"


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    words = rng.integers(3, 12, 50).astype(float)
    base = rng.integers(0, 4, 50).astype(float)
    idx = ea.resamples(50, CFG)
    assert np.array_equal(idx, ea.resamples(50, CFG)) and idx.shape == (CFG["bootstrap"]["n_resamples"], 50)
    same = ea.paired_delta(base, base, words, idx)
    assert same["delta_wer"] == 0 and same["ci"] == [0, 0] and same["p_boot"] == 1 and same["trials_same"] == 50
    worse = base + (np.arange(50) % 2)
    d = ea.paired_delta(worse, base, words, idx)
    assert d["delta_wer"] == pytest.approx(25 / words.sum()) and d["ci"][0] > 0 and d["p_boot"] < 0.01
    assert d["trials_worse"] == 25 and d["trials_better"] == 0


# ------------------------------------------------------------------ main() with a stub model

WORDS = ["alpha", "bravo", "charlie", "delta"]  # one per array: said when the array's TX columns are active
TIME = {True: "early", False: "late"}  # which half of the trial carries more activity


class StubTokenizer:
    """Word ids: 0..3 the arrays, 4/5 early/late, EOS 9."""
    vocab = WORDS + ["early", "late"]
    eos_token_id, vocab_size = 9, 10

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.vocab[i] for i in ids.tolist() if i < len(self.vocab))


def stub_words(x):
    """The stub decoder's rule: array k's word if its TX columns average > 0.5, then early / late by which half of
    the trial has the larger mean."""
    ids = [k for k in range(4) if x[:, 64 * k: 64 * k + 64].mean() > 0.5]
    half = len(x) // 2
    return ids + [4 if x[:half].mean() > x[half:].mean() else 5]


class StubModel:
    def eval(self):
        return self

    def generate(self, spikes, input_ids, attention_mask, **gen):
        return torch.tensor([stub_words(spikes[0].float().numpy()) + [9]])


def stub_collate(items):
    (item,) = items
    return {"spikes": torch.from_numpy(np.ascontiguousarray(item["spikes"]))[None],
            "input_ids": torch.tensor([[1, 2, 3]]), "targets": torch.full((1, 3), -100)}, {}


def stub_trials(n=4, t=40):
    rows, refs = [], []
    for i in range(n):
        x = np.zeros((t, 512), np.float32)
        active = [k for k in range(4) if (i + k) % 4 != 3]  # each trial: three arrays active
        for k in range(4):
            x[:, 64 * k: 64 * k + 64] = -2.0  # inactive (train mean of an array's TX: (3 * 1.2 - 2) / 4 = 0.4)
        for k in active:
            x[: t // 2, 64 * k: 64 * k + 64] = 2.0  # activity in the first half
            x[t // 2:, 64 * k: 64 * k + 64] = 0.4
        rows.append({"spikes": x})
        refs.append(" ".join([WORDS[k] for k in active] + ["early"]))
    return rows, refs


@pytest.fixture
def stub_results(tmp_path, monkeypatch):
    """RESULTS in tmp_path with two finished stub runs and one that fails to load."""
    results = tmp_path / "results" / "stage2"
    for run in ("stub_a", "stub_b", "stub_broken"):
        (results / run / "decodes").mkdir(parents=True)
        (results / run / "run.json").write_text(json.dumps({"n": {"test": 4}, "config": {"model": {
            "model_class": "BCI_DiffusionLLM"}}}))
        (results / run / "summary.json").write_text(json.dumps({"minutes": 1.0, "results": {
            "test_greedy_real": {"n": 600}}}))
    monkeypatch.setattr(ea, "RESULTS", results)
    rows, refs = stub_trials()
    saved = [" ".join(r.split()) for r in refs]
    saved[0] = "alpha"  # the stub's saved decode differs on one trial: matches_saved False for stub_b
    import pandas as pd

    pd.DataFrame({"trial": range(4), "reference": refs, "decoded": refs, "seen_in_train": [False] * 4}).to_csv(
        results / "stub_a" / "decodes" / "test_greedy_real.csv", index=False)
    pd.DataFrame({"trial": range(4), "reference": refs, "decoded": saved, "seen_in_train": [False] * 4}).to_csv(
        results / "stub_b" / "decodes" / "test_greedy_real.csv", index=False)

    def fake_load_run(run, splits, limit, device, cache, cfg):
        if run == "stub_broken":
            raise RuntimeError("cannot build this one")
        cache["loads"] = cache.get("loads", 0) + 1
        fill, info = ea.fill_vector(cfg, rows)
        return ea.Bundle(StubModel(), StubTokenizer(), stub_collate, {"test": rows}, {"test": refs},
                         {"test": np.zeros(4, bool)}, 5, fill,
                         {"run": run, "model_class": "stub", "llm": None, "prompt": "#", "checkpoint": "none",
                          "arm": "B", "seed": 42, "fill": info})

    monkeypatch.setattr(ea, "load_run", fake_load_run)
    monkeypatch.setattr(ea, "BIT", tmp_path)  # main() chdirs there
    monkeypatch.chdir(tmp_path)
    return results


def test_main_end_to_end_with_a_stub_model(stub_results, capsys):
    rc = ea.main(["--runs", "stub_a", "stub_broken", "stub_b", "--device", "cpu", "--budget-minutes", "0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[ablation] estimate stub_a:" in out and "[ablation] stub_broken run: ERROR" in out
    assert "cannot build this one" in (stub_results / "stub_broken" / "ablations" / "error.log").read_text()
    lines = [line for line in out.splitlines() if line.startswith("[ablation] stub_a ") and ": {" in line]
    assert len([line for line in lines if " phonetic:" not in line]) == len(CFG["default"])
    first = json.loads(lines[0].split(": ", 1)[1])
    assert first["split"] == "test" and first["wer"] == 0

    s = json.loads((stub_results / "stub_a" / "ablations" / "summary.json").read_text())
    r = s["results"]["test"]
    assert list(r) == CFG["default"] and not s["errors"]
    assert r["none"]["wer"] == 0 and r["none"]["matches_saved"] is True
    # every decode = the stub's rule on the ablated trial (computed here independently of the runner)
    rows, _ = stub_trials()
    fill, _ = ea.fill_vector(CFG, rows)
    tok = StubTokenizer()
    for name in CFG["default"]:
        fn = transform(name, fill)
        want = [tok.decode(torch.tensor(stub_words(x if fn is None else fn(x, i)))) for i, x in
                enumerate(r["spikes"] for r in rows)]
        df = ea.read_decodes(stub_results / "stub_a" / "ablations" / name / "test_greedy.csv")
        assert df["decoded"].tolist() == want, name
    # trial i has arrays (i+k) % 4 != 3 active: dropping array k loses its word in those 3 trials (4 words each)
    for a in ARRAYS:
        assert r[f"drop_{a}"]["delta_wer"] == pytest.approx(3 / 16) and r[f"drop_{a}"]["trials_worse"] == 3
        assert r[f"only_{a}"]["delta_wer"] > r[f"drop_{a}"]["delta_wer"]
    assert r["time_first_half"]["delta_wer"] == 0  # the information is in the first half
    assert r["time_second_half"]["delta_wer"] > 0 and r["reverse_time"]["delta_wer"] == pytest.approx(4 / 16)
    assert r["scramble_patches"]["trials_same"] + r["scramble_patches"]["trials_worse"] == 4
    meta = json.loads((stub_results / "stub_a" / "ablations" / "drop_v6v" / "test_greedy.meta.json").read_text())
    assert meta["condition"] == "ablation_drop_v6v" and meta["ablation"]["arrays"] == ["v6v"]
    sb = json.loads((stub_results / "stub_b" / "ablations" / "summary.json").read_text())["results"]["test"]
    assert sb["none"]["matches_saved"] is False and sb["none"]["trials_differing_from_saved"] == 1

    agg = json.loads((stub_results / "ablations_summary.json").read_text())
    assert set(agg["runs"]) == {"stub_a", "stub_b"} and "vector" not in agg["runs"]["stub_a"]["fill"]
    md = (stub_results / "ablations_summary.md").read_text()
    assert "`stub_a`, test" in md and "drop_v6v: ventral 6v removed" in md and "(≠ saved: 1)" in md
    if ea.phonetic_missing():
        assert r["drop_v6v"]["phonetic"]["skipped"].startswith("skipped") and "pending" in md
        assert ea.main(["--phonetic-only", "--runs", "stub_a"]) == 0  # says what is missing, still exits 0
    else:
        ph = r["reverse_time"]["phonetic"]
        assert ph["n_sub"] == 4 and ph["alike"]["obs"] == 0  # early -> late: not sound-alike
        assert r["none"]["phonetic"]["n_sub"] == 0


def test_phonetic_only_fills_pending_measures(stub_results, monkeypatch):
    if ea.phonetic_missing():
        pytest.skip("phonetic measures need pronouncing + wordfreq (project .venv)")
    monkeypatch.setattr(ea, "phonetic_missing", lambda: ["pronouncing"])
    assert ea.main(["--runs", "stub_a", "--device", "cpu", "--ablations", "reverse_time"]) == 0
    path = stub_results / "stub_a" / "ablations" / "summary.json"
    assert "skipped" in json.loads(path.read_text())["results"]["test"]["reverse_time"]["phonetic"]
    monkeypatch.undo()
    monkeypatch.setattr(ea, "RESULTS", stub_results)
    assert ea.main(["--phonetic-only", "--runs", "stub_a"]) == 0
    r = json.loads(path.read_text())["results"]["test"]
    assert r["reverse_time"]["phonetic"]["n_sub"] == 4 and "phon_excess_vs_none" not in r["none"]["phonetic"]


def test_budget_skips_later_runs_and_estimates_print(stub_results, capsys, monkeypatch):
    monkeypatch.setattr(ea, "sec_per_trial", lambda run, cfg: (60.0, "test"))  # 14 x 4 decodes x 1 min per run
    rc = ea.main(["--runs", "stub_a", "stub_b", "--device", "cpu", "--budget-minutes", "10"])
    out = capsys.readouterr().out
    assert rc == 0 and "[ablation] stub_b: skipped" in out
    assert (stub_results / "stub_a" / "ablations" / "summary.json").is_file()
    assert not (stub_results / "stub_b" / "ablations" / "summary.json").exists()
    assert ea.main(["--runs", "stub_a", "--estimate-only"]) == 0


def test_estimate_from_a_runs_final_evaluation(tmp_path, monkeypatch):
    results = tmp_path / "stage2"
    (results / "r").mkdir(parents=True)
    (results / "r" / "run.json").write_text(json.dumps({"n": {"test": 713, "val": 713}, "config": {"model": {
        "model_class": "BCI_TextLLM"}}}))
    monkeypatch.setattr(ea, "RESULTS", results)
    assert ea.sec_per_trial("r", CFG) == (0.14, "default for BCI_TextLLM")
    (results / "r" / "summary.json").write_text(json.dumps({"minutes": 110.0, "results": {
        "a": {"n": 713}, "b": {"n": 713}}}))
    (results / "r" / "log.jsonl").write_text('{"minutes": 50.0}\n{"minutes": 100.0}\n')
    spt, _ = ea.sec_per_trial("r", CFG)
    assert spt == pytest.approx(600 / 1426)
    est = ea.estimate("r", CFG, 14, ["test"], None)
    assert est["decodes"] == 14 * 713 and est["minutes"] == round(2 + 14 * 713 * spt / 60, 1)
    assert ea.estimate("r", CFG, 14, ["test"], 3)["decodes"] == 42


# ------------------------------------------------------------------ BIT's real build (BIT's environment only)

V, BOS, EOS, D, TINY_MASK = 96, 80, 81, 64, 95
TINY_AERO = "checkpoints/hf/Aero-tiny"  # BCI_AudioLLM's snapshot for the run without --llm (BIT's Aero path)


def _stub_missing_modules(monkeypatch):
    """BIT's trainer imports wandb, matplotlib and sklearn (logging and plots only); stand-ins where the CPU env copy
    lacks them."""
    for name in ("wandb", "matplotlib", "matplotlib.pyplot", "sklearn", "sklearn.metrics"):
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mod.__getattr__ = _stub_attr
        monkeypatch.setitem(sys.modules, name, mod)


def _stub_attr(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return lambda *a, **k: None


class WordTokenizer:
    """Whitespace words -> ids 0..79 (a fixed vocabulary, then a hash), BOS 80, EOS 81 ("<eos>"); 82..95 unused."""
    eos_token, eos_token_id, vocab_size = "<eos>", EOS, 82

    def __init__(self, words):
        self.words = sorted(set(words))[:60]
        self.ids = {w: k for k, w in enumerate(self.words)}

    def _id(self, w):
        return EOS if w == self.eos_token else self.ids.get(w, 60 + sum(map(ord, w)) % 20)

    def __call__(self, text, return_tensors="np"):
        return {"input_ids": np.array([[self._id(w) for w in text.replace("<eos>", " <eos> ").split()]])}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(self.words[i] if i < len(self.words) else f"w{i}" for i in ids if i < BOS)


TEXT = dict(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=2048, rope_theta=1e6, bos_token_id=BOS, eos_token_id=EOS,
            tie_word_embeddings=True, use_sliding_window=False)


def _tiny_llm(path, audio: bool, seed=0):
    """A random 2-layer, 64-d Qwen2 causal LM (or Qwen2-Audio with a 1-layer audio tower, as Aero-1-Audio's class),
    every weight random; decodes depend on the neural tokens."""
    from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    if audio:
        llm = Qwen2AudioForConditionalGeneration(Qwen2AudioConfig(
            audio_config={"model_type": "qwen2_audio_encoder", "d_model": 32, "encoder_layers": 1,
                          "encoder_attention_heads": 2, "encoder_ffn_dim": 64, "num_mel_bins": 16,
                          "max_source_positions": 50},
            text_config={"model_type": "qwen2", **TEXT}, audio_token_index=94))
    else:
        llm = Qwen2ForCausalLM(Qwen2Config(**TEXT))
    with torch.no_grad():
        for name, p in llm.named_parameters():
            p.uniform_(0.5, 1.5) if "norm" in name else p.normal_(0, 0.1)
    llm.save_pretrained(path)


@pytest.mark.parametrize("llm", ["qwen2.5-1.5b-instruct-diffusion", "qwen2.5-1.5b-instruct", None],
                         ids=["diffusion", "text-AR", "aero-AR"])
def test_real_bit_build_on_a_tiny_llm_three_trials(llm, tmp_path, monkeypatch, capsys):
    pytest.importorskip("peft")
    _stub_missing_modules(monkeypatch)
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")  # BIT's Trainer would otherwise pick a Mac's MPS device
    from safetensors.torch import save_file

    from bit_experiments import stage1, stage2, text_llm

    root = tmp_path / "repo"
    _tiny_llm(root / (text_llm.llm_entry(llm)["local_dir"] if llm else TINY_AERO), audio=llm is None)
    monkeypatch.setattr(text_llm, "ROOT", root)
    real_config = stage1.bit_sentence_config

    def tiny_config():
        config = real_config()
        config["model"]["projector"]["embed_size"] = D // 2  # the contrastive head expects hidden / 2
        config["training"]["mixed_precision"] = None  # accelerate has no bf16 mixed precision on this CPU
        config["method"]["model_kwargs"]["llm_path"] = str(root / TINY_AERO)  # BIT's default LLM (use_llm resets it)
        return config

    monkeypatch.setattr(stage1, "bit_sentence_config", tiny_config)
    results, checkpoints = root / "results" / "stage2", root / "checkpoints" / "stage2"
    monkeypatch.setattr(ea, "RESULTS", results)
    monkeypatch.setattr(ea, "CHECKPOINTS", checkpoints)

    rng = np.random.default_rng(0)
    sentences = ["the cat sat", "a dog ran far", "we went home now", "it is here", "go up", "see the sun"]
    raw = {s: [{"spikes": rng.standard_normal((int(rng.integers(30, 70)), 512)).astype(np.float32),
                "sentence": sentences[(k + 3 * j) % 6], "block": 0, "day": (2024, 1, 1),
                "block_idx": np.asarray(0), "day_idx": np.asarray(0)} for k in range(3)]
           for j, s in enumerate(("train", "val", "test"))}
    tokenizer = WordTokenizer(" ".join(sentences + [stage1.bit_sentence_config().data.prompt]).split())
    loads = []
    monkeypatch.setattr(ea, "load_raw", lambda data_config: loads.append(1) or raw)
    monkeypatch.setattr(ea, "load_tokenizer", lambda path: tokenizer)

    # a "finished run": run.json as train_stage2.py writes it, a slim BEST of a model with nonzero LoRA weights,
    # its saved greedy test decodes and summary.json
    run = f"B_tiny_{llm or 'aero'}_s42"
    monkeypatch.chdir(ea.BIT)
    config = tiny_config()
    config["seed"] = 42
    if llm:
        text_llm.use_llm(config, llm)
    if config["model"]["model_class"] == "BCI_DiffusionLLM":
        config["method"]["model_kwargs"]["diffusion_sampler"]["mask_id"] = TINY_MASK  # run.json's sampler is used
    from models import trainer as bit_trainer
    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn
    from utils.eval_phoneme import get_model_inputs

    torch.manual_seed(1)
    model = bit_trainer.NAME2MODEL[config.model.model_class](config.model, **config.method.model_kwargs,
                                                             **config.method.dataset_kwargs, features="all")
    stage2.fix_padding(model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    ckpt = checkpoints / run / "BEST"
    ckpt.mkdir(parents=True)
    save_file({k: v.detach().contiguous() for k, v in stage2.slim_state(model).items()}, str(ckpt / stage2.SLIM))
    dataset = text_llm.create_llm_labels({s: list(v) for s, v in raw.items()}, tokenizer, config.data.prompt)
    from functools import partial

    collate = partial(pad_collate_fn, model_inputs=get_model_inputs(model), **config.method.dataloader_kwargs)
    test_ds = SpikingDatasetForDecoding(dataset, "test", **config.method.dataset_kwargs)
    refs = [r["sentence"] for r in dataset["test"]]
    saved = stage2.generate(model, test_ds, collate, tokenizer, "greedy", device="cpu", dtype=torch.float32)
    (results / run / "decodes").mkdir(parents=True)
    stage2.write_decodes(results / run / "decodes" / "test_greedy_real.csv", refs, saved,
                         stage2.seen_mask(refs, [r["sentence"] for r in dataset["train"]]), {})
    (results / run / "run.json").write_text(json.dumps({
        "args": {"name": run, "arm": "B", "seed": 42, "llm": llm, "prompt": None}, "n": {"test": 3},
        "config": json.loads(json.dumps(config, default=str))}, default=str))
    (results / run / "summary.json").write_text(json.dumps({"minutes": 0.1, "results": {"x": {"n": 21}}}))

    # what reaches the encoder: the spikes of every decode (pre-smoothing)
    seen_spikes, models = [], []
    real_load_run = ea.load_run

    def spying_load_run(*a, **k):
        bundle = real_load_run(*a, **k)
        models.append(weakref.ref(bundle.model))
        bundle.model.neural_encoder.encoder.register_forward_pre_hook(
            lambda mod, args, kwargs: seen_spikes.append((args[0] if args else kwargs["spikes"]).detach().clone()),
            with_kwargs=True)
        return bundle

    monkeypatch.setattr(ea, "load_run", spying_load_run)
    names = ["none", "drop_v6v", "only_55b", "time_second_half", "scramble_patches", "reverse_time"]
    monkeypatch.setattr(stage2, "generate", _float32(stage2.generate))
    rc = ea.main(["--runs", run, "--device", "cpu", "--ablations", *names])
    out = capsys.readouterr().out
    assert rc == 0 and "ERROR" not in out, out[-3000:]
    gc.collect()
    assert len(models) == 1 and models[0]() is None  # the run's model is freed before the next run loads
    s = json.loads((results / run / "ablations" / "summary.json").read_text())
    r = s["results"]["test"]
    assert list(r) == names and not s["errors"] and len(loads) == 1
    assert r["none"]["matches_saved"] is True  # the rebuilt, BEST-loaded model decodes as the original
    decoded = {n: ea.read_decodes(results / run / "ablations" / n / "test_greedy.csv")["decoded"].tolist()
               for n in names}
    assert len(set(decoded["none"])) == 3 and any(decoded[n] != decoded["none"] for n in names[1:])  # not degenerate
    assert s["model_class"] == config.model.model_class and s["patch_size"] == 5
    assert s["load"]["file"] == stage2.SLIM
    if s["model_class"] == "BCI_DiffusionLLM":
        assert s["sampler"]["mask_id"] == TINY_MASK

    fill = np.array(s["fill"]["vector"], dtype=np.float32)
    assert np.array_equal(fill, ea.train_mean(raw["train"], 512)[0])
    assert np.allclose(fill, np.concatenate([x["spikes"] for x in raw["train"]]).mean(0), atol=1e-6)
    # one encoder call per AR decode, several per diffusion decode: keep each decode's first, in order
    per_decode = _first_call_per_trial(seen_spikes, [len(x["spikes"]) for x in raw["test"]], len(names))
    for k, (name, x) in enumerate(((n, x) for n in names for x in raw["test"])):
        got = per_decode[k][0].numpy()
        want = ea.make_transform(CFG["ablations"][name], CFG, fill, 5)
        want = x["spikes"] if want is None else want(x["spikes"], k % 3)
        assert got.shape == x["spikes"].shape and np.array_equal(got, want), (name, k)
    v6v = ea.array_columns(CFG, ["v6v"])
    assert np.allclose(per_decode[3][0].numpy()[:, v6v], fill[v6v])  # drop_v6v, trial 0


def _float32(generate):
    """stage2.generate in float32 (CPU bf16 autocast is slow and not what the GPU's numerics test anyway)."""
    def run(*a, **k):
        return generate(*a, **(k | {"dtype": torch.float32}))
    return run


def _first_call_per_trial(calls, lengths, n_conditions):
    """Group the encoder's recorded inputs by decode: a new decode starts whenever the trial's length changes or the
    same length recurs after a different input (consecutive identical inputs = one diffusion decode's passes)."""
    out = []
    for x in calls:
        if out and x.shape == out[-1].shape and torch.equal(x, out[-1]):
            continue
        out.append(x)
    assert len(out) == len(lengths) * n_conditions, (len(out), len(calls))
    return out
