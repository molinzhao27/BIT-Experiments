"""BIT SSL pretraining wrapper (Phase 4b) on a tiny synthetic corpus, CPU only.

The corpus is built by ssl_data.build_corpus from files in each source's documented format (as in
test_ssl_data.py), with low-dimensional smooth latents so that masked bins are predictable.
"""

import copy
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import scipy.io as sio
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")  # imported by BIT's models/ndt.py
if not (ROOT / "external" / "bit" / "models" / "ndt.py").exists():
    pytest.skip("external/bit not cloned", allow_module_level=True)

from kunzdecoder import bit_ssl, ssl_data  # noqa: E402

GROUPS = {"b2t25_T15": 256, "b2t24_T12": 128, "kunz_T12": 128}


# ---------------------------------------------------------------- synthetic corpus with structure

def _rates(T: int, w: np.ndarray, rng) -> np.ndarray:
    """(T, C) smooth signal: 3 slow sinusoids with random phase, mixed by w (3, C)."""
    t = np.arange(T)[:, None]
    z = np.sin(2 * np.pi * rng.uniform(0.01, 0.04, 3) * t + rng.uniform(0, 2 * np.pi, 3))
    return z @ w


def _write_corpus_sources(root: Path) -> None:
    rng = np.random.default_rng(0)
    w25, w24, wk = (rng.normal(size=(3, c)) / np.sqrt(3) for c in (256, 128, 128))
    # B2T'25: hdf5_data_final/<session>/data_<split>.hdf5, input_features = [TX 0:256 | SBP 256:512]
    for split, n in (("train", 24), ("val", 12)):
        p = root / "brandman_2024_text" / "hdf5_data_final" / "t15.2023.08.13" / f"data_{split}.hdf5"
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            for i in range(n):
                T = int(rng.integers(90, 200))
                x = np.concatenate([_rates(T, w25, rng) + 0.3 * rng.normal(size=(T, 256)),
                                    rng.normal(size=(T, 256))], 1).astype(np.float32)
                g = f.create_group(f"trial_{i:04d}")
                g["input_features"] = x
                g.attrs.update({"block_num": 1 + i // 8, "trial_num": i})
    # B2T'24: <split>/t12.<date>.mat, tx1 = 1 x n cell of (T, 256) counts, blockIdx
    for split, n in (("train", 24), ("test", 12)):
        (root / "willett_2023_text" / split).mkdir(parents=True, exist_ok=True)
        cells = np.empty((1, n), dtype=object)
        for i in range(n):
            T = int(rng.integers(90, 200))
            lam = np.exp(0.2 + 0.8 * np.concatenate([_rates(T, w24, rng), np.zeros((T, 128))], 1))
            cells[0, i] = rng.poisson(lam).astype(np.uint8)
        sio.savemat(root / "willett_2023_text" / split / "t12.2022.05.05.mat",
                    {"tx1": cells, "blockIdx": (1 + np.arange(n) // 6).astype(np.int32)[:, None]})
    # Kunz sentences: continuous 20 ms binnedTX + blockNum in sentenceData.zip
    T = 1500
    tx = rng.poisson(np.exp(0.2 + 0.8 * _rates(T, wk, rng))).astype(np.uint8)
    sio.savemat(root / "k.mat", {"binnedTX": tx, "blockNum": np.full((T, 1), 2, np.uint8), "binSize": np.array([[20]]),
                                 "goTrialEpochs": np.array([[1, 10]]), "delayTrialEpochs": np.array([[1, 2]])})
    with zipfile.ZipFile(root / "sentenceData.zip", "w") as zf:
        zf.write(root / "k.mat", "sentenceData/t12.2022.06.01_attempted_sentences.mat")


CORPUS_CFG = {
    "min_bins": 75, "max_bins": 2500, "stitch_by": "dataset_subject", "stitch_order": ["b2t25_T15", "b2t24_T12"],
    "normalization": {"scope": "day", "stats_split": "train", "min_sd": 0.01, "clip": 10,
                      "per_dataset": {"b2t25": {"scope": "none"}}},
    "datasets": {
        "b2t25": {"enabled": True, "subject": "T15", "path": ["brandman_2024_text"], "channels": [0, 256]},
        "b2t24": {"enabled": True, "subject": "T12", "path": ["willett_2023_text"], "feature": "tx1",
                  "channels": [0, 128]},
        "kunz": {"enabled": True, "isolated": None, "interleaved": None, "sentences": "sentenceData.zip",
                 "mode": "blocks", "chunk_bins": 150, "skip_eval_files": True}},
}


def _tiny_pretrain() -> dict:
    """ssl_human.yaml `pretrain`, shrunk for CPU: small transformer, 2 epochs, fp32."""
    p = yaml.safe_load((ROOT / "configs" / "phase4" / "ssl_human.yaml").read_text())["pretrain"]
    p["model"]["transformer"] |= {"hidden_size": 32, "n_layers": 2, "n_heads": 2, "inter_size": 64}
    p["model"]["embedder"]["max_F"] = 500
    p["training"] |= {"num_epochs": 2, "train_batch_size": 4, "test_batch_size": 8, "mixed_precision": None,
                      "save_every": None}
    p["optimizer"]["lr"] = 5e-3
    return p


@pytest.fixture(scope="module")
def bit():
    b = bit_ssl.import_bit()
    orig = {"to_idx": dict(b.datasets.DATASET_TO_IDX), "channels": copy.deepcopy(b.datasets.DATASET_CHANNELS)}
    yield SimpleNamespace(mod=b, orig=orig)
    bit_ssl.restore_stitch_tables(orig, b)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("ssl_src")
    _write_corpus_sources(root)
    out = root / "corpus"
    ssl_data.build_corpus(CORPUS_CFG, root, out, log=lambda s: None)
    return out


@pytest.fixture(scope="module")
def run(bit, corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("ssl_run")
    tr = bit_ssl.build_trainer({"pretrain": _tiny_pretrain()}, corpus, out, seed=42, device="cpu", bit=bit.mod)
    before = tr.evaluate("train")
    summary = tr.train()
    after = tr.evaluate("train")
    return SimpleNamespace(trainer=tr, out=out, before=before, after=after, summary=summary)


def _ndt(bit, features, encoder_cfg):
    with bit_ssl.bit_cwd(bit.dir):
        return bit.ndt.NDT({"encoder": encoder_cfg}, method_name="ssl", loss="mse", features=features)


# ---------------------------------------------------------------- tests

def test_table10_config(bit):
    p = yaml.safe_load((ROOT / "configs" / "phase4" / "ssl_human.yaml").read_text())
    cfg = bit_ssl.bit_config(p["pretrain"], 42, "b2t25_T15", bit.mod)
    with bit_ssl.bit_cwd(bit.mod.dir):
        enc = bit.mod.config.update_config(bit.mod.ndt.DEFAULT_CONFIG, cfg["model"])["encoder"]
    t, e = enc["transformer"], enc["embedder"]
    assert (t["hidden_size"], t["n_layers"], t["n_heads"], t["inter_size"], t["use_rope"]) == (384, 7, 6, 512, True)
    assert (t["dropout"], e["dropout"], e["patch_size"], e["max_F"]) == (0.4, 0.2, 5, 2500)
    assert e["masker"] == {"active": True, "ratio": 0.5, "expand_prob": 1.0, "max_timespan": 15}
    assert enc["smooth_and_noise"] == {"noise": True, "smooth_sd": 2, "white_noise_sd": 0.2, "constant_offset_sd": 0.05}
    assert cfg["method"]["model_kwargs"] == {"method_name": "ssl", "loss": "mse"}
    assert (cfg["training"]["num_epochs"], cfg["training"]["train_batch_size"], cfg["training"]["mixed_precision"]) \
        == (400, 64, "bf16")
    o = cfg["optimizer"]
    assert (o["lr"], o["wd"], o["scheduler"], o["gradient_accumulation_steps"]) == (5e-4, 1e-5, "cosine", 1)


def test_stitch_tables_patched_in_place(bit, corpus):
    D, N = bit.mod.datasets, bit.mod.ndt
    _, man = ssl_data.load_corpus(corpus)
    before = (dict(D.DATASET_TO_IDX), copy.deepcopy(D.DATASET_CHANNELS))
    with bit_ssl.stitch_tables(man, "tx1", bit.mod):
        assert D.DATASET_TO_IDX == {g: i for i, g in enumerate(GROUPS)}
        assert N.DATASET_CHANNELS is D.DATASET_CHANNELS and N.DATASET_CHANNELS["tx1"] == GROUPS
        assert D.DATASET_CHANNELS["all"] == bit.orig["channels"]["all"]  # other features untouched
        model = bit_ssl.build_ndt(bit_ssl.bit_config(_tiny_pretrain(), 42, "b2t25_T15", bit.mod), bit.mod)
        read_in = model.encoder.embedder.read_in.embed_dict
        assert [read_in[str(i)][1].in_features for i in range(3)] == [5 * c for c in GROUPS.values()]
        assert [model.decoder.read_out.readout_dict[str(i)][1].out_features for i in range(3)] == \
            [5 * c for c in GROUPS.values()]
    assert (dict(D.DATASET_TO_IDX), D.DATASET_CHANNELS) == before
    bad = copy.deepcopy(man)
    bad["stitch_groups"] = dict(reversed(list(man["stitch_groups"].items())))  # b2t25_T15 would get slot 2
    with pytest.raises(ValueError, match="slot"):
        bit_ssl.patch_stitch_tables(bad, "tx1", bit.mod)


def test_batches_are_single_group(run):
    tr = run.trainer
    seen = []
    for loader in (tr.train_loader, tr.eval_loaders["val"]):
        for inputs, unused in loader:
            g = inputs["dataset_name"].unique()
            assert len(g) == 1 and torch.equal(unused["stitch_idx"], inputs["dataset_name"])
            assert inputs["spikes"].shape[2] == list(GROUPS.values())[int(g)]
            seen += unused["source"]
    recs = tr.datasets["train"].dataset + tr.datasets["val"].dataset
    assert sorted(seen) == sorted(r["source"] for r in recs)  # every record exactly once per pass
    assert {r["stitch"] for r in tr.datasets["val"].dataset} == {"b2t25_T15", "b2t24_T12"}


def test_masked_r2_fixed_masks_deterministic(bit, run):
    tr = run.trainer
    a = tr.evaluate("val")
    torch.manual_seed(1234)
    torch.rand(1000)  # the global RNG must not matter
    b = tr.evaluate("val")
    assert a == b
    sel = tr.cfg["selection"]["val_groups"]
    assert a["selection_groups"] == sel
    assert a["r2_masked"] == pytest.approx(np.mean([a[f"r2_masked_{g}"] for g in sel]))
    assert all(a[f"n_bins_{g}"] > 0 for g in sel)
    other = bit_ssl.evaluate(tr.model, tr.eval_loaders["val"], tr.groups, 1, tr.device, selection_groups=sel)
    assert other["r2_masked"] != a["r2_masked"]  # a different mask seed gives different masks

    # bit_r2 is the value of BIT's own eval_neurons_metric
    inputs, unused = next(iter(tr.eval_loaders["val"]))
    tr.model.eval()
    with torch.no_grad():
        out = tr.model(**inputs)
    cfg = bit.mod.config.DictConfig({"method": {"model_kwargs": {"loss": "mse"},
                                                "dataloader_kwargs": {"pad_dict": bit_ssl.PAD_DICT}}})
    ref = bit.mod.eval_spike.eval_neurons_metric(tr.model, inputs, unused, out.to_dict(), cfg).item()
    assert bit_ssl.bit_r2(out.targets, out.preds) == pytest.approx(ref, abs=1e-5)


def test_two_epochs_reduce_masked_train_mse(run):
    for g in GROUPS:
        assert run.after[f"mse_masked_{g}"] < 0.98 * run.before[f"mse_masked_{g}"], g
    log = json.loads((run.out / "log.json").read_text())
    assert [r["epoch"] for r in log] == [1, 2]
    assert all(np.isfinite(r["train_loss"]) for r in log)
    for g in ("b2t25_T15", "b2t24_T12"):
        assert all(f"val_r2_masked_{g}" in r and f"val_bit_r2_{g}" in r for r in log)
    assert (run.out / "log.csv").read_text().splitlines()[0].startswith("epoch,train_loss")
    best = max(r["val_r2_masked"] for r in log)
    assert json.loads((run.out / "BEST" / "metrics.json").read_text())["r2_masked"] == pytest.approx(best)
    for d in ("BEST", "BEST_core", "LAST"):
        assert all((run.out / d / f).exists() for f in bit_ssl.CKPT_FILES), d
    assert (run.out / "LAST" / "optimizer.bin").exists()


def test_checkpoint_loads_with_bit_from_pt(bit, run):
    from safetensors.torch import load_file
    b, best, core = bit.mod, run.out / "BEST", run.out / "BEST_core"
    enc_cfg = torch.load(best / "encoder_config.pth", weights_only=True)  # plain dict (torch >= 2.6 default)
    torch.load(best / "trainer_config.pth", weights_only=True)
    saved = torch.load(best / "encoder.bin", weights_only=True)
    read_in = lambda keys, slots: sorted(k for k in keys if k.startswith(bit_ssl.READ_IN)  # noqa: E731
                                         and k[len(bit_ssl.READ_IN):].split(".")[0] in slots)
    patched = json.loads((best / "stitch_tables.json").read_text())
    assert patched["DATASET_TO_IDX"] == {g: i for i, g in enumerate(GROUPS)}

    # (a) pretraining tables: BIT's from_pt path, every key matches
    m = _ndt(b, "tx1", {"from_pt": str(best)})
    r = m.encoder.load_state_dict(saved, strict=False)
    assert r.missing_keys == [] and r.unexpected_keys == []
    assert all(torch.equal(v, saved[k]) for k, v in m.encoder.state_dict().items())

    # (b) BIT's own tx1 tables (2 slots, same shapes): only our extra read-in (kunz_T12, slot 2) is unexpected
    bit_ssl.restore_stitch_tables(bit.orig, b)
    try:
        m = _ndt(b, "tx1", {"from_pt": str(best)})
        r = m.encoder.load_state_dict(saved, strict=False)
        assert r.missing_keys == [] and sorted(r.unexpected_keys) == read_in(saved, {"2"})

        # (c) TX+SBP fine-tuning tables (512 / 256 ch): BIT's from_pt crashes on the read-in shapes ...
        with pytest.raises(RuntimeError, match="size mismatch"):
            _ndt(b, "all", {"from_pt": str(best)})
        # ... BEST_core loads through BIT's unmodified path: core = pretrained, read-ins fresh
        m = _ndt(b, "all", {"from_pt": str(core)})
        for k, v in m.encoder.state_dict().items():
            if k.startswith(bit_ssl.READ_IN):
                assert v.shape != saved[k].shape or k.endswith(("1.bias", "2.weight", "2.bias"))
            else:
                assert torch.equal(v, saved[k]), k
        # ... and load_core() loads the same core into a fresh TX+SBP model, reporting what it skipped
        fresh = _ndt(b, "all", enc_cfg)
        rep = bit_ssl.load_core(fresh, best, verbose=False)
        assert rep["reinit_read_in"] == ["0", "1"] and rep["skipped_shape"] == []
        assert rep["unexpected"] == read_in(saved, {"2"})
        assert rep["missing"] == read_in(fresh.encoder.state_dict(), {"0", "1"})
        assert all(torch.equal(fresh.encoder.state_dict()[k], saved[k]) for k in rep["loaded"])
        # (d) BCI_AudioLLM.load_safetensors path (sentence fine-tuning): model.safetensors, strict=False
        r = fresh.load_state_dict(load_file(str(core / "model.safetensors")), strict=False)
        assert r.unexpected_keys == []
        assert all(k.startswith(("encoder." + bit_ssl.READ_IN, "decoder." + bit_ssl.READ_OUT)) for k in r.missing_keys)
    finally:
        bit_ssl.patch_stitch_tables(run.trainer.manifest, "tx1", b)


def test_trivial_predictions_use_only_unmasked_bins():
    y = torch.zeros(1, 12, 2, dtype=torch.float64)
    y[0, :, 0] = torch.arange(12, dtype=torch.float64)
    y[0, :, 1] = 7.0
    masked = torch.zeros(1, 12, 2, dtype=torch.bool)
    masked[0, 4:8] = True
    y_leak = y.clone()
    y_leak[0, 4:8] = 1e6  # masked values must not influence either predictor
    valid = torch.ones_like(masked)
    for yy in (y, y_leak):
        q = bit_ssl.trivial_predictions(yy, masked, valid, width=3)
        seen = [0, 1, 2, 3, 8, 9, 10, 11]
        assert torch.allclose(q["trial_mean"][0, 5, 0], y[0, seen, 0].mean())
        # left stretch: mean of bins 2, 3 (bin 4 is masked) = 2.5; right: mean of bins 8, 9 = 8.5
        assert torch.allclose(q["neighbors"][0, 5, 0], torch.tensor(5.5, dtype=torch.float64))
        assert torch.allclose(q["neighbors"][0, 4:8, 1], torch.full((4,), 7.0, dtype=torch.float64))
