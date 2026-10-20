"""SSL corpus adapters (Phase 4b) on synthetic files in each source's documented format.

The B2T loaders are checked against BIT's own prepare_data.py (external/bit, read-only) so that our
"val" split is exactly BIT's split_val half.
"""

import importlib.util
import sys
import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import ssl_data  # noqa: E402

BIT = ROOT / "external" / "bit"


def _bit_module(rel: str, name: str):
    path = BIT / rel
    if not path.exists():
        pytest.skip("external/bit not cloned")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- fixtures in the documented formats

def make_b2t25(root: Path, sessions: dict, seed=0) -> Path:
    """hdf5_data_final/<session>/data_<split>.hdf5, one group per trial (Card et al. 2024 / BIT loader)."""
    rng = np.random.default_rng(seed)
    for session, splits in sessions.items():
        for split, blocks in splits.items():
            p = root / "hdf5_data_final" / session / f"data_{split}.hdf5"
            p.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(p, "w") as f:
                for i, b in enumerate(blocks):
                    g = f.create_group(f"trial_{i:04d}")
                    g["input_features"] = rng.normal(size=(80 + 7 * i + int(rng.integers(0, 5)), 512)).astype(np.float32)
                    g.attrs.update({"n_time_steps": len(g["input_features"]), "session": session, "block_num": b,
                                    "trial_num": i})
                    if split != "test":
                        g.attrs["sentence_label"] = f"sentence {i}"
    return root


def make_b2t24(root: Path, files: dict, seed=0, corrupt_holdout=False) -> Path:
    """competitionData-style <split>/t12.<date>.mat with tx1..tx4 / spikePow cells (Willett et al. 2023)."""
    rng = np.random.default_rng(seed)
    for split, dates in files.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for date, blocks in dates.items():
            p = root / split / f"t12.{date}.mat"
            if split == "competitionHoldOut" and corrupt_holdout:
                p.write_bytes(b"not a mat file")
                continue
            n = len(blocks)
            cells = {k: np.empty((1, n), dtype=object) for k in ("tx1", "tx2", "tx3", "tx4", "spikePow")}
            for i in range(n):
                t = 80 + 9 * i + int(rng.integers(0, 5))
                for k in cells:
                    cells[k][0, i] = (rng.poisson(1.0, size=(t, 256)).astype(np.uint8) if k != "spikePow"
                                      else rng.gamma(2.0, size=(t, 256)))
            sio.savemat(p, {**cells, "blockIdx": np.asarray(blocks, dtype=np.int32)[:, None],
                            "sentenceText": np.array([f"Hello world {i}." for i in range(n)])})
    return root


def make_kunz(path: Path, bin_ms=10, blocks=((3, 900), (5, 700)), n_ch=8, seed=0) -> dict:
    """Kunz-format session .mat: continuous binnedTX + blockNum, 1-based inclusive trial epochs."""
    rng = np.random.default_rng(seed)
    block = np.concatenate([np.full(n, b) for b, n in blocks])
    tx = rng.poisson(0.5, size=(len(block), n_ch)).astype(np.uint8)
    starts = np.cumsum([0] + [n for _, n in blocks])[:-1]
    delay = np.array([[s + 11, s + 210] for s in starts])  # 1-based inclusive
    go = np.array([[s + 211, s + 500] for s in starts])
    d = {"binnedTX": tx, "spikePow": rng.gamma(2.0, size=tx.shape), "blockNum": block[:, None].astype(np.uint8),
         "binSize": np.array([[bin_ms]]), "goTrialEpochs": go.astype(np.int32),
         "delayTrialEpochs": delay.astype(np.int32)}
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, d)
    return d


# ---------------------------------------------------------------- unit pieces

def test_rebin_split_even_block_runs():
    x = np.arange(14, dtype=np.uint8).reshape(7, 2)
    r = ssl_data.rebin(x, 2)
    assert r.dtype == np.uint16 and r.shape == (3, 2) and r[0].tolist() == [0 + 2, 1 + 3]
    assert ssl_data.split_even(1000, 500, 75) == [(0, 500), (500, 1000)]
    pieces = ssl_data.split_even(1001, 500, 75)
    assert len(pieces) == 3 and max(b - a for a, b in pieces) <= 500 and pieces[-1][1] == 1001
    assert ssl_data.split_even(50, 500, 75) == []
    assert ssl_data.block_runs(np.array([2, 2, 5, 5, 5, 2])) == [(2, 0, 2), (5, 2, 5), (2, 5, 6)]


def test_kunz_blocks_mode_rebins_within_blocks(tmp_path):
    d = make_kunz(tmp_path / "t15.2024.04.07_attempted_raw.mat")
    segs, info = ssl_data.kunz_segments(d, "f", "2024.04.07", "blocks", chunk_bins=200, min_bins=75)
    assert info["blocks"] == [3, 5]
    for b, n, s in ((3, 900, 0), (5, 700, 900)):
        mine = [g for g in segs if g.block == b]
        assert sum(len(g.x) for g in mine) == n // 2
        assert all(len(g.x) <= 200 for g in mine)
        # counts are summed in 10 ms pairs starting at the block's first bin (never across blocks)
        np.testing.assert_array_equal(np.concatenate([g.x for g in mine]),
                                      d["binnedTX"][s:s + n].reshape(-1, 2, 8).sum(1))
    segs, info = ssl_data.kunz_segments(d, "f", "2024.04.07", "blocks", 200, 75, exclude_blocks={5})
    assert {g.block for g in segs} == {3} and info["excluded_blocks"] == [5]


def test_kunz_trials_mode_and_unsupported_bin(tmp_path):
    d = make_kunz(tmp_path / "t15.2024.04.07_attempted_raw.mat")
    segs, _ = ssl_data.kunz_segments(d, "f", "2024.04.07", "trials", min_bins=75)
    # trial 0: delay onset bin 10 (0-based) .. go end 500 (exclusive) -> 245 bins at 20 ms
    assert [len(g.x) for g in segs] == [245, 245]
    np.testing.assert_array_equal(segs[0].x, d["binnedTX"][10:500].reshape(-1, 2, 8).sum(1))
    d["binSize"] = np.array([[33]])
    segs, info = ssl_data.kunz_segments(d, "f", "2024.04.07")
    assert segs == [] and "does not divide" in info["skipped"]


def test_kunz_files_skips_allbehaviors_and_reads_zip(tmp_path):
    make_kunz(tmp_path / "iso" / "t12.2023.08.15_attempted_raw.mat", bin_ms=20)
    make_kunz(tmp_path / "iso" / "t12.2023.08.15_allBehaviors_raw.mat", bin_ms=20)
    make_kunz(tmp_path / "s.mat")
    with zipfile.ZipFile(tmp_path / "sentenceData.zip", "w") as zf:
        zf.write(tmp_path / "s.mat", "sentenceData/t15.2024.03.03_attempted_50words.mat")
        zf.writestr("__MACOSX/sentenceData/._t15.2024.03.03_attempted_50words.mat", b"junk")
        zf.writestr("sentenceData/sentences_readme.txt", "readme")
    files = ssl_data.kunz_files(tmp_path, "iso", None, "sentenceData.zip")
    assert [(f.name, f.participant, f.date, f.collection) for f in files] == [
        ("t12.2023.08.15_attempted_raw.mat", "T12", "2023.08.15", "isolated"),
        ("t15.2024.03.03_attempted_50words.mat", "T15", "2024.03.03", "sentences")]
    assert files[1].load()["binnedTX"].shape == (1600, 8)


def test_fit_norm_uses_train_statistics_per_day():
    rng = np.random.default_rng(0)
    seg = lambda day, loc, n=500: ssl_data.Segment(rng.normal(loc, 2.0, (n, 3)).astype(np.float32), day, 1, "s")  # noqa: E731
    splits = {"train": [seg("d1", 5.0), seg("d2", -3.0)], "val": [seg("d1", 6.0), seg("d3", 1.0)]}
    stats = ssl_data.fit_norm(splits, "day")
    assert stats["d1"]["n_bins"] == 500 and stats["d1"]["source"] == "train"  # val bins not used
    assert stats["d3"]["source"] == "own"  # a day without training bins falls back to its own bins
    z = ssl_data.apply_norm(splits["train"][0], stats, "day")
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-4)
    assert abs(ssl_data.apply_norm(splits["val"][0], stats, "day").mean() - 0.5) < 0.15
    assert set(ssl_data.fit_norm(splits, "subject")) == {"all"}
    assert set(ssl_data.fit_norm(splits, "block")) == {("d1", 1), ("d2", 1), ("d3", 1)}


# ---------------------------------------------------------------- B2T adapters vs BIT's own loaders

B2T25_SESSIONS = {
    "t15.2023.08.13": {"train": [1, 1, 2, 2, 3], "val": [4, 4, 4], "test": [5, 5]},
    "t15.2023.08.11": {"train": [1, 2, 2]},
    "t15.2024.03.03": {"train": [1, 3, 3, 6], "val": [7, 7, 7, 7], "test": [8]},
}


def _same(a, b):
    return [x.shape for x in a] == [y.shape for y in b] and all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("as_zip", [False, True])
def test_b2t25_matches_bit_split(tmp_path, as_zip):
    root = make_b2t25(tmp_path / "brandman_2024_text", B2T25_SESSIONS)
    bit = _bit_module("data/brandman_2024_text/prepare_data.py", "bit_brandman_prepare")
    ref = bit.load_data(str(root), features=["tx1"], area_start=0, area_end=256)
    src = root
    if as_zip:
        src = tmp_path / "t15_copyTask_neuralData.zip"
        with zipfile.ZipFile(src, "w") as zf:
            for p in sorted(root.rglob("*.hdf5")):
                zf.write(p, p.relative_to(root).as_posix())
    ours = ssl_data.load_b2t25(src)
    assert _same([s.x for s in ours["train"]], [r["spikes"] for r in ref["train"]])
    assert _same([s.x for s in ours["val"]], [r["spikes"] for r in ref["val"]])
    assert not any(np.array_equal(s.x, r["spikes"]) for s in ours["val"] for r in ref["test"])
    assert not any("data_test" in s.source for v in ours.values() for s in v)
    assert [s.day for s in ours["train"]][:3] == ["2023.08.11"] * 3  # BIT's date-sorted session order
    assert ssl_data.b2t25_blocks(src, {"2024.03.03"}) == {"2024.03.03": {1, 3, 6, 7, 8}}


B2T24_FILES = {
    "train": {"2022.05.05": [2, 2, 3], "2022.04.28": [1, 1, 2, 4]},
    "test": {"2022.05.05": [5, 5, 5], "2022.04.28": [6, 6, 6, 6, 6]},
    "competitionHoldOut": {"2022.05.05": [7, 7]},
}


def test_b2t24_matches_bit_split(tmp_path):
    pytest.importorskip("transformers")  # imported at the top of BIT's willett prepare_data.py
    root = make_b2t24(tmp_path / "willett_2023_text", B2T24_FILES)
    bit = _bit_module("data/willett_2023_text/prepare_data.py", "bit_willett_prepare")
    ref = bit.load_data(str(root), features=["tx1"], area_start=0, area_end=128)
    ours = ssl_data.load_b2t24(root)
    assert _same([s.x for s in ours["train"]], [r["spikes"] for r in ref["train"]])
    assert _same([s.x for s in ours["val"]], [r["spikes"] for r in ref["val"]])
    assert [s.block - 1 for s in ours["train"]] == [r["block"] for r in ref["train"]]


def test_b2t24_never_reads_holdout_dir_or_tar(tmp_path):
    root = make_b2t24(tmp_path / "w", B2T24_FILES, corrupt_holdout=True)
    ours = ssl_data.load_b2t24(root)
    assert len(ours["train"]) == 7 and len(ours["val"]) == 4  # floor(8 / 2)
    tgz = tmp_path / "competitionData.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(root, arcname="./competitionData")
    from_tar = ssl_data.load_b2t24(tgz)
    assert _same([s.x for s in from_tar["val"]], [s.x for s in ours["val"]])
    assert ssl_data.first_existing(["w"], tmp_path) == root and ssl_data.first_existing(["nope"], tmp_path) is None


# ---------------------------------------------------------------- end to end

def _cfg():
    return {"min_bins": 75, "max_bins": 2500, "stitch_by": "dataset_subject",
            "normalization": {"scope": "day", "stats_split": "train", "min_sd": 0.01, "clip": None},
            "datasets": {
                "b2t25": {"enabled": True, "subject": "T15", "path": ["brandman_2024_text"], "channels": [0, 256]},
                "b2t24": {"enabled": True, "subject": "T12", "path": ["willett_2023_text"], "feature": "tx1",
                          "channels": [0, 128]},
                "kunz": {"enabled": True, "isolated": "iso", "interleaved": None, "sentences": "sentenceData.zip",
                         "mode": "blocks", "chunk_bins": 300, "exclude_blocks": {"T15": {"2024.03.03": [1]}}}},
            "stitch_order": ["b2t25_T15", "b2t24_T12"]}


def test_build_corpus_end_to_end(tmp_path):
    make_b2t25(tmp_path / "brandman_2024_text", B2T25_SESSIONS)
    make_b2t24(tmp_path / "willett_2023_text", B2T24_FILES, corrupt_holdout=True)
    make_kunz(tmp_path / "iso" / "t12.2023.08.15_attempted_raw.mat", bin_ms=20, n_ch=128)
    make_kunz(tmp_path / "iso" / "t12.2023.08.15_allBehaviors_raw.mat", bin_ms=20, n_ch=128)
    make_kunz(tmp_path / "s1.mat", blocks=((3, 900), (5, 700), (6, 800)), n_ch=256)  # blocks 3 and 6 are B2T'25 trials
    make_kunz(tmp_path / "s2.mat", blocks=((5, 600), (9, 800)), n_ch=256)  # block 5 repeats s1's block 5
    with zipfile.ZipFile(tmp_path / "sentenceData.zip", "w") as zf:
        zf.write(tmp_path / "s1.mat", "sentenceData/t15.2024.03.03_attempted_50words.mat")
        zf.write(tmp_path / "s2.mat", "sentenceData/t15.2024.03.03_imagined_50words.mat")
    out = tmp_path / "corpus"
    m = ssl_data.build_corpus(_cfg(), tmp_path, out, log=lambda s: None)

    assert m["stitch_groups"] == {"b2t24_T12": {"n_channels": 128, "subject": "T12", "datasets": ["b2t24"]},
                                  "b2t25_T15": {"n_channels": 256, "subject": "T15", "datasets": ["b2t25"]},
                                  "kunz_T12": {"n_channels": 128, "subject": "T12", "datasets": ["kunz"]},
                                  "kunz_T15": {"n_channels": 256, "subject": "T15", "datasets": ["kunz"]}}
    kunz_files = {f["file"]: f for f in m["sources"]["kunz"]["files"]}
    assert "t12.2023.08.15_allBehaviors_raw.mat" not in kunz_files
    s1, s2 = kunz_files["t15.2024.03.03_attempted_50words.mat"], kunz_files["t15.2024.03.03_imagined_50words.mat"]
    assert s1["blocks"] == [5] and s1["excluded_blocks"] == [3, 6]  # dynamic B2T'25 overlap
    assert s2["blocks"] == [9] and s2["duplicate_blocks"] == [5]
    assert m["totals"]["kunz:train"]["n_segments"] == 6 + 2 + 2 and "kunz:val" not in m["totals"]
    assert m["totals"]["b2t25:val"]["n_segments"] == 3 and m["totals"]["b2t24:val"]["n_segments"] == 4

    data, man = ssl_data.load_corpus(out)
    to_idx, channels = ssl_data.bit_stitch_tables(man)
    assert list(to_idx) == ["b2t25_T15", "b2t24_T12", "kunz_T12", "kunz_T15"]  # 0/1 as BIT's DATASET_TO_IDX
    assert list(to_idx) == list(channels["tx1"]) == list(man["stitch_groups"])
    rec = data["train"][0]
    assert {"spikes", "day_idx", "block_idx", "stitch_idx"} <= set(rec) and rec["spikes"].dtype == np.float32
    for r in data["train"] + data["val"]:
        assert r["spikes"].shape[1] == channels["tx1"][r["stitch"]] and 75 <= len(r["spikes"]) <= 2500
    # z-scored per day with training statistics
    by_day = {}
    for r in data["train"]:
        by_day.setdefault((r["stitch"], int(r["day_idx"])), []).append(r["spikes"])
    for x in by_day.values():
        x = np.concatenate(x)
        np.testing.assert_allclose(x.mean(0), 0, atol=1e-4)
        assert np.all((x.std(0) > 0.99) & (x.std(0) < 1.01))

    # BIT's collate function accepts the records (one stitch group per batch, as GroupBatchSampler does)
    torch = pytest.importorskip("torch")
    ds = _bit_module("utils/datasets.py", "bit_datasets")
    batch = [dict(r, spikes_mask=np.ones(len(r["spikes"]), np.int64), spikes_timestamp=np.arange(len(r["spikes"])))
             for r in data["val"] if r["stitch"] == "b2t25_T15"]
    pad = {"spikes": [{"dim": 0, "side": "right", "value": -100}], "spikes_mask": [{"dim": 0, "side": "right", "value": 0}],
           "spikes_timestamp": [{"dim": 0, "side": "right", "value": 0}]}
    model_inputs, _ = ds.pad_collate_fn(batch, ["spikes", "spikes_mask", "spikes_timestamp"], pad)
    assert model_inputs["spikes"].dtype == torch.float32 and model_inputs["spikes"].shape[2] == 256


def test_build_corpus_approved_defaults(tmp_path):
    """The approved Phase 4b choices: B2T'25 as released, block->day z elsewhere, clip 10, Kunz T15 sharing
    B2T'25's read-in, isolated/interleaved and *_eval files excluded."""
    make_b2t25(tmp_path / "brandman_2024_text", B2T25_SESSIONS)
    make_b2t24(tmp_path / "willett_2023_text", B2T24_FILES)
    make_kunz(tmp_path / "iso" / "t15.2024.04.07_attempted_raw.mat", bin_ms=10, n_ch=256)
    make_kunz(tmp_path / "s2.mat", blocks=((5, 600), (9, 800)), n_ch=256)
    make_kunz(tmp_path / "s3.mat", blocks=((2, 700),), n_ch=256, seed=3)
    with zipfile.ZipFile(tmp_path / "sentenceData.zip", "w") as zf:
        zf.write(tmp_path / "s2.mat", "sentenceData/t15.2024.03.03_imagined_50words.mat")
        zf.write(tmp_path / "s3.mat", "sentenceData/t15.2024.03.03_imagined_50words_eval.mat")
    cfg = _cfg()
    cfg["normalization"] = {"scope": "block_then_day", "stats_split": "train", "min_sd": 0.01, "clip": 10,
                            "per_dataset": {"b2t25": {"scope": "none"}}}
    cfg["stitch_merge"] = {"kunz_T15": "b2t25_T15"}
    cfg["datasets"]["kunz"] |= {"isolated": None, "skip_eval_files": True}
    m = ssl_data.build_corpus(cfg, tmp_path, tmp_path / "corpus", log=lambda s: None)

    assert m["stitch_groups"]["b2t25_T15"]["datasets"] == ["b2t25", "kunz"] and "kunz_T15" not in m["stitch_groups"]
    files = {f["file"] for f in m["sources"]["kunz"]["files"]}
    assert files == {"t15.2024.03.03_imagined_50words.mat"}  # no isolated, no *_eval
    data, _ = ssl_data.load_corpus(tmp_path / "corpus")
    # B2T'25 exactly as released (only clipped at 10)
    raw = ssl_data.load_b2t25(tmp_path / "brandman_2024_text")
    b25 = [r for r in data["val"] if r["stitch"] == "b2t25_T15"]
    for r, seg in zip(b25, raw["val"]):
        np.testing.assert_allclose(r["spikes"], np.clip(seg.x, -10, 10), rtol=0, atol=1e-6)
    # elsewhere: per-day z-scored train data, everything within the clip bound
    for name in ("b2t24", "kunz"):
        xs = [r["spikes"] for r in data["train"] if r["source"] and name in m["stitch_groups"][r["stitch"]]["datasets"]
              and r["stitch"] != "b2t25_T15"]
        if xs:
            x = np.concatenate(xs)
            assert np.abs(x).max() <= 10
    assert all(np.abs(r["spikes"]).max() <= 10 for r in data["train"] + data["val"])
    assert m["normalization"]["b2t25_T15"]["scope"] == "none" and m["normalization"]["b2t24_T12"]["scope"] == "block_then_day"
