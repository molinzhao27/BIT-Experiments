"""Human self-supervised pretraining corpus (Phase 4b) in BIT's record format.

Threshold crossings only, 20 ms bins. Plan, decisions and BIT references: docs/SSL_PLAN.md.

Sources (each adapter returns raw, un-normalized `Segment`s per split):
- Brain-to-Text '25, T15 (Card et al. 2024): `hdf5_data_final/<session>/data_{train,val,test}.hdf5`,
  one group per trial with `input_features` (T, 512) = [TX 0:256 | SBP 256:512]. The release is
  already z-scored (rolling, preceding 20 trials). Read from the extracted directory or the zip.
- Brain-to-Text '24, T12 (Willett et al. 2023): `competitionData/{train,test,competitionHoldOut}/t12.<date>.mat`,
  `tx1` = 1 x N cell of (T, 256) counts (-3.5 RMS); BIT uses electrodes 0:128 (area 6v). Read from the
  extracted directory or the .tar.gz.
- Kunz et al. 2025 (T12, T15, T16, T17): isolated / interleaved verbal-behavior files (`*_allBehaviors_raw.mat`
  excluded: they duplicate the per-behavior files) and the `.mat` members of `sentenceData.zip`. Continuous
  `binnedTX` at 10 or 20 ms, summed to 20 ms within each block.

Splits follow BIT exactly: B2T "val" is BIT's `split_val` half (seed 42) of the official val split; the other
half (BIT's "test") and the unlabeled competition test/holdout splits are never read as neural data.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

BIN_MS = 20
FORMAT_VERSION = 1
KUNZ_NAME_RE = re.compile(r"^(t\d+)\.(\d{4}\.\d{2}\.\d{2})_(.+?)(?:_raw)?\.mat$")
KUNZ_VARS = ["binnedTX", "blockNum", "binSize", "goTrialEpochs", "delayTrialEpochs"]


@dataclass
class Segment:
    """One pretraining sequence before normalization."""

    x: np.ndarray  # (T, C) float32 at 20 ms
    day: str  # "2023.08.15"
    block: int  # block number as recorded in the source (1-based)
    source: str  # "<file or session>#<trial / chunk>"


def _usable(p: Path) -> bool:
    """An archive file, or a directory in BIT's layout (B2T'25 hdf5_data_final/, B2T'24 [competitionData/]train/)."""
    if p.is_file():
        return p.suffix == ".zip" or p.name.endswith((".tar.gz", ".tgz"))
    return p.is_dir() and any((p / q).is_dir() for q in ("hdf5_data_final", "train", "competitionData/train"))


def first_existing(paths, root: Path) -> Path | None:
    """First usable candidate path (relative to root) or None."""
    for p in [paths] if isinstance(paths, str) else paths:
        if _usable(root / p):
            return root / p
    return None


def bit_val_half(records: list, seed: int = 42) -> list:
    """BIT's `split_val` (data/*/prepare_data.py): the first half of a seeded permutation is "val"."""
    perm = np.random.RandomState(seed).permutation(len(records))
    return [records[i] for i in perm[: len(records) // 2]]


def _date_key(name: str) -> tuple:
    return tuple(name.split(".")[1:4])


# ---------------------------------------------------------------- Brain-to-Text '25 (T15)

def _b2t25_files(root: Path, split: str):
    """[(session, opener)] in BIT's session order; opener() -> open h5py.File."""
    if root.suffix == ".zip":
        zf = zipfile.ZipFile(root)
        found = {}
        for n in zf.namelist():
            m = re.search(r"hdf5_data_final/([^/]+)/data_(\w+)\.hdf5$", n)
            if m and m.group(2) == split:
                found[m.group(1)] = n
        return [(s, lambda n=found[s]: h5py.File(io.BytesIO(zf.read(n)), "r"))
                for s in sorted(found, key=_date_key)]
    base = root / "hdf5_data_final" if (root / "hdf5_data_final").is_dir() else root
    sessions = sorted((d.name for d in base.iterdir() if d.is_dir()), key=_date_key)
    return [(s, lambda p=base / s / f"data_{split}.hdf5": h5py.File(p, "r"))
            for s in sessions if (base / s / f"data_{split}.hdf5").exists()]


def load_b2t25(root: Path, channels=(0, 256), val_seed: int = 42) -> dict[str, list[Segment]]:
    """Train trials and BIT's val half of the val trials; TX columns `channels` of `input_features`."""
    lo, hi = channels
    out = {}
    for split in ("train", "val"):
        segs = []
        for session, open_ in _b2t25_files(root, split):
            with open_() as f:
                for key in f.keys():  # BIT iterates f.keys() (sorted trial_XXXX)
                    g = f[key]
                    segs.append(Segment(np.asarray(g["input_features"][:, lo:hi], dtype=np.float32),
                                        session.split(".", 1)[1], int(g.attrs["block_num"]),
                                        f"{session}/data_{split}#{key}"))
        out[split] = segs
    out["val"] = bit_val_half(out["val"], val_seed)
    return out


def b2t25_blocks(root: Path, dates: set[str]) -> dict[str, set[int]]:
    """Block numbers per date over all splits (metadata only), for de-duplication against Kunz files."""
    blocks = defaultdict(set)
    for split in ("train", "val", "test"):
        for session, open_ in _b2t25_files(root, split):
            date = session.split(".", 1)[1]
            if date in dates:
                with open_() as f:
                    blocks[date] |= {int(f[k].attrs["block_num"]) for k in f.keys()}
    return dict(blocks)


# ---------------------------------------------------------------- Brain-to-Text '24 (T12)

def _b2t24_mats(root: Path, splits, variables):
    """{split: [(file name, loadmat dict)]} sorted by date (BIT's order). Reads a directory or the .tar.gz."""
    out = {s: [] for s in splits}
    if root.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(root, "r:gz") as tf:
            for m in tf:  # one sequential pass; competitionHoldOut members are skipped unread
                parts = Path(m.name).parts
                if m.isfile() and len(parts) >= 2 and parts[-2] in out and parts[-1].endswith(".mat"):
                    out[parts[-2]].append((parts[-1], sio.loadmat(io.BytesIO(tf.extractfile(m).read()),
                                                                  variable_names=variables)))
    else:
        base = root / "competitionData" if (root / "competitionData").is_dir() else root
        for s in splits:
            out[s] = [(p.name, sio.loadmat(p, variable_names=variables)) for p in (base / s).glob("*.mat")]
    return {s: sorted(v, key=lambda t: _date_key(t[0])) for s, v in out.items()}


def load_b2t24(root: Path, feature: str = "tx1", channels=(0, 128), val_seed: int = 42) -> dict[str, list[Segment]]:
    """Train trials and BIT's val half of the competition "test" folder (BIT's val). Holdout is never read."""
    lo, hi = channels
    mats = _b2t24_mats(root, ("train", "test"), [feature, "blockIdx"])
    out = {}
    for split, folder in (("train", "train"), ("val", "test")):
        segs = []
        for name, d in mats[folder]:
            cells, blocks = d[feature], np.asarray(d["blockIdx"]).ravel()
            for i in range(cells.shape[1]):
                segs.append(Segment(np.asarray(cells[0, i][:, lo:hi], dtype=np.float32),
                                    ".".join(name.split(".")[1:4]), int(blocks[i]), f"{folder}/{name}#{i}"))
        out[split] = segs
    out["val"] = bit_val_half(out["val"], val_seed)
    return out


# ---------------------------------------------------------------- Kunz et al. 2025

@dataclass
class KunzFile:
    name: str
    participant: str  # "T15"
    date: str
    tag: str  # behavior / sentence-task tag from the file name
    collection: str  # "isolated" | "interleaved" | "sentences"
    opener: object  # () -> path or file object accepted by scipy.io.loadmat

    def load(self) -> dict:
        return sio.loadmat(self.opener(), variable_names=KUNZ_VARS)


def kunz_files(root: Path, isolated=None, interleaved=None, sentences=None, archives=(),
               skip_eval: bool = False) -> list[KunzFile]:
    """Kunz session files in a fixed order. `allBehaviors` files are skipped (exact unions of the per-behavior files);
    with `skip_eval`, so are `*_eval.mat` files (closed-loop evaluation sessions, kept as held-out data).
    `archives`: further Kunz zips read like `sentences` (collection = zip stem)."""
    out = []

    def add(name, collection, opener):
        m = KUNZ_NAME_RE.match(name)
        if m and m.group(3) != "allBehaviors" and not (skip_eval and m.group(3).endswith("_eval")):
            out.append(KunzFile(name, m.group(1).upper(), m.group(2), m.group(3), collection, opener))

    for collection, d in (("isolated", isolated), ("interleaved", interleaved)):
        if d and (root / d).is_dir():
            for p in sorted((root / d).glob("*.mat")):
                add(p.name, collection, lambda p=p: p)
    zips = ([("sentences", sentences)] if sentences else []) + [(Path(a).stem, a) for a in archives or ()]
    for collection, a in zips:
        if (root / a).exists():
            zf = zipfile.ZipFile(root / a)
            for n in sorted(zf.namelist()):
                if "__MACOSX" not in n:
                    add(Path(n).name, collection, lambda n=n, zf=zf: io.BytesIO(zf.read(n)))
    return out


def block_runs(block: np.ndarray) -> list[tuple[int, int, int]]:
    """(block id, start, end) for each contiguous run of equal block ids."""
    edges = np.flatnonzero(np.diff(block)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(block)]
    return [(int(block[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def rebin(x: np.ndarray, factor: int) -> np.ndarray:
    """Sum `factor` consecutive bins (counts; kept as uint16 to halve memory); a trailing partial bin is dropped."""
    n = len(x) // factor
    dtype = np.uint16 if np.issubdtype(x.dtype, np.integer) else np.float32
    return x[: n * factor].reshape(n, factor, x.shape[1]).sum(1, dtype=dtype)


def split_even(n: int, max_len: int, min_len: int) -> list[tuple[int, int]]:
    """Cut [0, n) into ceil(n / max_len) near-equal pieces (each <= max_len); none if n < min_len."""
    if n < min_len:
        return []
    k = -(-n // max_len)
    b = np.linspace(0, n, k + 1).round().astype(int)
    return list(zip(b[:-1].tolist(), b[1:].tolist()))


def kunz_segments(d: dict, name: str, date: str, mode: str = "blocks", chunk_bins: int = 500,
                  min_bins: int = 150, max_bins: int = 2500, exclude_blocks=()) -> tuple[list[Segment], dict]:
    """Cut one Kunz file into 20 ms TX segments.

    mode "blocks": every recorded block (continuous) in near-equal chunks of <= `chunk_bins`.
    mode "trials": one segment per trial, [delay onset, go end), pieces of <= `max_bins`.
    Returns (segments, info) with info = bin size and the blocks skipped/used.
    """
    bin_ms = int(np.asarray(d["binSize"]).ravel()[0])
    info = {"bin_ms": bin_ms, "blocks": [], "excluded_blocks": []}
    if BIN_MS % bin_ms:
        info["skipped"] = f"binSize {bin_ms} ms does not divide {BIN_MS} ms"
        return [], info
    f = BIN_MS // bin_ms
    tx, block = d["binnedTX"], np.asarray(d["blockNum"]).ravel().astype(np.int64)
    segs = []
    runs = block_runs(block)
    for b, s, e in runs:
        if b in exclude_blocks:
            info["excluded_blocks"].append(b)
            continue
        info["blocks"].append(b)
        if mode == "blocks":
            xb = rebin(tx[s:e], f)
            for k, (c0, c1) in enumerate(split_even(len(xb), chunk_bins, min_bins)):
                segs.append(Segment(xb[c0:c1], date, b, f"{name}#b{b}c{k}"))
    if mode == "trials":
        starts = np.asarray(d["delayTrialEpochs"])[:, 0].astype(np.int64) - 1  # 1-based inclusive -> 0-based
        ends = np.asarray(d["goTrialEpochs"])[:, 1].astype(np.int64)
        for i, (t0, t1) in enumerate(zip(starts, ends)):
            b, s, e = next(r for r in runs if r[1] <= t0 < r[2])
            if b in exclude_blocks:
                continue
            t0 = s + (t0 - s + f - 1) // f * f  # first full 20 ms bin of the block's grid
            xt = rebin(tx[t0:min(t1, e)], f)
            for k, (c0, c1) in enumerate(split_even(len(xt), max_bins, min_bins)):
                segs.append(Segment(xt[c0:c1], date, b, f"{name}#t{i}p{k}"))
    elif mode != "blocks":
        raise ValueError(f"unknown Kunz mode {mode!r}")
    return segs, info


# ---------------------------------------------------------------- normalization

def _norm_key(scope: str, seg: Segment):
    return {"day": seg.day, "block": (seg.day, seg.block), "subject": "all"}[scope]


def fit_norm(splits: dict[str, list[Segment]], scope: str = "day", stats_split: str = "train",
             min_sd: float = 1e-2) -> dict:
    """Per-feature mean/sd per normalization key.

    scope "day": that day's `stats_split` bins (a day without such bins falls back to all of its own bins);
    "block": that block's own bins (label-free self-normalization); "subject": all `stats_split` bins.
    """
    acc = {}

    def add(seg, tag):
        k = _norm_key(scope, seg)
        a = acc.setdefault(k, {"train": [0, 0.0, 0.0], "own": [0, 0.0, 0.0]})[tag]
        x = seg.x.astype(np.float64)
        a[0] += len(x)
        a[1] = a[1] + x.sum(0)
        a[2] = a[2] + (x * x).sum(0)

    for split, segs in splits.items():
        for seg in segs:
            add(seg, "train" if (split == stats_split and scope != "block") else "own")
    stats = {}
    for k, a in acc.items():
        n, s1, s2 = a["train"] if a["train"][0] else a["own"]
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
        stats[k] = {"mean": mu, "sd": np.maximum(sd, min_sd), "n_bins": int(n),
                    "source": stats_split if a["train"][0] else "own"}
    return stats


def apply_norm(seg: Segment, stats: dict, scope: str, clip: float | None = None) -> np.ndarray:
    if scope == "none":
        z = seg.x.astype(np.float32)
    else:
        st = stats[_norm_key(scope, seg)]
        z = (seg.x - st["mean"]) / st["sd"]
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


def block_then_day(splits: dict[str, list[Segment]], stats_split: str = "train", min_sd: float = 1e-2):
    """BIT's Brain-to-Text '24 loader: z-score each block with its own bins, then each day with the day's
    `stats_split` (block-normalized) bins. Returns (splits of block-normalized Segments, day stats)."""
    bstats = fit_norm(splits, "block", stats_split, min_sd)
    zb = {sp: [Segment(apply_norm(seg, bstats, "block"), seg.day, seg.block, seg.source) for seg in v]
          for sp, v in splits.items()}
    return zb, fit_norm(zb, "day", stats_split, min_sd)




# ---------------------------------------------------------------- corpus files

def write_split(path: Path, zs, segs: list[Segment], day_idx: list[int], block_idx: list[int], attrs: dict) -> None:
    """Stream normalized segments `zs` into one HDF5 file: concatenated `spikes` + `offsets` + per-segment ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    c = attrs["n_channels"]
    with h5py.File(tmp, "w") as f:
        ds = f.create_dataset("spikes", shape=(0, c), maxshape=(None, c), dtype="f4", chunks=(1024, c),
                              compression="lzf")
        offsets = [0]
        for z in zs:
            ds.resize(offsets[-1] + len(z), axis=0)
            ds[offsets[-1]:] = z
            offsets.append(offsets[-1] + len(z))
        f["offsets"] = np.asarray(offsets, dtype=np.int64)
        f["day_idx"] = np.asarray(day_idx, dtype=np.int64)
        f["block_idx"] = np.asarray(block_idx, dtype=np.int64)
        f["block"] = np.asarray([s.block for s in segs], dtype=np.int64)
        f["source"] = np.asarray([s.source for s in segs], dtype=h5py.string_dtype())
        f.attrs["meta"] = json.dumps(attrs)
    tmp.rename(path)


def read_split(path: Path) -> tuple[list[np.ndarray], dict]:
    """(list of (T, C) float32 arrays, per-segment arrays + meta) from one corpus file."""
    with h5py.File(path, "r") as f:
        x, off = f["spikes"][:], f["offsets"][:]
        meta = {"meta": json.loads(f.attrs["meta"]), "day_idx": f["day_idx"][:], "block_idx": f["block_idx"][:],
                "block": f["block"][:], "source": f["source"].asstr()[:]}
    return [x[a:b] for a, b in zip(off[:-1], off[1:])], meta


def load_corpus(out_dir: Path, splits=("train", "val")) -> tuple[dict[str, list[dict]], dict]:
    """Corpus -> ({split: [BIT record]}, manifest). A record has BIT's keys `spikes`, `day_idx`, `block_idx`
    plus `stitch_idx` (read-in/out index, see `bit_stitch_tables`), `stitch` and `source`."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    stitch = list(manifest["stitch_groups"])
    data = {s: [] for s in splits}
    for entry in manifest["files"]:
        if entry["split"] not in data:
            continue
        xs, m = read_split(out_dir / entry["file"])
        si = stitch.index(entry["stitch"])
        data[entry["split"]] += [
            {"spikes": x, "day_idx": np.asarray(d), "block_idx": np.asarray(b), "stitch_idx": np.asarray(si),
             "stitch": entry["stitch"], "source": src}
            for x, d, b, src in zip(xs, m["day_idx"], m["block_idx"], m["source"])]
    return data, manifest


def bit_stitch_tables(manifest: dict, feature: str = "tx1") -> tuple[dict, dict]:
    """(DATASET_TO_IDX, DATASET_CHANNELS) for BIT's utils/datasets.py, in the order ReadIn/ReadOut enumerate."""
    groups = manifest["stitch_groups"]
    return {g: i for i, g in enumerate(groups)}, {feature: {g: groups[g]["n_channels"] for g in groups}}


# ---------------------------------------------------------------- builder

def order_stitch_groups(groups: dict, order: list[str]) -> dict:
    """Groups listed in `order` first (BIT keys read-in/read-out modules by position: brandman_2024_text = 0,
    willett_2023_text = 1), then the rest sorted by name."""
    names = [g for g in order if g in groups] + sorted(g for g in groups if g not in order)
    return {g: groups[g] for g in names}


def stitch_name(dataset: str, subject: str, by: str) -> str:
    return subject if by == "subject" else f"{dataset}_{subject}"


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def iter_groups(cfg: dict, root: Path, only=None, log=print):
    """Yield ((dataset, subject), {split: [Segment]}) one group at a time, then the provenance dict as
    (None, prov). Kunz blocks that are also B2T'25 trials (same participant, day, block) are dropped."""
    ds_cfg, prov = cfg["datasets"], {}
    want = lambda name: ds_cfg.get(name, {}).get("enabled", False) and (not only or name in only)  # noqa: E731
    kz = ds_cfg.get("kunz", {})
    kfiles = (kunz_files(root, kz.get("isolated"), kz.get("interleaved"), kz.get("sentences"), kz.get("archives", ()),
                         kz.get("skip_eval_files", False)) if want("kunz") else [])
    collections = ["isolated", "interleaved", "sentences"]
    kfiles.sort(key=lambda k: (k.participant, collections.index(k.collection) if k.collection in collections else 9,
                               k.collection, k.name))

    # static list from the config, extended by a scan of the B2T'25 files when they are present
    exclude = defaultdict(set)
    for subj, days in (kz.get("exclude_blocks") or {}).items():
        for day, blocks in days.items():
            exclude[(subj.upper(), day)] |= set(blocks)
    for name, loader in (("b2t25", load_b2t25), ("b2t24", load_b2t24)):
        if not want(name):
            continue
        c = ds_cfg[name]
        path = first_existing(c["path"], root)
        if path is None:
            prov[name] = {"status": "skipped: raw files absent", "candidates": c["path"]}
            log(f"[{name}] skipped (none of {c['path']} exists)")
            continue
        if name == "b2t25" and kfiles:
            dates = {k.date for k in kfiles if k.participant == c["subject"]}
            for day, blocks in b2t25_blocks(path, dates).items():
                if blocks - exclude[(c["subject"], day)]:
                    log(f"[b2t25] Kunz {c['subject']} {day}: also excluding blocks "
                        f"{sorted(blocks - exclude[(c['subject'], day)])} (missing from the config)")
                exclude[(c["subject"], day)] |= blocks
        log(f"[{name}] reading {path}")
        kw = {"channels": tuple(c["channels"]), "val_seed": c.get("val_seed", 42)}
        if name == "b2t24":
            kw["feature"] = c.get("feature", "tx1")
        prov[name] = {"status": "built", "path": _rel(path, root)}
        yield (name, c["subject"]), loader(path, **kw)

    if want("kunz") and not kfiles:
        prov["kunz"] = {"status": "skipped: raw files absent"}
    seen, files_info, segs_by_subj = set(), [], defaultdict(list)
    for i, kf in enumerate(kfiles):
        d = kf.load()
        dup = {b for b in np.unique(np.asarray(d["blockNum"])).tolist() if (kf.participant, kf.date, b) in seen}
        segs, info = kunz_segments(d, kf.name, kf.date, kz.get("mode", "blocks"), kz.get("chunk_bins", 500),
                                   cfg["min_bins"], cfg["max_bins"], exclude[(kf.participant, kf.date)] | dup)
        seen |= {(kf.participant, kf.date, b) for b in info["blocks"]}
        n_bins = sum(len(s.x) for s in segs)
        files_info.append(info | {"file": kf.name, "collection": kf.collection, "participant": kf.participant,
                                  "duplicate_blocks": sorted(dup), "n_channels": int(d["binnedTX"].shape[1]),
                                  "n_segments": len(segs), "hours": round(n_bins * BIN_MS / 3.6e6, 3)})
        segs_by_subj[kf.participant] += segs
        log(f"[kunz] {kf.name}: {len(segs)} segments" + (f" ({info['skipped']})" if "skipped" in info else ""))
        if i + 1 == len(kfiles) or kfiles[i + 1].participant != kf.participant:
            yield ("kunz", kf.participant), {"train": segs_by_subj.pop(kf.participant)}
    if kfiles:
        prov["kunz"] = {"status": "built", "mode": kz.get("mode", "blocks"), "files": files_info}
    yield None, prov


def build_corpus(cfg: dict, root: Path, out_dir: Path, only=None, log=print) -> dict:
    """Build the normalized corpus (one HDF5 per dataset x subject x split) and manifest.json in out_dir."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_all = cfg["normalization"]
    files, stitch_groups, norm_summary, days, blocks = [], {}, {}, {}, {}
    for key, splits in iter_groups(cfg, root, only, log):
        if key is None:
            prov = splits
            break
        dataset, subject = key
        norm = norm_all | (norm_all.get("per_dataset", {}).get(dataset) or {})
        norm.pop("per_dataset", None)
        scope, min_sd, clip = norm["scope"], norm.get("min_sd", 1e-2), norm.get("clip")
        stitch = stitch_name(dataset, subject, cfg.get("stitch_by", "dataset_subject"))
        stitch = (cfg.get("stitch_merge") or {}).get(stitch, stitch)  # shared read-in (verified channel identity)
        # drop sequences shorter than min_bins; cut longer than max_bins (BIT's max_F) into near-equal pieces
        splits = {s: [Segment(seg.x[a:b], seg.day, seg.block, seg.source + (f"p{k}" if len(seg.x) > cfg["max_bins"] else ""))
                      for seg in v if len(seg.x) >= cfg["min_bins"]
                      for k, (a, b) in enumerate(split_even(len(seg.x), cfg["max_bins"], 1))]
                  for s, v in splits.items()}
        n_ch = {seg.x.shape[1] for v in splits.values() for seg in v}
        if len(n_ch) != 1:
            raise ValueError(f"{dataset}/{subject}: channel counts {n_ch}")
        n_ch = n_ch.pop()
        g = stitch_groups.setdefault(stitch, {"n_channels": n_ch, "subject": subject, "datasets": []})
        if g["n_channels"] != n_ch:
            raise ValueError(f"stitch group {stitch}: {g['n_channels']} vs {n_ch} channels")
        g["datasets"].append(dataset)
        if scope == "block_then_day":
            splits, stats = block_then_day(splits, norm.get("stats_split", "train"), min_sd)
            scope = "day"
        elif scope == "none":
            stats = {}
        else:
            stats = fit_norm(splits, scope, norm.get("stats_split", "train"), min_sd)
        zmax, n_big, n_all = 0.0, 0, 0
        for split, segs in splits.items():
            if not segs:
                continue

            def normalized(segs=segs):
                nonlocal zmax, n_big, n_all
                for seg in segs:
                    z = apply_norm(seg, stats, scope, clip)
                    a = np.abs(z)
                    zmax, n_big, n_all = max(zmax, float(a.max())), n_big + int((a >= 10).sum()), n_all + a.size
                    yield z

            di = [days.setdefault((stitch, dataset, seg.day), len(days)) for seg in segs]
            bi = [blocks.setdefault((stitch, dataset, seg.day, seg.block), len(blocks)) for seg in segs]
            name = f"{dataset}_{subject.lower()}_{split}.h5"
            attrs = {"dataset": dataset, "subject": subject, "stitch": stitch, "split": split, "bin_ms": BIN_MS,
                     "n_channels": n_ch, "feature": "tx", "normalization": norm, "format_version": FORMAT_VERSION}
            write_split(out_dir / name, normalized(), segs, di, bi, attrs)
            lens = np.array([len(seg.x) for seg in segs])
            files.append({"file": name, "dataset": dataset, "subject": subject, "stitch": stitch, "split": split,
                          "n_segments": len(segs), "n_bins": int(lens.sum()),
                          "hours": round(int(lens.sum()) * BIN_MS / 3.6e6, 3), "n_days": len({s.day for s in segs}),
                          "len_bins_min_median_max": [int(lens.min()), int(np.median(lens)), int(lens.max())]})
            log(f"wrote {name}: {len(segs)} segments, {files[-1]['hours']} h")
        keys = list(stats)
        summary = {"scope": norm["scope"], "clip": clip, "max_abs_z": round(zmax, 2),
                   "frac_abs_z_gt_10": n_big / max(1, n_all)}
        if keys:
            np.savez(out_dir / f"norm_{dataset}_{subject.lower()}.npz", keys=np.asarray([json.dumps(k) for k in keys]),
                     mean=np.stack([stats[k]["mean"] for k in keys]), sd=np.stack([stats[k]["sd"] for k in keys]))
            sds = np.stack([stats[k]["sd"] for k in keys])
            summary |= {"n_keys": len(keys), "stats_source": sorted({stats[k]["source"] for k in keys}),
                        "median_sd": round(float(np.median(sds)), 4), "n_sd_at_floor": int((sds <= min_sd).sum())}
        norm_summary[f"{dataset}_{subject}"] = summary
        del splits
    totals = defaultdict(lambda: {"n_segments": 0, "hours": 0.0})
    for f in files:
        t = totals[f"{f['dataset']}:{f['split']}"]
        t["n_segments"] += f["n_segments"]
        t["hours"] = round(t["hours"] + f["hours"], 3)
    manifest = {
        "format_version": FORMAT_VERSION, "bin_ms": BIN_MS, "feature": "threshold crossings", "config": cfg,
        "stitch_groups": order_stitch_groups(stitch_groups, cfg.get("stitch_order", [])), "files": files,
        "totals": dict(totals),
        "total_hours": {s: round(sum(f["hours"] for f in files if f["split"] == s), 3) for s in ("train", "val")},
        "normalization": norm_summary, "days": [list(k) for k in days], "sources": prov}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest
