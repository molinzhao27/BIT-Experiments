"""Frame-level phoneme labels from CTC forced alignment with arm A's phoneme model, and batched frozen-encoder features
(docs/RETRAINING_ABLATIONS.md, docs/PROBES_FOILS.md). The labels fit the phoneme eraser (train split only) and train
and score the frame-level phoneme probes. They never train or select a decoder (DESIGN.md §5).

- **Label sequences:** BIT's own (`create_phonemes_ctc_labels`): g2p_en, i.e. the CMU dictionary plus its G2P for
  words it lacks, stress removed, SIL after every word. These are the sequences arm A was trained on. Vocabulary:
  external/bit/vocab.json (BLANK = 0, 39 phonemes, SIL = 40).
- **Frames:** the NDT's patch tokens (5 bins = 100 ms). They are the time base of every encoder output and of the CTC
  posteriors; a trial of T bins has ceil(T / 5) of them.
- **Alignment:** Viterbi over the CTC topology of the trial's label sequence (optional blanks between labels; a label
  repeated back to back needs a blank between its two copies). Trials too short for their labels stay unlabelled.
- **Segment labels:** each label owns its run of frames on the Viterbi path. The blank frames between two labels are
  split at the midpoint (the earlier label gets the middle frame). Up to `margin` blank frames before the first label
  (after the last) belong to it; the other frames are unlabelled (-1).

Python 3.10-compatible (runs in BIT's environment); numpy only, except `run_encoder` (torch, BIT on sys.path).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

BLANK = 0
N_VOCAB = 41      # BLANK, 39 phonemes, SIL (external/bit/vocab.json)
N_CLASSES = 40    # frame classes: vocab ids 1..40 -> 0..39 (phonemes + SIL; blank is never a segment label)
PATCH = 5         # bins per NDT patch token (configs/models/encoder/ndt.yaml embedder.patch_size)
NEG = -1e30


def n_patches(n_bins: int, patch: int = PATCH) -> int:
    return -(-int(n_bins) // patch)


# ---------------------------------------------------------------- forced alignment

def ctc_viterbi(logp: np.ndarray, labels: Sequence[int], blank: int = BLANK) -> np.ndarray | None:
    """Most likely CTC path of `labels` through log-posteriors logp [T, V]: the state per frame in the extended
    sequence [blank, l1, blank, l2, ..., lL, blank] (label j sits at state 2j + 1). None if T is too short."""
    logp = np.asarray(logp, dtype=np.float64)
    labels = [int(x) for x in labels]
    if not labels:
        return None
    ext = np.full(2 * len(labels) + 1, blank)
    ext[1::2] = labels
    n_t, n_s = logp.shape[0], ext.size
    skip = np.zeros(n_s, bool)  # s may be entered from s - 2 (a label different from the previous label)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    emit = logp[:, ext]  # [T, S]
    alpha = np.full(n_s, NEG)
    alpha[0], alpha[1] = emit[0, 0], emit[0, 1]
    back = np.zeros((n_t, n_s), np.int8)  # 0 stay, 1 from s - 1, 2 from s - 2
    for t in range(1, n_t):
        stay = alpha
        one = np.concatenate(([NEG], alpha[:-1]))
        two = np.where(skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        cand = np.stack([stay, one, two])
        back[t] = cand.argmax(0)
        alpha = cand.max(0) + emit[t]
    ends = [n_s - 1, n_s - 2]
    end = ends[int(np.argmax(alpha[ends]))]
    if alpha[end] <= NEG / 2:
        return None
    states = np.empty(n_t, np.int64)
    s = end
    for t in range(n_t - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    if states[0] > 1:
        return None
    return states


def path_score(logp: np.ndarray, states: np.ndarray, labels: Sequence[int], blank: int = BLANK) -> float:
    """Log-probability of a CTC path given as extended-sequence states."""
    ext = np.full(2 * len(labels) + 1, blank)
    ext[1::2] = labels
    return float(np.asarray(logp, np.float64)[np.arange(len(states)), ext[states]].sum())


def segment_labels(states: np.ndarray, labels: Sequence[int], margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(frame label [T] (vocab id, -1 = unlabelled), frame token [T] (index into labels, -1)) from a Viterbi path."""
    n_t, n_l = len(states), len(labels)
    token = np.full(n_t, -1, np.int64)
    on = states % 2 == 1
    token[on] = (states[on] - 1) // 2
    starts = np.array([np.flatnonzero(token == j)[0] for j in range(n_l)])
    ends = np.array([np.flatnonzero(token == j)[-1] for j in range(n_l)])
    for j in range(n_l - 1):  # blank frames between label j and j + 1: midpoint split, the middle one to j
        gap = np.arange(ends[j] + 1, starts[j + 1])
        if gap.size:
            cut = ends[j] + 1 + (gap.size + 1) // 2
            token[ends[j] + 1:cut], token[cut:starts[j + 1]] = j, j + 1
    token[max(0, starts[0] - margin):starts[0]] = 0
    token[ends[-1] + 1:ends[-1] + 1 + margin] = n_l - 1
    lab = np.where(token >= 0, np.asarray(labels, np.int64)[np.clip(token, 0, None)], -1)
    return lab, token


def align(logp: np.ndarray, labels: Sequence[int], margin: int = 2) -> tuple[np.ndarray, np.ndarray, bool]:
    """Segment labels of one trial; all -1 (and False) when the trial is too short for its label sequence."""
    states = ctc_viterbi(logp, labels)
    if states is None:
        return np.full(len(logp), -1, np.int64), np.full(len(logp), -1, np.int64), False
    lab, tok = segment_labels(states, labels, margin)
    return lab, tok, True


# ---------------------------------------------------------------- greedy CTC decoding (sanity check of arm A's model)

def greedy_ctc(logp: np.ndarray, blank: int = BLANK, bit: bool = False) -> list[int]:
    """Argmax path, collapsed. Standard CTC: repeats merge only when adjacent on the path. bit=True: BIT's
    format_ctc (utils/eval_llm.py), which also merges a label repeated across blanks."""
    out, last = [], -1
    for i in np.asarray(logp).argmax(-1).tolist():
        if bit:
            if i != last and i != blank:
                out.append(i)
                last = i
        else:
            if i != last and i != blank:
                out.append(i)
            last = i
    return out


def edit_distance(a: Sequence, b: Sequence) -> int:
    try:
        import editdistance

        return int(editdistance.eval(list(a), list(b)))
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def per(hyps: list[Sequence[int]], refs: list[Sequence[int]]) -> float:
    """Corpus phoneme error rate: total edits / total reference phonemes."""
    return sum(edit_distance(h, r) for h, r in zip(hyps, refs, strict=True)) / max(1, sum(map(len, refs)))


# ---------------------------------------------------------------- label sequences

def bit_phoneme_sequences(sentences: list[str], vocab_file: Path) -> list[list[int]]:
    """BIT's phoneme label sequences (vocab ids) for these sentences, with its own function (needs g2p_en)."""
    from data.willett_2023_text.prepare_data import create_phonemes_ctc_labels

    ds = create_phonemes_ctc_labels({"x": [{"sentence": s} for s in sentences]}, str(vocab_file))
    return [[int(i) for i in r["phonemes_idx"]] for r in ds["x"]]


# ---------------------------------------------------------------- cache (content-hashed, like text_targets)

def _digest(arrays: dict[str, np.ndarray], meta: dict) -> str:
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode())
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def pack(rows: list[np.ndarray], dtype=np.int16) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate variable-length rows; offsets [n + 1]."""
    offsets = np.zeros(len(rows) + 1, np.int64)
    offsets[1:] = np.cumsum([len(r) for r in rows])
    flat = np.concatenate([np.asarray(r) for r in rows]).astype(dtype) if rows else np.zeros(0, dtype)
    return flat, offsets


def unpack(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def save_labels(out_dir: Path, per_split: dict[str, dict[str, list]], meta: dict) -> Path:
    """per_split[split] = {"labels": [frame labels], "tokens": [frame tokens], "phonemes": [label sequences],
    "sentences": [str]}. Writes <out_dir>/<sha12>.npz + .json and LATEST."""
    arrays = {}
    for split, d in per_split.items():
        for key in ("labels", "tokens", "phonemes"):
            arrays[f"{split}/{key}"], arrays[f"{split}/{key}_offsets"] = pack(d[key])
        arrays[f"{split}/sentences"] = np.asarray(d["sentences"], dtype=object)
    plain = {k: v for k, v in arrays.items() if v.dtype != object}
    sha = _digest(plain, meta | {"sentences": {s: d["sentences"] for s, d in per_split.items()}})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sha[:12]}.npz"
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta | {"sha256": sha, "splits": sorted(per_split)}, indent=1,
                                                    default=str))
    (out_dir / "LATEST").write_text(path.name + "\n")
    return path


def load_labels(path: Path) -> tuple[dict[str, dict[str, list]], dict]:
    """Inverse of save_labels; verifies the content hash."""
    path = Path(path)
    if path.is_dir():
        path = path / (path / "LATEST").read_text().strip()
    z = np.load(path, allow_pickle=True)
    meta = json.loads(path.with_suffix(".json").read_text())
    out = {}
    for split in meta["splits"]:
        out[split] = {k: unpack(z[f"{split}/{k}"], z[f"{split}/{k}_offsets"]) for k in ("labels", "tokens", "phonemes")}
        out[split]["sentences"] = [str(s) for s in z[f"{split}/sentences"]]
    plain = {k: z[k] for k in z.files if z[k].dtype != object}
    stored = {k: v for k, v in meta.items() if k not in ("sha256", "splits")}
    if _digest(plain, stored | {"sentences": {s: d["sentences"] for s, d in out.items()}}) != meta["sha256"]:
        raise ValueError(f"{path}: content hash mismatch; the label cache was modified")
    return out, meta | {"file": str(path)}


def concept(labels: np.ndarray, tokens: np.ndarray, phonemes: np.ndarray, offsets: Sequence[int] = (0,)) -> np.ndarray:
    """Per frame, the class (0..39) of the label `o` positions away in the trial's label sequence, for each o in
    offsets ([T, len(offsets)], -1 where there is none). offsets (0,) = the frame's own label."""
    out = np.full((len(labels), len(offsets)), -1, np.int64)
    ph = np.asarray(phonemes, np.int64)
    for c, o in enumerate(offsets):
        j = tokens + o
        ok = (tokens >= 0) & (j >= 0) & (j < len(ph))
        out[ok, c] = ph[j[ok]] - 1
    return out


# ---------------------------------------------------------------- data and frozen encoders over a split

def load_t15(config, limit: int | None = None) -> dict:
    """BIT's T15 loader as train_stage2.py calls it (TX + SBP; train, BIT-val, BIT-test; the unlabeled holdout
    dropped), without the LLM labels. limit: first N trials per split (smoke tests)."""
    from registry import dataset_registry

    data_config = config.data
    data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/brandman_2024_text"
    module = __import__(dataset_registry["brandman_2024_text"], fromlist=["load_data"])
    ds = module.load_data(**data_config.dataset)
    ds.pop("holdout", None)
    return {k: v[:limit] for k, v in ds.items()} if limit else ds


def build_ndt_encoder(config, features: str = "all"):
    """Stage 2's NDT encoder (BIT's sentence-stage NDT: masking and noise off, smoothing 2, as stage1.build_encoder)
    for `features` ("all" = TX + SBP, "tx1" = TX only), with Stage 2's pad-masked smoothing; eval mode, random init."""
    from models.ndt import NDT

    from bit_experiments import stage1, stage2

    kw = dict(config.method.model_kwargs)
    kw["features"] = features
    with stage1.in_bit_dir():
        enc = NDT(config.model.encoder, **kw).encoder
    enc.embedder.mask_active, enc.embedder.mask_ratio = False, 0.0
    enc.smooth_and_noise.noise, enc.smooth_and_noise.smooth = False, 2.0
    enc.smooth_and_noise = stage2.PadMaskedSmoothing(enc.smooth_and_noise)
    return enc.eval()


def run_encoder(encode: Callable, records: list[dict], batch_size: int = 16, device="cpu",
                dtype=None) -> Iterator[tuple[int, np.ndarray]]:
    """(trial index, features [n_patches, d] float32) for every record, in length-sorted batches.

    encode(batch) -> (features [B, P, d], mask [B, P]) for a BIT-collated batch (stage1.collate's keys); features may
    also be a dict of such tensors (several representations from one pass), and then a dict of arrays is yielded.
    With stage2.PadMaskedSmoothing in the encoder, a trial's features in a batch equal its features alone. dtype:
    autocast dtype on CUDA (bf16, as Stage 2) or None (float32)."""
    import torch

    from bit_experiments import stage1

    device = torch.device(device)
    order = np.argsort([len(r["spikes"]) for r in records], kind="stable")
    items = [{"spikes": r["spikes"], "spikes_mask": np.ones(len(r["spikes"]), np.int64),
              "spikes_timestamp": np.arange(len(r["spikes"])), "dataset_name": np.array(0),
              "row": np.array(0)} for r in records]
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch = stage1.collate([items[j] for j in idx], device)
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype or torch.bfloat16,
                                             enabled=dtype is not None and device.type == "cuda"):
            feats, mask = encode(batch)
        many = isinstance(feats, dict)
        feats = {k: v.float().cpu().numpy() for k, v in (feats if many else {"x": feats}).items()}
        for k, j in enumerate(idx):
            n = n_patches(len(records[j]["spikes"]))
            if int(mask[k].sum()) != n:
                raise ValueError(f"trial {j}: {int(mask[k].sum())} valid patches, expected {n}")
            out = {name: v[k, :n] for name, v in feats.items()}
            yield int(j), (out if many else out["x"])
