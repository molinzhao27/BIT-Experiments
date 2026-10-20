"""Neural feature preprocessing (official Kunz et al. convention) and trial-window extraction.

Per session file: [TX | SBP] -> block-wise mean subtraction -> Gaussian smoothing
(within each block, MATLAB `gaussSmooth_fast` edge normalization) -> z-score over
all bins of the file. Results are cached in data/processed/ as float32.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d

from .io import Session, load_session

ARRAY_ALIASES = {"ventral6v": "i6v"}  # T16 per-behavior/interleaved files
CACHE_VERSION = 1


@dataclass
class Features:
    """Preprocessed session: x is (T, 2C) = [TX channels | SBP channels]."""

    x: np.ndarray
    session: Session

    def array_columns(self, array: str) -> np.ndarray:
        ch = self.session.arrays[array]
        return np.concatenate([ch, ch + self.session.tx.shape[1]])


def gauss_smooth(x: np.ndarray, sigma_bins: float, truncate: float = 5.0) -> np.ndarray:
    """Centered Gaussian smoothing along axis 0, renormalized at the edges (like gaussSmooth_fast)."""
    if sigma_bins <= 0:
        return x
    wing = int(np.ceil(truncate * sigma_bins))
    k = np.exp(-0.5 * (np.arange(-wing, wing + 1) / sigma_bins) ** 2)
    k /= k.sum()
    num = convolve1d(x, k, axis=0, mode="constant", cval=0.0)
    den = convolve1d(np.ones(len(x)), k, mode="constant", cval=0.0)
    return num / den[:, None]


def preprocess(s: Session, sigma_ms: float, truncate: float = 5.0, center_blocks: bool = True,
               zscore: bool = True) -> np.ndarray:
    """Official pipeline by default. center_blocks=False, zscore=False gives the smoothed raw
    features (used by the strict, train-statistics-only normalization in Phase 2)."""
    x = np.hstack([s.tx, s.sbp]).astype(np.float64)
    for b in np.unique(s.block):
        m = s.block == b
        if center_blocks:
            x[m] -= x[m].mean(0)
        x[m] = gauss_smooth(x[m], sigma_ms / s.bin_ms, truncate)
    if not zscore:
        return x.astype(np.float32)
    sd = x.std(0)
    sd[sd == 0] = 1.0
    return ((x - x.mean(0)) / sd).astype(np.float32)


def raw_block_means(s: Session) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw [TX | SBP] means: (block ids, per-block table (n_blocks, 2C), grand mean over all bins).

    Expand to bins with `table[np.searchsorted(ids, s.block)]`.
    """
    x = np.hstack([s.tx, s.sbp]).astype(np.float64)
    ids = np.unique(s.block)
    return ids, np.stack([x[s.block == b].mean(0) for b in ids]), x.mean(0)


def canonical_arrays(s: Session) -> Session:
    s.arrays = {ARRAY_ALIASES.get(k, k): v for k, v in s.arrays.items()}
    return s


def load_features(path: Path, cache_dir: Path, sigma_ms: float, truncate: float = 5.0,
                  center_blocks: bool = True, zscore: bool = True) -> Features:
    """Load a session and its preprocessed features, using/refreshing the on-disk cache."""
    s = canonical_arrays(load_session(path))
    params = {"v": CACHE_VERSION, "sigma_ms": sigma_ms, "truncate": truncate, "src": path.name,
              "size": path.stat().st_size}
    if not (center_blocks and zscore):  # keep the default pipeline's cache key unchanged
        params |= {"center_blocks": center_blocks, "zscore": zscore}
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    tag = "" if center_blocks and zscore else "_smoothraw"
    cache = cache_dir / f"{path.stem.removesuffix('_raw')}_sigma{sigma_ms:g}{tag}_{key}.npy"
    if cache.exists():
        x = np.load(cache)
    else:
        x = preprocess(s, sigma_ms, truncate, center_blocks, zscore)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npy")
        np.save(tmp, x)
        tmp.rename(cache)
        cache.with_suffix(".json").write_text(json.dumps(params, indent=1))
    return Features(x=x, session=s)


def window_means(x: np.ndarray, anchors: np.ndarray, block: np.ndarray, starts: np.ndarray,
                 length: int) -> np.ndarray:
    """Mean of x over [anchor + start, anchor + start + length) for every trial x start.

    Windows are clipped to the trial's own recording block (blocks are concatenated
    with unrecorded gaps between them); a window entirely outside its block falls back
    to the `length` bins at the nearest block edge. Returns (n_trials, n_starts, n_feat).
    """
    cs = np.vstack([np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    # block extent of each trial
    tb = block[anchors]
    edges = {b: (int(np.flatnonzero(block == b)[0]), int(np.flatnonzero(block == b)[-1]) + 1)
             for b in np.unique(tb)}
    b0 = np.array([edges[b][0] for b in tb])[:, None]
    b1 = np.array([edges[b][1] for b in tb])[:, None]
    lo = anchors[:, None] + starts[None, :]
    hi = lo + length
    lo_c, hi_c = np.maximum(lo, b0), np.minimum(hi, b1)
    after, before = lo_c >= b1, hi_c <= b0
    lo_c = np.where(after, np.maximum(b1 - length, b0), lo_c)
    hi_c = np.where(after, b1, hi_c)
    lo_c = np.where(before, b0, lo_c)
    hi_c = np.where(before, np.minimum(b0 + length, b1), hi_c)
    out = (cs[hi_c] - cs[lo_c]) / (hi_c - lo_c)[..., None]
    return out.astype(np.float32)
