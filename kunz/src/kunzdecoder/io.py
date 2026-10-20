"""Load Kunz et al. 2025 session .mat files (MATLAB v5) into 0-based numpy structures.

Conventions (see docs/DATA_NOTES.md):
- `goTrialEpochs` / `delayTrialEpochs` are 1-based, inclusive [start, end] bin
  indices in the .mat files; here they become 0-based half-open [start, end).
- `trialCues` stay 1-based cue ids so that `cue_names[cue - 1]` is the cue text.
- `chanSets` are 1-based channel indices into the per-feature channel axis;
  here they become 0-based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

FILENAME_RE = re.compile(r"^(t\d+)\.(\d{4}\.\d{2}\.\d{2})_(.+)_raw\.mat$")


def _str(x) -> str:
    return str(np.asarray(x).ravel()[0]).strip() if np.asarray(x).size else ""


@dataclass
class Session:
    path: Path
    participant: str  # "T12"
    date: str  # "2023.08.15"
    behavior: str  # file tag, e.g. "attempted", "allBehaviors", "interleavedVerbalBehaviors"
    bin_ms: int
    tx: np.ndarray  # (T, C) threshold-crossing counts
    sbp: np.ndarray  # (T, C) spike-band power
    block: np.ndarray  # (T,) block number per bin
    go: np.ndarray  # (N, 2) 0-based [start, end)
    delay: np.ndarray  # (N, 2) 0-based [start, end)
    cues: np.ndarray  # (N,) 1-based cue ids
    cue_names: list[str]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # name -> 0-based channel idx

    @property
    def n_trials(self) -> int:
        return len(self.cues)

    def trial_block(self) -> np.ndarray:
        """Block number of each trial (block of the bin at go onset)."""
        return self.block[self.go[:, 0]]


# The bare "imagined" cue in the interleaved sessions is motoric inner speech for T12
# and auditory inner speech for T16 (official repo, analysis/Fig5_ABCD.ipynb).
_INTERLEAVED_IMAGINED = {"T12": "imagined1stMotor", "T16": "imagined1stAuditory"}


def parse_cue(participant: str, name: str) -> tuple[str | None, str]:
    """(word or None for DO_NOTHING, behavior) for any cue label in the dataset.

    Handles the isolated `allBehaviors` labels ("ban-attempted") and the three
    interleaved label styles ("ban\\n(attempted)$1" / "ban$0" for T12,
    "ban (imaginedlistening)" for T15/T17, "ban,listen" for T16).
    """
    low = name.lower()
    m = re.match(r"[(]?([a-z_]+)", low)
    # T12 silent reading shows a blank cue for DO_NOTHING ("" / "-reading")
    word = None if m is None or m.group(1) in ("do_nothing", "donothing") else m.group(1)
    if "-" in name and "(" not in name:  # allBehaviors: "<word>-<behavior>"
        return word, name.split("-", 1)[1]
    if word is None:  # interleaved blocks share one DO_NOTHING cue across behaviors
        return None, "doNothing"
    if "imaginedlistening" in low:
        return word, "imaginedListening"
    if "passivelistening" in low or "listen" in low or low.endswith("$0"):
        return word, "listening"
    if "attempted" in low:
        return word, "attempted"
    if "mimed" in low:
        return word, "mouthed"
    if "imagined" in low:
        return word, _INTERLEAVED_IMAGINED[participant]
    return word, "unknown"


def load_session(path: str | Path) -> Session:
    path = Path(path)
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unexpected session file name: {path.name}")
    d = sio.loadmat(path)
    epochs = lambda k: d[k].astype(np.int64) - np.array([1, 0])  # noqa: E731
    names = [_str(n) for n in np.asarray(d["chanSetNames"]).ravel()]
    sets = [np.asarray(s).ravel().astype(np.int64) - 1 for s in np.asarray(d["chanSets"]).ravel()]
    return Session(
        path=path,
        participant=m.group(1).upper(),
        date=m.group(2),
        behavior=m.group(3),
        bin_ms=int(np.asarray(d["binSize"]).ravel()[0]),
        tx=d["binnedTX"],
        sbp=d["spikePow"],
        block=np.asarray(d["blockNum"]).ravel().astype(np.int64),
        go=epochs("goTrialEpochs"),
        delay=epochs("delayTrialEpochs"),
        cues=np.asarray(d["trialCues"]).ravel().astype(np.int64),
        cue_names=[_str(c) for c in np.asarray(d["cueList"]).ravel()],
        arrays=dict(zip(names, sets)),
    )
