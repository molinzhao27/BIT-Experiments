"""Candidate decoding windows and behavior 'events', shared by all phases.

All times are in ms relative to the go-epoch onset of each trial. A behavior's event is
the moment its word happens: stimulus onset for passive behaviors (the start of the
'delay' epoch = -nominal delay), the go cue for everything else. Transferring a window
between behaviors keeps its offset from the event: start_B = start_A - event_A + event_B.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def nominal_times(cfg1: dict, participant: str, behavior: str) -> tuple[float, float]:
    times = cfg1["behavior_times"][participant]
    return tuple(times.get(behavior, times["default"]))


def isolated_starts_ms(cfg1: dict, participant: str, behavior: str) -> np.ndarray:
    """Official candidate starts for the isolated task (see configs/phase1.yaml)."""
    delay_s, go_s = nominal_times(cfg1, participant, behavior)
    w = cfg1["window"]
    if behavior in cfg1["passive_behaviors"]:
        lo, hi = -delay_s * 1000, 0
    else:
        lo, hi = w["active_start_ms"], go_s * 1000 + w["active_end_after_go_ms"]
    return np.arange(lo, hi + 1e-6, w["step_ms"]).round().astype(int)


def isolated_event_ms(cfg1: dict, participant: str, behavior: str) -> int:
    if behavior in cfg1["passive_behaviors"]:
        return int(round(-nominal_times(cfg1, participant, behavior)[0] * 1000))
    return 0


def interleaved_starts_event(cfg1: dict, icfg: dict, behavior: str) -> tuple[np.ndarray, int]:
    """Candidate starts and event for one behavior of an interleaved session.

    icfg: {delay_s, go_s, listening_event: delay|go}. Listening is a stimulus-epoch
    behavior only when its audio plays in the delay (T12); otherwise it is go-aligned.
    """
    w = cfg1["window"]
    if behavior == "listening" and icfg["listening_event"] == "delay":
        lo, hi, event = -icfg["delay_s"] * 1000, 0, -icfg["delay_s"] * 1000
    else:
        lo, hi, event = w["active_start_ms"], icfg["go_s"] * 1000 + w["active_end_after_go_ms"], 0
    return np.arange(lo, hi + 1e-6, w["step_ms"]).round().astype(int), int(round(event))


def session_file(task_dir: Path, participant: str, behavior: str) -> Path:
    hits = sorted(task_dir.glob(f"{participant.lower()}.*_{behavior}_raw.mat"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{participant} {behavior}: {hits}")
    return hits[0]
