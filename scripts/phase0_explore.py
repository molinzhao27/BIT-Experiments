#!/usr/bin/env python
"""Phase 0: inventory every isolated / interleaved verbal-behavior session file.

    python scripts/phase0_explore.py            # writes results/phase0/inventory.json + sessions.csv

Prints per-file structure (arrays, bin size, cues, trials per cue, blocks,
epoch durations, data-quality checks) and checks that each `*_allBehaviors_raw.mat`
equals the union of that participant's per-behavior files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder.io import load_session  # noqa: E402

TASKS = ["isolatedVerbalBehaviors", "interleavedVerbalBehaviors"]


def describe(s) -> dict:
    ms = s.bin_ms
    tb = s.trial_block()
    word = np.array([s.cue_names[c - 1] for c in s.cues])
    behaviors_by_block = {}
    for b in np.unique(tb):
        names = word[tb == b]
        behaviors_by_block[int(b)] = dict(Counter(names))
    gaps = np.flatnonzero(np.diff(s.block) != 0)
    tx_var, sbp_var = s.tx.astype(float).var(0), s.sbp.var(0)
    return {
        "file": str(s.path.relative_to(ROOT)),
        "participant": s.participant,
        "date": s.date,
        "behavior": s.behavior,
        "bin_ms": ms,
        "n_bins": int(s.tx.shape[0]),
        "minutes": round(s.tx.shape[0] * ms / 60000, 1),
        "n_channels": int(s.tx.shape[1]),
        "arrays": {k: [int(v.min()), int(v.max()), int(v.size)] for k, v in s.arrays.items()},
        "dtypes": {"binnedTX": str(s.tx.dtype), "spikePow": str(s.sbp.dtype)},
        "n_cues": len(s.cue_names),
        "cue_names": s.cue_names,
        "n_trials": s.n_trials,
        "trials_per_cue": {s.cue_names[c - 1]: int(n) for c, n in zip(*np.unique(s.cues, return_counts=True))},
        "blocks": [int(b) for b in np.unique(s.block)],
        "trials_per_block": {int(b): int(n) for b, n in zip(*np.unique(tb, return_counts=True))},
        "cues_per_block": behaviors_by_block,
        "block_changes_in_time_series": int(gaps.size),
        "delay_ms_median": float(np.median(np.diff(s.delay, axis=1)) * ms),
        "go_ms_median": float(np.median(np.diff(s.go, axis=1)) * ms),
        "delay_ms_range": [int(np.diff(s.delay, axis=1).min() * ms), int(np.diff(s.delay, axis=1).max() * ms)],
        "go_ms_range": [int(np.diff(s.go, axis=1).min() * ms), int(np.diff(s.go, axis=1).max() * ms)],
        "delay_then_go": bool(np.all(s.delay[:, 1] <= s.go[:, 0])),
        "go_end_to_next_delay_bins": sorted(Counter((s.delay[1:, 0] - s.go[:-1, 1]).tolist()).most_common(3)),
        "trials_cross_block": int(np.sum(s.block[s.go[:, 0]] != s.block[np.minimum(s.go[:, 1] - 1, len(s.block) - 1)])),
        "nan": {"tx": int(np.isnan(s.tx.astype(float)).sum()), "sbp": int(np.isnan(s.sbp).sum())},
        "zero_var_channels": {"tx": int((tx_var == 0).sum()), "sbp": int((sbp_var == 0).sum())},
        "tx_max": int(s.tx.max()),
        "tx_mean_per_bin": round(float(s.tx.mean()), 4),
        "sbp_median": round(float(np.median(s.sbp)), 2),
        "sbp_p99": round(float(np.percentile(s.sbp, 99)), 2),
    }


def main() -> int:
    out_dir = ROOT / "results" / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        for f in sorted((ROOT / "data" / "raw" / task).glob("*_raw.mat")):
            s = load_session(f)
            r = describe(s)
            r["task"] = task
            rows.append(r)
            print(f"{s.participant} {s.date} {s.behavior:28s} bin={s.bin_ms}ms T={r['n_bins']:7d} "
                  f"({r['minutes']:5.1f} min) C={r['n_channels']} arrays={list(s.arrays)} "
                  f"trials={s.n_trials} blocks={r['blocks']} delay~{r['delay_ms_median']:.0f}ms "
                  f"go~{r['go_ms_median']:.0f}ms zero-var tx/sbp={r['zero_var_channels']}")

    # allBehaviors vs union of per-behavior files
    checks = {}
    iso = [r for r in rows if r["task"] == TASKS[0]]
    for p in sorted({r["participant"] for r in iso}):
        allb = next(r for r in iso if r["participant"] == p and r["behavior"] == "allBehaviors")
        singles = [r for r in iso if r["participant"] == p and r["behavior"] != "allBehaviors"]
        checks[p] = {
            "allBehaviors_trials": allb["n_trials"],
            "sum_single_trials": sum(r["n_trials"] for r in singles),
            "allBehaviors_bins": allb["n_bins"],
            "sum_single_bins": sum(r["n_bins"] for r in singles),
            "allBehaviors_blocks": allb["blocks"],
            "single_blocks": sorted(b for r in singles for b in r["blocks"]),
        }
        print(f"{p}: allBehaviors trials {checks[p]['allBehaviors_trials']} vs sum {checks[p]['sum_single_trials']}; "
              f"bins {checks[p]['allBehaviors_bins']} vs {checks[p]['sum_single_bins']}; "
              f"blocks equal: {checks[p]['allBehaviors_blocks'] == checks[p]['single_blocks']}")

    (out_dir / "inventory.json").write_text(json.dumps({"sessions": rows, "allBehaviors_checks": checks}, indent=1))
    flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
    flat.to_csv(out_dir / "sessions.csv", index=False)
    print(f"wrote {out_dir.relative_to(ROOT)}/inventory.json, sessions.csv")

    word_info_timecourse(ROOT / "figures" / "phase0_word_info_timecourse.png")
    word_info_timecourse(ROOT / "figures" / "phase0_word_info_timecourse_interleaved.png", task=TASKS[1])
    return 0


BEHAVIORS = ["attempted", "mouthed", "imagined1stMotor", "imagined1stAuditory",
             "imaginedListening", "listening", "reading"]


def _behavior_trials(task: str, participant: str, files: list[Path]):
    """Yield (behavior, Session, trial mask, word labels) for word trials of each behavior."""
    from kunzdecoder.io import parse_cue

    if task == TASKS[0]:
        for beh in BEHAVIORS:
            s = load_session(next(f for f in files if f.name.startswith(participant.lower())
                                  and f"_{beh}_raw" in f.name))
            yield beh, s, s.cues >= 2, s.cues  # cue 1 = DO_NOTHING
    else:
        s = load_session(next(f for f in files if f.name.startswith(participant.lower())))
        parsed = [parse_cue(s.participant, s.cue_names[c - 1]) for c in s.cues]
        words = np.array([w or "" for w, _ in parsed])
        behs = np.array([b for _, b in parsed])
        for beh in BEHAVIORS:
            mask = (behs == beh) & (words != "")
            if mask.any():
                yield beh, s, mask, words


def word_info_timecourse(fig_path: Path, task: str = TASKS[0], pre_ms: int = 2500, post_ms: int = 4500,
                         sigma_ms: int = 100) -> None:
    """Exploratory: mean one-way ANOVA F (7 words) across an array's 128 features, per bin, go-aligned.

    Features: block-mean-subtracted TX + SBP, Gaussian-smoothed (sigma_ms) within each block.
    F ~ 1 means no word information; this only shows *where in the trial* the information is.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    files = sorted((ROOT / "data" / "raw" / task).glob("*_raw.mat"))
    participants = sorted({f.name[:3].upper() for f in files})
    fig, axes = plt.subplots(len(participants), 4, figsize=(16, 3.2 * len(participants)), squeeze=False)
    colors = dict(zip(BEHAVIORS, plt.cm.tab10(np.arange(len(BEHAVIORS)))))
    cache: dict[Path, np.ndarray] = {}
    for pi, p in enumerate(participants):
        for beh, s, mask, labels in _behavior_trials(task, p, files):
            if s.path not in cache:
                cache.clear()
                x = np.hstack([s.tx, s.sbp]).astype(float)
                for b in np.unique(s.block):
                    m = s.block == b
                    x[m] -= x[m].mean(0)
                    x[m] = gaussian_filter1d(x[m], sigma_ms / s.bin_ms, axis=0, mode="nearest")
                cache[s.path] = x
            x = cache[s.path]
            pre, post = pre_ms // s.bin_ms, post_ms // s.bin_ms
            go0 = s.go[mask, 0]
            ok = (go0 - pre >= 0) & (go0 + post <= len(x))
            idx = go0[ok, None] + np.arange(-pre, post)[None]
            y = labels[mask][ok]
            t = np.arange(-pre, post) * s.bin_ms / 1000
            delay_s = -np.median(np.diff(s.delay[mask], axis=1)) * s.bin_ms / 1000
            for ai, (name, ch) in enumerate(s.arrays.items()):
                seg = x[:, np.concatenate([ch, ch + s.tx.shape[1]])][idx]  # trials x time x feat
                grand = seg.mean(0)
                k = np.unique(y)
                means = np.stack([seg[y == c].mean(0) for c in k])
                ss_b = sum((y == c).sum() * (means[i] - grand) ** 2 for i, c in enumerate(k))
                ss_w = sum(((seg[y == c] - means[i]) ** 2).sum(0) for i, c in enumerate(k))
                f = (ss_b / (len(k) - 1)) / (ss_w / (len(y) - len(k)))
                ax = axes[pi, ai]
                ax.plot(t, f.mean(1), color=colors[beh], lw=1, label=f"{beh} (n={len(y)})")
                ax.set_title(f"{p} {s.date} {name} ({ch.size} ch)", fontsize=9)
                ax.axvline(0, color="k", lw=0.5)
                ax.axhline(1, color="0.6", lw=0.5, ls=":")
                ax.axvline(delay_s, color=colors[beh], lw=0.4, ls="--")
        for ai in range(len(s.arrays), 4):
            axes[pi, ai].axis("off")
        axes[pi, 0].legend(fontsize=6, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("time from go-epoch onset (s); dashed = delay onset")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean F (7 words)")
    fig.suptitle(f"{task}: word information over the trial", y=1.0)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
