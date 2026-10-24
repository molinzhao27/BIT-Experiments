#!/usr/bin/env python
"""Paired comparisons of ablation runs against their controls (docs/RETRAINING_ABLATIONS.md §9), from saved decodes.
CPU, seconds; project venv or BIT's env.

    python scripts/ablation_report.py \
        --pair AR B_ssl_frozen_s42 B_ssl_phonerased_s42 \
        --pair diffusion B_ssl_aero-1-audio-diffusion_frozen_s42 B_ssl_aero-1-audio-diffusion_phonerased_s42 \
        --compare AR diffusion [--run-dir NAME=PATH ...] [--split test] [--out results/stage2/eraser_pair]

Per run, from <run>/decodes/<split>_greedy_{real,null_shuffled,null_prompt_only}.csv (BIT's WER, textmetrics):
- WER on the real input and under each null;
- the grounding gap G = WER(null_shuffled) - WER(real), how much the decoder's output depends on the trial's own
  neural data;
- the share of derailed sentences (sentence WER >= 1).

Per pair (control, ablated), with paired trial-bootstrap 95% CIs (the same resamples for every run and statistic):
- the cost, WER(ablated) - WER(control);
- the grounding kept, G(ablated) / G(control): 1 = the output still depends on the input as much as the control's; 0 = it
  no longer depends on it more than on another trial's (fluent output that ignores the signal).

Per --compare A B (two pairs on the same trials): B's cost - A's cost and B's grounding kept - A's, the
difference-in-differences that says which decoder loses more from the same ablation.
--run-dir points a run at another copy's results (e.g. NAME=/workspace/aero-diffusion/results/stage2/NAME).
Writes <out>.json and <out>.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments.textmetrics import word_errors  # noqa: E402

CONDITIONS = ("real", "null_shuffled", "null_prompt_only")


def load_run(path: Path, split: str) -> dict:
    """{condition: (refs, errors [T], n_words [T])} for the decodes that exist."""
    out = {}
    for cond in CONDITIONS:
        f = Path(path) / "decodes" / f"{split}_greedy_{cond}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, keep_default_na=False).sort_values("trial")
        err = np.array([word_errors(r, h) for r, h in zip(d.reference, d.decoded)])
        out[cond] = (d.reference.tolist(), err[:, 0].astype(float), err[:, 1].astype(float))
    if "real" not in out:
        raise FileNotFoundError(f"{path}: no {split}_greedy_real.csv")
    return out


def run_stats(e: dict, n: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """WER per condition and the grounding gap; idx: bootstrap resamples [B, T] (None = the sample itself)."""
    take = (lambda a: a.sum()) if idx is None else (lambda a: a[idx].sum(1))
    words = take(n)
    s = {c: take(v) / words for c, v in e.items()}
    if "null_shuffled" in s:
        s["gap"] = s["null_shuffled"] - s["real"]
    return s


def ci(x) -> list[float]:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=3, action="append", default=[], metavar=("LABEL", "CONTROL", "ABLATED"))
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("PAIR_A", "PAIR_B"))
    ap.add_argument("--run-dir", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "stage2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "stage2" / "eraser_pair")
    args = ap.parse_args(argv)
    if not args.pair:
        ap.error("at least one --pair")
    dirs = dict(x.split("=", 1) for x in args.run_dir)
    names = sorted({r for _, c, a in args.pair for r in (c, a)})
    runs = {r: load_run(Path(dirs.get(r, args.results / r)), args.split) for r in names}
    refs = runs[names[0]]["real"][0]
    for r, d in runs.items():
        for cond, (rf, _, _) in d.items():
            if rf != refs:
                raise ValueError(f"{r} {cond}: references differ from {names[0]}'s (not the same {args.split} trials)")
    n = runs[names[0]]["real"][2]
    errs = {r: {c: v[1] for c, v in d.items()} for r, d in runs.items()}
    idx = np.random.default_rng(args.seed).integers(0, len(n), (args.n_boot, len(n)))
    point = {r: run_stats(e, n) for r, e in errs.items()}
    boot = {r: run_stats(e, n, idx) for r, e in errs.items()}
    derail = {r: float(((e["real"] / np.maximum(n, 1)) >= 1).mean()) for r, e in errs.items()}
    report = {"split": args.split, "n_trials": int(len(n)), "n_boot": args.n_boot, "runs": {}, "pairs": {},
              "compare": {}}
    for r in names:
        report["runs"][r] = {k: float(v) for k, v in point[r].items()} | {"derailed": derail[r]} | {
            f"{k}_ci": ci(v) for k, v in boot[r].items()}
    pair_boot = {}
    for label, c, a in args.pair:
        p = {"control": c, "ablated": a, "cost": float(point[a]["real"] - point[c]["real"]),
             "cost_ci": ci(boot[a]["real"] - boot[c]["real"])}
        pb = {"cost": boot[a]["real"] - boot[c]["real"]}
        if "gap" in point[a] and "gap" in point[c]:
            p["grounding_kept"] = float(point[a]["gap"] / point[c]["gap"])
            pb["grounding_kept"] = boot[a]["gap"] / boot[c]["gap"]
            p["grounding_kept_ci"] = ci(pb["grounding_kept"])
        report["pairs"][label], pair_boot[label] = p, pb
    for a, b in args.compare:
        pa, pb = report["pairs"][a], report["pairs"][b]
        res = {"cost_diff": pb["cost"] - pa["cost"], "cost_diff_ci": ci(pair_boot[b]["cost"] - pair_boot[a]["cost"])}
        if "grounding_kept" in pa and "grounding_kept" in pb:
            res["grounding_kept_diff"] = pb["grounding_kept"] - pa["grounding_kept"]
            res["grounding_kept_diff_ci"] = ci(pair_boot[b]["grounding_kept"] - pair_boot[a]["grounding_kept"])
        report["compare"][f"{b} - {a}"] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = [f"# Ablation pairs ({args.split}, {len(n)} trials, greedy; paired trial bootstrap, {args.n_boot} resamples)",
             "", "| run | WER real | null shuffled | prompt only | grounding gap [95% CI] | derailed |",
             "|---|---|---|---|---|---|"]
    for r, s in report["runs"].items():
        gap = f"{pct(s['gap'])} [{pct(s['gap_ci'][0])}, {pct(s['gap_ci'][1])}]" if "gap" in s else ""
        lines.append(f"| {r} | {pct(s['real'])} | {pct(s.get('null_shuffled', float('nan')))} | "
                     f"{pct(s.get('null_prompt_only', float('nan')))} | {gap} | {pct(s['derailed'])}% |")
    lines += ["", "| pair | control | ablated | cost (pp) [95% CI] | grounding kept [95% CI] |", "|---|---|---|---|---|"]
    for label, p in report["pairs"].items():
        g = (f"{p['grounding_kept']:.2f} [{p['grounding_kept_ci'][0]:.2f}, {p['grounding_kept_ci'][1]:.2f}]"
             if "grounding_kept" in p else "")
        lines.append(f"| {label} | {p['control']} | {p['ablated']} | {pct(p['cost'])} [{pct(p['cost_ci'][0])}, "
                     f"{pct(p['cost_ci'][1])}] | {g} |")
    if report["compare"]:
        lines += ["", "| comparison | cost difference (pp) [95% CI] | grounding kept difference [95% CI] |",
                  "|---|---|---|"]
        for k, v in report["compare"].items():
            g = (f"{v['grounding_kept_diff']:+.2f} [{v['grounding_kept_diff_ci'][0]:+.2f}, "
                 f"{v['grounding_kept_diff_ci'][1]:+.2f}]" if "grounding_kept_diff" in v else "")
            lines.append(f"| {k} | {100 * v['cost_diff']:+.1f} [{100 * v['cost_diff_ci'][0]:+.1f}, "
                         f"{100 * v['cost_diff_ci'][1]:+.1f}] | {g} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
