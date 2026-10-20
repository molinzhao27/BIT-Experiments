#!/usr/bin/env python
"""Phase 1: reproduce Kunz et al. 2025 Fig 1E — 7-word decoding per participant x array x behavior.

    python scripts/phase1_isolated_words.py [--config configs/phase1.yaml] [--variants all|none|NAME ...]

Features (TX + SBP per array) -> block-mean subtraction -> Gaussian smoothing -> z-score;
500 ms window mean whose start is chosen by nested 10-fold CV (inner folds pick the
start, outer folds evaluate); pooled-variance Gaussian NB (diagonal LDA); Clopper-Pearson
95% CIs; significant if the lower bound > 1/7.

Outputs: results/phase1/{accuracy_table.csv, accuracy_table.md, predictions_*.csv.gz,
confusion_*.npz, inner_curves_*.npz, summary.json}, figures/phase1_fig1e_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder.decoding import clopper_pearson, confusion, nested_cv  # noqa: E402
from kunzdecoder.features import load_features, window_means  # noqa: E402
from kunzdecoder.plotting import accuracy_heatmap  # noqa: E402
from kunzdecoder.windows import isolated_starts_ms, session_file  # noqa: E402


def run_variant(cfg: dict, sigma_ms: float, kind: str, tag: str, out_dir: Path) -> pd.DataFrame:
    task_dir = ROOT / "data" / "raw" / cfg["task"]
    cache = ROOT / "data" / "processed"
    by_participant: dict[str, list] = {}
    for p, arr, label in cfg["arrays"]:
        by_participant.setdefault(p, []).append((arr, label))
    rows, preds, confs, curves = [], [], {}, {}
    for p, arrays in by_participant.items():
        for beh in cfg["behaviors"]:
            t0 = time.time()
            feats = load_features(session_file(task_dir, p, beh), cache, sigma_ms,
                                  cfg["preprocess"]["smoothing_truncate"])
            s = feats.session
            words = s.cues >= 2  # cue 1 = DO_NOTHING
            y, trial_idx = s.cues[words], np.flatnonzero(words)
            starts_ms = isolated_starts_ms(cfg, p, beh)
            starts = starts_ms // s.bin_ms
            length = cfg["window"]["length_ms"] // s.bin_ms
            for arr, label in arrays:
                X = window_means(feats.x[:, feats.array_columns(arr)], s.go[words, 0], s.block, starts, length)
                res = nested_cv(X, y, cfg["cv"]["outer_folds"], cfg["cv"]["inner_folds"], cfg["seed"], kind)
                k, n = res.n_correct, len(y)
                lo, hi = clopper_pearson(k, n, cfg["ci_level"])
                fold_starts = starts_ms[[res.window_idx[res.fold == f][0] for f in range(cfg["cv"]["outer_folds"])]]
                vals, cnt = np.unique(fold_starts, return_counts=True)
                rows.append({
                    "participant": p, "array": arr, "label": label, "behavior": beh,
                    "behavior_label": cfg["behaviors"][beh], "session": f"{s.participant} {s.date}",
                    "n_features": X.shape[2], "n_trials": n, "n_correct": k, "accuracy": k / n,
                    "ci_low": lo, "ci_high": hi, "significant": lo > cfg["chance"],
                    "window_start_ms_mode": int(vals[cnt.argmax()]),
                    "window_starts_ms_per_fold": " ".join(map(str, fold_starts)),
                    "n_candidate_windows": len(starts), "bin_ms": s.bin_ms,
                })
                preds.append(pd.DataFrame({
                    "participant": p, "array": arr, "behavior": beh, "trial": trial_idx, "fold": res.fold,
                    "y_true": res.y_true, "y_pred": res.y_pred, "window_start_ms": starts_ms[res.window_idx],
                }))
                key = f"{p}|{arr}|{beh}"
                confs[key] = confusion(res.y_true, res.y_pred, np.unique(y))
                curves[key] = np.vstack([starts_ms, res.inner_acc.mean(0)])
            print(f"[{tag}] {p} {beh:20s} {time.time() - t0:5.1f}s  " + "  ".join(
                f"{r['array']}={100 * r['accuracy']:.1f}{'' if r['significant'] else '(ns)'}"
                for r in rows[-len(arrays):]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"accuracy_{tag}.csv", index=False)
    pd.concat(preds).to_csv(out_dir / f"predictions_{tag}.csv.gz", index=False)
    np.savez_compressed(out_dir / f"confusion_{tag}.npz", **confs)
    np.savez_compressed(out_dir / f"inner_curves_{tag}.npz", **curves)
    return df


def pivot(df: pd.DataFrame, cfg: dict, value: str = "accuracy") -> pd.DataFrame:
    labels = [a[2] for a in cfg["arrays"]]
    return df.pivot(index="label", columns="behavior", values=value).loc[labels, list(cfg["behaviors"])]


def write_markdown_table(df: pd.DataFrame, cfg: dict, path: Path) -> None:
    acc, lo, hi, sig = (pivot(df, cfg, v) for v in ("accuracy", "ci_low", "ci_high", "significant"))
    head = "| Array | " + " | ".join(cfg["behaviors"].values()) + " |"
    lines = [head, "|" + "---|" * (len(cfg["behaviors"]) + 1)]
    for i, label in enumerate(acc.index):
        if i == cfg["n_paper_rows"]:
            lines.append("| *not in paper Fig 1E* |" + " |" * len(cfg["behaviors"]))
        cells = [f"{100 * acc.iloc[i, j]:.1f} [{100 * lo.iloc[i, j]:.0f}–{100 * hi.iloc[i, j]:.0f}]"
                 + ("" if sig.iloc[i, j] else " n.s.") for j in range(acc.shape[1])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    path.write_text("Accuracy % [Clopper-Pearson 95% CI]; n.s. = lower bound ≤ chance (14.3%).\n\n"
                    + "\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1.yaml")
    ap.add_argument("--variants", nargs="+", default=["all"], help="'all', 'none', or variant names")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = ROOT / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    sigma = cfg["preprocess"]["smoothing_sigma_ms"]
    runs = {"primary": {}}
    if args.variants != ["none"]:
        runs |= {k: v for k, v in cfg.get("variants", {}).items() if "all" in args.variants or k in args.variants}
    results = {}
    for tag, ov in runs.items():
        results[tag] = run_variant(cfg, ov.get("smoothing_sigma_ms", sigma), ov.get("classifier", cfg["classifier"]),
                                   tag, out_dir)

    df = results["primary"]
    df.to_csv(out_dir / "accuracy_table.csv", index=False)
    write_markdown_table(df, cfg, out_dir / "accuracy_table.md")
    accuracy_heatmap(
        100 * pivot(df, cfg).to_numpy(), pivot(df, cfg, "significant").to_numpy(),
        [a[2] for a in cfg["arrays"]], list(cfg["behaviors"].values()),
        ROOT / "figures" / "phase1_fig1e_heatmap.png",
        title=f"Isolated-word decoding (7 words), nested 10-fold CV, 500 ms window, σ = {sigma:g} ms",
        gap_after=cfg["n_paper_rows"], gap_label="not in paper Fig 1E ↓",
        note="Pooled-variance Gaussian NB (diagonal LDA). Hatched + n.s. = Clopper-Pearson 95% lower bound ≤ chance.",
    )

    checks = []
    for p, arr, beh, paper in cfg["paper_targets"]:
        row = {"participant": p, "array": arr, "behavior": beh, "paper": paper}
        for tag, d in results.items():
            r = d[(d.participant == p) & (d.array == arr) & (d.behavior == beh)].iloc[0]
            row[tag] = round(100 * r.accuracy, 1)
            if tag == "primary":
                row["ci"] = [round(100 * r.ci_low, 1), round(100 * r.ci_high, 1)]
                row["paper_within_ci"] = bool(r.ci_low <= paper / 100 <= r.ci_high)
        checks.append(row)
    comparison = pd.concat([d.set_index(["label", "behavior"])["accuracy"].rename(t) for t, d in results.items()],
                           axis=1)
    comparison.to_csv(out_dir / "variant_comparison.csv")
    import scipy
    import sklearn
    summary = {
        "sanity_checks": checks,
        "n_significant": {t: int(d.significant.sum()) for t, d in results.items()},
        "n_conditions": len(df),
        "variant_mean_accuracy": {t: round(float(d.accuracy.mean()), 4) for t, d in results.items()},
        "config": cfg,
        "runtime_s": round(time.time() - t_start, 1),
        "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "pandas": pd.__version__, "machine": platform.node()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(checks, indent=1))
    print(f"done in {summary['runtime_s']} s -> {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
