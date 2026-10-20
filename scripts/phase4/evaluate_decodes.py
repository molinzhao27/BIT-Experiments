#!/usr/bin/env python
"""Phase 4 shared evaluator: score decode CSVs (trial, reference, decoded) from every arm identically.

    python scripts/phase4/evaluate_decodes.py score DECODES.csv [...] [--arm A --seed S --split SP
        --decoding D --model M --condition real|null_shuffled|null_noise|null_prompt_only] [--meta F]
    python scripts/phase4/evaluate_decodes.py summarize
    python scripts/phase4/evaluate_decodes.py examples A.csv B.csv [...] [--labels a b ...]
    python scripts/phase4/evaluate_decodes.py nulls --real R.csv [--null-shuffled F] [--null-noise F]
        [--null-prompt-only F]

Run metadata (model, arm, seed, split, decoding, condition; condition defaults to "real") comes from
columns in the CSV, then a sidecar <stem>.meta.yaml|.meta.yml|.meta.json (or --meta), then CLI flags.
Settings: configs/phase4/eval.yaml (BERTScore roberta-large, layer 17, no idf, rescaled; τ grid; bootstrap).

score      -> results/phase4/{metrics.csv, error_analysis.csv, per_trial/<run>.csv} (rows of the same
              run are replaced; metrics.csv keeps existing columns and grows new ones)
summarize  -> results/phase4/metrics_summary.{csv,md} (mean ± SD over seeds)
examples   -> results/phase4/examples_side_by_side.md (same 20 trials for all files)
nulls      -> results/phase4/null_controls.csv (paired bootstrap, real vs each null)
`examples` and `nulls` accept decode CSVs or the per_trial/ files written by `score` (no re-scoring).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import phase4_eval as ev  # noqa: E402

OUT = ROOT / "results" / "phase4"


def _overrides(args) -> dict:
    return {k: getattr(args, k) for k in ev.META_KEYS}


def cmd_score(args, cfg) -> int:
    for path in args.decodes:
        meta = ev.run_meta(path, args.meta, _overrides(args))
        row, ea = ev.evaluate_file(path, meta, cfg, args.out_dir)
        d = ea[ea.is_default_tau].iloc[0]
        print(f"{path}: {meta['arm']} seed {meta['seed']} {meta['split']} [{meta['condition']}] "
              f"n={row['n_trials']} WER={100 * row['wer']:.2f}% sentWER={100 * row['mean_sent_wer']:.2f}% "
              f"SER={100 * row['ser']:.1f}% BERTScore-F1={row['bertscore_f1']:.4f} "
              f"meaning-changing@{d.tau:g}={d.frac_meaning_changing:.3f}")
    return 0


def cmd_summarize(args, cfg) -> int:
    s = ev.summarize(pd.read_csv(args.metrics))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s.to_csv(args.out_dir / "metrics_summary.csv", index=False)
    md = ev.summary_markdown(s)
    (args.out_dir / "metrics_summary.md").write_text(md)
    print(md)
    return 0


def cmd_examples(args, cfg) -> int:
    if args.labels and len(args.labels) != len(args.decodes):
        raise SystemExit("--labels needs one label per file")
    tables = {}
    for i, path in enumerate(args.decodes):
        label = args.labels[i] if args.labels else None
        if label is None:
            try:
                label = ev.run_meta(path).get("config")
            except ValueError:
                label = Path(path).stem
        if label in tables:
            label = f"{label} ({Path(path).stem})"
        tables[label] = ev.with_word_errors(ev.read_decodes(path))
    n = args.n or cfg["examples"]["n"]
    seed = cfg["examples"]["seed"] if args.seed is None else args.seed
    trials = ev.pick_examples(tables, n, seed)
    out = args.out or args.out_dir / "examples_side_by_side.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ev.examples_markdown(tables, trials, args.title))
    print(f"{len(trials)} trials -> {out}")
    return 0


def cmd_nulls(args, cfg) -> int:
    meta = ev.run_meta(args.real, args.meta, _overrides(args) | {"condition": "real"})
    real = ev.load_or_score(args.real, cfg)
    nulls = {}
    for cond in ev.NULLS:
        if (path := getattr(args, cond)) is None:
            continue
        try:
            other = ev.run_meta(path)
        except ValueError:
            other = {}
        for k in ("config", "arm"):
            if other.get(k) not in (None, meta[k]):
                print(f"warning: {path} is {k} {other[k]!r}, real run is {meta[k]!r}", file=sys.stderr)
        nulls[cond] = ev.load_or_score(path, cfg)
    if not nulls:
        raise SystemExit("give at least one of --null-shuffled / --null-noise / --null-prompt-only")
    b = cfg["bootstrap"]
    res = ev.null_controls(real, nulls, int(b["n_resamples"]), int(b["seed"]))
    ident = {k: meta[k] for k in ("model", "config", "arm", "seed", "split", "decoding")} | {"evaluator": ev.EVALUATOR}
    res = pd.concat([pd.DataFrame([ident] * len(res)), res], axis=1).assign(
        bertscore_hash=ev.bertscore_settings(cfg)["bertscore_hash"])
    ev.upsert_csv(args.out_dir / "null_controls.csv", res,
                  ("config", "arm", "seed", "split", "decoding", "evaluator", "null_condition", "metric"))
    print(res[["null_condition", "metric", "real", "null", "diff", "ci_low", "ci_high", "p_one_sided",
               "real_better", "beats_all_nulls"]].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4" / "eval.yaml")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--device", help="cpu | cuda | mps | auto (overrides the config)")
    ap.add_argument("--batch-size", type=int, help="BERTScore batch size (overrides the config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def meta_flags(p, condition: bool = False):
        p.add_argument("--meta", type=Path, help="YAML/JSON metadata (default: <stem>.meta.yaml sidecar)")
        p.add_argument("--model")
        p.add_argument("--arm")
        p.add_argument("--seed", type=int)
        p.add_argument("--split")
        p.add_argument("--decoding")
        if condition:
            p.add_argument("--condition", choices=ev.CONDITIONS)

    p = sub.add_parser("score", help="score decode files; update metrics.csv and error_analysis.csv")
    p.add_argument("decodes", nargs="+", type=Path)
    p.add_argument("--tau", type=float, help="default τ for meaning-preserving (overrides the config)")
    meta_flags(p, condition=True)

    p = sub.add_parser("summarize", help="mean ± SD over seeds -> metrics_summary.{csv,md}")
    p.add_argument("--metrics", type=Path, default=OUT / "metrics.csv")

    p = sub.add_parser("examples", help="side-by-side decodes of several arms on one split")
    p.add_argument("decodes", nargs="+", type=Path)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--title", default="")
    p.add_argument("--out", type=Path)

    p = sub.add_parser("nulls", help="real run vs its null controls (paired bootstrap)")
    p.add_argument("--real", type=Path, required=True)
    for cond in ev.NULLS:
        p.add_argument(f"--{cond.replace('_', '-')}", dest=cond, type=Path)
    meta_flags(p)

    args = ap.parse_args()
    if not hasattr(args, "condition"):
        args.condition = None
    cfg = ev.load_config(args.config)
    if args.device:
        cfg["bertscore"]["device"] = args.device
    if args.batch_size:
        cfg["bertscore"]["batch_size"] = args.batch_size
    if getattr(args, "tau", None) is not None:
        cfg["error_analysis"]["tau"] = args.tau
    return {"score": cmd_score, "summarize": cmd_summarize, "examples": cmd_examples, "nulls": cmd_nulls}[
        args.cmd](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
