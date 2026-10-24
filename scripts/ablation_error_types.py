#!/usr/bin/env python
"""Wrong words when brain input is removed: still sound-alike, or language-driven guesses? (CPU; saved decodes.)

    .venv/bin/python scripts/ablation_error_types.py [--config configs/ablation_error_types.yaml] [--runs RUN ...]
        [--out DIR]

Question: when the eval-only ablations (scripts/eval_ablations.py: an array dropped or kept alone, part of the trial,
patches shuffled, time reversed) remove brain input, do the decoders' substitutions turn into the LLM prior's guesses,
words that fit the true sentence context or are related in meaning? And does that differ by array?

No new measure. For every run x condition, scripts/phonetic_error_analysis.py's functions, on that decode set's
in-dictionary substitution pairs:
  sound-alike  share with phoneme similarity >= 0.5: observed (with its CI) and the length-matched null
  phon         phoneme-similarity excess (observed - null)
  ctx          excess masked-LM log-probability (nats) of the substitute in the reference word's slot of the true
               sentence (one-word-piece words only)
  related      excess share related in meaning (sem >= 0.4; sem = cosine of centred distilbert word embeddings)
and WER over the set's trials. Each decode set's null is its own substitutes, length-matched (phonetic_error_analysis
step 3). One lexicon for every decode set here, so one embedding centring: sem / related values differ a little from
phonetic_errors.md, which centred on its own words. One cluster bootstrap over trials (seeded) for every run and
condition, so every contrast is paired: each condition - the baseline (`none`, the unablated input re-decoded, checked
against the run's decodes/<split>_greedy_real.csv), and the config's pairs (drop_55b - drop_v6v). Slice mostly_right
repeats it on the trials each condition decodes with sentence WER <= 0.5, where the decoded context is mostly right.

Outputs: <out>/summary.json, <out>/tables.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ablation_error_types.yaml"
ABLATIONS_CONFIG = ROOT / "configs" / "eval_ablations.yaml"
PHONETIC_SCRIPT = ROOT / "scripts" / "phonetic_error_analysis.py"
CONTRAST_METRICS = ("phon", "ctx", "related", "sem", "phon|orth")  # excess differences, a - b
REFERENCE_LABELS = {"null_shuffled": "another trial's input (decodes/)", "real": "saved decodes (decodes/)"}


def load_phonetic_module():
    """scripts/phonetic_error_analysis.py as a module. Its HF_HUB_OFFLINE default is undone after the import (main()
    sets it before distilbert loads), so importing this file changes nothing else in the process."""
    if "phonetic_error_analysis" in sys.modules:
        return sys.modules["phonetic_error_analysis"]
    before = os.environ.get("HF_HUB_OFFLINE")
    spec = importlib.util.spec_from_file_location("phonetic_error_analysis", PHONETIC_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        if before is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = before
    sys.modules["phonetic_error_analysis"] = mod
    return mod


pea = load_phonetic_module()


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    clash = set(cfg["conditions"]) & set(cfg.get("reference_decodes") or [])
    assert not clash, f"reference_decodes reuse ablation names: {clash}"
    assert cfg["baseline"] in cfg["conditions"], "the baseline must be one of the conditions"
    return cfg


def condition_labels(cfg: dict) -> dict:
    labels = dict(REFERENCE_LABELS)
    if ABLATIONS_CONFIG.is_file():
        labels |= {k: v.get("label", k) for k, v in yaml.safe_load(ABLATIONS_CONFIG.read_text())["ablations"].items()}
    return {c: labels.get(c, c) for c in [*cfg["conditions"], *(cfg.get("reference_decodes") or [])]}


# ---------------------------------------------------------------- decode sets

def read_decodes(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path, keep_default_na=False)
    df["seen_in_train"] = df["seen_in_train"].astype(str).str.lower().eq("true")
    return df


def baseline_check(base: pd.DataFrame | None, saved: pd.DataFrame | None) -> dict:
    """Does the re-decoded baseline reproduce the run's own saved greedy decodes?"""
    if base is None or saved is None:
        return {"matches_saved": None}
    m = base.merge(saved, on="trial", suffixes=("", "_saved"))
    assert (m["reference"] == m["reference_saved"]).all(), "baseline and saved decodes have different references"
    differ = int((m["decoded"] != m["decoded_saved"]).sum())
    return {"matches_saved": differ == 0 and len(m) == len(base) == len(saved),
            "trials_differing_from_saved": differ, "n_trials": len(base), "n_saved": len(saved)}


def collect(cfg: dict, runs: list[str], results: Path) -> tuple[list[tuple[str, str, pd.DataFrame]], dict]:
    """[(run, condition, decodes)] for every decode set found (config order), and each run's baseline check."""
    split, sets, checks = cfg["split"], [], {}
    for run in runs:
        found = []
        for cond in cfg["conditions"]:
            df = read_decodes(results / run / "ablations" / cond / f"{split}_greedy.csv")
            if df is None:
                print(f"skip {run} {cond}: no ablations/{cond}/{split}_greedy.csv")
                continue
            found.append((run, cond, df))
        if not found:
            continue
        for name in cfg.get("reference_decodes") or []:
            df = read_decodes(results / run / "decodes" / f"{split}_greedy_{name}.csv")
            if df is None:
                print(f"skip {run} {name}: no decodes/{split}_greedy_{name}.csv")
                continue
            found.append((run, name, df))
        base = next((df for _, c, df in found if c == cfg["baseline"]), None)
        checks[run] = baseline_check(base, read_decodes(results / run / "decodes" / f"{split}_greedy_real.csv"))
        sets += found
    return sets, checks


# ---------------------------------------------------------------- statistics (phonetic_error_analysis.Stats resamples)

def ci(dist: np.ndarray) -> list[float]:
    return [float(x) for x in np.nanpercentile(dist, [2.5, 97.5])]


def wer_boot(stats, df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Corpus WER of a decode set and its distribution over the shared trial resamples."""
    e, n = np.zeros(len(stats.pos)), np.zeros(len(stats.pos))
    for t, r, h in zip(df["trial"], df["reference"], df["decoded"], strict=True):
        errs, words = pea.word_errors(r, h)
        e[stats.pos[int(t)]] += errs
        n[stats.pos[int(t)]] += words
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(e.sum() / max(n.sum(), 1)), (stats.counts @ e) / (stats.counts @ n)


def valid_pairs(ra, col: str, slice_: str) -> list[dict]:
    return [p for p in ra.pairs if ra.in_slice(p, slice_) and not np.isnan(p[col])]


def describe(ra, stats, slices: list[str]) -> tuple[dict, tuple[float, np.ndarray]]:
    """phonetic_error_analysis.summarize for one decode set, plus WER and the observed sound-alike share with CIs."""
    s = pea.summarize(ra, stats, slices, perm=False)
    w = wer, dist = wer_boot(stats, ra.df)
    assert abs(wer - s["wer"]) < 1e-9, f"{ra.name} {ra.condition}: WER {wer} != {s['wer']}"
    s["wer_ci"] = ci(dist)
    for sl in slices:
        pairs = valid_pairs(ra, "obs_alike", sl)
        if pairs and "alike" in s["slices"][sl]:
            s["slices"][sl]["alike"]["obs_ci"] = ci(stats.boot(pairs, "obs_alike"))
    return s, w


def contrast(stats, a, b, wer_a: tuple[float, np.ndarray], wer_b: tuple[float, np.ndarray], slice_: str) -> dict:
    """a - b on the same resampled trials: WER (slice all; (point, distribution) from wer_boot), the observed
    sound-alike share, and each excess (phonetic_error_analysis.contrast)."""
    row = {"run": a.name, "a": a.condition, "b": b.condition, "slice": slice_}
    if slice_ == "all":
        row["wer"] = {"diff": wer_a[0] - wer_b[0], "ci": ci(wer_a[1] - wer_b[1])}
    pa, pb = valid_pairs(a, "obs_alike", slice_), valid_pairs(b, "obs_alike", slice_)
    if pa and pb:
        row["alike_obs"] = {"diff": float(np.mean([p["obs_alike"] for p in pa]) - np.mean([p["obs_alike"] for p in pb])),
                            "ci": ci(stats.boot(pa, "obs_alike") - stats.boot(pb, "obs_alike"))}
    for m in CONTRAST_METRICS:
        if valid_pairs(a, f"exc_{m}", slice_) and valid_pairs(b, f"exc_{m}", slice_):
            c = pea.contrast(stats, a, b, m, slice_)
            row[m] = {"diff": c["diff"], "ci": c["ci"]}
    return row


# ---------------------------------------------------------------- the analysis

def analyse(cfg: dict, runs: list[str], results: Path) -> dict:
    sets, checks = collect(cfg, runs, results)
    if not sets:
        raise SystemExit("no ablation decodes found")
    refs: dict[int, str] = {}
    for run, cond, df in sets:  # every decode set decodes the same trials
        for t, r in zip(df["trial"], df["reference"], strict=True):
            assert refs.setdefault(int(t), r) == r, f"{run} {cond}: trial {t} has a different reference"
    loaded = [(run, cond, df, *pea.extract(df)) for run, cond, df in sets]
    words = {p[k] for *_, pairs, _, _ in loaded for p in pairs for k in ("ref", "hyp")}
    words |= {w for *_, rw in loaded for ws in rw.values() for w in ws}
    print(f"{len(loaded)} decode sets, {len(words)} distinct words; building the lexicon ...", flush=True)
    lex = pea.Lexicon(words, cfg.get("embed_model") or None)
    costs = pea.substitution_costs()
    analyses = [pea.RunAnalysis(run, cond, df, pairs, ops, rw, lex, costs, cfg["min_null"])
                for run, cond, df, pairs, ops, rw in loaded]
    ctx_lp = None
    if lex.emb is not None:
        slots = {(p["trial"], p["pos"]) for ra in analyses for p in ra.pairs}
        ref_words = {t: pea.normalize(r).split() for t, r in refs.items()}
        print(f"masked-LM context scores for {len(slots)} reference slots ...", flush=True)
        ctx_lp = lex.context_logprobs({(t, i): ref_words[t] for t, i in slots})

    stats = pea.Stats(sorted(refs), cfg["n_boot"], 1, cfg["seed"])
    slices = cfg.get("slices") or ["all"]
    summaries, dists = [], {}
    for ra in analyses:
        print(f"{ra.name} {ra.condition}: {len(ra.pairs)} pairs ...", flush=True)
        ra.build(ctx_lp)
        s, dists[(ra.name, ra.condition)] = describe(ra, stats, slices)
        summaries.append(s)

    by = {(ra.name, ra.condition): ra for ra in analyses}
    pairs = [(c, cfg["baseline"]) for c in [*cfg["conditions"], *(cfg.get("reference_decodes") or [])]
             if c != cfg["baseline"]] + [tuple(p) for p in cfg.get("contrasts") or []]
    contrasts = [contrast(stats, by[(run, a)], by[(run, b)], dists[(run, a)], dists[(run, b)], sl)
                 for run in dict.fromkeys(ra.name for ra in analyses) for a, b in pairs
                 if (run, a) in by and (run, b) in by for sl in slices]
    meta = {k: cfg[k] for k in ("split", "seed", "n_boot", "min_null", "embed_model", "baseline", "slices")}
    meta |= {"n_lexicon": len(lex.words), "n_trials": len(refs), "baseline_checks": checks,
             "cmudict": _version("cmudict"), "pronouncing": _version("pronouncing"), "wordfreq": _version("wordfreq")}
    return {"meta": meta, "runs": summaries, "contrasts": contrasts}


# ---------------------------------------------------------------- tables

def fmt(x: float | None, lo_hi: list[float] | None = None, scale: float = 1.0, digits: int = 3) -> str:
    if x is None or np.isnan(x):
        return "–"
    s = f"{scale * x:+.{digits}f}"
    return s if lo_hi is None else s + f" [{scale * lo_hi[0]:+.{digits}f}, {scale * lo_hi[1]:+.{digits}f}]"


def fmt_m(m: dict | None, scale: float = 1.0, digits: int = 3) -> str:
    return fmt(m["excess"], m["ci"], scale, digits) if m else "–"


def fmt_d(m: dict | None, scale: float = 1.0, digits: int = 3) -> str:
    return fmt(m["diff"], m["ci"], scale, digits) if m else "–"


def tables(result: dict, labels: dict) -> str:
    runs = list(dict.fromkeys(s["run"] for s in result["runs"]))
    checks = result["meta"]["baseline_checks"]
    lines = [f"# Error types under eval-only input ablations ({result['meta']['split']}; "
             "scripts/ablation_error_types.py)", "",
             "Excess = observed − length-matched null of that decode set's own substitutes (scripts/"
             "phonetic_error_analysis.py); 95% CIs from one cluster bootstrap over trials shared by every run and "
             f"condition ({result['meta']['n_boot']} resamples, seed {result['meta']['seed']}). Sound-alike = share "
             "of pairs with phoneme similarity ≥ 0.5; ctx = fit of the substitute in the true sentence (masked-LM "
             "log-prob, nats); related = share with sem ≥ 0.4, in percentage points. One seed per run.", ""]
    for run in runs:
        rows = [s for s in result["runs"] if s["run"] == run]
        chk = checks.get(run, {})
        same = {True: "= saved decodes", False: f"≠ saved decodes ({chk.get('trials_differing_from_saved')} trials)",
                None: "no saved decodes to check"}[chk.get("matches_saved")]
        lines += [f"## {pea.RUNS.get(run, run)} — `{run}` (baseline `{result['meta']['baseline']}` {same})", "",
                  "| Condition | WER [95% CI] | pairs | sound-alike obs [CI] / null | phon excess | ctx excess (nats) "
                  "| related obs / null | related excess (pp) |", "|---|---|---|---|---|---|---|---|"]
        for s in rows:
            a = s["slices"]["all"]
            al, rel = a.get("alike"), a.get("related")
            alike = (f"{100 * al['obs']:.1f}% [{100 * al['obs_ci'][0]:.1f}, {100 * al['obs_ci'][1]:.1f}] / "
                     f"{100 * al['null']:.1f}%" if al else "–")
            related = f"{100 * rel['obs']:.1f}% / {100 * rel['null']:.1f}%" if rel else "–"
            lines.append(f"| {s['condition']}: {labels.get(s['condition'], s['condition'])} "
                         f"| {100 * s['wer']:.1f}% [{100 * s['wer_ci'][0]:.1f}, {100 * s['wer_ci'][1]:.1f}] "
                         f"| {s['n_pairs']} | {alike} | {fmt_m(a.get('phon'))} | {fmt_m(a.get('ctx'), digits=2)} "
                         f"| {related} | {fmt_m(rel, 100, 1)} |")
        lines.append("")
    for sl in result["meta"]["slices"]:
        cs = [c for c in result["contrasts"] if c["slice"] == sl]
        if not cs:
            continue
        lines += [f"## Paired contrasts, a − b (slice {sl})", "",
                  "| Run | a − b | ΔWER (pp) | Δ sound-alike (pp) | Δ phon excess | Δ ctx excess (nats) "
                  "| Δ related excess (pp) | Δ sem excess |", "|---|---|---|---|---|---|---|---|"]
        for c in cs:
            lines.append(f"| {pea.RUNS.get(c['run'], c['run'])} | {c['a']} − {c['b']} | {fmt_d(c.get('wer'), 100, 1)} "
                         f"| {fmt_d(c.get('alike_obs'), 100, 1)} | {fmt_d(c.get('phon'))} "
                         f"| {fmt_d(c.get('ctx'), digits=2)} | {fmt_d(c.get('related'), 100, 1)} "
                         f"| {fmt_d(c.get('sem'))} |")
        lines.append("")
    lines += ["## Further measures (slice all) and the mostly-right slice (sentence WER ≤ 0.5)", "",
              "| Run | Condition | sem excess | fits context better than ref obs / null | ctx coverage "
              "| phon excess beyond spelling | mostly-right: pairs | phon excess | ctx excess | related excess (pp) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in result["runs"]:
        a, r = s["slices"]["all"], s["slices"].get("mostly_right", {})
        gt, ctx = a.get("ctx>r"), a.get("ctx")
        better = f"{100 * gt['obs']:.0f}% / {100 * gt['null']:.0f}%" if gt else "–"
        coverage = f"{100 * ctx['n'] / max(s['n_pairs'], 1):.0f}%" if ctx else "–"
        lines.append(f"| {pea.RUNS.get(s['run'], s['run'])} | {s['condition']} | {fmt_m(a.get('sem'))} "
                     f"| {better} | {coverage} "
                     f"| {fmt_m(a.get('phon|orth'))} | {r.get('n_pairs', '–')} | {fmt_m(r.get('phon'))} "
                     f"| {fmt_m(r.get('ctx'), digits=2)} | {fmt_m(r.get('related'), 100, 1)} |")
    return "\n".join(lines) + "\n"


def _version(pkg: str) -> str:
    from importlib.metadata import version

    return version(pkg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--runs", nargs="+", help="runs under results/stage2 (default: the config's)")
    ap.add_argument("--results", type=Path, default=pea.RESULTS, help="the stage2 results directory")
    ap.add_argument("--out", type=Path, help="default: the config's out")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # distilbert from the local cache only
    result = analyse(cfg, args.runs or cfg["runs"], args.results)
    out = args.out or ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    result["meta"]["config"] = cfg
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    (out / "tables.md").write_text(tables(result, condition_labels(cfg)))
    print(f"wrote {out}/summary.json, tables.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
