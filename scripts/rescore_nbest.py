#!/usr/bin/env python
"""Rescore and select N-best candidates: the rescoring SIDE TEST (docs/RESCORING.md). Never a headline number:
docs/DESIGN.md keeps LM rescoring out of every headline result.

    <bit env>/python scripts/rescore_nbest.py opt --run B_ssl_s42 [RUN ...] [--device mps] [--skip-mac-checks]
    python scripts/rescore_nbest.py report --run B_ssl_s42 [--no-bertscore]
    python scripts/rescore_nbest.py table                       # the cross-run table only

opt     OPT-6.7B (configs/rescoring.yaml `opt`, pinned revision, fp16) log-likelihood of every unique candidate text
        of the runs' candidate files (results/rescoring/<run>/candidates_<split>.jsonl), cached per text in
        results/rescoring/opt_cache_opt-6.7b.jsonl (shared by all runs; resume-safe, written per batch chunk). As Card et
        al. 2024: </s> (OPT's BOS) first, every later token scored, no end token (opt_eos is cached too). On a Mac it
        first checks that free memory is >= 35% (memory_pressure) and that no other process used the GPU in a 10-s
        window (ioreg accumulatedGPUTime per client), and refuses otherwise. Needs transformers (BIT's env).
report  per split, the unique candidates of every trial with their features (rescoring.trial_table); rule weights
        tuned on BIT-val only, applied unchanged to BIT-test. Rules (configs/rescoring.yaml selection.rules): greedy;
        opt = OPT log-likelihood only; dec_opt = decoder log-prob + alpha * OPT + beta * words. Metrics on BIT-test:
        corpus WER (BIT's normalization) with a paired trial bootstrap vs greedy (10,000 resamples, seed 0), oracle WER
        (best candidate per trial), seen / unseen sentences, and BERTScore F1 of the chosen sentences
        (scripts/evaluate_decodes.py, configs/eval.yaml, run in the project .venv). Outputs
        results/rescoring/<run>/{summary.json, summary.md, selected/<split>_<rule>.csv, bertscore/} and the cross-run
        results/rescoring/summary.md. numpy / pandas / editdistance / yaml only (any env).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments import rescoring as rs  # noqa: E402

OUT = ROOT / "results" / "rescoring"
CONFIG = ROOT / "configs" / "rescoring.yaml"
VENV = ROOT / ".venv" / "bin" / "python"
SPLITS = ("val", "test")


def say(msg: str) -> None:
    print(f"[rescore {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def opt_cache_path(cfg: dict) -> Path:
    return OUT / f"opt_cache_{cfg['opt']['repo_id'].split('/')[-1]}.jsonl"


# ---------------------------------------------------------------- Mac guards

def mac_free_percent() -> int | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def gpu_time_by_process() -> dict[str, int]:
    """Accumulated GPU time (ns) per client process, from ioreg's IOAccelerator user clients."""
    out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-l", "-w", "0"], capture_output=True, text=True).stdout
    usage: dict[str, int] = {}
    who = None
    for line in out.splitlines():
        m = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', line)
        if m:
            who = f"{m.group(1)} {m.group(2)}"
            continue
        if who and '"AppUsage"' in line:
            usage[who] = usage.get(who, 0) + sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', line))
    return usage


# Interactive apps that always touch the GPU a little (UI compositing, video decode); they never block scoring.
# ioreg truncates process names to 16 characters, so these are prefixes.
INTERACTIVE = ("WindowServer", "Claude Helper", "Google Chrome", "VTDecoderXPC", "NotificationCen", "Spotify",
               "Safari", "loginwindow", "ControlCenter", "Dock")


def mac_gpu_users(window_s: float = 10.0, min_share: float = 0.15) -> dict[str, float]:
    """Processes (other than interactive apps and this one) that used >= min_share of the GPU in the window."""
    a = gpu_time_by_process()
    time.sleep(window_s)
    b = gpu_time_by_process()
    me = str(os.getpid())
    busy = {}
    for k, v in b.items():
        share = (v - a.get(k, 0)) / (window_s * 1e9)
        if share >= min_share and not any(app in k for app in INTERACTIVE) and k.split()[0] != me:
            busy[k] = round(share, 3)
    return busy


# ---------------------------------------------------------------- opt

def candidate_files(run: str) -> list[Path]:
    return [p for s in SPLITS if (p := OUT / run / f"candidates_{s}.jsonl").exists()]


def cmd_opt(args, cfg) -> int:
    ocfg = cfg["opt"]
    cache = rs.ScoreCache(opt_cache_path(cfg))
    texts: set[str] = set()
    for run in args.run:
        files = candidate_files(run)
        if not files:
            raise SystemExit(f"{run}: no candidate files in {OUT / run}")
        for f in files:
            texts |= {rs.opt_text(r["candidate"], ocfg) for r in rs.CandidateFile(f).rows}
    todo = cache.missing(texts)
    say(f"{len(texts)} unique candidate texts, {len(texts) - len(todo)} cached, {len(todo)} to score")
    if not todo:
        return 0
    if sys.platform == "darwin" and not args.skip_mac_checks:
        free = mac_free_percent()
        users = mac_gpu_users()
        say(f"Mac checks: free memory {free}%, other GPU users {users or 'none'}")
        if free is None or free < 35 or users:
            raise SystemExit("the Mac is busy (free memory < 35% or another process on the GPU): OPT scoring refused")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[ocfg["dtype"]]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ocfg["repo_id"], revision=ocfg["revision"])
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(ocfg["repo_id"], revision=ocfg["revision"], torch_dtype=dtype,
                                                 low_cpu_mem_usage=True, device_map={"": str(device)})
    model.eval()
    say(f"{ocfg['repo_id']} loaded on {device} ({ocfg['dtype']}) in {time.time() - t0:.0f} s")
    todo.sort(key=len)  # similar lengths per batch: less padding
    t1, chunk = time.time(), 512
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        scores = rs.opt_logprobs(part, model, tok, device, int(ocfg["batch_size"]))
        if not all(np.isfinite([s["opt_sum"] for s in scores])):
            raise SystemExit("non-finite OPT log-probs (try dtype float32 or bfloat16)")
        cache.add(dict(zip(part, scores)))
        el = time.time() - t1
        done = start + len(part)
        say(f"OPT: {done}/{len(todo)} texts, {el / done * 1000:.1f} ms/text, ETA {el / done * (len(todo) - done) / 60:.1f} min")
    meta = {"model": ocfg["repo_id"], "revision": ocfg["revision"], "dtype": ocfg["dtype"], "device": str(device),
            "texts_scored": len(todo), "seconds": round(time.time() - t1, 1),
            "ms_per_text": round((time.time() - t1) / len(todo) * 1000, 2), "torch": torch.__version__}
    for run in args.run:
        (OUT / run / "opt_meta.json").write_text(json.dumps(meta, indent=1))
    return 0


# ---------------------------------------------------------------- report

def tables(run: str, cfg: dict) -> tuple[dict[str, pd.DataFrame], int]:
    """Per split, the trial tables; without OPT columns while any candidate text still lacks an OPT score (then
    the rules that need OPT are skipped). Returns (tables, number of texts without an OPT score)."""
    ocfg = cfg["opt"]
    cache = rs.ScoreCache(opt_cache_path(cfg))
    cands = {}
    for s in SPLITS:
        path = OUT / run / f"candidates_{s}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing")
        cands[s] = rs.read_candidates(path)
    missing = len(cache.missing({rs.opt_text(c, ocfg) for s in SPLITS for c in cands[s].candidate}))
    opt = cache.data if missing == 0 else None
    return {s: rs.trial_table(cands[s], opt, ocfg) for s in SPLITS}, missing


def write_selected(path: Path, t: pd.DataFrame, rows: np.ndarray, meta: dict) -> None:
    sel = t.iloc[rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trial": sel.trial.to_numpy(), "reference": sel.reference.to_numpy(), "decoded": sel.candidate.to_numpy(),
                  "seen_in_train": sel.seen_in_train.to_numpy() if "seen_in_train" in sel else None}).to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))


def run_bertscore(run: str, files: list[Path], device: str) -> dict[str, pd.DataFrame]:
    """BERTScore of each selected-sentence CSV with scripts/evaluate_decodes.py (project .venv, offline)."""
    out_dir = OUT / run / "bertscore"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [str(VENV), str(ROOT / "scripts" / "evaluate_decodes.py"), "--out-dir", str(out_dir), "--device", device,
           "score", *map(str, files)]
    say(f"BERTScore of {len(files)} files ...")
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    from bit_experiments import evaluation as ev  # its metadata and file naming, as evaluate_decodes.py used them

    return {f.stem: pd.read_csv(out_dir / "per_trial" / f"{ev.slug(ev.run_meta(f))}.csv", keep_default_na=False)
            for f in files}


def split_metrics(t: pd.DataFrame, rows: np.ndarray) -> dict:
    sel = t.iloc[rows]
    e, w = sel.errors.to_numpy(float), sel.ref_words.to_numpy(float)
    res = {"wer": float(e.sum() / max(w.sum(), 1)), "n": len(sel), "changed_vs_greedy": float((~sel.is_greedy).mean()),
           "exact": float((e == 0).mean())}
    if "seen_in_train" in sel and sel.seen_in_train.notna().all():
        seen = sel.seen_in_train.astype(bool).to_numpy()
        for name, m in (("seen", seen), ("unseen", ~seen)):
            res[f"wer_{name}"] = float(e[m].sum() / max(w[m].sum(), 1)) if m.any() else float("nan")
            res[f"n_{name}"] = int(m.sum())
    if "opt_mean" in sel:
        res["opt_mean_logprob"] = float(sel.opt_mean.mean())
    return res


def oracle(t: pd.DataFrame) -> dict:
    g = t.groupby("trial")
    e, w = g.errors.min().to_numpy(float), g.ref_words.first().to_numpy(float)
    res = {"wer": float(e.sum() / max(w.sum(), 1))}
    if "seen_in_train" in t and t.seen_in_train.notna().all():
        seen = g.seen_in_train.first().astype(bool).to_numpy()
        for name, m in (("seen", seen), ("unseen", ~seen)):
            res[f"wer_{name}"] = float(e[m].sum() / max(w[m].sum(), 1)) if m.any() else float("nan")
    return res


def cmd_report(args, cfg) -> int:
    for run in args.run:
        report(run, cfg, not args.no_bertscore)
    write_table()
    return 0


def report(run: str, cfg: dict, bertscore: bool) -> dict:
    scfg, mcfg = cfg["selection"], cfg["metrics"]
    t, opt_missing = tables(run, cfg)
    if opt_missing:
        say(f"{run}: {opt_missing} candidate texts have no OPT score yet (run `opt`): OPT rules skipped")
    sel = {s: rs.Selector(t[s]) for s in SPLITS}
    meta_c = json.loads((OUT / run / "candidates_test.meta.json").read_text())
    arm = run.split("_")[0]
    summary = {"run": run, "side_test": "N-best rescoring; never a headline number (docs/RESCORING.md)",
               "decoder": meta_c.get("model_class"), "llm": meta_c.get("llm"),
               "candidates": {s: {"trials": int(t[s].trial.nunique()), "samples_per_trial": float(t[s].n_samples.mean()),
                                  "unique_per_trial": float(t[s].groupby("trial").size().mean())} for s in SPLITS},
               "opt": {k: cfg["opt"][k] for k in ("repo_id", "revision", "dtype", "text", "eos")},
               "opt_texts_missing": opt_missing, "rules": {}, "oracle": {s: oracle(t[s]) for s in SPLITS}}
    test = sel["test"]
    idx = rs.bootstrap_indices(test.trials.size, int(mcfg["bootstrap"]["n_resamples"]), int(mcfg["bootstrap"]["seed"]))
    greedy_rows = test.select(rs.rule_weights("greedy"))
    base_err = test.errors[greedy_rows]
    files = []
    for rule in scfg["rules"]:
        needs = rs.rule_weights(rule, {"alpha": 1, "beta": 1, "gamma": 1})  # every feature the rule can use
        if not (sel["val"].available(needs) and test.available(needs)):
            say(f"{rule}: features missing ({', '.join(k for k in needs if k not in test.t)}), skipped")
            continue
        params, val_wer = rs.tune(rule, sel["val"], scfg)  # BIT-val only
        weights = rs.rule_weights(rule, params)
        rows = test.select(weights)
        r = {"params": params, "val_wer_tuned": val_wer, "test": split_metrics(t["test"], rows),
             "vs_greedy": rs.paired_bootstrap(test.errors[rows], base_err, test.words, idx)}
        summary["rules"][rule] = r
        f = OUT / run / "selected" / f"test_{rule}.csv"
        write_selected(f, t["test"], rows, {"model": f"rescoring {run} {rule}", "arm": arm,
                                            "seed": meta_c.get("seed"), "split": "brandman_2024_text:test",
                                            "decoding": f"nbest-{rule}", "condition": "real",
                                            "side_test": "N-best rescoring (docs/RESCORING.md)"})
        files.append(f)
        say(f"{run} {rule}: params {params}, val {100 * val_wer:.2f}%, test {100 * r['test']['wer']:.2f}% "
            f"(vs greedy {100 * r['vs_greedy']['delta']:+.2f} pp [{100 * r['vs_greedy']['ci_low']:+.2f}, "
            f"{100 * r['vs_greedy']['ci_high']:+.2f}])")
    if bertscore and mcfg.get("bertscore", True):
        bs = run_bertscore(run, files, mcfg.get("bertscore_device", "cpu"))
        g = bs.get("test_greedy")
        for rule, r in summary["rules"].items():
            f1 = bs[f"test_{rule}"].bs_f1.to_numpy(float)
            r["test"]["bertscore_f1"] = float(f1.mean())
            if g is not None:
                d = f1[idx].mean(1) - g.bs_f1.to_numpy(float)[idx].mean(1)
                r["vs_greedy"]["bertscore_delta"] = float(f1.mean() - g.bs_f1.mean())
                r["vs_greedy"]["bertscore_ci"] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
    for name in ("opt_meta.json",):
        p = OUT / run / name
        if p.exists():
            summary["opt_timing"] = json.loads(p.read_text())
    summary["generation_timing"] = {s: json.loads((OUT / run / f"candidates_{s}.meta.json").read_text())["timing"]
                                    for s in SPLITS if (OUT / run / f"candidates_{s}.meta.json").exists()}
    (OUT / run / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / run / "summary.md").write_text(summary_md(summary))
    say(f"-> {OUT / run / 'summary.md'}")
    return summary


def pct(x, nd=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}"


def summary_md(s: dict) -> str:
    c = s["candidates"]
    lines = [f"# N-best rescoring side test: {s['run']}", "",
             "**Side test, never a headline number** (docs/DESIGN.md: no LM rescoring in headline results; "
             "docs/RESCORING.md). Weights are tuned on BIT-val only and applied unchanged to BIT-test.", "",
             f"Decoder {s['decoder']} ({s['llm']}). Candidates per trial: {c['test']['samples_per_trial']:.0f} "
             f"(greedy + nucleus samples), {c['test']['unique_per_trial']:.2f} unique on BIT-test "
             f"({c['val']['unique_per_trial']:.2f} on BIT-val). Rescorer: {s['opt']['repo_id']} ({s['opt']['dtype']})."
             + (f" **OPT scores pending** ({s['opt_texts_missing']} texts unscored): the OPT rules are not in this table yet."
                if s.get("opt_texts_missing") else ""), "",
             "| rule | params (tuned on val) | val WER % | test WER % | Δ vs greedy, pp [95% CI] | p(not better) | "
             "changed vs greedy % | seen % | unseen % | BERTScore F1 | ΔBERTScore [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rule, r in s["rules"].items():
        v, te = r["vs_greedy"], r["test"]
        bci = v.get("bertscore_ci")
        lines.append(
            f"| {rule} | {json.dumps(r['params']) if r['params'] else '-'} | {pct(r['val_wer_tuned'])} | {pct(te['wer'])} | "
            f"{100 * v['delta']:+.2f} [{100 * v['ci_low']:+.2f}, {100 * v['ci_high']:+.2f}] | {v['p_not_better']:.4f} | "
            f"{pct(te['changed_vs_greedy'], 1)} | {pct(te.get('wer_seen'))} | {pct(te.get('wer_unseen'))} | "
            f"{te.get('bertscore_f1', float('nan')):.4f} | "
            + (f"{v['bertscore_delta']:+.4f} [{bci[0]:+.4f}, {bci[1]:+.4f}]" if bci else "") + " |")
    o = s["oracle"]
    lines += ["", f"Oracle WER (best candidate per trial, the headroom): BIT-test {pct(o['test']['wer'])}% "
                  f"(seen {pct(o['test'].get('wer_seen'))}%, unseen {pct(o['test'].get('wer_unseen'))}%), "
                  f"BIT-val {pct(o['val']['wer'])}%."]
    gt = s.get("generation_timing", {})
    if gt:
        lines += ["", "Runtime: generation " + ", ".join(
            f"{k} {v['seconds_this_session'] / 60:.1f} min ({v['sec_per_candidate']} s/candidate)" for k, v in gt.items())
            + (f"; OPT {s['opt_timing']['seconds'] / 60:.1f} min for {s['opt_timing']['texts_scored']} texts "
               f"({s['opt_timing']['ms_per_text']} ms/text, {s['opt_timing']['device']})" if "opt_timing" in s else "") + "."]
    return "\n".join(lines) + "\n"


def write_table() -> Path:
    rows = []
    for p in sorted(OUT.glob("*/summary.json")):
        s = json.loads(p.read_text())
        r = s["rules"]
        row = {"run": s["run"], "decoder": s.get("llm"), "greedy": r.get("greedy", {}).get("test", {}).get("wer")}
        for rule in ("opt", "dec_opt"):
            if rule in r:
                row[rule] = r[rule]["test"]["wer"]
                row[f"{rule}_d"] = r[rule]["vs_greedy"]
        row["oracle"] = s["oracle"]["test"]["wer"]
        rows.append(row)
    lines = ["# N-best rescoring side test: all runs", "",
             "**Side test, never a headline number** (docs/RESCORING.md). BIT-test corpus WER %, weights tuned on "
             "BIT-val only; Δ vs greedy in pp with a paired 95% bootstrap CI.", "",
             "| run | decoder | greedy | OPT only (Δ) | decoder + α·OPT + β·words (Δ) | oracle |", "|---|---|---|---|---|---|"]

    def cell(row, rule):
        if rule not in row:
            return ""
        d = row[f"{rule}_d"]
        return f"{pct(row[rule])} ({100 * d['delta']:+.2f} [{100 * d['ci_low']:+.2f}, {100 * d['ci_high']:+.2f}])"
    for row in rows:
        lines.append(f"| {row['run']} | {row['decoder']} | {pct(row['greedy'])} | {cell(row, 'opt')} | "
                     f"{cell(row, 'dec_opt')} | {pct(row['oracle'])} |")
    path = OUT / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("opt")
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--skip-mac-checks", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--no-bertscore", action="store_true")
    sub.add_parser("table")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.cmd == "table":
        print(write_table())
        return 0
    return {"opt": cmd_opt, "report": cmd_report}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
