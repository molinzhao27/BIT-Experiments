#!/usr/bin/env python
"""Phase 4a: run BIT's own pipeline on Brain-to-Text '25 (T15) and collect the results.

    python scripts/phase4/run_bit_baseline.py [--stages phoneme sentence eval] [--splits val test]
    python scripts/phase4/run_bit_baseline.py --stages phoneme eval_phoneme --phoneme-init checkpoints/ssl_human/BEST_core_ft \
        --tag armA_ssl --out results/phase4/armA_ssl_phoneme     # arm A's phoneme stage from the SSL encoder

Stages (BIT's scripts, BIT's environment, cwd = external/bit):
  phoneme   train.py --training_mode train_from_scratch --task phoneme   (CTC, 800 epochs)
  sentence  train.py --training_mode finetune --task sentence --ft_ckpt <phoneme BEST>  (LLM, 150 epochs)
  eval      eval_phoneme.py and eval_llm.py on each split (as shipped, --nbest 0: ONE teacher-forced forward
            pass over prompt + reference, argmax per position; its generation settings are never used)
  eval_phoneme  eval_phoneme.py only (PER per split -> <out>/phoneme_summary.json)
With --phoneme-init, the phoneme stage is BIT's `--training_mode finetune --task phoneme --ft_ckpt <init>` instead
(same 800 epochs; BIT loads the encoder core, the read-in for TX+SBP starts fresh). --tag prefixes the log names.
Splits: BIT halves the official val split (seed 42) into "val" (used for checkpoint selection and
its reported numbers) and "test" (untouched; our clean held-out number). The official test split
has no labels and is never used.

Writes results/bit_baseline/{decodes_<split>.csv, summary.json}, appends results/phase4/metrics.csv,
logs to logs/bit/. Records wall time and peak GPU memory per stage. Run it through the shared GPU
lock (scripts/remote/gpu_job.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder.textmetrics import word_errors, wer  # noqa: E402

DATASET = "brandman_2024_text"


class GpuPeak:
    """Poll nvidia-smi for the peak memory of our process tree's GPU use."""

    def __init__(self, every_s: float = 5.0):
        self.peak_mib, self.every, self._stop = 0, every_s, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            if out:
                self.peak_mib = max(self.peak_mib, int(out.splitlines()[0]))
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def log_path(cfg: dict, stage: str) -> Path:
    return ROOT / "logs" / "bit" / f"{cfg['tag'] + '_' if cfg['tag'] else ''}{stage}.log"


def run(stage: str, args: list[str], cfg: dict, timings: dict) -> None:
    bit = ROOT / cfg["bit"]["path"]
    py = Path(cfg["env"]["prefix"]) / "bin" / "python"
    log = log_path(cfg, stage)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1", "HF_HOME": "/workspace/.cache/huggingface",
                        # deepspeed (imported, unused) needs a CUDA toolkit + C compiler on import
                        "CC": "/opt/worm-build/bin/gcc", "CUDA_HOME": "/opt/worm-build", "DS_SKIP_CUDA_CHECK": "1",
                        "DS_BUILD_OPS": "0"}
    print(f"[{time.strftime('%H:%M:%S')}] {stage}: {' '.join(args)}", flush=True)
    t0 = time.time()
    with GpuPeak() as gpu, open(log, "a") as fh:
        fh.write(f"\n===== {time.strftime('%F %T')} {' '.join(args)}\n")
        fh.flush()
        rc = subprocess.run([str(py), *args], cwd=bit, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    timings[stage] = {"wall_h": round((time.time() - t0) / 3600, 3), "peak_gpu_mib": gpu.peak_mib, "rc": rc}
    print(f"[{time.strftime('%H:%M:%S')}] {stage}: rc={rc} {timings[stage]}", flush=True)
    if rc != 0:
        raise SystemExit(f"{stage} failed (see {log})")


def ckpt(cfg: dict, mode: str, task: str) -> Path:
    """BIT's checkpoint dir: <checkpoint_dir>/<mode>-ndt-<task>-<dataset>-feat_all/BEST."""
    base = Path(cfg.get("ckpt_root") or cfg["overrides"]["all"]["dirs.checkpoint_dir"])
    return base / f"{mode}-ndt-{task}-{DATASET}-feat_all" / "BEST"


def last_line_with(path: Path, needle: str) -> str | None:
    lines = [ln for ln in path.read_text().splitlines() if needle in ln] if path.exists() else []
    return lines[-1] if lines else None


def collect(cfg: dict, splits: list[str], timings: dict) -> None:
    out = cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    sent = ckpt(cfg, "finetune", "sentence")
    rows = []
    for split in splits:
        preds = sorted(sent.glob(f"baseline_{split}_predicted_sentences_*.csv"))[-1]
        labels = np.load(sent / f"{split}_sentence_label.npy", allow_pickle=True)
        hyp = pd.read_csv(preds, keep_default_na=False).text.astype(str).tolist()
        refs = [str(x) for x in labels]
        errs = [word_errors(r, h) for r, h in zip(refs, hyp, strict=True)]
        pd.DataFrame({"trial": range(len(refs)), "reference": refs, "decoded": hyp,
                      "word_errors": [e for e, _ in errs], "ref_words": [n for _, n in errs]}
                     ).to_csv(out / f"decodes_{split}.csv", index=False)
        per_line = last_line_with(log_path(cfg, f"eval_phoneme_{split}"), "PER on")
        rows.append({"model": "BIT released pipeline (sentence stage does not load the phoneme encoder)", "arm": "bit_baseline", "seed": 42,
                     "split": f"{DATASET}:{split}", "decoding": "teacher-forced argmax (BIT eval_llm.py default, --nbest 0; not free generation)",
                     "n_trials": len(refs), "wer": wer(refs, hyp),
                     "per": float(per_line.split()[0]) if per_line else None})
    summary = {"llm": cfg["llm"], "bit_commit": cfg["bit"]["commit"], "stages": timings, "metrics": rows,
               "time": time.strftime("%F %T")}
    old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    summary["stages"] = old.get("stages", {}) | timings
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if cfg["ckpt_root"] is None:  # smoke tests don't touch the shared metrics file
        m = ROOT / "results" / "phase4" / "metrics.csv"
        m.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(m, mode="a", header=not m.exists(), index=False)
    print(json.dumps(rows, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4" / "bit_baseline.yaml")
    ap.add_argument("--stages", nargs="+", default=["phoneme", "sentence", "eval"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--ckpt-root", help="evaluate checkpoints under this root instead (smoke tests)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bit_baseline")
    ap.add_argument("--phoneme-init", help="SSL checkpoint for the phoneme stage (BIT finetune mode)")
    ap.add_argument("--tag", default="", help="prefix for log names (e.g. armA_ssl)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg["ckpt_root"] = args.ckpt_root
    cfg["out"] = args.out if args.out.is_absolute() else ROOT / args.out
    cfg["tag"] = args.tag
    timings: dict = {}
    common = ["--dataset", DATASET, "--features", "all", "--encoder", "ndt"]
    ph_mode = "finetune" if args.phoneme_init else "train_from_scratch"
    if args.phoneme_init and {"sentence", "eval"} & set(args.stages):
        raise SystemExit("--phoneme-init: only the phoneme and eval_phoneme stages (the sentence-stage checkpoint "
                         "dir is shared with 4a; arms train their sentence stage with train_stage2.py)")
    if "phoneme" in args.stages:
        init = ["--ft_ckpt", str(ROOT / args.phoneme_init)] if args.phoneme_init else []
        run("phoneme", ["train.py", "--training_mode", ph_mode, *common, "--task", "phoneme", *init], cfg, timings)
    if "eval_phoneme" in args.stages:
        per = {}
        for split in args.splits:
            run(f"eval_phoneme_{split}", ["eval_phoneme.py", "--model_path", str(ckpt(cfg, ph_mode, "phoneme")),
                                          "--eval_split", split], cfg, timings)
            line = last_line_with(log_path(cfg, f"eval_phoneme_{split}"), "PER on")
            per[split] = float(line.split()[0]) if line else None
        cfg["out"].mkdir(parents=True, exist_ok=True)
        (cfg["out"] / "phoneme_summary.json").write_text(json.dumps(
            {"checkpoint": str(ckpt(cfg, ph_mode, "phoneme")), "init": args.phoneme_init, "per": per,
             "stages": timings, "bit_commit": cfg["bit"]["commit"], "time": time.strftime("%F %T")}, indent=1))
        print(json.dumps(per), flush=True)
    if "sentence" in args.stages:
        run("sentence", ["train.py", "--training_mode", "finetune", *common, "--task", "sentence",
                         "--ft_ckpt", str(ckpt(cfg, ph_mode, "phoneme"))], cfg, timings)
    if "eval" in args.stages:
        for split in args.splits:
            run(f"eval_phoneme_{split}", ["eval_phoneme.py", "--model_path",
                                          str(ckpt(cfg, ph_mode, "phoneme")), "--eval_split", split],
                cfg, timings)
            run(f"eval_llm_{split}", ["eval_llm.py", "--model_path", str(ckpt(cfg, "finetune", "sentence")),
                                      "--eval_split", split], cfg, timings)
        collect(cfg, args.splits, timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
