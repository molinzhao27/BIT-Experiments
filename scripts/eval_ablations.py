#!/usr/bin/env python
"""Eval-only input ablations of trained Stage 2 runs: which parts of the neural input do the decoders use?

    <bit env>/python scripts/eval_ablations.py [--runs RUN ...] [--splits test] [--ablations NAME ...]
        [--device cuda] [--budget-minutes 75] [--limit N] [--config configs/eval_ablations.yaml]
    .venv/bin/python scripts/eval_ablations.py --phonetic-only [--runs RUN ...]

Same weights, greedy decoding; only each trial's neural features change, before BIT's collate, so the encoder's
smoothing, patching and read-in see the ablated input. The ablations are defined in configs/eval_ablations.yaml
(docs/EVAL_ABLATIONS.md has the method and the channel map with its source):
  none              the unablated input, decoded again: the pairing baseline, and a check that the rebuilt model
                    reproduces the run's own <split>_greedy_real decodes (`matches_saved`)
  drop_<array>      one electrode array's features (threshold crossings and spike-band power) set to the fill
  only_<array>      every other array's features set to the fill
  time_*            only part of each trial kept (first / second / middle half of its bins); the rest of the bins
                    = the fill on every feature; the trial keeps its length
  scramble_patches  the trial's encoder patches (5 bins = 100 ms) in a seeded random order (trial i:
                    default_rng([seed, i]), the same order for every run); bins inside a patch keep their order;
                    a trailing partial patch stays last
  reverse_time      the trial's bins in reverse order
Fill = each feature's mean over every training-set bin (`fill: train_mean`; the features are z-scored per channel, so
~0), or 0 (`fill: zero`). The models never saw a constant feature.

Each run is rebuilt eval-only from its run.json as scripts/decoding_order_sweep.py and train_stage2.py --eval-only do
(BIT's sentence config, the run's LLM entry and prompt, stage2.fix_padding, BEST via stage2.load_model_state), for
every model class (BCI_AudioLLM = BIT's Aero path, BCI_TextLLM, BCI_DiffusionLLM; a diffusion run decodes with its
run.json sampler). Only finished runs (summary.json) are evaluated. One process loads the raw dataset once and each
model once, and decodes all of its ablations. A failing run or ablation is logged (results/stage2/<run>/ablations/
error.log, `[ablation] <run> <name>: ERROR ...`) and skipped; the process exits 0. The estimated GPU time is printed
first; a run that would push the batch past --budget-minutes is skipped (the first run always runs).

Per (run, split, ablation): decodes (stage2.write_decodes), WER (stage2.score), a paired trial bootstrap of
WER(ablation) - WER(none) with resamples shared within the run, and the word substitutions' sound-alike share and
phoneme-similarity excess (scripts/phonetic_error_analysis.py's functions; needs pronouncing + wordfreq: if BIT's env
lacks them, run --phonetic-only in the project .venv on the saved decodes afterwards).
Outputs: results/stage2/<run>/ablations/{summary.json, error.log, <name>/<split>_greedy.csv + .meta.json};
results/stage2/ablations_summary.{json,md} over every run with ablations (rewritten after each run).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

from bit_experiments import stage2  # noqa: E402
from bit_experiments.textmetrics import word_errors  # noqa: E402

DATASET = "brandman_2024_text"
RESULTS = ROOT / "results" / ("phase4" if (ROOT / "results" / "phase4").is_dir() else "") / "stage2"
CHECKPOINTS = ROOT / "checkpoints" / "stage2"
CONFIG = ROOT / "configs" / "eval_ablations.yaml"
PHONETIC_SCRIPT = ROOT / "scripts" / "phonetic_error_analysis.py"
PHONETIC_DEPS = ("editdistance", "pronouncing", "wordfreq", "pandas")


# ---------------------------------------------------------------- config and ablation transforms

def load_config(path: Path = CONFIG) -> dict:
    """The ablation config, with the channel map checked: every array covers each feature type with disjoint
    ranges, and the arrays together cover the n_features columns exactly once."""
    cfg = yaml.safe_load(Path(path).read_text())
    cols = np.concatenate([array_columns(cfg, [a]) for a in cfg["arrays"]])
    if sorted(cols.tolist()) != list(range(cfg["n_features"])):
        raise ValueError(f"{path}: the arrays do not cover columns 0..{cfg['n_features'] - 1} exactly once")
    for name in cfg["default"]:
        if name not in cfg["ablations"]:
            raise ValueError(f"{path}: default ablation {name!r} is not defined")
    return cfg


def array_columns(cfg: dict, arrays: list[str], feature_types: list[str] | None = None) -> np.ndarray:
    """Model input columns of `arrays` (every feature type, or only `feature_types`), sorted."""
    cols = [c for a in arrays for f in (feature_types or cfg["feature_types"]) for c in range(*cfg["arrays"][a][f])]
    return np.array(sorted(cols), dtype=int)


def make_transform(spec: dict, cfg: dict, fill: np.ndarray, patch: int):
    """fn(spikes (T, C), trial index) -> ablated copy (same shape and dtype); None for `none`."""
    kind = spec["kind"]
    if kind == "none":
        return None
    if kind in ("drop_arrays", "keep_arrays", "drop_features"):
        if kind == "drop_features":
            cols = array_columns(cfg, list(cfg["arrays"]), spec["features"])
        else:
            cols = array_columns(cfg, spec["arrays"])
            if kind == "keep_arrays":
                cols = np.setdiff1d(np.arange(cfg["n_features"]), cols)

        def fill_columns(x, i):
            y = x.copy()
            y[:, cols] = fill[cols]
            return y
        return fill_columns
    if kind == "keep_window":
        lo, hi = spec["window"]

        def keep_window(x, i):
            a, b = window_bins(len(x), lo, hi)
            y = np.broadcast_to(fill, x.shape).astype(x.dtype)
            y[a:b] = x[a:b]
            return y
        return keep_window
    if kind == "scramble_patches":
        p = patch if spec.get("patch", "encoder") == "encoder" else int(spec["patch"])
        seed = int(spec.get("seed", cfg["seed"]))

        def scramble(x, i):
            n = len(x) // p
            order = np.random.default_rng([seed, i]).permutation(n)
            y = x.copy()
            if n:
                y[: n * p] = x[: n * p].reshape(n, p, x.shape[1])[order].reshape(n * p, x.shape[1])
            return y
        return scramble
    if kind == "reverse":
        return lambda x, i: x[::-1].copy()
    raise ValueError(f"unknown ablation kind {kind!r}")


def window_bins(n_bins: int, lo: float, hi: float) -> tuple[int, int]:
    """The kept bins [a, b) of a trial of n_bins: floor(lo * n) .. floor(hi * n)."""
    return int(np.floor(lo * n_bins)), int(np.floor(hi * n_bins))


class Ablated:
    """A SpikingDatasetForDecoding with fn applied to each trial's spikes (what stage2.generate needs: len, [i])."""

    def __init__(self, ds, fn):
        self.ds, self.fn = ds, fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = dict(self.ds[i])  # never write into the wrapped dataset's rows
        if self.fn is not None:
            x = np.asarray(item["spikes"])
            y = self.fn(x, i)
            if y.shape != x.shape or y.dtype != x.dtype:
                raise RuntimeError(f"ablation changed the input from {x.shape} {x.dtype} to {y.shape} {y.dtype}")
            item["spikes"] = y
        return item


def train_mean(rows: list[dict], n_features: int) -> tuple[np.ndarray, int]:
    """Each feature's mean over every bin of every training trial (float64 sums), and the number of bins."""
    total, n = np.zeros(n_features), 0
    for r in rows:
        x = np.asarray(r["spikes"])
        total += x.sum(0, dtype=np.float64)
        n += len(x)
    return (total / max(n, 1)).astype(np.float32), n


def fill_vector(cfg: dict, train_rows: list[dict]) -> tuple[np.ndarray, dict]:
    if cfg["fill"] == "zero":
        fill, n = np.zeros(cfg["n_features"], dtype=np.float32), 0
    elif cfg["fill"] == "train_mean":
        fill, n = train_mean(train_rows, cfg["n_features"])
    else:
        raise ValueError(f"unknown fill {cfg['fill']!r}")
    info = {"kind": cfg["fill"], "n_train_trials": len(train_rows), "n_train_bins": n}
    for f in cfg["feature_types"]:
        v = fill[array_columns(cfg, list(cfg["arrays"]), [f])]
        info[f] = {"mean": float(v.mean()), "mean_abs": float(np.abs(v).mean()), "max_abs": float(np.abs(v).max())}
    info["vector"] = [float(v) for v in fill]  # exact (float32 values)
    return fill, info


# ---------------------------------------------------------------- statistics

def per_trial_errors(refs: list[str], hyps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    e = np.array([word_errors(r, h) for r, h in zip(refs, hyps, strict=True)], dtype=float).reshape(-1, 2)
    return e[:, 0], e[:, 1]


def resamples(n: int, cfg: dict) -> np.ndarray:
    """(n_resamples, n) trial indices, one draw per run, shared by its ablations (paired across ablations too)."""
    b = cfg["bootstrap"]
    return np.random.default_rng(b["seed"]).integers(0, n, size=(b["n_resamples"], n), dtype=np.int32)


def paired_delta(err: np.ndarray, base: np.ndarray, words: np.ndarray, idx: np.ndarray) -> dict:
    """WER(ablation) - WER(none) on the same trials, with the trial bootstrap's 95% CI and two-sided p."""
    d = (err[idx].sum(1) - base[idx].sum(1)) / np.maximum(words[idx].sum(1), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_wer": float((err.sum() - base.sum()) / max(words.sum(), 1)), "ci": [float(lo), float(hi)],
            "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "trials_worse": int((err > base).sum()), "trials_better": int((err < base).sum()),
            "trials_same": int((err == base).sum())}


# ---------------------------------------------------------------- phonetic error pattern (phonetic_error_analysis.py)

def phonetic_missing() -> list[str]:
    return [m for m in PHONETIC_DEPS if importlib.util.find_spec(m) is None]


def load_phonetic_module():
    """scripts/phonetic_error_analysis.py as a module. It sets HF_HUB_OFFLINE on import; that is undone here, so the
    next run's model loading is unaffected (no embeddings are used: sem / ctx are skipped)."""
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


def decodes_frame(refs: list[str], hyps: list[str], seen) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    return pd.DataFrame({"trial": range(len(hyps)), "reference": refs[: len(hyps)], "decoded": hyps,
                         "seen_in_train": np.asarray(seen[: len(hyps)], dtype=bool)})


def read_decodes(path: Path):
    import pandas as pd

    df = pd.read_csv(path, keep_default_na=False)
    df["seen_in_train"] = df["seen_in_train"].astype(str).str.lower().eq("true")
    return df


def phonetic(run: str, frames: dict, pcfg: dict) -> dict:
    """Per ablation: the substitutions' sound-alike share (phon >= 0.5) and phoneme-similarity excess over the
    length-matched null of that decode set's own substitutes (also beyond spelling, phon|orth), with trial-bootstrap
    CIs, as phonetic_error_analysis.py computes them for a run; and the paired change in excess vs `none`."""
    pea = load_phonetic_module()
    loaded = {name: (df, *pea.extract(df)) for name, df in frames.items()}
    words = {p[k] for _, pairs, _, _ in loaded.values() for p in pairs for k in ("ref", "hyp")}
    words |= {w for *_, rw in loaded.values() for ws in rw.values() for w in ws}
    lex = pea.Lexicon(words, None)
    costs = pea.substitution_costs()
    trials = sorted({int(t) for df, *_ in loaded.values() for t in df["trial"]})
    stats = pea.Stats(trials, pcfg["n_boot"], 1, pcfg["seed"])
    out, analyses = {}, {}
    for name, (df, pairs, ops, rw) in loaded.items():
        try:
            ra = pea.RunAnalysis(run, name, df, pairs, ops, rw, lex, costs, pcfg["min_null"])
            ra.build(None)
            s = pea.summarize(ra, stats, ["all"], perm=False)
            a = s["slices"]["all"]
            out[name] = {"n_sub": s["n_sub"], "n_pairs": s["n_pairs"], "oov_rate": s["oov_rate"], "ops": s["ops"]}
            out[name] |= {k: a[k] for k in ("alike", "phon", "phon|orth", "homophone") if k in a}
            if ra.pairs:
                analyses[name] = ra
        except Exception as e:  # e.g. no substitutions at all
            out[name] = {"error": repr(e)}
    base = analyses.get("none")
    for name, ra in analyses.items():
        if base is not None and name != "none":
            for metric in ("alike", "phon"):
                c = pea.contrast(stats, ra, base, metric)
                out[name][f"{metric}_excess_vs_none"] = {"diff": c["diff"], "ci": c["ci"]}
    return out


# ---------------------------------------------------------------- rebuilding a finished run (eval only)

@dataclass
class Bundle:
    """A loaded run: the model and everything stage2.generate needs, per split."""
    model: object
    tokenizer: object
    collate: object
    splits: dict
    refs: dict
    seen: dict
    patch: int
    fill: np.ndarray
    info: dict
    trainer: object = None


def load_raw(data_config) -> dict:
    """BIT's brandman_2024_text loader (train / val / test, no holdout), as train_stage2.load_dataset."""
    from registry import dataset_registry

    module = __import__(dataset_registry[DATASET], fromlist=["load_data"])
    ds = module.load_data(**data_config.dataset)
    ds.pop("holdout", None)  # no labels; never used
    return ds


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, add_bos_token=False, add_eos_token=False)


def load_dataset(config, limit: int | None, cache: dict):
    """The run's dataset with its LLM labels; the raw data is loaded once per process (create_llm_labels rewrites
    every row's token arrays from its sentence, so rows are shared across runs)."""
    from bit_experiments import text_llm

    data_config = config.data
    data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{DATASET}"
    key = json.dumps(data_config.dataset, sort_keys=True, default=str)
    if cache.get("key") != key:
        cache.clear()
        cache.update(key=key, raw=load_raw(data_config))
    ds = cache["raw"]
    if limit:
        ds = {k: v[:limit] for k, v in ds.items()}
    tokenizer = load_tokenizer(data_config.tokenizer_path)
    return text_llm.create_llm_labels(ds, tokenizer, data_config.prompt), tokenizer


def checkpoint_for(run: str, run_json: dict) -> Path:
    ckpt = CHECKPOINTS / run / "BEST"
    if ckpt.is_dir():
        return ckpt
    d = (run_json.get("eval_checkpoint") or {}).get("dir")  # e.g. an eval-only run of BIT's own checkpoint
    if d and Path(d).is_dir():
        return Path(d)
    raise FileNotFoundError(f"no checkpoint: {ckpt} (run.json eval_checkpoint: {d})")


def rebuilt_config(run: str, run_json: dict):
    """train_stage2.py's config for this run (eval only), checked against what run.json recorded."""
    from bit_experiments import stage1, text_llm

    rargs = run_json["args"]
    config = stage1.bit_sentence_config()
    config["seed"] = rargs["seed"]
    config["savestring"] = run
    config["dirs"]["checkpoint_dir"] = str(CHECKPOINTS)
    config["log_to_wandb"] = False
    if rargs.get("llm"):
        text_llm.use_llm(config, rargs["llm"])
    if rargs.get("prompt") is not None:
        config["data"]["prompt"] = rargs["prompt"]
    saved = run_json["config"]
    kw, saved_kw = config["method"]["model_kwargs"], saved["method"]["model_kwargs"]
    checks = {"model_class": (config["model"]["model_class"], saved["model"]["model_class"]),
              "prompt": (config["data"]["prompt"], saved["data"]["prompt"]),
              "llm": (Path(kw["llm_path"]).name, Path(saved_kw["llm_path"]).name),
              "features": (list(config["data"]["dataset"]["features"]), list(saved["data"]["dataset"]["features"])),
              "area": ([config["data"]["dataset"][k] for k in ("area_start", "area_end")],
                       [saved["data"]["dataset"][k] for k in ("area_start", "area_end")])}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(f"{run}: the rebuilt config differs from run.json: {bad}")
    sampler = None
    if saved_kw.get("diffusion_sampler") is not None:  # decode with the run's own sampler
        sampler = dict(saved_kw["diffusion_sampler"])
        kw["diffusion_sampler"] = sampler
    return config, sampler


def load_run(run: str, splits: list[str], limit: int | None, device: str, cache: dict, cfg: dict) -> Bundle:
    from models.trainer import Trainer
    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn
    from utils.eval_phoneme import get_model_inputs

    run_dir = RESULTS / run
    if not (run_dir / "summary.json").is_file():
        raise FileNotFoundError(f"{run_dir}/summary.json missing: the run is not finished")
    run_json = json.loads((run_dir / "run.json").read_text())
    ckpt = checkpoint_for(run, run_json)
    config, sampler = rebuilt_config(run, run_json)
    dataset, tokenizer = load_dataset(config, limit, cache)
    train_sentences = [r["sentence"] for r in dataset["train"]]

    class EvalTrainer(Trainer):
        def set_model(self, model, extra_model_kwargs=None):
            super().set_model(model, extra_model_kwargs)
            self.padding_report = stage2.fix_padding(self.model)  # as every Stage 2 run

    trainer = EvalTrainer(config, dataset=dataset, metric_fns={}, extra_model_kwargs={"features": "all"},
                          eval_mode=True)
    model = trainer.accelerator.unwrap_model(trainer.model)
    load = stage2.load_model_state(model, ckpt)
    model.to(device)
    collate = partial(pad_collate_fn, model_inputs=get_model_inputs(model), **config.method.dataloader_kwargs)
    ds = {s: SpikingDatasetForDecoding(dataset, s, **config.method.dataset_kwargs) for s in splits}
    refs = {s: [r["sentence"] for r in dataset[s]] for s in splits}
    seen = {s: stage2.seen_mask(refs[s], train_sentences) for s in splits}
    n_features = ds[splits[0]][0]["spikes"].shape[1]
    if n_features != cfg["n_features"]:
        raise RuntimeError(f"{run}: the input has {n_features} features, the channel map {cfg['n_features']}")
    fill_key = ("fill", cfg["fill"], limit)
    if fill_key not in cache:
        cache[fill_key] = fill_vector(cfg, dataset["train"])
    fill, fill_info = cache[fill_key]
    info = {"run": run, "model_class": config["model"]["model_class"], "llm": run_json["args"].get("llm"),
            "prompt": config["data"]["prompt"], "checkpoint": str(ckpt), "load": load, "sampler": sampler,
            "padding": trainer.padding_report, "arm": run_json["args"].get("arm"),
            "seed": run_json["args"].get("seed"), "fill": fill_info}
    return Bundle(model, tokenizer, collate, ds, refs, seen, int(model.neural_encoder.encoder.embedder.patch_size),
                  fill, info, trainer)


def release(bundle: Bundle) -> None:
    """Drop the run's model (and accelerate's references to it) before the next run is loaded."""
    import torch

    if bundle.trainer is not None:
        try:
            bundle.trainer.accelerator.free_memory()
        except Exception:  # noqa: BLE001 - cleanup only
            pass
    bundle.model = bundle.trainer = bundle.splits = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- timing estimate

def sec_per_trial(run: str, cfg: dict) -> tuple[float, str]:
    """Seconds per decoded trial from the run's own final evaluation (summary.json minutes minus the last
    log.jsonl minutes = loading BEST + its decode sets, nulls and nucleus included), else the config's class default."""
    run_dir = RESULTS / run
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
        n = sum(r["n"] for r in summary["results"].values())
        minutes = summary["minutes"]
        log = run_dir / "log.jsonl"
        if log.is_file():
            logged = [json.loads(line).get("minutes") for line in log.read_text().splitlines() if line.strip()]
            logged = [m for m in logged if m is not None]
            minutes -= max(logged, default=0)
        if n and minutes > 0:
            return minutes * 60 / n, f"own final evaluation ({len(summary['results'])} decode sets)"
    except (OSError, KeyError, ValueError):
        pass
    try:
        cls = json.loads((run_dir / "run.json").read_text())["config"]["model"]["model_class"]
    except (OSError, KeyError, ValueError):
        return float("nan"), "no run.json"
    return cfg["timing"]["default_sec_per_trial"][cls], f"default for {cls}"


def estimate(run: str, cfg: dict, n_ablations: int, splits: list[str], limit: int | None) -> dict:
    spt, source = sec_per_trial(run, cfg)
    try:
        n = json.loads((RESULTS / run / "run.json").read_text())["n"]
        trials = sum(min(n[s], limit) if limit else n[s] for s in splits)
    except (OSError, KeyError, ValueError):
        trials = None
    minutes = (cfg["timing"]["load_minutes"] + n_ablations * trials * spt / 60) if trials else float("nan")
    return {"run": run, "sec_per_trial": round(spt, 4), "source": source, "trials": trials,
            "decodes": n_ablations * trials if trials else None, "minutes": round(minutes, 1)}


# ---------------------------------------------------------------- one run

def log_error(run: str, where: str, exc: BaseException) -> str:
    """Traceback to results/stage2/<run>/ablations/error.log (results/stage2/ablations_errors.log for an unknown run)."""
    if (RESULTS / run).is_dir():
        (RESULTS / run / "ablations").mkdir(parents=True, exist_ok=True)
        path = RESULTS / run / "ablations" / "error.log"
    else:
        path = RESULTS / "ablations_errors.log"
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with open(path, "a") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {run} {where}\n{text}\n")
    print(f"[ablation] {run} {where}: ERROR {exc!r} (traceback in {path})", flush=True)
    return repr(exc)


def summary_path(run: str) -> Path:
    return RESULTS / run / "ablations" / "summary.json"


def evaluate_run(run: str, names: list[str], splits: list[str], args, cfg: dict, cache: dict) -> dict:
    """Load the run once, decode every (split, ablation), score, pair with `none`, and write as it goes. Results of
    an earlier batch with the same --limit are kept (and updated) unless re-run."""
    import torch

    out = RESULTS / run / "ablations"
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # max_mem_gb = this run's peak
    bundle = load_run(run, splits, args.limit, args.device, cache, cfg)
    try:
        path = summary_path(run)
        summary = json.loads(path.read_text()) if path.is_file() else {}
        if summary.get("limit") != args.limit or summary.get("config", {}).get("fill") != cfg["fill"]:
            summary = {}
        summary.setdefault("results", {})
        summary.setdefault("errors", {})
        summary |= bundle.info | {"limit": args.limit, "patch_size": bundle.patch, "config": {
            k: cfg[k] for k in ("seed", "fill", "arrays", "feature_types", "bootstrap", "phonetic")}}
        summary["ablations"] = summary.get("ablations", {}) | {n: cfg["ablations"][n] for n in names}
        for split in splits:
            refs, seen = bundle.refs[split], bundle.seen[split]
            results = summary["results"].setdefault(split, {})
            idx, base, words = None, None, None
            hyps_by = {}
            for name in names:
                where = f"{split} {name}"
                try:
                    fn = make_transform(cfg["ablations"][name], cfg, bundle.fill, bundle.patch)
                    t = time.time()
                    hyps = stage2.generate(bundle.model, Ablated(bundle.splits[split], fn), bundle.collate,
                                           bundle.tokenizer, "greedy", device=args.device)
                    res = {"split": split} | stage2.score(refs, hyps, seen)
                    res["sec_per_trial"] = round((time.time() - t) / max(len(hyps), 1), 4)
                    res["mean_words"] = float(np.mean([len(h.split()) for h in hyps])) if hyps else 0.0
                    err, n_words = per_trial_errors(refs, hyps)
                    if name == "none":
                        base, words = err, n_words
                        idx = resamples(len(hyps), cfg)
                        res |= saved_match(run, split, hyps)
                    elif base is not None:
                        res |= paired_delta(err, base, words, idx)
                    meta = {"model": f"stage2 {run} ablation {name}", "arm": bundle.info["arm"],
                            "seed": bundle.info["seed"], "split": f"{DATASET}:{split}", "decoding": "greedy",
                            "condition": f"ablation_{name}", "ablation": cfg["ablations"][name],
                            "fill": cfg["fill"], "patch_size": bundle.patch, "checkpoint": bundle.info["checkpoint"],
                            "llm": bundle.info["llm"], "prompt": bundle.info["prompt"], "source_run": run}
                    stage2.write_decodes(out / name / f"{split}_greedy.csv", refs, hyps, seen, meta)
                    hyps_by[name] = hyps
                    results[name] = res
                    summary["errors"].pop(where, None)
                    print(f"[ablation] {run} {name}: {json.dumps(res)}", flush=True)
                except Exception as e:  # noqa: BLE001 - one failing ablation must not stop the others
                    summary["errors"][where] = log_error(run, where, e)
                path.write_text(json.dumps(summary, indent=1, default=float))
            add_phonetic(run, split, {n: decodes_frame(refs, h, seen) for n, h in hyps_by.items()}, summary, cfg)
            path.write_text(json.dumps(summary, indent=1, default=float))
        summary["minutes"] = round((time.time() - t0) / 60, 2)
        if torch.cuda.is_available():
            summary["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        path.write_text(json.dumps(summary, indent=1, default=float))
        return summary
    finally:
        release(bundle)


def saved_match(run: str, split: str, hyps: list[str]) -> dict:
    """Does the re-decoded unablated input reproduce the run's own greedy decodes (first len(hyps) trials)?"""
    path = RESULTS / run / "decodes" / f"{split}_greedy_real.csv"
    if not path.is_file():
        return {"matches_saved": None}
    saved = read_decodes(path)["decoded"].tolist()[: len(hyps)]
    differ = sum(a != b for a, b in zip(hyps, saved, strict=True))
    return {"matches_saved": differ == 0, "trials_differing_from_saved": differ}


def add_phonetic(run: str, split: str, frames: dict, summary: dict, cfg: dict) -> None:
    results = summary["results"][split]
    missing = phonetic_missing()
    if missing:
        note = f"skipped: {', '.join(missing)} not installed here; run --phonetic-only in the project .venv"
        for name in frames:
            results[name]["phonetic"] = {"skipped": note}
        print(f"[ablation] {run} phonetic {note}", flush=True)
        return
    try:
        for name, v in phonetic(run, frames, cfg["phonetic"]).items():
            results[name]["phonetic"] = v
            print(f"[ablation] {run} {name} phonetic: {json.dumps(compact_phonetic(v))}", flush=True)
    except Exception as e:  # noqa: BLE001
        summary["errors"][f"{split} phonetic"] = log_error(run, f"{split} phonetic", e)


def compact_phonetic(v: dict) -> dict:
    if "alike" not in v:
        return v
    out = {"n_sub": v["n_sub"], "n_pairs": v["n_pairs"], "sound_alike": round(v["alike"]["obs"], 4),
           "sound_alike_null": round(v["alike"]["null"], 4), "phon_excess": round(v["phon"]["excess"], 4),
           "phon_excess_ci": [round(x, 4) for x in v["phon"]["ci"]]}
    if "phon_excess_vs_none" in v:
        out["phon_excess_vs_none"] = round(v["phon_excess_vs_none"]["diff"], 4)
    return out


# ---------------------------------------------------------------- summaries over runs

def run_label(run: str) -> str:
    try:
        return load_phonetic_module().RUNS.get(run, run)
    except Exception:  # noqa: BLE001 - labels only
        return run


def write_aggregate(cfg: dict) -> None:
    """results/stage2/ablations_summary.{json,md} from every results/stage2/<run>/ablations/summary.json."""
    order = {r: k for k, r in enumerate(cfg["runs"])}
    runs = {}
    for f in sorted(RESULTS.glob("*/ablations/summary.json"), key=lambda p: (order.get(p.parent.parent.name, 99),
                                                                            p.parent.parent.name)):
        s = json.loads(f.read_text())
        runs[f.parent.parent.name] = {k: s.get(k) for k in ("model_class", "llm", "checkpoint", "limit", "patch_size",
                                                            "minutes", "errors")}
        runs[f.parent.parent.name] |= {"fill": {k: v for k, v in (s.get("fill") or {}).items() if k != "vector"},
                                       "results": s.get("results", {})}
    (RESULTS / "ablations_summary.json").write_text(json.dumps({"runs": runs}, indent=1, default=float))
    (RESULTS / "ablations_summary.md").write_text(markdown(runs, cfg))


def fmt_ci(x: dict | None, key: str, scale: float = 100, digits: int = 1, sign: bool = True) -> str:
    if not x or x.get(key) is None:
        return "–"
    s = "+" if sign else ""
    return f"{scale * x[key]:{s}.{digits}f} [{scale * x['ci'][0]:{s}.{digits}f}, {scale * x['ci'][1]:{s}.{digits}f}]"


def markdown(runs: dict, cfg: dict) -> str:
    lines = ["# Eval-only input ablations (scripts/eval_ablations.py)", "",
             "Same weights, greedy decoding; only the neural input changes (configs/eval_ablations.yaml, "
             "docs/EVAL_ABLATIONS.md). ΔWER = WER(ablation) − WER(none), percentage points, paired trial bootstrap "
             "95% CI. Sound-alike = share of in-dictionary substitutions with phoneme similarity ≥ 0.5, observed / "
             "length-matched null. Phon excess = mean phoneme similarity of the substitutions − null (95% CI); "
             "vs none = paired change in that excess. One seed per run.", ""]
    for run, s in runs.items():
        for split, res in s["results"].items():
            fill = s.get("fill") or {}
            lines += [f"## {run_label(run)} — `{run}`, {split}", "",
                      f"{s.get('model_class')}, fill {fill.get('kind')} (|fill| mean "
                      f"{fill.get('tx', {}).get('mean_abs', float('nan')):.3f} TX / "
                      f"{fill.get('sbp', {}).get('mean_abs', float('nan')):.3f} SBP), "
                      f"{s.get('minutes')} min" + (f", limit {s['limit']}" if s.get("limit") else ""), "",
                      "| Ablation | WER | ΔWER [95% CI] | p | trials worse / better | words / trial "
                      "| subs | sound-alike obs / null | phon excess [95% CI] | phon excess vs none |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for name, r in res.items():
                ph = r.get("phonetic") or {}
                label = (cfg["ablations"].get(name) or {}).get("label", name)
                if name == "none":
                    m = r.get("matches_saved")
                    delta = "baseline" + ("" if m is None else (" (= saved)" if m else
                                                               f" (≠ saved: {r['trials_differing_from_saved']})"))
                else:
                    delta = fmt_ci(r, "delta_wer")
                alike = (f"{100 * ph['alike']['obs']:.1f}% / {100 * ph['alike']['null']:.1f}%"
                         if "alike" in ph else ("–" if not ph.get("skipped") else "pending"))
                phon = (f"{ph['phon']['excess']:+.3f} [{ph['phon']['ci'][0]:+.3f}, {ph['phon']['ci'][1]:+.3f}]"
                        if "phon" in ph else "–")
                vs = ph.get("phon_excess_vs_none")
                vs = f"{vs['diff']:+.3f} [{vs['ci'][0]:+.3f}, {vs['ci'][1]:+.3f}]" if vs else "–"
                p = "–" if r.get("p_boot") is None else (f"< {1 / cfg['bootstrap']['n_resamples']:.0e}"
                                                          if r["p_boot"] == 0 else f"{r['p_boot']:.3g}")
                wb = f"{r['trials_worse']} / {r['trials_better']}" if "trials_worse" in r else "–"
                lines.append(f"| {name}: {label} | {100 * r['wer']:.1f}% | {delta} | {p} | {wb} "
                             f"| {r.get('mean_words', float('nan')):.1f} | {ph.get('n_sub', '–')} | {alike} "
                             f"| {phon} | {vs} |")
            lines.append("")
        if s.get("errors"):
            lines += [f"Errors in `{run}`: " + "; ".join(f"{k}: {v}" for k, v in s["errors"].items()), ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- entry points

def phonetic_only(runs: list[str], cfg: dict) -> int:
    """The phonetic measures on saved ablation decodes (project .venv), into each run's summary and the aggregate."""
    missing = phonetic_missing()
    if missing:
        print(f"[ablation] phonetic: {', '.join(missing)} not installed in this environment", flush=True)
        return 0
    for run in runs:
        path = summary_path(run)
        if not path.is_file():
            print(f"[ablation] {run}: no {path}", flush=True)
            continue
        try:
            summary = json.loads(path.read_text())
            for split, res in summary["results"].items():
                frames = {n: read_decodes(f) for n in res
                          if (f := RESULTS / run / "ablations" / n / f"{split}_greedy.csv").is_file()}
                add_phonetic(run, split, frames, summary, cfg)
            path.write_text(json.dumps(summary, indent=1, default=float))
        except Exception as e:  # noqa: BLE001
            log_error(run, "phonetic-only", e)
    write_aggregate(cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--runs", nargs="+", help="finished Stage 2 runs (default: the config's runs, in order)")
    ap.add_argument("--splits", nargs="+", choices=["val", "test"], help="default: the config's (test)")
    ap.add_argument("--ablations", nargs="+", help="names in the config (default: its `default`; none is always first)")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--budget-minutes", type=float, help="skip runs past this estimate (0 = none; default: config)")
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    ap.add_argument("--phonetic-only", action="store_true", help="phonetic measures on saved decodes, no decoding")
    ap.add_argument("--estimate-only", action="store_true", help="print the GPU estimate and exit")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    runs = args.runs or cfg["runs"]
    splits = args.splits or cfg["splits"]
    names = args.ablations or cfg["default"]
    unknown = [n for n in names if n not in cfg["ablations"]]
    if unknown:
        ap.error(f"unknown ablations {unknown} (have: {', '.join(cfg['ablations'])})")
    names = ["none"] + [n for n in dict.fromkeys(names) if n != "none"]
    if args.phonetic_only:
        return phonetic_only(runs, cfg)
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    budget = cfg["budget_minutes"] if args.budget_minutes is None else args.budget_minutes
    exempt = set(cfg["runs"][: cfg.get("budget_exempt", 0)])  # the priority runs; the batch's first run too

    plan = [estimate(r, cfg, len(names), splits, args.limit) for r in runs]
    total = 0.0
    print(f"[ablation] plan: {len(names)} conditions ({', '.join(names)}) x splits {splits}, budget "
          f"{budget or 'none'} min (always run: the first run and {', '.join(sorted(exempt)) or 'none'})", flush=True)
    for p in plan:
        total += 0 if np.isnan(p["minutes"]) else p["minutes"]
        print(f"[ablation] estimate {p['run']}: {p['minutes']} min ({p['decodes']} decodes x {p['sec_per_trial']} s, "
              f"{p['source']}; cumulative {total:.1f} min)", flush=True)
    if args.estimate_only:
        return 0

    os.chdir(BIT)  # BIT opens configs/ by relative path
    t0, cache, actual, predicted = time.time(), {}, 0.0, 0.0
    for k, (run, p) in enumerate(zip(runs, plan, strict=True)):
        elapsed = (time.time() - t0) / 60
        # this batch's measured speed vs the estimates so far, within [0.5, 2]
        scale = min(2.0, max(0.5, actual / predicted)) if predicted else 1.0
        if (k and run not in exempt and budget and not np.isnan(p["minutes"])
                and elapsed + scale * p["minutes"] > budget):
            print(f"[ablation] {run}: skipped (est. {scale * p['minutes']:.1f} min after {elapsed:.1f} min "
                  f"> budget {budget} min)", flush=True)
            continue
        t = time.time()
        try:
            evaluate_run(run, names, splits, args, cfg, cache)
            if not np.isnan(p["minutes"]):
                actual, predicted = actual + (time.time() - t) / 60, predicted + p["minutes"]
        except Exception as e:  # noqa: BLE001 - one failing run must not stop the batch
            log_error(run, "run", e)
        finally:
            try:
                write_aggregate(cfg)
            except Exception as e:  # noqa: BLE001
                log_error(run, "aggregate", e)
        print(f"[ablation] {run}: done in {(time.time() - t) / 60:.1f} min (estimate {p['minutes']} min)",
              flush=True)
    print(f"[ablation] batch done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
