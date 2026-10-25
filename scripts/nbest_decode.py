#!/usr/bin/env python
"""N-best candidates from a trained Stage 2 decoder, for the rescoring SIDE TEST (docs/RESCORING.md). Never a headline
number: docs/DESIGN.md keeps LM rescoring out of every headline result.

    <bit env>/python scripts/nbest_decode.py --run B_ssl_s42 [--splits val test] [--device auto|cuda|mps|cpu]
        [--data-root DIR] [--hf-root DIR] [--ckpt-root DIR] [--results-root DIR] [--out DIR] [--limit N]
        [--config configs/rescoring.yaml]
    <bit env>/python scripts/nbest_decode.py --run B_ssl_s42 --gate [--gate-limit N]   # reproduction check only

The run is rebuilt eval-only as train_stage2.py --eval-only builds it: BIT's sentence config, the run's LLM entry and
prompt from its run.json (checked against it), stage2.fix_padding, BEST via stage2.load_model_state. It is built
directly, without BIT's Trainer (whose module imports deepspeed and wandb). Paths are arguments, so the same code runs
on any machine: --data-root holds brandman_2024_text/hdf5_data_final (only data_val.hdf5 is needed: BIT-val and
BIT-test are both drawn from it), --hf-root the LLM snapshots (by the directory name the run's config names), and
--ckpt-root <run>/BEST. --results-root/<run> holds the run's run.json and its own decodes (the reproduction check and
the seen-in-train flags come from them).

Per trial (BIT-val and BIT-test, as configured): the greedy decode exactly as stage2.generate decodes it (one trial at a
time, BIT's prompt, autocast bf16 over fp32 weights, <= 25 new tokens), then K nucleus samples (p 0.9, T 0.7, <= 25
tokens; torch seeded with seed * 100003 + trial once per trial, as stage2.generate). AR decoders draw the K samples in
one generate call (num_return_sequences); a block-diffusion decoder runs its plain sampler K times (and K more per
extra sub-block size in diffusion_orders). Per candidate, the decoder's own score: AR = summed and mean log-probability
of its tokens (up to and including EOS) from the raw logits of the generate call; diffusion = the same over each token's
log-probability when the sampler committed it (BCI_DiffusionLLM.commit_logprobs).

BIT's Aero path loads Qwen2-Audio's audio tower, which BIT never runs (project_to_audio false). It is replaced by an
empty stand-in before the model moves to the device (a 12 GB card holds the rest; --keep-audio-tower keeps it).

Outputs: <out>/candidates_<split>.jsonl (one line per candidate: split, trial, k, n, reference, seen_in_train,
candidate, is_greedy, decoding, seed, scores{dec_sum, dec_mean, dec_n_tokens, ended_eos}), resume-safe (a rerun skips
complete trials), and candidates_<split>.meta.json (run, device, versions, timing, greedy agreement with the run's own
greedy decodes). --gate writes <out>/gate/: the greedy BIT-test decodes, their exact-string match rate and corpus WER
against results/stage2/<run>/decodes/test_greedy_real.csv, and a cross-check that the first trials equal
stage2.generate's own output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import types
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

from bit_experiments import rescoring, stage2  # noqa: E402
from bit_experiments.textmetrics import normalize  # noqa: E402

DATASET = "brandman_2024_text"
CONFIG = ROOT / "configs" / "rescoring.yaml"


def say(msg: str) -> None:
    print(f"[nbest {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


# ---------------------------------------------------------------- the run, rebuilt eval-only

def ensure_trainer_module() -> None:
    """text_llm.use_llm registers its model class in BIT's models.trainer, whose imports (deepspeed, wandb) an
    inference environment may lack: then a stand-in with NAME2MODEL (as tests/test_aero_diffusion.py's fixture)."""
    try:
        from models import trainer  # noqa: F401
    except ImportError:
        import models

        stub = types.ModuleType("models.trainer")
        stub.NAME2MODEL = {}
        sys.modules["models.trainer"] = stub
        models.trainer = stub


PATH_KEYS = {"method.model_kwargs.llm_path", "data.tokenizer_path", "data.dataset.data_dir"}


def config_diff(a, b, path="") -> list[str]:
    """Differences between two JSON-like config trees (dotted paths), ignoring PATH_KEYS."""
    if path.rstrip(".") in PATH_KEYS:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) if f"{path}{k}" not in PATH_KEYS for d in
                (config_diff(a.get(k), b.get(k), f"{path}{k}.") if k in a and k in b else [f"{path}{k} (one side only)"])]
    return [] if a == b else [f"{path.rstrip('.')}: {json.dumps(a, default=str)[:80]} vs {json.dumps(b, default=str)[:80]}"]


def rebuilt_config(run_json: dict, data_root: Path, hf_root: Path):
    """train_stage2.py's config for the run (eval only) with local paths. This repository's copy of BIT's YAMLs is the
    unpatched one (the GPU host applies scripts/patch_bit_configs.py to its copy), so the patched model keys are taken
    from run.json (the projector: inter_size 1024); afterwards the model, method and data sections must equal what
    run.json recorded, paths aside."""
    from bit_experiments import stage1, text_llm

    rargs = run_json["args"]
    if rargs.get("ablation"):
        raise SystemExit("runs with --ablation are not supported here")
    saved = run_json["config"]
    config = stage1.bit_sentence_config()
    config["seed"] = rargs["seed"]
    config["log_to_wandb"] = False
    config["dirs"]["data_dir"] = str(data_root)
    if rargs.get("llm"):
        ensure_trainer_module()
        text_llm.use_llm(config, rargs["llm"])
    if rargs.get("prompt") is not None:
        config["data"]["prompt"] = rargs["prompt"]
    for k, v in saved["model"]["projector"].items():  # bit_baseline.yaml's projector override, as the run had it
        config["model"]["projector"][k] = v
    kw, saved_kw = config["method"]["model_kwargs"], saved["method"]["model_kwargs"]
    if saved_kw.get("diffusion_sampler") is not None:  # decode with the run's own sampler
        kw["diffusion_sampler"] = dict(saved_kw["diffusion_sampler"])
    snapshot = hf_root / Path(saved_kw["llm_path"]).name
    if rargs.get("llm") and Path(kw["llm_path"]).name != snapshot.name:
        raise SystemExit(f"registry snapshot {Path(kw['llm_path']).name} != the run's {snapshot.name}")
    if not (snapshot / "config.json").exists():
        raise SystemExit(f"LLM snapshot missing: {snapshot}")
    kw["llm_path"] = config["data"]["tokenizer_path"] = str(snapshot)
    mine = json.loads(json.dumps({k: config[k] for k in ("model", "method", "data")}, default=str))
    bad = config_diff(mine, {k: saved[k] for k in ("model", "method", "data")})
    if bad:
        raise SystemExit(f"the rebuilt config differs from run.json: {bad}")
    return config


def model_class(name: str):
    if name == "BCI_AudioLLM":
        from models.bci import BCI_AudioLLM

        return BCI_AudioLLM
    if name == "BCI_TextLLM":
        from bit_experiments.text_llm import BCI_TextLLM

        return BCI_TextLLM
    if name == "BCI_DiffusionLLM":
        from bit_experiments.diffusion_llm import BCI_DiffusionLLM

        return BCI_DiffusionLLM
    raise SystemExit(f"unknown model class {name}")


class UnusedAudioTower:
    """Stand-in for Qwen2-Audio's audio tower (BIT never feeds it audio). It keeps one fp32 parameter so that the
    LLM's .device / .dtype (its first parameter's) stay what they were, and refuses to run."""

    @staticmethod
    def make():
        import torch

        class _Tower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

            @property
            def device(self):  # Qwen2AudioForConditionalGeneration.forward reads audio_tower.device on every call
                return self.anchor.device

            @property
            def dtype(self):
                return self.anchor.dtype

            def forward(self, *args, **kwargs):
                raise RuntimeError("the audio tower was replaced: BIT never uses it (project_to_audio false)")

        return _Tower()


def drop_audio_tower(model) -> dict:
    if getattr(model, "project_to_audio", False):
        raise RuntimeError("project_to_audio is on: the audio tower is used")
    for module in model.llm.modules():
        tower = module._modules.get("audio_tower")  # the module that owns it (PEFT wrappers only forward getattr)
        if tower is not None:
            n = sum(p.numel() for p in tower.parameters())
            module.audio_tower = UnusedAudioTower.make()
            return {"audio_tower_dropped_params": n}
    return {"audio_tower_dropped_params": 0}


def build(run_json: dict, ckpt: Path | None, config, device, keep_tower: bool):
    import torch

    from bit_experiments import stage1

    cls = model_class(config["model"]["model_class"])
    with stage1.in_bit_dir(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(run_json["args"]["seed"])
        model = cls(config.model, **config.method.model_kwargs, **config.method.dataset_kwargs, features="all")
    info = {"padding": stage2.fix_padding(model),
            "load": stage2.load_model_state(model, ckpt) if ckpt else "none: structural check (untrained weights)"}
    if not keep_tower:
        info |= drop_audio_tower(model)
    model.to(device).eval()
    return model, info


def load_splits(config, splits: list[str]):
    """BIT's brandman_2024_text loader (train_stage2.load_dataset without the label fraction), the run's prompt."""
    from transformers import AutoTokenizer

    from bit_experiments import text_llm
    from registry import dataset_registry

    data_config = config.data
    data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{DATASET}"
    module = __import__(dataset_registry[DATASET], fromlist=["load_data"])
    ds = module.load_data(**data_config.dataset)
    ds.pop("holdout", None)
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_path, add_bos_token=False, add_eos_token=False)
    ds = text_llm.create_llm_labels({s: ds[s] for s in splits}, tokenizer, data_config.prompt)
    return ds, tokenizer


def seen_flags(results_dir: Path, split: str, refs: list[str]) -> list:
    """seen_in_train per trial from the run's own decodes (the train split is not needed here); None if absent."""
    path = results_dir / "decodes" / f"{split}_greedy_real.csv"
    if not path.exists():
        return [None] * len(refs)
    df = pd.read_csv(path, keep_default_na=False)
    if len(df) != len(refs) or any(normalize(a) != normalize(b) for a, b in zip(df.reference, refs)):
        raise SystemExit(f"{path}: references differ from the loaded {split} split (data or split order changed?)")
    return [bool(v) if str(v) not in ("True", "False") else str(v) == "True" for v in df.seen_in_train]


# ---------------------------------------------------------------- decoding

class Decoder:
    def __init__(self, model, tokenizer, device, dtype_name: str, cfg: dict):
        import torch

        self.torch = torch
        self.model, self.tok, self.device, self.cfg = model, tokenizer, device, cfg
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
        self.diffusion = hasattr(model, "denoising_ce")
        self.eos = tokenizer.eos_token_id
        self.base = {"pad_token_id": tokenizer.vocab_size, "eos_token_id": tokenizer.eos_token_id}

    def autocast(self):
        return self.torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != self.torch.float32)

    def inputs(self, ds, i):
        """stage2.generate's per-trial inputs: BIT's collate of the lone trial, the prompt as input_ids."""
        torch = self.torch
        inputs, _ = self.collate([ds[i]])
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        prompt = inputs["input_ids"][inputs["targets"] == -100].unsqueeze(0)
        inputs.update(input_ids=prompt, attention_mask=torch.ones_like(prompt))
        inputs.pop("targets", None)
        return inputs

    def text(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def ar(self, inputs, gen: dict, n: int) -> list[dict]:
        extra = {"num_return_sequences": n} if n > 1 else {}
        with self.torch.no_grad(), self.autocast():
            out = self.model.generate(**inputs, **self.base, **gen, **extra, return_dict_in_generate=True,
                                      output_logits=True)
        ids = out.sequences
        scores = rescoring.ar_sequence_scores(self.torch.stack(out.logits, 1), ids, self.eos)
        return [{"candidate": self.text(ids[b]), "scores": scores[b]} for b in range(ids.shape[0])]

    def diff(self, inputs, gen: dict) -> dict:
        self.model.commit_logprobs = []
        try:
            with self.torch.no_grad(), self.autocast():
                ids = self.model.generate(**inputs, **self.base, **gen)
        finally:
            recs, self.model.commit_logprobs = self.model.commit_logprobs, None
        rec = recs[-1]
        if rec["ids"] != ids[0].tolist():
            raise RuntimeError("commit log-probs recorded for another decode")
        s, n = float(sum(rec["logprobs"])), len(rec["ids"])
        return {"candidate": self.text(ids[0]),
                "scores": {"dec_sum": s, "dec_mean": s / max(n, 1), "dec_n_tokens": n, "ended_eos": self.eos in rec["ids"]}}

    def greedy(self, inputs) -> dict:
        gen = {"max_new_tokens": self.cfg["greedy"]["max_new_tokens"], "do_sample": False}
        return self.diff(inputs, gen) if self.diffusion else self.ar(inputs, gen, 1)[0]

    def samples(self, inputs, seed: int) -> list[dict]:
        nuc = self.cfg["nucleus"]
        gen = {"max_new_tokens": nuc["max_new_tokens"], "do_sample": True, "top_p": nuc["top_p"],
               "temperature": nuc["temperature"]}
        k, out = int(self.cfg["n_samples"]), []
        self.torch.manual_seed(seed)
        if not self.diffusion:
            batch = max(1, int(self.cfg.get("ar_batch", k)))
            while len(out) < k:
                out += [c | {"decoding": "nucleus"} for c in self.ar(inputs, gen, min(batch, k - len(out)))]
            return out
        own = self.model.sampler["small_block_size"]
        try:
            for sbs in [own] + [s for s in self.cfg.get("diffusion_orders", []) if s != own]:
                self.model.sampler["small_block_size"] = sbs
                out += [self.diff(inputs, gen) | {"decoding": "nucleus", "sbs": sbs} for _ in range(k)]
        finally:
            self.model.sampler["small_block_size"] = own
        return out


def run_split(dec: Decoder, ds, split: str, refs: list[str], seen: list, out_file: Path, seed: int, limit, log_every=20):
    """Write every trial's candidates (greedy first) to out_file, skipping complete trials. Returns timing info."""
    cf = rescoring.CandidateFile(out_file)
    n = len(ds) if limit is None else min(limit, len(ds))
    todo = [i for i in range(n) if i not in cf.done]
    say(f"{split}: {n - len(todo)} of {n} trials already written, {len(todo)} to go -> {out_file}")
    t0, n_cands = time.time(), 0
    for j, i in enumerate(todo):
        inputs = dec.inputs(ds, i)
        trial_seed = seed * 100003 + i
        cands = [dec.greedy(inputs) | {"decoding": "greedy", "is_greedy": True}]
        cands += [c | {"is_greedy": False} for c in dec.samples(inputs, trial_seed)]
        rows = [{"split": split, "trial": i, "k": k, "n": len(cands), "reference": refs[i], "seen_in_train": seen[i],
                 "candidate": c["candidate"], "is_greedy": c["is_greedy"], "decoding": c["decoding"],
                 **({"sbs": c["sbs"]} if "sbs" in c else {}), "seed": trial_seed, "scores": c["scores"]}
                for k, c in enumerate(cands)]
        cf.append(rows)
        n_cands += len(rows)
        if (j + 1) == log_every or (j + 1) % 100 == 0:
            el = time.time() - t0
            say(f"{split}: {j + 1}/{len(todo)} trials in {el / 60:.1f} min ({el / (j + 1):.2f} s/trial, "
                f"{el / n_cands:.3f} s/candidate); ETA {el / (j + 1) * (len(todo) - j - 1) / 60:.0f} min")
    el = time.time() - t0
    return {"trials_this_session": len(todo), "seconds_this_session": round(el, 1),
            "sec_per_trial": round(el / max(len(todo), 1), 3), "sec_per_candidate": round(el / max(n_cands, 1), 4),
            "n_trials": n, "rows": len(cf.rows)}


def greedy_agreement(cf_rows: list[dict], host_csv: Path) -> dict:
    """The candidates' greedy decodes vs the run's own greedy decodes (exact strings, and after normalization)."""
    if not host_csv.exists():
        return {"host_decodes": None}
    host = pd.read_csv(host_csv, keep_default_na=False).set_index("trial").decoded.astype(str)
    mine = {r["trial"]: r["candidate"] for r in cf_rows if r["is_greedy"]}
    common = sorted(set(mine) & set(host.index))
    exact = float(np.mean([mine[t] == host[t] for t in common])) if common else float("nan")
    normed = float(np.mean([normalize(mine[t]) == normalize(host[t]) for t in common])) if common else float("nan")
    return {"host_decodes": str(host_csv.name), "n": len(common), "exact_match": exact, "normalized_match": normed}


# ---------------------------------------------------------------- the reproduction check

def gate(dec: Decoder, ds, refs, seen, results_dir: Path, out: Path, limit, crosscheck: int) -> dict:
    host_csv = results_dir / "decodes" / "test_greedy_real.csv"
    host = pd.read_csv(host_csv, keep_default_na=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    t0, hyps = time.time(), []
    for i in range(n):
        hyps.append(dec.greedy(dec.inputs(ds, i))["candidate"])
        if i + 1 == 20:
            say(f"gate: 20 trials in {time.time() - t0:.0f} s; ETA {(time.time() - t0) / 20 * (n - 20) / 60:.1f} min")
    sec = (time.time() - t0) / n
    # the candidate path's greedy decode == stage2.generate's own (same function the host evaluation ran)
    own = stage2.generate(dec.model, ds, dec.collate, dec.tok, "greedy", device=dec.device,
                          dtype=dec.dtype, limit=min(crosscheck, n))
    ref_host = host.decoded.astype(str).tolist()[:n]
    host_score = stage2.score(refs[:n], ref_host, np.array(seen[:n], bool) if seen[0] is not None else None)
    mine_score = stage2.score(refs[:n], hyps, np.array(seen[:n], bool) if seen[0] is not None else None)
    exact = [a == b for a, b in zip(hyps, ref_host)]
    res = {"n": n, "device": str(dec.device), "dtype": str(dec.dtype), "sec_per_trial": round(sec, 3),
           "exact_match": float(np.mean(exact)),
           "normalized_match": float(np.mean([normalize(a) == normalize(b) for a, b in zip(hyps, ref_host)])),
           "wer": mine_score["wer"], "host_wer": host_score["wer"], "delta_wer_pp": 100 * (mine_score["wer"] - host_score["wer"]),
           "mine": mine_score, "host": host_score,
           "stage2_generate_crosscheck": {"n": len(own), "equal": all(a == b for a, b in zip(own, hyps))},
           "passed": bool(np.mean(exact) >= 0.97 and abs(mine_score["wer"] - host_score["wer"]) <= 0.005),
           "criterion": "strict: exact >= 97% and |delta WER| <= 0.5 pp (ops' cell job proceeds at >= 90% and <= 0.5 pp)"}
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{dec.device.type}_{str(dec.dtype).split('.')[-1]}"
    stage2.write_decodes(out / f"test_greedy_{tag}.csv", refs[:n], hyps, np.array(seen[:n], dtype=object),
                         {"side_test": "rescoring reproduction check", "device": str(dec.device)})
    (out / f"gate_{tag}.json").write_text(json.dumps(res, indent=1))
    diffs = [(i, a, b) for i, (a, b, e) in enumerate(zip(hyps, ref_host, exact)) if not e][:10]
    for i, a, b in diffs:
        say(f"gate differs, trial {i}: here {a!r} | host {b!r}")
    return res


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", help="default: the config's (val test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], help="default: the config's (bf16, as stage2)")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--hf-root", type=Path, default=ROOT / "checkpoints" / "hf")
    ap.add_argument("--ckpt-root", type=Path, default=ROOT / "checkpoints" / "stage2")
    ap.add_argument("--checkpoint", type=Path, help="default: <ckpt-root>/<run>/BEST")
    ap.add_argument("--results-root", type=Path, default=ROOT / "results" / "stage2")
    ap.add_argument("--out", type=Path, help="default: results/rescoring/<run>")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    ap.add_argument("--n-samples", type=int, help="override the config's K")
    ap.add_argument("--gate", action="store_true", help="reproduction check only: greedy BIT-test vs the run's decodes")
    ap.add_argument("--gate-limit", type=int)
    ap.add_argument("--gate-crosscheck", type=int, default=20)
    ap.add_argument("--keep-audio-tower", action="store_true")
    ap.add_argument("--structural", action="store_true",
                    help="build the run's model without its BEST (untrained heads) and decode --limit trials: checks "
                         "the model path before a checkpoint exists; needs --out")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (torch); the loader is single-threaded")
    args = ap.parse_args()
    paths = {k: Path(getattr(args, k)).resolve() for k in ("data_root", "hf_root", "ckpt_root", "results_root")}
    cfg = yaml.safe_load(args.config.read_text())["candidates"]
    if args.n_samples is not None:
        cfg["n_samples"] = args.n_samples
    splits = ["test"] if args.gate else (args.splits or cfg["splits"])
    results_dir = paths["results_root"] / args.run
    run_json = json.loads((results_dir / "run.json").read_text())
    ckpt = (args.checkpoint or paths["ckpt_root"] / args.run / "BEST").resolve()
    out = (args.out or ROOT / "results" / "rescoring" / args.run).resolve()
    os.chdir(BIT)  # BIT opens configs/ by relative path

    import torch
    import transformers

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    t0 = time.time()
    config = rebuilt_config(run_json, paths["data_root"], paths["hf_root"])
    dataset, tokenizer = load_splits(config, splits)
    for s in splits:
        if len(dataset[s]) != run_json["n"][s]:
            raise SystemExit(f"{s}: {len(dataset[s])} trials here, {run_json['n'][s]} in run.json")
    if args.structural and (args.out is None or args.gate):
        raise SystemExit("--structural needs --out (and no --gate)")
    model, info = build(run_json, None if args.structural else ckpt, config, device, args.keep_audio_tower)
    say(f"{args.run}: {config['model']['model_class']} on {device}, built and loaded in {time.time() - t0:.0f} s "
        f"({info['load']}, audio tower dropped: {info.get('audio_tower_dropped_params', 0):,} params)")
    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn
    from utils.eval_phoneme import get_model_inputs

    dec = Decoder(model, tokenizer, device, args.dtype or cfg["dtype"], cfg)
    dec.collate = partial(pad_collate_fn, model_inputs=get_model_inputs(model), **config.method.dataloader_kwargs)
    ds = {s: SpikingDatasetForDecoding(dataset, s, **config.method.dataset_kwargs) for s in splits}
    refs = {s: [r["sentence"] for r in dataset[s]] for s in splits}
    seen = {s: seen_flags(results_dir, s, refs[s]) for s in splits}

    if args.gate:
        res = gate(dec, ds["test"], refs["test"], seen["test"], results_dir, out / "gate", args.gate_limit,
                   args.gate_crosscheck)
        say(f"gate: exact {res['exact_match']:.4f}, normalized {res['normalized_match']:.4f}, WER {100 * res['wer']:.2f}% "
            f"vs host {100 * res['host_wer']:.2f}% ({res['delta_wer_pp']:+.2f} pp), stage2.generate cross-check "
            f"{res['stage2_generate_crosscheck']}, {res['sec_per_trial']} s/trial -> "
            f"strict check (>= 97% exact, <= 0.5 pp) {'passed' if res['passed'] else 'not passed'}")
        return 0 if res["passed"] else 3

    seed = run_json["args"]["seed"] if cfg.get("seed") is None else int(cfg["seed"])
    for s in splits:
        f = out / f"candidates_{s}.jsonl"
        timing = run_split(dec, ds[s], s, refs[s], seen[s], f, seed, args.limit)
        cf = rescoring.CandidateFile(f)
        meta = {"side_test": "N-best rescoring (docs/RESCORING.md); never a headline number", "run": args.run,
                "split": s, "checkpoint": str(ckpt), "model_class": config["model"]["model_class"],
                "llm": run_json["args"].get("llm") or "aero-1-audio", "prompt": config["data"]["prompt"],
                "sampler": getattr(model, "sampler", None), "device": str(device), "dtype": args.dtype or cfg["dtype"],
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "seed": seed,
                "candidates": cfg, "load": info, "timing": timing, "limit": args.limit,
                "greedy_vs_run": greedy_agreement(cf.rows, results_dir / "decodes" / f"{s}_greedy_real.csv")}
        f.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
        say(f"{s}: done ({timing}); greedy vs the run's decodes: {meta['greedy_vs_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
