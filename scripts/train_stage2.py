#!/usr/bin/env python
"""Phase 4 Stage 2 (every arm): sentence-level LLM training from an arm's intermediate checkpoint, then evaluation.

    <bit env>/python scripts/train_stage2.py --name A_ssl_s42 --arm A --init <phoneme BEST> [--seed 42]
        [--epochs 150] [--eval-every 5] [--label-fraction 1.0] [--limit N] [--no-train] [--llm NAME] [--prompt TEXT]
    <bit env>/python scripts/train_stage2.py --name B_rand_bit4a --arm B --eval-only <BIT sentence BEST>

--llm / --prompt (LLM-swap side question, docs/DESIGN.md §8): a decoder from configs/llms.yaml instead
of BIT's Aero-1-Audio, and a prompt instead of BIT's ("#" marks the neural tokens; "#" alone = no prompt, which also
drops the prompt-only null). Without them nothing changes.

--ablation (retraining ablations, docs/RETRAINING_ABLATIONS.md): a configs/ablations.yaml entry that changes only
what reaches the projector and LLM (frozen encoder, phoneme bottleneck, phoneme erasure). Without it nothing changes.

--init: arm A = CTC phoneme checkpoint, B / B-long = checkpoints/ssl_human/BEST_core, C / E = Stage 1 checkpoint
(encoder + projector), "none" = random init. Model, data, LoRA, loss (CE + BIT's contrastive term), optimizer and
epochs are BIT's sentence stage as in 4a (configs/bit_baseline.yaml), the same for every arm, except that
training sees each trial as evaluation does (no batch-padding effects; stage2.fix_padding, DESIGN.md §2).
Selection: free-generation (greedy) WER on BIT-val every --eval-every epochs; BIT-test is never used for a choice.
After training, BEST is evaluated: BIT-val and BIT-test, greedy and nucleus, plus null controls on BIT-test.

Outputs: checkpoints/stage2/<name>/BEST (BIT's format, slimmed after evaluation: slim.safetensors holds everything
but the frozen base LLM), results/stage2/<name>/
{run.json, log.jsonl, summary.json, decodes/<split>_<decoding>_<condition>.csv + .meta.json}. Score the decodes with
scripts/evaluate_decodes.py (project venv). Code: src/bit_experiments/stage2.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

from bit_experiments import stage1, stage2  # noqa: E402
from bit_experiments.performance import configure_collators, configure_runtime  # noqa: E402

DATASET = "brandman_2024_text"


def load_dataset(config, fraction: float, seed: int, limit: int | None):
    """train.py's load_brain2text_dataset (BIT), then the label fraction and the smoke-test limit."""
    from transformers import AutoTokenizer

    from bit_experiments.text_llm import create_llm_labels  # BIT's, with integer arrays also for an empty prompt half
    from registry import dataset_registry

    data_config = config.data
    data_config["dataset"]["data_dir"] = f"{config.dirs.data_dir}/{DATASET}"
    module = __import__(dataset_registry[DATASET], fromlist=["load_data"])
    ds = module.load_data(**data_config.dataset)
    ds.pop("holdout", None)  # no labels; never used
    if limit:
        ds = {k: v[:limit] for k, v in ds.items()}
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_path, add_bos_token=False, add_eos_token=False)
    ds = create_llm_labels(ds, tokenizer, data_config.prompt)
    return stage2.subsample_train(ds, fraction, seed), tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True, help="A | B | B-long | C | D | E (metadata only; the recipe is shared)")
    ap.add_argument("--init", default="none", help="intermediate checkpoint dir, or none (random init)")
    ap.add_argument("--eval-only", help="evaluate this trained sentence checkpoint (model.safetensors) instead")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, help="default: BIT's (150, configs/bit_baseline.yaml)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    ap.add_argument("--null-decodings", nargs="*", default=["greedy"], help="decodings for the null controls (none: skip)")
    ap.add_argument("--no-train", action="store_true", help="build, load the init, evaluate at step 0 (smoke tests)")
    ap.add_argument("--llm", help="a configs/llms.yaml entry (DESIGN.md §8); default: BIT's Aero-1-Audio")
    ap.add_argument("--prompt", help='LLM prompt, "#" = the neural tokens ("#" alone: no prompt); default: BIT\'s')
    ap.add_argument("--ablation", help="a configs/ablations.yaml entry (docs/RETRAINING_ABLATIONS.md); default: none")
    ap.add_argument("--runtime", choices=("reference", "optimized"), default="optimized",
                    help="runtime overhead reductions; reference restores the previous workload (same recipe)")
    args = ap.parse_args()
    if args.prompt is not None and args.prompt.count("#") != 1:
        ap.error('--prompt needs exactly one "#" (where the neural tokens go)')
    t0 = time.time()
    out = ROOT / "results" / "stage2" / args.name
    run_ckpt = ROOT / "checkpoints" / "stage2" / args.name
    exists = (run_ckpt.is_dir() and any(run_ckpt.iterdir())) or (out / "summary.json").exists()
    if exists and not args.eval_only and not args.no_train:
        # a finished run's BEST/slim.safetensors would be evaluated (and kept) instead of the new model
        ap.error(f"a run named {args.name!r} already exists ({run_ckpt} / {out}): remove both or pick a new --name")
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(BIT)  # BIT opens configs/ by relative path

    from models.trainer import Trainer
    from utils.datasets import SpikingDatasetForDecoding, pad_collate_fn
    from utils.eval_phoneme import get_model_inputs

    config = stage1.bit_sentence_config()
    config["seed"] = args.seed
    config["savestring"] = args.name
    config["dirs"]["checkpoint_dir"] = str(ROOT / "checkpoints" / "stage2")
    config["log_to_wandb"] = False
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    config["training"]["eval_every"] = args.eval_every
    config["training"]["save_every"] = None  # BEST and LAST only (each a full 8.5 GB copy until slimmed below)
    config["method"]["metric_kwargs"]["metric_name"] = "WER"  # = free-generation greedy WER on BIT-val (below)
    if args.llm:
        from bit_experiments import text_llm

        text_llm.use_llm(config, args.llm)
    if args.prompt is not None:
        config["data"]["prompt"] = args.prompt
    llm = args.llm or "aero-1-audio"
    init = None if args.init == "none" else Path(args.init if Path(args.init).is_absolute() else ROOT / args.init)
    dataset, tokenizer = load_dataset(config, args.label_fraction, args.seed, args.limit)
    train_sentences = [r["sentence"] for r in dataset["train"]]
    log = open(out / "log.jsonl", "a")

    class Stage2Trainer(Trainer):
        """BIT's Trainer; the encoder (and projector) are loaded after the model is built; evaluate() returns
        free-generation WER as the selection metric."""

        def set_model(self, model, extra_model_kwargs=None):
            super().set_model(model, extra_model_kwargs)
            self.padding_report = stage2.fix_padding(self.model)  # every run, also eval-only (DESIGN.md §2)
            self.init_report = stage2.load_init(self.model, init) if not args.eval_only else {"init": None}
            if args.ablation:  # after the init is loaded, before the optimizer is built (it sees the new projector)
                from bit_experiments import neural_ablations

                self.init_report["ablation"] = neural_ablations.apply(
                    self.model, args.ablation, None if args.eval_only else init, args.seed)
            self.runtime_report = configure_runtime(self.model, args.runtime)

        def build_dataloaders(self):
            super().build_dataloaders()
            configure_collators(self, args.runtime)

        def train(self):
            if args.runtime == "reference":
                return super().train()
            from bit_experiments.training_runtime import train_stage2_deferred

            return train_stage2_deferred(self)

        def evaluate(self, additional_metric_fns=None, eval_train_set=False):
            loss, metrics, gt, preds = super().evaluate(additional_metric_fns, eval_train_set)
            model = self.accelerator.unwrap_model(self.model)
            hyps = stage2.generate(model, val_ds, collate, tokenizer, "greedy", device=self.accelerator.device)
            res = stage2.score(val_refs, hyps, val_seen)
            metrics["WER"] = res["wer"]
            log.write(json.dumps({"epoch": self._epoch(), "val_loss_tf": loss, "val_free_greedy": res,
                                  "minutes": round((time.time() - t0) / 60, 2)}) + "\n")
            log.flush()
            print(f"[stage2] val loss {loss:.4f} | free-generation WER {res['wer']:.4f} "
                  f"(unseen {res.get('wer_unseen', float('nan')):.4f})", flush=True)
            model.train()
            return loss, metrics, gt, preds

        def _epoch(self):
            self.n_evals = getattr(self, "n_evals", 0) + 1
            return self.n_evals * args.eval_every

    trainer = Stage2Trainer(config, dataset=dataset, metric_fns={}, extra_model_kwargs={"features": "all"},
                            eval_mode=bool(args.eval_only))
    model = trainer.accelerator.unwrap_model(trainer.model)
    if args.runtime == "optimized":
        from bit_experiments.exact_data import exact_pad_collate_fn as pad_collate_fn
    collate = partial(pad_collate_fn, model_inputs=get_model_inputs(model), **config.method.dataloader_kwargs)
    splits = {s: SpikingDatasetForDecoding(dataset, s, **config.method.dataset_kwargs) for s in ("val", "test")}
    refs = {s: [r["sentence"] for r in dataset[s]] for s in splits}
    seen = {s: stage2.seen_mask(refs[s], train_sentences) for s in splits}
    val_ds, val_refs, val_seen = splits["val"], refs["val"], seen["val"]
    run = {"args": vars(args), "init_report": trainer.init_report, "padding": trainer.padding_report,
           "runtime": trainer.runtime_report,
           "n": {s: len(v) for s, v in dataset.items()},
           "n_seen": {s: int(m.sum()) for s, m in seen.items()}, "config": json.loads(json.dumps(config, default=str)),
           "torch": torch.__version__}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    print(json.dumps({"init": run["init_report"].get("init"), "n": run["n"],
                      "encoder_loaded": run["init_report"].get("encoder", {}).get("loaded"),
                      "read_in_loaded": run["init_report"].get("read_in_loaded"),
                      "projector_loaded": "projector" in run["init_report"]}), flush=True)

    ckpt = Path(args.eval_only) if args.eval_only else Path(config.dirs.checkpoint_dir) / args.name / "BEST"
    if not args.eval_only and not args.no_train:
        trainer.train()
    if not args.no_train:
        run["eval_checkpoint"] = {"dir": str(ckpt), "load": stage2.load_model_state(model, ckpt)}
        if args.ablation:
            from bit_experiments import neural_ablations

            run["eval_checkpoint"]["ablation"] = neural_ablations.verify(model)
        (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    device = trainer.accelerator.device

    summary = {"name": args.name, "arm": args.arm, "seed": args.seed, "llm": llm, "checkpoint": str(ckpt),
               "results": {}}
    has_prompt = bool((np.asarray(dataset["test"][0]["labels"]) == -100).any())  # "#" alone: nothing to condition on
    jobs = [(s, d, "real") for s in ("val", "test") for d in stage2.DECODINGS]
    jobs += [("test", d, c) for d in args.null_decodings for c in stage2.CONDITIONS[1:]
             if has_prompt or c != "null_prompt_only"]
    for split, decoding, condition in jobs:
        hyps = stage2.generate(model, splits[split], collate, tokenizer, decoding, condition, seed=args.seed,
                               device=device)
        key = f"{split}_{decoding}_{condition}"
        summary["results"][key] = stage2.score(refs[split], hyps, seen[split])
        meta = {"model": f"stage2 {args.name}", "arm": args.arm, "seed": args.seed, "split": f"{DATASET}:{split}",
                "decoding": decoding, "condition": condition, "init": str(init), "checkpoint": str(ckpt),
                "label_fraction": args.label_fraction, "llm": llm, "prompt": config.data.prompt}
        if args.ablation:
            meta["ablation"] = args.ablation
        stage2.write_decodes(out / "decodes" / f"{key}.csv", refs[split], hyps, seen[split], meta)
        print(f"[stage2] {key}: {json.dumps(summary['results'][key])}", flush=True)
    if not args.eval_only and not args.no_train:  # keep BEST without the frozen base LLM; LAST is a full duplicate
        summary["slim"] = stage2.slim_checkpoint(model, ckpt)
        shutil.rmtree(ckpt.parent / "LAST", ignore_errors=True)
    summary["minutes"] = round((time.time() - t0) / 60, 2)
    if torch.cuda.is_available():
        summary["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
