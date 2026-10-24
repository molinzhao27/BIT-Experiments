#!/usr/bin/env python
"""Probes, RSA and foil contrasts for a list of runs (DESIGN.md §5; docs/PROBES_FOILS.md). One host job, in BIT's
environment, under the GPU lock.

    <bit env>/python scripts/run_probes_foils.py --runs input random ssl A_ctc C_stage1 B_ssl_s42 \
        B_ssl_fast-dllm-v2-1.5b_s42 [--parts probe rsa foils] [--config configs/probes.yaml] [--device auto]
        [--limit N] [--out results/probes]
    <bit env>/python scripts/run_probes_foils.py --summarize          # results/probes/SUMMARY.md from what exists

Runs: a `sources` key of configs/probes.yaml (input, random, ssl, A_ctc, C_stage1), or else a Stage 2 run name.
A Stage 2 run uses checkpoints/stage2/<name>/BEST and results/stage2/<name>/run.json (its LLM, prompt and ablation).

Representations (frozen, eval mode, Stage 2's input pipeline, bf16 autocast on CUDA):
- `encoder`: encoder output (384-d);
- `projector`: projector output (LLM embedding width), for Stage 1 / Stage 2;
- `ablated`: after the run's ablation module, for an ablation run;
- `posterior`: arm A's 41 log-posteriors;
- `input`: the patched input.

Parts, per run:
- `probe`: a linear frame-level phoneme probe per representation, trained on train frames (seeded subsample),
  L2 chosen on val, scored on test (probes.score_probe), with the majority-class baseline.
- `rsa`: BIT App. L.1 RSA per representation against the cached LLM sentence embeddings (emb, layer14),
  plain and partial (phonetic; phonetic + length + duration), BIT's 120-200 patch selection and all trials, on test.
- `foils`: the forced-choice contrasts (bit_experiments.foils). Stage 2 runs score `loglik` (AR) or `denoise`
  (block diffusion) and `aligner` (BIT's contrastive head); C_stage1 scores `stage1` (its own head against the
  fixed layer-14 target, recomputed for the foils with Aero's LM). Conditions: real, null_shuffled (the derangement of
  the run's own null decodes), null_prompt_only.

Outputs: results/probes/<run>/{probe.json, rsa.json, foil_scores.csv, foils.json, timing.json}. Every part logs its
wall time (timing.json), which is the GPU time it held the lock for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

from bit_experiments import foils as fo  # noqa: E402
from bit_experiments import neural_ablations as na  # noqa: E402
from bit_experiments import phoneme_frames as pf  # noqa: E402
from bit_experiments import probes, stage1, stage2, text_targets  # noqa: E402

PAD = stage2.PAD_VALUE
READ_IN0 = "embedder.read_in.embed_dict.0."  # the brandman_2024_text (T15) read-in slot


def say(msg: str) -> None:
    print(f"[probes_foils] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def load_all(module: torch.nn.Module, path: Path, allow_missing=(), allow_unexpected=False) -> dict:
    """Load a state file into `module`; every tensor of the module must come from it (except allow_missing prefixes)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    rep = stage2._shape_load(module, state)
    missing = [k for k in rep["not_in_checkpoint"] if not k.startswith(tuple(allow_missing))]
    if missing or rep["skipped_shape"] or (rep["unexpected"] and not allow_unexpected):
        raise RuntimeError(f"{path}: missing {missing[:5]}, wrong shape {rep['skipped_shape'][:5]}, "
                           f"unexpected {rep['unexpected'][:5]}")
    return rep


def projector_from(config, path: Path) -> torch.nn.Module:
    """BIT's projector with the widths of the saved one (projector.bin)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    first = state.get("0.weight", state.get("weight"))
    last = state.get("2.weight", state.get("weight"))
    proj = stage1.build_projector(config, first.shape[1], last.shape[0])
    load_all(proj, path)
    return proj.eval()


# ---------------------------------------------------------------- sources

class Source(types.SimpleNamespace):
    """name, kind, reps (names), encode(batch) -> ({rep: [B, P, d]}, mask [B, P]), modules (to move), info."""

    def to(self, device):
        for m in self.modules:
            m.to(device)
        return self


def make_source(name: str, cfg: dict, config, seed: int) -> Source:
    spec = cfg["sources"].get(name)
    kind = spec["kind"] if spec else "stage2"
    info = {"kind": kind}
    if kind == "input":
        enc = pf.build_ndt_encoder(config)
        emb = enc.embedder

        def encode(b):
            x = enc.smooth_and_noise(b["spikes"])
            x, m, ts, _ = emb.pad_for_patching(x, b["spikes_mask"], b["spikes_timestamp"])
            x, m, _, _ = emb.patch_data(x, m, ts)
            return {"input": x.masked_fill(x == PAD, 0)}, m

        return Source(name=name, kind=kind, reps=["input"], encode=encode, modules=[enc], info=info)
    if kind == "random":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            enc = pf.build_ndt_encoder(config)
        info["seed"] = seed
        return Source(name=name, kind=kind, reps=["encoder"], encode=_plain(enc), modules=[enc], info=info)
    if kind == "ssl":
        enc = pf.build_ndt_encoder(config, features="tx1")
        path = na.resolve(spec["dir"]) / "encoder.bin"
        state = torch.load(path, map_location="cpu", weights_only=True)
        rep = stage2._shape_load(enc, state)
        missing = [k for k in rep["not_in_checkpoint"] if not k.startswith("embedder.read_in.embed_dict.1.")]
        if missing or not any(k.startswith(READ_IN0) for k in state):
            raise RuntimeError(f"{path}: T15 read-in or core not loaded: {missing[:5]}")
        n_tx = enc.embedder.read_in.embed_dict["0"][1].in_features // pf.PATCH
        info |= {"dir": spec["dir"], "input": f"TX only (first {n_tx} features)"}

        def encode(b):
            h, m, *_ = enc(b["spikes"][..., :n_tx], b["spikes_mask"], b["spikes_timestamp"], b["dataset_name"],
                           force_mask=False)
            return {"encoder": h}, m

        return Source(name=name, kind=kind, reps=["encoder"], encode=encode, modules=[enc], info=info)
    if kind == "ctc":
        d = na.resolve(spec["dir"])
        enc = pf.build_ndt_encoder(config)
        load_all(enc, d / "encoder.bin")
        readout = na.PhonemeReadout.from_checkpoint(d)
        info["dir"] = spec["dir"]

        def encode(b):
            h, m, *_ = enc(b["spikes"], b["spikes_mask"], b["spikes_timestamp"], b["dataset_name"], force_mask=False)
            return {"encoder": h, "posterior": readout.log_posteriors(h)}, m

        return Source(name=name, kind=kind, reps=["encoder", "posterior"], encode=encode, modules=[enc, readout],
                      info=info)
    if kind == "stage1":
        d = na.resolve(spec["dir"])
        enc = pf.build_ndt_encoder(config)
        load_all(enc, d / "encoder.bin")
        proj = projector_from(config, d / "projector.bin")
        ckpt = torch.load(d / "align_head.pt", map_location="cpu", weights_only=False)
        c = ckpt["config"]
        width = (proj[-1] if isinstance(proj, torch.nn.Sequential) else proj).out_features
        head = stage1.AlignHead(width, width, c["head"]["proj_dim"], c["loss"], c["head"]["init_temp"],
                                c["head"]["max_scale"])
        head.load_state_dict(ckpt["state"])
        info |= {"dir": spec["dir"], "target": c["targets"]["name"], "epoch": ckpt.get("epoch")}
        return Source(name=name, kind=kind, reps=["encoder", "projector"], encode=_with_projector(enc, None, proj),
                      modules=[enc, proj, head.eval()], info=info, head=head, encoder=enc, projector=proj)
    # a Stage 2 run
    run = json.loads((ROOT / "results" / "stage2" / name / "run.json").read_text())
    ckpt = stage2_checkpoint(name, run)
    a = run["args"]
    enc = pf.build_ndt_encoder(config)
    load_all(enc, ckpt / "encoder.bin")
    module, reps = None, ["encoder", "projector"]
    if a.get("ablation"):
        module, expect, _ = na.build_module(na.entry(a["ablation"]))
        if expect is not None and na.state_digest(enc.state_dict()) != expect:
            raise RuntimeError(f"{name}: its encoder is not the one ablation {a['ablation']} was fit on")
        reps = ["encoder", "ablated", "projector"]
    proj = projector_from(config, ckpt / "projector.bin")
    info |= {"checkpoint": str(ckpt), "llm": a.get("llm") or "aero-1-audio", "prompt": a.get("prompt"),
             "ablation": a.get("ablation"), "seed": a.get("seed", 42), "arm": a.get("arm")}
    return Source(name=name, kind=kind, reps=reps, encode=_with_projector(enc, module, proj),
                  modules=[m for m in (enc, module, proj) if m is not None], info=info, run=run, checkpoint=ckpt)


def stage2_checkpoint(name: str, run: dict) -> Path:
    """checkpoints/stage2/<name>/BEST, or the checkpoint an eval-only run evaluated (e.g. B_rand_bit4a: BIT's own
    sentence BEST). An absolute path from another code copy is re-rooted at its `checkpoints/` part."""
    p = run["args"].get("eval_only")
    if not p:
        return ROOT / "checkpoints" / "stage2" / name / "BEST"
    p = Path(p) if Path(p).is_absolute() else ROOT / p
    if not p.exists() and "checkpoints" in p.parts:
        p = ROOT.joinpath(*p.parts[p.parts.index("checkpoints"):])
    return p


def _plain(enc):
    def encode(b):
        h, m, *_ = enc(b["spikes"], b["spikes_mask"], b["spikes_timestamp"], b["dataset_name"], force_mask=False)
        return {"encoder": h}, m
    return encode


def _with_projector(enc, module, proj):
    def encode(b):
        h, m, *_ = enc(b["spikes"], b["spikes_mask"], b["spikes_timestamp"], b["dataset_name"], force_mask=False)
        out = {"encoder": h}
        x = h
        if module is not None:
            x = out["ablated"] = module(h, m)
        out["projector"] = proj(x)
        return out, m
    return encode


# ---------------------------------------------------------------- probe + RSA

def extract(src: Source, ds: dict, labels: dict, cfg: dict, device, want_probe: bool, want_rsa: bool) -> dict:
    """Labelled frames per split and representation (train: seeded subsample), and the test trials' 10-segment
    pooled vectors (RSA)."""
    seed, pc, rc = cfg["seed"], cfg["probe"], cfg["rsa"]
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else None
    out = {"frames": {}, "pooled": {r: [] for r in src.reps}, "n_frames": []}
    splits = ["train", "val", "test"] if want_probe else []
    if want_rsa and rc["split"] not in splits:
        splits.append(rc["split"])
    for split in splits:
        lab = labels[split]["labels"]
        if len(lab) != len(ds[split]):
            raise ValueError(f"label cache has {len(lab)} {split} trials, the dataset {len(ds[split])}")
        n_lab = sum(int((x >= 0).sum()) for x in lab)
        p = min(1.0, pc["max_train_frames"] / max(1, n_lab)) if split == "train" else 1.0
        rng = np.random.default_rng([seed, fo.SPLIT_CODE[split]])
        xs, ys, ts = {r: [] for r in src.reps}, [], []
        for j, reps in pf.run_encoder(src.encode, ds[split], cfg["batch_size"], device, dtype):
            y = lab[j]
            if len(y) != len(next(iter(reps.values()))):
                raise ValueError(f"{split} trial {j}: frame count differs from the label cache")
            if want_probe:
                keep = (y >= 0) & (rng.random(len(y)) < p)
                for r, h in reps.items():
                    xs[r].append(h[keep])
                ys.append(y[keep] - 1)
                ts.append(np.full(int(keep.sum()), j))
            if want_rsa and split == rc["split"]:
                for r, h in reps.items():
                    out["pooled"][r].append((j, probes.segment_pool(h, rc["n_segments"]).astype(np.float32)))
                out["n_frames"].append((j, len(y)))
        if want_probe:
            out["frames"][split] = ({r: np.concatenate(v) for r, v in xs.items()}, np.concatenate(ys),
                                    np.concatenate(ts))
    for r in out["pooled"]:
        out["pooled"][r] = np.stack([v for _, v in sorted(out["pooled"][r], key=lambda x: x[0])]) \
            if out["pooled"][r] else None
    out["n_frames"] = np.array([n for _, n in sorted(out["n_frames"])])
    return out


def run_probe(src: Source, ex: dict, cfg: dict, device) -> dict:
    pc = cfg["probe"]
    (xtr, ytr, _), (xva, yva, _), (xte, yte, tte) = (ex["frames"][s] for s in ("train", "val", "test"))
    res = {}
    for r in src.reps:
        t0 = time.time()
        probe = probes.fit_linear_probe(xtr[r], ytr, xva[r], yva, pf.N_CLASSES, pc["l2_grid"], pc["max_iter"],
                                        device=device)
        res[r] = {"dim": int(xtr[r].shape[1]), "test": probes.score_probe(probe, xte[r], yte, tte, pc["n_boot"],
                                                                         cfg["seed"]),
                  "val": probes.score_probe(probe, xva[r], yva, None, 0), "n_train_frames": int(len(ytr)),
                  "seconds": round(time.time() - t0, 1)} | probes.strip(probe)
        say(f"probe {src.name}/{r}: test acc {res[r]['test']['accuracy']:.3f} (majority "
            f"{res[r]['test']['majority']:.3f}), {res[r]['test']['info_bits']:.2f} bits/frame")
    return res


def rsa_models(ds: dict, labels: dict, cfg: dict, n_frames: np.ndarray) -> dict:
    """Model RDMs (text targets) and covariate RDMs for the RSA split."""
    rc = cfg["rsa"]
    split = rc["split"]
    sentences = [r["sentence"] for r in ds[split]]
    if labels[split]["sentences"] != sentences:
        raise ValueError("label cache sentences differ from the dataset's")
    rows = {}
    for t in rc["targets"]:
        row_of, mat, _ = text_targets.load(na.resolve(cfg["text_targets"]), t)
        rows[t] = probes.cosine_rdm(mat[[row_of[s] for s in sentences]])
    ph = labels[split]["phonemes"]
    cov = {"phonemes": probes.sequence_rdm([list(p) for p in ph]),
           "n_phonemes": probes.abs_diff_rdm([len(p) for p in ph]), "n_frames": probes.abs_diff_rdm(n_frames)}
    return {"targets": rows, "covariates": cov}


def run_rsa(src: Source, ex: dict, models: dict, cfg: dict) -> dict:
    rc = cfg["rsa"]
    n_frames = ex["n_frames"]
    selections = {}
    if rc.get("patch_range"):
        lo, hi = rc["patch_range"]
        selections[f"patches_{lo}_{hi}"] = np.flatnonzero((n_frames >= lo) & (n_frames <= hi))
    if rc.get("also_all_trials", True) or not selections:
        selections["all"] = np.arange(len(n_frames))
    res = {}
    for r in src.reps:
        neural_full = probes.cosine_rdm(ex["pooled"][r])
        for sel_name, idx in selections.items():
            if len(idx) < 4:
                res[f"{r}/{sel_name}"] = {"n_trials": int(len(idx)), "skipped": "fewer than 4 trials"}
                continue
            sub = np.ix_(idx, idx)
            neural = neural_full[sub]
            cov = {k: v[sub] for k, v in models["covariates"].items()}
            entry = {"n_trials": int(len(idx)),
                     "phonetic": probes.rsa(neural, cov["phonemes"], (), rc["n_perm"], rc["n_boot"], cfg["seed"])}
            for t, model_rdm in models["targets"].items():
                m = model_rdm[sub]
                entry[t] = {"plain": probes.rsa(neural, m, (), rc["n_perm"], rc["n_boot"], cfg["seed"]),
                            "target_vs_phonetic": probes.rsa(m, cov["phonemes"], (), 0, 0)["r"]}
                for cname, keys in rc["covariates"].items():
                    entry[t][f"partial_{cname}"] = probes.rsa(neural, m, [cov[k] for k in keys], rc["n_perm"],
                                                              rc["n_boot"], cfg["seed"])
            res[f"{r}/{sel_name}"] = entry
            say(f"rsa {src.name}/{r}/{sel_name}: " + ", ".join(
                f"{t} r={entry[t]['plain']['r']:.3f} partial={entry[t]['partial_phonetic']['r']:.3f}"
                for t in models["targets"]))
    return res


# ---------------------------------------------------------------- foils

def stage2_model(src: Source, config, device, seed: int):
    """The run's full Stage 2 model as train_stage2.py builds it, its BEST loaded (and its ablation re-applied)."""
    from transformers import AutoTokenizer

    from bit_experiments import text_llm

    a = src.run["args"]
    if a.get("llm"):
        text_llm.use_llm(config, a["llm"])
    if a.get("prompt") is not None:
        config["data"]["prompt"] = a["prompt"]
    if config.model.model_class == "BCI_AudioLLM":
        from models.bci import BCI_AudioLLM as cls
    else:
        from models.trainer import NAME2MODEL

        cls = NAME2MODEL[config.model.model_class]
    with stage1.in_bit_dir(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = cls(config.model, **config.method.model_kwargs, **config.method.dataset_kwargs, features="all")
    stage2.fix_padding(model)
    if a.get("ablation"):
        na.apply(model, a["ablation"], None, seed)
    load = stage2.load_model_state(model, src.checkpoint)
    if a.get("ablation"):
        load["ablation"] = na.verify(model)
    tok = AutoTokenizer.from_pretrained(config.data.tokenizer_path, add_bos_token=False, add_eos_token=False)
    diffusion = hasattr(model, "denoising_ce")
    return model.to(device).eval(), tok, config.data.prompt, diffusion, load


def aero_lm(config, device):
    from transformers import AutoTokenizer, Qwen2AudioForConditionalGeneration

    path = config.method.model_kwargs.llm_path
    tok = AutoTokenizer.from_pretrained(path, add_bos_token=False, add_eos_token=False)
    tok.padding_side = "right"
    lm = Qwen2AudioForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32).language_model
    return lm.to(device).eval(), tok


def check_references(df: pd.DataFrame, ds: dict) -> None:
    for split, g in df.groupby("split"):
        sents = [r["sentence"] for r in ds[split]]
        bad = [(t, r) for t, r in zip(g.trial, g.reference) if t >= len(sents) or sents[t] != r]
        if bad:
            raise ValueError(f"foil set does not match BIT's {split} sentences, e.g. trial {bad[0][0]}: {bad[0][1]!r}")


def score_foils(src: Source, ds: dict, foil_df: pd.DataFrame, cfg: dict, config, device) -> pd.DataFrame:
    fc = cfg["foils"]
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else None
    seed = src.info.get("seed", 42)
    rows = []
    if src.kind == "stage2":
        model, tok, prompt, diffusion, load = stage2_model(src, config, device, seed)
        src.info["foil_model"] = {"diffusion": diffusion, "load": load}
        scorers = ["denoise" if diffusion else "loglik", "aligner"]
    else:  # C_stage1: its own head against recomputed fixed targets
        lm, ltok = aero_lm(config, device)
        sentences = sorted(set(foil_df.reference) | set(foil_df.foil))
        with torch.no_grad():
            tgt = text_targets.compute_targets(lm, ltok, sentences, device=str(device))[src.info["target"]]
        row_of, cached, _ = text_targets.load(na.resolve(cfg["text_targets"]), src.info["target"])
        refs = sorted(set(foil_df.reference))
        cos = (tgt[[sentences.index(s) for s in refs]] * cached[[row_of[s] for s in refs]]).sum(1)
        if cos.min() < 0.999:
            raise RuntimeError(f"recomputed {src.info['target']} targets differ from the cache (min cos {cos.min():.4f})")
        src.info["target_check_min_cos"] = float(cos.min())
        target = {s: torch.as_tensor(tgt[i], device=device) for i, s in enumerate(sentences)}
        del lm
        scorers = ["stage1"]
    for split in (fc.get("diffusion_splits", fc["splits"]) if src.kind == "stage2" and diffusion else fc["splits"]):
        g = foil_df[foil_df.split == split]
        recs = ds[split]
        perm = stage2.derangement(len(recs), np.random.default_rng(seed))  # stage2.generate's null_shuffled pairing
        t0 = time.time()
        for trial, gt in g.groupby("trial"):
            texts = [gt.reference.iloc[0]] + gt.foil.tolist()
            for cond in fc["conditions"]:
                if cond == "null_prompt_only" and (src.kind != "stage2" or prompt.strip() == "#"):
                    continue
                spikes = recs[int(perm[trial])]["spikes"] if cond == "null_shuffled" else recs[trial]["spikes"]
                scores = {}
                if src.kind == "stage2":
                    po = cond == "null_prompt_only"
                    if diffusion:
                        s, se = fo.diffusion_denoise(model, spikes, texts, tok, prompt, fc["diffusion_draws"],
                                                     seed * 1_000_003 + fo.SPLIT_CODE[split] * 100_003 + trial,
                                                     device, dtype, prompt_only=po)
                        scores["denoise"] = (s, se)
                    else:
                        scores["loglik"] = (fo.ar_loglik(model, spikes, texts, tok, prompt, device, dtype, po), None)
                    if not po:
                        scores["aligner"] = (fo.aligner_cosine(model, spikes, texts, tok, prompt, device, dtype), None)
                else:
                    scores["stage1"] = (stage1_cosine(src, spikes, texts, target, device, dtype), None)
                for scorer, (s, se) in scores.items():
                    for i, r in enumerate(gt.itertuples()):
                        rows.append({"split": split, "trial": trial, "type": r.type, "variant": r.variant,
                                     "foil": r.foil, "phon_dist": r.phon_dist, "condition": cond, "scorer": scorer,
                                     "score_true": float(s[0]), "score_foil": float(s[i + 1]),
                                     "se_true": None if se is None else float(se[0]),
                                     "se_foil": None if se is None else float(se[i + 1])})
        say(f"foils {src.name}/{split}: {g.trial.nunique()} trials, {len(g)} foils, {time.time() - t0:.0f} s")
    if src.kind == "stage2":
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


@torch.no_grad()
def stage1_cosine(src: Source, spikes, texts, target: dict, device, dtype) -> np.ndarray:
    nin = {k: v.to(device) for k, v in fo.neural_inputs(spikes, 1).items()}
    with torch.autocast(torch.device(device).type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
        h, m, *_ = src.encoder(nin["spikes"], nin["spikes_mask"], nin["spikes_timestamp"], nin["dataset_name"],
                               force_mask=False)
        zn = src.head.embed_neural(src.projector(h), m)
    zt = src.head.embed_text(torch.stack([target[t] for t in texts]))
    return (zn @ zt.T)[0].float().cpu().numpy()


def analyse_foils(scores: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    for scorer, g in scores.groupby("scorer"):
        for name, sub in (("test", g[g.split == "test"]), ("val+test", g)):
            if sub.empty:
                continue
            key = sub.assign(trial=sub.split.map(fo.SPLIT_CODE) * 100_000 + sub.trial)  # trials unique across splits
            res = {"forced_choice": fo.forced_choice(key, cfg["foils"]["n_boot"], cfg["seed"])}
            for null in [c for c in key.condition.unique() if c != "real"]:
                try:
                    res[f"type_effects_{null}"] = fo.type_effects(key, null, n_boot=cfg["foils"]["n_boot"],
                                                                  seed=cfg["seed"])
                except (KeyError, np.linalg.LinAlgError, ValueError) as err:
                    res[f"type_effects_{null}"] = {"error": str(err)}
            out[f"{scorer}/{name}"] = res
    return out


# ---------------------------------------------------------------- summary

def summarize(out_dir: Path) -> Path:
    lines = ["# Probes, RSA and foils: summary", "", "Generated by `scripts/run_probes_foils.py --summarize`. Test "
             "split. Probe: linear frame-level phoneme probe (40 classes), accuracy [95% CI] and information "
             "(bits/frame). RSA: Pearson r of 1 - cosine RDMs vs the LLM layer-14 sentence embeddings, plain / "
             "partial (phonetic regressed out), BIT's 120-200 patch trials.", "",
             "| run | representation | probe acc | majority | bits/frame | RSA layer14 | partial | phonetic RSA |",
             "|---|---|---|---|---|---|---|---|"]
    for d in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        pr = json.loads((d / "probe.json").read_text()) if (d / "probe.json").exists() else {}
        rs = json.loads((d / "rsa.json").read_text()) if (d / "rsa.json").exists() else {}
        for r in sorted((set(pr) | {k.split("/")[0] for k in rs}) - {"source"}):
            p = pr.get(r, {}).get("test", {})
            sel = next((v for k, v in rs.items() if k.startswith(r + "/patches") and "skipped" not in v), None) \
                or rs.get(f"{r}/all", {})
            l14 = sel.get("layer14", {})
            fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
            ci = p.get("accuracy_ci")
            lines.append(f"| {d.name} | {r} | {fmt(p.get('accuracy'))}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                         + f" | {fmt(p.get('majority'))} | {fmt(p.get('info_bits'))} | "
                         f"{fmt(l14.get('plain', {}).get('r'))} | {fmt(l14.get('partial_phonetic', {}).get('r'))} | "
                         f"{fmt(sel.get('phonetic', {}).get('r'))} |")
    lines += ["", "## Foils (test): accuracy real / shuffled / prompt-only, neural evidence (real - shuffled)", "",
              "| run | scorer | type | n | acc real | acc shuffled | acc prompt-only | real - shuffled [95% CI] |",
              "|---|---|---|---|---|---|---|---|"]
    for d in sorted(p for p in out_dir.iterdir() if p.is_dir() and (p / "foils.json").exists()):
        fj = json.loads((d / "foils.json").read_text())
        for key, res in fj["results"].items():
            scorer, split = key.split("/", 1)
            if split != "test":
                continue
            for t, v in res["forced_choice"].items():
                if v["level"] != "type":
                    continue
                ci = v.get("acc_real_minus_null_shuffled_ci")
                lines.append(f"| {d.name} | {scorer} | {t} | {v['n']} | {v.get('acc_real', float('nan')):.3f} | "
                             f"{v.get('acc_null_shuffled', float('nan')):.3f} | "
                             f"{v.get('acc_null_prompt_only', float('nan')):.3f} | "
                             f"{v.get('acc_real_minus_null_shuffled', float('nan')):+.3f}"
                             + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "") + " |")
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--parts", nargs="*", default=["probe", "rsa", "foils"], choices=["probe", "rsa", "foils"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "probes.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "probes")
    ap.add_argument("--summarize", action="store_true", help="only (re)write <out>/SUMMARY.md")
    args = ap.parse_args()
    if args.summarize:
        say(f"wrote {summarize(args.out)}")
        return 0
    if not args.runs:
        ap.error("--runs is required (or --summarize)")
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    t0 = time.time()
    config = stage1.bit_sentence_config()
    ds = pf.load_t15(config, args.limit)
    labels = pf.load_labels(na.resolve(cfg["labels"]))[0] if {"probe", "rsa"} & set(args.parts) else None
    foil_df = None
    if "foils" in args.parts:
        foil_df, manifest = fo.load_foils(na.resolve(cfg["foil_set"]))
        foil_df = foil_df[foil_df.split.isin(cfg["foils"]["splits"])]
        if args.limit:
            foil_df = foil_df[foil_df.trial < args.limit]
        check_references(foil_df, ds)
    models = None
    say(f"data loaded ({time.time() - t0:.0f} s, device {device})")
    for name in args.runs:
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        timing = {"device": str(device)}
        src = make_source(name, cfg, stage1.bit_sentence_config(), cfg["seed"]).to(device)
        if {"probe", "rsa"} & set(args.parts):
            t = time.time()
            ex = extract(src, ds, labels, cfg, device, "probe" in args.parts, "rsa" in args.parts)
            timing["extract_min"] = round((time.time() - t) / 60, 2)
            if "probe" in args.parts:
                t = time.time()
                (out / "probe.json").write_text(json.dumps(jsonable({"source": src.info,
                                                                     **run_probe(src, ex, cfg, device)}), indent=1))
                timing["probe_min"] = round((time.time() - t) / 60, 2)
            if "rsa" in args.parts:
                t = time.time()
                models = models or rsa_models(ds, labels, cfg, ex["n_frames"])
                (out / "rsa.json").write_text(json.dumps(jsonable({"source": src.info,
                                                                   **run_rsa(src, ex, models, cfg)}), indent=1))
                timing["rsa_min"] = round((time.time() - t) / 60, 2)
            del ex
        if "foils" in args.parts and src.kind in ("stage2", "stage1"):
            t = time.time()
            scores = score_foils(src, ds, foil_df, cfg, stage1.bit_sentence_config(), device)
            scores.to_csv(out / "foil_scores.csv", index=False)
            (out / "foils.json").write_text(json.dumps(jsonable({"source": src.info, "foil_set": manifest["dir"],
                                                                 "results": analyse_foils(scores, cfg)}), indent=1))
            timing["foils_min"] = round((time.time() - t) / 60, 2)
        (out / "timing.json").write_text(json.dumps(timing, indent=1))
        say(f"{name} done: {timing}")
    summarize(args.out)
    say(f"all done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
