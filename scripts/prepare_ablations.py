#!/usr/bin/env python
"""Artifacts for the retraining ablations (docs/RETRAINING_ABLATIONS.md). BIT's environment, on the GPU (bf16
autocast, as in Stage 2; about 15 min for all steps) or CPU (float32, for smoke tests only: the artifacts must be
computed the way Stage 2 computes its features).

    <bit env>/python scripts/prepare_ablations.py labels       # frame phoneme labels, all splits (+ arm A's PER)
    <bit env>/python scripts/prepare_ablations.py bottleneck   # per-class statistics of arm A's log-posteriors, train
    <bit env>/python scripts/prepare_ablations.py export-encoder [--run B_ssl_s42] [--best DIR] [--erasure ENTRY]
    <bit env>/python scripts/prepare_ablations.py erasure      # LEACE on train frames + held-out check (exit 4 = fail)
    <bit env>/python scripts/prepare_ablations.py all          # the four in order
    common: [--device auto|cuda|cpu] [--limit N] [--batch-size 16] [--seed 0]

Outputs:
- labels: data/processed/phoneme_frames/<sha12>.npz + .json (and LATEST). Per split, per trial, it holds the frame
  labels (vocab id or -1), the frame's index in the label sequence, and the label sequence (BIT's g2p_en
  phonemes + SIL). The .json reports arm A's greedy PER through Stage 2's input pipeline; exit 3 if test PER is
  more than --per-tolerance from --expect-test-per.
- bottleneck: configs/ablations.yaml phoneme-bottleneck `stats` (+ .json), tied to arm A's encoder and read-out
  digests.
- export-encoder: checkpoints/ablations/<run>-encoder/{encoder.bin, encoder_config.pth, export.json}, the run's
  BEST encoder (checked against its slim.safetensors), the --init for frozen-B and phoneme-erasure.
- erasure: the phoneme-erasure `eraser` (+ eraser_check.json next to it). The check:
  - a linear probe trained on train frames and scored on held-out val frames, before and after erasure, with its
    L2 chosen by val log-loss (the best-fitting probe on the held-out frames themselves);
  - an MLP probe (non-linear leftover);
  - probes for the neighbouring labels;
  - the in-sample cross-covariance;
  - the share of feature variance removed.

  Pass: after erasure, val accuracy within --tol-acc of the majority class and information within --tol-bits of 0.

Labels and probes never train or select a decoder (DESIGN.md §5).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BIT = ROOT / "external" / "bit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BIT))

from bit_experiments import neural_ablations as na  # noqa: E402
from bit_experiments import phoneme_frames as pf  # noqa: E402
from bit_experiments import probes, stage1, stage2  # noqa: E402

LABELS = ROOT / "data" / "processed" / "phoneme_frames"
SPLITS = ("train", "val", "test")


def say(msg: str) -> None:
    print(f"[prepare_ablations] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def frozen_encoder(config, ckpt_dir: Path, device) -> torch.nn.Module:
    """Stage 2's NDT encoder (phoneme_frames.build_ndt_encoder), every tensor loaded from ckpt_dir/encoder.bin."""
    enc = pf.build_ndt_encoder(config)
    stage2._shape_load(enc, torch.load(Path(ckpt_dir) / "encoder.bin", map_location="cpu", weights_only=True))
    na._check_from_init(enc, Path(ckpt_dir))
    return enc.to(device).eval()


def encoder_fn(enc, head=None):
    def encode(b):
        h, m, *_ = enc(b["spikes"], b["spikes_mask"], b["spikes_timestamp"], b["dataset_name"], force_mask=False)
        return (h if head is None else head(h, m)), m
    return encode


def autocast_dtype(device) -> torch.dtype | None:
    return torch.bfloat16 if torch.device(device).type == "cuda" else None


# ---------------------------------------------------------------- steps

def make_labels(config, ds, phoneme_model: Path, out: Path, device, batch_size=16, margin=2, sequences=None,
                expect_test_per=None, per_tolerance=0.015) -> tuple[Path, dict]:
    enc = frozen_encoder(config, phoneme_model, device)
    readout = na.PhonemeReadout.from_checkpoint(phoneme_model).to(device)
    lp_fn = encoder_fn(enc, lambda h, m: readout.log_posteriors(h))
    sequences = sequences or (lambda s: pf.bit_phoneme_sequences(s, BIT / "vocab.json"))
    per_split, report = {}, {}
    for split in [s for s in SPLITS if s in ds]:
        recs = ds[split]
        sentences = [r["sentence"] for r in recs]
        seqs = sequences(sentences)
        lab, tok, hyp, hyp_bit, ok = ([None] * len(recs) for _ in range(5))
        for j, lp in pf.run_encoder(lp_fn, recs, batch_size, device, autocast_dtype(device)):
            lab[j], tok[j], ok[j] = pf.align(lp, seqs[j], margin)
            hyp[j], hyp_bit[j] = pf.greedy_ctc(lp), pf.greedy_ctc(lp, bit=True)
        flat = np.concatenate(lab)
        per_split[split] = {"labels": lab, "tokens": tok, "phonemes": [np.asarray(s) for s in seqs],
                            "sentences": sentences}
        report[split] = {"n_trials": len(recs), "n_unaligned": int(len(recs) - sum(ok)), "n_frames": int(len(flat)),
                         "labelled": float((flat >= 0).mean()),
                         "frames_per_label": float((flat >= 0).sum() / max(1, sum(map(len, seqs)))),
                         "per": pf.per(hyp, seqs), "per_bit_collapse": pf.per(hyp_bit, seqs),
                         "class_counts": np.bincount(flat[flat >= 0], minlength=pf.N_VOCAB).tolist()}
        say(f"labels {split}: {json.dumps({k: v for k, v in report[split].items() if k != 'class_counts'})}")
    meta = {"phoneme_model": str(phoneme_model), "encoder_digest": na.state_digest(enc.state_dict()),
            "readout_digest": na.state_digest({"weight": readout.weight, "bias": readout.bias}), "margin": margin,
            "vocab": "external/bit/vocab.json", "sequences": "BIT create_phonemes_ctc_labels (g2p_en)",
            "device": str(device), "autocast": str(autocast_dtype(device)), "report": report}
    if expect_test_per is not None and "test" in report:
        meta["per_check"] = {"expected": expect_test_per, "tolerance": per_tolerance,
                             "ok": abs(report["test"]["per_bit_collapse"] - expect_test_per) <= per_tolerance}
    return pf.save_labels(out, per_split, meta), meta


def make_bottleneck_stats(config, ds, e: dict, device, batch_size=16) -> dict:
    model_dir = na.resolve(e["phoneme_model"])
    enc = frozen_encoder(config, model_dir, device)
    readout = na.PhonemeReadout.from_checkpoint(model_dir, feature=e["feature"], floor=e["floor"]).to(device)
    s1, s2, n = torch.zeros(na.N_VOCAB, dtype=torch.float64), torch.zeros(na.N_VOCAB, dtype=torch.float64), 0
    for _, x in pf.run_encoder(encoder_fn(enc, lambda h, m: readout.raw(h)), ds["train"], batch_size, device,
                               autocast_dtype(device)):
        x = torch.from_numpy(x).double()
        s1, s2, n = s1 + x.sum(0), s2 + (x ** 2).sum(0), n + len(x)
    mean = s1 / n
    sd = (s2 / n - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-6)
    stats = {"feature": e["feature"], "floor": e["floor"], "n_frames": int(n), "split": "train",
             "mean": None if e["feature"] == "one_hot" else mean.float(),
             "sd": None if e["feature"] == "one_hot" else sd.float(),
             "encoder_digest": na.state_digest(enc.state_dict()),
             "readout_digest": na.state_digest({"weight": readout.weight, "bias": readout.bias}),
             "phoneme_model": e["phoneme_model"], "device": str(device), "autocast": str(autocast_dtype(device))}
    path = na.resolve(e["stats"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    summary = {k: v for k, v in stats.items() if k not in ("mean", "sd")} | {
        "mean": None if stats["mean"] is None else mean.tolist(), "sd": None if stats["sd"] is None else sd.tolist()}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    say(f"bottleneck stats: {n} train frames -> {path}")
    return summary


def export_encoder(config, run: str, out: Path, best: Path | None = None) -> dict:
    """The run's BEST encoder.bin (checked against its slim.safetensors) -> out. best: the BEST directory if it is not
    checkpoints/stage2/<run>/BEST of this copy (e.g. a run trained from another code copy)."""
    src = Path(best) if best else ROOT / "checkpoints" / "stage2" / run / "BEST"
    state = torch.load(src / "encoder.bin", map_location="cpu", weights_only=True)
    slim = src / stage2.SLIM
    checked = None
    if slim.exists():
        from safetensors.torch import load_file

        full = load_file(str(slim))
        own = {k[len("neural_encoder.encoder."):]: v for k, v in full.items() if k.startswith("neural_encoder.encoder.")}
        if set(own) != set(state) or any(not torch.equal(own[k], state[k].to(own[k].dtype)) for k in own):
            raise RuntimeError(f"{src}: encoder.bin differs from slim.safetensors' neural_encoder.encoder")
        checked = stage2.SLIM
    enc = frozen_encoder(config, src, "cpu")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "encoder.bin", out / "encoder.bin")
    if (src / "encoder_config.pth").exists():
        shutil.copy2(src / "encoder_config.pth", out / "encoder_config.pth")
    info = {"run": run, "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
            "encoder_bin_sha256": na.file_digest(out / "encoder.bin"), "encoder_digest": na.state_digest(enc.state_dict()),
            "checked_against": checked, "note": "encoder only: --init with this directory loads no projector"}
    (out / "export.json").write_text(json.dumps(info, indent=1))
    say(f"exported {src} -> {out} (digest {info['encoder_digest'][:12]})")
    return info


def _frames(enc, recs, lab_split, device, batch_size, keep_prob, rng, offsets=(0,), acc=None, n_classes=40):
    """Labelled frames of one split: features, own class, neighbour classes (-1, +1), trial ids; optionally a
    seeded subsample (keep_prob) and the LEACE sums of every labelled frame (acc)."""
    xs, ys, nbs, ts = [], [], [], []
    for j, h in pf.run_encoder(encoder_fn(enc), recs, batch_size, device, autocast_dtype(device)):
        lab, tok, ph = lab_split["labels"][j], lab_split["tokens"][j], lab_split["phonemes"][j]
        if len(lab) != len(h):
            raise ValueError(f"trial {j}: {len(h)} frames, {len(lab)} labels (labels from another dataset?)")
        sel = lab >= 0
        if acc is not None and sel.any():
            acc.update(torch.from_numpy(h[sel]), na.one_hot_concept(torch.from_numpy(pf.concept(lab, tok, ph, offsets)
                                                                                     [sel]), n_classes))
        keep = sel & (rng.random(len(lab)) < keep_prob)
        xs.append(h[keep])
        ys.append(lab[keep] - 1)
        nbs.append(pf.concept(lab, tok, ph, (-1, 1))[keep])
        ts.append(np.full(int(keep.sum()), j))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(nbs), np.concatenate(ts)


def make_eraser(config, ds, labels: dict, labels_meta: dict, e: dict, device, batch_size=16, seed=0,
                max_train_frames=200_000, max_val_frames=100_000, tol_acc=0.02, tol_bits=0.02,
                probe_device=None) -> tuple[dict, bool]:
    enc_dir = na.resolve(e["encoder"])
    enc = frozen_encoder(config, enc_dir, device)
    digest = na.state_digest(enc.state_dict())
    export = enc_dir / "export.json"
    if export.exists() and json.loads(export.read_text())["encoder_digest"] != digest:
        raise RuntimeError(f"{enc_dir}: encoder digest differs from export.json")
    offsets = list(e.get("offsets", [0]))
    n_lab = {s: sum(int((x >= 0).sum()) for x in labels[s]["labels"]) for s in ("train", "val")}
    rng = np.random.default_rng(seed)
    acc = na.CovAccumulator(enc.hidden_size, pf.N_CLASSES * len(offsets))
    say(f"erasure: train frames ({n_lab['train']} labelled), offsets {offsets}")
    xtr, ytr, nbtr, ttr = _frames(enc, ds["train"], labels["train"], device, batch_size,
                                  min(1, max_train_frames / max(1, n_lab["train"])), rng, offsets, acc)
    fit = na.leace(*acc.moments())
    A, b = fit["A"].float(), fit["b"].float()
    eraser = {"A": A, "b": b, "rank": fit["rank"], "cov_rank": fit["cov_rank"], "offsets": offsets,
              "n_classes": pf.N_CLASSES, "n_frames": acc.n, "encoder": e["encoder"], "encoder_digest": digest,
              "labels_sha256": labels_meta["sha256"], "mean": acc.moments()[0].float(), "device": str(device),
              "autocast": str(autocast_dtype(device)), "seed": seed}
    path = na.resolve(e["eraser"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(eraser, path)
    say(f"LEACE: rank {fit['rank']} of {enc.hidden_size} (Cov rank {fit['cov_rank']}), {acc.n} frames -> {path}")

    xva, yva, nbva, tva = _frames(enc, ds["val"], labels["val"], device, batch_size,
                                  min(1, max_val_frames / max(1, n_lab["val"])), rng)
    erase = lambda x: x @ A.numpy().T.astype(np.float64) + b.numpy()  # noqa: E731
    xtr_e, xva_e = erase(xtr), erase(xva)
    pdev = probe_device or device
    check = {"n_train_frames_probe": int(len(ytr)), "n_val_frames": int(len(yva))}
    for name, (a, v) in {"before": (xtr, xva), "after": (xtr_e, xva_e)}.items():
        lin = probes.fit_linear_probe(a, ytr, v, yva, pf.N_CLASSES, device=pdev)
        mlp = probes.fit_mlp_probe(a, ytr, v, yva, pf.N_CLASSES, seed=seed, device=pdev)
        check[name] = {"linear": probes.score_probe(lin, v, yva, tva, seed=seed) | probes.strip(lin),
                       "mlp": probes.score_probe(mlp, v, yva, tva, seed=seed) | probes.strip(mlp)}
    for c, o in enumerate((-1, 1)):
        ok_tr, ok_va = nbtr[:, c] >= 0, nbva[:, c] >= 0
        lin = probes.fit_linear_probe(xtr_e[ok_tr], nbtr[ok_tr, c], xva_e[ok_va], nbva[ok_va, c], pf.N_CLASSES,
                                      device=pdev)
        check["after"][f"linear_label{o:+d}"] = probes.score_probe(lin, xva_e[ok_va], nbva[ok_va, c], tva[ok_va],
                                                                    seed=seed) | probes.strip(lin)
    z = na.one_hot_concept(torch.from_numpy(ytr[:, None]), pf.N_CLASSES).numpy()
    cov = lambda x: np.abs((x - x.mean(0)).T @ (z - z.mean(0)) / len(x)).max()  # noqa: E731
    check["max_abs_cross_cov"] = {"before": float(cov(xtr)), "after": float(cov(xtr_e))}
    check["variance_removed_val"] = float(((xva_e - xva) ** 2).sum(1).mean() / ((xva - xva.mean(0)) ** 2).sum(1).mean())
    after = check["after"]["linear"]
    check["pass"] = bool(after["accuracy"] <= after["majority"] + tol_acc and after["info_bits"] <= tol_bits)
    check["criteria"] = {"tol_acc": tol_acc, "tol_bits": tol_bits, "probe": "linear, val (held out)"}
    path.with_name("eraser_check.json").write_text(json.dumps(check, indent=1))
    say(f"erasure check: linear val acc {check['before']['linear']['accuracy']:.3f} -> {after['accuracy']:.3f} "
        f"(majority {after['majority']:.3f}), MLP {check['before']['mlp']['accuracy']:.3f} -> "
        f"{check['after']['mlp']['accuracy']:.3f}; pass={check['pass']}")
    return check, check["pass"]


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["labels", "bottleneck", "export-encoder", "erasure", "all"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phoneme-model", default="checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST")
    ap.add_argument("--margin", type=int, default=2, help="blank frames before / after the labels that still count")
    ap.add_argument("--expect-test-per", type=float, default=0.0925, help="arm A's test PER (dashboard, 2026-10-20)")
    ap.add_argument("--per-tolerance", type=float, default=0.015)
    ap.add_argument("--labels", type=Path, default=LABELS, help="label cache (dir with LATEST, or a .npz)")
    ap.add_argument("--bottleneck", default="phoneme-bottleneck", help="configs/ablations.yaml entry")
    ap.add_argument("--erasure", default="phoneme-erasure", help="configs/ablations.yaml entry")
    ap.add_argument("--run", default="B_ssl_s42", help="export-encoder: the Stage 2 run")
    ap.add_argument("--best", type=Path, help="export-encoder: its BEST dir (default checkpoints/stage2/<run>/BEST)")
    ap.add_argument("--max-train-frames", type=int, default=200_000)
    ap.add_argument("--max-val-frames", type=int, default=100_000)
    ap.add_argument("--tol-acc", type=float, default=0.02)
    ap.add_argument("--tol-bits", type=float, default=0.02)
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed)
    config = stage1.bit_sentence_config()
    steps = ["labels", "bottleneck", "export-encoder", "erasure"] if args.step == "all" else [args.step]
    ds = pf.load_t15(config, args.limit) if set(steps) - {"export-encoder"} else None
    rc = 0
    for step in steps:
        if step == "labels":
            path, meta = make_labels(config, ds, na.resolve(args.phoneme_model), LABELS, device, args.batch_size,
                                     args.margin, expect_test_per=None if args.limit else args.expect_test_per,
                                     per_tolerance=args.per_tolerance)
            say(f"labels -> {path}")
            if not meta.get("per_check", {"ok": True})["ok"]:
                say(f"PER CHECK FAILED: {meta['per_check']} (test {meta['report']['test']})")
                return 3
        elif step == "bottleneck":
            make_bottleneck_stats(config, ds, na.entry(args.bottleneck), device, args.batch_size)
        elif step == "export-encoder":
            out = na.resolve(na.entry(args.erasure)["encoder"])
            export_encoder(config, args.run, out, args.best)
        else:
            labels, meta = pf.load_labels(args.labels)
            _, ok = make_eraser(config, ds, labels, meta, na.entry(args.erasure), device, args.batch_size, args.seed,
                                args.max_train_frames, args.max_val_frames, args.tol_acc, args.tol_bits)
            rc = 0 if ok else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
