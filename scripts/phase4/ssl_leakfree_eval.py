#!/usr/bin/env python
"""Phase 4b check: masked-bin val R^2 of the SSL encoder without the smoothing leak (AUDIT 2026-10-19 item 7).

    python scripts/phase4/ssl_leakfree_eval.py [--ckpt checkpoints/ssl_human/BEST] [--config configs/phase4/ssl_human.yaml]
                                               [--threads 8] [--out results/phase4/ssl_leakfree.json]

BIT's NDT Gaussian-smooths the raw input (sd 2 bins, 13 taps) before patching and masking (external/bit/models/
ndt.py:454, model_utils.py:61-68), so visible bins next to a masked span carry part of the masked raw values (weights
0.176 / 0.121 / 0.065 / 0.027 / 0.009 / 0.002 at offsets 1-6). The no-learning baselines see raw unmasked bins only.
This script re-scores the trained checkpoint (no retraining) on exactly the fixed selection masks and loader of
`bit_ssl.evaluate` (same masks, same R^2), with the smoothed model input built in different ways:

  input           masked raw bins, before smoothing             pad bins (-100), before smoothing
  bit             kept (as trained and reported: leaks)         kept (BIT: tails of shorter trials pulled to -100)
  zero            set to 0 (the z-scored channel mean)          kept
  resample        replaced by a random visible bin, same trial  kept
  *_padfix        as above                                      set to 0, i.e. each trial smoothed as if alone
  visible_only    excluded: smoothing renormalized over the visible bins only (pad bins excluded too)

`zero`, `resample` and `visible_only` give the model only the trial's unmasked raw bins, the same information as the
baselines (`bit_ssl.trivial_predictions`). Each R^2 is also split into masked bins within 6 bins of a visible bin
(`edge`, the only ones that can leak) and the rest (`interior`). `leak_probe`: least squares of the target on the pure
leaked signal at the nearest visible bin on each side (smoothing of the masked raw bins only) plus the trial mean, per
(left, right) distance to the gap edge; `leak_probe_ref` is the same fit without the leaked signal. Both are fit and
scored on the same bins (<= 256 parameters per group against >= 6 M values).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import bit_ssl, ssl_data  # noqa: E402

PAD = bit_ssl.PAD_VALUE
EDGE = 6  # kernel half-width (smooth_sd 2 -> 13 taps)
DCAP = 7  # probe buckets: distance to the gap edge 1..6, 7 = no leak (>= 7 or no visible bin on that side)
VARIANTS = ("bit", "bit_padfix", "zero", "zero_padfix", "resample", "resample_padfix", "visible_only")


class GivenInput(torch.nn.Module):
    """Stands in for NeuralEncoder.smooth_and_noise: returns the precomputed smoothed input."""

    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, spikes):
        if self.x is None or self.x.shape != spikes.shape:
            raise RuntimeError("precomputed input missing or of the wrong shape")
        return self.x


def build(cfg: dict, ckpt: Path, device):
    """The model and the val loader of `bit_ssl.build_trainer` (same config, batches and order), val split only."""
    from safetensors.torch import load_file
    bit = bit_ssl.import_bit()
    pretrain = cfg["pretrain"]
    seed = pretrain.get("seeds", [cfg.get("seed", 42)])[0]
    data, manifest = ssl_data.load_corpus(ROOT / cfg["corpus"]["out_dir"], splits=("val",))
    bit_ssl.patch_stitch_tables(manifest, pretrain.get("features", "tx1"), bit)
    run_cfg = bit_ssl.bit_config(pretrain, seed, next(iter(manifest["stitch_groups"])), bit)
    model = bit_ssl.build_ndt(run_cfg, bit).to(device)
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")))
    model.eval()
    ds = bit_ssl.multi_spiking_dataset(bit)(data, "val", **run_cfg["method"]["dataset_kwargs"])
    loader = bit_ssl.make_loader(bit, ds, run_cfg["training"]["test_batch_size"], False, run_cfg["seed"],
                                 bit_ssl.model_inputs(model), False, False)
    return model, loader, list(manifest["stitch_groups"]), run_cfg


def inputs_for(x, valid, masked, visible, smooth, gen):
    """Smoothed model input per variant (B, T, C)."""
    v3, m3 = valid.unsqueeze(-1), masked.unsqueeze(-1)
    padfix = lambda z: torch.where(v3, smooth(torch.where(v3, z, 0.0)), torch.full_like(z, PAD))  # noqa: E731
    xz = torch.where(m3, 0.0, x)
    xr = xz.clone()
    for b in range(x.shape[0]):
        vi, mi = visible[b].nonzero().squeeze(1), masked[b].nonzero().squeeze(1)
        if len(mi) and len(vi):
            xr[b, mi] = x[b, vi[torch.randint(len(vi), (len(mi),), generator=gen)]]
    w = visible.unsqueeze(-1).to(x.dtype)
    vis = smooth(torch.where(visible.unsqueeze(-1), x, 0.0)) / smooth(w).clamp(min=1e-6)
    vis = torch.where(visible.unsqueeze(-1), vis, torch.where(v3, torch.zeros_like(x), torch.full_like(x, PAD)))
    return {"bit": smooth(x), "bit_padfix": padfix(x), "zero": smooth(xz), "zero_padfix": padfix(xz),
            "resample": smooth(xr), "resample_padfix": padfix(xr), "visible_only": vis}


def gap_distances(visible):
    """Distance (bins) from each bin to the nearest visible bin on its left / right (inf if none), and their indices."""
    B, T = visible.shape
    t = torch.arange(T, device=visible.device).expand(B, T)
    left = torch.where(visible, t, torch.full_like(t, -1)).cummax(1).values
    right = torch.where(visible, t, torch.full_like(t, T)).flip(1).cummin(1).values.flip(1)
    dl = torch.where(left >= 0, (t - left).double(), math.inf)
    dr = torch.where(right < T, (right - t).double(), math.inf)
    return dl, dr, left.clamp(min=0), right.clamp(max=T - 1)


def add(acc: dict, key, y, p, sel):
    """Per-channel sums of `bit_ssl.evaluate` over the bins in sel (B, T)."""
    m = sel.unsqueeze(-1).double()
    a = acc.setdefault(key, {"n": 0.0, "sy": 0.0, "syy": 0.0, "res": 0.0})
    a["n"] = a["n"] + m.sum((0, 1))
    a["sy"] = a["sy"] + (y * m).sum((0, 1))
    a["syy"] = a["syy"] + (y * y * m).sum((0, 1))
    a["res"] = a["res"] + ((p - y) ** 2 * m).sum((0, 1))


def r2(a: dict) -> float:
    n = a["n"].clamp(min=1)
    ss_tot = (a["syy"] - a["sy"] ** 2 / n).sum().item()
    return 1.0 - a["res"].sum().item() / ss_tot if ss_tot > 0 else float("nan")


@torch.no_grad()
def run(model, loader, groups, mask_seed: int, device, resample_seed: int = 0) -> dict:
    enc, emb = model.encoder, model.encoder.embedder
    smooth_and_noise = enc.smooth_and_noise
    given = GivenInput()
    ps = emb.patch_size
    gen_mask, gen_rs = torch.Generator(), torch.Generator().manual_seed(resample_seed)
    acc, probe, stats = {}, {}, defaultdict(list)
    count = defaultdict(int)
    checked = False
    try:
        for inputs, _ in loader:
            t0 = time.time()
            inputs = bit_ssl._to(inputs, device)
            g = int(inputs["dataset_name"].unique().item())
            gen_mask.manual_seed(int(mask_seed) * 1_000_003 + g * 10_007 + count[g])  # = bit_ssl.fixed_masks reseed
            count[g] += 1
            x = inputs["spikes"]
            B, T, C = x.shape
            pmask = bit_ssl._bit_masker(emb, B, math.ceil(T / ps), device, gen_mask)
            emb.masker = lambda batch_size, num_patches, dev: pmask  # noqa: E731 (shape checked by the assert below)
            valid = inputs["spikes_mask"].bool()
            mbins = pmask.repeat_interleave(ps, 1)[:, :T]
            masked, visible = mbins & valid, valid & ~mbins
            y = x.double()
            dl, dr, il, ir = gap_distances(visible)
            d = torch.minimum(dl, dr)
            subsets = {"all": masked, "edge": masked & (d <= EDGE), "interior": masked & (d > EDGE)}

            # batch composition: padding smoothed into shorter trials' tails (SSL_PLAN.md 1.3)
            lens = valid.sum(1)
            stats["group"].append(g)
            stats["len_min"].append(int(lens.min()))
            stats["len_max"].append(int(lens.max()))
            stats["n_shorter"].append(int((lens < T).sum()))
            stats["n_trials"].append(B)
            stats["n_valid_bins"].append(int(valid.sum()))
            stats["n_tail_bins"].append(int(((lens < T).unsqueeze(1) & valid
                                             & (torch.arange(T, device=device) >= (lens - EDGE).unsqueeze(1))).sum()))

            # model under each input
            if not checked:  # the substitute path reproduces BIT's own forward exactly
                ref = model(**inputs)
            enc.smooth_and_noise = given
            try:
                for name, xin in inputs_for(x, valid, masked, visible, smooth_and_noise, gen_rs).items():
                    given.x = xin
                    out = model(**inputs)
                    if not torch.equal(out.mask.bool()[:, :T, 0], masked):
                        raise RuntimeError("model mask differs from the precomputed mask")
                    if not checked and name == "bit":
                        if not torch.equal(out.preds, ref.preds):
                            raise RuntimeError("substitute input path does not reproduce BIT's forward")
                        checked = True
                    p = out.preds.double()[:, :T]
                    for s, sel in subsets.items():
                        add(acc, (g, name, s), y, p, sel)
            finally:
                enc.smooth_and_noise = smooth_and_noise
                given.x = None

            # no-learning baselines (raw unmasked bins only), as bit_ssl.evaluate(baselines=True)
            base = bit_ssl.trivial_predictions(y, masked.unsqueeze(-1).expand_as(y), (x != PAD))
            for k, q in base.items():
                for s, sel in subsets.items():
                    add(acc, (g, f"baseline_{k}", s), y, q, sel)

            # leak probe: target ~ [leak at nearest visible bin left, right, trial mean, 1] per (dl, dr) bucket
            leak = smooth_and_noise(torch.where(masked.unsqueeze(-1), x, 0.0)).double()  # what smoothing leaks
            tm = base["trial_mean"]
            bi, ti = masked.nonzero(as_tuple=True)
            ql = torch.where((dl[bi, ti] <= EDGE).unsqueeze(1), leak[bi, il[bi, ti]], 0.0)
            qr = torch.where((dr[bi, ti] <= EDGE).unsqueeze(1), leak[bi, ir[bi, ti]], 0.0)
            feats = torch.stack([ql, qr, tm[bi, ti], torch.ones_like(ql)], -1)  # (n, C, 4)
            bucket = (dl[bi, ti].clamp(max=DCAP) * (DCAP + 1) + dr[bi, ti].clamp(max=DCAP)).long()
            yy = y[bi, ti]
            pr = probe.setdefault(g, {"A": torch.zeros((DCAP + 1) ** 2, 4, 4, dtype=torch.float64),
                                      "b": torch.zeros((DCAP + 1) ** 2, 4, dtype=torch.float64),
                                      "yy": torch.zeros((DCAP + 1) ** 2, dtype=torch.float64)})
            pr["A"].index_add_(0, bucket.cpu(), torch.einsum("ncj,nck->njk", feats, feats).cpu())
            pr["b"].index_add_(0, bucket.cpu(), torch.einsum("ncj,nc->nj", feats, yy).cpu())
            pr["yy"].index_add_(0, bucket.cpu(), (yy * yy).sum(1).cpu())
            print(f"group {groups[g]} batch {count[g]}: B {B} T {T} ({time.time() - t0:.1f} s)", flush=True)
    finally:
        if "masker" in vars(emb):
            del emb.masker

    res = {"r2_masked": {}, "batches": {}, "leak_probe": {}}
    for (g, name, s), a in sorted(acc.items()):
        res["r2_masked"].setdefault(groups[g], {}).setdefault(name, {})[s] = r2(a)
        if name == "bit":
            res["r2_masked"][groups[g]].setdefault("n_bins", {})[s] = int(a["n"][0].item())
    for g, pr in sorted(probe.items()):
        ss_tot = None
        for (gg, name, s), a in acc.items():
            if gg == g and name == "bit" and s == "all":
                n = a["n"].clamp(min=1)
                ss_tot = (a["syy"] - a["sy"] ** 2 / n).sum().item()
        out = {}
        for key, cols in (("leak_probe", [0, 1, 2, 3]), ("leak_probe_ref", [2, 3])):
            A, bv = pr["A"][:, cols][:, :, cols], pr["b"][:, cols]
            beta = torch.einsum("kij,kj->ki", torch.linalg.pinv(A), bv)
            ss_res = (pr["yy"] - 2 * (beta * bv).sum(1) + torch.einsum("ki,kij,kj->k", beta, A, beta)).sum().item()
            out[key] = 1.0 - ss_res / ss_tot
        res["leak_probe"][groups[g]] = out
    for g in sorted(set(stats["group"])):
        idx = [i for i, gg in enumerate(stats["group"]) if gg == g]
        pick = lambda k: [stats[k][i] for i in idx]  # noqa: E731
        res["batches"][groups[g]] = {
            "n_batches": len(idx),
            "n_batches_mixed_lengths": sum(a != b for a, b in zip(pick("len_min"), pick("len_max"))),
            "n_trials": sum(pick("n_trials")),
            "n_trials_shorter_than_batch_max": sum(pick("n_shorter")),
            "frac_valid_bins_in_padded_tails": sum(pick("n_tail_bins")) / sum(pick("n_valid_bins")),
            "len_min_max_per_batch": list(zip(pick("len_min"), pick("len_max"))),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4" / "ssl_human.yaml")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "ssl_human" / "BEST")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reference", type=Path, default=ROOT / "results" / "phase4" / "ssl_baselines.json",
                    help="saved bit_ssl.evaluate result the `bit` input must reproduce")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "phase4" / "ssl_leakfree.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(args.config.read_text())
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    device = torch.device(args.device)
    model, loader, groups, run_cfg = build(cfg, ckpt, device)
    res = run(model, loader, groups, run_cfg["selection"]["eval_mask_seed"], device)
    ref = json.loads(args.reference.read_text()) if args.reference.exists() else {}
    res["reproduces_reference"] = {
        g: {"saved": ref.get(f"r2_masked_{g}"), "bit": v["bit"]["all"],
            "match": ref.get(f"r2_masked_{g}") is not None and abs(ref[f"r2_masked_{g}"] - v["bit"]["all"]) < 1e-5}
        for g, v in res["r2_masked"].items()}
    res = {"checkpoint": str(ckpt.relative_to(ROOT)), "split": "val", "variants": list(VARIANTS), **res}
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for g, v in res["r2_masked"].items():
        print(g)
        for name, s in v.items():
            if name != "n_bins":
                print(f"  {name:28s} " + "  ".join(f"{k} {x:+.4f}" for k, x in s.items()))
        print(f"  n_bins {v['n_bins']}  leak probe {res['leak_probe'][g]}")
    print(json.dumps(res["reproduces_reference"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
