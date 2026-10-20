#!/usr/bin/env python
"""Phase 4b check: is the SSL encoder's masked-bin val R^2 more than local smoothing?

    python scripts/ssl_baselines.py [--ckpt checkpoints/ssl_human/BEST] [--config configs/ssl_human.yaml]
                                           [--device cpu] [--out results/ssl_baselines.json]

Scores the checkpoint and two no-learning predictors (bit_experiments.bit_ssl.trivial_predictions: each channel's trial
mean of the unmasked bins, and the mean of the nearest unmasked 100 ms on either side of each gap) on exactly the
same masked val bins (the fixed selection masks), with the same R^2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments import bit_ssl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ssl_human.yaml")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "ssl_human" / "BEST")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ssl_baselines.json")
    args = ap.parse_args()
    from safetensors.torch import load_file

    cfg = yaml.safe_load(args.config.read_text())
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    seed = cfg["pretrain"].get("seeds", [cfg.get("seed", 42)])[0]
    tr = bit_ssl.build_trainer(cfg, ROOT / cfg["corpus"]["out_dir"], ckpt.parent / "_baselines_unused", seed,
                               device=args.device)
    tr.model.load_state_dict(load_file(str(ckpt / "model.safetensors")))
    sel = tr.cfg["selection"]
    res = bit_ssl.evaluate(tr.model, tr.eval_loaders["val"], tr.groups, sel["eval_mask_seed"], tr.device,
                           selection_groups=sel.get("val_groups"), baselines=True)
    res = {"checkpoint": str(ckpt.relative_to(ROOT)), "split": "val", **res}
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("r2_masked")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
