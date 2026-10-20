#!/usr/bin/env python
"""Phase 4b: BIT masked-modeling SSL pretraining of the NDT encoder on the human corpus (threshold crossings).

    python scripts/phase4/pretrain_ssl.py [--config configs/phase4/ssl_human.yaml] [--corpus data/processed/ssl_human]
                                          [--out checkpoints/ssl_human] [--seed 42] [--device auto]
                                          [--max-epochs N] [--limit-records N]

Runs in BIT's environment (external/bit importable; no accelerate / deepspeed / wandb needed). Settings come from
`pretrain:` in the config (BIT Table 10). Writes to --out: BEST/ (max masked-bin val R^2, fixed eval masks, equal
weight over pretrain.selection.val_groups), BEST_core/ (BEST without read-in/read-out: use it as BIT's --ft_ckpt
when the fine-tuning features differ, e.g. --features all), LAST/ (+ optimizer/scheduler state), EPOCH<n>/
(every save_every), log.json / log.csv (per epoch: train masked MSE, per-group val R^2 / MSE, BIT's own R^2),
run.json. Code: src/kunzdecoder/bit_ssl.py. Plan: docs/phase4/SSL_PLAN.md §6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import bit_ssl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4" / "ssl_human.yaml")
    ap.add_argument("--corpus", type=Path, help="corpus directory (default: corpus.out_dir of the config)")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "ssl_human")
    ap.add_argument("--seed", type=int, help="default: first of pretrain.seeds")
    ap.add_argument("--device", default="auto", help="auto (cuda if available, else cpu) | cuda | cuda:1 | cpu")
    ap.add_argument("--max-epochs", type=int, help="override pretrain.training.num_epochs (smoke tests)")
    ap.add_argument("--limit-records", type=int, help="first N records per (split, stitch group) (smoke tests)")
    ap.add_argument("--bit-dir", type=Path, default=bit_ssl.BIT_DIR)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = args.seed if args.seed is not None else cfg["pretrain"].get("seeds", [cfg.get("seed", 42)])[0]
    corpus = args.corpus or ROOT / cfg["corpus"]["out_dir"]
    trainer = bit_ssl.build_trainer(cfg, corpus, args.out, seed, device=args.device, max_epochs=args.max_epochs,
                                    limit=args.limit_records, bit=bit_ssl.import_bit(args.bit_dir))
    print(json.dumps({"device": str(trainer.device), "autocast": str(trainer.dtype),
                      "records": {s: len(d) for s, d in trainer.datasets.items()},
                      "batches_per_epoch": len(trainer.train_loader), "out": str(args.out)}), flush=True)
    print(json.dumps(trainer.train(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
