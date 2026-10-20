#!/usr/bin/env python
"""Apply configs/bit_baseline.yaml overrides to BIT's own trainer YAMLs (idempotent).

    python scripts/patch_bit_configs.py [--check]

BIT reads configs/<mode>/<task>/ndt/trainer.yaml from its repo; passing paths through its
--kwargs would leak them into the run name (= checkpoint directory) that eval_llm.py parses,
so the README's documented route (edit the YAML) is used. Only the listed keys change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINER_YAMLS = ["configs/train_from_scratch/phoneme/ndt/trainer.yaml", "configs/finetune/phoneme/ndt/trainer.yaml",
                 "configs/finetune/sentence/ndt/trainer.yaml"]
MODEL_YAMLS = ["configs/models/projector/bci-audio.yaml"]  # only keys listed for them are changed


def set_key(cfg: dict, dotted: str, value) -> bool:
    *parents, leaf = dotted.split(".")
    cur = cfg
    for p in parents:
        cur = cur.setdefault(p, {})
    changed = cur.get(leaf) != value
    cur[leaf] = value
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "bit_baseline.yaml")
    ap.add_argument("--check", action="store_true", help="only report what would change")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    bit = ROOT / cfg["bit"]["path"]
    for rel in TRAINER_YAMLS + MODEL_YAMLS:
        if rel in MODEL_YAMLS and rel not in cfg["overrides"]:
            continue
        path = bit / rel
        text = path.read_text()
        data = yaml.safe_load(text)
        overrides = ({**cfg["overrides"]["all"]} if rel in TRAINER_YAMLS else {}) | cfg["overrides"].get(rel, {})
        changed = [k for k, v in overrides.items() if set_key(data, k, v)]
        print(f"{rel}: {', '.join(changed) or 'up to date'}")
        if changed and not args.check:
            # BIT's "include:" directives are plain strings, so a YAML round trip preserves them
            path.write_text(yaml.safe_dump(data, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
