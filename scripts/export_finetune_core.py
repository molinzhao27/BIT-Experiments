#!/usr/bin/env python
"""Make an SSL core checkpoint fine-tune like training from scratch (BIT's --ft_ckpt), weights unchanged.

    python scripts/export_finetune_core.py [--core checkpoints/ssl_human/BEST_core]
        [--dst checkpoints/ssl_human/BEST_core_ft] [--dataset brandman_2024_text]

BIT merges the checkpoint's encoder_config over the task config, which would keep the SSL masker (50% of patches)
on during CTC fine-tuning. The masker is set to the task's own (BIT registry `embedder_defaults`: inactive for
brandman_2024_text, as in the 4a from-scratch run). See bit_experiments.bit_ssl.export_finetune_core.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments import bit_ssl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", type=Path, default=ROOT / "checkpoints" / "ssl_human" / "BEST_core")
    ap.add_argument("--dst", type=Path, default=ROOT / "checkpoints" / "ssl_human" / "BEST_core_ft")
    ap.add_argument("--dataset", default="brandman_2024_text")
    args = ap.parse_args()
    with bit_ssl.bit_cwd():
        from registry import embedder_defaults
    masker = embedder_defaults[args.dataset]["ndt"]["masker"]
    info = bit_ssl.export_finetune_core(ROOT / args.core, ROOT / args.dst, {"masker": masker})
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
