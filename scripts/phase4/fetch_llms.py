#!/usr/bin/env python
"""Download the pinned LLM snapshots of configs/phase4/llms.yaml (LLM-swap question, DESIGN.md §8).

    <bit env>/python scripts/phase4/fetch_llms.py [name ...]      # default: every entry

Each snapshot goes to its `local_dir` at its pinned `revision` (as setup_bit.sh does for Aero-1-Audio), with
weights, config and tokenizer files only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]


def main(names: list[str]) -> int:
    entries = yaml.safe_load((ROOT / "configs" / "phase4" / "llms.yaml").read_text())
    for name in names or entries:
        e = entries[name]
        path = snapshot_download(e["repo_id"], revision=e["revision"], local_dir=ROOT / e["local_dir"],
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        print(f"{name}: {e['repo_id']}@{e['revision']} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
