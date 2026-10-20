#!/usr/bin/env python
"""Phase 4 (arms C, E): cache the fixed LLM sentence targets for every Brain-to-Text '25 sentence, once.

    <bit env>/python scripts/phase4/cache_text_targets.py [--device cpu] [--batch-size 64]

Run with BIT's environment (transformers 4.48.2; loads Aero-1-Audio via Qwen2AudioForConditionalGeneration exactly
as BIT does). Sentences come from BIT's own Brain-to-Text '25 loader (train / BIT-val / BIT-test; the unlabeled
competition test split is never read), normalized as BIT normalizes them. Writes data/processed/text_targets/
<sha12>.npz + .json (teacher, revision, layers, pooling, normalization, tokenizer settings, content hash).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))

from kunzdecoder import text_targets  # noqa: E402


def bit_sentences(data_dir: Path) -> dict[str, list[str]]:
    """BIT's train / val / test sentences in BIT's order (one channel loaded to keep memory small)."""
    from data.brandman_2024_text.prepare_data import load_data

    ds = load_data(data_dir=str(data_dir), features=["tx1"], area_start=0, area_end=1)
    return {s: [r["sentence"] for r in ds[s]] for s in ("train", "val", "test")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4" / "bit_baseline.yaml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "brandman_2024_text")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "text_targets")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, Qwen2AudioForConditionalGeneration

    llm = yaml.safe_load(args.config.read_text())["llm"]
    model_dir = ROOT / llm["local_dir"]
    t0 = time.time()
    splits = bit_sentences(args.data_dir)
    sentences = sorted({s for v in splits.values() for s in v})
    print(f"{len(sentences)} distinct sentences ({ {k: len(v) for k, v in splits.items()} } trials)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, add_bos_token=False, add_eos_token=False)
    tok.padding_side = "right"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32)
    lm = model.language_model.to(args.device)
    torch.set_grad_enabled(False)
    arrays = text_targets.compute_targets(lm, tok, sentences, batch_size=args.batch_size, device=args.device)
    meta = {"teacher": f"{llm['repo_id']}@{llm['revision']} language_model (Qwen2.5-1.5B), frozen",
            "loader": "Qwen2AudioForConditionalGeneration.from_pretrained(...).language_model (as BIT)",
            "dtype": "float32", "device": args.device,
            "input": "sentence alone: BIT-normalized text, add_special_tokens=False (no prompt, no BOS/EOS)",
            "pooling": "mean over the sentence's tokens (attention-mask aware)", "normalization": "L2",
            "targets_doc": {"emb": "input-embedding mean pool", "layer<k>": "hidden_states[k] mean pool",
                            "layers5": "mean of pooled layers 5,10,15,20,25"},
            "splits": {k: len(v) for k, v in splits.items()}, "data_dir": str(args.data_dir.relative_to(ROOT)),
            "seconds": round(time.time() - t0, 1)}
    path = text_targets.save(args.out, sentences, arrays, meta)
    print(f"wrote {path.relative_to(ROOT)} ({len(sentences)} x {arrays['emb'].shape[1]}) in {meta['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
