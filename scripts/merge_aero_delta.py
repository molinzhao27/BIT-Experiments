#!/usr/bin/env python
"""Aero block diffusion, path 2 option (d) (docs/AERO_DIFFUSION.md): add Aero-1-Audio's audio post-training to
Fast-dLLM v2 1.5B by weight arithmetic, without training. Per language-model tensor:

    W = W_base + alpha * (W_donor - W_reference)
    base = Fast-dLLM v2 1.5B, donor = Aero-1-Audio's LM, reference = Qwen2.5-1.5B-Instruct (both fine-tunes' start)

    <bit env>/python scripts/merge_aero_delta.py --out checkpoints/hf/Fast_dLLM_v2_1.5B-aero [--alpha 1.0]
        [--base fast-dllm-v2-1.5b] [--donor aero-1-audio-diffusion] [--reference qwen2.5-1.5b-instruct]

Arguments are configs/llms.yaml entries or snapshot directories. Fast-dLLM v2 moved Instruct's weights by ~5.2%
(relative norm, sampled layers) and Aero by ~2.0%, in nearly orthogonal directions (cosine of the two changes
~0.01): the condition under which task vectors add without much interference. Whether the audio skill survives in
the block-diffusion layout is the open question; the doc's checks decide before any Stage 2 run.

- Embedding (tied with the head): only the base's rows (151,936). Aero's 64 extra rows are unused and dropped. The
  base's own added tokens (|<MASK>| = 151665 in its added_tokens.json) keep the base's rows: NVIDIA trained the mask
  row, and Aero's change to that row is only the drift shared by all its unused rows.
- Output: <out>/model.safetensors in fp32 (small changes survive) under the base's tensor names, the base's config,
  tokenizer and added-tokens files (so it loads as the base does: loader diffusion, |<MASK>| from added_tokens.json),
  and <out>/merge.json: args, snapshots, per-tensor relative change and the cosine between the two changes.
CPU only, tensor by tensor; peak memory about one fp32 copy of the model (6.2 GB). Seed-free (deterministic).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))

from bit_experiments.diffusion_llm import AERO_LM_PREFIX, MASK_TOKEN  # noqa: E402

EMBED = "model.embed_tokens.weight"


class LMTensors:
    """A snapshot's language-model tensors by Qwen2ForCausalLM name, read one at a time (fp32). Aero-1-Audio's
    `language_model.` prefix is stripped and its audio tensors are not listed."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = path
        config = json.loads((path / "config.json").read_text())
        prefix = AERO_LM_PREFIX if config.get("model_type") == "aero" else ""
        index = path / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else \
            ["model.safetensors"]
        self.where = {}
        for name in files:
            with safe_open(path / name, framework="pt") as f:
                self.where |= {k[len(prefix):]: (path / name, k) for k in f.keys() if k.startswith(prefix)}

    def __getitem__(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        file, key = self.where[name]
        with safe_open(file, framework="pt") as f:
            return f.get_tensor(key).float()


def snapshot_dir(name_or_dir: str) -> tuple[Path, dict | None]:
    """A registry entry's local snapshot, or a directory."""
    if Path(name_or_dir).is_dir():
        return Path(name_or_dir), None
    from bit_experiments.text_llm import llm_entry

    entry = llm_entry(name_or_dir)
    return Path(entry["path"]), entry


def merge(base: LMTensors, donor: LMTensors, reference: LMTensors, alpha: float, keep_rows: list[int]):
    """The merged state dict under the base's names, and per-tensor statistics."""
    out, stats = {}, {}
    dot = norm_d = norm_b = 0.0
    for name in sorted(base.where):
        if name == "lm_head.weight":
            continue  # tied: the embedding
        w = base[name]
        d, r = donor[name], reference[name]
        if name == EMBED:
            d, r = d[: w.shape[0]], r[: w.shape[0]]  # Aero's 152,000 rows -> the base's vocabulary
        if d.shape != w.shape or r.shape != w.shape:
            raise ValueError(f"{name}: base {tuple(w.shape)}, donor {tuple(d.shape)}, reference {tuple(r.shape)}")
        delta = d - r
        if name == EMBED and keep_rows:
            delta[keep_rows] = 0  # the base's own added tokens (the mask row) stay the base's
        own = w - r  # the base's change from the reference (the diffusion conversion)
        out[name] = w + alpha * delta
        stats[name] = {"donor_change": round(float(delta.norm() / r.norm().clamp_min(1e-12)), 5),
                       "base_change": round(float(own.norm() / r.norm().clamp_min(1e-12)), 5),
                       "cos": round(float(torch.nn.functional.cosine_similarity(delta.flatten(), own.flatten(), dim=0)),
                                    4)}
        dot, norm_d, norm_b = dot + float((delta * own).sum()), norm_d + float(delta.norm()) ** 2, \
            norm_b + float(own.norm()) ** 2
    summary = {"cos_pooled": round(dot / max((norm_d * norm_b) ** 0.5, 1e-30), 4), "tensors": len(out)}
    return out, stats, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output snapshot directory (new)")
    ap.add_argument("--base", default="fast-dllm-v2-1.5b", help="registry entry or directory (block diffusion)")
    ap.add_argument("--donor", default="aero-1-audio-diffusion", help="registry entry or directory (Aero-1-Audio)")
    ap.add_argument("--reference", default="qwen2.5-1.5b-instruct", help="registry entry or directory (their start)")
    ap.add_argument("--alpha", type=float, default=1.0, help="scale of the donor's change")
    args = ap.parse_args(argv)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} exists and is not empty")
    (base_dir, _), (donor_dir, _), (ref_dir, _) = map(snapshot_dir, (args.base, args.donor, args.reference))
    added = base_dir / "added_tokens.json"
    keep_rows = [json.loads(added.read_text())[MASK_TOKEN]] if added.exists() else []
    merged, stats, summary = merge(LMTensors(base_dir), LMTensors(donor_dir), LMTensors(ref_dir), args.alpha,
                                   keep_rows)

    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in merged.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in base_dir.iterdir():  # config, tokenizer, added tokens, generation config, code: the base's
        weights = f.suffix in (".safetensors", ".bin", ".pt") or f.name.endswith(".index.json")
        if f.is_file() and not weights:
            shutil.copy2(f, out / f.name)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "merge.json").write_text(json.dumps({
        "args": vars(args), "base": str(base_dir), "donor": str(donor_dir), "reference": str(ref_dir),
        "kept_rows": keep_rows, "summary": summary, "tensors": stats, "git": git}, indent=1))
    print(f"[merge] {out}: {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
