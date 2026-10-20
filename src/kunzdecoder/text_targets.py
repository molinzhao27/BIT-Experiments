"""Fixed sentence targets for Stage 1 alignment (Phase 4, arms C and E), cached once so they never move.

Teacher: the language model inside Aero-1-Audio (Qwen2.5-1.5B), i.e. the same LLM that decodes in Stage 2, run
frozen over each cued sentence ON ITS OWN: no prompt, no BOS/EOS, BIT's sentence normalization. Pooling: mean
over the sentence's tokens. Normalization: L2. Targets:
  emb        input-embedding mean pool (the clean version of BIT's in-loop aligner target, which also pools the
             shared prompt and padding)
  layer<k>   mean pool of hidden_states[k] (output of decoder layer k; 0 = embeddings)
  layers5    mean of the pooled layers 5, 10, 15, 20, 25 (Zufle & Niehues: deeper/all-layer targets)
Python 3.10-compatible (runs in BIT's environment).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_LAYERS = (7, 14, 21, 28)
EVERY5 = (5, 10, 15, 20, 25)


def compute_targets(lm, tokenizer, sentences: list[str], layers=DEFAULT_LAYERS, batch_size: int = 64,
                    device: str = "cpu") -> dict[str, np.ndarray]:
    """{target name: (len(sentences), hidden) float32, L2-normalized} for a causal LM (HF, output_hidden_states)."""
    import torch

    need = sorted(set(layers) | set(EVERY5))
    out = {"emb": [], **{f"layer{k}": [] for k in layers}, "layers5": []}
    lm.eval()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            hs = lm(**enc, output_hidden_states=True).hidden_states  # (n_layers + 1) x (B, L, H)
        m = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        pooled = {k: (hs[k] * m).sum(1) / m.sum(1).clamp_min(1) for k in [0] + need}
        out["emb"].append(pooled[0])
        for k in layers:
            out[f"layer{k}"].append(pooled[k])
        out["layers5"].append(torch.stack([pooled[k] for k in EVERY5]).mean(0))
    res = {}
    for name, parts in out.items():
        x = torch.cat(parts).float()
        res[name] = torch.nn.functional.normalize(x, dim=-1).cpu().numpy().astype(np.float32)
    return res


def digest(arrays: dict[str, np.ndarray], sentences: list[str]) -> str:
    h = hashlib.sha256()
    for s in sentences:
        h.update(s.encode() + b"\0")
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


def save(out_dir: Path, sentences: list[str], arrays: dict[str, np.ndarray], meta: dict) -> Path:
    """Write <out_dir>/<sha12>.npz (+ .json manifest); the file name is the content hash."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = digest(arrays, sentences)
    path = out_dir / f"{sha[:12]}.npz"
    np.savez(path, sentences=np.asarray(sentences, dtype=object), **arrays)
    (out_dir / f"{sha[:12]}.json").write_text(json.dumps(meta | {"sha256": sha, "n_sentences": len(sentences),
                                                                  "targets": sorted(arrays)}, indent=1))
    (out_dir / "LATEST").write_text(path.name + "\n")
    return path


def load(path: Path, target: str) -> tuple[dict[str, int], np.ndarray, dict]:
    """(sentence -> row, (N, H) target matrix, manifest); verifies the content hash."""
    path = Path(path)
    if path.is_dir():
        path = path / (path / "LATEST").read_text().strip()
    z = np.load(path, allow_pickle=True)
    sentences = [str(s) for s in z["sentences"]]
    meta = json.loads(path.with_suffix(".json").read_text())
    arrays = {k: z[k] for k in meta["targets"]}
    if digest(arrays, sentences) != meta["sha256"]:
        raise ValueError(f"{path}: content hash mismatch; the target cache was modified")
    return {s: i for i, s in enumerate(sentences)}, arrays[target], meta
