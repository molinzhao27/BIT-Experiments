"""Phase 4 Stage 1: align the neural encoder to FIXED LLM sentence targets, before any LLM training (arms C, E).

Neural side: BIT NDT encoder (no masking, no noise, smoothing 2, exactly as in BIT's sentence stage) -> BIT's MLP
projector into the LLM embedding space (the same module Stage 2 feeds to the LLM, full sequence) -> masked mean pool
-> linear -> L2. Text side: cached target (kunzdecoder.text_targets) -> linear -> L2 (InfoNCE), or the target itself
(cosine / MSE regression, arm E).

Loss (arm C): symmetric InfoNCE with a learnable temperature (init 0.1, logit scale capped at 100, as BIT); pairs that
share a normalized sentence are masked out of the denominator (false negatives). Large batches via GradCache
(Gao et al. 2021): embed the whole batch without gradients, take the loss gradient w.r.t. the embeddings, then
re-run each chunk with gradients, replaying its RNG state so dropout masks match. The gradient equals full-batch
backprop. Every step logs the number of distinct sentences in the InfoNCE denominator.

Outputs a BIT-loadable checkpoint dir: encoder.bin, encoder_config.pth, decoder.bin (NDT.save_checkpoint),
model.safetensors (NDT state dict, what BCI_AudioLLM.load_neural_encoder reads), projector.bin,
projector_config.pth (BCI_AudioLLM format), align_head.pt. Python 3.10-compatible (runs in BIT's environment).
"""

from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BIT_DIR = ROOT / "external" / "bit"


# ---------------------------------------------------------------- objective

class AlignHead(nn.Module):
    """Pooling + projection heads and the alignment loss. Pooling happens only here (amendment 1)."""

    def __init__(self, dim_neural: int = 1536, dim_text: int = 1536, proj_dim: int = 512, loss: str = "infonce",
                 init_temp: float = 0.1, max_scale: float = 100.0):
        super().__init__()
        self.kind, self.max_scale = loss, max_scale
        if loss == "infonce":
            self.neural = nn.Linear(dim_neural, proj_dim)
            self.text = nn.Linear(dim_text, proj_dim)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))
        elif loss in ("cosine", "mse"):  # arm E: regress onto the fixed target; no text head (it could collapse)
            self.neural = nn.Linear(dim_neural, dim_text)
            self.text = None
        else:
            raise ValueError(loss)

    def embed_neural(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return F.normalize(self.neural(pooled).float(), dim=-1)

    def embed_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize((self.text(t) if self.text is not None else t).float(), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_scale)

    def loss(self, zn: torch.Tensor, zt: torch.Tensor, sid: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """zn, zt: (B, d) L2-normalized; sid: (B,) sentence ids (equal ids = same normalized sentence)."""
        if self.kind == "cosine":
            return (1 - (zn * zt).sum(-1)).mean(), {}
        if self.kind == "mse":
            return ((zn - zt) ** 2).sum(-1).mean(), {}
        logits = self.scale() * zn @ zt.T
        same = sid[:, None] == sid[None, :]
        eye = torch.eye(len(sid), dtype=torch.bool, device=sid.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
        labels = torch.arange(len(sid), device=zn.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        stats = {"distinct_in_denominator": int(torch.unique(sid).numel()), "masked_false_negatives":
                 int((same & ~eye).sum().item()) // 2, "logit_scale": float(self.scale().detach()),
                 "in_batch_top1": float((logits.argmax(1) == labels).float().mean())}
        return loss, stats


# ---------------------------------------------------------------- GradCache

def _rng_state(device: torch.device):
    return torch.get_rng_state(), (torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _set_rng_state(state, device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def gradcache_step(embed: Callable[[object], torch.Tensor], chunks: list, zt: torch.Tensor, sid: torch.Tensor,
                   head: AlignHead, device: torch.device) -> tuple[float, dict]:
    """One large-batch alignment step; leaves gradients in every parameter (call optimizer.step() after).

    embed(chunk) -> (b, d) normalized neural embeddings (with dropout); zt: (B, d) text embeddings WITH grad.
    """
    states, parts = [], []
    with torch.no_grad():
        for ch in chunks:
            states.append(_rng_state(device))
            parts.append(embed(ch))
    zn = torch.cat(parts).detach().requires_grad_(True)
    loss, stats = head.loss(zn, zt, sid)
    loss.backward()  # text head, temperature, and d loss / d zn
    grad, start = zn.grad, 0
    for ch, st in zip(chunks, states):
        _set_rng_state(st, device)
        z = embed(ch)
        (z * grad[start:start + len(z)]).sum().backward()
        start += len(z)
    return float(loss), stats


# ---------------------------------------------------------------- BIT pieces

@contextlib.contextmanager
def in_bit_dir():
    """BIT resolves configs by relative path."""
    cwd = os.getcwd()
    os.chdir(BIT_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)


def bit_sentence_config():
    """BIT's full config for `finetune --task sentence --dataset brandman_2024_text --features all` (= Stage 2)."""
    from types import SimpleNamespace

    from utils.config import ConfigBuilder

    args = SimpleNamespace(training_mode="finetune", encoder="ndt", task="sentence", dataset="brandman_2024_text",
                           features="all", ft_ckpt="none", ds_config="none", kwargs=None)
    with in_bit_dir():
        config, _ = ConfigBuilder(args).build()
    return config


def load_matching(module: nn.Module, state: dict) -> dict:
    """Load parameters whose names and shapes match; report the rest (e.g. a read-in whose width changed)."""
    own = module.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    module.load_state_dict(ok, strict=False)
    return {"loaded": len(ok), "skipped_shape": sorted(k for k in state if k in own and own[k].shape != state[k].shape),
            "missing": sorted(k for k in own if k not in state), "unexpected": sorted(k for k in state if k not in own)}


def build_encoder(config, init_dir: Path | None = None):
    """BIT NDT for the sentence stage (TX+SBP, method endtoend), masking/noise off as BCI_AudioLLM.disable_masking.
    init_dir: a BIT/SSL encoder checkpoint; matching-shape parameters are loaded (the core), others reported."""
    from models.ndt import NDT

    kw = dict(config.method.model_kwargs)
    kw["features"] = "all"
    with in_bit_dir():
        ndt = NDT(config.model.encoder, **kw)
    report = None
    if init_dir is not None:
        report = load_matching(ndt.encoder, torch.load(Path(init_dir) / "encoder.bin", map_location="cpu"))
    enc = ndt.encoder
    enc.embedder.mask_active, enc.embedder.mask_ratio = False, 0.0
    enc.smooth_and_noise.noise, enc.smooth_and_noise.smooth = False, 2.0
    return ndt, report


def build_projector(config, hidden: int, llm_hidden: int = 1536) -> nn.Module:
    """Exactly BCI_AudioLLM.build_projector (project_to_audio False)."""
    from transformers.activations import ACT2FN

    p = config.model.projector
    if p.inter_size is not None:
        return nn.Sequential(nn.Linear(hidden * p.stacking, p.inter_size, bias=p.bias), ACT2FN[p.act],
                             nn.Linear(p.inter_size, llm_hidden, bias=p.bias))
    return nn.Linear(hidden * p.stacking, llm_hidden, bias=p.bias)


def save_checkpoint(out: Path, ndt, projector: nn.Module, head: AlignHead, config, extra: dict) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    ndt.save_checkpoint(str(out))  # encoder.bin, encoder_config.pth, decoder.bin
    save_file({k: v.contiguous() for k, v in ndt.state_dict().items()}, str(out / "model.safetensors"))
    torch.save(projector.state_dict(), out / "projector.bin")
    torch.save(dict(config.model.projector), out / "projector_config.pth")
    torch.save({"state": head.state_dict(), **extra}, out / "align_head.pt")


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def retrieval(zn: np.ndarray, rows: np.ndarray, cand_rows: np.ndarray, zt_cand: np.ndarray,
              subset: np.ndarray | None = None, ks=(1, 5)) -> dict:
    """Sentence retrieval: rank the candidate sentences (distinct val sentences) for every val trial."""
    sims = zn @ zt_cand.T  # (N, M)
    true_col = {r: j for j, r in enumerate(cand_rows)}
    tc = np.array([true_col[r] for r in rows])
    rank = (sims > sims[np.arange(len(tc)), tc][:, None]).sum(1)  # 0 = best
    sel = np.ones(len(rows), bool) if subset is None else subset
    return {f"top{k}": float((rank[sel] < k).mean()) for k in ks} | {"n": int(sel.sum()),
                                                                      "n_candidates": int(len(cand_rows))}


def masked_items(records: list[dict], target_rows: dict[str, int]) -> list[dict]:
    """BIT-format decoding items (as SpikingDatasetForDecoding) + target row / sentence id."""
    return [{"spikes": r["spikes"], "spikes_mask": np.ones(len(r["spikes"]), np.int64),
             "spikes_timestamp": np.arange(len(r["spikes"])), "dataset_name": np.array(0),
             "row": np.array(target_rows[r["sentence"]])} for r in records]


PAD = {"spikes": [{"dim": 0, "side": "right", "value": -100}],
       "spikes_mask": [{"dim": 0, "side": "right", "value": 0}],
       "spikes_timestamp": [{"dim": 0, "side": "right", "value": 0}]}


def collate(items: list[dict], device: torch.device) -> dict:
    from utils.datasets import pad_collate_fn

    batch, _ = pad_collate_fn(items, ["spikes", "spikes_mask", "spikes_timestamp", "dataset_name", "row"], PAD)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
