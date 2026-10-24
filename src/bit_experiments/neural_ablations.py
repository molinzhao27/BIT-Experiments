"""Phase 4 retraining ablations (docs/RETRAINING_ABLATIONS.md): Stage 2 with only what reaches the projector and LLM
changed. Selected with `train_stage2.py --ablation NAME` (an entry of configs/ablations.yaml); without the flag
nothing changes.

Kinds:
- `frozen`: the encoder loaded from --init is frozen. It always runs in eval mode (no dropout, no gradient), and its
  output goes to the projector unchanged. This is the control for the other two.
- `phoneme_bottleneck`: the frozen encoder is arm A's CTC phoneme model. Its CTC read-out (decoder.bin: Linear 384 ->
  41, then log-softmax) gives per-frame log-posteriors over BIT's 41 classes (blank, 39 phonemes, SIL). They are
  floored at log(floor) and standardized per class with fixed train-split statistics. These 41 numbers per frame are
  the only neural input to the projector, whose first Linear is re-sized to 41 inputs (the rest of BIT's
  projector is kept as built).
- `phoneme_erasure`: the frozen encoder's output passes a fixed LEACE eraser (Belrose et al. 2023), then the
  projector. scripts/prepare_ablations.py fits the eraser on the train split, with frame labels from CTC forced
  alignment with arm A's model (bit_experiments.phoneme_frames).

Mechanics:
- The model's `prepare_neural_embeds` is replaced on the instance, as stage2.fix_padding replaces `merge_embeds`. This
  covers BCI_AudioLLM, BCI_TextLLM and BCI_DiffusionLLM alike (all reach the encoder through `prepare_embeds`).
- The frozen encoder runs under the ambient autocast (bf16 in Stage 2 on CUDA, as when the artifacts were fit). The
  read-out, standardization and eraser run in float32.
- Artifacts (read-out weights, statistics, eraser) are non-persistent buffers, loaded from their files and checked by
  content hash. Slim checkpoints keep their format, and any checkpoint can be evaluated with an ablation inserted
  (`--eval-only`).
- A new layer is built under a forked RNG seeded with the run's seed, so the global random stream (dropout, LoRA)
  continues as without the ablation.
"""

from __future__ import annotations

import hashlib
import math
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "ablations.yaml"
KINDS = ("frozen", "phoneme_bottleneck", "phoneme_erasure")
FEATURES = ("log_posterior", "posterior", "one_hot")
N_VOCAB = 41
READ_OUT = "read_out.readout_dict.{}."  # NDT DecoderWrapper -> ReadOut (n_outputs = vocab) for one dataset index


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def entry(name: str) -> dict:
    entries = yaml.safe_load(REGISTRY.read_text())
    if name not in entries:
        raise KeyError(f"{name!r} not in {REGISTRY} (have: {', '.join(entries)})")
    e = entries[name] | {"name": name}
    if e["kind"] not in KINDS:
        raise ValueError(f"{name}: unknown kind {e['kind']!r}")
    return e


def state_digest(state: dict) -> str:
    """sha256 over sorted names, shapes and float32 values: the same for equal tensors on any device / dtype."""
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k].detach().to("cpu", torch.float32).contiguous()
        h.update(f"{k}:{tuple(v.shape)};".encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- modules between encoder and projector

class Passthrough(nn.Module):
    def forward(self, h, mask):
        return h


class PhonemeReadout(nn.Module):
    """Arm A's CTC read-out, frozen, and the bottleneck's feature of its log-posteriors (float32)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, feature: str = "log_posterior", floor: float = 1e-8,
                 mean: torch.Tensor | None = None, sd: torch.Tensor | None = None):
        super().__init__()
        if feature not in FEATURES:
            raise ValueError(f"feature {feature!r} not in {FEATURES}")
        self.feature, self.log_floor = feature, math.log(floor)
        self.register_buffer("weight", weight.float(), persistent=False)
        self.register_buffer("bias", bias.float(), persistent=False)
        self.register_buffer("mean", None if mean is None else mean.float(), persistent=False)
        self.register_buffer("sd", None if sd is None else sd.float(), persistent=False)

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def log_posteriors(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):
            return F.log_softmax(F.linear(h.float(), self.weight, self.bias), dim=-1)

    def raw(self, h: torch.Tensor) -> torch.Tensor:
        """The feature before standardization."""
        lp = self.log_posteriors(h)
        if self.feature == "log_posterior":
            return lp.clamp_min(self.log_floor)
        if self.feature == "posterior":
            return lp.exp()
        return F.one_hot(lp.argmax(-1), lp.shape[-1]).float()

    def forward(self, h, mask):
        x = self.raw(h)
        if self.mean is not None:
            with torch.autocast(h.device.type, enabled=False):
                x = (x - self.mean) / self.sd
        return x

    @classmethod
    def from_checkpoint(cls, ckpt_dir: Path, dataset_idx: int = 0, **kw) -> "PhonemeReadout":
        """From an NDT CTC checkpoint's decoder.bin (BIT's Trainer.save_checkpoint)."""
        state = torch.load(Path(ckpt_dir) / "decoder.bin", map_location="cpu", weights_only=True)
        prefix = READ_OUT.format(dataset_idx)
        w, b = state[prefix + "weight"], state[prefix + "bias"]
        if w.shape[0] != N_VOCAB:
            raise ValueError(f"{ckpt_dir}: read-out has {w.shape[0]} outputs, expected BIT's {N_VOCAB}")
        return cls(w, b, **kw)


class LinearEraser(nn.Module):
    """x -> A x + b in float32 (A, b from `leace`)."""

    def __init__(self, A: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A.float(), persistent=False)
        self.register_buffer("b", b.float(), persistent=False)

    def forward(self, h, mask):
        with torch.autocast(h.device.type, enabled=False):
            return F.linear(h.float(), self.A, self.b)


# ---------------------------------------------------------------- LEACE

class CovAccumulator:
    """Running float64 sums for the mean of x, Cov(x) and Cov(x, z) over (x [N, d], z [N, k]) pairs."""

    def __init__(self, d: int, k: int):
        self.n = 0
        self.sx, self.sz = torch.zeros(d, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
        self.sxx = torch.zeros(d, d, dtype=torch.float64)
        self.sxz = torch.zeros(d, k, dtype=torch.float64)

    def update(self, x: torch.Tensor, z: torch.Tensor) -> None:
        x, z = torch.as_tensor(x).double().cpu(), torch.as_tensor(z).double().cpu()
        self.n += x.shape[0]
        self.sx += x.sum(0)
        self.sz += z.sum(0)
        self.sxx += x.T @ x
        self.sxz += x.T @ z

    def moments(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.n
        mx, mz = self.sx / n, self.sz / n
        return mx, self.sxx / n - torch.outer(mx, mx), self.sxz / n - torch.outer(mx, mz)


def one_hot_concept(classes: torch.Tensor, n_classes: int) -> torch.Tensor:
    """classes [N, c] (-1 = none) -> concatenated one-hots [N, c * n_classes]."""
    classes = torch.as_tensor(classes).long()
    if classes.dim() == 1:
        classes = classes[:, None]
    out = torch.zeros(classes.shape[0], classes.shape[1] * n_classes, dtype=torch.float64)
    for c in range(classes.shape[1]):
        ok = classes[:, c] >= 0
        out[ok.nonzero().flatten(), c * n_classes + classes[ok, c]] = 1
    return out


def leace(mean: torch.Tensor, cov_xx: torch.Tensor, cov_xz: torch.Tensor, tol: float = 1e-10) -> dict:
    """LEACE eraser (Belrose et al., NeurIPS 2023): r(x) = x - W+ P W (x - mean), with W = (Cov_xx^1/2)+ the
    whitening, W+ = Cov_xx^1/2 and P the orthogonal projection onto colspace(W Cov_xz). Then Cov(r(X), Z) = 0: every
    class-conditional mean equals the overall mean (linear guardedness), with the least change to X. Pseudo-inverses
    handle a singular Cov_xx (an encoder output after LayerNorm lies in a hyperplane). Returns A = I - W+ P W,
    b = W+ P W mean (float64), the rank of P and the eigenvalue floor used."""
    evals, evecs = torch.linalg.eigh(cov_xx)
    keep = evals > tol * evals.max()
    sqrt = evecs[:, keep] @ torch.diag(evals[keep].sqrt()) @ evecs[:, keep].T
    isqrt = evecs[:, keep] @ torch.diag(evals[keep].rsqrt()) @ evecs[:, keep].T
    u, s, _ = torch.linalg.svd(isqrt @ cov_xz, full_matrices=False)
    rank = int((s > tol ** 0.5 * s.max()).sum()) if s.numel() and s.max() > 0 else 0
    u = u[:, :rank]
    erase = sqrt @ (u @ u.T) @ isqrt
    A = torch.eye(cov_xx.shape[0], dtype=torch.float64) - erase
    return {"A": A, "b": erase @ mean, "rank": rank, "cov_rank": int(keep.sum())}


# ---------------------------------------------------------------- Stage 2 hook

def ablated_prepare_neural_embeds(self, spikes, spikes_mask, spikes_timestamp, dataset_name=None, day_idx=None):
    """BCI_AudioLLM.prepare_neural_embeds with the frozen encoder (eval mode, no gradient), then the ablation module."""
    enc = self.neural_encoder.encoder
    was = enc.training
    enc.eval()
    try:
        with torch.no_grad():
            h, mask, _, _, _ = enc(spikes, spikes_mask, spikes_timestamp, dataset_name, force_mask=False)
    finally:
        enc.train(was)
    return self.neural_ablation(h, mask), mask


def _check_from_init(enc: nn.Module, init: Path) -> dict:
    """Every encoder tensor comes from init/encoder.bin (a frozen random read-in would be meaningless)."""
    state = torch.load(init / "encoder.bin", map_location="cpu", weights_only=True)
    own = enc.state_dict()
    missing = sorted(k for k in own if k not in state)
    differ = sorted(k for k in own if k in state and not torch.equal(own[k].cpu(), state[k].to(own[k].dtype)))
    if missing or differ:
        raise RuntimeError(f"frozen encoder not fully loaded from {init}: missing {missing[:5]}, differ {differ[:5]}")
    return {"encoder_from_init": True, "n_tensors": len(own)}


def projector_in_features(proj: nn.Module) -> int:
    """Input width of BIT's projector: a Linear (the repo's released config, inter_size null) or an MLP
    Sequential(Linear, act, Linear) (inter_size 1024: configs/bit_baseline.yaml, every run on the host)."""
    return (proj[0] if isinstance(proj, nn.Sequential) else proj).in_features


def resize_projector_input(model, in_features: int, seed: int) -> dict:
    """Give the projector `in_features` inputs, in place: only its first Linear is replaced, by a fresh nn.Linear with
    the same outputs and bias (PyTorch's default init, as BIT builds it), under a forked RNG seeded with `seed`. An
    MLP projector keeps its activation and second Linear exactly as the model built them."""
    proj = model.projector
    old = proj[0] if isinstance(proj, nn.Sequential) else proj
    device = old.weight.device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        new = nn.Linear(in_features, old.out_features, bias=old.bias is not None).to(device)
    if isinstance(proj, nn.Sequential):
        proj[0] = new
    else:
        model.projector = new
    return {"in_features": in_features, "seed": seed, "replaced": f"Linear({old.in_features}, {old.out_features})",
            "kept": [type(m).__name__ for m in list(proj)[1:]] if isinstance(proj, nn.Sequential) else []}


def build_module(e: dict) -> tuple[nn.Module, str | None, dict]:
    """The module between encoder and projector for registry entry `e`, the digest of the encoder its artifacts were
    fit on (None for `frozen`), and a report of the artifacts (file, sha256, ...)."""
    if e["kind"] == "frozen":
        return Passthrough(), None, {}
    if e["kind"] == "phoneme_bottleneck":
        stats = torch.load(resolve(e["stats"]), map_location="cpu", weights_only=True)
        module = PhonemeReadout.from_checkpoint(resolve(e["phoneme_model"]), feature=e["feature"], floor=e["floor"],
                                                mean=stats.get("mean"), sd=stats.get("sd"))
        if stats["feature"] != e["feature"] or stats["floor"] != e["floor"]:
            raise ValueError(f"{e['stats']}: computed for {stats['feature']} / floor {stats['floor']}")
        if state_digest({"weight": module.weight, "bias": module.bias}) != stats["readout_digest"]:
            raise ValueError(f"{e['stats']}: computed with a different read-out than {e['phoneme_model']}")
        return module, stats["encoder_digest"], {"stats": {"file": e["stats"], "sha256": file_digest(resolve(e["stats"])),
                                                           "n_frames": stats["n_frames"]}}
    eraser = torch.load(resolve(e["eraser"]), map_location="cpu", weights_only=True)
    if list(eraser["offsets"]) != list(e.get("offsets", [0])):
        raise ValueError(f"{e['eraser']}: fit for offsets {eraser['offsets']}, the entry has {e.get('offsets')}")
    return LinearEraser(eraser["A"], eraser["b"]), eraser["encoder_digest"], {
        "eraser": {"file": e["eraser"], "sha256": file_digest(resolve(e["eraser"])), "rank": eraser["rank"],
                   "offsets": eraser["offsets"], "n_frames": eraser["n_frames"]}}


def apply(model, name: str, init: Path | None, seed: int) -> dict:
    """Install ablation `name` in a freshly built Stage 2 model (after stage2.fix_padding and stage2.load_init), in
    place. init = the --init directory (None with --eval-only: the trained checkpoint is loaded afterwards; call
    `verify` then). Returns the report for run.json."""
    e = entry(name)
    enc = model.neural_encoder.encoder
    rep = {"name": name, "kind": e["kind"], "entry": e}
    if init is not None:
        if not (e.get("phoneme_model") or e.get("encoder")):
            raise ValueError(f"--ablation {name} names no encoder: it is for --eval-only only")
        source = resolve(e.get("phoneme_model") or e["encoder"])
        if Path(init).resolve() != source.resolve():
            raise ValueError(f"--ablation {name} expects --init {source}, got {init}")
        rep |= _check_from_init(enc, Path(init))
    for p in enc.parameters():
        p.requires_grad_(False)
    module, expect, artifacts = build_module(e)
    rep |= artifacts
    device = next(enc.parameters()).device
    model.neural_ablation = module.to(device)
    width = module.out_features if isinstance(module, PhonemeReadout) else enc.hidden_size
    if projector_in_features(model.projector) != width:
        rep["projector_resized"] = resize_projector_input(model, width, seed)
    model.prepare_neural_embeds = types.MethodType(ablated_prepare_neural_embeds, model)
    model._ablation = {"name": name, "expect_encoder_digest": expect,
                       "encoder_digest_at_start": state_digest(enc.state_dict()) if init is not None else None}
    if init is not None and expect is not None and model._ablation["encoder_digest_at_start"] != expect:
        raise ValueError(f"{name}: the artifacts were fit on a different encoder than {init}")
    rep["encoder_digest"] = model._ablation["encoder_digest_at_start"]
    rep["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rep["projector_in_features"] = width
    return rep


def verify(model) -> dict:
    """After a checkpoint is loaded: the encoder is still frozen and is the one the artifacts were fit on (and, after
    training, the one training started from)."""
    info, enc = model._ablation, model.neural_encoder.encoder
    digest = state_digest(enc.state_dict())
    rep = {"encoder_digest": digest, "frozen": not any(p.requires_grad for p in enc.parameters())}
    for key in ("expect_encoder_digest", "encoder_digest_at_start"):
        if info[key] is not None:
            rep[key.replace("_digest", "") + "_matches"] = info[key] == digest
            if info[key] != digest:
                raise RuntimeError(f"ablation {info['name']}: encoder digest {digest[:12]} != {key} {info[key][:12]}")
    if not rep["frozen"]:
        raise RuntimeError(f"ablation {info['name']}: encoder parameters became trainable")
    return rep
