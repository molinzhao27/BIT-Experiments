"""BIT masked-modeling SSL pretraining (Phase 4b) on the human corpus of `ssl_data`.

Shared step for every Phase 4 arm (DESIGN.md §0, tier 1). Plan and BIT references: docs/phase4/SSL_PLAN.md.
BIT = external/bit (github.com/yzhang511/bit @ f107ae9), read-only. Used unchanged from BIT: `NDT` with
`method_name: ssl` (masking, MSE on masked unpadded bins, read-in/read-out per stitch group), `SpikingDataset`,
`GroupBatchSampler`, `pad_collate_fn`, `update_config` / `default_trainer_config`, `NDT.save_checkpoint`, and the
value of `eval_neurons_metric` (logged for reference). Ours:

- `patch_stitch_tables`: BIT's `DATASET_TO_IDX` / `DATASET_CHANNELS[feature]` replaced in place from the manifest.
- `multi_spiking_dataset`: `SpikingDataset` whose read-in index is each record's `stitch_idx`.
- `evaluate`: R^2 on masked, unpadded bins with fixed (seeded) masks, pooled per stitch group.
- `SSLPretrainer`: the loop of BIT's `Trainer.train` (same optimizer, schedule, autocast, step order and
  checkpoint files) without accelerate / deepspeed / wandb / the LLM imports, plus a per-epoch log.
- `export_core` / `load_core`: checkpoints whose transformer core loads into a fine-tuning NDT with a
  different read-in shape (TX+SBP), which BIT's own `from_pt` path cannot do (size mismatch error).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kunzdecoder import ssl_data

ROOT = Path(__file__).resolve().parents[2]
BIT_DIR = ROOT / "external" / "bit"
PAD_VALUE = -100  # BIT's spike padding value (utils/datasets.py PAD_VALUE)
PAD_DICT = {  # SSL_PLAN.md §3.2; the same entries as BIT's phoneme trainer.yaml
    "spikes": [{"dim": 0, "side": "right", "value": PAD_VALUE, "truncate": None, "min_length": None, "max_length": None}],
    "spikes_mask": [{"dim": 0, "side": "right", "value": 0, "truncate": None, "min_length": None, "max_length": None}],
    "spikes_timestamp": [{"dim": 0, "side": "right", "value": 0, "truncate": None, "min_length": None,
                          "max_length": None}],
}
# Our stitch groups -> (BIT's dataset, its fixed DATASET_TO_IDX slot at f107ae9). Fine-tuning loads read-ins by
# position, so these groups must keep BIT's slots
BIT_SLOTS = {"b2t25_T15": ("brandman_2024_text", 0), "b2t24_T12": ("willett_2023_text", 1)}
READ_IN = "embedder.read_in.embed_dict."  # NeuralEncoder state-dict prefix of the per-group read-ins
READ_OUT = "read_out."  # DecoderWrapper state-dict prefix of the per-group read-outs
CKPT_FILES = ("encoder.bin", "encoder_config.pth", "decoder.bin", "trainer_config.pth", "model.safetensors",
              "epoch.pth")


# ---------------------------------------------------------------- BIT imports

@contextmanager
def bit_cwd(bit_dir: Path = BIT_DIR):
    """BIT on sys.path and as cwd (it opens configs/*.yaml by relative path)."""
    bit_dir = str(Path(bit_dir).resolve())
    if bit_dir not in sys.path:
        sys.path.insert(0, bit_dir)
    old = os.getcwd()
    os.chdir(bit_dir)
    try:
        yield
    finally:
        os.chdir(old)


def import_bit(bit_dir: Path = BIT_DIR) -> SimpleNamespace:
    """BIT's modules (imported once per process; `models.ndt` holds the same table objects as `utils.datasets`)."""
    with bit_cwd(bit_dir):
        import models.ndt as ndt
        import utils.config as config
        import utils.datasets as datasets
        import utils.eval_spike as eval_spike
    return SimpleNamespace(ndt=ndt, datasets=datasets, config=config, eval_spike=eval_spike, dir=Path(bit_dir))


# ---------------------------------------------------------------- stitch tables

def patch_stitch_tables(manifest: dict, feature: str = "tx1", bit: SimpleNamespace | None = None) -> dict:
    """Replace BIT's `DATASET_TO_IDX` and `DATASET_CHANNELS[feature]` in place with the corpus stitch groups.

    Other features' tables are kept. Returns the previous tables (for `restore_stitch_tables`). Raises if a
    group in `BIT_SLOTS` would not get the index of its BIT dataset (b2t25_T15 = 0, b2t24_T12 = 1)."""
    bit = bit or import_bit()
    D = bit.datasets
    to_idx, channels = ssl_data.bit_stitch_tables(manifest, feature)
    if list(to_idx.values()) != list(range(len(to_idx))) or list(to_idx) != list(channels[feature]):
        raise ValueError(f"stitch tables out of order: {to_idx} / {channels[feature]}")
    for ours, (theirs, slot) in BIT_SLOTS.items():
        if ours in to_idx and to_idx[ours] != slot:
            raise ValueError(f"{ours} has read-in slot {to_idx[ours]}, BIT's {theirs} has {slot}")
    prev = {"to_idx": dict(D.DATASET_TO_IDX), "channels": copy.deepcopy(D.DATASET_CHANNELS)}
    D.DATASET_TO_IDX.clear()
    D.DATASET_TO_IDX.update(to_idx)
    D.DATASET_CHANNELS.setdefault(feature, {})
    D.DATASET_CHANNELS[feature].clear()
    D.DATASET_CHANNELS[feature].update(channels[feature])
    assert bit.ndt.DATASET_CHANNELS is D.DATASET_CHANNELS  # NDT reads the patched object at construction
    return prev


def restore_stitch_tables(prev: dict, bit: SimpleNamespace | None = None) -> None:
    bit = bit or import_bit()
    D = bit.datasets
    D.DATASET_TO_IDX.clear()
    D.DATASET_TO_IDX.update(prev["to_idx"])
    for feature in list(D.DATASET_CHANNELS):
        if feature not in prev["channels"]:
            del D.DATASET_CHANNELS[feature]
    for feature, table in prev["channels"].items():
        D.DATASET_CHANNELS.setdefault(feature, {})
        D.DATASET_CHANNELS[feature].clear()
        D.DATASET_CHANNELS[feature].update(table)


@contextmanager
def stitch_tables(manifest: dict, feature: str = "tx1", bit: SimpleNamespace | None = None):
    prev = patch_stitch_tables(manifest, feature, bit)
    try:
        yield
    finally:
        restore_stitch_tables(prev, bit)


# ---------------------------------------------------------------- dataset / loaders

_DATASET_CLASS = {}


def multi_spiking_dataset(bit: SimpleNamespace | None = None):
    """BIT's `SpikingDataset` with `dataset_name` (the read-in/read-out index) taken from each record."""
    bit = bit or import_bit()
    base = bit.datasets.SpikingDataset
    if base not in _DATASET_CLASS:
        class MultiSpikingDataset(base):
            """Items as `SpikingDataset.__getitem__`, but `dataset_name` = record `stitch_idx`. The record is copied
            shallowly (BIT deep-copies): nothing downstream writes into the arrays (pad_collate_fn copies), and
            `GroupBatchSampler` reads every item once at construction."""

            def __getitem__(self, idx):
                inputs = dict(self.dataset[idx])
                spikes = inputs.pop(self.spikes_name)
                inputs.update({
                    "spikes": spikes,
                    "spikes_mask": np.ones(spikes.shape[0], dtype=np.int64),
                    "spikes_timestamp": np.arange(0, spikes.shape[0]),
                    "spikes_spacestamp": np.arange(0, spikes.shape[1]),
                    "spikes_lengths": np.asarray(spikes.shape[0]),
                    "dataset_name": np.array(int(inputs["stitch_idx"])),
                })
                return inputs

        _DATASET_CLASS[base] = MultiSpikingDataset
    return _DATASET_CLASS[base]


def make_loader(bit: SimpleNamespace, dataset, batch_size: int, shuffle: bool, seed: int, model_inputs,
                drop_last: bool = False, pin_memory: bool = False) -> DataLoader:
    """BIT's loader construction (models/trainer.py:199-227): one stitch group per batch."""
    sampler = bit.datasets.GroupBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                             seed=seed)
    collate = partial(bit.datasets.pad_collate_fn, model_inputs=list(model_inputs), pad_dict=PAD_DICT)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, pin_memory=pin_memory)


def limit_records(data: dict, n: int | None) -> dict:
    """First `n` records per (split, stitch group); None keeps everything."""
    if n is None:
        return data
    out = {}
    for split, recs in data.items():
        seen = defaultdict(int)
        out[split] = []
        for r in recs:
            if seen[r["stitch"]] < n:
                seen[r["stitch"]] += 1
                out[split].append(r)
    return out


# ---------------------------------------------------------------- config

def bit_config(pretrain: dict, seed: int, first_group: str, bit: SimpleNamespace | None = None) -> dict:
    """BIT trainer config (configs/trainer.yaml layout) from ssl_human.yaml `pretrain`. The encoder section only
    overrides BIT's configs/models/encoder/ndt.yaml (NDT merges it); dataset defaults are not applied, so the
    masker stays on (SSL_PLAN.md §1.2 item 3)."""
    bit = bit or import_bit()
    cfg = {
        "seed": seed, "savestring": pretrain.get("savestring", "ssl-ndt-human-tx"), "log_to_wandb": False,
        "verbosity": 0,
        "training": {"shuffle_test_dataloader": False, "drop_last_train_dataloader": False,
                     **pretrain["training"]},
        "model": {"model_class": "NDT", "encoder": {**copy.deepcopy(pretrain["model"]), "from_pt": None},
                  "decoder": {"from_pt": None}},
        "data": {"dataset_class": "ssl_multi", "train_name": "train", "val_name": "val", "train_len": None,
                 "val_len": None},
        "method": {"model_kwargs": dict(pretrain["method"]),
                   "dataset_kwargs": {"dataset_name": first_group},  # SpikingDataset.__init__ only; per record
                   "metric_kwargs": {"metric_name": pretrain["selection"]["metric"]},
                   "dataloader_kwargs": {"pad_dict": copy.deepcopy(PAD_DICT)}},
        "optimizer": dict(pretrain["optimizer"]),
        "features": pretrain.get("features", "tx1"),
        "selection": copy.deepcopy(pretrain["selection"]),
    }
    with bit_cwd(bit.dir):
        merged = bit.config.update_config(bit.config.default_trainer_config(), cfg)
    return _plain(merged)


def _plain(x):
    """Nested plain dicts/lists (BIT's DictConfig is a dict subclass; torch.load(weights_only=True) needs plain)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def build_ndt(cfg: dict, bit: SimpleNamespace | None = None):
    """BIT's NDT for the (already patched) stitch tables, as `Trainer.set_model` builds it."""
    bit = bit or import_bit()
    with bit_cwd(bit.dir):
        return bit.ndt.NDT(copy.deepcopy(cfg["model"]), **cfg["method"]["model_kwargs"],
                           **cfg["method"]["dataset_kwargs"], features=cfg["features"])


def model_inputs(model) -> list[str]:
    import inspect
    return list(inspect.signature(model.forward).parameters)


# ---------------------------------------------------------------- fixed-mask evaluation

def _bit_masker(emb, batch_size: int, num_patches: int, device, generator: torch.Generator) -> torch.Tensor:
    """BIT's `NeuralEmbeddingLayer.masker` (models/ndt.py:249-277), drawing from `generator` (CPU)."""
    if torch.bernoulli(torch.tensor(float(emb.expand_prob)), generator=generator):
        span = int(torch.randint(1, emb.max_timespan + 1, (1,), generator=generator))
    else:
        span = 1
    n_spans = int(emb.mask_ratio / span * num_patches)
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    for b in range(batch_size):
        for s in torch.randperm(num_patches - span + 1, generator=generator)[:n_spans].tolist():
            mask[b, s:s + span] = True
    return mask.to(device)


@contextmanager
def fixed_masks(model, seed: int):
    """Swap the model's masker for BIT's algorithm on a CPU generator; `reseed(group, k)` before batch k of a group.
    Masks then depend only on (seed, group, batch position, batch shape): not on the global RNG or the device."""
    emb = model.encoder.embedder
    gen = torch.Generator()
    emb.masker = lambda batch_size, num_patches, device: _bit_masker(emb, batch_size, num_patches, device, gen)
    try:
        yield SimpleNamespace(reseed=lambda group, k: gen.manual_seed(int(seed) * 1_000_003 + int(group) * 10_007 + k))
    finally:
        del emb.masker


def bit_r2(targets: torch.Tensor, preds: torch.Tensor, pad_value: float = PAD_VALUE) -> float:
    """The value BIT's `eval_neurons_metric` returns for loss mse (utils/eval_spike.py:146-171): sklearn's r2_score
    pooled over ALL bins (masked and unmasked, padding zeroed) of the 1,000 most active channels. BIT computes the
    same pooled score once per channel (on CPU) and averages the identical copies; here it is computed once, in
    float64 on the tensors' device, with sklearn's convention for a zero denominator."""
    pad = targets == pad_value
    gt = targets.masked_fill(pad, 0).double()
    pr = preds.masked_fill(pad, 0).double()
    idx = torch.topk(gt.abs().sum(dim=(0, 1)), min(1_000, gt.shape[-1])).indices
    gt, pr = gt[..., idx], pr[..., idx]
    ss_res = ((gt - pr) ** 2).sum().item()
    ss_tot = ((gt - gt.mean()) ** 2).sum().item()
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _to(inputs: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}


def _fill(v: torch.Tensor, avail: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry the last available value of v (B, T, C) forward in time (backward if `reverse`); also return where one
    exists."""
    if reverse:
        v, avail = v.flip(1), avail.flip(1)
    t = torch.arange(v.shape[1], device=v.device).view(1, -1, 1).expand_as(v)
    idx = torch.where(avail, t, torch.full_like(t, -1)).cummax(dim=1).values
    out, ok = v.gather(1, idx.clamp(min=0)), idx >= 0
    return (out.flip(1), ok.flip(1)) if reverse else (out, ok)


def trivial_predictions(y: torch.Tensor, masked: torch.Tensor, valid: torch.Tensor, width: int = 5) -> dict:
    """Reference predictors for the masked bins, built from the same trial's UNMASKED bins only (no learning):
    `trial_mean`: each channel's mean over the trial's unmasked bins.
    `neighbors`: the mean of the nearest unmasked `width`-bin (100 ms) stretches before and after the gap, each
    channel separately; one side at the trial edges; the trial mean if the trial has no unmasked bin."""
    seen = valid & ~masked
    w = seen.to(y.dtype)
    cnt = w.sum(1, keepdim=True)
    trial_mean = (y * w).sum(1, keepdim=True) / cnt.clamp(min=1)
    box = lambda x: F.avg_pool1d(x.transpose(1, 2), width, 1, width // 2, count_include_pad=False).transpose(1, 2)  # noqa: E731
    local = box(y * w) / box(w).clamp(min=1e-9)
    left, has_l = _fill(local, seen)
    right, has_r = _fill(local, seen, reverse=True)
    both = has_l.to(y.dtype) + has_r.to(y.dtype)
    nb = (left * has_l + right * has_r) / both.clamp(min=1)
    nb = torch.where(both > 0, nb, trial_mean.expand_as(y))
    return {"trial_mean": trial_mean.expand_as(y), "neighbors": nb}


@torch.no_grad()
def evaluate(model, loader: DataLoader, groups: list[str], mask_seed: int, device, dtype=torch.float32,
             selection_groups: list[str] | None = None, baselines: bool = False) -> dict:
    """Masked-bin reconstruction metrics with fixed masks.

    Per stitch group g, over all masked, unpadded bins b (the bins BIT's SSL loss uses) and channels c of the split:
    `r2_masked_<g>` = 1 - sum_bc (y - yhat)^2 / sum_c sum_b (y - mean_b y_c)^2 (variance-weighted over channels),
    `mse_masked_<g>` = sum (y - yhat)^2 / (#bins x #channels). `r2_masked` = equal-weight mean over
    `selection_groups` that are present. `bit_r2` = BIT's metric averaged over batches (as BIT's Trainer does),
    on the same forward passes (fixed masks), also per group. With `baselines`, the same R^2 for the
    `trivial_predictions` on the same bins: `r2_masked_<g>_<baseline>`."""
    model.eval()
    acc = {}
    bit_vals = defaultdict(list)
    count = defaultdict(int)
    with fixed_masks(model, mask_seed) as fm:
        for inputs, _ in loader:
            g = int(inputs["dataset_name"].unique().item())
            fm.reseed(g, count[g])
            count[g] += 1
            with torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype != torch.float32):
                out = model(**_to(inputs, device))
            y, p = out.targets.double(), out.preds.double()
            m = out.mask.bool()
            a = acc.setdefault(g, {"n": 0.0, "sy": 0.0, "syy": 0.0, "res": 0.0})
            a["n"] = a["n"] + m.sum(dim=(0, 1))
            a["sy"] = a["sy"] + (y * m).sum(dim=(0, 1))
            a["syy"] = a["syy"] + (y * y * m).sum(dim=(0, 1))
            a["res"] = a["res"] + ((p - y) ** 2 * m).sum(dim=(0, 1))
            if baselines:
                for k, q in trivial_predictions(y, m, out.targets != PAD_VALUE).items():
                    a[f"res_{k}"] = a.get(f"res_{k}", 0.0) + ((q - y) ** 2 * m).sum(dim=(0, 1))
            bit_vals[g].append(bit_r2(out.targets, out.preds))
    res = {}
    for g, a in sorted(acc.items()):
        name = groups[g]
        n = a["n"].clamp(min=1)
        ss_tot = (a["syy"] - a["sy"] ** 2 / n).sum().item()
        ss_res = a["res"].sum().item()
        res[f"r2_masked_{name}"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        res[f"mse_masked_{name}"] = ss_res / max(1.0, a["n"].sum().item())
        res[f"n_bins_{name}"] = int(a["n"][0].item())
        res[f"bit_r2_{name}"] = float(np.nanmean(bit_vals[g]))
        for k in (k for k in a if k.startswith("res_")):
            res[f"r2_masked_{name}_{k[4:]}"] = 1.0 - a[k].sum().item() / ss_tot if ss_tot > 0 else float("nan")
    sel = [g for g in (selection_groups or [groups[g] for g in sorted(acc)]) if f"r2_masked_{g}" in res]
    res["r2_masked"] = float(np.mean([res[f"r2_masked_{g}"] for g in sel])) if sel else float("nan")
    res["selection_groups"] = sel
    all_bit = [v for g in sorted(bit_vals) for v in bit_vals[g]]
    res["bit_r2"] = float(np.nanmean(all_bit)) if all_bit else float("nan")
    return res


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(model, cfg: dict, ckpt_dir: Path, epoch: int, extra: dict | None = None,
                    optimizer=None, scheduler=None) -> Path:
    """BIT's checkpoint directory (Trainer.save_checkpoint + accelerator.save_state's model file):
    encoder.bin, encoder_config.pth, decoder.bin (NDT.save_checkpoint), trainer_config.pth, model.safetensors
    (whole-NDT state dict, read by BCI_AudioLLM.load_safetensors), epoch.pth; plus stitch_tables.json and
    metrics.json. Written to a temporary directory, then renamed."""
    from safetensors.torch import save_file
    ckpt_dir = Path(ckpt_dir)
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_checkpoint(str(tmp))
    torch.save(_plain(cfg), tmp / "trainer_config.pth")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(tmp / "model.safetensors"))
    torch.save(epoch, tmp / "epoch.pth")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.bin")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp / "scheduler.bin")
    (tmp / "metrics.json").write_text(json.dumps({"epoch": epoch, **(extra or {})}, indent=1, default=str))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    tmp.rename(ckpt_dir)
    return ckpt_dir


def export_core(src: Path, dst: Path) -> dict:
    """Copy of checkpoint `src` without read-in and read-out weights. BIT's unmodified `--ft_ckpt dst` (NDT
    `from_pt`, strict=False) then loads the transformer core and mask token and leaves every read-in/read-out at its
    fresh initialization, whatever the fine-tuning channel counts (e.g. TX+SBP = 512 for T15)."""
    from safetensors.torch import load_file, save_file
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("optimizer.bin", "scheduler.bin"))
    removed = {}
    enc = torch.load(src / "encoder.bin", map_location="cpu", weights_only=True)
    removed["encoder.bin"] = sorted(k for k in enc if k.startswith(READ_IN))
    torch.save({k: v for k, v in enc.items() if not k.startswith(READ_IN)}, tmp / "encoder.bin")
    dec = torch.load(src / "decoder.bin", map_location="cpu", weights_only=True)
    removed["decoder.bin"] = sorted(k for k in dec if k.startswith(READ_OUT))
    torch.save({k: v for k, v in dec.items() if not k.startswith(READ_OUT)}, tmp / "decoder.bin")
    full = load_file(str(src / "model.safetensors"))
    drop = lambda k: k.startswith("encoder." + READ_IN) or k.startswith("decoder." + READ_OUT)  # noqa: E731
    removed["model.safetensors"] = sorted(k for k in full if drop(k))
    save_file({k: v for k, v in full.items() if not drop(k)}, str(tmp / "model.safetensors"))
    info = {"source": str(src), "removed": removed}
    (tmp / "core.json").write_text(json.dumps(info, indent=1))
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return info


def export_finetune_core(core: Path, dst: Path, embedder: dict) -> dict:
    """Copy of a core checkpoint for BIT's `--ft_ckpt`, with the fine-tuning task's embedder settings.

    BIT's NDT merges the checkpoint's encoder_config.pth over the task config (models/ndt.py, `from_pt`), so a
    core exported from SSL would switch the SSL masker (active, ratio 0.5) back on during CTC fine-tuning, where
    `force_mask = self.training`. Writing the task's own embedder entries (e.g. BIT's registry `embedder_defaults`
    for brandman_2024_text: masker inactive) makes SSL-initialized fine-tuning differ from training from scratch
    only in its initial weights. The weights are copied unchanged."""
    core, dst = Path(core), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(core, tmp)
    cfg = torch.load(core / "encoder_config.pth", map_location="cpu", weights_only=False)
    before = copy.deepcopy(cfg["embedder"])
    cfg["embedder"].update(copy.deepcopy(embedder))
    torch.save(cfg, tmp / "encoder_config.pth")
    info = {"source": str(core), "embedder_before": _plain(before), "embedder_after": _plain(cfg["embedder"])}
    (tmp / "finetune_core.json").write_text(json.dumps(info, indent=1, default=str))
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return info


def load_core(model, ckpt_dir: Path, exclude_read_in: bool = False, verbose: bool = True) -> dict:
    """Load `encoder.bin` into an NDT (or its NeuralEncoder), keeping every parameter whose shape matches.

    A read-in slot (`embedder.read_in.embed_dict.<i>.*`) is loaded whole or not at all: if any of its tensors has a
    different shape (TX-only checkpoint, TX+SBP model) the slot keeps its initialization. With `exclude_read_in`,
    no read-in is loaded. Returns {"loaded", "reinit_read_in", "skipped_shape", "unexpected", "missing"}."""
    target = model.encoder if hasattr(model, "decoder") and hasattr(model, "encoder") else model
    state = torch.load(Path(ckpt_dir) / "encoder.bin", map_location="cpu", weights_only=True)
    own = target.state_dict()
    slot = lambda k: k[len(READ_IN):].split(".", 1)[0] if k.startswith(READ_IN) else None  # noqa: E731
    bad_slots = {slot(k) for k, v in state.items() if slot(k) is not None and k in own and own[k].shape != v.shape}
    rep = {"loaded": [], "reinit_read_in": sorted(bad_slots), "skipped_shape": [], "unexpected": [], "missing": []}
    keep = {}
    for k, v in state.items():
        s = slot(k)
        if k not in own:
            rep["unexpected"].append(k)
        elif s is not None and (exclude_read_in or s in bad_slots):
            continue
        elif own[k].shape != v.shape:
            rep["skipped_shape"].append(f"{k}: {tuple(v.shape)} -> {tuple(own[k].shape)}")
        else:
            keep[k] = v
    if exclude_read_in:
        rep["reinit_read_in"] = sorted({slot(k) for k in own if slot(k) is not None})
    target.load_state_dict(keep, strict=False)
    rep["loaded"], rep["unexpected"] = sorted(keep), sorted(rep["unexpected"])
    rep["missing"] = sorted(k for k in own if k not in keep)
    if verbose:
        print(f"load_core({ckpt_dir}): {len(keep)} tensors loaded; read-in slots re-initialized "
              f"{rep['reinit_read_in']}; {len(rep['unexpected'])} unexpected; {len(rep['skipped_shape'])} "
              f"other shape mismatches {rep['skipped_shape']}")
    return rep


# ---------------------------------------------------------------- training

def set_seed(seed: int) -> None:
    """Trainer.set_seed (models/trainer.py:98-103) plus numpy / random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device: str = "auto") -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)


class SSLPretrainer:
    """BIT `Trainer.train` for NDT SSL on one device, with fixed-mask selection and a per-epoch log.

    Same as BIT: AdamW(lr, wd, eps) on all parameters; `OneCycleLR` cosine over num_epochs x batches per epoch
    (max_lr = lr, pct_start = warmup_pct, div_factor), stepped every batch; loss = BIT's summed masked MSE,
    backward per batch; torch.autocast(bf16) for the forward; GroupBatchSampler(seed) batches; eval every
    `eval_every`; BEST on improvement of the selection metric; EPOCH<n> every `save_every`; LAST at the end."""

    def __init__(self, cfg: dict, data: dict, manifest: dict, out_dir: Path, device="auto",
                 bit: SimpleNamespace | None = None):
        self.bit = bit or import_bit()
        self.cfg, self.manifest, self.out_dir = cfg, manifest, Path(out_dir)
        self.groups = list(manifest["stitch_groups"])
        self.device = resolve_device(device)
        mp = cfg["training"].get("mixed_precision")
        self.dtype = torch.bfloat16 if (mp == "bf16" and self.device.type == "cuda") else torch.float32
        if cfg["optimizer"].get("gradient_accumulation_steps", 1) != 1:
            raise NotImplementedError("gradient_accumulation_steps != 1 (BIT's Table-10 run uses 1)")
        set_seed(cfg["seed"])
        self.model = build_ndt(cfg, self.bit).to(self.device)
        self.inputs = model_inputs(self.model)
        Dataset = multi_spiking_dataset(self.bit)
        kw = cfg["method"]["dataset_kwargs"]
        self.datasets = {s: Dataset(data, s, **kw) for s in data}
        tr, pin = cfg["training"], self.device.type == "cuda"
        self.train_loader = make_loader(self.bit, self.datasets["train"], tr["train_batch_size"], True, cfg["seed"],
                                        self.inputs, tr.get("drop_last_train_dataloader", False), pin)
        self.eval_loaders = {s: make_loader(self.bit, self.datasets[s], tr["test_batch_size"], False, cfg["seed"],
                                            self.inputs, False, pin) for s in self.datasets}
        opt = cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt["lr"], weight_decay=opt["wd"],
                                           eps=opt["eps"])
        if opt["scheduler"] != "cosine":
            raise NotImplementedError(f"scheduler {opt['scheduler']!r} (Table 10: cosine)")
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, total_steps=tr["num_epochs"] * len(self.train_loader), max_lr=opt["lr"],
            pct_start=opt["warmup_pct"], div_factor=opt["div_factor"])
        self.log: list[dict] = []

    def evaluate(self, split: str = "val") -> dict:
        sel = self.cfg["selection"]
        return evaluate(self.model, self.eval_loaders[split], self.groups, sel["eval_mask_seed"], self.device,
                        self.dtype, sel.get("val_groups"))

    def _write_log(self) -> None:
        (self.out_dir / "log.json").write_text(json.dumps(self.log, indent=1))
        cols = list(dict.fromkeys(k for row in self.log for k in row))
        with open(self.out_dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: (";".join(v) if isinstance(v, list) else v) for k, v in row.items()} for row in self.log])

    def _save(self, name: str, epoch: int, metrics: dict, state: bool = False) -> None:
        save_checkpoint(self.model, self.cfg, self.out_dir / name, epoch, metrics,
                        self.optimizer if state else None, self.scheduler if state else None)
        (self.out_dir / name / "stitch_tables.json").write_text(json.dumps(
            {"feature": self.cfg["features"], "DATASET_TO_IDX": {g: i for i, g in enumerate(self.groups)},
             "DATASET_CHANNELS": {g: self.manifest["stitch_groups"][g]["n_channels"] for g in self.groups}}, indent=1))

    def train(self) -> dict:
        tr = self.cfg["training"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if "val" not in self.datasets or not len(self.datasets["val"]):
            raise ValueError("corpus has no val split: the selection metric needs the B2T val halves")
        n_params = sum(p.numel() for p in self.model.parameters())
        core = sum(p.numel() for n, p in self.model.named_parameters() if "embed_dict" not in n and "decoder" not in n)
        (self.out_dir / "run.json").write_text(json.dumps({
            "config": self.cfg, "device": str(self.device), "autocast": str(self.dtype), "n_params": n_params,
            "n_core_params": core, "stitch_groups": self.manifest["stitch_groups"],
            "n_records": {s: len(d) for s, d in self.datasets.items()}, "batches_per_epoch": len(self.train_loader),
            "torch": torch.__version__}, indent=1, default=str))
        best, epoch, metrics = -math.inf, 0, {}
        for epoch in range(1, tr["num_epochs"] + 1):
            t0 = time.time()
            self.model.train()
            loss_sum, n_sum = 0.0, 0
            for inputs, _ in self.train_loader:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                                    enabled=self.dtype != torch.float32):
                    out = self.model(**_to(inputs, self.device))
                out.loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                loss_sum += out.loss.detach().item()
                n_sum += int(out.n_examples.item())
            row = {"epoch": epoch, "train_loss": loss_sum / n_sum if n_sum else float("nan"),
                   "lr": self.scheduler.get_last_lr()[0], "train_seconds": round(time.time() - t0, 2)}
            if tr.get("eval_every") and epoch % tr["eval_every"] == 0:
                metrics = self.evaluate("val")
                row |= {f"val_{k}": v for k, v in metrics.items()}
                row["best"] = bool(metrics["r2_masked"] > best)
                if row["best"]:
                    best = metrics["r2_masked"]
                    self._save("BEST", epoch, metrics)
                    export_core(self.out_dir / "BEST", self.out_dir / "BEST_core")
            if self.device.type == "cuda":
                row["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            row["seconds"] = round(time.time() - t0, 2)
            self.log.append(row)
            self._write_log()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                              if not k.startswith("val_n_bins")}), flush=True)
            if tr.get("save_every") and epoch % tr["save_every"] == 0:
                self._save(f"EPOCH{epoch}", epoch, metrics)
        self._save("LAST", epoch, metrics, state=True)
        return {"best_r2_masked": best, "epochs": epoch, "out_dir": str(self.out_dir)}


def build_trainer(cfg: dict, corpus_dir: Path, out_dir: Path, seed: int, *, device: str = "auto",
                  max_epochs: int | None = None, limit: int | None = None,
                  bit: SimpleNamespace | None = None) -> SSLPretrainer:
    """Pretrainer for ssl_human.yaml (whole file or its `pretrain` section) on the corpus in `corpus_dir`.

    Patches BIT's stitch tables for this process (they stay patched; NDT and the datasets read them).
    `max_epochs` replaces num_epochs (the cosine schedule spans the shortened run, as BIT's would);
    `limit` keeps the first N records per (split, stitch group)."""
    bit = bit or import_bit()
    pretrain = copy.deepcopy(cfg.get("pretrain", cfg))
    if max_epochs is not None:
        pretrain["training"]["num_epochs"] = max_epochs
    data, manifest = ssl_data.load_corpus(Path(corpus_dir))
    data = limit_records(data, limit)
    patch_stitch_tables(manifest, pretrain.get("features", "tx1"), bit)
    run_cfg = bit_config(pretrain, seed, next(iter(manifest["stitch_groups"])), bit)
    run_cfg["corpus_dir"] = str(Path(corpus_dir).resolve())
    return SSLPretrainer(run_cfg, data, manifest, out_dir, device, bit)
