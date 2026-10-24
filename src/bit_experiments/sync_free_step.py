"""The packed block-diffusion step without host waits (fast_step + sync_free: true; docs/DIFFUSION_SPEEDUP.md "Host
waits"). It gives fast_step's masks, loss, logits and gradients bit for bit, and makes the same random draws in the
same order, so the generator's state after a step is the same too. Only where the CPU waits for the GPU changes.

The fast step (BCI_DiffusionLLM.forward, then fast_block_diffusion_ce) makes the CPU wait for the GPU about 45 times
per step of 8 trials:
- per trial, int() and bool() of GPU tensors, and boolean indexing of the neural tokens (5 each);
- the pack plan's labels.cpu();
- blocking copies of small id tensors to the GPU (torch.tensor(..., device=...));
- the boolean selection of the decoded outputs. It sits after the decoder, so the CPU cannot queue the head, the loss
  and the backward until the whole forward has run.
Each wait empties the GPU's queue, and the GPU then idles while Python works.

This path waits once, right after the encoder: one copy to the CPU of the batch's small integer tensors and of the
encoder's neural-token mask, which exists only on the GPU. Everything that depends on lengths is then computed on
the CPU and sent in non-blocking copies from pinned memory:
- the [B, L] clean sequences, as one gather from [text embeddings ‖ neural tokens ‖ EOS ‖ filler]. These are the
  same rows that forward concatenates trial by trial;
- the labels, the pack plan and the flex block mask. The mask is built at block level: each packed 32-token block has
  one mask value per pair of blocks, so the 128-token tiles follow from a (T/32)² table. This replaces evaluating the
  mask at T² positions on the GPU;
- the decoded outputs. Each target position is masked in exactly one copy, so there are exactly as many as there
  are targets. Which copy masks a target is drawn on the GPU, and a GPU sort of their keys gives the boolean
  selection's order without its wait.
With edit_rate > 0 (T2T), the number of decoded outputs is random, so that case keeps block_diffusion_ce's boolean
selection (one wait after the decoder) and gains only the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, _create_sparse_block_from_block_mask

from models.bci import BCIOutput
from bit_experiments import fast_block_diffusion as fast
from bit_experiments.diffusion_llm import EPS, FAST_STEP_DEFAULTS, LENGTH_BUCKET, block_diffusion_ce


def pack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Integer or bool tensors as one flat int64 tensor (to_device's single copy)."""
    return torch.cat([t.reshape(-1).to(torch.int64) for t in tensors])


def unpack(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    """pack's inverse: each tensor of `like` back as its own contiguous tensor of its shape and dtype."""
    out, start = [], 0
    for t in like:
        out.append(flat[start: start + t.numel()].view(t.shape).to(t.dtype, copy=True))
        start += t.numel()
    return out


def to_device(tensors: list[torch.Tensor], device) -> list[torch.Tensor]:
    """CPU integer or bool tensors on `device`, with one non-blocking copy from pinned memory (no host wait). On the
    CPU they are returned as they are."""
    if torch.device(device).type == "cpu":
        return tensors
    return unpack(pack(tensors).pin_memory().to(device, non_blocking=True), tensors)


@dataclass
class Layout:
    """forward's batch layout, computed on the CPU. gather [B * L] holds, for each clean position, its row in the
    source table [text embeddings (B * Lt rows) ‖ neural tokens (B * Ls) ‖ EOS ‖ filler]. labels is [B, L], as
    forward builds it for block_diffusion_ce."""
    gather: torch.Tensor
    labels: torch.Tensor


def layout(attention_mask, targets, input_ids, input_split, token_mask, block_size: int, eos_id: int,
           align: bool) -> Layout:
    """BCI_DiffusionLLM.forward's per-trial layout (and _prefix and _align's), on CPU tensors: the same rows in the
    same places, the same labels, the same length bucket."""
    b, lt = input_ids.shape
    ls = token_mask.shape[1]
    eos_row, filler_row = b * (lt + ls), b * (lt + ls) + 1
    rows = []
    for i in range(b):
        n_text = int(attention_mask[i].sum())
        is_target = targets[i, :n_text] != -100
        n_prompt = int((~is_target).sum())
        if not bool(is_target[n_prompt:].all()):
            raise ValueError("expected [prompt tokens, sentence + EOS] per trial (BIT's create_llm_labels)")
        split = int(input_split[i])
        text = torch.arange(i * lt, (i + 1) * lt)
        neural = b * lt + i * ls + token_mask[i].bool().nonzero().flatten()
        prefix = torch.cat([text[:split], neural, text[split:n_prompt]])
        n_fill = (1 - prefix.numel()) % block_size
        if align and n_fill:
            prefix = torch.cat([prefix.new_full((n_fill,), filler_row), prefix])
        target = input_ids[i, n_prompt:n_text]
        pad = (-(prefix.numel() + target.numel())) % block_size
        rows.append((prefix, torch.cat([text[n_prompt:n_text], text.new_full((pad,), eos_row)]),
                     torch.cat([target, target.new_full((pad,), eos_id)])))
    length = max(p.numel() + t.numel() for p, t, _ in rows)
    length = -(-length // LENGTH_BUCKET) * LENGTH_BUCKET
    gather, labels = [], []
    for prefix, target_rows, target in rows:
        n_pad = length - prefix.numel() - target_rows.numel()
        gather.append(torch.cat([prefix, target_rows, prefix.new_full((n_pad,), eos_row)]))
        labels.append(torch.cat([target.new_full((prefix.numel(),), -100), target, target.new_full((n_pad,), -100)]))
    return Layout(torch.cat(gather), torch.stack(labels))


def block_mask_tiles(plan: fast.PackPlan, block_size: int):
    """create_block_mask(plan.mask_mod(), None, None, T, T)'s partial and full tile maps ([1, 1, T/128, T/128], int8),
    computed at block level on the CPU. The packed tokens come in whole blocks with one (noisy, blk, ctx, grp) per
    block, and so do the pads, so the mask is constant on each block x block square."""
    t, tile = len(plan), fast.FLEX_TILE
    if tile % block_size or t % tile:
        raise ValueError(f"block size {block_size} and packed length {t} should divide / be multiples of {tile}")
    for f in (plan.noisy, plan.blk, plan.ctx, plan.grp):
        per_block = f.view(-1, block_size)
        if not bool((per_block == per_block[:, :1]).all()):
            raise ValueError("pack plan tables are not constant within blocks")
    rep = torch.arange(0, t, block_size)
    sub = plan.mask_mod()(None, None, rep[:, None], rep[None, :])
    r, n = tile // block_size, t // tile
    sub = sub.view(n, r, n, r).permute(0, 2, 1, 3).reshape(n, n, r * r)
    full = sub.all(-1)
    partial = sub.any(-1) & ~full
    return partial.to(torch.int8)[None, None], full.to(torch.int8)[None, None]


def plan_block_mask(plan_cpu: fast.PackPlan, plan_dev: fast.PackPlan, block_size: int) -> BlockMask:
    """The plan's flex block mask on plan_dev's device, built on the CPU (block_mask_tiles, then torch's own
    _create_sparse_block_from_block_mask, as create_block_mask does), with the same fields as create_block_mask
    would give. Its tensors go over in one non-blocking copy, and its mask_mod reads plan_dev's tables."""
    t = len(plan_cpu)
    host = _create_sparse_block_from_block_mask(block_mask_tiles(plan_cpu, block_size), None, (t, t),
                                                fast.FLEX_TILE, fast.FLEX_TILE)
    names = ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices", "q_num_blocks", "q_indices",
             "full_q_num_blocks", "full_q_indices")
    moved = dict(zip(names, to_device([getattr(host, k) for k in names], plan_dev.src.device)))
    return BlockMask(host.seq_lengths, **moved, BLOCK_SIZE=host.BLOCK_SIZE, mask_mod=plan_dev.mask_mod())


def sync_free_ce(decoder, lm_head, embed, clean: torch.Tensor, labels: torch.Tensor, host_labels: torch.Tensor,
                 block_size: int, mask_id: int, edit_rate: float = 0.0, n_ordinary: int | None = None,
                 share_clean=False, quantum=fast.PACK_QUANTUM, attention: str | None = None):
    """fast_block_diffusion_ce, bit for bit, without host waits (module docstring). host_labels are the labels on the
    CPU, which the layout already knows. With edit_rate > 0 it keeps block_diffusion_ce's selection."""
    (b, length), dev = labels.shape, labels.device
    plan_cpu = fast.pack_plan(host_labels, block_size, share_clean, quantum)
    flex = (attention or ("flex" if dev.type == "cuda" else "dense")) == "flex"
    lookup = torch.full((2 * b * 2 * length,), -1, dtype=torch.long)  # packed token of each computed noisy slot
    lookup[plan_cpu.out_src] = plan_cpu.out_tok
    decoded = host_labels != -100  # target j is decoded from position j - 1, in the copy that masks it
    decoded[:, 0] = False
    target_pos = decoded.flatten().nonzero().flatten()
    fields = ("src", "pos", "noisy", "blk", "ctx", "grp", "out_tok", "out_src")
    moved = to_device([getattr(plan_cpu, f) for f in fields] + [lookup, target_pos], dev)
    plan = fast.PackPlan(**dict(zip(fields, moved[:8])), n_real=plan_cpu.n_real)
    lookup, target_pos = moved[8:]
    block_mask = plan_block_mask(plan_cpu, plan, block_size) if flex else None
    if edit_rate:  # the number of decoded outputs is random: block_diffusion_ce's boolean selection
        return block_diffusion_ce(fast.PackedDecoder(decoder, plan, attention, block_mask), lm_head, embed, clean,
                                  labels, block_size, mask_id, edit_rate, n_ordinary)

    # block_diffusion_ce's draws and noisy copies, op for op (edit_rate 0); the mask row without a host copy
    n_blocks = length // block_size
    ratio = ((1 - EPS) * torch.rand(b * n_blocks, device=dev) + EPS)[:, None].repeat(1, block_size)
    masked = (torch.rand(b * n_blocks, block_size, device=dev) < ratio).reshape(b, length)
    is_target = labels != -100
    mask_embed = embed(torch.full((1,), mask_id, device=dev)).to(clean.dtype)

    def noisy_copy(m):
        m = m & is_target
        return torch.cat([torch.where(m[..., None], mask_embed, clean), clean], dim=1), torch.where(m, labels, -100)

    (x1, l1), (x2, l2) = noisy_copy(masked), noisy_copy(~masked)
    x, lab = torch.cat([x1, x2]), torch.cat([l1, l2])
    h = fast.packed_forward(decoder, x.reshape(-1, x.shape[-1])[plan.src][None], plan, attention, block_mask)[0]

    # the decoded outputs in the boolean selection's order (row-major over [2B, L - 1]), without its host wait
    i, j = target_pos // length, target_pos % length
    row = (~masked.reshape(-1)[target_pos]).long() * b + i  # copy 1 decodes the masked targets, copy 2 the others
    key = (row * (length - 1) + j - 1).sort().values
    row, p = key // (length - 1), key % (length - 1)
    tok = lookup[row * 2 * length + p]
    torch._assert_async((tok >= 0).all())  # every decoded output was computed
    logits = lm_head(h[tok])
    target = lab.reshape(-1)[row * length + p + 1]
    ce = F.cross_entropy(logits.float(), target, reduction="sum")
    return ce, (lab[:, 1:] != -100).sum(), logits, target


def forward(model, input_ids, attention_mask, input_split, spikes, spikes_mask, spikes_timestamp, spikes_lengths,
            targets=None, dataset_name=None, day_idx=None, attention: str | None = None) -> BCIOutput:
    """BCI_DiffusionLLM.forward with fast_step, bit for bit, with one host wait (module docstring)."""
    spikes_embeds, text_embeds, token_mask, contrastive_loss = model.prepare_embeds(
        input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx,
    )
    decoder, lm_head, embed = model._parts()
    dev, dtype, d = text_embeds.device, text_embeds.dtype, text_embeds.shape[-1]
    meta = (attention_mask, targets, input_ids, input_split, token_mask)
    host = torch.cat([m.reshape(-1).long() for m in meta]).cpu()  # the one wait: after the encoder
    host = [h.view(m.shape) for h, m in zip(host.split([m.numel() for m in meta]), meta)]
    lay = layout(*host, model.block_size, model.eos_id, model.revision["align_sentence_block"])
    gather, labels = to_device([lay.gather, lay.labels], dev)
    special = torch.full((2,), model.eos_id, device=dev)
    special[1] = model.filler_id
    table = torch.cat([text_embeds.reshape(-1, d), spikes_embeds.to(dtype).reshape(-1, d), embed(special).to(dtype)])
    clean = table[gather].view(*labels.shape, d)
    opt = FAST_STEP_DEFAULTS | model.sampler
    ce, n_supervised, logits, lab = sync_free_ce(
        decoder, lm_head, embed, clean, labels, lay.labels, model.block_size, model.mask_id,
        model.revision["edit_rate"], model.filler_id, opt["share_clean"], opt["pack_quantum"] or fast.PACK_QUANTUM,
        attention)
    loss = ce if contrastive_loss is None else ce + contrastive_loss
    return BCIOutput(loss=loss, n_examples=n_supervised, preds=logits, targets=lab)
