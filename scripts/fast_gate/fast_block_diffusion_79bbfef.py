"""A faster block-diffusion training step with the same maths (docs/DIFFUSION_SPEEDUP.md). fast_block_diffusion_ce
gives diffusion_llm.block_diffusion_ce's loss, logits and gradients, but skips the computation that no decoded output
depends on. It is off by default and turned on by `fast_step: true` in a configs/llms.yaml entry's `sampler`. Run 3
used the plain step.

block_diffusion_ce feeds the decoder [noisy ‖ clean] at the batch length L, once per trial and complementary copy:
2B rows of 2L tokens. Its attention lets noisy block k see itself and clean blocks < k, and clean block k see clean
blocks ≤ k. The noisy output at position p is decoded only when p + 1 is supervised. That output depends on noisy
block(p) and on the clean blocks before it, and on nothing else. So each row needs only:
- noisy blocks: the blocks holding a position whose next position is a target. For BIT, these run from the block of
  the last prefix token to the last target block. The first is the sentence's first block, or, when the prefix fills
  whole blocks, the prefix block just before it;
- clean blocks: block 0 up to the last needed noisy block, exclusive.
Three things are never computed: the noisy copy of the prefix's other blocks, the clean copy of the last target block,
and the batch's padding blocks in both halves. Which blocks are needed depends only on the target labels, never on the
random masks, so the shapes do not depend on the draws.

PackedDecoder computes only those tokens. The needed blocks of all rows are packed into one sequence, which is padded
to a multiple of pack_quantum. Each token keeps its row, half, block and position. RoPE uses the original position.
Token q attends token k iff both belong to the same row and block_diff_mask holds between their original positions.
The pad tokens form a context of their own. The outputs are scattered back to their [2B, 2L] slots, so
block_diffusion_ce's own masking, token shift, selection and cross-entropy run unchanged, with the same random draws
in the same order.

share_clean (off by default): both complementary copies of a trial read one clean copy of its context instead of
two. Without dropout this is exact too. With LoRA dropout, the two copies' predictions share one dropout draw on the
context instead of two independent ones. The expected loss and gradient are the same, but the noise is correlated, so
this changes the recipe's noise slightly and has its own flag.

RNG: the masks and T2T draws are block_diffusion_ce's, so a seed gives the same masks. LoRA dropout draws a mask for
each token copy that is computed, which is fewer copies than the plain step. Two things follow. (i) The needed copies
get different dropout masks than in the plain step, with the same distribution and independence. (ii) The generator's
state after the step differs, and so do all later draws. A fast-step run is therefore the same recipe, but not a
bit-for-bit replay of a plain-step run with the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.modeling_outputs import BaseModelOutput

from bit_experiments.diffusion_llm import block_diffusion_ce

PACK_QUANTUM = 512  # packed length rounded up to a multiple of this: few distinct shapes for the compiled flex attention
FLEX_TILE = 128     # flex attention's block-mask tile (create_block_mask's default BLOCK_SIZE)
# Flex kernel tiles for the packed step, explicit in both directions (docs/DIFFUSION_SPEEDUP.md "Kernel tiles"). The
# GPU (RTX PRO 6000 Blackwell, sm_120) has 101,376 bytes of shared memory per block. Torch 2.7 treats it as an H100 and
# runs the forward at (BLOCK_M 128, BLOCK_N 64, 8 warps, 3 stages). The plain step's mask (index arithmetic) fits:
# 3 stages x (K + V) 64 x 128 bf16 tiles = 98,304 bytes. The packed mask reads its tables, and converting its
# 128 x 64 boolean output takes 128 x 72 = 9,216 more: 107,520, the failed first GPU smoke (2026-10-23). At 64 x 64
# with 2 stages: 2 x 32,768 + 64 x 72 = 70,144 bytes. Even counting a resident Q tile (16,384) and an fp32 mask
# scratch (64 x 68 x 4 = 17,408), it is 99,328. The backward tiles are the vendored FLEX_KERNEL_OPTIONS, which fit for
# run 3; the packed mask adds at most a 64 x 32 tile's scratch to them. All tiles divide FLEX_TILE.
PACKED_FLEX_KERNEL_OPTIONS = {"fwd_BLOCK_M": 64, "fwd_BLOCK_N": 64, "fwd_num_warps": 4, "fwd_num_stages": 2,
                              "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 64, "bwd_BLOCK_M2": 64, "bwd_BLOCK_N2": 32,
                              "bwd_num_warps": 4, "bwd_num_stages": 2}


@torch.compile(fullgraph=True)
def packed_flex_attention(q, k, v, mask=None):
    """The packed step's compiled flex attention: the vendored fused_flex_attention with PACKED_FLEX_KERNEL_OPTIONS
    (the vendored one, used by the plain step, is left as it is)."""
    return flex_attention(q, k, v, block_mask=mask, enable_gqa=True, kernel_options=PACKED_FLEX_KERNEL_OPTIONS)


def _compiled_flex(device) -> bool:
    """Whether flex attention runs compiled (packed_flex_attention) or eagerly (the reference, for CPU tests)."""
    return device.type == "cuda"


def packed_mask(b, h, q, kv, noisy, blk, ctx, grp):
    """block_diff_mask on packed tokens (module docstring). A noisy token sees the noisy tokens of its own row and
    block, and the clean tokens of its context in earlier blocks. A clean token sees the clean tokens of its context in
    its own and earlier blocks. It is used as flex attention's mask_mod and on index grids (the dense CPU mask)."""
    same_noisy_block = noisy[q] & noisy[kv] & (grp[q] == grp[kv]) & (blk[q] == blk[kv])
    clean_context = ~noisy[kv] & (ctx[q] == ctx[kv]) & ((blk[kv] < blk[q]) | (~noisy[q] & (blk[kv] == blk[q])))
    return same_noisy_block | clean_context


@dataclass
class PackPlan:
    """The tokens of block_diffusion_ce's [2B, 2L] decoder input that get computed, in packed order with the pads
    last. Each token has:
    - src: its flat index in that input (row * 2L + position in [noisy ‖ clean]);
    - pos: its RoPE position; noisy: whether it is in the noisy half; blk: its block;
    - ctx: the clean context it belongs to or reads (-1 for pads); grp: for a noisy token, its row (-1 for pads).
    out_tok and out_src give the packed indices of the noisy tokens and their src: the outputs block_diffusion_ce
    decodes. n_real is the number of tokens before the pads."""
    src: torch.Tensor
    pos: torch.Tensor
    noisy: torch.Tensor
    blk: torch.Tensor
    ctx: torch.Tensor
    grp: torch.Tensor
    out_tok: torch.Tensor
    out_src: torch.Tensor
    n_real: int

    def __len__(self):
        return self.src.numel()

    def to(self, device) -> "PackPlan":
        return PackPlan(**{f.name: getattr(self, f.name).to(device, non_blocking=True) if f.name != "n_real"
                           else self.n_real for f in fields(self)})

    def mask_mod(self):
        return partial(packed_mask, noisy=self.noisy, blk=self.blk, ctx=self.ctx, grp=self.grp)

    def dense_mask(self) -> torch.Tensor:
        """The [T, T] boolean mask (True = may attend), from the same mask_mod as the flex block mask."""
        idx = torch.arange(len(self), device=self.src.device)
        return self.mask_mod()(None, None, idx[:, None], idx[None, :])


def needed_blocks(labels: torch.Tensor, block_size: int):
    """For each trial: the noisy blocks that are needed ([B, n_blocks], the blocks holding a position whose next
    position is a target), and the number of clean blocks that are needed (block 0 up to the last needed noisy block,
    exclusive). labels [B, L] are block_diffusion_ce's (-100 = context)."""
    b, length = labels.shape
    nxt = torch.zeros(b, length, dtype=torch.bool, device=labels.device)
    nxt[:, :-1] = labels[:, 1:] != -100
    noisy = nxt.view(b, length // block_size, block_size).any(-1)
    k = torch.arange(noisy.shape[1], device=labels.device)
    return noisy, torch.where(noisy, k, 0).amax(-1)


def pack_plan(labels: torch.Tensor, block_size: int, share_clean=False, quantum=PACK_QUANTUM) -> PackPlan:
    """The packed layout for block_diffusion_ce's labels [B, L] (module docstring), on the CPU. Row r = c * B + i is
    trial i's copy c, as in block_diffusion_ce's torch.cat([copy 1, copy 2]). Tokens are grouped by trial for
    locality: copy 1's clean then noisy blocks, then copy 2's (with share_clean, copy 2 has no clean blocks and both
    copies read copy 1's). The plan depends only on the labels, not on the masks."""
    if quantum % FLEX_TILE:
        raise ValueError(f"pack_quantum {quantum} should be a multiple of {FLEX_TILE}")
    labels = labels.detach().cpu()
    b, length = labels.shape
    noisy_blocks, n_clean = needed_blocks(labels, block_size)
    entries = []  # one per computed block: (row, half: 0 noisy / 1 clean, block, ctx, grp)
    for i in range(b):
        ks = noisy_blocks[i].nonzero().flatten().tolist()
        for c in range(2):
            r = c * b + i
            ctx = i if share_clean else r
            if c == 0 or not share_clean:
                entries += [(r, 1, k, ctx, r) for k in range(int(n_clean[i]))]
            entries += [(r, 0, k, ctx, r) for k in ks]
    row, half, blk, ctx, grp = torch.tensor(entries, dtype=torch.long).view(-1, 5).repeat_interleave(block_size, 0).T
    pos = blk * block_size + torch.arange(block_size).repeat(len(entries))
    src = row * 2 * length + half * length + pos
    noisy = half == 0
    n_real = src.numel()
    n_pad = (-n_real) % quantum or (quantum if n_real == 0 else 0)
    pad = torch.zeros(n_pad, dtype=torch.long)
    out_tok = noisy.nonzero().flatten()
    return PackPlan(src=torch.cat([src, pad]), pos=torch.cat([pos, pad]), noisy=torch.cat([noisy, pad.bool()]),
                    blk=torch.cat([blk, pad]), ctx=torch.cat([ctx, pad - 1]), grp=torch.cat([grp, pad - 1]),
                    out_tok=out_tok, out_src=src[out_tok], n_real=n_real)


def _decoder_layer(layer, h, cos, sin, attend):
    """Fast_dLLM_QwenDecoderLayer.forward with its attention's training branch, on one packed sequence. RoPE is
    applied to the whole sequence at the plan's positions, rather than to each half at positions 0..L-1. The values
    per position are the same."""
    from bit_experiments.fast_dllm_v2.modeling import apply_rotary_pos_emb

    attn, shape = layer.self_attn, (*h.shape[:-1], -1, layer.self_attn.head_dim)
    x = layer.input_layernorm(h)
    q = attn.q_proj(x).view(shape).transpose(1, 2)
    k = attn.k_proj(x).view(shape).transpose(1, 2)
    v = attn.v_proj(x).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    o = attend(q, k, v).transpose(1, 2).contiguous()
    h = h + attn.o_proj(o.reshape(*h.shape[:-1], -1).contiguous())
    return h + layer.mlp(layer.post_attention_layernorm(h))


def packed_forward(decoder, x: torch.Tensor, plan: PackPlan, attention: str | None = None) -> torch.Tensor:
    """Fast_dLLM_QwenModel.forward in its training layout, on the packed tokens x [1, T, d], with the plan's
    positions and mask. Returns the final normed hidden states [1, T, d]. The attention options are:
    - "flex": flex attention with the plan's block mask (on CUDA, packed_flex_attention with its tiles);
    - "dense": SDPA with the boolean mask (the vendored CPU fallback);
    - None: flex on CUDA, dense elsewhere, as the vendored model chooses."""
    attention = attention or ("flex" if x.device.type == "cuda" else "dense")
    if attention == "flex":
        block_mask = create_block_mask(plan.mask_mod(), None, None, len(plan), len(plan), device=x.device)

        def attend(q, k, v):  # q, k leave RoPE in fp32 under autocast; flex needs one dtype (as the vendored model)
            q, k = q.to(v.dtype), k.to(v.dtype)
            if _compiled_flex(x.device):
                return packed_flex_attention(q, k, v, mask=block_mask)
            return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)  # eager reference (CPU tests)
    elif attention == "dense":
        dense = plan.dense_mask()

        def attend(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=True)
    else:
        raise ValueError(f"attention {attention!r}: expected 'flex', 'dense' or None")
    cos, sin = decoder.rotary_emb(x, plan.pos[None])
    h = x
    for layer in decoder.layers[: decoder.config.num_hidden_layers]:
        h = _decoder_layer(layer, h, cos, sin, attend)
    return decoder.norm(h)


class PackedDecoder(nn.Module):
    """Stands in for the decoder inside block_diffusion_ce. It computes only the plan's tokens and returns the noisy
    outputs in their [2B, 2L] slots. Every other slot is zero and is never decoded: an async assert checks that each
    decoded output was computed."""

    def __init__(self, decoder, plan: PackPlan, attention: str | None = None):
        super().__init__()
        self.decoder, self.plan, self.attention = decoder, plan, attention

    def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor | None = None, **unused):
        rows, width, d = inputs_embeds.shape
        plan = self.plan
        if labels is not None:  # decoded: noisy position p with p + 1 supervised (block_diffusion_ce's token shift)
            decoded = torch.zeros(rows, width, dtype=torch.bool, device=labels.device)
            decoded[:, : width // 2 - 1] = labels[:, 1:] != -100
            decoded.view(-1)[plan.out_src] = False
            torch._assert_async(~decoded.any())
        h = packed_forward(self.decoder, inputs_embeds.reshape(-1, d)[plan.src][None], plan, self.attention)[0]
        out = h.new_zeros(rows * width, d).index_copy(0, plan.out_src, h[plan.out_tok])
        return BaseModelOutput(last_hidden_state=out.view(rows, width, d))


def fast_block_diffusion_ce(decoder, lm_head, embed, clean: torch.Tensor, labels: torch.Tensor, block_size: int,
                            mask_id: int, edit_rate: float = 0.0, n_ordinary: int | None = None, with_corrupted=False,
                            share_clean=False, quantum=PACK_QUANTUM, attention: str | None = None):
    """block_diffusion_ce with the same arguments and returns, computing only what the decoded outputs depend on
    (module docstring). share_clean, quantum and attention: see the module docstring, pack_plan and packed_forward."""
    plan = pack_plan(labels, block_size, share_clean, quantum).to(clean.device)
    return block_diffusion_ce(PackedDecoder(decoder, plan, attention), lm_head, embed, clean, labels, block_size,
                              mask_id, edit_rate, n_ordinary, with_corrupted)


def step_cost(labels: torch.Tensor, block_size: int, share_clean=False, quantum=PACK_QUANTUM, tile=FLEX_TILE) -> dict:
    """The decoder work of one step, plain against packed: tokens through the layers, allowed (query, key) pairs, and
    flex attention tiles (tile × tile, any allowed pair; flex computes whole tiles). Plain = 2B rows of 2L under
    block_diff_mask, padding blocks included. Used for docs/DIFFUSION_SPEEDUP.md."""
    from bit_experiments.fast_dllm_v2.modeling import block_diff_mask

    b, length = labels.shape
    idx = torch.arange(2 * length)
    plain = block_diff_mask(None, None, idx[:, None], idx[None, :], block_size=block_size, n=length)
    plan = pack_plan(labels, block_size, share_clean, quantum)
    packed = plan.dense_mask()

    def tiles(m):
        n_q, n_k = -(-m.shape[0] // tile), -(-m.shape[1] // tile)
        m = F.pad(m, (0, n_k * tile - m.shape[1], 0, n_q * tile - m.shape[0]))
        return int(m.view(n_q, tile, n_k, tile).any(3).any(1).sum())

    return {"plain": {"tokens": 2 * b * 2 * length, "pairs": 2 * b * int(plain.sum()), "tiles": 2 * b * tiles(plain)},
            "packed": {"tokens": len(plan), "real_tokens": plan.n_real, "pairs": int(packed.sum()),
                       "tiles": tiles(packed)},
            "supervised": int((labels != -100).sum())}
