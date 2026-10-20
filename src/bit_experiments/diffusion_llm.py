"""BIT's end-to-end model with a block-diffusion decoder: Fast-dLLM v2 1.5B (Phase 4 LLM-swap run 3, DESIGN.md §8).

Encoder, projector, prompt, LoRA and BIT's contrastive term are as in BCI_TextLLM (run 1, Qwen2.5-1.5B-Instruct, the
model Fast-dLLM v2 was converted from). The decoder objective and the decoding are Fast-dLLM v2's, rewritten to take
embeddings: its released `forward` (training) and `generate` take token ids only, and the neural tokens are not tokens.

Sequence per trial: [prompt_a, neural tokens, prompt_b] = a clean prefix that is never masked, then the sentence +
EOS, padded with EOS to the end of its 32-token block (LLaDA-style; the padding is supervised too). Batches are padded
with whole blocks, which no real position attends to (attention is causal across blocks).

Training step (Fast-dLLM v2 `Fast_dLLM_QwenForCausalLM.forward`):
- per 32-token block a mask ratio t ~ U(eps, 1); target tokens are replaced by |<MASK>| with probability t;
- a complementary copy masks exactly the other target tokens, so every target token is predicted once per pair;
- the input is [noisy ‖ clean] with its block-diffusion attention (noisy block: itself + earlier clean blocks;
  clean block: itself + earlier clean blocks), and only the noisy half is decoded;
- token shift: masked position j is predicted from the output at position j - 1 (as its shifted causal-LM loss).
The cross-entropy is summed over the masked tokens of both copies, as BIT sums its CE in run 1: every target token
(padding EOS included) counts once per complementary pair, so a sentence token weighs what it weighs in run 1 next to
BIT's summed contrastive term. (Fast-dLLM v2 averages instead; a mean over the masked tokens, about half of them
padding, would halve the sentence tokens' weight relative to the contrastive term.)

Decoding (Fast-dLLM v2 `generate`, without its block-level KV cache): the prefix's complete blocks go into the KV
cache; each block is filled sub-block by sub-block (small_block_size), each forward unmasking every masked position of
the sub-block whose proposal has probability > threshold, and at least the most confident one. A finished block is
committed to the cache and the next block's first token is its argmax next token. Stops at EOS (no mask before it) or
max_new_tokens. Greedy = argmax proposals; nucleus = top-p sampled proposals (temperature, top_p as for the AR runs).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from models.bci import BCIOutput
from bit_experiments.text_llm import BCI_TextLLM

MASK_TOKEN = "|<MASK>|"
EPS = 1e-3          # minimum mask ratio, as Fast-dLLM v2
LENGTH_BUCKET = 128  # batch lengths are padded to multiples of this (fewer shapes for the compiled flex attention)


class BCI_DiffusionLLM(BCI_TextLLM):
    """BCI_TextLLM with Fast-dLLM v2 (block diffusion) as the decoder."""

    def __init__(self, config, llm_path, lora=None, freeze_llm=False, diffusion_sampler=None, **kwargs):
        """diffusion_sampler: block_size, small_block_size, threshold, mask_id (set by text_llm.use_llm)."""
        super().__init__(config, llm_path, lora, freeze_llm, **kwargs)
        self.sampler = dict(diffusion_sampler)
        cfg = self.llm_config.text_config
        self.block_size, self.eos_id, self.mask_id = cfg.bd_size, cfg.eos_token_id, int(self.sampler["mask_id"])
        if self.sampler["block_size"] != self.block_size:
            raise ValueError(f"sampler block_size {self.sampler['block_size']} != the model's bd_size {self.block_size}")

    @staticmethod
    def from_pretrained(path):
        from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM

        return Fast_dLLM_QwenForCausalLM.from_pretrained(path, torch_dtype=torch.float32)

    # ------------------------------------------------------------ pieces of the (LoRA-wrapped) LLM

    def _parts(self):
        base = self.llm.get_base_model()
        return base.model, base.lm_head, base.get_input_embeddings()

    @contextmanager
    def _layout(self, training: bool):
        """The vendored decoder switches mask, RoPE split and attention kernel on `.training`. Set it on those modules
        only (the LoRA dropout keeps the model's own mode), e.g. diffusion-layout validation loss in eval mode."""
        from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenAttention, Fast_dLLM_QwenModel

        decoder = self._parts()[0]
        mods = [m for m in decoder.modules() if isinstance(m, (Fast_dLLM_QwenModel, Fast_dLLM_QwenAttention))]
        before = [m.training for m in mods]
        for m in mods:
            m.training = training
        try:
            yield
        finally:
            for m, b in zip(mods, before):
                m.training = b

    def _prefix(self, text_embeds, spikes_embeds, spikes_mask, split: int, n_prompt: int):
        """[prompt_a, valid neural tokens, prompt_b] for one trial."""
        return torch.cat([text_embeds[:split], spikes_embeds[spikes_mask.bool()], text_embeds[split:n_prompt]])

    # ------------------------------------------------------------ training

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_split: torch.LongTensor,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        spikes_lengths: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        dataset_name: torch.Tensor | None = None,
        day_idx: torch.LongTensor | None = None,
    ) -> BCIOutput:
        spikes_embeds, text_embeds, spikes_mask, contrastive_loss = self.prepare_embeds(
            input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx,
        )
        decoder, lm_head, embed = self._parts()
        bs, dev = self.block_size, text_embeds.device
        dtype = text_embeds.dtype

        rows = []
        for i in range(input_ids.shape[0]):
            n_text = int(attention_mask[i].sum())
            is_target = targets[i, :n_text] != -100
            n_prompt = int((~is_target).sum())
            if not bool(is_target[n_prompt:].all()):
                raise ValueError("expected [prompt tokens, sentence + EOS] per trial (BIT's create_llm_labels)")
            prefix = self._prefix(text_embeds[i], spikes_embeds[i].to(dtype), spikes_mask[i], int(input_split[i]),
                                  n_prompt)
            target = input_ids[i, n_prompt:n_text]
            pad = (-(prefix.shape[0] + target.numel())) % bs
            rows.append((prefix, torch.cat([target, target.new_full((pad,), self.eos_id)])))

        length = max(p.shape[0] + t.numel() for p, t in rows)
        length = -(-length // LENGTH_BUCKET) * LENGTH_BUCKET
        pad_embed = embed(torch.tensor([self.eos_id], device=dev)).to(dtype)
        clean, labels = [], []
        for prefix, target in rows:
            n_pad = length - prefix.shape[0] - target.numel()
            clean.append(torch.cat([prefix, embed(target).to(dtype), pad_embed.expand(n_pad, -1)]))
            labels.append(torch.cat([target.new_full((prefix.shape[0],), -100), target, target.new_full((n_pad,), -100)]))
        clean, labels = torch.stack(clean), torch.stack(labels)

        ce, n_supervised, logits, lab = self.denoising_ce(clean, labels)
        loss = ce if contrastive_loss is None else ce + contrastive_loss
        return BCIOutput(loss=loss, n_examples=n_supervised, preds=logits, targets=lab)

    def denoising_ce(self, clean: torch.Tensor, labels: torch.Tensor):
        """Fast-dLLM v2's training step on embeddings. clean [B, L, d] (L a multiple of the block size), labels [B, L]
        (-100 = never masked). The random draws are the same calls, in the same order and shapes, as in its
        `Fast_dLLM_QwenForCausalLM.forward`, so a shared seed gives the same masks. Logits are computed only where the
        next token is masked (token shift), which gives the same loss without the [2B, L, vocab] tensor. Returns
        (summed CE over the masked tokens of both copies, their number, their logits, their labels)."""
        decoder, lm_head, embed = self._parts()
        (b, length), bs, dev = labels.shape, self.block_size, labels.device
        n_blocks = length // bs
        ratio = ((1 - EPS) * torch.rand(b * n_blocks, device=dev) + EPS)[:, None].repeat(1, bs)
        masked = (torch.rand(b * n_blocks, bs, device=dev) < ratio).reshape(b, length)
        is_target = labels != -100
        mask_embed = embed(torch.tensor([self.mask_id], device=dev)).to(clean.dtype)

        def noisy_copy(m):
            m = m & is_target
            return torch.cat([torch.where(m[..., None], mask_embed, clean), clean], dim=1), torch.where(m, labels, -100)

        (x1, l1), (x2, l2) = noisy_copy(masked), noisy_copy(~masked)
        x, lab = torch.cat([x1, x2]), torch.cat([l1, l2])
        with self._layout(training=True):
            hidden = decoder(inputs_embeds=x, labels=lab).last_hidden_state[:, :length]
        shifted = lab[:, 1:]
        supervised = shifted != -100
        logits = lm_head(hidden[:, :-1][supervised])
        ce = F.cross_entropy(logits.float(), shifted[supervised], reduction="sum")
        return ce, supervised.sum(), logits, shifted[supervised]

    # ------------------------------------------------------------ decoding

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_split: torch.LongTensor,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        spikes_lengths: torch.LongTensor,
        targets: torch.LongTensor | None = None,
        dataset_name: torch.Tensor | None = None,
        day_idx: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        max_new_tokens: int = 25,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        **unused,
    ) -> torch.LongTensor:
        """One trial (batch 1), as bit_experiments.stage2.generate calls it. `input_ids` is the prompt; `inputs_embeds`
        (prompt-only null control) replaces the whole prefix. Returns the new token ids, [1, n]."""
        if input_ids.shape[0] != 1:
            raise ValueError("block-diffusion decoding is one trial at a time")
        if inputs_embeds is None:
            spikes_embeds, text_embeds, spikes_mask, _ = self.prepare_embeds(
                input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx,
            )
            prefix = self._prefix(text_embeds[0], spikes_embeds[0].to(text_embeds.dtype), spikes_mask[0],
                                  int(input_split[0]), text_embeds.shape[1])
        else:
            prefix = inputs_embeds[0]
        eos = self.eos_id if eos_token_id is None else int(eos_token_id)
        with self._layout(training=False):
            ids = self._denoise(prefix.to(self.llm.dtype), max_new_tokens, do_sample, top_p, temperature, eos)
        return torch.tensor([ids], device=prefix.device, dtype=torch.long)

    def _denoise(self, prefix, max_new_tokens, do_sample, top_p, temperature, eos) -> list[int]:
        decoder, lm_head, embed = self._parts()
        bs, sbs, threshold = self.block_size, self.sampler["small_block_size"], self.sampler["threshold"]
        cache = DynamicCache()

        def run(x, commit):
            h = decoder(inputs_embeds=x[None], past_key_values=cache, use_cache=True, update_past_key_values=commit,
                        block_size=bs).last_hidden_state
            logits = lm_head(h)[0].float()
            logits[:, self.mask_id] = -torch.inf  # an accepted mask proposal would leave its slot masked forever
            return logits

        n_full = prefix.shape[0] // bs * bs
        tail = prefix[n_full:]
        slots = torch.full((bs - tail.shape[0],), self.mask_id, device=prefix.device)
        if n_full:
            logits = run(prefix[:n_full], commit=True)
            if tail.shape[0] == 0:
                slots[0] = logits[-1].argmax()  # the block's first token is the AR next token, as Fast-dLLM v2

        out: list[int] = []
        while True:
            for start in range(0, bs, sbs):
                lo, hi = max(start - tail.shape[0], 0), max(start + sbs - tail.shape[0], 0)
                while bool((slots[lo:hi] == self.mask_id).any()) and not self._finished(slots, eos):
                    logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=False)
                    logits = torch.cat([logits[:1], logits[:-1]])[tail.shape[0]:][lo:hi]  # token shift
                    proposal, confidence = self._propose(logits, do_sample, top_p, temperature)
                    is_masked = slots[lo:hi] == self.mask_id
                    confidence = torch.where(is_masked, confidence, -torch.inf)
                    accept = confidence > threshold
                    accept[confidence.argmax()] = True
                    accept &= is_masked
                    slots[lo:hi] = torch.where(accept, proposal, slots[lo:hi])
                if self._finished(slots, eos):
                    break
            out += slots.tolist()
            if eos in out or len(out) >= max_new_tokens:
                break
            logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=True)
            tail = prefix[:0]
            slots = torch.full((bs,), self.mask_id, device=prefix.device)
            slots[0] = logits[-1].argmax()
        if eos in out:
            out = out[: out.index(eos) + 1]
        return out[:max_new_tokens]

    def _finished(self, slots, eos) -> bool:
        """EOS decoded with no mask before it (the block's remaining masks come after the sentence)."""
        hits = (slots == eos).nonzero()
        return bool(hits.numel()) and not bool((slots[: int(hits[0])] == self.mask_id).any())

    @staticmethod
    def _propose(logits, do_sample, top_p, temperature):
        """Per position: proposed token and its probability. Greedy: argmax of the softmax. Nucleus: top-p sample."""
        if not do_sample:
            confidence, proposal = logits.softmax(-1).max(-1)
            return proposal, confidence
        probs = (logits / temperature).softmax(-1)
        sorted_probs, order = probs.sort(-1, descending=True)
        sorted_probs[(sorted_probs.cumsum(-1) - sorted_probs) > top_p] = 0
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True)
        proposal = torch.multinomial(probs, 1).squeeze(-1)
        return proposal, probs.gather(-1, proposal[:, None]).squeeze(-1)
