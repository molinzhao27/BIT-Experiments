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
The step is the module-level block_diffusion_ce. text_block_diffusion_ce applies it to token ids alone, for
post-training the text model itself (docs/REVISION_RUN.md, two-step design).

Decoding (Fast-dLLM v2 `generate`, without its block-level KV cache): the prefix's complete blocks go into the KV
cache; each block is filled sub-block by sub-block (small_block_size), each forward unmasking every masked position of
the sub-block whose proposal has probability > threshold, and at least the most confident one. A finished block is
committed to the cache and the next block's first token is its argmax next token. Stops at EOS (no mask before it) or
max_new_tokens. Greedy = argmax proposals; nucleus = top-p sampled proposals (temperature, top_p as for the AR runs).

Revision options (docs/REVISION_RUN.md), keys of the configs/llms.yaml entry's `sampler`. All are off by default
(REVISION_DEFAULTS), and then training and decoding are exactly run 3's.
- align_sentence_block: the prefix is left-padded with <|endoftext|> (the config's bos token, Qwen's document
  separator) until its last token opens a block, in training and in decoding alike. That token predicts the sentence's
  first word (token shift), so the whole sentence + EOS (≤ 26 tokens) is denoised inside one block, and with
  editing on stays revisable until that block is committed.
- edit_rate: a token-to-token term next to the mask-to-token one (LLaDA2.1's T2T). Each visible target token of a copy
  is replaced by a uniformly random ordinary token (ids below <|endoftext|>) with this probability and supervised with
  its original token, in the same summed CE. Every target token still counts once per pair as a masked token, as in
  run 3; the replaced ones count once more. The corruption is drawn after the masks, so the masks do not change.
- editing_threshold, max_post_steps, max_edits: the editing sampler. On every pass, each committed token of the current
  block (up to its first EOS) whose new proposal differs and has probability > editing_threshold is overwritten, with
  the fill step's commits, from the same forward. When the block has nothing left to fill (no mask before an EOS),
  passes that only edit continue until one edits nothing, or max_post_steps of them have run. max_edits caps the edits
  per position (None: no cap). Earlier blocks are in the KV cache and never change.
- remask_threshold, remask_rounds: the re-masking sampler (Mask-Predict / Mask-CTC style; needs no training, only the
  mask-to-token skill every run has). When a block is done, its committed tokens (up to the first EOS) whose
  probability when committed was below remask_threshold are masked again and re-filled by the same sampler, now with
  the rest of the block visible; up to remask_rounds times, stopping early when a round changes nothing.

Instruct control (configs/llms.yaml qwen2.5-1.5b-instruct-diffusion, docs/INSTRUCT_DIFFUSION_CONTROL.md): the same
model, objective and sampler from Qwen2.5-1.5B-Instruct's own weights instead of NVIDIA's conversion. from_pretrained
loads a Qwen2 snapshot into the vendored architecture (qwen2_as_fast_dllm_config). The mask is Instruct's unused
embedding row 151665 (Fast-dLLM v2's |<MASK>| id), frozen under LoRA as run 3's mask row is.

Aero block diffusion (configs/llms.yaml aero-1-audio-diffusion, docs/AERO_DIFFUSION.md): the same model, objective and
sampler from Aero-1-Audio's language model, the LM BIT decodes with. from_pretrained reads it out of BIT's own Aero
snapshot (aero_language_model: the `language_model.` tensors, without the audio tower and projector) and converts its
Qwen2 text_config as for the control. The mask is again row 151665 (`<image>` in Aero's tokenizer, never used by Aero;
the same untrained vector as every unused row), frozen.

Training speed (docs/DIFFUSION_SPEEDUP.md), keys of the entry's `sampler`. They are off by default
(FAST_STEP_DEFAULTS), and then the step is run 3's. fast_step: true swaps block_diffusion_ce for
bit_experiments.fast_block_diffusion's packed step. It has the same masks, loss and gradients but skips the tokens no
decoded output depends on (mainly the noisy copy of the prefix). share_clean and pack_quantum are its options, and
sync_free runs the same step, bit for bit, with one host wait instead of ~45 (bit_experiments.sync_free_step).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from models.bci import BCIOutput
from bit_experiments.text_llm import BCI_TextLLM

MASK_TOKEN = "|<MASK>|"
EPS = 1e-3          # minimum mask ratio, as Fast-dLLM v2
LENGTH_BUCKET = 128  # batch lengths are padded to multiples of this (fewer shapes for the compiled flex attention)
REVISION_DEFAULTS = {"align_sentence_block": False, "edit_rate": 0.0, "editing_threshold": None, "max_post_steps": 16,
                     "max_edits": None, "remask_threshold": None, "remask_rounds": 1}  # all off = run 3 (docstring)
FAST_STEP_DEFAULTS = {"fast_step": False, "share_clean": False, "pack_quantum": None,  # off = run 3's step (docstring)
                      "sync_free": False}
BD_SIZE = 32  # block size given to a converted Qwen2 snapshot: Fast-dLLM v2 1.5B's bd_size (its config.json)
AERO_LM_PREFIX = "language_model."  # Aero-1-Audio's LM (AeroForConditionalGeneration.language_model, a Qwen2ForCausalLM)


def qwen2_as_fast_dllm_config(config: dict):
    """A Qwen2 causal LM's config.json as a Fast-dLLM v2 config: every field kept, plus bd_size. Fast-dLLM v2 is a
    Qwen2 with the same modules and parameter names (q/k/v biases, tied embeddings per the config), so the weights
    load as they are. Only the attention mask, the RoPE split of the [noisy ‖ clean] input and the KV cache differ, as
    for run 3. Refuses configs the vendored model would treat differently (sliding window, RoPE scaling)."""
    from bit_experiments.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig

    if config.get("model_type") != "qwen2":
        raise ValueError(f"expected a qwen2 config, got model_type {config.get('model_type')!r}")
    if config.get("use_sliding_window") or config.get("rope_scaling") or config.get("hidden_act") != "silu":
        raise ValueError("sliding-window attention, RoPE scaling and activations other than silu are not converted")
    kept = {k: v for k, v in config.items() if k not in ("architectures", "model_type", "transformers_version")}
    return Fast_dLLM_QwenConfig(**kept, bd_size=BD_SIZE)


def aero_language_model(path) -> tuple[dict, dict]:
    """Aero-1-Audio's language model from its snapshot (config.json model_type "aero"): its Qwen2 text_config and its
    `language_model.` tensors under Qwen2ForCausalLM's names. The audio tower and audio projector are left out (BIT
    never uses them). Aero ships lm_head and embed_tokens as two tensors, equal, with tie_word_embeddings in its
    text_config, so BIT's tied load holds that one matrix: lm_head is checked equal and dropped (refused if not)."""
    from safetensors import safe_open

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "aero":
        raise ValueError(f"expected an Aero-1-Audio snapshot, got model_type {config.get('model_type')!r}")
    text_config = {k: v for k, v in config["text_config"].items() if k != "_name_or_path"}  # Aero's base model's name
    index = path / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    state = {}
    for name in files:
        with safe_open(path / name, framework="pt") as f:
            for key in f.keys():
                if key.startswith(AERO_LM_PREFIX):
                    state[key[len(AERO_LM_PREFIX):]] = f.get_tensor(key)
    if text_config.get("tie_word_embeddings", True) and "lm_head.weight" in state:
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError("Aero's lm_head differs from its embeddings: a tied load would keep only one of them")
        del state["lm_head.weight"]
    return text_config, state


@contextmanager
def diffusion_layout(decoder, training: bool):
    """The vendored decoder switches mask, RoPE split and attention kernel on `.training`. Set it on those modules
    only (the LoRA dropout keeps the model's own mode), e.g. diffusion-layout validation loss in eval mode."""
    from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenAttention, Fast_dLLM_QwenModel

    mods = [m for m in decoder.modules() if isinstance(m, (Fast_dLLM_QwenModel, Fast_dLLM_QwenAttention))]
    before = [m.training for m in mods]
    for m in mods:
        m.training = training
    try:
        yield
    finally:
        for m, b in zip(mods, before):
            m.training = b


def block_diffusion_ce(decoder, lm_head, embed, clean: torch.Tensor, labels: torch.Tensor, block_size: int,
                       mask_id: int, edit_rate: float = 0.0, n_ordinary: int | None = None, with_corrupted=False):
    """Fast-dLLM v2's training step on embeddings, with the optional token-to-token term (module docstring). Used by
    BCI_DiffusionLLM.denoising_ce (Stage 2) and text_block_diffusion_ce (text-only post-training).
    clean [B, L, d] (L a multiple of block_size), labels [B, L] (-100 = context: never masked or corrupted). The
    random draws are the same calls, in the same order and shapes, as in its `Fast_dLLM_QwenForCausalLM.forward`, so a
    shared seed gives the same masks. Logits are computed only where the next token is supervised (token shift),
    which gives the same loss without the [2B, L, vocab] tensor. With edit_rate, each visible target token is replaced
    by a uniformly random token from [0, n_ordinary) with that probability and supervised with its original, after
    the masks are drawn. Returns (summed CE over the supervised tokens of both copies, their number, their logits,
    their labels), and with with_corrupted, which of them were corrupted (T2T) rather than masked (M2T)."""
    (b, length), bs, dev = labels.shape, block_size, labels.device
    n_blocks = length // bs
    ratio = ((1 - EPS) * torch.rand(b * n_blocks, device=dev) + EPS)[:, None].repeat(1, bs)
    masked = (torch.rand(b * n_blocks, bs, device=dev) < ratio).reshape(b, length)
    is_target = labels != -100
    mask_embed = embed(torch.tensor([mask_id], device=dev)).to(clean.dtype)
    if edit_rate:  # one draw for both copies: a target token is visible in exactly one of them
        corrupt = (torch.rand(b, length, device=dev) < edit_rate) & is_target
        noise = embed(torch.randint(n_ordinary, (b, length), device=dev)).to(clean.dtype)

    def noisy_copy(m):
        m = m & is_target
        if edit_rate:
            c = corrupt & ~m  # this copy's corrupted visible target tokens
            x = torch.where(c[..., None], noise, torch.where(m[..., None], mask_embed, clean))
            return torch.cat([x, clean], dim=1), torch.where(m | c, labels, -100)
        return torch.cat([torch.where(m[..., None], mask_embed, clean), clean], dim=1), torch.where(m, labels, -100)

    (x1, l1), (x2, l2) = noisy_copy(masked), noisy_copy(~masked)
    x, lab = torch.cat([x1, x2]), torch.cat([l1, l2])
    with diffusion_layout(decoder, training=True):
        hidden = decoder(inputs_embeds=x, labels=lab).last_hidden_state[:, :length]
    shifted = lab[:, 1:]
    supervised = shifted != -100
    logits = lm_head(hidden[:, :-1][supervised])
    ce = F.cross_entropy(logits.float(), shifted[supervised], reduction="sum")
    if with_corrupted:
        corrupted = torch.cat([corrupt & ~masked, corrupt & masked]) if edit_rate else lab != lab
        return ce, supervised.sum(), logits, shifted[supervised], corrupted[:, 1:][supervised]
    return ce, supervised.sum(), logits, shifted[supervised]


def text_block_diffusion_ce(model, input_ids: torch.LongTensor, mask_id: int, labels: torch.LongTensor | None = None,
                            edit_rate: float = 0.0, with_corrupted=False):
    """block_diffusion_ce on token ids, for post-training a Fast-dLLM v2 (or its PEFT wrapper) on text. labels default
    to input_ids (every token a target; -100 = context). Rows must be a multiple of the block size long. Random T2T
    tokens are the ids below the config's bos (<|endoftext|>, Qwen's first special token). The mean loss is ce / n.
    Used by scripts/posttrain_edit.py."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    embed = base.get_input_embeddings()
    return block_diffusion_ce(base.model, base.lm_head, embed, embed(input_ids),
                              input_ids if labels is None else labels, base.config.bd_size, mask_id, edit_rate,
                              base.config.bos_token_id, with_corrupted)


class BCI_DiffusionLLM(BCI_TextLLM):
    """BCI_TextLLM with Fast-dLLM v2 (block diffusion) as the decoder."""

    def __init__(self, config, llm_path, lora=None, freeze_llm=False, diffusion_sampler=None, **kwargs):
        """diffusion_sampler: block_size, small_block_size, threshold, mask_id (set by text_llm.use_llm), and the
        revision options (REVISION_DEFAULTS)."""
        super().__init__(config, llm_path, lora, freeze_llm, **kwargs)
        self._configure(diffusion_sampler)

    def _configure(self, diffusion_sampler):
        self.sampler = dict(diffusion_sampler)
        cfg = self.llm_config.text_config
        self.block_size, self.eos_id, self.mask_id = cfg.bd_size, cfg.eos_token_id, int(self.sampler["mask_id"])
        if self.sampler["block_size"] != self.block_size:
            raise ValueError(f"sampler block_size {self.sampler['block_size']} != the model's bd_size {self.block_size}")
        if not 0 <= self.mask_id < cfg.vocab_size:
            raise ValueError(f"mask id {self.mask_id} has no embedding row (vocab_size {cfg.vocab_size})")
        unknown = set(self.sampler) - {"block_size", "small_block_size", "threshold", "mask_id", *REVISION_DEFAULTS,
                                       *FAST_STEP_DEFAULTS}
        if unknown:
            raise ValueError(f"unknown sampler options {sorted(unknown)}")
        fast = FAST_STEP_DEFAULTS | self.sampler
        if not all(isinstance(fast[k], bool) for k in ("fast_step", "share_clean", "sync_free")):
            raise ValueError("fast_step, share_clean and sync_free should be true or false")
        if not fast["fast_step"] and (fast["share_clean"] or fast["pack_quantum"] is not None or fast["sync_free"]):
            raise ValueError("share_clean, pack_quantum and sync_free are options of fast_step, which is off")
        if fast["pack_quantum"] is not None and not (isinstance(fast["pack_quantum"], int) and fast["pack_quantum"] > 0
                                                     and fast["pack_quantum"] % 128 == 0):
            raise ValueError(f"pack_quantum {fast['pack_quantum']} should be a positive multiple of 128")
        opt = self.revision
        if not 0 <= opt["edit_rate"] < 1:
            raise ValueError(f"edit_rate {opt['edit_rate']} not in [0, 1)")
        if opt["editing_threshold"] is not None and not 0 < opt["editing_threshold"] < 1:
            raise ValueError(f"editing_threshold {opt['editing_threshold']} not in (0, 1); None turns editing off")
        if opt["remask_threshold"] is not None and not 0 < opt["remask_threshold"] <= 1:
            raise ValueError(f"remask_threshold {opt['remask_threshold']} not in (0, 1]; None turns re-masking off")
        if not (isinstance(opt["remask_rounds"], int) and opt["remask_rounds"] >= 1):
            raise ValueError(f"remask_rounds {opt['remask_rounds']} should be an integer >= 1")
        # <|endoftext|>: the filler, and the first special token (random T2T tokens are drawn from the ids below it)
        self.filler_id = cfg.bos_token_id
        if (opt["align_sentence_block"] or opt["edit_rate"]) and not self.filler_id < min(self.eos_id, self.mask_id):
            raise ValueError(f"bos {self.filler_id} should be the first special token (eos {self.eos_id}, "
                             f"mask {self.mask_id})")
        self.trace = None  # set to a list to record every decode (_denoise)
        self.commit_logprobs = None  # set to a list to record each decode's per-token commit log-probs (_denoise)

    @property
    def revision(self) -> dict:
        """The revision options in effect: the sampler's, else REVISION_DEFAULTS (off)."""
        return REVISION_DEFAULTS | self.sampler

    @staticmethod
    def from_pretrained(path):
        """The vendored Fast-dLLM v2 from its snapshot, or from a Qwen2 causal LM's (the Instruct control), whose
        config is converted by qwen2_as_fast_dllm_config, or from Aero-1-Audio's language model (aero_language_model,
        converted the same way)."""
        from bit_experiments.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM

        config = json.loads((Path(path) / "config.json").read_text())
        if config.get("model_type") == "qwen2":
            return Fast_dLLM_QwenForCausalLM.from_pretrained(path, config=qwen2_as_fast_dllm_config(config),
                                                             torch_dtype=torch.float32)
        if config.get("model_type") == "aero":
            text_config, state = aero_language_model(path)
            llm = Fast_dLLM_QwenForCausalLM.from_pretrained(None, config=qwen2_as_fast_dllm_config(text_config),
                                                            state_dict=state, torch_dtype=torch.float32)
            llm.name_or_path = llm.config.name_or_path = str(path)
            return llm
        return Fast_dLLM_QwenForCausalLM.from_pretrained(path, torch_dtype=torch.float32)

    # ------------------------------------------------------------ pieces of the (LoRA-wrapped) LLM

    def _parts(self):
        base = self.llm.get_base_model()
        return base.model, base.lm_head, base.get_input_embeddings()

    def _layout(self, training: bool):
        """diffusion_layout on this model's decoder."""
        return diffusion_layout(self._parts()[0], training)

    def _prefix(self, text_embeds, spikes_embeds, spikes_mask, split: int, n_prompt: int):
        """[prompt_a, valid neural tokens, prompt_b] for one trial."""
        return torch.cat([text_embeds[:split], spikes_embeds[spikes_mask.bool()], text_embeds[split:n_prompt]])

    def _align(self, prefix):
        """align_sentence_block: <|endoftext|> fillers before the prefix, so that its last token opens a block (the
        sentence then fills that block from its second position). Otherwise the prefix as is."""
        n = (1 - prefix.shape[0]) % self.block_size
        if not self.revision["align_sentence_block"] or n == 0:
            return prefix
        filler = self._parts()[2](torch.tensor([self.filler_id], device=prefix.device)).to(prefix.dtype)
        return torch.cat([filler.expand(n, -1), prefix])

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
        if (FAST_STEP_DEFAULTS | self.sampler)["sync_free"]:  # the same step with one host wait (sync_free_step)
            from bit_experiments import sync_free_step

            return sync_free_step.forward(self, input_ids, attention_mask, input_split, spikes, spikes_mask,
                                          spikes_timestamp, spikes_lengths, targets, dataset_name, day_idx)
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
            prefix = self._align(self._prefix(text_embeds[i], spikes_embeds[i].to(dtype), spikes_mask[i],
                                              int(input_split[i]), n_prompt))
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
        """block_diffusion_ce with this model's decoder, block size, mask and edit_rate. With fast_step, it uses the
        packed equivalent (bit_experiments.fast_block_diffusion)."""
        fast = FAST_STEP_DEFAULTS | self.sampler
        if fast["fast_step"]:
            from bit_experiments import fast_block_diffusion

            return fast_block_diffusion.fast_block_diffusion_ce(
                *self._parts(), clean, labels, self.block_size, self.mask_id, self.revision["edit_rate"],
                self.filler_id, share_clean=fast["share_clean"],
                quantum=fast["pack_quantum"] or fast_block_diffusion.PACK_QUANTUM)
        return block_diffusion_ce(*self._parts(), clean, labels, self.block_size, self.mask_id,
                                  self.revision["edit_rate"], self.filler_id)

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
        prefix = self._align(prefix)
        eos = self.eos_id if eos_token_id is None else int(eos_token_id)
        with self._layout(training=False):
            ids = self._denoise(prefix.to(self.llm.dtype), max_new_tokens, do_sample, top_p, temperature, eos)
        return torch.tensor([ids], device=prefix.device, dtype=torch.long)

    def _denoise(self, prefix, max_new_tokens, do_sample, top_p, temperature, eos) -> list[int]:
        """The sampler (module docstring). With self.trace a list, it appends a record of the decode in
        scripts/decoding_order_sweep.py's traced_denoise format: prefix_len, tail (prefix tokens sharing the first
        block), out (the returned ids), and events, in order:
        {"b": block, "first": [pos, token]} = a block's first token, the AR next token of the committed block before it;
        {"b": block, "m": [[pos, proposal, probability, committed 0/1], ...]} = one pass, every masked position of the
        sub-block being filled ([] in a pass that only edits). pos counts output tokens from 0.
        With editing on, every pass also has "e": [[pos, token, proposal, probability, edited 0/1], ...] = every
        committed position of the block up to its first EOS whose proposal differs from its token, before the pass.
        With re-masking on, {"b": block, "r": [[pos, token, probability when committed], ...]} = tokens masked again
        before a re-fill round; the passes that re-fill them list them in "m" again.
        With self.commit_logprobs a list (N-best rescoring side test, docs/RESCORING.md; off by default), it appends
        {"ids": the returned ids, "logprobs": per returned id, the model's log-probability of that token in the pass
        that committed it (log-softmax of the untempered logits, the mask excluded as in the proposals; a block's AR
        first token: from the commit pass of the block before; an edited or re-filled token: from its last write)}.
        Recording reads the same logits and changes no decode."""
        decoder, lm_head, embed = self._parts()
        bs, sbs, threshold = self.block_size, self.sampler["small_block_size"], self.sampler["threshold"]
        opt = self.revision
        edit, cap = opt["editing_threshold"], float("inf") if opt["max_edits"] is None else opt["max_edits"]
        remask, rounds = opt["remask_threshold"], opt["remask_rounds"] if opt["remask_threshold"] is not None else 0
        cache = DynamicCache()
        rec = {"prefix_len": int(prefix.shape[0]), "events": []}
        if self.trace is not None:
            self.trace.append(rec)
        score = self.commit_logprobs is not None
        out_lp: list[float] = []

        def run(x, commit):
            h = decoder(inputs_embeds=x[None], past_key_values=cache, use_cache=True, update_past_key_values=commit,
                        block_size=bs).last_hidden_state
            logits = lm_head(h)[0].float()
            logits[:, self.mask_id] = -torch.inf  # an accepted mask proposal would leave its slot masked forever
            return logits

        n_full = prefix.shape[0] // bs * bs
        tail = prefix[n_full:]
        rec["tail"] = int(tail.shape[0])
        slots = torch.full((bs - tail.shape[0],), self.mask_id, device=prefix.device)
        lp = torch.zeros(slots.shape, device=prefix.device) if score else None
        if n_full:
            logits = run(prefix[:n_full], commit=True)
            if tail.shape[0] == 0:
                slots[0] = logits[-1].argmax()  # the block's first token is the AR next token, as Fast-dLLM v2
                rec["events"].append({"b": 0, "first": [0, int(slots[0])]})
                if score:
                    lp[0] = logits[-1].log_softmax(-1)[slots[0]]

        out: list[int] = []
        block = 0
        while True:
            chunks = [(max(s - tail.shape[0], 0), max(s + sbs - tail.shape[0], 0)) for s in range(0, bs, sbs)]
            n_edits, post = torch.zeros_like(slots), 0
            conf = torch.ones(slots.shape, device=slots.device)  # probability when committed (AR first token: 1)
            before = None
            for round_ in range(rounds + 1):  # the block's denoising, then up to `rounds` re-mask + re-fill rounds
                while True:  # one pass each: fill the first sub-block that has a mask (unless finished), and edit
                    finished = self._finished(slots, eos)
                    lo, hi = next(((lo, hi) for lo, hi in chunks
                                   if not finished and bool((slots[lo:hi] == self.mask_id).any())), (0, 0))
                    if hi == lo:  # nothing to fill
                        if edit is None or post == opt["max_post_steps"]:
                            break
                        post += 1
                    logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=False)
                    logits = torch.cat([logits[:1], logits[:-1]])[tail.shape[0]:]  # token shift
                    if edit is None:
                        fill = self._propose(logits[lo:hi], do_sample, top_p, temperature)
                    else:  # proposals for the whole block: the sub-block's fill them, the others may edit
                        proposal, probability = self._propose(logits, do_sample, top_p, temperature)
                        fill = proposal[lo:hi], probability[lo:hi]
                    is_masked = accept = slots[lo:hi] == self.mask_id
                    if hi > lo:
                        confidence = torch.where(is_masked, fill[1], -torch.inf)
                        accept = confidence > threshold
                        accept[confidence.argmax()] = True
                        accept &= is_masked
                    if edit is not None:
                        differs = (slots != self.mask_id) & (proposal != slots)
                        eos_at = (slots == eos).nonzero()
                        if eos_at.numel():
                            differs[int(eos_at[0]) + 1:] = False  # after the first EOS: padding
                        if tail.shape[0] == 0:
                            differs[0] = False  # the AR first token: predicted from the cached block, not from this one
                        change = differs & (probability > edit) & (n_edits < cap)
                    if self.trace is not None:
                        event = {"b": block, "m": [
                            [len(out) + lo + i, int(fill[0][i]), round(float(fill[1][i]), 4), int(accept[i])]
                            for i in is_masked.nonzero().flatten().tolist()]}
                        if edit is not None:
                            event["e"] = [[len(out) + i, int(slots[i]), int(proposal[i]),
                                           round(float(probability[i]), 4), int(change[i])]
                                          for i in differs.nonzero().flatten().tolist()]
                        rec["events"].append(event)
                    if score:
                        lsm = logits.log_softmax(-1)
                        lp[lo:hi] = torch.where(accept, lsm[lo:hi].gather(-1, fill[0][:, None]).squeeze(-1), lp[lo:hi])
                    slots[lo:hi] = torch.where(accept, fill[0], slots[lo:hi])
                    if remask is not None:
                        conf[lo:hi] = torch.where(accept, fill[1], conf[lo:hi])
                    if edit is not None:
                        if score:
                            lp = torch.where(change, lsm.gather(-1, proposal[:, None]).squeeze(-1), lp)
                        slots = torch.where(change, proposal, slots)
                        n_edits += change
                        if remask is not None:
                            conf = torch.where(change, probability, conf)
                        if hi == lo and not bool(change.any()):
                            break
                if round_ == rounds or (before is not None and torch.equal(slots, before)):
                    break  # out of rounds, or the last re-fill changed nothing
                low = (slots != self.mask_id) & (conf < remask)
                eos_at = (slots == eos).nonzero()
                if eos_at.numel():
                    low[int(eos_at[0]) + 1:] = False  # after the first EOS: padding
                if not bool(low.any()):
                    break
                if self.trace is not None:
                    rec["events"].append({"b": block, "r": [[len(out) + i, int(slots[i]), round(float(conf[i]), 4)]
                                                            for i in low.nonzero().flatten().tolist()]})
                before = slots.clone()
                slots[low] = self.mask_id
            out += slots.tolist()
            if score:
                out_lp += lp.tolist()
            if eos in out or len(out) >= max_new_tokens:
                break
            logits = run(torch.cat([tail, embed(slots).to(tail.dtype)]), commit=True)
            tail = prefix[:0]
            slots = torch.full((bs,), self.mask_id, device=prefix.device)
            slots[0] = logits[-1].argmax()
            block += 1
            rec["events"].append({"b": block, "first": [len(out), int(slots[0])]})
            if score:
                lp = torch.zeros(slots.shape, device=prefix.device)
                lp[0] = logits[-1].log_softmax(-1)[slots[0]]
        if eos in out:
            out = out[: out.index(eos) + 1]
        rec["out"] = out[:max_new_tokens]
        if score:
            self.commit_logprobs.append({"ids": out[:max_new_tokens], "logprobs": out_lp[: len(out[:max_new_tokens])]})
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
