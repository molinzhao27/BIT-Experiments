"""Opt-in supervised-position vocabulary projection for BIT's Qwen2 AR models.

The transformer still processes the identical full causal sequence, including all
prefix and padding positions and all LoRA dropout calls. Only its frozen final
linear projection is restricted to hidden states with a supervised next token.
The summed cross-entropy objective and contrastive term are unchanged. Smaller
GEMMs and loss reductions can introduce ordinary floating-point rounding changes.

This is a training/teacher-forced-loss interface: BCIOutput.preds is [N, vocab]
and targets is [N], in row-major shifted-target order. Consumers requiring full
sequence logits must use the original forward. generate() is never modified.
Installation is explicit and per instance; unsupported models fail closed.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import Optional

import torch
from torch import nn
from peft import LoraConfig, PeftModel, PeftModelForCausalLM
from transformers import Qwen2AudioForConditionalGeneration, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from models.bci import BCI_AudioLLM, BCIOutput


def _parts(model):
    """Resolve existing modules, without replacing or copying any adapter."""
    wrapper = model.llm
    if isinstance(wrapper, PeftModel):
        if type(wrapper) not in (PeftModel, PeftModelForCausalLM):
            raise ValueError("selective AR supports only ordinary LoRA PEFT wrappers")
        if not all(isinstance(cfg, LoraConfig) for cfg in wrapper.peft_config.values()):
            raise ValueError("selective AR supports LoRA adapters only")
        base = wrapper.get_base_model()
    else:
        base = wrapper
    if type(base) is Qwen2ForCausalLM:
        lm, route = base, "qwen2-text"
    elif type(base) is Qwen2AudioForConditionalGeneration and type(base.language_model) is Qwen2ForCausalLM:
        lm, route = base.language_model, "qwen2-audio-language-model"
    else:
        raise ValueError("selective AR requires a supported Qwen2 causal or Qwen2Audio language model")
    if type(lm.model) is not Qwen2Model:
        raise ValueError("selective AR requires the standard Qwen2 decoder")
    head = lm.lm_head
    if type(head) is not nn.Linear or "forward" in head.__dict__ or any(p.requires_grad for p in head.parameters()):
        raise ValueError("selective AR requires a frozen, deterministic plain Linear vocabulary head")
    if bool(getattr(base, "is_quantized", False)) or bool(getattr(lm, "is_quantized", False)):
        raise ValueError("selective AR does not support quantized models")
    return wrapper, base, lm, route


def install_selective_ar_head(model) -> dict:
    """Install the candidate forward; idempotent, and leaves generate/state_dict intact."""
    if getattr(model, "_selective_ar_original_forward", None) is not None:
        _, _, lm, route = _parts(model)
        return {"enabled": True, "route": route, "vocab_size": lm.config.vocab_size,
                "predictions": "supervised_shifted_positions"}
    if not isinstance(model, BCI_AudioLLM) or getattr(model.forward, "__func__", None) is not BCI_AudioLLM.forward:
        raise ValueError("selective AR requires BIT's unmodified autoregressive forward")
    if model.project_to_audio:
        raise ValueError("selective AR does not support project_to_audio")
    loss = model.loss_fn
    if (type(loss) is not nn.CrossEntropyLoss or loss.reduction != "sum" or loss.ignore_index != -100
            or loss.weight is not None or loss.label_smoothing != 0):
        raise ValueError("selective AR requires summed unweighted CE with ignore_index=-100 and no smoothing")
    _, _, lm, route = _parts(model)
    model._selective_ar_original_forward = model.forward
    model._selective_ar_had_instance_forward = "forward" in model.__dict__
    model.forward = MethodType(selective_ar_forward, model)
    return {"enabled": True, "route": route, "vocab_size": lm.config.vocab_size,
            "predictions": "supervised_shifted_positions"}


def uninstall_selective_ar_head(model) -> bool:
    """Restore the exact original per-instance/class forward lookup."""
    original = getattr(model, "_selective_ar_original_forward", None)
    if original is None:
        return False
    if model._selective_ar_had_instance_forward:
        model.forward = original
    else:
        del model.forward
    del model._selective_ar_original_forward
    del model._selective_ar_had_instance_forward
    return True


def selective_ar_forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    input_split: torch.LongTensor,
    spikes: torch.FloatTensor,
    spikes_mask: torch.LongTensor,
    spikes_timestamp: torch.FloatTensor,
    spikes_lengths: torch.LongTensor,
    targets: Optional[torch.LongTensor] = None,
    dataset_name: Optional[torch.Tensor] = None,
    day_idx: Optional[torch.LongTensor] = None,
) -> BCIOutput:
    """BIT's AR loss with compact supervised logits; requires teacher-forcing targets."""
    if targets is None:
        raise ValueError("selective AR forward requires targets; use generate() for generation")
    wrapper, base, lm, _ = _parts(self)
    spikes_embeds, text_embeds, spikes_mask, contrastive_loss = self.prepare_embeds(
        input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx,
    )
    inputs_embeds, attention_mask, targets = self.merge_embeds(
        text_embeds, spikes_embeds, spikes_mask, attention_mask, input_split, targets,
    )
    # These are the options the outer Qwen2/Qwen2Audio forward supplies to the decoder. In particular,
    # preserve config.use_cache, causal attention, positions, dropout, and gradient checkpointing.
    context = wrapper._enable_peft_forward_hooks() if isinstance(wrapper, PeftModel) else nullcontext()
    with context:
        hidden = lm.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True,
            output_attentions=base.config.output_attentions,
            output_hidden_states=base.config.output_hidden_states,
        ).last_hidden_state
        shifted = targets[..., 1:]
        selected = shifted != -100
        logits = lm.lm_head(hidden[..., :-1, :][selected])
    labels = shifted[selected].long()
    # Sum reduction also handles N=0: empty CE is a differentiable zero, so the decoder and its
    # adapters still receive zero gradients instead of disappearing from the optimizer's step.
    loss = self.loss_fn(logits.float(), labels)
    if contrastive_loss is not None:
        loss += contrastive_loss
    return BCIOutput(loss=loss, n_examples=selected.sum(), preds=logits, targets=labels)
