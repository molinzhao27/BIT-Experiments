"""BIT's end-to-end model with a plain causal LM as the decoder (Phase 4 LLM-swap question, DESIGN.md §8).

BIT's BCI_AudioLLM loads the LLM as Qwen2AudioForConditionalGeneration and reads its sizes from
`llm_config.text_config`. BCI_TextLLM changes only that:

- the LLM is loaded with AutoModelForCausalLM (no audio tower; BIT never uses it, `project_to_audio` is false);
- `llm_config` exposes the LM's own config as `text_config`, so BIT's projector, loss and generation code run as is;
- the checkpoint's generation defaults are replaced by neutral ones built from its model config (bos/eos only, which
  is what Aero-1-Audio ships). Qwen2.5-Instruct ships repetition_penalty 1.1, top_k 20 and top_p 0.8, which
  `generate` would otherwise apply on top of Stage 2's greedy / nucleus settings.

Encoder, projector, prompt splice, LoRA, losses and generation are BIT's. Runs in BIT's environment with BIT on
sys.path (as stage1 / stage2).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig

from models.bci import BCI_AudioLLM
from utils.config import DictConfig

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "llms.yaml"


def llm_entry(name: str) -> dict:
    """A pinned entry of configs/llms.yaml, with `path` = its local snapshot."""
    entries = yaml.safe_load(REGISTRY.read_text())
    if name not in entries:
        raise KeyError(f"{name!r} not in {REGISTRY} (have: {', '.join(entries)})")
    return entries[name] | {"name": name, "path": str(ROOT / entries[name]["local_dir"])}


class BCI_TextLLM(BCI_AudioLLM):
    """BCI_AudioLLM with an AutoModelForCausalLM decoder."""

    def __init__(self, config, llm_path, lora=None, freeze_llm=False, **kwargs):
        super().__init__(config, llm_path, lora, freeze_llm, **kwargs)
        hidden = self.llm_config.text_config.hidden_size
        if self.modality_aligner is not None and self.modality_aligner.spike_proj.in_features != hidden:
            # BIT sizes its contrastive head as projector.embed_size * 2 (= 1536 for Qwen2.5-1.5B)
            raise ValueError(f"contrastive head expects {self.modality_aligner.spike_proj.in_features}-d embeddings, "
                             f"the LLM has {hidden}: set model.projector.embed_size = {hidden // 2}")

    @property
    def llm_config(self):
        return SimpleNamespace(text_config=self.llm.config)

    @llm_config.setter
    def llm_config(self, value):
        """BCI_AudioLLM.__init__ assigns self.llm.config; the getter derives the view from self.llm instead."""

    @staticmethod
    def from_pretrained(path):
        return AutoModelForCausalLM.from_pretrained(path)

    def load_llm(self, llm_path, pt_path, lora, freeze_llm):
        """As BCI_AudioLLM.load_llm (same dtype, LoRA and freezing), with `from_pretrained` (AutoModelForCausalLM)."""
        llm = self.from_pretrained(pt_path or llm_path)
        llm.generation_config = GenerationConfig.from_model_config(llm.config)
        if lora and pt_path is None:
            peft = DictConfig(lora)
            llm = get_peft_model(llm, LoraConfig(
                inference_mode=False,
                r=peft.r,
                lora_alpha=peft.alpha,
                lora_dropout=peft.dropout,
                target_modules=peft.target_modules,
                modules_to_save=peft.modules_to_save,
            ))
        if freeze_llm:
            for p in llm.parameters():
                p.requires_grad = False
        return llm

    def load_checkpoint(self, load_dir):
        raise NotImplementedError("load trained Stage 2 weights with bit_experiments.stage2.load_model_state")


def create_llm_labels(dataset: dict, tokenizer, prompt: str) -> dict:
    """BIT's create_llm_labels with integer token arrays. With an empty prompt half ("#" alone = no prompt), numpy
    returns the empty tokenization as float64, and BIT's concatenation turns input_ids / attention_mask / labels into
    floats. For a non-empty prompt the cast changes nothing."""
    from data.willett_2023_text.prepare_data import create_llm_labels as bit_create_llm_labels

    dataset = bit_create_llm_labels(dataset, tokenizer, prompt)
    for rows in dataset.values():
        for row in rows:
            for key in ("input_ids", "attention_mask", "labels"):
                row[key] = row[key].astype(np.int64)
    return dataset


def use_llm(config, name: str) -> dict:
    """Point a BIT sentence config at a registry LLM, in place: model class, LLM and tokenizer paths.
    Registers BCI_TextLLM with BIT's Trainer. Returns the entry (for run.json)."""
    from models import trainer as bit_trainer

    entry = llm_entry(name)
    if entry["loader"] == "text":
        model_class = BCI_TextLLM
    elif entry["loader"] == "diffusion":
        from bit_experiments.diffusion_llm import MASK_TOKEN, BCI_DiffusionLLM as model_class

        if "mask_id" in entry:  # an AR snapshot without |<MASK>|: one of its unused embedding rows (the Instruct control)
            mask_id = entry["mask_id"]
        else:
            mask_id = json.loads((Path(entry["path"]) / "added_tokens.json").read_text())[MASK_TOKEN]
        config["method"]["model_kwargs"]["diffusion_sampler"] = dict(entry["sampler"], mask_id=mask_id)
    else:
        raise ValueError(f"{name}: unknown loader {entry['loader']!r}")
    bit_trainer.NAME2MODEL[model_class.__name__] = model_class
    config["model"]["model_class"] = model_class.__name__
    config["method"]["model_kwargs"]["llm_path"] = entry["path"]
    config["data"]["tokenizer_path"] = entry["path"]
    return entry
