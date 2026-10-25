"""Runtime-only controls, separate from model and scientific sampler configuration."""

from functools import partial


def configure_runtime(model, mode="optimized"):
    """Enable bounded mask reuse and skip unused diffusion prefill projections.

    Does not change parameters, precision, dropout, attention kernels or samplers.
    Reference mode restores the previous workloads for in-process comparisons.
    """
    if mode not in ("reference", "optimized"):
        raise ValueError(f"unknown runtime: {mode}")
    enabled = mode == "optimized"
    decoders = 0
    for module in model.modules():
        if hasattr(module, "configure_mask_cache"):
            module.configure_mask_cache(32 if enabled else 0)
            decoders += 1
    model._skip_unused_prefill_logits = enabled
    return {"mode": mode, "mask_cache_entries": 32 if enabled else 0,
            "cached_decoders": decoders, "skip_unused_prefill_logits": enabled,
            "deferred_log_chunk": 64 if enabled else 1}


def configure_collators(trainer, mode="optimized"):
    """Keep BIT's datasets, samplers, loader RNG and worker settings unchanged."""
    if mode not in ("reference", "optimized"):
        raise ValueError(f"unknown runtime: {mode}")
    if mode == "optimized":
        from bit_experiments.exact_data import exact_pad_collate_fn as collate
    else:
        from utils.datasets import pad_collate_fn as collate
    for loader in (trainer.train_dataloader, trainer.val_dataloader):
        loader.collate_fn = partial(collate, model_inputs=trainer.model_inputs,
                                   **trainer.config.method.dataloader_kwargs)

