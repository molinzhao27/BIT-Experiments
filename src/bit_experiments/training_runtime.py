"""Stage 2 runtime improvements that leave BIT's training recipe unchanged.

The upstream loop reads two CUDA scalars after every batch, draining the device
queue before the next batch can be prepared. Buffering the already-reduced,
detached scalars removes most of those waits. We keep the individual Python
values and their original summation order: a GPU reduction of the whole epoch
would change the reported loss through floating-point reassociation.
"""

from __future__ import annotations

import torch
from tqdm import tqdm


class DeferredScalars:
    """A bounded device buffer of scalar values, converted in insertion order.

    ``append`` takes ownership of an immutable scalar result (the caller must
    not subsequently modify its storage). Training passes a fresh ``sum()``
    result. Autograd graphs are detached immediately. A dtype/device transition
    flushes the current chunk instead of silently promoting its values.
    """

    def __init__(self, chunk_size: int = 64):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("log chunk size must be a positive integer")
        self.chunk_size = chunk_size
        self._pending: list[torch.Tensor] = []
        self._values: list[float | int | bool] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def append(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("DeferredScalars expects a scalar tensor")
        value = value.detach()
        if self._pending:
            first = self._pending[0]
            if value.dtype != first.dtype or value.device != first.device:
                self.flush()
        self._pending.append(value)
        if len(self._pending) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            # stack copies scalars without arithmetic; tolist performs the same
            # per-element Python conversion as item(), including integer counts.
            self._values.extend(torch.stack(self._pending).cpu().tolist())
            self._pending.clear()

    def values(self) -> list[float | int | bool]:
        self.flush()
        return self._values


def train_stage2_deferred(trainer, log_chunk_size: int = 64) -> None:
    """BIT's ``Trainer.train`` for ``endtoend``, with batched scalar transfers.

    Keep this aligned with external/bit/models/trainer.py. In particular, BIT
    steps on global steps 1, 1+A, ... for accumulation A, even across epochs.
    This function deliberately preserves that behavior and all collective,
    optimizer, scheduler, validation, selection and checkpoint call ordering.
    It delegates evaluation and checkpoint serialization to the original trainer.
    """
    config = trainer.config
    if config.method.model_kwargs.method_name != "endtoend":
        raise ValueError("deferred Stage 2 training supports endtoend only")
    train_loss = DeferredScalars(log_chunk_size)
    train_n_examples = DeferredScalars(log_chunk_size)
    global_step = 1
    best_metric = -float("inf")

    for epoch in range(trainer.start_epoch, config.training.num_epochs + 1):
        trainer.print_v(f"Epoch {epoch}", verbosity=1)
        trainer.model.train()

        for step, (model_inputs, unused_inputs) in enumerate(tqdm(trainer.train_dataloader)):
            if (global_step - 1) % config.optimizer.gradient_accumulation_steps == 0:
                with torch.autocast(device_type=trainer.device_type, dtype=trainer.dtype):
                    outputs = trainer.model(**model_inputs)
                    n_examples = outputs.n_examples
                    loss = outputs.loss / config.optimizer.gradient_accumulation_steps
                trainer.accelerator.backward(loss)
                trainer.optimizer.step()
                if config.optimizer.scheduler in ["linear", "cosine"]:
                    trainer.lr_scheduler.step()
                trainer.optimizer.zero_grad()
            else:
                with trainer.accelerator.no_sync(trainer.model):
                    with torch.autocast(device_type=trainer.device_type, dtype=trainer.dtype):
                        outputs = trainer.model(**model_inputs)
                        n_examples = outputs.n_examples
                        loss = outputs.loss / config.optimizer.gradient_accumulation_steps
                        trainer.accelerator.backward(loss)

            train_loss.append(trainer.accelerator.gather(loss).sum().detach())
            train_n_examples.append(trainer.accelerator.gather(n_examples).sum().detach())
            global_step += 1

        if config.training.eval_every and (epoch % config.training.eval_every == 0):
            trainer.print_v(f"Eval Epoch {epoch}", verbosity=1)
            test_avg_loss, test_avg_metrics, gt, preds = trainer.evaluate()
            loss_values, count_values = train_loss.values(), train_n_examples.values()
            train_avg_loss = sum(loss_values) / sum(count_values) if sum(count_values) > 0 else 0
            trainer.print_v(f"{train_avg_loss=}" + "\n" + f"{test_avg_loss=}" + "\n"
                            + f"{test_avg_metrics=}", verbosity=1)
            val_metric = 1 - test_avg_metrics[trainer.metric_kwargs.metric_name]

            if config.log_to_wandb and trainer.accelerator.is_main_process:
                import wandb

                wandb.log({"epoch": epoch, "train_avg_loss": train_avg_loss,
                           "test_avg_loss": test_avg_loss, **test_avg_metrics})

            if val_metric > best_metric:
                best_metric = val_metric
                trainer.save_checkpoint(save_name="BEST", epoch=epoch)
                trainer.save_accelerator_state_to_resume(save_name="BEST", epoch=epoch)

            train_loss = DeferredScalars(log_chunk_size)
            train_n_examples = DeferredScalars(log_chunk_size)
            trainer.model.train()

        if config.training.save_every and (epoch % config.training.save_every == 0):
            trainer.save_checkpoint(save_name="EPOCH", epoch=epoch)
            trainer.save_accelerator_state_to_resume(save_name="EPOCH", epoch=epoch)

    trainer.save_checkpoint(save_name="LAST", epoch=epoch)
    trainer.save_accelerator_state_to_resume(save_name="LAST", epoch=epoch)
    trainer.print_v("Training completed!", verbosity=1)
