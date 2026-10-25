"""Deferred Stage 2 metrics against the actual vendored BIT training loop.

No model/data downloads or GPU required. A multi-epoch stochastic model exercises
optimizer/scheduler state, leftover accumulation, validation, checkpoint choice,
logging, batch order and RNG. The oracle is extracted from the pinned upstream
file rather than another hand-written version of its loop.
"""

import ast
import copy
from contextlib import contextmanager
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bit_experiments.training_runtime import DeferredScalars, train_stage2_deferred


def _upstream_train(wandb):
    tree = ast.parse((ROOT / "external/bit/models/trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"torch": torch, "tqdm": lambda x: x, "wandb": wandb}
    exec(compile(module, "BIT.Trainer.train", "exec"), namespace)
    return namespace["train"]


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
        assert _bytes(a) == _bytes(b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, float):
        assert struct.pack("d", a) == struct.pack("d", b)
    else:
        assert a == b


class _Model(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(0.2)
        self.events = events

    def forward(self, x, sample_id):
        self.events.append(("batch", int(sample_id)))
        pred = self.proj(self.dropout(x))
        return NS(loss=pred.square().sum(), n_examples=torch.tensor(x.shape[0]), preds=pred)


class _Accelerator:
    is_main_process = True

    def __init__(self, events):
        self.events = events

    def backward(self, loss):
        self.events.append("backward")
        loss.backward()

    def gather(self, value):
        self.events.append(("gather", str(value.dtype)))
        # Exercise reduction of more than one gathered scalar without changing
        # anything based on the materialization timing.
        return torch.stack([value, value * 2])

    @contextmanager
    def no_sync(self, model):
        self.events.append("no_sync_enter")
        yield
        self.events.append("no_sync_exit")


class _Trainer:
    def __init__(self, accumulation, eval_every, scheduler, log_to_wandb):
        self.config = NS(training=NS(num_epochs=4, eval_every=eval_every, save_every=2),
                         optimizer=NS(gradient_accumulation_steps=accumulation, scheduler=scheduler),
                         method=NS(model_kwargs=NS(method_name="endtoend")), log_to_wandb=log_to_wandb)
        self.events, self.printed, self.checkpoints = [], [], []
        self.model = _Model(self.events)
        self.metric_fns = {"ignored_train_metric": None}
        self.metric_kwargs = NS(metric_name="WER")
        self.device_type, self.dtype = "cpu", torch.bfloat16
        self.start_epoch, self.n_evals = 1, 0
        self.accelerator = _Accelerator(self.events)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1 / (step + 1))
        data = [{"x": torch.randn(2, 3), "sample_id": torch.tensor(i)} for i in range(5)]
        # DataLoader consumes a fresh CPU base_seed on every epoch; shuffle also
        # draws from the global RNG. Both must remain unchanged across epochs.
        self.train_dataloader = torch.utils.data.DataLoader(
            [(item, {}) for item in data], batch_size=None, shuffle=True)

    def print_v(self, value, verbosity=0):
        self.printed.append(value)

    def evaluate(self):
        self.events.append("evaluate")
        self.model.eval()
        self.n_evals += 1
        # These deliberately tie and regress to check strict best selection.
        wer = [0.8, 0.4, 0.4, 0.5][self.n_evals - 1]
        return float(torch.rand(())), {"WER": wer}, None, None

    def _snapshot(self):
        return copy.deepcopy({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                              "scheduler": self.lr_scheduler.state_dict(), "rng": torch.get_rng_state(),
                              "grads": [p.grad for p in self.model.parameters()]})

    def save_checkpoint(self, save_name, epoch):
        self.events.append(("save", save_name, epoch))
        self.checkpoints.append(("save", save_name, epoch, self._snapshot()))

    def save_accelerator_state_to_resume(self, save_name, epoch):
        self.events.append(("resume", save_name, epoch))
        self.checkpoints.append(("resume", save_name, epoch, self._snapshot()))


@pytest.mark.parametrize("accumulation,eval_every,scheduler", [(1, 1, "linear"), (3, 2, "cosine"),
                                                            (4, 1, "step"), (2, None, "linear")])
@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_multi_epoch_training_matches_upstream(accumulation, eval_every, scheduler, chunk_size, monkeypatch):
    monkeypatch.setattr("bit_experiments.training_runtime.tqdm", lambda x: x)
    logs = []
    fake_wandb = NS(log=lambda row: logs.append(copy.deepcopy(row)))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    upstream = _upstream_train(fake_wandb)
    results = []
    for fn in [upstream, lambda trainer: train_stage2_deferred(trainer, chunk_size)]:
        torch.manual_seed(91)
        trainer = _Trainer(accumulation, eval_every, scheduler, True)
        logs.clear()
        fn(trainer)
        results.append({"state": trainer._snapshot(), "checkpoints": trainer.checkpoints,
                        "printed": trainer.printed, "events": trainer.events, "logs": copy.deepcopy(logs)})
    _same(*results)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64,
                                  torch.int32, torch.int64, torch.bool])
def test_scalar_conversion_preserves_dtype_values_and_order(dtype):
    values = [0, -0.0, 1, -2, 3]
    if dtype.is_floating_point:
        values += [float("inf"), -float("inf"), float("nan")]
    tensors = [torch.tensor(x, dtype=dtype) for x in values]
    scalars = DeferredScalars(chunk_size=3)
    for t in tensors:
        scalars.append(t)
        assert scalars.pending_count < 3
    _same([t.item() for t in tensors], scalars.values())
    assert scalars.pending_count == 0


def test_dtype_changes_never_promote_integer_counts():
    tensors = [torch.tensor(2**60 + 1), torch.tensor(1.5), torch.tensor(2**60 + 3),
               torch.tensor(1.0 + 2**-30, dtype=torch.float64)]
    scalars = DeferredScalars(64)
    for t in tensors:
        scalars.append(t)
    _same([t.item() for t in tensors], scalars.values())


def test_buffer_does_not_retain_autograd_graph_or_change_rng():
    state = torch.get_rng_state().clone()
    x = torch.tensor(2.0, requires_grad=True)
    scalars = DeferredScalars(3)
    scalars.append((x * x).sum())
    assert all(not t.requires_grad and t.grad_fn is None for t in scalars._pending)
    assert scalars.values() == [4.0]
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("size", [0, -1, 1.0, True, None])
def test_bad_chunk_size_is_rejected(size):
    with pytest.raises(ValueError, match="positive integer"):
        DeferredScalars(size)


def test_non_scalar_or_wrong_training_recipe_rejected():
    with pytest.raises(ValueError, match="scalar tensor"):
        DeferredScalars().append(torch.ones(2))
    trainer = NS(config=NS(method=NS(model_kwargs=NS(method_name="ssl"))))
    with pytest.raises(ValueError, match="endtoend only"):
        train_stage2_deferred(trainer)
