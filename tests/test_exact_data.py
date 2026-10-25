"""Exact collation against the vendored BIT implementation, including RNG."""

import copy
import pickle
import random
import sys
from functools import partial
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))
if not (ROOT / "external" / "bit" / "utils" / "datasets.py").exists():
    pytest.skip("external/bit not cloned", allow_module_level=True)

from bit_experiments.exact_data import exact_pad_collate_fn  # noqa: E402
from utils import datasets as bit_data  # noqa: E402


def assert_bytes(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes()
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_bytes(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_bytes(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("truncate,minimum", [(None, None), (3, None), (0, None), (10, 8)])
def test_padding_matches_reference(dtype, side, truncate, minimum):
    batch = [{"x": np.arange(n * 3).reshape(n, 3).astype(dtype)} for n in (0, 2, 6)]
    options = {"model_inputs": ["x"], "pad_dict": {"x": {"dim": 0, "side": side, "value": 0,
               "truncate": truncate, "min_length": minimum, "max_length": 100}}}
    assert_bytes(bit_data.pad_collate_fn(batch, **options), exact_pad_collate_fn(batch, **options))


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("pad_value", [-0.0, float("nan"), -100.0])
def test_nan_payloads_signed_zero_and_noncontiguous_arrays(side, pad_value):
    # Distinct NaN payloads and signed zeros must survive copies bit for bit.
    x = np.array([0x00000000, 0x80000000, 0x7FC00001, 0x7FC12345,
                  0x3F800000, 0xBF800000], dtype=np.uint32).view(np.float32)
    batch = [{"x": x[::-2, None]}, {"x": x[:, None]}]
    options = {"model_inputs": ["x"], "pad_dict": {"x": {"dim": 0, "side": side, "value": pad_value}}}
    assert_bytes(bit_data.pad_collate_fn(batch, **options), exact_pad_collate_fn(batch, **options))


@pytest.mark.parametrize("kind", ["multidim", "axis1", "mixed_dtype", "fortran", "endian_float", "endian_int"])
def test_unusual_padding_uses_reference_fallback(monkeypatch, kind):
    batch = [{"x": np.arange(n * 2, dtype=np.float32).reshape(n, 2)} for n in (2, 4)]
    pad = [{"dim": 0, "side": "right", "value": -100}]
    if kind == "multidim":
        pad.append({"dim": 1, "side": "left", "value": 0, "max_length": 4})
    elif kind == "axis1":
        batch = [{"x": np.arange(2 * n, dtype=np.float32).reshape(2, n)} for n in (2, 4)]
        pad = [{"dim": 1, "side": "left", "value": 0, "max_length": 4}]
    elif kind == "mixed_dtype":
        batch[1]["x"] = batch[1]["x"].astype(np.float64)
    elif kind == "fortran":
        batch = [{"x": np.asfortranarray(row["x"])} for row in batch]
    else:
        dtype = np.dtype(np.float32 if kind == "endian_float" else np.int32).newbyteorder("S")
        batch = [{"x": row["x"].astype(dtype)} for row in batch]
    options = {"model_inputs": ["x"], "pad_dict": {"x": pad}}
    expected = bit_data.pad_collate_fn(batch, **options)
    calls = []
    original = bit_data.padded_array
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(bit_data, "padded_array", record)
    assert_bytes(expected, exact_pad_collate_fn(batch, **options))
    assert calls == [True]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("value", [(1, 2), [1, 2], np.asarray(3), "4", None])
def test_non_scalar_or_nonnumeric_constant_uses_reference_fallback(monkeypatch, side, value):
    batch = [{"x": np.arange(n * 2, dtype=np.float32).reshape(n, 2)} for n in (1, 3)]
    options = {"model_inputs": ["x"], "pad_dict": {"x": {"dim": 0, "side": side, "value": value}}}
    expected = bit_data.pad_collate_fn(batch, **options)
    calls = []
    original = bit_data.padded_array
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(bit_data, "padded_array", record)
    assert_bytes(expected, exact_pad_collate_fn(batch, **options))
    assert calls == [True]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("dtype,value", [
    (np.float32, np.float32(-0.0)),
    (np.float32, np.array(0x7FC12345, dtype=np.uint32).view(np.float32)[()]),
    (np.int32, float("nan")),
    (np.float32, complex(1, 2)),
    (np.uint8, np.int64(-1)),
    (np.uint64, np.uint64(2**64 - 1)),
    (np.float32, 1e300),
    (np.int32, np.float64(1.5)),
    (np.complex64, np.complex64(1 - 2j)),
    (np.bool_, True),
])
def test_scalar_constant_casting_matches_numpy_pad(dtype, value, side):
    batch = [{"x": np.arange(n * 2).reshape(n, 2).astype(dtype)} for n in (1, 3)]
    options = {"model_inputs": ["x"], "pad_dict": {"x": {"dim": 0, "side": side, "value": value}}}
    # NumPy versions differ in whether a lossy/out-of-range cast warns or raises.
    # Match the installed reference's output or error type without changing it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            expected = bit_data.pad_collate_fn(batch, **options)
        except (ValueError, TypeError, OverflowError) as error:
            with pytest.raises(type(error)):
                exact_pad_collate_fn(batch, **options)
        else:
            assert_bytes(expected, exact_pad_collate_fn(batch, **options))


def test_all_collation_branches_order_and_ownership():
    batch = [{"x": np.arange(n * 2, dtype=np.float64).reshape(n, 2),
              "scalar": np.asarray(i), "fixed": np.arange(3), "ragged": np.arange(n),
              "strings": np.asarray(["trial", str(i)]), "name": str(i)}
             for i, n in enumerate((2, 4))]
    options = {"model_inputs": ["x", "scalar", "fixed", "ragged"],
               "pad_dict": {"x": {"dim": 0, "side": "right", "value": -100}}}
    expected = bit_data.pad_collate_fn([batch[:1], batch[1:]], **options)
    actual = exact_pad_collate_fn([batch[:1], batch[1:]], **options)
    assert_bytes(expected, actual)
    assert actual[0]["x"].dtype == torch.float32
    actual[0]["x"].fill_(999)
    actual[0]["fixed"].fill_(999)
    assert np.all(batch[0]["x"] != 999) and np.all(batch[0]["fixed"] != 999)
    # Preserve the original ragged-array aliasing contract, too.
    actual[0]["ragged"][0][0] = 88
    assert batch[0]["ragged"][0] == 88


def test_invalid_padding_keeps_reference_validation():
    batch = [{"x": np.arange(3)}]
    options = {"model_inputs": ["x"], "pad_dict": {"x": {"dim": 0, "side": "right", "value": 0,
               "truncate": 2, "min_length": 4}}}
    for collate in (bit_data.pad_collate_fn, exact_pad_collate_fn):
        with pytest.raises(AssertionError, match="Can't truncate below minimum"):
            collate(batch, **options)


def test_pickling_and_three_epoch_sampler_rng_trajectory():
    rows = [{"spikes": np.full((n, 3), i, dtype=np.float32), "labels": np.arange(i % 3 + 1),
             "day_idx": np.asarray(0), "block_idx": np.asarray(0), "trial": i}
            for i, n in enumerate((3, 7, 5, 4, 6, 2, 8))]
    ds = bit_data.SpikingDatasetForDecoding({"train": rows}, "train", targets_name="labels",
                                          dataset_name="brandman_2024_text")
    pad = {k: {"dim": 0, "side": "right", "value": v}
           for k, v in (("spikes", -100), ("spikes_mask", 0), ("spikes_timestamp", 0),
                        ("targets", -100), ("targets_mask", 0))}
    inputs = ["spikes", "spikes_mask", "spikes_timestamp", "targets", "targets_mask"]
    def run(fn):
        torch.manual_seed(301)
        np.random.seed(301)
        random.seed(301)
        sampler = bit_data.GroupBatchSampler(ds, batch_size=3, shuffle=True, seed=42)
        collate = pickle.loads(pickle.dumps(partial(fn, model_inputs=inputs, pad_dict=pad)))
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
        output, states = [], []
        for _ in range(3):
            output.append(list(loader))
            states.append((torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()),
                           random.getstate(), copy.deepcopy(sampler.rng.bit_generator.state)))
            # Exercise the shared main-process RNG between epochs, as training does.
            torch.rand(11)
        return output, states
    assert_bytes(run(bit_data.pad_collate_fn), run(exact_pad_collate_fn))
