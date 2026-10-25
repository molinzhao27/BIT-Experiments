"""Byte-preserving collation for BIT's deterministic datasets.

The common padding case allocates its final batch once, instead of padding
each example, stacking those copies, and cloning the resulting tensor.  Dataset
access, sampling, loader settings, random draws, dtype conversion, dictionary
order, and the treatment of unused inputs remain BIT's.  Unsupported padding
layouts use BIT's original implementation.
"""

from __future__ import annotations

import numpy as np
import torch


def _single_axis_pad(arrays, pad_config):
    """Return a fresh batch for uniform-dtype axis-0 padding, or None.

    BIT pads before truncating, including on the left.  An overlong example
    therefore keeps its *first* ``truncate`` entries on either side.  Restrict
    this fast path to the common case; BIT handles other dimensions, dtype
    promotion, and validation for unusual configurations.
    """
    if len(pad_config) != 1 or not arrays:
        return None
    p = pad_config[0]
    if p["dim"] != 0 or p["side"] not in ("left", "right"):
        return None
    # np.pad also accepts pairs and per-axis constants.  Assigning one of those
    # directly could broadcast it across features instead of choosing a side.
    if not np.isscalar(p["value"]):
        return None
    constant = np.asarray(p["value"])
    if constant.dtype.kind not in "biufc":
        return None
    # np.pad normalizes scalar constants to NumPy scalars before assignment.
    # Keep its casting behavior (e.g. complex-to-real and integer overflow).
    constant = constant[()]
    first = arrays[0]
    if first.ndim == 0 or first.dtype.kind not in "biufc" or not first.dtype.isnative:
        return None
    if any(a.dtype != first.dtype or a.ndim != first.ndim or a.shape[1:] != first.shape[1:]
           or not a.flags.c_contiguous
           for a in arrays):
        return None
    maximum = max(a.shape[0] for a in arrays)
    truncate = p.get("truncate")
    truncate = maximum if truncate is None else truncate
    minimum = p.get("min_length")
    minimum = 0 if minimum is None else minimum
    if (not isinstance(truncate, (int, np.integer)) or not isinstance(minimum, (int, np.integer))
            or truncate < 0 or minimum < 0 or minimum > truncate):
        return None
    # max_length is intentionally ignored for dim 0 by BIT's padded_array.
    length = min(truncate, max(maximum, minimum))
    if length == 0 or any(size == 0 for size in first.shape[1:]):
        return None  # Preserve the reference tensor's empty-allocation strides.
    out = np.empty((len(arrays), length, *first.shape[1:]), dtype=first.dtype)
    for i, a in enumerate(arrays):
        copied = min(a.shape[0], length)
        if p["side"] == "left":
            offset = max(0, length - a.shape[0])
            out[i, :offset] = constant
            out[i, offset:offset + copied] = a[:copied]
        else:
            out[i, :copied] = a[:copied]
            out[i, copied:] = constant
    return out


def _pad_tensor(arrays, pad_config):
    padded = _single_axis_pad(arrays, pad_config)
    if padded is not None:
        # The NumPy allocation belongs exclusively to this tensor.  No source
        # array is aliased, so the reference collator's clone is redundant.
        return torch.from_numpy(padded)

    from utils.datasets import padded_array

    return torch.from_numpy(padded_array(
        arrays,
        dim=[p["dim"] for p in pad_config],
        side=[p["side"] for p in pad_config],
        value=[p["value"] for p in pad_config],
        truncate=[p.get("truncate") for p in pad_config],
        min_length=[p.get("min_length") for p in pad_config],
        max_length=[p.get("max_length") for p in pad_config],
    )).clone()


def exact_pad_collate_fn(batch, model_inputs, pad_dict):
    """Picklable drop-in for ``utils.datasets.pad_collate_fn``.

    All branches other than padding retain the upstream behavior, including
    unused inputs and the existing aliasing of unpadded, ragged arrays.
    """
    def convert_dtype(tensor):
        return tensor.float() if tensor.dtype in (torch.float64, torch.double) else tensor

    if isinstance(batch[0], list):
        batch = [row for sub_batch in batch for row in sub_batch]

    keys = batch[0].keys()
    array_keys = [k for k in keys if isinstance(batch[0][k], np.ndarray)]
    string_array_keys = [k for k in array_keys if batch[0][k].dtype.type == np.str_]
    pad_keys = pad_dict.keys()
    assert set(pad_keys).issubset(array_keys), f"Can't pad non-array keys: {set(pad_keys)-set(array_keys)}"

    padded_batch, unused_inputs = {}, {}
    for key in keys:
        if key in array_keys:
            if key in pad_keys:
                pad_config = pad_dict[key]
                if isinstance(pad_config, dict):
                    pad_config = [pad_config]
                value = convert_dtype(_pad_tensor([row[key] for row in batch], pad_config))
            elif key in string_array_keys:
                value = np.stack([row[key] for row in batch], axis=0)
            elif len(set(row[key].shape for row in batch)) == 1:
                value = convert_dtype(torch.from_numpy(np.stack([row[key] for row in batch], axis=0)))
            else:
                value = [convert_dtype(torch.from_numpy(row[key])) for row in batch]
        else:
            value = [row[key] for row in batch if key in row]
        if key in model_inputs:
            padded_batch[key] = value
        else:
            unused_inputs[key] = value
    return padded_batch, unused_inputs
