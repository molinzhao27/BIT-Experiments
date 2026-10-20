#!/usr/bin/env python3
"""CPU-only counterexample for the Phase 2 source-label-shuffle transfer null.

The source remains strongly word-informative under the no-transfer null. Target
features contain no information about their independently randomized balanced
labels. Shuffling individual source labels destroys source word structure and
can change target prediction diversity, producing the wrong reference null.

This uses exactly the RNG order of the original audit command. There is one
candidate window, so window selection is immaterial. fit_at/predict_at are the
same functions called by batched.transfer after selecting its window.

Run from any directory:
  /Users/molin/Documents/KunzDecoder/.venv/bin/python \
    docs/audits/2026-10-19/transfer_null_repro.py

All tensors are explicitly CPU. Constants bound work to 1,000 source label
permutations and 1,000 independent target-label assignments; the original run
completed in under one second including interpreter startup.
"""

import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = None
sys.path.insert(0, str(ROOT / "src"))
from kunzdecoder import batched


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    n, k = 140, 7
    y = np.repeat(np.arange(k), 20)

    # Seven centered prototypes with positive, equal within-class variances.
    # The same zero-mean noise template is used for each source class.
    noise = .05 * np.vstack([np.eye(k), -np.eye(k), np.zeros((6, k))])
    Xa = np.eye(k)[y] - 1 / k + np.tile(noise, (k, 1))

    # First RNG call: target features, independent of every target-label draw.
    Xb = rng.normal(0, .001, size=(n, k))
    # Next RNG calls: exactly 1,000 independent permutations of source labels.
    rows = batched.permuted_labels(y, 1000, rng)
    lab = torch.tensor(rows, device="cpu")
    w = torch.zeros(len(rows), dtype=torch.long, device="cpu")
    m, v = batched.fit_at(torch.tensor(Xa[:, None], device="cpu"), lab, k, w)
    pred = batched.predict_at(
        m, v, torch.tensor(Xb[:, None], device="cpu"), w
    ).numpy()

    reject, total_correct = 0, 0
    # Remaining RNG calls: 1,000 independently randomized balanced target sets.
    for _ in range(1000):
        target = rng.permutation(y)
        scores = (pred == target).sum(1)
        total_correct += scores[0]
        # Exactly the p-value used in scripts/phase2_transfer.py:p_value.
        reject += (1 + (scores[1:] >= scores[0]).sum()) / 1001 < .05

    result = {
        "seed": 42,
        "n_source_trials": n,
        "n_target_trials": n,
        "n_classes": k,
        "candidate_windows": 1,
        "source_label_permutations": 1000,
        "independent_target_label_assignments": 1000,
        "device": "cpu",
        "min_true_model_variance": float(v[0].min()),
        "true_pred_class_counts": np.bincount(pred[0], minlength=k).tolist(),
        "null_predictors_with_one_class": int(
            sum(len(np.unique(p)) == 1 for p in pred[1:])
        ),
        "chance_mean_accuracy": float(total_correct / 140000),
        "theoretical_chance": 1 / k,
        "nominal_alpha": .05,
        "rejections": int(reject),
        "false_positive_rate_at_05": float(reject / 1000),
        "runtime_seconds_excluding_imports": time.perf_counter() - started,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
