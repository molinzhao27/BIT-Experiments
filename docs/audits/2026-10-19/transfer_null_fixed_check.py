#!/usr/bin/env python3
"""The audit's transfer counterexample (transfer_null_repro.py), scored with the corrected null.

Same source/target construction and seed. The corrected null keeps the decoder trained on the
true source labels fixed and permutes the target labels within blocks (here 2 word-balanced blocks
of 70), as scripts/phase2_transfer.py now does. Target labels are independent of the target
features, so the false-positive rate at 0.05 should be at most ~0.05 (the old null gave 0.361).

  /Users/molin/Documents/KunzDecoder/.venv/bin/python docs/audits/2026-10-19/transfer_null_fixed_check.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from kunzdecoder import batched  # noqa: E402


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    n, k = 140, 7
    y = np.repeat(np.arange(k), 20)
    noise = .05 * np.vstack([np.eye(k), -np.eye(k), np.zeros((6, k))])
    Xa = np.eye(k)[y] - 1 / k + np.tile(noise, (k, 1))
    Xb = rng.normal(0, .001, size=(n, k))
    w = torch.zeros(1, dtype=torch.long)
    m, v = batched.fit_at(torch.tensor(Xa[:, None]), torch.tensor(y[None]), k, w)
    pred = batched.predict_at(m, v, torch.tensor(Xb[:, None]), w).numpy()[0]  # fixed decoder

    blocks = np.repeat([0, 1], n // 2)
    y_block = np.tile(np.repeat(np.arange(k), 10), 2)  # 10 per word per block
    pvals = []
    for _ in range(1000):  # independent balanced target labels, randomized within blocks
        target = batched.permuted_labels(y_block, 1, rng, blocks)[1]
        rows = batched.permuted_labels(target, 1000, rng, blocks)
        hits = (rows == pred[None]).sum(1)
        pvals.append((1 + (hits[1:] >= hits[0]).sum()) / 1001)
    pvals = np.array(pvals)
    print(json.dumps({"seed": 42, "fixed_pred_class_counts": np.bincount(pred, minlength=k).tolist(),
                      "target_label_assignments": 1000, "permutations_each": 1000,
                      "false_positive_rate_at_05": float((pvals < .05).mean()),
                      "false_positive_rate_at_01": float((pvals < .01).mean()),
                      "old_null_false_positive_rate_at_05": 0.361}, indent=2))


if __name__ == "__main__":
    main()
