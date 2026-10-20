"""The batched (torch, permutation-axis) pipeline must equal the numpy Phase 1 pipeline."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kunzdecoder import batched, decoding, models  # noqa: E402


def _data(seed=0, n=140, w=5, f=16):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(7), n // 7)
    X = rng.normal(size=(n, w, f))
    X[:, 2] += 0.9 * (y[:, None] == np.arange(f)[None] % 7)  # window 2 is the informative one
    X[:, 3] += 0.4 * (y[:, None] == np.arange(f)[None] % 7)
    return X.astype(np.float32), y


def _folds(folds):
    return [(torch.as_tensor(a), torch.as_tensor(b)) for a, b in folds]


def test_nested_cv_matches_numpy():
    X, y = _data()
    ref = decoding.nested_cv(X, y, 10, 10, seed=42)
    outer, inner = decoding.nested_folds(y, 10, 10, 42)
    out = batched.nested_cv(torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(y[None]), 7,
                            _folds(outer), [_folds(f) for f in inner])
    assert int(out["correct"][0]) == ref.n_correct
    assert [int(w) for w in out["windows"][0]] == [ref.window_idx[ref.fold == k][0] for k in range(10)]


def test_transfer_matches_numpy():
    X, y = _data(1)
    Xb, yb = _data(2)
    best, _ = decoding.select_window(X, y, np.arange(7), 10, 42, "diag_lda")
    m = models.fit(X[:, [best]], y, np.arange(7))
    ref = int((models.predict(m, Xb[:, [best]])[:, 0] == yb).sum())
    labels = torch.as_tensor(np.vstack([y, np.random.default_rng(0).permutation(y)]))
    out = batched.transfer(torch.as_tensor(X, dtype=torch.float64), labels, 7,
                           _folds(decoding.stratified_folds(y, 10, 42)),
                           {"b": (torch.as_tensor(Xb, dtype=torch.float64), torch.as_tensor(yb))})
    assert int(out["windows"][0]) == best
    assert int(out["b"][0]) == ref


def test_nested_cv_refold_matches_fixed_folds_per_row():
    X, y = _data(3)
    rng = np.random.default_rng(1)
    rows = np.vstack([y, rng.permutation(y), rng.permutation(y)])
    folds_of = lambda yy: decoding.nested_folds(yy, 10, 10, 42)  # noqa: E731
    Xt = torch.as_tensor(X, dtype=torch.float64)
    out = batched.nested_cv_refold(Xt, torch.as_tensor(rows), 7, folds_of)
    for p, row in enumerate(rows):  # each row = the fixed-fold pipeline run on that row's own folds
        outer, inner = folds_of(row)
        ref = batched.nested_cv(Xt, torch.as_tensor(row[None]), 7, _folds(outer), [_folds(f) for f in inner])
        assert int(out["correct"][p]) == int(ref["correct"][0])
        assert out["windows"][p].tolist() == ref["windows"][0].tolist()
    assert int(out["correct"][0]) == decoding.nested_cv(X, y, 10, 10, seed=42).n_correct


def test_permuted_labels_within_groups():
    y = np.tile(np.arange(7), 6)
    groups = np.repeat([0, 1, 2], 14)
    rows = batched.permuted_labels(y, 50, np.random.default_rng(0), groups)
    assert (rows[0] == y).all() and len(rows) == 51
    for g in range(3):
        cols = groups == g
        assert all(np.bincount(r[cols], minlength=7).tolist() == [2] * 7 for r in rows)
    assert (rows[1:] != y).any()
    plain = batched.permuted_labels(y, 3, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    assert (plain == np.vstack([y] + [rng.permutation(y) for _ in range(3)])).all()
